"""
Verbose-hooks runner for generate_infinitetalk.py.

Invoked by api.py instead of calling generate_infinitetalk.py directly.
It:

  1. Forces the child's logging to INFO and pipes everything through a
     flushing stdout handler.
  2. Installs monkey-patches that emit a log line at every meaningful
     step of model loading, audio prep, diffusion, decode, and encode.
  3. Then hands argv off to generate_infinitetalk.generate() so the CLI
     flags you already know still work.

Run exactly like the original script -- same argv:
    python -u api_verbose_hooks.py --task infinitetalk-14B --size infinitetalk-480 ...
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path

# --- step 0. Unbuffered stdout ------------------------------------------------
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass


# --- step 1. Unified logging --------------------------------------------------

class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


def _install_logging() -> None:
    fmt = "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-26s | %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    h = _FlushHandler(stream=sys.stdout)
    h.setFormatter(logging.Formatter(fmt, dfmt))
    root.addHandler(h)
    root.setLevel(logging.INFO)
    # Noisy libraries we *don't* want at INFO.
    for noisy in ("urllib3", "filelock", "huggingface_hub", "h5py", "PIL.PngImagePlugin"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_install_logging()
VLOG = logging.getLogger("infinitetalk.hooks")


def _banner(msg: str) -> None:
    VLOG.info("-" * 90)
    VLOG.info(msg)
    VLOG.info("-" * 90)


# --- step 2. Small helper for wrap-and-log -----------------------------------

def _wrap(klass_or_mod, name: str, emit):
    """Replace klass_or_mod.name with a wrapper that calls emit(args, kwargs, result_or_exc, dt)."""
    orig = getattr(klass_or_mod, name)

    def wrapper(*args, **kwargs):
        t0 = time.time()
        try:
            result = orig(*args, **kwargs)
        except Exception as e:
            dt = time.time() - t0
            emit(args, kwargs, ("exc", e), dt)
            raise
        dt = time.time() - t0
        emit(args, kwargs, ("ok", result), dt)
        return result

    wrapper.__wrapped__ = orig  # type: ignore[attr-defined]
    wrapper.__name__ = f"verbose_{name}"
    setattr(klass_or_mod, name, wrapper)


def _fmt_bytes(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


# --- step 3. Safetensors / torch.load hooks (shards) --------------------------

def _install_loader_hooks() -> None:
    import safetensors.torch as st
    import torch as _torch

    orig_load_file = st.load_file

    def verbose_load_file(path, device="cpu", *a, **kw):
        size = 0
        try:
            size = Path(path).stat().st_size
        except Exception:
            pass
        VLOG.info(f"safetensors.load_file: START {path} ({_fmt_bytes(size)})")
        t0 = time.time()
        try:
            sd = orig_load_file(path, device=device, *a, **kw)
        except Exception as e:
            VLOG.error(f"safetensors.load_file: FAILED {path}: {e}")
            raise
        n_keys = len(sd) if hasattr(sd, "__len__") else -1
        VLOG.info(f"safetensors.load_file: DONE  {path}  "
                  f"keys={n_keys} in {time.time()-t0:.2f}s")
        return sd

    st.load_file = verbose_load_file

    # Also patch the 'from safetensors.torch import load_file' alias inside wan.multitalk
    try:
        import wan.multitalk as _wm
        if hasattr(_wm, "load_file"):
            _wm.load_file = verbose_load_file
    except Exception as e:
        VLOG.warning(f"could not patch wan.multitalk.load_file: {e}")

    orig_torch_load = _torch.load

    def verbose_torch_load(path, *a, **kw):
        size = 0
        try:
            size = Path(str(path)).stat().st_size
        except Exception:
            pass
        VLOG.info(f"torch.load: START {path} ({_fmt_bytes(size)})")
        t0 = time.time()
        obj = orig_torch_load(path, *a, **kw)
        VLOG.info(f"torch.load: DONE  {path} in {time.time()-t0:.2f}s")
        return obj

    _torch.load = verbose_torch_load


# --- step 4. Pipeline + sub-module hooks --------------------------------------

def _install_pipeline_hooks() -> None:
    import wan.multitalk as mt
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.clip import CLIPModel
    from wan.modules.vae import WanVAE
    from wan.modules.multitalk_model import WanModel
    from wan.wan_lora import WanLoraWrapper

    # ---- InfiniteTalkPipeline.__init__: step-by-step narration ----
    orig_init = mt.InfiniteTalkPipeline.__init__

    def verbose_init(self, *args, **kwargs):
        _banner("InfiniteTalkPipeline.__init__ START")
        VLOG.info(f"args keys: {list(kwargs.keys())}")
        VLOG.info(f"  checkpoint_dir   = {kwargs.get('checkpoint_dir')}")
        VLOG.info(f"  quant_dir        = {kwargs.get('quant_dir')}")
        VLOG.info(f"  infinitetalk_dir = {kwargs.get('infinitetalk_dir')}")
        VLOG.info(f"  lora_dir         = {kwargs.get('lora_dir')}")
        VLOG.info(f"  quant            = {kwargs.get('quant')}")
        VLOG.info(f"  use_usp          = {kwargs.get('use_usp')}")
        t0 = time.time()
        try:
            orig_init(self, *args, **kwargs)
        except Exception as e:
            VLOG.error(f"InfiniteTalkPipeline.__init__ FAILED after {time.time()-t0:.1f}s: {e}")
            for line in traceback.format_exc().splitlines():
                VLOG.error(line)
            raise
        dt = time.time() - t0

        # Post-init report: which sub-modules loaded, where they live, param count.
        try:
            import torch
            def _describe(mod, label):
                if mod is None:
                    VLOG.info(f"  {label:<14}= None")
                    return
                target = mod if isinstance(mod, torch.nn.Module) else getattr(mod, "model", mod)
                try:
                    n = sum(p.numel() for p in target.parameters())
                    dev = next(target.parameters()).device
                    dtype = next(target.parameters()).dtype
                    VLOG.info(f"  {label:<14}= {target.__class__.__name__} "
                              f"params={n:,} device={dev} dtype={dtype}")
                except StopIteration:
                    VLOG.info(f"  {label:<14}= {target.__class__.__name__} (no parameters)")
            VLOG.info("pipeline components loaded:")
            _describe(getattr(self, "text_encoder", None), "T5")
            _describe(getattr(self, "vae", None),          "VAE")
            _describe(getattr(self, "clip", None),         "CLIP")
            _describe(getattr(self, "model", None),        "DiT")
        except Exception as e:
            VLOG.warning(f"post-init describe failed: {e}")

        VLOG.info(f"InfiniteTalkPipeline.__init__ DONE in {dt:.1f}s")
        _banner("InfiniteTalkPipeline ready")

    mt.InfiniteTalkPipeline.__init__ = verbose_init

    # ---- Sub-module constructors (so we time each one independently) ----
    for mod, name in [(T5EncoderModel, "T5EncoderModel"),
                      (CLIPModel,      "CLIPModel"),
                      (WanVAE,         "WanVAE"),
                      (WanModel,       "WanModel")]:
        _orig = mod.__init__

        def _make(orig, name):
            def patched(self, *a, **kw):
                path_hint = (kw.get("checkpoint_path") or kw.get("vae_pth")
                             or kw.get("tokenizer_path") or "")
                VLOG.info(f"{name}.__init__ START  path={path_hint}")
                t0 = time.time()
                try:
                    orig(self, *a, **kw)
                except Exception as e:
                    VLOG.error(f"{name}.__init__ FAILED after {time.time()-t0:.1f}s: {e}")
                    raise
                VLOG.info(f"{name}.__init__ DONE in {time.time()-t0:.1f}s")
            patched.__name__ = f"verbose_{name}_init"
            return patched

        mod.__init__ = _make(_orig, name)

    # ---- LoRA wrapper: each apply ----
    if hasattr(WanLoraWrapper, "apply_lora"):
        _wrap(WanLoraWrapper, "apply_lora",
              lambda a, kw, res, dt: VLOG.info(
                  f"WanLoraWrapper.apply_lora DONE in {dt:.2f}s "
                  f"(args={a[1:] if len(a)>1 else ()}, scale={kw.get('scale') if 'scale' in kw else (a[2] if len(a)>2 else '?')})"))
    if hasattr(WanLoraWrapper, "load_lora"):
        _wrap(WanLoraWrapper, "load_lora",
              lambda a, kw, res, dt: VLOG.info(
                  f"WanLoraWrapper.load_lora  DONE in {dt:.2f}s path={a[1] if len(a)>1 else ''}"))

    # ---- Vram / device management ----
    _wrap(mt.InfiniteTalkPipeline, "enable_vram_management",
          lambda a, kw, res, dt: VLOG.info(
              f"enable_vram_management DONE in {dt:.1f}s "
              f"(num_persistent_param_in_dit={kw.get('num_persistent_param_in_dit')})"))
    _wrap(mt.InfiniteTalkPipeline, "enable_cpu_offload",
          lambda a, kw, res, dt: VLOG.info(f"enable_cpu_offload DONE"))
    _wrap(mt.InfiniteTalkPipeline, "load_models_to_device",
          lambda a, kw, res, dt: VLOG.info(
              f"load_models_to_device DONE in {dt:.2f}s (want={a[1] if len(a)>1 else kw.get('loadmodel_names')})"))

    # ---- generate_infinitetalk: narrate stages ----
    orig_gen = mt.InfiniteTalkPipeline.generate_infinitetalk

    def verbose_generate(self, input_data, *a, **kw):
        _banner("generate_infinitetalk START")
        VLOG.info(f"  prompt           = {input_data.get('prompt')!r}")
        VLOG.info(f"  cond_video       = {input_data.get('cond_video')}")
        VLOG.info(f"  size_buckget     = {kw.get('size_buckget') or (a[0] if a else '?')}")
        VLOG.info(f"  frame_num        = {kw.get('frame_num')}")
        VLOG.info(f"  sampling_steps   = {kw.get('sampling_steps')}")
        VLOG.info(f"  text_guide_scale = {kw.get('text_guide_scale')}")
        VLOG.info(f"  audio_guide_scale= {kw.get('audio_guide_scale')}")
        VLOG.info(f"  max_frames_num   = {kw.get('max_frames_num')}")
        t0 = time.time()
        try:
            video = orig_gen(self, input_data, *a, **kw)
        except Exception as e:
            VLOG.error(f"generate_infinitetalk FAILED after {time.time()-t0:.1f}s: {e}")
            for line in traceback.format_exc().splitlines():
                VLOG.error(line)
            raise
        dt = time.time() - t0
        try:
            import torch
            if isinstance(video, torch.Tensor):
                VLOG.info(f"generate_infinitetalk DONE in {dt:.1f}s "
                          f"-> video tensor shape={tuple(video.shape)} dtype={video.dtype}")
            else:
                VLOG.info(f"generate_infinitetalk DONE in {dt:.1f}s -> {type(video).__name__}")
        except Exception:
            VLOG.info(f"generate_infinitetalk DONE in {dt:.1f}s")
        _banner("generate_infinitetalk END")
        return video

    mt.InfiniteTalkPipeline.generate_infinitetalk = verbose_generate

    # ---- Per-iteration diffusion step logging ----
    # Wrap WanModel.forward to emit a log line at the *first* forward of each
    # new timestep (the CFG loop calls forward 2-3x per step). We track state
    # on the pipeline object so counters reset per generate_infinitetalk call.
    orig_forward = WanModel.forward

    def _iter_state(self_pipeline):
        # One state dict per WanModel instance.
        s = getattr(self_pipeline, "_verbose_iter_state", None)
        if s is None:
            s = {"last_t": None, "i": 0, "total": None,
                 "t0": None, "step_t0": None, "sub": 0}
            self_pipeline._verbose_iter_state = s
        return s

    def verbose_forward(self, x, *args, **kwargs):
        t = kwargs.get("t")
        # Fallback: `t` might be positional. Try to find it.
        if t is None and args:
            for a in args:
                try:
                    import torch
                    if isinstance(a, torch.Tensor) and a.numel() <= 8:
                        t = a
                        break
                except Exception:
                    pass

        s = _iter_state(self)
        import torch
        t_val = None
        try:
            if isinstance(t, torch.Tensor):
                t_val = float(t.flatten()[0].item())
        except Exception:
            pass

        now = time.time()
        is_new_step = (t_val is not None and t_val != s["last_t"])
        if is_new_step:
            if s["t0"] is None:
                s["t0"] = now
            s["i"] += 1
            s["step_t0"] = now
            s["sub"] = 1
            elapsed = now - s["t0"]
            avg = elapsed / max(1, s["i"])

            # shape peek
            try:
                if isinstance(x, list) and x and isinstance(x[0], torch.Tensor):
                    shape = tuple(x[0].shape)
                    dtype = x[0].dtype
                else:
                    shape, dtype = None, None
            except Exception:
                shape, dtype = None, None

            total = s["total"]
            eta_str = ""
            pct_str = ""
            if total:
                eta = avg * max(0, total - s["i"])
                eta_str = f" eta={eta:.1f}s"
                pct_str = f"  ({s['i']}/{total}, {s['i']/total*100:.1f}%)"

            VLOG.info(
                f"iter step {s['i']}{pct_str}  t={t_val:.2f}  "
                f"latent_shape={shape} dtype={dtype}  "
                f"step_dt~{avg:.2f}s  elapsed={elapsed:.1f}s{eta_str}"
            )
            s["last_t"] = t_val
        else:
            s["sub"] += 1

        return orig_forward(self, x, *args, **kwargs)

    WanModel.forward = verbose_forward

    # Hook the pipeline to reset iter state + populate total when a new
    # generate_infinitetalk call begins. We already wrapped generate_infinitetalk
    # above, so extend that wrapper: wrap once more that resets/sets total.
    orig_gen2 = mt.InfiniteTalkPipeline.generate_infinitetalk

    def generate_with_iter_reset(self, input_data, *a, **kw):
        # Reset per-call counter on the DiT module.
        if getattr(self, "model", None) is not None:
            self.model._verbose_iter_state = {
                "last_t": None, "i": 0,
                "total": kw.get("sampling_steps"),
                "t0": None, "step_t0": None, "sub": 0,
            }
            VLOG.info(f"iter-state reset: total={self.model._verbose_iter_state['total']}")
        return orig_gen2(self, input_data, *a, **kw)

    mt.InfiniteTalkPipeline.generate_infinitetalk = generate_with_iter_reset

    # ---- VAE encode/decode ----
    try:
        orig_decode = mt.WanVAE.decode
        def verbose_vae_decode(self, *a, **kw):
            VLOG.info(f"VAE.decode START (latent batches={len(a[0]) if a else '?'})")
            t0 = time.time()
            out = orig_decode(self, *a, **kw)
            VLOG.info(f"VAE.decode DONE in {time.time()-t0:.1f}s")
            return out
        mt.WanVAE.decode = verbose_vae_decode
    except Exception as e:
        VLOG.warning(f"could not patch WanVAE.decode: {e}")


# --- step 5. Patch tqdm so each bar also logs occasional progress ------------

def _install_tqdm_hook() -> None:
    """Wrap tqdm so diffusion step progress prints through our logger too.

    tqdm's own \\r bars keep animating via stdout (because we set bufsize=0 in
    api.py) -- this is an additional, persistent log line per N steps so the
    structured log file has discrete timestamped entries.
    """
    try:
        import tqdm as _tqdm
    except Exception:
        return

    orig_update = _tqdm.std.tqdm.update

    def verbose_update(self, n=1):
        ret = orig_update(self, n)
        try:
            if self.total and (self.n % max(1, self.total // 20) == 0 or self.n == self.total):
                VLOG.info(f"tqdm[{self.desc or 'bar'}] {self.n}/{self.total} "
                          f"({(self.n/self.total*100):.1f}%) "
                          f"elapsed={self.format_dict.get('elapsed', 0):.1f}s")
        except Exception:
            pass
        return ret

    _tqdm.std.tqdm.update = verbose_update


# --- step 6. Entrypoint -------------------------------------------------------

def main() -> int:
    _banner(f"api_verbose_hooks starting — pid={os.getpid()}")
    VLOG.info(f"python={sys.version.split()[0]}  argv={sys.argv}")
    try:
        import torch
        VLOG.info(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
                  f"device_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            VLOG.info(f"cuda device = {torch.cuda.get_device_name(0)}  "
                      f"runtime={torch.version.cuda}")
    except Exception as e:
        VLOG.warning(f"torch import failed: {e}")

    # Install hooks BEFORE importing the generate module so its `from wan import ...`
    # picks up the patched versions. We import wan.multitalk here to make sure
    # the symbols exist, install patches, then import the script module.
    import wan  # noqa: F401
    _install_loader_hooks()
    _install_pipeline_hooks()
    _install_tqdm_hook()
    VLOG.info("all hooks installed, handing off to generate_infinitetalk.generate()")

    import generate_infinitetalk as gi

    # Pretend we're running the script: let its argparse read sys.argv.
    args = gi._parse_args()
    _banner("parsed CLI args")
    for k, v in sorted(vars(args).items()):
        VLOG.info(f"  {k:<32}= {v}")

    gi.generate(args)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception as e:
        VLOG.error(f"FATAL: {e}")
        for line in traceback.format_exc().splitlines():
            VLOG.error(line)
        rc = 1
    sys.exit(rc)
