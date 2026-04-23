"""
InfiniteTalk REST API

Endpoints:
  POST /jobs                    create job (multipart: image, audio, [prompt, size, mode])
  GET  /jobs/{job_id}           status + progress
  GET  /jobs/{job_id}/download  final .mp4
  GET  /jobs/{job_id}/log       raw subprocess log
  GET  /health                  CUDA / config sanity

EVERYTHING streams to ONE terminal (stdout), unbuffered, in full detail:
  - server lifecycle (startup banner, env, torch, CUDA)
  - every HTTP request (method, path, client, status, duration)
  - every job lifecycle transition
  - every byte of generate_infinitetalk.py's stdout+stderr, live (tqdm bars included)
  - full tracebacks on failure

Run:
    python api.py --host 0.0.0.0 --port 8000 \
        --ckpt-dir        weights/Wan2.1-I2V-14B-480P \
        --wav2vec-dir     weights/chinese-wav2vec2-base \
        --infinitetalk-dir weights/InfiniteTalk/single/infinitetalk.safetensors
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import queue
import shlex
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ----------------------------- unbuffered stdout FIRST -----------------------
# Must happen before anything prints, so every subsequent line flushes immediately.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)  # py3.7+
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import uvicorn


# ----------------------------- unified logging -------------------------------

LOG_FMT = "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-22s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


class _FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


def _install_root_logging() -> None:
    """One handler, stdout, flushes on every emit. Tame all known library loggers."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = _FlushStreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FMT, DATE_FMT))
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Make sure uvicorn/fastapi/etc. all flow through the root handler (no isolation).
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi",
                 "asyncio", "multipart", "PIL"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(logging.INFO)


_install_root_logging()
log = logging.getLogger("infinitetalk.api")
sub_log = logging.getLogger("infinitetalk.sub")


def _banner(msg: str) -> None:
    bar = "=" * 90
    log.info(bar)
    log.info(msg)
    log.info(bar)


# ----------------------------- job model -------------------------------------

STATUSES = ("queued", "running", "completed", "failed")


@dataclass
class Job:
    id: str
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    prompt: str = ""
    size: str = "infinitetalk-480"
    mode: str = "clip"
    image_path: str = ""
    audio_path: str = ""
    output_path: str = ""
    # Length cap for generation (frames at the pipeline's 25 fps output rate).
    # 0 means "auto-derive from audio duration + small safety margin".
    max_frames: int = 0
    # When True, generated video length is capped by max_frames alone. When
    # False (default), the pipeline auto-stops once the audio is consumed.
    offload_model: bool = False
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def touch(self) -> None:
        self.updated_at = datetime.utcnow().isoformat()


# ----------------------------- helpers ---------------------------------------

def _audio_duration_seconds(path: Path) -> float:
    """Read duration of an audio file. Uses soundfile (fast, header-only read).
    Falls back to librosa if soundfile can't handle the format."""
    try:
        import soundfile as sf
        with sf.SoundFile(str(path)) as f:
            return float(f.frames) / float(f.samplerate)
    except Exception as e:
        log.warning(f"soundfile failed on {path}: {e}; falling back to librosa")
        import librosa
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return float(len(y)) / float(sr)


def _audio_to_max_frames(duration_s: float, fps: int = 25,
                         safety_frames: int = 25) -> int:
    """Convert audio duration in seconds to a frame-count cap for the pipeline.
    Uses 25 fps (the output save rate used by save_video_ffmpeg) plus a 1-second
    safety margin so the pipeline's chunked loop doesn't stop one chunk short."""
    return int(duration_s * fps) + safety_frames


# ----------------------------- job store -------------------------------------

class JobStore:
    """Thread-safe in-memory job registry + single-GPU serial worker."""

    def __init__(self, storage_root: Path, runner) -> None:
        self.root = storage_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.q: "queue.Queue[str]" = queue.Queue()
        self.runner = runner
        self.worker = threading.Thread(target=self._loop, name="job-worker", daemon=True)
        self.worker.start()
        log.info("Job worker thread started.")

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def add(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.id] = job
        self.q.put(job.id)
        log.info(f"[{job.id}] queued (pending={self.q.qsize()})")

    def get(self, job_id: str) -> Optional[Job]:
        with self.lock:
            return self.jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> None:
        with self.lock:
            j = self.jobs.get(job_id)
            if not j:
                return
            for k, v in kwargs.items():
                setattr(j, k, v)
            j.touch()
        # Log every state change so the terminal shows it.
        detail = ", ".join(f"{k}={v}" for k, v in kwargs.items() if k != "error")
        if detail:
            log.info(f"[{job_id}] update: {detail}")
        if "error" in kwargs and kwargs["error"]:
            log.error(f"[{job_id}] error set:\n{kwargs['error']}")

    def _loop(self) -> None:
        while True:
            job_id = self.q.get()
            job = self.get(job_id)
            if job is None:
                continue
            try:
                _banner(f"[{job_id}] STARTING generation")
                self.update(job_id, status="running", stage="starting", progress=0.01)
                self.runner(job, self.update)
                _banner(f"[{job_id}] COMPLETED -> {job.output_path}")
            except Exception as e:
                tb = traceback.format_exc()
                log.error(f"[{job_id}] FAILED: {e}")
                for line in tb.splitlines():
                    log.error(f"[{job_id}] {line}")
                self.update(job_id, status="failed", stage="error",
                            error=f"{e}\n{tb}")


# ----------------------------- runner ----------------------------------------

class InfiniteTalkRunner:
    """Shells out to generate_infinitetalk.py, streams every byte live."""

    def __init__(self, args) -> None:
        self.args = args
        self.repo_root = Path(__file__).resolve().parent

    def __call__(self, job: Job, update) -> None:
        jdir = Path(self.args.storage_root) / job.id
        jdir.mkdir(parents=True, exist_ok=True)

        # ---- 1. Build the pipeline's input JSON ----
        update(job.id, stage="preparing_inputs", progress=0.05)
        input_json = {
            "prompt": job.prompt or "A person talking to the camera.",
            "cond_video": str(Path(job.image_path).resolve()),
            "cond_audio": {"person1": str(Path(job.audio_path).resolve())},
        }
        input_json_path = jdir / "input.json"
        input_json_path.write_text(json.dumps(input_json, indent=2), encoding="utf-8")
        log.info(f"[{job.id}] wrote input.json:\n{json.dumps(input_json, indent=2)}")

        save_file_stem = str(jdir / "output")
        job.output_path = save_file_stem + ".mp4"

        # ---- 2. Assemble CLI ----
        # We invoke api_verbose_hooks.py, which installs per-stage/per-iteration
        # logging hooks before handing argv to generate_infinitetalk.generate().
        cmd = [
            sys.executable, "-u", str(self.repo_root / "api_verbose_hooks.py"),
            "--task", "infinitetalk-14B",
            "--size", job.size,
            "--mode", job.mode,
            "--ckpt_dir", self.args.ckpt_dir,
            "--wav2vec_dir", self.args.wav2vec_dir,
            "--infinitetalk_dir", self.args.infinitetalk_dir,
            "--input_json", str(input_json_path),
            "--save_file", save_file_stem,
            "--audio_save_dir", str(jdir / "audio_cache"),
        ]
        if self.args.quant:
            cmd += ["--quant", self.args.quant]
            if self.args.quant_dir:
                cmd += ["--quant_dir", self.args.quant_dir]
        if self.args.num_persistent_param_in_dit is not None:
            cmd += ["--num_persistent_param_in_dit",
                    str(self.args.num_persistent_param_in_dit)]
        if self.args.use_teacache:
            cmd += ["--use_teacache"]
        if self.args.lora_dir:
            cmd += ["--lora_dir", *self.args.lora_dir]
            if self.args.lora_scale:
                cmd += ["--lora_scale", *[str(s) for s in self.args.lora_scale]]

        # Frame budget: pass our per-job cap so streaming mode covers the full
        # audio duration instead of truncating at the upstream default of 1000
        # frames (= 40s @ 25fps).
        if job.max_frames > 0:
            cmd += ["--max_frame_num", str(job.max_frames)]

        # Offload default: on single A100 (80GB or 40GB) we have plenty of VRAM
        # to keep the model resident across chunks. The upstream auto-default
        # offloads the 18GB DiT to CPU at every chunk boundary, costing ~10s
        # per chunk of PCIe transfer. Only opt in if the user explicitly asks.
        cmd += ["--offload_model", "True" if job.offload_model else "False"]

        if self.args.extra_args:
            cmd += self.args.extra_args

        pretty = " ".join(shlex.quote(c) for c in cmd)
        log.info(f"[{job.id}] launching subprocess:\n  {pretty}")

        # ---- 3. Launch with fully inherited/unbuffered env ----
        update(job.id, stage="loading_models", progress=0.10)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # Make tqdm output visible (tqdm checks TTY; some builds silence otherwise).
        env.setdefault("TQDM_DISABLE", "0")
        env.setdefault("FORCE_COLOR", "0")  # avoid ANSI junk in the log file

        log_file = jdir / "log.txt"
        with log_file.open("w", encoding="utf-8", buffering=1) as lf:
            lf.write(f"$ {pretty}\n\n")

            proc = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,               # completely unbuffered
                env=env,
            )

            stage_signals = [
                ("Creating infinitetalk pipeline", "loading_pipeline",   0.15),
                ("Load checkpoint",                "loading_checkpoint", 0.25),
                ("Generating video",               "generating",         0.40),
                ("Saving generated video",         "encoding_video",     0.92),
            ]
            seen: set[str] = set()
            prefix = f"[{job.id}] "

            # Byte-by-byte read so \r-animated tqdm bars render live and we
            # never block waiting on a full line.
            assert proc.stdout is not None
            buf = bytearray()
            while True:
                chunk = proc.stdout.read(1)
                if not chunk:
                    break
                # Mirror raw bytes to terminal AND log file so progress bars look right.
                try:
                    sys.stdout.buffer.write(b"" if not buf and chunk in (b"\n", b"\r") else b"")
                except Exception:
                    pass

                if chunk in (b"\n", b"\r"):
                    line = buf.decode("utf-8", errors="replace")
                    buf.clear()
                    if line:
                        # Per-line: route through logger so timestamp+job-id appear.
                        sub_log.info(prefix + line.rstrip())
                        lf.write(line + ("\n" if chunk == b"\n" else "\r"))
                        for needle, stage, prog in stage_signals:
                            if needle in line and needle not in seen:
                                seen.add(needle)
                                update(job.id, stage=stage, progress=prog)
                    else:
                        # Bare \r or \n — still mirror the byte for liveness.
                        try:
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.flush()
                        except Exception:
                            pass
                        lf.write(chunk.decode("utf-8", errors="replace"))
                else:
                    buf.extend(chunk)
                    # Also mirror the raw byte to stdout so tqdm carriage-return
                    # progress bars animate in the terminal.
                    try:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.flush()
                    except Exception:
                        pass

            # Drain any trailing partial line.
            if buf:
                tail = buf.decode("utf-8", errors="replace")
                sub_log.info(prefix + tail.rstrip())
                lf.write(tail + "\n")

            rc = proc.wait()
            log.info(f"[{job.id}] subprocess exit code: {rc}")
            lf.write(f"\n[exit code: {rc}]\n")

            if rc != 0:
                raise RuntimeError(
                    f"generate_infinitetalk.py exited with code {rc}. "
                    f"Full log: {log_file}"
                )

        if not Path(job.output_path).exists():
            raise RuntimeError(f"Expected output not found: {job.output_path}")

        update(job.id, status="completed", stage="completed", progress=1.0)
        log.info(f"[{job.id}] output file ready: {job.output_path} "
                 f"({Path(job.output_path).stat().st_size} bytes)")


# ----------------------------- FastAPI app -----------------------------------

def create_app(runner_args) -> FastAPI:
    app = FastAPI(title="InfiniteTalk API", version="1.0")
    storage = Path(runner_args.storage_root).resolve()
    store = JobStore(storage, InfiniteTalkRunner(runner_args))

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.time()
        client = f"{request.client.host}:{request.client.port}" if request.client else "?"
        log.info(f"HTTP --> {request.method} {request.url.path} from {client}")
        try:
            response = await call_next(request)
        except Exception as e:
            log.error(f"HTTP !! {request.method} {request.url.path} raised {e}")
            for line in traceback.format_exc().splitlines():
                log.error(line)
            raise
        dt_ms = (time.time() - t0) * 1000
        log.info(f"HTTP <-- {request.method} {request.url.path} "
                 f"{response.status_code} in {dt_ms:.1f}ms")
        return response

    @app.on_event("startup")
    def _startup():
        _banner("InfiniteTalk API ready")
        log.info(f"listening on http://{runner_args.host}:{runner_args.port}")
        log.info(f"storage_root    = {storage}")
        log.info(f"ckpt_dir        = {runner_args.ckpt_dir}")
        log.info(f"wav2vec_dir     = {runner_args.wav2vec_dir}")
        log.info(f"infinitetalk_dir= {runner_args.infinitetalk_dir}")

    @app.get("/health")
    def health():
        info: dict = {"ok": True, "pending_jobs": store.q.qsize(),
                      "storage_root": str(storage),
                      "ckpt_dir": runner_args.ckpt_dir}
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = (torch.cuda.get_device_name(0)
                                   if torch.cuda.is_available() else None)
            info["torch_version"] = torch.__version__
        except Exception as e:
            info["torch_import_error"] = str(e)
        return info

    @app.post("/jobs")
    async def create_job(
        image: UploadFile = File(..., description="Reference image (png/jpg) or short video (mp4)"),
        audio: UploadFile = File(..., description="Speech audio, wav preferred"),
        prompt: str = Form("A person talking to the camera."),
        size: str = Form("infinitetalk-480"),
        mode: str = Form("streaming"),
        max_frames: int = Form(0, description="Cap on generated frames (25fps). 0 = auto from audio length."),
        offload_model: bool = Form(False, description="Offload DiT to CPU between chunks. Slower; use only if VRAM < 24 GB."),
    ):
        if size not in ("infinitetalk-480", "infinitetalk-720"):
            raise HTTPException(400, "size must be infinitetalk-480 or infinitetalk-720")
        if mode not in ("clip", "streaming"):
            raise HTTPException(400, "mode must be 'clip' or 'streaming'")

        job_id = uuid.uuid4().hex[:12]
        jdir = store.job_dir(job_id)
        jdir.mkdir(parents=True, exist_ok=True)

        img_ext = Path(image.filename or "input.png").suffix or ".png"
        aud_ext = Path(audio.filename or "input.wav").suffix or ".wav"
        img_path = jdir / f"input_image{img_ext}"
        aud_path = jdir / f"input_audio{aud_ext}"

        img_bytes = await image.read()
        aud_bytes = await audio.read()
        img_path.write_bytes(img_bytes)
        aud_path.write_bytes(aud_bytes)

        log.info(f"[{job_id}] upload: image={image.filename} -> {img_path.name} "
                 f"({len(img_bytes)} B, ct={image.content_type})")
        log.info(f"[{job_id}] upload: audio={audio.filename} -> {aud_path.name} "
                 f"({len(aud_bytes)} B, ct={audio.content_type})")

        # Auto-derive max_frames from audio duration if the caller didn't set one.
        # Upstream's default of 1000 caps output at 40s @ 25 fps, silently
        # truncating any longer audio. We size the cap to cover the whole audio.
        if max_frames <= 0:
            try:
                dur = _audio_duration_seconds(aud_path)
                max_frames = _audio_to_max_frames(dur)
                log.info(f"[{job_id}] audio duration={dur:.2f}s -> "
                         f"auto max_frames={max_frames} (25 fps + 1s safety)")
            except Exception as e:
                log.warning(f"[{job_id}] could not read audio duration ({e}); "
                            f"falling back to upstream default (1000 frames / 40s)")
                max_frames = 0  # pipeline keeps its default
        else:
            log.info(f"[{job_id}] caller-set max_frames={max_frames}")

        log.info(f"[{job_id}] params: size={size} mode={mode} "
                 f"offload_model={offload_model} prompt={prompt!r}")

        job = Job(
            id=job_id, prompt=prompt, size=size, mode=mode,
            image_path=str(img_path), audio_path=str(aud_path),
            max_frames=max_frames, offload_model=offload_model,
        )
        store.add(job)
        return {"job_id": job_id, "status": job.status,
                "max_frames": max_frames, "offload_model": offload_model}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        j = store.get(job_id)
        if not j:
            raise HTTPException(404, "job not found")
        return asdict(j)

    @app.get("/jobs/{job_id}/download")
    def job_download(job_id: str):
        j = store.get(job_id)
        if not j:
            raise HTTPException(404, "job not found")
        if j.status != "completed":
            raise HTTPException(409, f"job not ready (status={j.status})")
        if not j.output_path or not Path(j.output_path).exists():
            raise HTTPException(500, "output file missing on disk")
        log.info(f"[{job_id}] serving download: {j.output_path}")
        return FileResponse(j.output_path, media_type="video/mp4",
                            filename=f"infinitetalk_{job_id}.mp4")

    @app.get("/jobs/{job_id}/log")
    def job_log_ep(job_id: str):
        j = store.get(job_id)
        if not j:
            raise HTTPException(404, "job not found")
        log_path = store.job_dir(job_id) / "log.txt"
        if not log_path.exists():
            return JSONResponse({"log": ""})
        return FileResponse(log_path, media_type="text/plain")

    return app


# ----------------------------- CLI -------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--storage-root", default="./api_storage")

    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--wav2vec-dir", required=True)
    p.add_argument("--infinitetalk-dir", required=True)

    p.add_argument("--quant", choices=["int8", "fp8"], default=None)
    p.add_argument("--quant-dir", default=None)
    p.add_argument("--num-persistent-param-in-dit", type=int, default=None,
                   help="Lower this (e.g. 0) for low-VRAM cards.")
    p.add_argument("--use-teacache", action="store_true")
    p.add_argument("--lora-dir", nargs="+", default=None)
    p.add_argument("--lora-scale", nargs="+", type=float, default=None)
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                   help="Anything after this flag is passed through verbatim.")

    return p.parse_args()


def _log_environment():
    _banner("InfiniteTalk API starting")
    log.info(f"python      = {sys.version.split()[0]} ({sys.executable})")
    log.info(f"platform    = {platform.platform()}")
    log.info(f"cwd         = {os.getcwd()}")
    log.info(f"argv        = {sys.argv}")
    try:
        import torch
        log.info(f"torch       = {torch.__version__}")
        log.info(f"cuda avail  = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"cuda device = {torch.cuda.get_device_name(0)}")
            log.info(f"device count= {torch.cuda.device_count()}")
            log.info(f"cuda runtime= {torch.version.cuda}")
        else:
            log.warning("CUDA not available — generation will fail.")
    except Exception as e:
        log.warning(f"torch import failed: {e}")


def main():
    args = _parse()
    _log_environment()
    for path, label in [
        (args.ckpt_dir, "ckpt_dir"),
        (args.wav2vec_dir, "wav2vec_dir"),
        (args.infinitetalk_dir, "infinitetalk_dir"),
    ]:
        marker = "OK " if Path(path).exists() else "MISSING"
        log.info(f"{marker} {label} = {path}")

    app = create_app(args)

    # Uvicorn config: inherit our root logger (no separate log config).
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=None,       # don't clobber our handlers
        access_log=True,
        use_colors=False,
    )


if __name__ == "__main__":
    main()
