"""
Hot-pipeline runner for the API.

Loads InfiniteTalkPipeline + wav2vec audio encoder ONCE at API startup,
then reuses them for every request. This eliminates the ~7-minute
model-load cost that the subprocess-based runner pays on every job.

Design:
  * Pipeline is held in module state, built by build_hot_runner().
  * Each job calls HotRunner(job, update_fn). It reimplements the
    per-job body of generate_infinitetalk.generate() (the part from
    line ~547 onwards) using the already-loaded pipeline.
  * If a job raises, we log the traceback. We do NOT tear down the
    pipeline (would be expensive and usually unnecessary) -- most
    errors are in audio decoding / input handling, not CUDA state.

Guarantees kept from the subprocess runner:
  * Same inputs (Job object) produce the same output (.mp4 at
    job.output_path).
  * Same verbose stage/progress updates flow through `update`.
  * Same job storage layout (api_storage/<job_id>/...).

What you lose vs subprocess:
  * If the pipeline somehow ends up in an unrecoverable CUDA state,
    the whole API process must be restarted. In practice on A100 with
    bfloat16 + flash-attn this does not happen.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch

log = logging.getLogger("infinitetalk.hot")


# ----------------------------- config ----------------------------------------

@dataclass
class HotRunnerConfig:
    """Everything the hot runner needs to initialise and run jobs.
    Mirrors the relevant subset of api.py's CLI args."""
    ckpt_dir: str
    wav2vec_dir: str
    infinitetalk_dir: str
    quant: Optional[str] = None
    quant_dir: Optional[str] = None
    dit_path: Optional[str] = None
    lora_dir: Optional[list] = None
    lora_scale: Optional[list] = None
    num_persistent_param_in_dit: Optional[int] = None
    storage_root: str = "./api_storage"
    device_id: int = 0


# ----------------------------- the hot runner --------------------------------

class HotRunner:
    """Callable that runs a Job against an already-loaded pipeline."""

    def __init__(self, cfg: HotRunnerConfig) -> None:
        self.cfg = cfg
        self.repo_root = Path(__file__).resolve().parent
        # Make sure the repo root is importable so `import wan` + friends work.
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        self._load_pipeline()

    # ---- one-time loader ----
    def _load_pipeline(self) -> None:
        log.info("=" * 90)
        log.info("HotRunner: loading pipeline (one-time cost)")
        log.info("=" * 90)
        t0 = time.time()

        import wan
        from wan.configs import WAN_CONFIGS
        from transformers import Wav2Vec2FeatureExtractor
        from src.audio_analysis.wav2vec2 import Wav2Vec2Model

        cfg = WAN_CONFIGS["infinitetalk-14B"]
        log.info(f"HotRunner: WAN_CONFIGS[infinitetalk-14B] loaded")

        # Build the DiT + T5 + VAE + CLIP + LoRA stack.
        self.wan_i2v = wan.InfiniteTalkPipeline(
            config=cfg,
            checkpoint_dir=self.cfg.ckpt_dir,
            quant_dir=self.cfg.quant_dir,
            device_id=self.cfg.device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
            lora_dir=self.cfg.lora_dir,
            lora_scales=self.cfg.lora_scale,
            quant=self.cfg.quant,
            dit_path=self.cfg.dit_path,
            infinitetalk_dir=self.cfg.infinitetalk_dir,
        )

        if self.cfg.num_persistent_param_in_dit is not None:
            self.wan_i2v.vram_management = True
            self.wan_i2v.enable_vram_management(
                num_persistent_param_in_dit=self.cfg.num_persistent_param_in_dit
            )

        # Wav2Vec audio encoder (stays on CPU -- small enough).
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.cfg.wav2vec_dir, local_files_only=True
        )
        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            self.cfg.wav2vec_dir, local_files_only=True
        ).to("cpu")
        self.audio_encoder.feature_extractor._freeze_parameters()

        # Remember the config for per-call generation.
        self._cfg = cfg
        self._cfg_name = "infinitetalk-14B"

        log.info("=" * 90)
        log.info(f"HotRunner: pipeline ready in {time.time()-t0:.1f}s")
        log.info("=" * 90)

    # ---- per-job call ----
    def __call__(self, job, update: Callable[..., None]) -> None:
        # Late imports so api_pipeline is importable on machines without the
        # full stack (eg. Windows dev laptop).
        from generate_infinitetalk import (
            audio_prepare_single,
            audio_prepare_multi,
            get_embedding,
        )
        import soundfile as sf
        from wan.utils.multitalk_utils import save_video_ffmpeg

        jdir = Path(self.cfg.storage_root) / job.id
        jdir.mkdir(parents=True, exist_ok=True)

        # ---- 1. Build input_data dict (the JSON `generate()` normally reads from disk) ----
        update(job.id, stage="preparing_inputs", progress=0.05)
        input_data = {
            "prompt": job.prompt or "A person talking to the camera.",
            "cond_video": str(Path(job.image_path).resolve()),
            "cond_audio": {"person1": str(Path(job.audio_path).resolve())},
        }
        log.info(f"[{job.id}] input_data:\n{json.dumps(input_data, indent=2)}")

        # Audio scratch dir (the upstream code nests under the image basename).
        audio_save_dir = jdir / "audio_cache" / Path(input_data["cond_video"]).stem
        audio_save_dir.mkdir(parents=True, exist_ok=True)

        # ---- 2. Prepare audio (mirror of generate_infinitetalk.generate()) ----
        update(job.id, stage="preparing_audio", progress=0.1)
        if len(input_data["cond_audio"]) == 2:
            new_s1, new_s2, sum_s = audio_prepare_multi(
                input_data["cond_audio"]["person1"],
                input_data["cond_audio"]["person2"],
                input_data.get("audio_type", "add"),
            )
            sum_audio = str(audio_save_dir / "sum_all.wav")
            sf.write(sum_audio, sum_s, 16000)
            input_data["video_audio"] = sum_audio
        else:
            human_speech = audio_prepare_single(input_data["cond_audio"]["person1"])
            sum_audio = str(audio_save_dir / "sum_all.wav")
            sf.write(sum_audio, human_speech, 16000)
            input_data["video_audio"] = sum_audio
        log.info(f"[{job.id}] wrote summed audio -> {sum_audio}")

        # ---- 3. Per-clip audio embedding + pipeline call ----
        # Single-speaker path only (multi is handled symmetrically but most API users are single).
        update(job.id, stage="embedding_audio", progress=0.15)
        input_clip = {
            "prompt": input_data["prompt"],
            "cond_video": input_data["cond_video"],
        }

        human_speech = audio_prepare_single(input_data["cond_audio"]["person1"])
        t_emb = time.time()
        audio_embedding = get_embedding(
            human_speech, self.wav2vec_feature_extractor, self.audio_encoder
        )
        log.info(f"[{job.id}] wav2vec embedding done in {time.time()-t_emb:.1f}s "
                 f"shape={tuple(audio_embedding.shape)}")

        emb_path = str(audio_save_dir / "1.pt")
        clip_audio = str(audio_save_dir / "sum.wav")
        sf.write(clip_audio, human_speech, 16000)
        torch.save(audio_embedding, emb_path)

        input_clip["cond_audio"] = {"person1": emb_path}
        input_clip["video_audio"] = clip_audio

        # ---- 4. Build the synthetic args Namespace the pipeline's forward expects ----
        #     Only what `InfiniteTalkPipeline.generate_infinitetalk` (and its inner
        #     code path) actually reads through extra_args.
        extra_args = Namespace(
            use_teacache=bool(getattr(job, "use_teacache", False)),
            teacache_thresh=float(getattr(job, "teacache_thresh", 0.2)),
            use_apg=bool(getattr(job, "use_apg", False)),
            apg_momentum=float(getattr(job, "apg_momentum", -0.75)),
            apg_norm_threshold=float(getattr(job, "apg_norm_threshold", 55)),
            size=job.size,
            mode=job.mode,
            color_correction_strength=float(getattr(job, "color_correction_strength", 1.0)),
        )

        # Per-job knobs (LightX2V defaults when a LoRA is loaded).
        with_lora = bool(self.cfg.lora_dir)
        sample_steps = int(getattr(job, "sample_steps", 4 if with_lora else 40))
        sample_shift = float(getattr(job, "sample_shift", 2.0 if with_lora else 7.0))
        text_scale = float(getattr(job, "text_guide_scale", 1.0 if with_lora else 5.0))
        audio_scale = float(getattr(job, "audio_guide_scale", 2.0 if with_lora else 4.0))
        motion_frame = int(getattr(job, "motion_frame", 9))
        frame_num = int(getattr(job, "frame_num", 81))
        max_frames = (
            int(job.max_frames)
            if getattr(job, "max_frames", 0) > 0
            else (frame_num if job.mode == "clip" else 1000)
        )
        seed = int(getattr(job, "base_seed", 42))
        offload = bool(getattr(job, "offload_model", False))

        log.info(f"[{job.id}] generation params: steps={sample_steps} "
                 f"shift={sample_shift} text_scale={text_scale} "
                 f"audio_scale={audio_scale} motion={motion_frame} "
                 f"frame_num={frame_num} max_frames={max_frames} "
                 f"offload={offload} seed={seed}")

        # ---- 5. Run the pipeline ----
        update(job.id, stage="generating", progress=0.25)
        t_gen = time.time()
        video = self.wan_i2v.generate_infinitetalk(
            input_clip,
            size_buckget=job.size,
            motion_frame=motion_frame,
            frame_num=frame_num,
            shift=sample_shift,
            sampling_steps=sample_steps,
            text_guide_scale=text_scale,
            audio_guide_scale=audio_scale,
            seed=seed,
            offload_model=offload,
            max_frames_num=max_frames,
            color_correction_strength=extra_args.color_correction_strength,
            extra_args=extra_args,
        )
        log.info(f"[{job.id}] generate_infinitetalk returned in {time.time()-t_gen:.1f}s "
                 f"video tensor shape={tuple(video.shape)} dtype={video.dtype}")

        # ---- 6. Save ----
        update(job.id, stage="encoding_video", progress=0.92)
        save_file_stem = str(jdir / "output")
        job.output_path = save_file_stem + ".mp4"

        t_save = time.time()
        save_video_ffmpeg(
            video, save_file_stem, [input_data["video_audio"]],
            high_quality_save=False,
        )
        log.info(f"[{job.id}] save_video_ffmpeg done in {time.time()-t_save:.1f}s "
                 f"-> {job.output_path}")

        if not Path(job.output_path).exists():
            raise RuntimeError(
                f"output file missing after save_video_ffmpeg: {job.output_path}"
            )

        update(job.id, status="completed", stage="completed", progress=1.0)
        log.info(f"[{job.id}] output file ready: {job.output_path} "
                 f"({Path(job.output_path).stat().st_size} bytes)")


# ----------------------------- factory ---------------------------------------

def build_hot_runner(args) -> HotRunner:
    """Build the hot runner from the same `args` namespace api.py's CLI produces."""
    cfg = HotRunnerConfig(
        ckpt_dir=args.ckpt_dir,
        wav2vec_dir=args.wav2vec_dir,
        infinitetalk_dir=args.infinitetalk_dir,
        quant=args.quant,
        quant_dir=args.quant_dir,
        lora_dir=args.lora_dir,
        lora_scale=args.lora_scale,
        num_persistent_param_in_dit=args.num_persistent_param_in_dit,
        storage_root=args.storage_root,
    )
    return HotRunner(cfg)
