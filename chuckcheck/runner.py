"""High-level helpers that tie pose extraction, analysis, rendering and reporting together."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .config import AnalysisConfig
from .pipeline import AnalysisResult, analyze_sequence
from .pose import PoseExtractor, PoseSequence, iter_frames
from .render import render_video
from .report import write_outputs
from .synthetic import SyntheticBowler, SyntheticSpec

Progress = Optional[Callable[[float, str], None]]


def _stage(progress: Progress, lo: float, hi: float) -> Progress:
    if progress is None:
        return None
    return lambda f, msg: progress(lo + (hi - lo) * f, msg)


def run_video(video: str | Path, out_dir: str | Path, cfg: AnalysisConfig | None = None,
              render: bool = True, overrides: dict | None = None, progress: Progress = None,
              sequence: PoseSequence | None = None) -> tuple[AnalysisResult, dict]:
    """Full pipeline for a real video. Pass ``sequence`` to skip pose estimation."""
    cfg = cfg or AnalysisConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if sequence is None:
        sequence = PoseExtractor(cfg).extract(video, _stage(progress, 0.0, 0.6 if render else 0.9))
        sequence.save(out / "pose.npz")
    res = analyze_sequence(sequence, cfg, overrides)
    paths = {"pose": str(out / "pose.npz")}
    if render:
        paths["video"] = render_video(res, iter_frames(video), out / "annotated.mp4", _stage(progress, 0.6, 0.95))
    paths.update(write_outputs(res, out, lambda: iter_frames(video)))
    if progress:
        progress(1.0, "Done")
    return res, paths


def run_demo(out_dir: str | Path, extension_deg: float = 22.0, fps: float = 120.0, side: str = "right",
             cfg: AnalysisConfig | None = None, render: bool = True, progress: Progress = None
             ) -> tuple[AnalysisResult, dict, SyntheticBowler]:
    """Synthetic bowler with a known elbow extension: no video or model download needed."""
    cfg = cfg or AnalysisConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bowler = SyntheticBowler(SyntheticSpec(extension_deg=extension_deg, fps=fps, side=side))
    res = analyze_sequence(bowler.sequence, cfg)
    paths = {}
    if render:
        paths["video"] = render_video(res, bowler.frames(), out / "annotated.mp4", _stage(progress, 0.0, 0.9))
    paths.update(write_outputs(res, out, bowler.frames))
    if progress:
        progress(1.0, "Done")
    return res, paths, bowler
