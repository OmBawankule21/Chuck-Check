"""Command-line interface.

    chuckcheck analyze my_bowling.mp4 --out runs/ball1
    chuckcheck demo --extension 24
    chuckcheck reanalyze runs/ball1 --uah 212 --release 236
    chuckcheck download-models
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .config import AnalysisConfig


def _progress():
    last = {"msg": "", "t": 0.0}

    def cb(frac: float, msg: str):
        now = time.time()
        if now - last["t"] < 0.2 and frac < 1.0:
            return
        last["t"] = now
        bar = "#" * int(30 * frac)
        sys.stderr.write(f"\r[{bar:<30}] {frac * 100:5.1f}%  {msg[:60]:<60}")
        sys.stderr.flush()
        if frac >= 1.0 and msg == "Done":
            sys.stderr.write("\n")
    return cb


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--out", default=None, help="output folder (default: runs/<name>)")
    p.add_argument("--arm", choices=["auto", "left", "right"], default="auto", help="bowling arm")
    p.add_argument("--model", choices=["lite", "full", "heavy"], default="full", help="pose model size")
    p.add_argument("--model-path", default=None, help="local pose_landmarker .task file")
    p.add_argument("--release-method", choices=["arm-angle", "wrist-speed"], default="arm-angle")
    p.add_argument("--release-offset", type=float, default=15.0,
                   help="degrees past peak upper-arm elevation at which release is assumed")
    p.add_argument("--pose-error", type=float, default=4.0, help="assumed pose error (deg) for the band")
    p.add_argument("--slowmo", type=float, default=0.25, help="slow-motion speed around release (1 = off)")
    p.add_argument("--no-video", action="store_true", help="skip rendering the annotated video")


def _cfg(a) -> AnalysisConfig:
    return AnalysisConfig(model=a.model, model_path=a.model_path, bowling_arm=a.arm,
                          release_method=a.release_method, release_offset_deg=a.release_offset,
                          pose_error_deg=a.pose_error, slowmo=a.slowmo)


def _summary(res, paths):
    print()
    for dl in res.deliveries:
        lg = dl.legality
        print(f"Delivery #{dl.index + 1}: {lg.verdict:<13} extension {lg.extension_deg:5.1f}°  "
              f"(band {lg.band_low_deg:.1f}–{lg.band_high_deg:.1f}°, quality {lg.quality}, "
              f"UAH f{dl.events.uah} → release f{dl.events.release})")
        for w in lg.warnings:
            print(f"   ! {w}")
    for w in res.warnings:
        print(f"   ! {w}")
    print("\nOutputs:")
    for k, v in paths.items():
        print(f"  {k:<6} {v}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="chuckcheck", description="Cricket bowling-action legality analyser")
    ap.add_argument("--version", action="version", version=f"chuckcheck {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze", help="analyse a bowling video")
    pa.add_argument("video")
    _add_common(pa)

    pd_ = sub.add_parser("demo", help="run on a synthetic bowler with a known elbow extension")
    pd_.add_argument("--extension", type=float, default=22.0, help="true elbow extension (deg)")
    pd_.add_argument("--fps", type=float, default=120.0)
    pd_.add_argument("--side", choices=["left", "right"], default="right")
    _add_common(pd_)

    pr = sub.add_parser("reanalyze", help="re-run analysis from a saved run folder (fast, no pose model)")
    pr.add_argument("run_dir")
    pr.add_argument("--video", default=None, help="original video (for rendering/key frames)")
    pr.add_argument("--uah", type=int, default=None, help="manual upper-arm-horizontal frame")
    pr.add_argument("--release", type=int, default=None, help="manual ball-release frame")
    pr.add_argument("--delivery", type=int, default=0, help="which delivery the manual frames apply to")
    _add_common(pr)

    sub.add_parser("download-models", help="pre-download the pose models")

    a = ap.parse_args(argv)

    if a.cmd == "download-models":
        from .models import get_model_path
        for v in ("lite", "full", "heavy"):
            print(v, "->", get_model_path(v))
        return 0

    from .pipeline import AnalysisError
    from .runner import run_demo, run_video
    try:
        if a.cmd == "analyze":
            out = a.out or str(Path("runs") / Path(a.video).stem)
            res, paths = run_video(a.video, out, _cfg(a), render=not a.no_video, progress=_progress())
        elif a.cmd == "demo":
            out = a.out or f"runs/demo_ext{a.extension:g}"
            res, paths, b = run_demo(out, a.extension, a.fps, a.side, _cfg(a), render=not a.no_video,
                                     progress=_progress())
            print(f"\nGround truth extension: {a.extension:.1f}°  (UAH f{b.true_uah}, release f{b.true_release})")
        else:
            from .pose import PoseSequence
            seq = PoseSequence.load(Path(a.run_dir) / "pose.npz")
            ov = {a.delivery: {"uah": a.uah, "release": a.release}} if (a.uah or a.release) else None
            video = a.video or seq.source
            if not Path(video).is_file():
                from .pipeline import analyze_sequence
                from .report import write_outputs
                res = analyze_sequence(seq, _cfg(a), ov)
                paths = write_outputs(res, a.out or a.run_dir)
            else:
                res, paths = run_video(video, a.out or a.run_dir, _cfg(a), render=not a.no_video,
                                       overrides=ov, progress=_progress(), sequence=seq)
    except AnalysisError as e:
        print(f"\nAnalysis failed: {e}", file=sys.stderr)
        return 2
    _summary(res, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
