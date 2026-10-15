"""ICC-style elbow-extension measurement, uncertainty and verdict."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product

import numpy as np

from .kinematics import Events, elbow_flexion
from .filters import odd_window, smooth

VERDICT_STYLE = {
    "LEGAL": ("#22c55e", "Elbow extension is comfortably within the 15° limit."),
    "LIKELY LEGAL": ("#a3e635", "Best estimate is within the limit, but the error band crosses 15°."),
    "BORDERLINE": ("#f59e0b", "Best estimate is above 15°, but the error band includes legal values. A lab test is advised."),
    "SUSPECT": ("#ef4444", "Elbow extension is above 15° even at the low end of the error band."),
}


def max_extension(flex: np.ndarray, i0: int, i1: int) -> tuple[float, int, int]:
    """Largest decrease of elbow flexion from any frame to any later frame in [i0, i1].

    This is the amount the elbow *straightens* inside the ICC window
    (upper-arm-horizontal -> ball release). Returns (degrees, from_idx, to_idx).
    """
    i0, i1 = int(max(0, i0)), int(min(len(flex) - 1, i1))
    if i1 <= i0:
        return 0.0, i0, i0
    best, bf, bt = 0.0, i0, i0
    run_max, run_arg = flex[i0], i0
    for j in range(i0 + 1, i1 + 1):
        if flex[j - 1] > run_max:
            run_max, run_arg = flex[j - 1], j - 1
        drop = run_max - flex[j]
        if drop > best:
            best, bf, bt = float(drop), run_arg, j
    return best, bf, bt


@dataclass
class LegalityResult:
    extension_deg: float                 # best estimate (ICC window, max straightening)
    band_low_deg: float                  # sensitivity band incl. pose error
    band_high_deg: float
    peak_to_release_deg: float           # alt. definition: max flexion in window - flexion at release
    flexion_at_uah_deg: float
    flexion_at_release_deg: float
    max_flexion_deg: float
    extension_from_frame: int
    extension_to_frame: int
    verdict: str
    verdict_color: str
    verdict_text: str
    quality: str
    window_frames: int
    window_ms: float
    sensitivity: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def classify(estimate: float, low: float, high: float, limit: float) -> str:
    if high <= limit:
        return "LEGAL"
    if estimate <= limit:
        return "LIKELY LEGAL"
    if low <= limit:
        return "BORDERLINE"
    return "SUSPECT"


def assess(world_filled: np.ndarray, side: str, events: Events, fps: float, cfg,
           arm_visibility: np.ndarray | None = None) -> LegalityResult:
    """Measure elbow extension inside [UAH, release] with a sensitivity analysis.

    The sensitivity grid perturbs the smoothing window and the event frames by
    about +/-10 ms, then the assumed systematic pose error is added on both sides.
    """
    base_w = odd_window(cfg.smooth_window_s, fps)
    flex = elbow_flexion(smooth(world_filled, base_w, cfg.smooth_polyorder), side)
    est, f_from, f_to = max_extension(flex, events.uah, events.release)
    win = flex[events.uah:events.release + 1]
    p2r = float(np.max(win) - flex[events.release]) if len(win) else 0.0

    shift = max(1, int(round(0.010 * fps)))
    samples = []
    for wscale, du, dr in product((0.6, 1.0, 1.4), (-shift, 0, shift), (-shift, 0, shift)):
        w = odd_window(cfg.smooth_window_s * wscale, fps)
        fl = flex if w == base_w else elbow_flexion(smooth(world_filled, w, cfg.smooth_polyorder), side)
        u, r = events.uah + du, events.release + dr
        if r <= u:
            continue
        samples.append(max_extension(fl, u, r)[0])
    samples = np.array(samples) if samples else np.array([est])
    lo = max(0.0, float(np.percentile(samples, 10)) - cfg.pose_error_deg)
    hi = float(np.percentile(samples, 90)) + cfg.pose_error_deg
    lo, hi = min(lo, est), max(hi, est)

    verdict = classify(est, lo, hi, cfg.legal_limit_deg)
    color, text = VERDICT_STYLE[verdict]

    n_win = events.release - events.uah + 1
    warnings, grades = [], []
    grades.append("A" if fps >= 100 else "B" if fps >= 50 else "C")
    if fps < 50:
        warnings.append(f"Video is {fps:.0f} fps. The UAH-to-release window lasts only ~{n_win} frames; "
                        "record at 120-240 fps (slow-motion mode) for a reliable measurement.")
    grades.append("A" if n_win >= 8 else "B" if n_win >= 4 else "C")
    if arm_visibility is not None and len(arm_visibility):
        mv = float(np.mean(arm_visibility[events.uah:events.release + 1]))
        grades.append("A" if mv >= 0.8 else "B" if mv >= 0.6 else "C")
        if mv < 0.6:
            warnings.append(f"Bowling arm is poorly visible in the delivery window (confidence {mv:.2f}). "
                            "Film side-on with the bowling arm facing the camera.")
    if not events.uah_detected:
        warnings.append("Upper arm was never seen rising through horizontal; the window start is approximate.")
    quality = max(grades)  # 'C' > 'B' > 'A' alphabetically -> worst grade

    return LegalityResult(
        extension_deg=round(est, 2), band_low_deg=round(lo, 2), band_high_deg=round(hi, 2),
        peak_to_release_deg=round(p2r, 2),
        flexion_at_uah_deg=round(float(flex[events.uah]), 2),
        flexion_at_release_deg=round(float(flex[events.release]), 2),
        max_flexion_deg=round(float(np.max(win)) if len(win) else 0.0, 2),
        extension_from_frame=int(f_from), extension_to_frame=int(f_to),
        verdict=verdict, verdict_color=color, verdict_text=text, quality=quality,
        window_frames=int(n_win), window_ms=round(1000.0 * (n_win - 1) / fps, 1),
        sensitivity=[round(float(s), 2) for s in samples], warnings=warnings,
    )
