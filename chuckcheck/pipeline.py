"""End-to-end analysis: PoseSequence -> deliveries, events, legality and metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import landmarks as L
from .config import AnalysisConfig
from .kinematics import (Events, angular_speed, detect_bowling_arm, detect_events, elbow_flexion,
                         find_deliveries, knee_flexion, tilt_from_vertical, upper_arm_elevation)
from .legality import LegalityResult, assess
from .pose import PoseExtractor, PoseSequence, ProgressFn
from .filters import fill_gaps, odd_window, smooth


class AnalysisError(RuntimeError):
    pass


@dataclass
class DeliveryResult:
    index: int
    events: Events
    legality: LegalityResult
    metrics: dict

    def to_dict(self, fps: float) -> dict:
        return {"index": self.index, "events": self.events.to_dict(fps),
                "legality": self.legality.to_dict(), "metrics": self.metrics}


@dataclass
class AnalysisResult:
    fps: float
    width: int
    height: int
    n_frames: int
    source: str
    side: str
    side_scores: dict
    detection_rate: float
    times: np.ndarray
    flex: np.ndarray
    elev: np.ndarray
    arm_speed: np.ndarray
    image_s: np.ndarray
    world_s: np.ndarray
    world_filled: np.ndarray
    valid: np.ndarray
    visibility: np.ndarray
    deliveries: list[DeliveryResult]
    config: AnalysisConfig
    warnings: list = field(default_factory=list)

    @property
    def primary(self) -> DeliveryResult:
        return self.deliveries[0]

    def to_dict(self, include_series: bool = True) -> dict:
        d = {
            "tool": "ChuckCheck",
            "source": self.source,
            "video": {"fps": self.fps, "width": self.width, "height": self.height,
                      "frames": self.n_frames, "duration_s": round(self.n_frames / self.fps, 3)},
            "bowling_arm": self.side,
            "bowling_arm_scores": {k: round(v, 1) for k, v in self.side_scores.items()},
            "detection_rate": round(self.detection_rate, 3),
            "deliveries": [dl.to_dict(self.fps) for dl in self.deliveries],
            "warnings": self.warnings,
            "config": self.config.to_dict(),
        }
        if include_series:
            d["series"] = {
                "time_s": np.round(self.times, 4).tolist(),
                "elbow_flexion_deg": np.round(self.flex, 2).tolist(),
                "upper_arm_elevation_deg": np.round(self.elev, 2).tolist(),
                "upper_arm_speed_dps": np.round(self.arm_speed, 1).tolist(),
            }
        return d


def _torso_length_px(image: np.ndarray, valid: np.ndarray) -> float:
    sh = (image[:, L.L_SHOULDER, :2] + image[:, L.R_SHOULDER, :2]) / 2
    hp = (image[:, L.L_HIP, :2] + image[:, L.R_HIP, :2]) / 2
    ok = valid[:, [L.L_SHOULDER, L.R_SHOULDER, L.L_HIP, L.R_HIP]].all(axis=1)
    d = np.linalg.norm(sh - hp, axis=1)
    return float(np.median(d[ok])) if ok.any() else float(np.median(d))


def knee_action(ffc: float, rel: float) -> str:
    delta = rel - ffc
    if delta > 10:
        return "Flexor"
    if delta < -10:
        return "Extender"
    return "Braced" if ffc <= 25 else "Constant (flexed)"


def compute_metrics(world_s: np.ndarray, side: str, ev: Events, fps: float,
                    arm_speed: np.ndarray) -> dict:
    r = ev.release
    front = L.other(side)
    hi = min(len(arm_speed), r + int(0.05 * fps) + 1)
    peak_speed = float(np.max(arm_speed[ev.uah:hi]))

    feet = [L.L_HEEL, L.R_HEEL, L.L_FOOT, L.R_FOOT, L.L_ANKLE, L.R_ANKLE]
    ground_y = float(np.max(world_s[r, feet, 1]))
    _, _, w = L.arm(side)
    release_h = ground_y - float(world_s[r, w, 1])

    mid_sh = (world_s[:, L.L_SHOULDER] + world_s[:, L.R_SHOULDER]) / 2
    mid_hp = (world_s[:, L.L_HIP] + world_s[:, L.R_HIP]) / 2
    trunk = tilt_from_vertical(mid_sh - mid_hp)

    m = {
        "bowling_arm": f"{side.capitalize()}-arm",
        "peak_upper_arm_speed_dps": round(peak_speed, 0),
        "release_height_m": round(release_h, 2),
        "release_height_note": "approximate: MediaPipe world scale",
        "trunk_tilt_at_release_deg": round(float(trunk[r]), 1),
        "uah_to_release_ms": round(1000 * (r - ev.uah) / fps, 1),
        "front_foot_contact_frame": ev.ffc,
        "ffc_to_release_ms": None,
        "front_knee_flexion_at_ffc_deg": None,
        "front_knee_flexion_at_release_deg": None,
        "front_knee_action": None,
    }
    if ev.ffc is not None and ev.ffc < r:
        knee = knee_flexion(world_s, front)
        kf, kr = float(knee[ev.ffc]), float(knee[r])
        m.update({
            "ffc_to_release_ms": round(1000 * (r - ev.ffc) / fps, 1),
            "front_knee_flexion_at_ffc_deg": round(kf, 1),
            "front_knee_flexion_at_release_deg": round(kr, 1),
            "front_knee_action": knee_action(kf, kr),
        })
    return m


def analyze_sequence(seq: PoseSequence, cfg: Optional[AnalysisConfig] = None,
                     overrides: Optional[dict] = None) -> AnalysisResult:
    """Analyse an already-extracted pose sequence.

    overrides: {delivery_index: {"uah": frame, "release": frame}} to correct events by hand.
    """
    cfg = cfg or AnalysisConfig()
    overrides = overrides or {}
    fps = seq.fps
    det_rate = float(seq.detected.mean()) if seq.n_frames else 0.0
    if seq.n_frames < int(0.5 * fps) or det_rate < 0.2:
        raise AnalysisError(
            f"Bowler detected in only {det_rate:.0%} of {seq.n_frames} frames. "
            "Use a clip where the full body is visible during the delivery stride.")

    valid = seq.detected[:, None] & (seq.visibility >= cfg.min_visibility)
    max_gap = max(1, int(round(cfg.max_gap_s * fps)))
    world_f, valid_w = fill_gaps(seq.world, valid, max_gap)
    image_f, _ = fill_gaps(seq.image, valid, max_gap)
    win = odd_window(cfg.smooth_window_s, fps)
    world_s = smooth(world_f, win, cfg.smooth_polyorder)
    image_s = smooth(image_f, odd_window(cfg.smooth_window_s * 0.6, fps), 2)

    warnings: list[str] = []
    if cfg.bowling_arm == "auto":
        side, scores = detect_bowling_arm(world_s, fps)
    else:
        side, scores = cfg.bowling_arm, {}
    s, e, w = L.arm(side)
    flex = elbow_flexion(world_s, side)
    elev = upper_arm_elevation(world_s, side)
    arm_speed = smooth(angular_speed(world_s[:, e] - world_s[:, s], fps), odd_window(0.05, fps, 3), 2)
    torso_px = _torso_length_px(image_f, valid_w)
    front_ankle = image_s[:, L.leg(L.other(side))[2]]

    arm_valid = valid_w[:, [s, e, w]].all(axis=1)
    peaks = find_deliveries(arm_speed, elev, fps, cfg.min_delivery_separation_s, arm_valid)
    if not peaks:
        raise AnalysisError("No bowling action found (the arm never rotates over the shoulder).")

    arm_vis = seq.visibility[:, [s, e, w]].min(axis=1) * seq.detected
    deliveries = []
    for i, t_fast in enumerate(peaks):
        ev = detect_events(elev, world_s[:, w], front_ankle, torso_px, fps, t_fast,
                           cfg.release_method, cfg.release_offset_deg)
        ov = overrides.get(i) or overrides.get(str(i))
        if ov:
            if ov.get("uah") is not None:
                ev.uah = int(ov["uah"])
            if ov.get("release") is not None:
                ev.release = int(ov["release"])
            ev.release = max(ev.release, ev.uah + 1)
            ev.manual = True
        leg = assess(world_f, side, ev, fps, cfg, arm_vis)
        met = compute_metrics(world_s, side, ev, fps, arm_speed)
        deliveries.append(DeliveryResult(i, ev, leg, met))

    if det_rate < 0.7:
        warnings.append(f"Bowler detected in {det_rate:.0%} of frames; gaps were interpolated.")
    if len(deliveries) > 1:
        warnings.append(f"{len(deliveries)} deliveries detected in this clip.")

    return AnalysisResult(
        fps=fps, width=seq.width, height=seq.height, n_frames=seq.n_frames, source=seq.source,
        side=side, side_scores=scores, detection_rate=det_rate, times=seq.times, flex=flex,
        elev=elev, arm_speed=arm_speed, image_s=image_s, world_s=world_s, world_filled=world_f,
        valid=valid_w, visibility=seq.visibility, deliveries=deliveries, config=cfg,
        warnings=warnings,
    )


def analyze_video(path: str | Path, cfg: Optional[AnalysisConfig] = None,
                  progress: ProgressFn = None) -> tuple[AnalysisResult, PoseSequence]:
    cfg = cfg or AnalysisConfig()
    seq = PoseExtractor(cfg).extract(path, progress)
    return analyze_sequence(seq, cfg), seq
