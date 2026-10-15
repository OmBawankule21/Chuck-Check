"""Joint angles, bowling-arm detection and delivery event detection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from scipy.signal import find_peaks, medfilt

from . import landmarks as L
from .filters import odd_window, smooth

UP = np.array([0.0, -1.0, 0.0])  # MediaPipe world y-axis points down


# ----------------------------------------------------------------------------- angles
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-9, None)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle ABC in degrees (180 = straight)."""
    u, v = _unit(a - b), _unit(c - b)
    return np.degrees(np.arccos(np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)))


def elbow_flexion(world: np.ndarray, side: str) -> np.ndarray:
    """Elbow flexion in degrees: 0 = fully straight arm."""
    s, e, w = L.arm(side)
    return 180.0 - joint_angle(world[:, s], world[:, e], world[:, w])


def knee_flexion(world: np.ndarray, side: str) -> np.ndarray:
    h, k, a, _, _ = L.leg(side)
    return 180.0 - joint_angle(world[:, h], world[:, k], world[:, a])


def upper_arm_elevation(world: np.ndarray, side: str) -> np.ndarray:
    """Angle of the upper arm above the horizontal plane, degrees.
    -90 = hanging down, 0 = horizontal, +90 = pointing straight up."""
    s, e, _ = L.arm(side)
    u = _unit(world[:, e] - world[:, s])
    return np.degrees(np.arcsin(np.clip(u @ UP, -1.0, 1.0)))


def angular_speed(vec: np.ndarray, fps: float) -> np.ndarray:
    """Angular speed (deg/s) of a direction vector over time (central difference)."""
    u = _unit(vec)
    T = len(u)
    out = np.zeros(T)
    if T < 3:
        return out
    d = np.clip(np.sum(u[2:] * u[:-2], axis=1), -1.0, 1.0)
    out[1:-1] = np.degrees(np.arccos(d)) * fps / 2.0
    out[0], out[-1] = out[1], out[-2]
    return out


def tilt_from_vertical(vec: np.ndarray) -> np.ndarray:
    u = _unit(vec)
    return np.degrees(np.arccos(np.clip(u @ UP, -1.0, 1.0)))


# ------------------------------------------------------------------- bowling arm
def detect_bowling_arm(world: np.ndarray, fps: float) -> tuple[str, dict]:
    """The bowling arm whirls over the top: it has the fastest upper-arm rotation
    and reaches a high elevation."""
    scores = {}
    k = odd_window(0.04, fps, minimum=3)
    box = np.ones(max(1, int(0.4 * fps)))
    for side in ("left", "right"):
        s, e, _ = L.arm(side)
        speed = medfilt(angular_speed(world[:, e] - world[:, s], fps), k)  # kills 1-2 frame glitches
        travel = np.convolve(speed / fps, box, mode="same")  # degrees rotated in any 0.4 s window
        elev = upper_arm_elevation(world, side)
        score = float(np.max(travel))
        if np.nanmax(elev) < 45:
            score *= 0.5
        scores[side] = score
    side = max(scores, key=scores.get)
    return side, scores


def find_deliveries(arm_speed: np.ndarray, elev: np.ndarray, fps: float,
                    min_sep_s: float = 1.5, arm_valid: Optional[np.ndarray] = None) -> list[int]:
    """Frames of peak upper-arm angular speed, one per delivery.

    A peak only counts as a delivery if the arm rises from below horizontal to
    well above it (over-arm action) and rotates through a large angle.
    """
    if len(arm_speed) < 3 or np.nanmax(arm_speed) <= 0:
        return []
    peak = float(np.nanmax(arm_speed))
    if arm_valid is not None:
        arm_speed = np.where(arm_valid, arm_speed, 0.0)
        if np.max(arm_speed) <= 0:
            return []
        peak = float(np.max(arm_speed))
    idx, _ = find_peaks(arm_speed, height=0.25 * peak, distance=max(1, int(0.3 * fps)))
    good = []
    for i in idx:
        if arm_valid is not None:
            win = arm_valid[max(0, i - int(0.4 * fps)):i + int(0.25 * fps) + 1]
            if win.mean() < 0.8:   # mostly interpolated: not trustworthy
                continue
        up = elev[min(len(elev) - 1, i):min(len(elev), i + int(0.25 * fps) + 1)]
        pre = elev[max(0, i - int(0.4 * fps)):i + 1]
        before = elev[max(0, i - int(0.7 * fps)):i + 1]
        a, b = max(0, i - int(0.3 * fps)), min(len(arm_speed), i + int(0.3 * fps) + 1)
        travel = float(np.sum(arm_speed[a:b]) / fps)
        high = max(np.nanmax(pre), np.nanmax(up) if len(up) else -90)
        if high > 50 and np.nanmin(before) < 0 and travel >= 150:
            good.append(int(i))
    # keep the strongest qualifying peak in each min_sep_s neighbourhood
    keep: list[int] = []
    for i in sorted(good, key=lambda k: -arm_speed[k]):
        if all(abs(i - k) >= min_sep_s * fps for k in keep):
            keep.append(i)
    if keep:
        strongest = max(arm_speed[k] for k in keep)
        keep = [k for k in keep if arm_speed[k] >= 0.4 * strongest]
    return sorted(keep)


# ------------------------------------------------------------------------ events
@dataclass
class Events:
    uah: int                 # upper arm horizontal (start of ICC window)
    vertical: int            # upper arm at max elevation
    release: int             # estimated ball release (end of ICC window)
    ffc: Optional[int]       # front-foot contact
    uah_detected: bool = True
    release_method: str = "arm-angle"
    manual: bool = False

    def to_dict(self, fps: float) -> dict:
        d = asdict(self)
        for k in ("uah", "vertical", "release", "ffc"):
            d[f"{k}_time_s"] = None if d[k] is None else round(d[k] / fps, 4)
        return d


def detect_events(elev: np.ndarray, wrist_world: np.ndarray, front_ankle_img: np.ndarray,
                  torso_px: float, fps: float, t_fast: int, release_method: str = "arm-angle",
                  release_offset_deg: float = 15.0) -> Events:
    T = len(elev)
    lo = max(0, t_fast - int(0.35 * fps))
    hi = min(T, t_fast + int(0.25 * fps) + 1)
    vertical = lo + int(np.nanargmax(elev[lo:hi]))

    # Upper-arm-horizontal: last upward zero-crossing of elevation before vertical.
    uah, found = None, False
    start = max(1, vertical - int(0.6 * fps))
    for k in range(vertical, start - 1, -1):
        if elev[k - 1] < 0.0 <= elev[k]:
            uah, found = (k if abs(elev[k]) <= abs(elev[k - 1]) else k - 1), True
            break
    if uah is None:  # arm never seen below horizontal: take the lowest point in the window
        uah = start + int(np.nanargmin(elev[start:vertical + 1]))

    # Ball release
    max_after = min(T - 1, vertical + int(0.2 * fps))
    release = None
    if release_method == "wrist-speed":
        sp = np.linalg.norm(np.gradient(wrist_world, axis=0), axis=1) * fps
        end = min(T, vertical + int(0.15 * fps) + 1)
        release = vertical + int(np.nanargmax(sp[vertical:end]))
    else:
        target = elev[vertical] - release_offset_deg
        for k in range(vertical + 1, max_after + 1):
            if elev[k] <= target:
                release = k if abs(elev[k] - target) <= abs(elev[k - 1] - target) else k - 1
                break
    if release is None or release <= vertical - 1:
        release = min(T - 1, vertical + max(1, int(round(0.03 * fps))))
    release = max(release, uah + 1)

    ffc = detect_front_foot_contact(front_ankle_img, torso_px, fps, vertical)
    return Events(uah=int(uah), vertical=int(vertical), release=int(release), ffc=ffc,
                  uah_detected=found, release_method=release_method)


def detect_front_foot_contact(ankle_img: np.ndarray, torso_px: float, fps: float,
                              vertical: int) -> Optional[int]:
    """Start of the stationary phase of the front ankle that contains arm-vertical."""
    if torso_px <= 0 or not np.isfinite(torso_px):
        return None
    sp = np.linalg.norm(np.gradient(ankle_img[:, :2], axis=0), axis=1) * fps / torso_px
    sp = smooth(sp, odd_window(0.05, fps, 3), 2)
    thr = 1.2  # torso lengths per second
    lo = max(0, vertical - int(0.6 * fps))
    if sp[vertical] >= thr:
        # foot still moving at vertical: look for the slowest frame just before it
        k = lo + int(np.argmin(sp[lo:vertical + 1]))
        return k if sp[k] < thr else None
    k = vertical
    while k - 1 >= lo and sp[k - 1] < thr:
        k -= 1
    return k if k > lo else None
