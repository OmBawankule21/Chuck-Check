"""Synthetic bowler with a *known* elbow extension.

Used for the demo (no video needed) and for testing that the measurement
recovers the ground-truth extension.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from . import landmarks as L
from .pose import PoseSequence

THIGH, SHANK = 0.45, 0.43
HIP_H = 0.86            # hip height above ground (m)
TORSO = 0.52
UPPER_ARM, FOREARM = 0.30, 0.27
SHOULDER_HW, HIP_HW = 0.19, 0.12
PX_PER_M = 190.0


@dataclass
class SyntheticSpec:
    extension_deg: float = 22.0     # true elbow straightening between UAH and release
    side: str = "right"
    fps: float = 120.0
    duration_s: float = 2.0
    release_s: float = 1.1
    noise_m: float = 0.004
    width: int = 1280
    height: int = 720
    seed: int = 7


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _ik_knee(hip, ankle):
    """2-link IK in the x-y plane (y down). Knee bends forward (+x)."""
    d_vec = ankle - hip
    d = np.clip(np.linalg.norm(d_vec), 1e-6, THIGH + SHANK - 1e-6)
    theta = np.arctan2(d_vec[1], d_vec[0])
    a = np.arccos(np.clip((THIGH ** 2 + d ** 2 - SHANK ** 2) / (2 * THIGH * d), -1, 1))
    cands = [hip + THIGH * np.array([np.cos(theta + s * a), np.sin(theta + s * a)]) for s in (1, -1)]
    return max(cands, key=lambda k: k[0])


class SyntheticBowler:
    def __init__(self, spec: SyntheticSpec | None = None):
        self.spec = spec or SyntheticSpec()
        self._build()

    # -------------------------------------------------------------- kinematics
    def arm_angle(self, t):
        """Bowling-arm rotation phi (deg): 0 = hanging down, 90 = horizontal behind,
        180 = vertical, 270 = horizontal in front."""
        s = self.spec
        t_a, t_r = s.release_s - 0.42, s.release_s
        phi_r = 195.0
        v = 2 * (phi_r - 20.0) / (t_r - t_a)
        if t < t_a:
            return 20.0
        if t <= t_r:
            tau = (t - t_a) / (t_r - t_a)
            return 20.0 + (phi_r - 20.0) * tau ** 2
        dt = min(t - t_r, 0.3)
        return phi_r + v * dt - 0.5 * (v / 0.3) * dt ** 2

    def flexion(self, phi):
        s = self.spec
        f_uah = max(s.extension_deg + 6.0, 24.0)
        if phi < 90:
            return f_uah
        if phi <= 195:
            return f_uah - s.extension_deg * _smoothstep((phi - 90) / 105)
        return f_uah - s.extension_deg

    def _hip_x(self, t):
        s = self.spec
        t_b = s.release_s - 0.45
        if t < t_b:
            return -2.9 + 4.0 * t
        x_b = -2.9 + 4.0 * t_b
        if t < s.release_s:
            dt = t - t_b
            return x_b + 4.0 * dt - 0.5 * (2.0 / 0.45) * dt ** 2
        x_r = x_b + 4.0 * 0.45 - 0.5 * (2.0 / 0.45) * 0.45 ** 2
        return x_r + 2.0 * (t - s.release_s) - 0.5 * 2.0 * (t - s.release_s) ** 2

    def _build(self):
        s = self.spec
        rng = np.random.default_rng(s.seed)
        T = int(round(s.duration_s * s.fps))
        times = np.arange(T) / s.fps
        t_b, t_f = s.release_s - 0.45, s.release_s - 0.13
        bowl, front = s.side, L.other(s.side)
        zs = {"left": -1.0, "right": 1.0}
        x_bfc = self._hip_x(t_b) - 0.15
        x_land = self._hip_x(t_f) + 0.55
        self.crease_x = x_land + 0.10

        P = np.zeros((T, L.N_LANDMARKS, 3))
        for i, t in enumerate(times):
            hx = self._hip_x(t)
            hip_c = np.array([hx, -HIP_H, 0.0])
            sh_c = hip_c + np.array([0.06, -TORSO, 0.0])
            pts = {}
            for side in ("left", "right"):
                z = zs[side]
                pts[f"{side}_hip"] = hip_c + [0, 0, z * HIP_HW]
                pts[f"{side}_shoulder"] = sh_c + [0, 0, z * SHOULDER_HW]
            # legs
            for side in ("left", "right"):
                hip = pts[f"{side}_hip"][:2]
                if t < t_b:
                    ph = 2 * np.pi * 1.6 * t + (0 if side == front else np.pi)
                    ax = hx + 0.38 * np.sin(ph)
                    ay = -0.06 - 0.16 * max(0.0, np.cos(ph))
                elif side == front:
                    if t < t_f:
                        k = _smoothstep((t - t_b) / (t_f - t_b))
                        x0 = hx + 0.38 * np.sin(2 * np.pi * 1.6 * t_b)
                        ax = x0 + (x_land - x0) * k
                        ay = -0.06 - 0.22 * np.sin(np.pi * k)
                    else:
                        ax, ay = x_land, -0.06
                else:
                    t_lift = s.release_s + 0.05
                    if t < t_lift:
                        ax, ay = x_bfc, -0.06
                    else:
                        k = _smoothstep((t - t_lift) / 0.5)
                        ax = x_bfc + (x_land + 0.4 - x_bfc) * k
                        ay = -0.06 - 0.25 * np.sin(np.pi * k)
                ankle = np.array([ax, ay])
                d = ankle - hip
                if np.linalg.norm(d) > THIGH + SHANK:
                    ankle = hip + d / np.linalg.norm(d) * (THIGH + SHANK - 1e-4)
                knee = _ik_knee(hip, ankle)
                z = pts[f"{side}_hip"][2]
                pts[f"{side}_knee"] = np.array([*knee, z])
                pts[f"{side}_ankle"] = np.array([*ankle, z])
                pts[f"{side}_heel"] = np.array([ankle[0] - 0.06, ankle[1] + 0.05, z])
                pts[f"{side}_foot_index"] = np.array([ankle[0] + 0.16, ankle[1] + 0.06, z])
            # bowling arm
            phi = np.radians(self.arm_angle(t))
            fl = np.radians(self.flexion(np.degrees(phi)))
            u = np.array([-np.sin(phi), np.cos(phi), 0.0])
            f = np.array([-np.sin(phi + fl), np.cos(phi + fl), 0.0])
            sh = pts[f"{bowl}_shoulder"]
            pts[f"{bowl}_elbow"] = sh + UPPER_ARM * u
            pts[f"{bowl}_wrist"] = pts[f"{bowl}_elbow"] + FOREARM * f
            # non-bowling arm: raised in front during the gather, then pulled down through release
            t1, t2, t3 = s.release_s - 0.6, s.release_s - 0.3, s.release_s + 0.1
            base = 30 + 15 * np.sin(2 * np.pi * 1.6 * min(t, t1))
            if t < t1:
                psi_deg = base
            elif t < t2:
                psi_deg = base + (150 - base) * _smoothstep((t - t1) / (t2 - t1))
            else:
                psi_deg = 150 - 120 * _smoothstep((t - t2) / (t3 - t2))
            psi = np.radians(psi_deg)
            u2 = np.array([np.sin(psi), np.cos(psi), 0.0])
            f2 = np.array([np.sin(psi + 0.5), np.cos(psi + 0.5), 0.0])
            sh2 = pts[f"{front}_shoulder"]
            pts[f"{front}_elbow"] = sh2 + UPPER_ARM * u2
            pts[f"{front}_wrist"] = pts[f"{front}_elbow"] + FOREARM * f2
            for side in ("left", "right"):
                wv = pts[f"{side}_wrist"] - pts[f"{side}_elbow"]
                wv = wv / (np.linalg.norm(wv) + 1e-9)
                for name, off in (("pinky", 0.08), ("index", 0.09), ("thumb", 0.06)):
                    pts[f"{side}_{name}"] = pts[f"{side}_wrist"] + off * wv + [0, 0, 0.01 * zs[side]]
            head = sh_c + np.array([0.10, -0.20, 0.0])
            pts["nose"] = head + [0.06, 0.0, 0.0]
            for side in ("left", "right"):
                z = zs[side]
                pts[f"{side}_eye_inner"] = head + [0.05, -0.03, 0.015 * z]
                pts[f"{side}_eye"] = head + [0.05, -0.03, 0.03 * z]
                pts[f"{side}_eye_outer"] = head + [0.04, -0.03, 0.045 * z]
                pts[f"{side}_ear"] = head + [-0.03, -0.02, 0.07 * z]
                pts[f"mouth_{side}"] = head + [0.05, 0.04, 0.025 * z]
            for j, name in enumerate(L.NAMES):
                P[i, j] = pts[name]

        P_noisy = P + rng.normal(0, s.noise_m, P.shape)
        hip_c = (P_noisy[:, L.L_HIP] + P_noisy[:, L.R_HIP]) / 2
        self.ground_frame = P
        world = P_noisy - hip_c[:, None, :]
        img = np.empty_like(P_noisy)
        img[..., 0] = s.width / 2 + PX_PER_M * P_noisy[..., 0]
        img[..., 1] = s.height * 0.86 + PX_PER_M * P_noisy[..., 1]
        img[..., 2] = PX_PER_M * P_noisy[..., 2]
        self.times = times
        self.sequence = PoseSequence(
            fps=s.fps, width=s.width, height=s.height, image=img.astype(np.float32),
            world=world.astype(np.float32), visibility=np.full((T, L.N_LANDMARKS), 0.95, np.float32),
            detected=np.ones(T, bool), source=f"synthetic://bowler?extension={s.extension_deg}",
        )
        # ground-truth event frames
        phis = np.array([self.arm_angle(t) for t in times])
        self.true_uah = int(np.argmin(np.abs(np.where(times < s.release_s, phis, 1e9) - 90)))
        self.true_release = int(round(s.release_s * s.fps))

    # ---------------------------------------------------------------- rendering
    def frames(self) -> Iterator[np.ndarray]:
        """Render a simple broadcast-style stick-figure video of the synthetic bowler."""
        s = self.spec
        W, H = s.width, s.height
        bg = np.zeros((H, W, 3), np.uint8)
        g = np.linspace(0, 1, H)[:, None]
        sky = (np.array([60, 40, 25]) * (1 - g) + np.array([90, 70, 45]) * g)
        bg[:] = sky[:, None, :].astype(np.uint8)
        horizon = int(H * 0.55)
        grass = np.linspace(0, 1, H - horizon)[:, None]
        bg[horizon:] = (np.array([40, 110, 50]) * (1 - grass) + np.array([30, 150, 60]) * grass)[:, None, :].astype(np.uint8)
        ground = int(H * 0.86)
        cv2.rectangle(bg, (0, ground - 18), (W, ground + 30), (120, 175, 205), -1)  # pitch strip
        cx = int(W / 2 + PX_PER_M * self.crease_x)
        cv2.line(bg, (cx, ground - 18), (cx, ground + 30), (245, 245, 245), 3)
        cv2.putText(bg, "SYNTHETIC DEMO BOWLER", (W - 330, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (230, 230, 230), 2, cv2.LINE_AA)
        img = self.sequence.image
        for i in range(self.sequence.n_frames):
            fr = bg.copy()
            p = img[i, :, :2].astype(int)
            order = sorted(L.SKELETON_EDGES, key=lambda ed: -img[i, ed[0], 2] - img[i, ed[1], 2])
            for a, b in order:
                cv2.line(fr, tuple(p[a]), tuple(p[b]), (235, 235, 240), 14, cv2.LINE_AA)
            head = p[L.NOSE] + np.array([-8, 0])
            cv2.circle(fr, tuple(head), 24, (235, 235, 240), -1, cv2.LINE_AA)
            yield fr
