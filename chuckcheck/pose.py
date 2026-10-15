"""Pose extraction: video -> PoseSequence (2D image + 3D world landmarks per frame)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from .config import AnalysisConfig
from .landmarks import L_HIP, N_LANDMARKS, R_HIP
from .models import get_model_path

ProgressFn = Optional[Callable[[float, str], None]]


@dataclass
class PoseSequence:
    """Per-frame landmarks of the tracked bowler.

    image:      (T, 33, 3)  x, y in pixels; z in pixel-scaled relative depth
    world:      (T, 33, 3)  metres, origin between the hips, y axis points DOWN
    visibility: (T, 33)     0..1 landmark confidence
    detected:   (T,)        whether a person was found in that frame
    """

    fps: float
    width: int
    height: int
    image: np.ndarray
    world: np.ndarray
    visibility: np.ndarray
    detected: np.ndarray
    source: str = ""

    @property
    def n_frames(self) -> int:
        return int(self.image.shape[0])

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) / self.fps

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, fps=self.fps, width=self.width, height=self.height, image=self.image,
            world=self.world, visibility=self.visibility, detected=self.detected,
            source=np.array(self.source),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PoseSequence":
        d = np.load(path, allow_pickle=False)
        return cls(
            fps=float(d["fps"]), width=int(d["width"]), height=int(d["height"]),
            image=d["image"], world=d["world"], visibility=d["visibility"],
            detected=d["detected"], source=str(d["source"]),
        )


def video_info(path: str | Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Video has no readable frames: {path}")
    h, w = frame.shape[:2]
    return float(fps), w, h, n


def iter_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield BGR frames (OpenCV applies the phone rotation flag automatically)."""
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _hip_center(lms) -> np.ndarray:
    return np.array([(lms[L_HIP].x + lms[R_HIP].x) / 2, (lms[L_HIP].y + lms[R_HIP].y) / 2])


def _bbox_area(lms) -> float:
    xs = [p.x for p in lms]
    ys = [p.y for p in lms]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _vis(p) -> float:
    v = getattr(p, "visibility", None)
    pr = getattr(p, "presence", None)
    v = 1.0 if v is None else float(v)
    pr = 1.0 if pr is None else float(pr)
    return min(v, pr)


class PoseExtractor:
    """Runs MediaPipe PoseLandmarker over a video and tracks a single person."""

    def __init__(self, config: AnalysisConfig | None = None):
        self.cfg = config or AnalysisConfig()

    def _create(self):
        import mediapipe as mp  # imported lazily so the analytics work without it
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = get_model_path(self.cfg.model, self.cfg.model_path)
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self.cfg.num_poses,
            min_pose_detection_confidence=self.cfg.min_detection_confidence,
            min_pose_presence_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        return mp, vision.PoseLandmarker.create_from_options(opts)

    def extract(self, video_path: str | Path, progress: ProgressFn = None) -> PoseSequence:
        fps, width, height, n_est = video_info(video_path)
        mp, landmarker = self._create()
        images, worlds, vis, det = [], [], [], []
        prev_center, last_seen = None, -10**9
        max_jump_frames = int(0.5 * fps)
        scale = min(1.0, self.cfg.max_infer_side / max(width, height))
        try:
            for i, frame in enumerate(iter_frames(video_path)):
                small = frame if scale >= 1.0 else cv2.resize(
                    frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                ts_ms = int(round(i * 1000.0 / fps))
                res = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ts_ms)

                img = np.full((N_LANDMARKS, 3), np.nan, np.float32)
                wld = np.full((N_LANDMARKS, 3), np.nan, np.float32)
                v = np.zeros(N_LANDMARKS, np.float32)
                found = False
                k = self._choose(res.pose_landmarks, prev_center) if res.pose_landmarks else None
                if k is not None and prev_center is not None and i - last_seen <= max_jump_frames:
                    # a sudden jump means a different person (umpire, batter): ignore that frame
                    if np.linalg.norm(_hip_center(res.pose_landmarks[k]) - prev_center) > 0.25:
                        k = None
                if k is not None:
                    lms, wlms = res.pose_landmarks[k], res.pose_world_landmarks[k]
                    last_seen = i
                    img[:] = [(p.x * width, p.y * height, p.z * width) for p in lms]
                    wld[:] = [(p.x, p.y, p.z) for p in wlms]
                    v[:] = [_vis(p) for p in lms]
                    prev_center = _hip_center(lms)
                    found = True
                images.append(img); worlds.append(wld); vis.append(v); det.append(found)
                if progress and (i % 5 == 0):
                    frac = (i + 1) / n_est if n_est > 0 else 0.0
                    progress(min(frac, 0.99), f"Pose estimation: frame {i + 1}/{max(n_est, i + 1)}")
        finally:
            landmarker.close()
        if not images:
            raise IOError(f"No frames decoded from {video_path}")
        if progress:
            progress(1.0, "Pose estimation complete")
        return PoseSequence(
            fps=fps, width=width, height=height, image=np.stack(images), world=np.stack(worlds),
            visibility=np.stack(vis), detected=np.array(det, bool), source=str(video_path),
        )

    @staticmethod
    def _choose(candidates, prev_center) -> int:
        """Keep tracking the same person: nearest hip-centre to the previous frame,
        otherwise the largest person in view."""
        if len(candidates) == 1:
            return 0
        if prev_center is not None:
            d = [np.linalg.norm(_hip_center(c) - prev_center) for c in candidates]
            return int(np.argmin(d))
        return int(np.argmax([_bbox_area(c) for c in candidates]))
