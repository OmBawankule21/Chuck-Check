"""Download and cache the MediaPipe pose landmarker models."""
from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def model_dir() -> Path:
    d = Path(os.environ.get("CHUCKCHECK_MODEL_DIR", Path.home() / ".cache" / "chuckcheck" / "models"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_model_path(variant: str = "full", model_path: str | None = None) -> str:
    """Return a local path to the pose model, downloading it on first use."""
    if model_path:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return str(model_path)
    if variant not in MODEL_URLS:
        raise ValueError(f"Unknown model variant {variant!r}; choose from {list(MODEL_URLS)}")
    dest = model_dir() / f"pose_landmarker_{variant}.task"
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return str(dest)
    url = MODEL_URLS[variant]
    print(f"[chuckcheck] downloading {variant} pose model -> {dest}")
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dest)
    except Exception as exc:  # pragma: no cover - network dependent
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the pose model from {url}.\n"
            f"Download it manually and pass --model-path, or place it at {dest}.\n"
            f"Original error: {exc}"
        ) from exc
    return str(dest)
