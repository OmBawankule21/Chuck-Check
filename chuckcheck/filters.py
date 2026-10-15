"""Gap filling, smoothing and differentiation of landmark trajectories."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def odd_window(seconds: float, fps: float, minimum: int = 5) -> int:
    w = max(minimum, int(round(seconds * fps)))
    return w if w % 2 == 1 else w + 1


def fill_gaps(x: np.ndarray, valid: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate invalid samples along axis 0.

    x: (T, ...) array; valid: (T,) or broadcastable to x[..., 0] shape (T, J).
    Gaps longer than ``max_gap`` frames (and leading/trailing gaps) are filled
    by nearest value but flagged as invalid in the returned mask.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    T = x.shape[0]
    flat = x.reshape(T, -1)
    vmask = np.asarray(valid, bool)
    if vmask.ndim == 1:
        vcols = np.repeat(vmask[:, None], flat.shape[1], axis=1)
    else:
        per = flat.shape[1] // vmask.shape[1]
        vcols = np.repeat(vmask, per, axis=1)
    vcols &= np.isfinite(flat)
    out_valid = vcols.copy()
    t = np.arange(T)
    for c in range(flat.shape[1]):
        good = vcols[:, c]
        if good.sum() == 0:
            flat[:, c] = 0.0
            continue
        if good.all():
            continue
        flat[:, c] = np.interp(t, t[good], flat[good, c])
        # mark long gaps as still-invalid
        bad = ~good
        start = None
        for i in range(T + 1):
            if i < T and bad[i]:
                if start is None:
                    start = i
            elif start is not None:
                length = i - start
                interior = start > 0 and i < T
                out_valid[start:i, c] = interior and length <= max_gap
                start = None
    filled = flat.reshape(x.shape)
    if vmask.ndim == 1:
        return filled, out_valid.all(axis=1)
    return filled, out_valid.reshape(T, vmask.shape[1], -1).all(axis=2)


def smooth(x: np.ndarray, window: int, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing along axis 0 (keeps peaks better than a moving average)."""
    T = x.shape[0]
    if T < 5:
        return np.array(x, dtype=np.float64)
    w = min(window, T if T % 2 == 1 else T - 1)
    p = min(polyorder, w - 2)
    if w < 3 or p < 1:
        return np.array(x, dtype=np.float64)
    return savgol_filter(x, w, p, axis=0, mode="interp")


def derivative(x: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(x, axis=0) * fps
