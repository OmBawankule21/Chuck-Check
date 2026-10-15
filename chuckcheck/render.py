"""Annotated, broadcast-style output video."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import landmarks as L
from .pipeline import AnalysisResult

AMBER = (11, 158, 245)        # BGR
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
PANEL = (24, 20, 18)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False):
    try:
        import matplotlib
        base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        return ImageFont.truetype(os.path.join(base, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
    except Exception:  # pragma: no cover
        return ImageFont.load_default()


def _panel(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.62) -> None:
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + np.array(PANEL) * alpha).astype(np.uint8)


class _Text:
    """Batch text drawing with antialiased TrueType fonts via PIL."""

    def __init__(self):
        self.items = []

    def add(self, xy, text, size, color=(240, 240, 240), bold=False, anchor="la"):
        self.items.append((xy, text, size, color, bold, anchor))

    def draw(self, img_bgr: np.ndarray) -> np.ndarray:
        if not self.items:
            return img_bgr
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for xy, text, size, color, bold, anchor in self.items:
            d.text(xy, text, font=_font(int(size), bold), fill=color, anchor=anchor)
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _phase(i: int, res: AnalysisResult) -> tuple[str, Optional[int]]:
    for dl in res.deliveries:
        ev = dl.events
        if ev.uah <= i <= ev.release:
            return "ICC WINDOW: arm horizontal → release", dl.index
        if ev.ffc is not None and ev.ffc <= i < ev.uah:
            return "Front-foot contact", dl.index
        if ev.release < i <= ev.release + int(0.6 * res.fps):
            return "Follow-through", dl.index
    return "Run-up / gather", None


def _draw_elbow_arc(img, sh, el, wr, flex, s):
    a1 = np.degrees(np.arctan2(sh[1] - el[1], sh[0] - el[0]))
    a2 = np.degrees(np.arctan2(wr[1] - el[1], wr[0] - el[0]))
    diff = (a2 - a1 + 540) % 360 - 180
    r = int(28 * s)
    cv2.ellipse(img, tuple(map(int, el)), (r, r), 0, a1, a1 + diff, AMBER, max(2, int(3 * s)), cv2.LINE_AA)


def render_video(result: AnalysisResult, frames: Iterable[np.ndarray], out_path: str | Path,
                 progress: Optional[Callable[[float, str], None]] = None) -> str:
    """Overlay skeleton, live elbow angle, event markers, timeline and verdict card."""
    import imageio_ffmpeg

    cfg = result.config
    fps = result.fps
    W, H = result.width - result.width % 2, result.height - result.height % 2
    s = min(W, H) / 720.0
    out_fps = float(min(fps, 60.0))
    side = result.side
    si, ei, wi = L.arm(side)
    hl = L.arm_edges(side)
    T = result.n_frames

    # slow-motion zones around each release
    slow = np.zeros(T, bool)
    if cfg.slowmo < 1.0:
        half = int(cfg.slowmo_window_s * fps)
        for dl in result.deliveries:
            r = dl.events.release
            slow[max(0, r - half): min(T, r + half + 1)] = True

    # timeline geometry (bottom strip)
    tl_x0, tl_x1 = int(24 * s), W - int(24 * s)
    tl_y0, tl_y1 = H - int(120 * s), H - int(22 * s)
    fmax = max(40.0, float(np.nanmax(result.flex)) + 5)
    xs = np.linspace(tl_x0, tl_x1, T)
    ys = tl_y1 - (np.clip(result.flex, 0, fmax) / fmax) * (tl_y1 - tl_y0 - int(18 * s))
    poly = np.stack([xs, ys], 1).astype(np.int32)

    writer = imageio_ffmpeg.write_frames(
        str(out_path), (W, H), fps=out_fps, codec="libx264", pix_fmt_in="rgb24",
        output_params=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast"],
        macro_block_size=1, ffmpeg_log_level="error")
    writer.send(None)
    acc = 0.0
    last = None
    try:
        for i, frame in enumerate(frames):
            if i >= T:
                break
            img = frame[:H, :W].copy()
            p = result.image_s[i, :, :2]
            ok = result.valid[i]
            # skeleton
            for a, b in L.SKELETON_EDGES:
                if not (ok[a] and ok[b]):
                    continue
                is_arm = (a, b) in hl or (b, a) in hl
                cv2.line(img, tuple(p[a].astype(int)), tuple(p[b].astype(int)),
                         AMBER if is_arm else WHITE, max(2, int((6 if is_arm else 2) * s)), cv2.LINE_AA)
            for j in range(11, 33):
                if ok[j]:
                    cv2.circle(img, tuple(p[j].astype(int)), max(2, int(3 * s)), WHITE, -1, cv2.LINE_AA)
            txt = _Text()
            if ok[si] and ok[ei] and ok[wi]:
                _draw_elbow_arc(img, p[si], p[ei], p[wi], result.flex[i], s)
                ex, ey = p[ei].astype(int)
                _panel(img, ex + int(30 * s), ey - int(20 * s), ex + int(118 * s), ey + int(16 * s), 0.7)
                txt.add((ex + int(36 * s), ey - int(16 * s)), f"{result.flex[i]:5.1f}°", 24 * s, (245, 158, 11), True)

            # HUD
            phase, di = _phase(i, result)
            _panel(img, int(16 * s), int(16 * s), int(16 * s) + int(420 * s), int(150 * s))
            txt.add((int(30 * s), int(26 * s)), "CHUCKCHECK", 18 * s, (245, 158, 11), True)
            txt.add((int(196 * s), int(29 * s)), f"{side.upper()}-ARM  •  {fps:.0f} fps", 14 * s, (180, 180, 180))
            txt.add((int(30 * s), int(56 * s)), "Elbow flexion", 15 * s, (180, 180, 180))
            txt.add((int(30 * s), int(76 * s)), f"{result.flex[i]:.1f}°", 34 * s, (255, 255, 255), True)
            txt.add((int(200 * s), int(56 * s)), "Phase", 15 * s, (180, 180, 180))
            txt.add((int(200 * s), int(80 * s)), phase, 15 * s, (255, 255, 255), True)
            txt.add((int(30 * s), int(124 * s)), f"t = {i / fps:6.3f} s   frame {i}" + ("   ▶ SLOW-MO" if slow[i] else ""),
                    13 * s, (170, 170, 170))

            # event flash labels
            for dl in result.deliveries:
                ev = dl.events
                for f_ev, label in ((ev.uah, "UPPER ARM HORIZONTAL"), (ev.release, "BALL RELEASE"),
                                    (ev.ffc, "FRONT-FOOT CONTACT")):
                    if f_ev is not None and 0 <= i - f_ev < max(2, int(0.12 * fps)):
                        _panel(img, W // 2 - int(190 * s), int(18 * s), W // 2 + int(190 * s), int(64 * s), 0.75)
                        txt.add((W // 2, int(41 * s)), label, 22 * s, (245, 158, 11), True, "mm")

            # timeline
            _panel(img, tl_x0 - int(10 * s), tl_y0 - int(8 * s), tl_x1 + int(10 * s), tl_y1 + int(10 * s), 0.6)
            for dl in result.deliveries:
                ev = dl.events
                cv2.rectangle(img, (int(xs[ev.uah]), tl_y0), (int(xs[ev.release]) + 1, tl_y1), (40, 90, 140), -1)
            lim_y = int(tl_y1 - (cfg.legal_limit_deg / fmax) * (tl_y1 - tl_y0 - int(18 * s)))
            cv2.line(img, (tl_x0, lim_y), (tl_x1, lim_y), (90, 90, 90), 1, cv2.LINE_AA)
            cv2.polylines(img, [poly], False, (200, 200, 200), max(1, int(2 * s)), cv2.LINE_AA)
            cv2.polylines(img, [poly[: i + 1]], False, AMBER, max(2, int(3 * s)), cv2.LINE_AA)
            cv2.line(img, (int(xs[i]), tl_y0), (int(xs[i]), tl_y1), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(img, tuple(poly[i]), max(3, int(5 * s)), AMBER, -1, cv2.LINE_AA)
            txt.add((tl_x0, tl_y0 - int(4 * s)), "Elbow flexion over time  (shaded = ICC window)", 12 * s, (190, 190, 190), False, "lb")
            img = txt.draw(img)

            rate = out_fps / fps / (cfg.slowmo if slow[i] else 1.0)
            acc += rate
            while acc >= 1.0:
                writer.send(np.ascontiguousarray(img[:, :, ::-1]))
                acc -= 1.0
            last = img
            if progress and i % 10 == 0:
                progress(min(0.99, (i + 1) / T), f"Rendering frame {i + 1}/{T}")

        if last is not None:
            card = _verdict_card(last, result, s)
            for _ in range(int(cfg.verdict_hold_s * out_fps)):
                writer.send(np.ascontiguousarray(card[:, :, ::-1]))
    finally:
        writer.close()
    if progress:
        progress(1.0, "Video rendered")
    return str(out_path)


def _verdict_card(img: np.ndarray, result: AnalysisResult, s: float) -> np.ndarray:
    H, W = img.shape[:2]
    card = (img * 0.35).astype(np.uint8)
    dl = result.primary
    lg = dl.legality
    col = _hex_to_rgb(lg.verdict_color)
    cw, ch = int(560 * s), int(300 * s)
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    _panel(card, x0, y0, x0 + cw, y0 + ch, 0.9)
    cv2.rectangle(card, (x0, y0), (x0 + int(8 * s), y0 + ch), col[::-1], -1)
    t = _Text()
    t.add((x0 + int(32 * s), y0 + int(24 * s)), "CHUCKCHECK VERDICT", 16 * s, (180, 180, 180), True)
    t.add((x0 + int(32 * s), y0 + int(52 * s)), lg.verdict, 44 * s, col, True)
    t.add((x0 + int(32 * s), y0 + int(118 * s)), f"Elbow extension  {lg.extension_deg:.1f}°", 26 * s, (255, 255, 255), True)
    t.add((x0 + int(32 * s), y0 + int(156 * s)),
          f"Error band {lg.band_low_deg:.1f}° – {lg.band_high_deg:.1f}°   •   ICC limit {result.config.legal_limit_deg:.0f}°",
          16 * s, (200, 200, 200))
    t.add((x0 + int(32 * s), y0 + int(186 * s)),
          f"Window {lg.window_ms:.0f} ms ({lg.window_frames} frames)   •   Data quality {lg.quality}", 16 * s, (200, 200, 200))
    t.add((x0 + int(32 * s), y0 + int(240 * s)),
          "Screening estimate from monocular video – not an official ICC test.", 13 * s, (150, 150, 150))
    return t.draw(card)
