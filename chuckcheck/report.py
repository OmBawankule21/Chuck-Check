"""Write analysis outputs: JSON, CSV, key frames and a self-contained HTML report card."""
from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import pandas as pd

from . import __version__
from . import landmarks as L
from .pipeline import AnalysisResult
from .plots import elevation_figure, flexion_figure, skeleton_3d_figure, speed_figure

FrameFactory = Callable[[], Iterable[np.ndarray]]


def timeseries_frame(res: AnalysisResult) -> pd.DataFrame:
    df = pd.DataFrame({
        "frame": np.arange(res.n_frames),
        "time_s": np.round(res.times, 5),
        "elbow_flexion_deg": np.round(res.flex, 3),
        "upper_arm_elevation_deg": np.round(res.elev, 3),
        "upper_arm_speed_dps": np.round(res.arm_speed, 2),
        "pose_detected": res.valid[:, L.arm(res.side)[1]],
    })
    df["phase"] = ""
    for dl in res.deliveries:
        ev = dl.events
        df.loc[ev.uah:ev.release, "phase"] = f"icc_window_{dl.index}"
    return df


def grab_frames(factory: FrameFactory, indices: Iterable[int]) -> dict[int, np.ndarray]:
    want = sorted(set(int(i) for i in indices if i is not None))
    out = {}
    if not want:
        return out
    for i, fr in enumerate(factory()):
        if i in want:
            out[i] = fr.copy()
        if i >= want[-1]:
            break
    return out


def annotate_keyframe(img: np.ndarray, res: AnalysisResult, i: int, label: str) -> np.ndarray:
    img = img.copy()
    p = res.image_s[i, :, :2]
    s = min(img.shape[:2]) / 720.0
    arm = L.arm_edges(res.side)
    for a, b in L.SKELETON_EDGES:
        col = (11, 158, 245) if (a, b) in arm else (240, 240, 240)
        cv2.line(img, tuple(p[a].astype(int)), tuple(p[b].astype(int)), col, max(2, int((6 if (a, b) in arm else 2) * s)), cv2.LINE_AA)
    # crop around the bowler
    pts = p[11:][res.valid[i, 11:]]
    if len(pts):
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) * 0.75 + 40
        X0, Y0 = int(max(0, cx - half)), int(max(0, cy - half))
        X1, Y1 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
        if X1 - X0 > 50 and Y1 - Y0 > 50:
            img = img[Y0:Y1, X0:X1]
    h = 420
    img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    return img


def _b64jpg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _fmt(v, unit="", nd=1):
    if v is None:
        return "–"
    if isinstance(v, (int, float, np.floating)):
        return f"{v:.{nd}f}{unit}"
    return html.escape(str(v))


CSS = """
:root{--page:#0d0d0d;--surface:#1a1a19;--surface2:#232321;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--line:rgba(255,255,255,.10);--accent:#f5a60b;}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}
.wrap{max-width:1120px;margin:0 auto;padding:32px 16px 64px}
.brand{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.brand h1{margin:0;font-size:20px;letter-spacing:.14em;color:var(--accent)}
.brand .meta{color:var(--muted);font-size:13px}
.hero{margin-top:20px;background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:28px;
display:grid;grid-template-columns:1.2fr 1fr;gap:24px;position:relative;overflow:hidden}
.hero:before{content:"";position:absolute;left:0;top:0;bottom:0;width:6px;background:var(--vcol)}
.badge{display:inline-flex;align-items:center;gap:10px;font-weight:800;font-size:34px;color:var(--vcol);letter-spacing:.02em}
.badge .icon{font-size:26px}
.vtext{color:var(--ink2);margin:6px 0 0}
.big{font-size:64px;font-weight:800;line-height:1}
.big small{font-size:22px;color:var(--ink2);font-weight:600;margin-left:6px}
.band{color:var(--ink2);margin-top:8px}
.bar{position:relative;height:10px;border-radius:5px;background:var(--surface2);margin:18px 0 6px}
.bar .rng{position:absolute;top:0;bottom:0;border-radius:5px;background:rgba(57,135,229,.45)}
.bar .est{position:absolute;top:-5px;width:4px;height:20px;border-radius:2px;background:var(--ink)}
.bar .lim{position:absolute;top:-8px;bottom:-8px;width:2px;background:var(--ink2)}
.scale{display:flex;justify-content:space-between;color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.tile .v{font-size:22px;font-weight:700;margin-top:4px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:16px;margin-top:16px}
h2{font-size:15px;color:var(--ink2);letter-spacing:.08em;text-transform:uppercase;margin:32px 0 4px}
.frames{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.frames figure{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.frames img{width:100%;display:block}
.frames figcaption{padding:8px 12px;color:var(--ink2);font-size:13px}
.warn{background:rgba(250,178,25,.08);border:1px solid rgba(250,178,25,.35);border-radius:12px;padding:12px 16px;margin-top:16px;color:var(--ink)}
.warn li{margin:4px 0}
table{width:100%;border-collapse:collapse;font-size:14px}
td,th{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--muted);font-weight:600}td.num{font-variant-numeric:tabular-nums}
.method{color:var(--ink2);font-size:14px}
.foot{color:var(--muted);font-size:12px;margin-top:40px}
@media (max-width:820px){.hero{grid-template-columns:1fr}.grid{grid-template-columns:repeat(2,1fr)}.frames{grid-template-columns:1fr}.big{font-size:52px}}
"""

ICONS = {"LEGAL": "✔", "LIKELY LEGAL": "✔", "BORDERLINE": "⚠", "SUSPECT": "✖"}


def build_html(res: AnalysisResult, keyframes: dict[int, np.ndarray]) -> str:
    import plotly.offline

    dl = res.primary
    lg, ev, m = dl.legality, dl.events, dl.metrics
    lim = res.config.legal_limit_deg
    scale_max = max(30.0, lg.band_high_deg + 5)
    pct = lambda v: f"{100 * min(max(v, 0), scale_max) / scale_max:.2f}%"

    figs = [flexion_figure(res, dl), elevation_figure(res, dl), speed_figure(res, dl), skeleton_3d_figure(res, dl)]
    fig_html = "".join(
        f'<div class="card">{f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</div>'
        for f in figs)

    tiles = [
        ("Bowling arm", m["bowling_arm"]),
        ("Flexion at UAH", _fmt(lg.flexion_at_uah_deg, "°")),
        ("Flexion at release", _fmt(lg.flexion_at_release_deg, "°")),
        ("ICC window", f"{lg.window_ms:.0f} ms"),
        ("Peak arm speed", _fmt(m["peak_upper_arm_speed_dps"], "°/s", 0)),
        ("Release height*", _fmt(m["release_height_m"], " m", 2)),
        ("Trunk tilt at release", _fmt(m["trunk_tilt_at_release_deg"], "°")),
        ("Front knee", _fmt(m["front_knee_action"])),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in tiles)

    cap = {ev.ffc: "Front-foot contact", ev.uah: "Upper arm horizontal (window start)",
           ev.release: "Ball release (window end)"}
    frames_html = "".join(
        f'<figure><img alt="{html.escape(cap.get(i, ""))}" src="{_b64jpg(annotate_keyframe(img, res, i, cap.get(i, "")))}">'
        f'<figcaption>{html.escape(cap.get(i, ""))} · frame {i} · flexion {res.flex[i]:.1f}°</figcaption></figure>'
        for i, img in sorted(keyframes.items()))

    warns = list(dict.fromkeys(lg.warnings + res.warnings))
    warn_html = ("<div class='warn'><b>⚠ Notes on reliability</b><ul>" +
                 "".join(f"<li>{html.escape(w)}</li>" for w in warns) + "</ul></div>") if warns else ""

    rows = "".join(
        f"<tr><td>#{d.index + 1}</td><td class='num'>{d.events.release / res.fps:.2f} s</td>"
        f"<td class='num'>{d.legality.extension_deg:.1f}°</td>"
        f"<td class='num'>{d.legality.band_low_deg:.1f}–{d.legality.band_high_deg:.1f}°</td>"
        f"<td>{ICONS[d.legality.verdict]} {d.legality.verdict}</td><td>{d.legality.quality}</td></tr>"
        for d in res.deliveries)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ChuckCheck Report</title>
<style>{CSS}</style><script>{plotly.offline.get_plotlyjs()}</script></head>
<body><div class="wrap">
<div class="brand"><h1>CHUCKCHECK</h1><div class="meta">{html.escape(Path(res.source).name or res.source)} ·
{res.fps:.0f} fps · {res.width}×{res.height} · {datetime.now():%d %b %Y %H:%M}</div></div>

<section class="hero" style="--vcol:{lg.verdict_color}">
 <div>
  <div class="badge"><span class="icon">{ICONS[lg.verdict]}</span>{lg.verdict}</div>
  <p class="vtext">{html.escape(lg.verdict_text)}</p>
  <div style="margin-top:22px;color:var(--muted);font-size:13px;text-transform:uppercase;letter-spacing:.06em">Elbow extension in ICC window</div>
  <div class="big">{lg.extension_deg:.1f}°<small>/ {lim:.0f}° limit</small></div>
  <div class="band">Error band {lg.band_low_deg:.1f}° – {lg.band_high_deg:.1f}° · data quality <b>{lg.quality}</b></div>
 </div>
 <div>
  <div style="color:var(--muted);font-size:13px">Where this delivery sits</div>
  <div class="bar"><div class="rng" style="left:{pct(lg.band_low_deg)};width:calc({pct(lg.band_high_deg)} - {pct(lg.band_low_deg)})"></div>
   <div class="lim" style="left:{pct(lim)}" title="ICC limit"></div><div class="est" style="left:{pct(lg.extension_deg)}" title="estimate"></div></div>
  <div class="scale"><span>0°</span><span>{lim:.0f}° limit</span><span>{scale_max:.0f}°</span></div>
  <table style="margin-top:18px"><tr><th>Measure</th><th>Value</th></tr>
   <tr><td>Max straightening (primary)</td><td class="num">{lg.extension_deg:.1f}°</td></tr>
   <tr><td>Peak flexion − flexion at release</td><td class="num">{lg.peak_to_release_deg:.1f}°</td></tr>
   <tr><td>Max flexion in window</td><td class="num">{lg.max_flexion_deg:.1f}°</td></tr>
   <tr><td>Event detection</td><td>{"manual" if ev.manual else "automatic"} ({ev.release_method})</td></tr></table>
 </div>
</section>
{warn_html}
<div class="grid">{tiles_html}</div>

<h2>Key frames</h2><div class="frames">{frames_html}</div>
<h2>Kinematics</h2>{fig_html}
<h2>All deliveries in clip</h2><div class="card"><table><tr><th>#</th><th>Release</th><th>Extension</th><th>Band</th><th>Verdict</th><th>Quality</th></tr>{rows}</table></div>

<h2>Method</h2><div class="card method">
<p><b>Pose.</b> MediaPipe BlazePose estimates 33 body landmarks per frame in image space and as metric 3D
"world" coordinates. The bowler is tracked across frames; short occlusions are interpolated and the
trajectories are smoothed with a Savitzky–Golay filter.</p>
<p><b>Events.</b> The bowling arm is the one that rotates fastest over the shoulder. <i>Upper arm horizontal</i>
is the last upward crossing of 0° elevation before the arm reaches vertical. <i>Ball release</i> is estimated as
the moment the upper arm has rotated {res.config.release_offset_deg:.0f}° past its highest point
(or the wrist-speed peak), and can be corrected by hand.</p>
<p><b>Measurement.</b> Elbow flexion is 180° minus the 3D shoulder–elbow–wrist angle. Following the ICC regulation,
extension is the largest decrease in flexion between upper-arm-horizontal and release. The error band combines a
sensitivity analysis (smoothing window ×0.6–1.4, event timing ±10 ms) with an assumed
±{res.config.pose_error_deg:.0f}° systematic error of monocular 3D pose.</p>
<p>*Release height uses MediaPipe's approximate metric scale.</p></div>

<p class="foot">ChuckCheck v{__version__} — a screening and coaching tool. Official assessments use calibrated
multi-camera motion capture in an ICC-accredited lab; this report is not an official ruling.</p>
</div></body></html>"""


def write_outputs(res: AnalysisResult, out_dir: str | Path, frame_factory: FrameFactory | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    (out / "analysis.json").write_text(json.dumps(res.to_dict(), indent=2, default=_json_default))
    paths["json"] = str(out / "analysis.json")
    timeseries_frame(res).to_csv(out / "timeseries.csv", index=False)
    paths["csv"] = str(out / "timeseries.csv")
    keyframes = {}
    if frame_factory is not None:
        ev = res.primary.events
        keyframes = grab_frames(frame_factory, [ev.ffc, ev.uah, ev.release])
        for i, img in keyframes.items():
            cv2.imwrite(str(out / f"keyframe_{i:05d}.jpg"), annotate_keyframe(img, res, i, ""))
    (out / "report.html").write_text(build_html(res, keyframes), encoding="utf-8")
    paths["html"] = str(out / "report.html")
    return paths


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))
