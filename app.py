"""ChuckCheck web app.  Run with:  streamlit run app.py"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from chuckcheck import AnalysisConfig, __version__
from chuckcheck.pipeline import AnalysisError, analyze_sequence
from chuckcheck.plots import elevation_figure, flexion_figure, skeleton_3d_figure, speed_figure
from chuckcheck.pose import PoseExtractor, iter_frames
from chuckcheck.render import render_video
from chuckcheck.report import annotate_keyframe, grab_frames, write_outputs
from chuckcheck.synthetic import SyntheticBowler, SyntheticSpec

st.set_page_config(page_title="ChuckCheck", page_icon="🏏", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1200px}
.cc-brand{font-weight:800;letter-spacing:.16em;color:#f5a60b;font-size:1.4rem}
.cc-sub{color:#898781;margin-top:-4px}
.cc-hero{background:#1a1a19;border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:22px 26px;
 border-left:6px solid var(--vcol);margin:8px 0 16px}
.cc-verdict{font-size:2rem;font-weight:800;color:var(--vcol)}
.cc-big{font-size:3.4rem;font-weight:800;line-height:1.05;color:#f0efec}
.cc-big small{font-size:1.1rem;color:#c3c2b7;font-weight:600}
.cc-muted{color:#c3c2b7}
</style>""", unsafe_allow_html=True)

ICONS = {"LEGAL": "✔", "LIKELY LEGAL": "✔", "BORDERLINE": "⚠", "SUSPECT": "✖"}
ss = st.session_state
if "workdir" not in ss:
    ss.workdir = Path(tempfile.mkdtemp(prefix="chuckcheck_"))

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="cc-brand">CHUCKCHECK</div><div class="cc-sub">Bowling-action legality from video</div>',
                unsafe_allow_html=True)
    st.divider()
    source = st.radio("Input", ["Upload a video", "Synthetic demo bowler"], horizontal=False)
    if source == "Upload a video":
        upl = st.file_uploader("Bowling clip (side-on, full body, ideally 120–240 fps)",
                               type=["mp4", "mov", "avi", "mkv", "m4v"])
        demo_ext = None
    else:
        upl = None
        demo_ext = st.slider("True elbow extension of the demo bowler (°)", 0.0, 40.0, 22.0, 0.5)
        demo_fps = st.select_slider("Demo frame rate", [30, 60, 120, 240], value=120)
    with st.expander("Settings", expanded=False):
        arm = st.selectbox("Bowling arm", ["auto", "right", "left"])
        model = st.selectbox("Pose model", ["full", "heavy", "lite"],
                             help="heavy = most accurate, lite = fastest")
        rmethod = st.selectbox("Release detection", ["arm-angle", "wrist-speed"])
        roffset = st.slider("Release: degrees past peak arm elevation", 0.0, 40.0, 15.0, 1.0)
        perr = st.slider("Assumed pose error (°)", 0.0, 10.0, 4.0, 0.5)
        slowmo = st.select_slider("Slow-mo speed around release", [1.0, 0.5, 0.25, 0.125], value=0.25)
        do_render = st.checkbox("Render annotated video", value=True)
    run = st.button("Analyse", type="primary", width="stretch")
    st.caption(f"v{__version__} · screening tool, not an official ICC test")

cfg = AnalysisConfig(model=model, bowling_arm=arm, release_method=rmethod, release_offset_deg=roffset,
                     pose_error_deg=perr, slowmo=slowmo)


def frames_factory():
    if ss.get("bowler") is not None:
        return ss.bowler.frames
    return lambda: iter_frames(ss.video_path)


def finish(res, render: bool):
    out = ss.workdir / "out"
    out.mkdir(exist_ok=True)
    bar = st.progress(0.0, "Rendering…")
    paths = {}
    if render:
        paths["video"] = render_video(res, frames_factory()(), out / "annotated.mp4",
                                      lambda f, m: bar.progress(min(f, 1.0), m))
    paths.update(write_outputs(res, out, frames_factory()))
    bar.empty()
    ss.result, ss.paths = res, paths


# ----------------------------------------------------------------------------- run
if run:
    try:
        if source == "Upload a video":
            if upl is None:
                st.warning("Upload a video first.")
                st.stop()
            vp = ss.workdir / upl.name
            vp.write_bytes(upl.getbuffer())
            ss.video_path, ss.bowler = str(vp), None
            bar = st.progress(0.0, "Loading pose model…")
            seq = PoseExtractor(cfg).extract(vp, lambda f, m: bar.progress(min(f, 1.0), m))
            bar.empty()
        else:
            b = SyntheticBowler(SyntheticSpec(extension_deg=demo_ext, fps=float(demo_fps)))
            ss.bowler, ss.video_path, seq = b, None, b.sequence
        ss.seq = seq
        res = analyze_sequence(seq, cfg)
        finish(res, do_render)
    except AnalysisError as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

# ----------------------------------------------------------------------------- results
if "result" not in ss:
    st.markdown('<div class="cc-brand" style="font-size:2.2rem">CHUCKCHECK</div>', unsafe_allow_html=True)
    st.markdown("#### Is that bowling action legal? Measure elbow extension from a phone video.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Film**  \nSide-on to the bowling arm, whole body in frame, slow-motion (120–240 fps) if your phone has it.")
    c2.markdown("**2 · Analyse**  \n3D pose → upper-arm-horizontal & release detection → elbow extension with an error band.")
    c3.markdown("**3 · Review**  \nAnnotated slow-mo video, charts, 3D replay, key frames and a shareable report card.")
    st.info("No clip handy? Choose **Synthetic demo bowler** in the sidebar and press **Analyse**.")
    st.stop()

res = ss.result
if len(res.deliveries) > 1:
    di = st.selectbox("Delivery", range(len(res.deliveries)),
                      format_func=lambda i: f"#{i + 1} · release at {res.deliveries[i].events.release / res.fps:.2f}s")
else:
    di = 0
dl = res.deliveries[di]
lg, ev, m = dl.legality, dl.events, dl.metrics

st.markdown(f"""
<div class="cc-hero" style="--vcol:{lg.verdict_color}">
 <div class="cc-verdict">{ICONS[lg.verdict]} {lg.verdict}</div>
 <div class="cc-muted">{lg.verdict_text}</div>
 <div style="margin-top:14px" class="cc-big">{lg.extension_deg:.1f}°<small> elbow extension · limit {cfg.legal_limit_deg:.0f}°</small></div>
 <div class="cc-muted">Error band {lg.band_low_deg:.1f}° – {lg.band_high_deg:.1f}° · data quality <b>{lg.quality}</b> ·
 window {lg.window_ms:.0f} ms ({lg.window_frames} frames)</div>
</div>""", unsafe_allow_html=True)
if ss.get("bowler") is not None:
    st.caption(f"Synthetic ground truth: {ss.bowler.spec.extension_deg:.1f}° (UAH frame {ss.bowler.true_uah}, "
               f"release frame {ss.bowler.true_release})")
for w in dict.fromkeys(lg.warnings + res.warnings):
    st.warning(w, icon="⚠️")

cols = st.columns(4)
cols[0].metric("Bowling arm", m["bowling_arm"])
cols[1].metric("Flexion at UAH", f"{lg.flexion_at_uah_deg:.1f}°")
cols[2].metric("Flexion at release", f"{lg.flexion_at_release_deg:.1f}°")
cols[3].metric("Peak arm speed", f"{m['peak_upper_arm_speed_dps']:.0f}°/s")
cols = st.columns(4)
cols[0].metric("Release height*", f"{m['release_height_m']:.2f} m")
cols[1].metric("Trunk tilt at release", f"{m['trunk_tilt_at_release_deg']:.1f}°")
cols[2].metric("Front knee", m["front_knee_action"] or "–")
cols[3].metric("FFC → release", f"{m['ffc_to_release_ms']:.0f} ms" if m["ffc_to_release_ms"] else "–")

t_vid, t_chart, t_3d, t_kf, t_fix, t_dl = st.tabs(
    ["🎬 Annotated video", "📈 Kinematics", "🧍 3D replay", "🖼 Key frames", "🛠 Correct events", "⬇ Downloads"])

with t_vid:
    if "video" in ss.paths:
        st.video(ss.paths["video"])
    else:
        st.info("Video rendering was turned off in Settings.")
with t_chart:
    st.plotly_chart(flexion_figure(res, dl), width="stretch")
    st.plotly_chart(elevation_figure(res, dl), width="stretch")
    st.plotly_chart(speed_figure(res, dl), width="stretch")
with t_3d:
    st.plotly_chart(skeleton_3d_figure(res, dl), width="stretch")
with t_kf:
    idx = [ev.ffc, ev.uah, ev.release]
    frames = grab_frames(frames_factory(), idx)
    names = {ev.ffc: "Front-foot contact", ev.uah: "Upper arm horizontal", ev.release: "Ball release"}
    kc = st.columns(len(frames) or 1)
    for c, (i, img) in zip(kc, sorted(frames.items())):
        c.image(annotate_keyframe(img, res, i, "")[:, :, ::-1], caption=f"{names[i]} · frame {i} · {res.flex[i]:.1f}°")
with t_fix:
    st.markdown("Automatic event detection can be off by a frame or two. Step through the video, then set the "
                "frames where the **upper arm is horizontal** and where the **ball leaves the hand**.")
    c1, c2 = st.columns(2)
    uah = c1.number_input("Upper-arm-horizontal frame", 0, res.n_frames - 2, int(ev.uah))
    rel = c2.number_input("Ball-release frame", 1, res.n_frames - 1, int(ev.release))
    peek = st.slider("Preview frame", 0, res.n_frames - 1, int(ev.release))
    pf = grab_frames(frames_factory(), [peek])
    if pf:
        st.image(annotate_keyframe(pf[peek], res, peek, "")[:, :, ::-1],
                 caption=f"frame {peek} · t={peek / res.fps:.3f}s · flexion {res.flex[peek]:.1f}° · "
                         f"elevation {res.elev[peek]:.1f}°", width=420)
    if st.button("Re-analyse with these frames", type="primary"):
        res2 = analyze_sequence(ss.seq, cfg, {di: {"uah": int(uah), "release": int(rel)}})
        finish(res2, "video" in ss.paths)
        st.rerun()
with t_dl:
    p = ss.paths
    st.download_button("Report card (HTML)", Path(p["html"]).read_bytes(), "chuckcheck_report.html", "text/html")
    st.download_button("Analysis (JSON)", Path(p["json"]).read_bytes(), "analysis.json", "application/json")
    st.download_button("Time series (CSV)", Path(p["csv"]).read_bytes(), "timeseries.csv", "text/csv")
    if "video" in p:
        st.download_button("Annotated video (MP4)", Path(p["video"]).read_bytes(), "annotated.mp4", "video/mp4")
    with st.expander("Raw analysis JSON"):
        st.json(json.loads(Path(p["json"]).read_text())["deliveries"])
st.caption("*Release height uses MediaPipe's approximate metric scale. ChuckCheck is a screening and coaching "
           "tool; official assessments use calibrated multi-camera motion capture.")
