"""Interactive Plotly figures for the report and the web app."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import landmarks as L
from .pipeline import AnalysisResult, DeliveryResult

# Dark chart chrome (validated reference palette, dark mode)
SURFACE = "#1a1a19"
INK = "#f0efec"
INK2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
SERIES = "#3987e5"        # categorical slot 1 (dark)
SERIES2 = "#d95926"       # categorical slot 2 (dark)
WINDOW = "rgba(57,135,229,0.14)"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def _layout(fig: go.Figure, title: str, ytitle: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0.01, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(family=FONT, color=INK2, size=12),
        margin=dict(l=56, r=20, t=48, b=44), hovermode="x unified", showlegend=False,
        hoverlabel=dict(bgcolor="#262624", bordercolor=AXIS, font=dict(color=INK)),
    )
    fig.update_xaxes(title="Time (s)", gridcolor=GRID, zeroline=False, linecolor=AXIS,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False, linecolor=AXIS,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    return fig


def _focus_range(res: AnalysisResult, dl: DeliveryResult, before=0.6, after=0.4):
    ev = dl.events
    t0 = max(0.0, (ev.ffc if ev.ffc is not None else ev.uah) / res.fps - before)
    t1 = min(res.times[-1], ev.release / res.fps + after)
    return [t0, t1]


def _event_lines(fig, res, dl, y_top):
    ev = dl.events
    marks = [(ev.ffc, "FFC"), (ev.uah, "UAH"), (ev.release, "Release")]
    placed = []
    for f, name in sorted([m for m in marks if m[0] is not None]):
        t = f / res.fps
        # stagger labels that would collide
        level = sum(1 for p in placed if abs(p - t) < 0.06)
        placed.append(t)
        fig.add_vline(x=t, line=dict(color=AXIS, width=1, dash="dot"))
        fig.add_annotation(x=t, y=y_top, text=name, showarrow=False, yshift=10 + 16 * level,
                           font=dict(size=11, color=INK2), bgcolor=SURFACE)


def flexion_figure(res: AnalysisResult, dl: DeliveryResult) -> go.Figure:
    lg, ev = dl.legality, dl.events
    t = res.times
    fig = go.Figure()
    fig.add_vrect(x0=ev.uah / res.fps, x1=ev.release / res.fps, fillcolor=WINDOW, line_width=0)
    fig.add_trace(go.Scatter(x=t, y=res.flex, mode="lines", line=dict(color=SERIES, width=2),
                             name="Elbow flexion", hovertemplate="%{y:.1f}°<extra>Elbow flexion</extra>"))
    # limit line: 15 degrees below the flexion where the straightening starts
    f0 = res.flex[lg.extension_from_frame]
    lim = f0 - res.config.legal_limit_deg
    fig.add_shape(type="line", x0=ev.uah / res.fps, x1=ev.release / res.fps, y0=lim, y1=lim,
                  line=dict(color=INK2, width=1.5, dash="dash"))
    fig.add_annotation(x=ev.release / res.fps, y=lim, text=f"{res.config.legal_limit_deg:.0f}° limit",
                       showarrow=False, xanchor="left", xshift=6, font=dict(size=11, color=INK2))
    # extension arrow
    tf, tt = lg.extension_from_frame / res.fps, lg.extension_to_frame / res.fps
    fig.add_trace(go.Scatter(x=[tf, tt], y=[res.flex[lg.extension_from_frame], res.flex[lg.extension_to_frame]],
                             mode="markers", marker=dict(size=9, color=SERIES, line=dict(color=SURFACE, width=2)),
                             hoverinfo="skip"))
    fig.add_annotation(x=tt, y=res.flex[lg.extension_to_frame], ax=tf, ay=res.flex[lg.extension_from_frame],
                       xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2,
                       arrowcolor=INK, arrowwidth=1.5, text="")
    fig.add_annotation(x=(tf + tt) / 2, y=(res.flex[lg.extension_from_frame] + res.flex[lg.extension_to_frame]) / 2,
                       text=f"<b>{lg.extension_deg:.1f}°</b> extension", showarrow=False, xanchor="right",
                       xshift=-10, font=dict(size=13, color=INK))
    y_top = float(np.nanmax(res.flex)) + 4
    _event_lines(fig, res, dl, y_top)
    _layout(fig, "Elbow flexion (0° = straight arm); shaded = ICC window", "Flexion (°)")
    fig.update_xaxes(range=_focus_range(res, dl))
    fig.update_yaxes(range=[min(0, float(np.nanmin(res.flex)) - 2), y_top + 6])
    return fig


def elevation_figure(res: AnalysisResult, dl: DeliveryResult) -> go.Figure:
    ev = dl.events
    fig = go.Figure()
    fig.add_vrect(x0=ev.uah / res.fps, x1=ev.release / res.fps, fillcolor=WINDOW, line_width=0)
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.add_annotation(x=_focus_range(res, dl)[0], y=0, text="horizontal", showarrow=False, xanchor="left",
                       yshift=8, font=dict(size=11, color=MUTED))
    fig.add_trace(go.Scatter(x=res.times, y=res.elev, mode="lines", line=dict(color=SERIES, width=2),
                             hovertemplate="%{y:.1f}°<extra>Upper-arm elevation</extra>"))
    _event_lines(fig, res, dl, 95)
    _layout(fig, "Upper-arm elevation (−90° down, 0° horizontal, +90° vertical)", "Elevation (°)", 300)
    fig.update_xaxes(range=_focus_range(res, dl))
    fig.update_yaxes(range=[-100, 110])
    return fig


def speed_figure(res: AnalysisResult, dl: DeliveryResult) -> go.Figure:
    ev = dl.events
    fig = go.Figure()
    fig.add_vrect(x0=ev.uah / res.fps, x1=ev.release / res.fps, fillcolor=WINDOW, line_width=0)
    fig.add_trace(go.Scatter(x=res.times, y=res.arm_speed, mode="lines", line=dict(color=SERIES, width=2),
                             hovertemplate="%{y:.0f}°/s<extra>Upper-arm angular speed</extra>"))
    _event_lines(fig, res, dl, float(np.nanmax(res.arm_speed)) * 1.05)
    _layout(fig, "Upper-arm angular speed", "°/s", 280)
    fig.update_xaxes(range=_focus_range(res, dl))
    return fig


def skeleton_3d_figure(res: AnalysisResult, dl: DeliveryResult, max_frames: int = 80) -> go.Figure:
    """Rotatable 3D replay of the delivery (world landmarks), bowling arm highlighted."""
    ev = dl.events
    a = max(0, (ev.ffc if ev.ffc is not None else ev.uah) - int(0.25 * res.fps))
    b = min(res.n_frames - 1, ev.release + int(0.3 * res.fps))
    idx = np.unique(np.linspace(a, b, min(max_frames, b - a + 1)).astype(int))
    W = res.world_s
    # plotly axes: x = forward, y = depth, z = up
    X, Y, Z = W[..., 0], W[..., 2], -W[..., 1]
    arm = L.arm_edges(res.side)
    body_edges = [e for e in L.SKELETON_EDGES if e not in arm and e[0] >= 11]

    def seg(i, edges):
        xs, ys, zs = [], [], []
        for p, q in edges:
            xs += [X[i, p], X[i, q], None]; ys += [Y[i, p], Y[i, q], None]; zs += [Z[i, p], Z[i, q], None]
        return xs, ys, zs

    def traces(i):
        bx, by, bz = seg(i, body_edges)
        ax_, ay, az = seg(i, list(arm))
        s, e, w = L.arm(res.side)
        tr = [go.Scatter3d(x=bx, y=by, z=bz, mode="lines", line=dict(color=INK2, width=5), hoverinfo="skip"),
              go.Scatter3d(x=ax_, y=ay, z=az, mode="lines", line=dict(color=SERIES2, width=10), hoverinfo="skip"),
              go.Scatter3d(x=[X[i, e]], y=[Y[i, e]], z=[Z[i, e]], mode="markers+text",
                           marker=dict(size=5, color=SERIES2), text=[f"{res.flex[i]:.1f}°"],
                           textfont=dict(color=INK, size=13), textposition="top center", hoverinfo="skip")]
        return tr

    fig = go.Figure(data=traces(idx[0]))
    frames = []
    for i in idx:
        tag = " • UAH" if i == ev.uah else " • RELEASE" if i == ev.release else ""
        frames.append(go.Frame(data=traces(i), name=str(i),
                               layout=go.Layout(title_text=f"3D replay  •  t = {i / res.fps:.3f} s{tag}")))
    fig.frames = frames
    sel = W[idx][:, 11:]
    lo, hi = np.nanmin(sel.reshape(-1, 3), 0), np.nanmax(sel.reshape(-1, 3), 0)
    pad = 0.15
    rng = lambda k: [float(lo[k] - pad), float(hi[k] + pad)]
    steps = [dict(method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                         transition=dict(duration=0))],
                  label=f"{i / res.fps:.2f}") for i in idx]
    fig.update_layout(
        title=dict(text="3D replay", font=dict(size=15, color=INK), x=0.01),
        paper_bgcolor=SURFACE, font=dict(family=FONT, color=INK2), height=560, showlegend=False,
        margin=dict(l=0, r=0, t=48, b=0),
        scene=dict(
            xaxis=dict(range=rng(0), title="forward (m)", backgroundcolor=SURFACE, gridcolor=GRID, color=MUTED),
            yaxis=dict(range=rng(2), title="depth (m)", backgroundcolor=SURFACE, gridcolor=GRID, color=MUTED),
            zaxis=dict(range=[-float(hi[1]) - pad, -float(lo[1]) + pad], title="up (m)",
                       backgroundcolor=SURFACE, gridcolor=GRID, color=MUTED),
            aspectmode="data", camera=dict(eye=dict(x=0.2, y=-2.0, z=0.4)),
        ),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                          bgcolor="#262624", font=dict(color=INK),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True)]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(steps=steps, x=0.2, len=0.78, y=0.02, currentvalue=dict(prefix="t = ", suffix=" s",
                      font=dict(color=INK2)), font=dict(color=MUTED), bgcolor="#262624")],
    )
    return fig
