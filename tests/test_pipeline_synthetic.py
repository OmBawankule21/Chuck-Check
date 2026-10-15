"""The synthetic bowler has a known elbow extension; the pipeline must recover it."""
import numpy as np
import pytest

from chuckcheck import AnalysisConfig
from chuckcheck.pipeline import AnalysisError, analyze_sequence
from chuckcheck.pose import PoseSequence
from chuckcheck.synthetic import SyntheticBowler, SyntheticSpec


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("ext", [5.0, 14.0, 22.0, 32.0])
@pytest.mark.parametrize("fps", [240.0, 120.0])
def test_recovers_known_extension(side, ext, fps):
    b = SyntheticBowler(SyntheticSpec(extension_deg=ext, fps=fps, side=side))
    res = analyze_sequence(b.sequence)
    d = res.primary
    assert res.side == side
    assert len(res.deliveries) == 1
    assert abs(d.events.uah - b.true_uah) <= max(2, int(0.01 * fps))
    assert abs(d.events.release - b.true_release) <= max(2, int(0.01 * fps))
    assert abs(d.legality.extension_deg - ext) < 2.5
    assert d.legality.band_low_deg <= ext + 2.5 and d.legality.band_high_deg >= ext - 2.5


def test_verdicts_follow_extension():
    v = {e: analyze_sequence(SyntheticBowler(SyntheticSpec(extension_deg=e)).sequence).primary.legality.verdict
         for e in (4.0, 30.0)}
    assert v[4.0] == "LEGAL"
    assert v[30.0] == "SUSPECT"


def test_low_fps_warns_and_degrades_quality():
    res = analyze_sequence(SyntheticBowler(SyntheticSpec(extension_deg=20, fps=30)).sequence)
    lg = res.primary.legality
    assert lg.quality == "C"
    assert any("fps" in w for w in lg.warnings)


def test_occlusions_are_bridged():
    b = SyntheticBowler(SyntheticSpec(extension_deg=20, fps=120))
    seq = b.sequence
    rng = np.random.default_rng(1)
    drop = rng.random(seq.n_frames) < 0.08
    drop[b.true_release - 2:b.true_release + 3] = False
    det = seq.detected.copy(); det[drop] = False
    world = seq.world.copy(); world[drop] = np.nan
    s2 = PoseSequence(seq.fps, seq.width, seq.height, seq.image, world, seq.visibility, det, seq.source)
    res = analyze_sequence(s2)
    assert abs(res.primary.legality.extension_deg - 20) < 3


def test_two_deliveries_in_one_clip():
    a = SyntheticBowler(SyntheticSpec(extension_deg=8, fps=120, seed=1)).sequence
    b = SyntheticBowler(SyntheticSpec(extension_deg=26, fps=120, seed=2)).sequence
    cat = lambda x, y: np.concatenate([x, y])
    seq = PoseSequence(120.0, a.width, a.height, cat(a.image, b.image), cat(a.world, b.world),
                       cat(a.visibility, b.visibility), cat(a.detected, b.detected), "two")
    res = analyze_sequence(seq)
    assert len(res.deliveries) == 2
    e = [d.legality.extension_deg for d in res.deliveries]
    assert abs(e[0] - 8) < 2.5 and abs(e[1] - 26) < 2.5


def test_manual_override():
    b = SyntheticBowler(SyntheticSpec(extension_deg=20, fps=120))
    res = analyze_sequence(b.sequence, overrides={0: {"uah": b.true_uah, "release": b.true_uah + 5}})
    d = res.primary
    assert d.events.manual and d.events.release == b.true_uah + 5
    assert d.legality.extension_deg < 20


def test_rejects_clip_without_bowling():
    b = SyntheticBowler(SyntheticSpec(extension_deg=20, fps=120))
    seq = b.sequence
    keep = slice(0, int(0.35 * seq.n_frames))  # run-up only, no delivery
    s2 = PoseSequence(seq.fps, seq.width, seq.height, seq.image[keep], seq.world[keep],
                      seq.visibility[keep], seq.detected[keep], "runup")
    with pytest.raises(AnalysisError):
        analyze_sequence(s2)


def test_pose_sequence_roundtrip(tmp_path):
    seq = SyntheticBowler(SyntheticSpec(fps=60)).sequence
    seq.save(tmp_path / "p.npz")
    s2 = PoseSequence.load(tmp_path / "p.npz")
    assert s2.n_frames == seq.n_frames and np.allclose(s2.world, seq.world)


def test_config_release_method_wrist_speed():
    b = SyntheticBowler(SyntheticSpec(extension_deg=20, fps=240))
    res = analyze_sequence(b.sequence, AnalysisConfig(release_method="wrist-speed"))
    assert res.primary.events.release_method == "wrist-speed"
    assert res.primary.events.release > res.primary.events.vertical - 1
