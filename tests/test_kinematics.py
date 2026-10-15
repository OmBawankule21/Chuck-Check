import numpy as np

from chuckcheck.kinematics import elbow_flexion, joint_angle, upper_arm_elevation
from chuckcheck.legality import classify, max_extension
from chuckcheck.filters import fill_gaps, odd_window, smooth


def test_joint_angle_right_angle():
    a = np.array([[1.0, 0, 0]]); b = np.zeros((1, 3)); c = np.array([[0, 1.0, 0]])
    assert np.isclose(joint_angle(a, b, c)[0], 90.0)


def test_straight_arm_has_zero_flexion_and_elevation_signs():
    w = np.zeros((2, 33, 3))
    # right shoulder, elbow, wrist in a straight vertical line pointing up (y is down)
    w[:, 12] = [0, -0.5, 0]; w[:, 14] = [0, -0.8, 0]; w[:, 16] = [0, -1.1, 0]
    assert np.allclose(elbow_flexion(w, "right"), 0, atol=1e-6)
    assert np.allclose(upper_arm_elevation(w, "right"), 90)
    w[:, 14] = [0.3, -0.5, 0]  # horizontal upper arm
    assert np.allclose(upper_arm_elevation(w, "right"), 0, atol=1e-6)


def test_max_extension_is_largest_later_drop():
    flex = np.array([10, 30, 25, 28, 12, 20, 5], float)
    ext, i, j = max_extension(flex, 0, 6)
    assert ext == 25 and i == 1 and j == 6
    # flexion only increasing -> no extension
    assert max_extension(np.arange(10.0), 0, 9)[0] == 0


def test_classify_bands():
    assert classify(8, 3, 13, 15) == "LEGAL"
    assert classify(12, 7, 18, 15) == "LIKELY LEGAL"
    assert classify(17, 12, 22, 15) == "BORDERLINE"
    assert classify(25, 19, 30, 15) == "SUSPECT"


def test_fill_gaps_interpolates_short_and_flags_long():
    x = np.arange(20, dtype=float)[:, None]
    valid = np.ones(20, bool); valid[5:7] = False; valid[10:18] = False
    filled, ok = fill_gaps(x, valid, max_gap=3)
    assert np.allclose(filled[:, 0], np.arange(20))
    assert ok[5] and ok[6] and not ok[12]


def test_smoothing_keeps_shape():
    t = np.linspace(0, 1, 121)
    y = np.sin(2 * np.pi * t)[:, None] + np.random.default_rng(0).normal(0, 0.05, (121, 1))
    ys = smooth(y, odd_window(0.1, 120), 3)
    assert ys.shape == y.shape
    assert np.abs(ys[:, 0] - np.sin(2 * np.pi * t)).mean() < 0.03
