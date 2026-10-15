"""MediaPipe BlazePose 33-landmark topology and helpers."""
from __future__ import annotations

NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]
N_LANDMARKS = len(NAMES)

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

SKELETON_EDGES = [
    (11, 12), (11, 23), (12, 24), (23, 24),          # torso
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # left arm/hand
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # right arm/hand
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),  # right leg
    (0, 2), (0, 5), (2, 7), (5, 8), (9, 10),          # face
]


def arm(side: str) -> tuple[int, int, int]:
    """(shoulder, elbow, wrist) indices for 'left' or 'right'."""
    if side == "left":
        return L_SHOULDER, L_ELBOW, L_WRIST
    if side == "right":
        return R_SHOULDER, R_ELBOW, R_WRIST
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def leg(side: str) -> tuple[int, int, int, int, int]:
    """(hip, knee, ankle, heel, foot_index) indices."""
    if side == "left":
        return L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_FOOT
    if side == "right":
        return R_HIP, R_KNEE, R_ANKLE, R_HEEL, R_FOOT
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def other(side: str) -> str:
    return "left" if side == "right" else "right"


def arm_edges(side: str) -> set[tuple[int, int]]:
    s, e, w = arm(side)
    return {(s, e), (e, w)}
