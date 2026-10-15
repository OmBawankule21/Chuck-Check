"""Analysis configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

Side = Literal["auto", "left", "right"]


@dataclass
class AnalysisConfig:
    # --- pose estimation -------------------------------------------------
    model: Literal["lite", "full", "heavy"] = "full"
    model_path: Optional[str] = None      # use a local .task file instead of downloading
    num_poses: int = 2                    # >1 lets us track the bowler when others are in frame
    max_infer_side: int = 1280            # downscale big frames before inference (speed)
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # --- signal processing ----------------------------------------------
    min_visibility: float = 0.5           # landmarks below this are treated as missing
    max_gap_s: float = 0.12               # interpolate gaps shorter than this
    smooth_window_s: float = 0.10         # Savitzky-Golay window length
    smooth_polyorder: int = 3

    # --- event detection -------------------------------------------------
    bowling_arm: Side = "auto"
    release_method: Literal["arm-angle", "wrist-speed"] = "arm-angle"
    release_offset_deg: float = 15.0      # release ~= upper arm this far past its peak elevation
    min_delivery_separation_s: float = 1.5

    # --- legality --------------------------------------------------------
    legal_limit_deg: float = 15.0         # ICC limit on elbow extension
    pose_error_deg: float = 4.0           # assumed systematic error of monocular 3D pose

    # --- rendering -------------------------------------------------------
    slowmo: float = 0.25                  # playback speed inside the delivery window (1 = off)
    slowmo_window_s: float = 0.35         # +/- around release
    verdict_hold_s: float = 2.5           # how long the verdict card is shown at the end

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
