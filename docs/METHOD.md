# ChuckCheck method notes

## 1. Pose estimation
* MediaPipe **PoseLandmarker** (BlazePose GHUM, `lite` / `full` / `heavy`) runs in VIDEO mode, so it tracks between frames.
* Each frame yields 33 landmarks as normalised image coordinates and as **world coordinates** in metres, with the origin between the hips and the y axis pointing down.
* Up to `num_poses = 2` people are detected. The bowler is the person whose hip centre is nearest to the previous frame's. If the detection jumps more than 25% of the frame within 0.5 s, that frame is treated as a mis-detection (for example, the umpire).
* Frames larger than 1280 px are downscaled before inference. Landmarks are normalised, so accuracy in the original pixel space is kept.

## 2. Signal conditioning
* Landmarks with visibility below 0.5 are treated as missing. Gaps up to 0.12 s are linearly interpolated; longer gaps are flagged invalid.
* World and image trajectories are smoothed with a **Savitzky–Golay** filter (window 0.10 s, order 3), which keeps peaks better than a moving average.

## 3. Bowling arm and deliveries
* For each arm, the angular speed of the upper-arm direction vector is median-filtered, then integrated over any 0.4 s window. The arm that rotates furthest (and rises above 45°) is the bowling arm.
* Deliveries are peaks of upper-arm angular speed that also show the arm rising from below horizontal to above 50° elevation, rotating at least 150° within ±0.3 s, and being at least 80% genuinely observed (not interpolated). Peaks closer than 1.5 s are merged.

## 4. Events
| Event | Definition |
|---|---|
| Arm vertical | max upper-arm elevation within −0.35 s / +0.25 s of the speed peak |
| **UAH** (window start) | last upward 0° crossing of elevation before vertical (nearest frame) |
| **Release** (window end) | first frame after vertical where elevation has dropped `release_offset_deg` (15°) below its peak; *or* the wrist-speed peak within 150 ms after vertical |
| Front-foot contact | start of the stationary phase of the front ankle (image speed < 1.2 torso-lengths/s) that contains arm-vertical |

All events can be overridden: use `reanalyze --uah --release` on the command line, or the **Correct events** tab in the app.

## 5. Elbow extension
* `flexion(t) = 180° − angle(shoulder, elbow, wrist)` in 3D.
* `extension = max over UAH ≤ i ≤ j ≤ release of [ flexion(i) − flexion(j) ]`, the largest straightening anywhere in the window. This is ≥ the simpler `max flexion − flexion at release`, which is also reported.

## 6. Uncertainty
27 re-measurements with smoothing window ×{0.6, 1, 1.4}, UAH ±10 ms and release ±10 ms. The band is
`[P10 − e, P90 + e]`, where `e` is the assumed systematic pose error (default 4°, configurable).

## 7. Data-quality grade
The worst of three grades:
* frame rate: ≥100 fps = A, ≥50 fps = B, otherwise C
* frames in the ICC window: ≥8 = A, ≥4 = B, otherwise C
* mean visibility of the bowling arm in the window: ≥0.8 = A, ≥0.6 = B, otherwise C

## 8. Other metrics
* **Peak upper-arm angular speed** (°/s), from UAH to release + 50 ms.
* **Release height**: wrist height above the lowest foot landmark at release. This is approximate, because MediaPipe's world scale is learned, not calibrated.
* **Trunk tilt at release**: angle of the mid-hip → mid-shoulder vector from vertical.
* **Front-knee action** from knee flexion at FFC and at release: `Braced` (|Δ| ≤ 10°, flexion at FFC ≤ 25°), `Flexor` (Δ > 10°), `Extender` (Δ < −10°), otherwise `Constant (flexed)`.

## 9. Validation
`scripts/validate_synthetic.py` uses `chuckcheck/synthetic.py`. That generator is a kinematic bowler with a configurable,
known elbow-extension profile, landmark noise, run-up, delivery stride, braced front leg and follow-through.
It validates the measurement chain end to end. It does not validate the pose model's accuracy on real people.
