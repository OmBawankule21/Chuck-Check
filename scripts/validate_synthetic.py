"""Accuracy check against synthetic bowlers with known elbow extension.

    python scripts/validate_synthetic.py
Prints a markdown table of absolute error by frame rate.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chuckcheck.pipeline import analyze_sequence  # noqa: E402
from chuckcheck.synthetic import SyntheticBowler, SyntheticSpec  # noqa: E402

rows = []
for fps in (240, 120, 60, 30):
    errs, correct, ev_err = [], 0, []
    cases = [(e, side, seed) for e in (0, 5, 10, 15, 20, 25, 30, 35) for side in ("right", "left") for seed in (1, 2)]
    for ext, side, seed in cases:
        b = SyntheticBowler(SyntheticSpec(extension_deg=ext, fps=fps, side=side, seed=seed))
        d = analyze_sequence(b.sequence).primary
        errs.append(abs(d.legality.extension_deg - ext))
        ev_err.append(1000 * abs(d.events.release - b.true_release) / fps)
        correct += (d.legality.band_low_deg <= ext <= d.legality.band_high_deg)
    rows.append((fps, np.mean(errs), np.max(errs), 100 * correct / len(cases), np.mean(ev_err), len(cases)))

print("| Frame rate | Mean abs. error | Max error | Truth inside error band | Release timing error | Cases |")
print("|---|---|---|---|---|---|")
for fps, me, mx, cov, ev, n in rows:
    print(f"| {fps} fps | {me:.2f}° | {mx:.2f}° | {cov:.0f}% | {ev:.1f} ms | {n} |")
