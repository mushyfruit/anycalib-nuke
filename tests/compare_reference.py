"""The plugin's engine and fit (anycalib_cli) against AnyCalib's own pipeline
(test-assets/reference.json), as the angle between the rays each assigns to a 21 x 21 grid
of plate pixels. k1/k2 alone trade off against each other and focal, so aren't compared.

    python compare_reference.py build/anycalib_cli models/anycalib_dist.pt test-assets [cuda|cpu]

The metric is model-agnostic: the angle between the rays the two calibrations
assign to the same plate pixels (a 21x21 grid), which is AnyCalib's own residual
space. Focal and principal point are reported too. k1/k2 alone are not compared:
for weak distortion they trade off against each other and against focal.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


def unproject(fx, fy, cx, cy, k1, k2, u, v):
    x0, y0 = (u - cx) / fx, (v - cy) / fy
    rd = np.hypot(x0, y0)
    ru = rd.copy()
    for _ in range(50):
        r2 = ru * ru
        f = ru * (1 + k1 * r2 + k2 * r2 * r2) - rd
        df = np.maximum(1 + 3 * k1 * r2 + 5 * k2 * r2 * r2, 1e-9)
        ru = np.maximum(ru - f / df, 0)
    sc = np.where(rd > 1e-12, ru / np.maximum(rd, 1e-12), 1.0)
    d = np.stack([x0 * sc, y0 * sc, np.ones_like(x0)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def main(cli, model, assets, device="cuda"):
    ref = json.load(open(os.path.join(assets, "reference.json")))
    tmp = tempfile.mkdtemp()
    names, ppms = [], []
    for name in sorted(ref):
        p = os.path.join(tmp, os.path.splitext(name)[0] + ".ppm")
        Image.open(os.path.join(assets, name)).convert("RGB").save(p)
        names.append(name)
        ppms.append(p)
    out = subprocess.run([cli, model, device] + ppms, capture_output=True, text=True).stdout
    ours = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    print("%-42s %9s %9s %9s %9s  %s" % ("image", "f ref", "f ours", "max deg", "mean deg", "centre diff px"))
    worst = 0.0
    for name, o in zip(names, ours):
        r = ref[name]
        W, H = r["width"], r["height"]
        u, v = np.meshgrid(np.linspace(0.5, W - 0.5, 21), np.linspace(0.5, H - 0.5, 21))
        a = unproject(r["fx"], r["fy"], r["cx"], r["cy"], r["k1"], r["k2"], u, v)
        c = o["opencv"]
        b = unproject(c["fx"], c["fy"], c["cx"], c["cy"], c["k1"], c["k2"], u, v)
        ang = np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))
        worst = max(worst, ang.max())
        print("%-42s %9.1f %9.1f %9.3f %9.3f  (%.1f, %.1f)" % (
            name, 0.5 * (r["fx"] + r["fy"]), 0.5 * (c["fx"] + c["fy"]), ang.max(), ang.mean(),
            c["cx"] - r["cx"], c["cy"] - r["cy"]))
    print("\nworst ray-angle difference: %.3f deg" % worst)


if __name__ == "__main__":
    main(*sys.argv[1:])
