"""The plugin's engine and fit (anycalib_cli) against AnyCalib's own pipeline
(test-assets/reference.json), as the angle between the rays each assigns to a 21 x 21 grid
of plate pixels. k1/k2 alone trade off against each other and focal, so aren't compared.

    python compare_reference.py build/anycalib_cli models/anycalib_dist.pt test-assets [auto|cpu]
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


def rays(f, cx, cy, k1, k2, u, v, s=None):
    """Unit rays of a division camera: ((u - cx) / s, (v - cy) / s, (f / s) D(r)), with r
    the radius in units of s (the focal length by default) and D = 1 + k1 r^2 + k2 r^4."""
    s = f if s is None else s
    x, y = (u - cx) / s, (v - cy) / s
    r2 = x * x + y * y
    d = np.stack([x, y, f / s * (1 + k1 * r2 + k2 * r2 * r2)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def main(cli, model, assets, device="auto"):
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
        a = rays(r["f"], r["cx"], r["cy"], r["k1"], r["k2"], u, v)
        n, S = o["nuke"], 0.5 * max(W, H)
        cx, cy = 0.5 * W + n["centre"][0] * S, 0.5 * H - n["centre"][1] * S  # Nuke is y up
        b = rays(n["focal"], cx, cy, n["distortion1"], n["distortion2"], u, v, S)
        ang = np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))
        worst = max(worst, ang.max())
        print("%-42s %9.1f %9.1f %9.3f %9.3f  (%.1f, %.1f)" % (
            name, r["f"], n["focal"], ang.max(), ang.mean(), cx - r["cx"], cy - r["cy"]))
    print("\nworst ray-angle difference: %.3f deg" % worst)


if __name__ == "__main__":
    main(*sys.argv[1:])
