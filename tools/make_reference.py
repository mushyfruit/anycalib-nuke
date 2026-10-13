#!/usr/bin/env python3
"""Reference intrinsics from AnyCalib's own Python pipeline (not the exported network), for
tests/compare_reference.py. Runs AnyCalib.predict on every image in --images-dir as in
AnyCalib's README and writes {filename: {width, height, fx, fy, cx, cy, k1, k2, ...}}.
Pixel centres are at +0.5 (OpenCV puts them at integers).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", default="test-assets")
    p.add_argument("--output", default="test-assets/reference.json")
    p.add_argument("--model-id", default="anycalib_dist")
    p.add_argument("--cam-id", default="radial:2")
    p.add_argument("--weights", default=None, help="optional fine-tuned checkpoint (see export_anycalib.py)")
    p.add_argument("--non-strict", action="store_true")
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)

    from anycalib import AnyCalib
    from anycalib.cameras import CameraFactory

    dev = torch.device(a.device)
    model = AnyCalib(model_id=a.model_id)
    if a.weights:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from export_anycalib import load_weights

        load_weights(model, a.weights, strict=not a.non_strict)
    model = model.to(dev).eval()
    cam = CameraFactory.create_from_id(a.cam_id)

    paths = sorted(q for q in Path(a.images_dir).iterdir() if q.suffix.lower() in IMG_EXT)
    ref = {}
    for path in paths:
        image = np.array(Image.open(path).convert("RGB"))
        image = torch.tensor(image, dtype=torch.float32, device=dev).permute(2, 0, 1) / 255
        out = model.predict(image, cam_id=a.cam_id)
        intr = out["intrinsics"].double().cpu()
        names = list(cam.PARAMS_IDX)[: intr.numel()]

        ho, wo = image.shape[-2:]
        rec = {"width": int(wo), "height": int(ho), "cam_id": a.cam_id}
        rec |= {n: float(v) for n, v in zip(names, intr)}
        rec |= {
            "success": bool(out["success"]),
            "model_id": a.model_id + (f" + {Path(a.weights).name}" if a.weights else ""),
        }
        ref[path.name] = rec
        print(f"{path.name:42s} {wo}x{ho}  " + "  ".join(
            f"{n}={v:.4f}" if n.startswith("k") else f"{n}={v:.1f}" for n, v in zip(names, intr))
            + f"  success={rec['success']}", flush=True)

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(ref, indent=2) + "\n")
    print(f"wrote {a.output} ({len(ref)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
