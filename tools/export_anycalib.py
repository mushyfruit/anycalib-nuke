# Copyright the anycalib-nuke authors.
# SPDX-License-Identifier: Apache-2.0
"""Export the AnyCalib network to TorchScript (image -> rays) for the libtorch Nuke ships.

Run with torch 2.7.1, the version in Nuke 17. The camera fit happens in the plugin
(src/CameraFit.cpp), so only the network is exported. DINOv2's positional-embedding
interpolation is rewritten with shape-generic ops so the trace runs at any resolution
and still matches upstream.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import types
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _exact_interp(vit):
    """DINOv2's interpolate_pos_encoding (bicubic, A = -0.75, with its interpolate_offset
    and square short-circuit) as a resampling matrix per axis, so the trace stays dynamic in
    the input size. Tensors from new_ones() trace without a device, so it runs on CPU and CUDA.
    """
    M = math.isqrt(vit.pos_embed.shape[1] - 1)  # side of the pretrained patch grid
    off = float(vit.interpolate_offset)
    if vit.interpolate_antialias:
        raise SystemExit("antialiased pos-embed interpolation is not supported")
    A = -0.75

    def c1(x):
        return ((A + 2) * x - (A + 3)) * x * x + 1

    def c2(x):
        return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A

    def arange(ref, n):
        return ref.new_ones(n).cumsum(0) - 1

    def axis_weights(n, ref):
        i = arange(ref, n)
        n_f = torch.ones_like(i).sum()  # n as a tensor keeps the trace dynamic
        src = (i + 0.5) * (M / (n_f + off) if off else M / n_f) - 0.5
        f = torch.clamp(torch.floor(src), max=M - 1)
        t = torch.clamp(src - f, 0.0, 1.0)
        cols = arange(ref, M)
        W = 0.0
        for k, wk in enumerate((c2(t + 1), c1(t), c1(1 - t), c2(2 - t))):
            idx = torch.clamp(f + (k - 1), 0, M - 1)
            W = W + wk[:, None] * (idx[:, None] == cols[None, :]).float()
        eye = (i[:, None] == cols[None, :]).float()
        return W, eye, n_f

    def interpolate_pos_encoding(self, x, h, w):  # upstream names them (w, h) but passes (H, W)
        pe = self.pos_embed.float()
        Wr, Ir, nr = axis_weights(h // self.patch_size, pe)
        Wc, Ic, nc = axis_weights(w // self.patch_size, pe)
        square = (nr == M) & (nc == M) & (pe.new_ones(h).sum() == pe.new_ones(w).sum())
        Wr = torch.where(square, Ir, Wr)
        Wc = torch.where(square, Ic, Wc)
        grid = pe[0, 1:].reshape(M, M, -1).permute(2, 0, 1)  # (dim, M, M)
        grid = Wr @ grid @ Wc.T  # (dim, rows, cols)
        return torch.cat((pe[:, :1], grid.flatten(1).T[None]), dim=1).to(x.dtype)

    return interpolate_pos_encoding


class RayField(nn.Module):
    """AnyCalib's network without its calibrator: image (B, 3, H, W) -> unit rays (B, H*W, 3).
    `resolution`, the input pixel budget, is stored for the plugin to read."""

    def __init__(self, model, resolution):
        super().__init__()
        self.backbone, self.decoder, self.head = model.backbone, model.decoder, model.head
        self.register_buffer("resolution", torch.tensor(resolution))

    def forward(self, image):
        rays = self.head(self.decoder(self.backbone(image)))["rays"]  # (B, 3, H, W)
        return rays.flatten(2).transpose(1, 2)


def load_weights(m: nn.Module, path: str | os.PathLike, strict: bool = True) -> None:
    """Load fine-tuned weights into the stock AnyCalib model: a state_dict (.pt, .pth,
    .ckpt, .safetensors) or an AnyCalib training checkpoint (.tar)."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(path)
    else:
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            # AnyCalib's training checkpoints hold numpy scalars, which weights_only refuses.
            print("[weights] falling back to full unpickling: only use checkpoints you trust")
            sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict"):
        if isinstance(sd.get(key), Mapping):
            sd = sd[key]
            break
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(
        f"[weights] {path}: {len(sd)} tensors, {len(missing)} missing, {len(unexpected)} unexpected"
    )
    if strict and (missing or unexpected):
        raise SystemExit(
            f"[weights] keys don't match (--non-strict loads anyway): "
            f"missing {missing[:8]}, unexpected {unexpected[:8]}"
        )


def load_plugin_input(image_path, resolution):
    """The image as the plugin feeds it to the network: AnyCalib's own preprocessing."""
    from anycalib import AnyCalib
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    t = torch.from_numpy(np.array(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    h, w = t.shape[-2:]
    ar = max(AnyCalib.AR_RANGE[0], min(h / w, AnyCalib.AR_RANGE[1]))
    hp, wp = AnyCalib.compute_target_size(AnyCalib, resolution, ar)
    x, _, _ = AnyCalib.set_im_size(t, (hp, wp))
    return x.contiguous()


def ray_diff(a, b):
    """Max absolute difference and max angle (degrees) between two ray fields."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    an = a / np.linalg.norm(a, axis=-1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=-1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((an * bn).sum(-1), -1.0, 1.0)))
    return float(np.abs(a - b).max()), float(ang.max())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="models/anycalib_dist.pt", help="TorchScript file to write")
    ap.add_argument("--model-id", default="anycalib_dist", help="AnyCalib's pretrained model")
    ap.add_argument("--weights", help="fine-tuned checkpoint loaded over --model-id")
    ap.add_argument("--non-strict", action="store_true", help="allow missing or unexpected keys")
    ap.add_argument(
        "--resolution",
        type=int,
        default=102400,
        help="input pixels the weights were trained at (102400 for AnyCalib's own)",
    )
    ap.add_argument("--verify-image", help="compare the TorchScript and PyTorch rays on this image")
    args = ap.parse_args(argv)
    if not torch.__version__.startswith("2.7.1"):
        sys.exit(f"torch {torch.__version__}: export with 2.7.1, the libtorch in Nuke 17")

    os.environ["XFORMERS_DISABLED"] = "1"  # before anycalib loads: plain attention traces
    from anycalib import AnyCalib

    model = AnyCalib(model_id=args.model_id).eval()
    if args.weights:
        load_weights(model, args.weights, strict=not args.non_strict)
    net = RayField(model, args.resolution).eval()
    if args.verify_image:
        x = load_plugin_input(args.verify_image, args.resolution)
        with torch.inference_mode():
            upstream = net(x).numpy()

    vit = model.backbone.model
    vit.interpolate_pos_encoding = types.MethodType(_exact_interp(vit), vit)
    with warnings.catch_warnings(), torch.no_grad():
        # The tracer warns about the shape asserts it bakes in; they don't change the graph.
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(net, torch.rand(1, 3, 238, 420), check_trace=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.output)

    traced = torch.jit.load(args.output)  # as the plugin loads it
    assert int(traced.resolution) == args.resolution
    with torch.inference_mode():
        for h, w in ((238, 420), (322, 322)):  # 322 x 322 takes upstream's square short-circuit
            assert traced(torch.rand(1, 3, h, w)).shape == (1, h * w, 3)
    ok = True
    if args.verify_image:
        with torch.inference_mode():
            rays = traced(x).numpy()
            d_patched, ang_patched = ray_diff(rays, net(x).numpy())
        _, ang_upstream = ray_diff(rays, upstream)
        ok = d_patched <= 1e-3
        print(
            f"TorchScript vs patched model: max |d| {d_patched:.2e}, max {ang_patched:.4f} deg "
            f"-> {'PASS' if ok else 'FAIL'}"
        )
        print(f"TorchScript vs upstream AnyCalib: max {ang_upstream:.4f} deg")
    print(f"wrote {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
