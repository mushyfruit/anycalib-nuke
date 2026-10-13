# AnyCalib for Nuke

A Nuke 17 node (Linux) that estimates lens distortion from the plate with the
[AnyCalib](https://github.com/javrtg/AnyCalib) network, and produces what a comp needs:

- `distortion1`, `distortion2`, `distortionCenter` in Foundry's NukeX Classic model, the
  values CameraTracker produces and LensDistortion2 takes as `distortionDenominator0/1` and
  `centre`.
- STMaps laid out as LensDistortion2 writes them, with `write STMaps` on: `forward.u/v`
  undistorts, `backward.u/v` redistorts. The bbox then includes the overscan.
- `lens/*` metadata, which a Write (EXR, *all metadata*) puts in the header.
- Create LensDistortion buttons that build LensDistortion2 nodes, as CameraTracker does.
- A viewer overlay of detected lines: green if they're straight under the estimate, red if not.

## Install

Needs Nuke 17 on Linux. Build the plugin as described in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md),
which gives an `anycalib` folder holding the plugin and the network. Add that folder to Nuke's
plugin path, for example in `~/.nuke/init.py`:

```python
nuke.pluginAddPath('/path/to/anycalib')
```

The node is under Transform > AnyCalib.

## Use

1. Connect the plate. Analyze converts it from the project's working space to the sRGB, with
   normalised exposure, that AnyCalib was trained on.
2. Optionally, connect a matte to the `mask` input and pick its channel in `mask`, as with
   CameraTracker: white areas (moving objects, burn-ins, letterbox bars) are left out.
   `lock centre` keeps `distortionCenter` at 0.
3. Press Analyze. It samples up to `max frames` frames evenly over the input range (or a
   custom one) and pools them into one fit. The lens is assumed fixed over the range.
4. Press Create LensDistortion, or turn on `write STMaps` and undistort with an STMap
   (`uv = forward`) or redistort CG with `uv = backward`. CG needs the overscan shown in the
   `overscan` knob.
5. Check the overlay: lines that should be straight should be green.

CUDA is used when available, otherwise the CPU. On an RTX 3090 a 2048×1536 frame takes about
50 ms, after about 2 s to load the network; the first two frames take 0.5 s and 1.5 s while
TorchScript optimises the graph. On the CPU a frame takes about 0.7 s.

### Checks

- `fit RMS`: the RMS angle between the network's rays and the fitted lens. It's large when
  the lens model can't follow the network, as on a fisheye, but says nothing about whether
  the network is right.
- `frame spread`: the standard deviation, over the analysed frames, of each frame's largest
  corner shift in pixels. 0 for a single frame.
- `plumb-line scale`: a check that doesn't involve the network. The distortion is scaled by
  α ∈ [0, 2] to find where the detected lines are straightest. α near 1 agrees with the
  estimate. It's 1 when the lines don't react to α.

### Metadata

| Key | |
|---|---|
| `lens/model` | `nukex_classic` |
| `lens/k1`, `lens/k2` | `distortion1`, `distortion2` |
| `lens/centre`, `lens/centre_px` | `distortionCenter` (normalised, y up), and in pixels |
| `lens/fit_rms_deg`, `lens/frame_spread_px`, `lens/plumbline_alpha` | see above |
| `lens/focal_px` | focal length of the fit, in pixels |
| `lens/frames` | analysed range, e.g. `1001-1100 (24 frames)` |
| `lens/estimator`, `lens/estimator_version` | `AnyCalib`; plugin and libtorch versions, CUDA or CPU |
| `lens/model_sha256` | SHA-256 of the network file |
| `lens/overscan` | left, bottom, right, top, in pixels |
| `lens/source` | `AnyCalib`, or `manual` before any analysis |

The EXR writer prefixes them with `nuke/` unless *do not attach prefix* is on.

## Limitations

- Linux only.
- Spherical lenses only. Non-square pixels are handled through the format's pixel aspect, but
  anamorphic squeeze isn't estimated.
- AnyCalib's `radial:2` model breaks down past about 70° off-axis, so fisheyes are out of reach.
  On the GoPro HERO4 plate the plumb-line check flags the result.
- One lens per analysed range: zooms and focus breathing need separate ranges.
- LensDistortion2 writes (−1, −1) in its undistort map at some off-plate pixels; this node
  only does so where there's no inverse.
