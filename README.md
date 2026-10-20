# AnyCalib for Nuke

A Nuke 17 node (Linux) that estimates lens distortion from the plate. It runs the
[AnyCalib](https://github.com/javrtg/AnyCalib) network, fine-tuned on renders through Nuke's
lens model and on real footage, refines the result with straight edges in the plate, and
produces what a comp needs:

- `distortion1`, `distortion2`, `distortionCenter` in Foundry's NukeX Classic model, the
  values CameraTracker produces and LensDistortion2 takes as `distortionDenominator0/1` and
  `centre`.
- STMaps laid out as LensDistortion2 writes them, with `write STMaps` on: `forward.u/v`
  undistorts, `backward.u/v` redistorts. The bbox then includes the overscan.
- `lens/*` metadata, which a Write (EXR, *all metadata*) puts in the header.
- Create LensDistortion buttons that build LensDistortion2 nodes, as CameraTracker does.
- A viewer overlay of detected lines: green if they're straight under the estimate, red if not.

## Install

Needs Nuke 17 on Linux. Unpack the [release](../../releases) and add its `anycalib` folder,
which holds the plugin and the network, to Nuke's plugin path, for example in `~/.nuke/init.py`:

```python
nuke.pluginAddPath('/path/to/anycalib')
```

The node is under Transform > AnyCalib. To build it yourself, see
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Use

1. Connect the plate. Analyze converts it from the project's working space to the sRGB, with
   normalised exposure, that AnyCalib was trained on.
2. Optionally, connect a matte to the `mask` input and pick its channel in `mask`, as with
   CameraTracker: white areas (moving objects, burn-ins, letterbox bars) are left out.
   `lock centre` keeps `distortionCenter` at 0.
3. Press Analyze. It samples up to `max frames` frames evenly over the input range (or a
   custom one) and pools them into one fit, which `refine with lines` then adjusts so that
   straight edges in the plate come out straight. The lens is assumed fixed over the range.
   It uses the active viewer's view, or the first project view headlessly. Analyze stereo
   views separately. Format changes, failures and incomplete runs keep the previous result.
4. Press Create LensDistortion, or turn on `write STMaps` and undistort with an STMap
   (`uv = forward`) or redistort CG with `uv = backward`. CG needs the overscan shown in the
   `overscan` knob.
5. Check the overlay: lines that should be straight should be green.

CUDA is used when available, otherwise the CPU. On an RTX 3090 a frame takes about 70 ms,
after about 2 s to load the network; the first two frames take longer while TorchScript
optimises the graph. On the CPU a frame takes about 1.4 s.

### Checks

On the Advanced tab, rounded to 0.1 px:

- `frame variation`: how much the analysed frames disagree about the lens, in pixels on the
  undistorted plate, where they disagree most. 0 for a single frame. A large value means the
  lens changed over the range (zoom, focus), so analyse those parts separately.
- `lines used`: straight edges that refined the network's estimate. Only long edges away from
  the centre count, where distortion bends them; lines through the centre don't bend. Below 2
  the network's estimate is kept as it is.
- `line correction`: how far the lines moved the estimate, in pixels at the plate's corners. A
  large correction on a plate full of curved architecture deserves a look; turn off
  `refine with lines` to compare.

### Metadata

| Key | |
|---|---|
| `lens/model` | `nukex_classic` |
| `lens/k1`, `lens/k2` | `distortion1`, `distortion2` |
| `lens/centre`, `lens/centre_px` | `distortionCenter` (normalised, y up), and in pixels |
| `lens/frame_spread_px` | frame variation |
| `lens/lines_used`, `lens/line_correction_px` | see Checks |
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
- Fisheyes are out of reach. The fit follows them, but undistorting one to a pinhole image
  moves its corners thousands of pixels.
- One lens per analysed range: zooms and focus breathing need separate ranges.
- LensDistortion2 writes (−1, −1) in its undistort map at some off-plate pixels; this node
  only does so where there's no inverse.
