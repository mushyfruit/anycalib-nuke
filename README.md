# AnyCalib for Nuke

A Nuke node that estimates lens distortion from the plate, with no grid or camera move. It runs
a fine-tuned [AnyCalib](https://github.com/javrtg/AnyCalib), then refines the estimate so that
straight edges come out straight. [Write-up](https://www.andrewboyles.com/thoughts/anycalib-nuke).

![A barrel-distorted plate wiping to the node's undistortion; the edges it used turn from red to green](docs/images/wipe.gif)

## Install

Needs Nuke 17.0 or 17.1 on Linux or Windows. Unpack the plugin from the
[release](../../releases) and download the network into its `models` folder:

```sh
tar xzf anycalib-nuke-*-nuke17.1-linux-plugin.tar.gz   # or nuke17.0
curl -L -o anycalib/models/anycalib_dist.pt \
    https://huggingface.co/mushyfruit/anycalib-nuke/resolve/main/anycalib_dist.pt
```

On Windows, in PowerShell:

```powershell
Expand-Archive anycalib-nuke-*-nuke17.1-windows-plugin.zip .   # or nuke17.0
curl.exe -L -o anycalib\models\anycalib_dist.pt https://huggingface.co/mushyfruit/anycalib-nuke/resolve/main/anycalib_dist.pt
```

Then add the `anycalib` folder to Nuke's plugin path, for example in `~/.nuke/init.py` (forward
slashes on Windows too):

```python
nuke.pluginAddPath("/path/to/anycalib")
```

The node is under Transform > AnyCalib. To build it yourself, see
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Use

1. Connect the plate, in the working space.
2. Press Analyze. It fits one lens over the range, so analyse zooms and focus pulls in parts,
   and stereo views separately.
3. Check the overlay: edges that should be straight should be green.
4. Press Create LensDistortion, or turn on `write STMaps` and use an STMap: `uv = forward` to
   undistort, `uv = backward` to redistort CG rendered with the overscan in `overscan`.

`distortion1`, `distortion2` and `distortionCenter` are NukeX Classic values, as CameraTracker
gives them. A frame takes about 70 ms on an RTX 3090 and 1.4 s on the CPU.

## Limitations

- Spherical lenses only. Pixel aspect is handled; anamorphic squeeze isn't estimated.
- No fisheyes.
- On long lenses and good primes, assuming no distortion is often closer.

## Results

Error is how far points undistorted with the estimate land from the calibrated lens, in pixels
at 1280 wide. On 341 real cine plates from [InFlux](https://influx.cs.princeton.edu/)
(CC BY 4.0), shot at locations held out of training but on the same camera and zooms:

- fine-tuned: 5.36 px
- assuming no distortion: 8.34 px
- stock AnyCalib: 9.97 px

One of the better plates, each arrow drawn 4× its length:

![Undistortion error on one plate: stock AnyCalib 16.6 px mean, fine-tuned 1.0 px](docs/images/error-field.jpg)
