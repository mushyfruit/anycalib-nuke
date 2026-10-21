# Development

Commands are run from the repo root.

## Network

The network is exported to TorchScript from AnyCalib's PyTorch weights, with torch 2.7.1 to
match Nuke:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r tools/requirements.txt
.venv/bin/python tools/export_anycalib.py --output models/anycalib_dist.pt \
    --verify-image test-assets/gopro_hero4_assembly_hall.jpg
```

For fine-tuned weights, add `--weights checkpoint.tar` (an AnyCalib training checkpoint, a
state_dict or `.safetensors`). The shipped network is a LoRA fine-tune merged into AnyCalib's
weights and exported with `--resolution 204800`, the input size it was trained at; the engine
reads that size from the file. The node looks for the network in `model override` (Advanced
tab), then `$ANYCALIB_MODEL_PATH`, then `models/anycalib_dist.pt` next to the plugin.

The engine and fit reproduce `AnyCalib.predict` with its `simple_division:2` camera (Nuke's
model, normalised by the focal length) to within 0.007° of ray angle on the test plates.

## Build

Needs Nuke 17, CMake 3.20+ and a C++17 compiler. The network runs on the libtorch Nuke ships
(PyTorch 2.7.1), so nothing is bundled. CMake downloads the libtorch 2.7.1 headers, or takes an
extracted release with `-DLIBTORCH_ROOT=...`. On Windows, `LIBTORCH_ROOT` must be the
2.7.1+cu128 Windows release: Nuke ships the torch DLLs but not their import libraries. The
plugin has to be built against an MSVC runtime no newer than the `msvcp140.dll` the Nuke
version ships (14.36 for 17.0, 14.44 for 17.1).

```sh
cmake -S . -B build -DNUKE_ROOT=$HOME/nuke/Nuke17.0v3 -DCMAKE_INSTALL_PREFIX=$PWD/dist \
      -DANYCALIB_MODEL_FILE=$PWD/models/anycalib_dist.pt   # optional: bundle the network
cmake --build build -j
cmake --install build
```

Add `dist/anycalib` to `NUKE_PATH`. The node is under Transform > AnyCalib.

## Tests

The lens and shot-analysis tests also build without Nuke, Torch, a GPU or downloads:

```sh
cmake -S . -B /tmp/anycalib-core -DANYCALIB_BUILD_NUKE=OFF
cmake --build /tmp/anycalib-core -j
ctest --test-dir /tmp/anycalib-core --output-on-failure
```

```sh
ctest --test-dir build    # lens maths, fits, line refinement, frame pooling
# end to end in Nuke: maps against LensDistortion2 and the legacy node, buttons, persistence
NUKE_PATH=dist/anycalib nuke --nukex -ti tests/nuke/test_anycalib.py test-assets /tmp/anycalib-test
# engine and fit against AnyCalib's Python pipeline (test-assets/reference.json); needs the
# stock network, exported without --weights
.venv/bin/python tests/compare_reference.py build/anycalib_cli models/anycalib_dist.pt test-assets
```

Test plates and their licences are listed in `test-assets/SOURCES.md`.

## Layout

- `src/AnyCalibEngine`: the network's input crop and resize, inference, rays back to plate pixels.
- `src/CameraFit`, `src/LensModel`: the fit and Nuke's lens model.
- `src/LineDetect`: edges for the overlay and the line refinement.
- `src/ShotAnalysis`: pools frames into one fit, keeping at most 100,000 rays. A failed or
  incomplete run keeps the previous result.
- `src/AnyCalibNuke.cpp`: the node. `python/anycalib_nuke.py`: its buttons.
