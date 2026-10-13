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
state_dict or `.safetensors`). The node looks for the network in `model override` (Advanced
tab), then `$ANYCALIB_MODEL_PATH`, then `models/anycalib_dist.pt` next to the plugin. The
file's SHA-256 goes into the metadata, so renders say which weights they came from.

The engine reproduces `AnyCalib.predict` (`radial:2`) to within 0.016° of ray angle on
the real test plates.

## Build

Needs Nuke 17, CMake 3.20+ and a C++17 compiler. The network runs on the libtorch Nuke ships
(PyTorch 2.7.1), so nothing is bundled. CMake downloads the libtorch 2.7.1 headers, or takes an
extracted release with `-DLIBTORCH_ROOT=...`.

```sh
cmake -S . -B build -DNUKE_ROOT=$HOME/nuke/Nuke17.0v3 -DCMAKE_INSTALL_PREFIX=$PWD/dist \
      -DANYCALIB_MODEL_FILE=$PWD/models/anycalib_dist.pt   # optional: bundle the network
cmake --build build -j
cmake --install build
```

Add `dist/anycalib` to `NUKE_PATH`. The node is under Transform > AnyCalib.

## Tests

```sh
ctest --test-dir build    # lens maths and fits on synthetic rays, no network
# end to end in Nuke: maps against LensDistortion2 and the legacy node, metadata, buttons, persistence
NUKE_PATH=dist/anycalib nuke --nukex -ti tests/nuke/test_anycalib.py test-assets /tmp/anycalib-test
# engine and fit against AnyCalib's Python pipeline (test-assets/reference.json)
.venv/bin/python tests/compare_reference.py build/anycalib_cli models/anycalib_dist.pt test-assets cuda
```

Test plates and their licences are listed in `test-assets/SOURCES.md`.
