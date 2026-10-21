"""The AnyCalib node's buttons."""

import math
import os

import nuke

IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1]


def working_space():
    """Matrix from the project's working space to linear Rec.709 (row-major)."""
    root = nuke.root()
    if root["colorManagement"].value() != "OCIO":
        return IDENTITY
    import PyOpenColorIO as ocio

    config = ocio.Config.CreateFromFile(root["OCIOConfigPath"].value())
    ref = ocio.Config.CreateFromFile("ocio://cg-config-latest")
    try:  # through both configs' aces_interchange role
        cpu = ocio.Config.GetProcessorFromConfigs(
            config, root["workingSpaceLUT"].value(), ref, "lin_rec709"
        ).getDefaultCPUProcessor()
    except ocio.Exception:  # no interchange role: nuke-default, whose linear is Rec.709
        return IDENTITY
    cols = [cpu.applyRGB(v) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
    return [cols[c][r] for r in range(3) for c in range(3)]


def model_file(n):
    """The network: model override, then $ANYCALIB_MODEL_PATH, then models/ next to the plugin."""
    return (
        n["model"].evaluate()
        or os.environ.get("ANYCALIB_MODEL_PATH")
        or os.path.join(os.path.dirname(__file__), "models", "anycalib_dist.pt")
    )


def analyze(n):
    src = n.input(0)
    if src is None:
        nuke.message("AnyCalib: connect a plate to the input first.")
        return
    n["inputMatrix"].setValue(working_space())
    path = model_file(n)
    if not os.path.isfile(path):
        nuke.message(
            f"AnyCalib: no network at {path}. Set 'model override' or $ANYCALIB_MODEL_PATH, "
            "or install models/anycalib_dist.pt next to the plugin."
        )
        return
    n["analysisModelFile"].setValue(path)

    if n["useInputRange"].value():
        first, last = int(src.firstFrame()), int(src.lastFrame())
    else:
        first, last = int(n["analysisFirst"].value()), int(n["analysisLast"].value())
    if last < first:
        first, last = last, first
    step = max(1, math.ceil((last - first + 1) / max(1, int(n["maxFrames"].value()))))
    n["analysisFrameCount"].setValue(len(range(first, last + 1, step)))
    # Each view may have a different lens. Never pool stereo views into one estimate.
    viewer = nuke.activeViewer()
    view = viewer.view() if viewer is not None else nuke.views()[0]
    nuke.execute(n, first, last, step, views=[view])


def create_lens_distortion(n, undistort):
    """A LensDistortion2 with the node's values, as CameraTracker makes them."""
    for s in nuke.selectedNodes():
        s.setSelected(False)
    ld = nuke.nodes.LensDistortion2()
    ld["distortionModelPreset"].setValue("NukeX Classic")
    ld["projection"].setValue("None (Rectilinear)")
    ld["distortionDenominator0"].setValue(n["distortion1"].value())
    ld["distortionDenominator1"].setValue(n["distortion2"].value())
    ld["centre"].setValue([n["distortionCenter"].value(0), n["distortionCenter"].value(1)])
    ld["output"].setValue("Undistort" if undistort else "Redistort")
    ld["label"].setValue(f"{'undistort' if undistort else 'redistort'}\nfrom {n.name()}")
    ld.setXYpos(n.xpos() + (-110 if undistort else 110), n.ypos() + 60)
    if undistort and n.input(0) is not None:
        ld.setInput(0, n.input(0))
    ld.setSelected(True)
