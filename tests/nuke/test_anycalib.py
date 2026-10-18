"""End-to-end test of the AnyCalib Nuke plugin. Run headless:

    NUKE_PATH=dist/anycalib nuke --nukex -ti tests/nuke/test_anycalib.py <assets> <outdir>

<assets> must contain synthetic_barrel_k1-0.12_k2+0.02.png and the three
gopro_hero7_*.jpg plates (see test-assets/SOURCES.md). Prints PASS/FAIL per
check and exits non-zero on any failure.
"""
import json
import math
import os
import shutil
import sys

import nuke

ASSETS, OUT = sys.argv[1], sys.argv[2]
SYN = os.path.join(ASSETS, "synthetic_barrel_k1-0.12_k2+0.02.png")
os.makedirs(OUT, exist_ok=True)
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print("%s  %s  %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def read(path, first=1, last=1):
    r = nuke.nodes.Read(file=path, first=first, last=last, origfirst=first, origlast=last)
    r["on_error"].setValue("black")
    return r


def sample_uv(node, layer, x, y):
    return (nuke.sample(node, layer + ".u", x, y), nuke.sample(node, layer + ".v", x, y))


def make_grid(w, h):
    """13 x 9 pixel centres spanning the format."""
    return [(int(round(i * (w - 1) / 12.0)) + 0.5, int(round(j * (h - 1) / 8.0)) + 0.5)
            for j in range(9) for i in range(13)]


def bbox_diff(ours, theirs):
    """Largest difference between two nodes' bboxes, whether ours contains theirs, and both."""
    a, b = [(n.bbox().x(), n.bbox().y(), n.bbox().x() + n.bbox().w(), n.bbox().y() + n.bbox().h())
            for n in (ours, theirs)]
    contains = a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]
    return max(abs(p - q) for p, q in zip(a, b)), contains, "ours %s, LensDistortion2 %s" % (a, b)


nuke.load("AnyCalib")

# Multi-frame analysis over a 3-frame GoPro sequence (same camera)
seq_dir = os.path.join(OUT, "seq")
os.makedirs(seq_dir, exist_ok=True)
plates = sorted(f for f in os.listdir(ASSETS) if f.startswith("gopro_hero7_"))
for i, f in enumerate(plates):
    shutil.copy(os.path.join(ASSETS, f), os.path.join(seq_dir, "plate.%04d.jpg" % (i + 1)))
seq = read(os.path.join(seq_dir, "plate.####.jpg"), 1, len(plates))
ac = nuke.nodes.AnyCalib(inputs=[seq])
ac["analyze"].execute()

check("analysis ran over the range", int(ac["framesAnalysed"].value()) == len(plates),
      "framesAnalysed=%d framesInfo=%r" % (ac["framesAnalysed"].value(), ac["framesInfo"].value()))
check("checks recorded", ac["rmsDegrees"].value() > 0 and ac["frameSpread"].value() > 0,
      "fit RMS %.3f deg, frame spread %.2f px, %d lines, line correction %.2f px" % (
          ac["rmsDegrees"].value(), ac["frameSpread"].value(), ac["linesUsed"].value(),
          ac["lineCorrection"].value()))
check("estimator version + model digest recorded",
      ac["estimatorVersion"].value().startswith("anycalib-nuke/") and len(ac["modelSha256"].value()) == 64,
      ac["estimatorVersion"].value())
check("maps off by default: plate passes through, bbox unchanged",
      "forward.u" not in ac.channels() and ac.bbox().w() == seq.bbox().w() and ac.bbox().h() == seq.bbox().h())

def analysis_snapshot(node):
    return {name: node[name].toScript() for name in (
        "distortion1", "distortion2", "distortionCenter", "rmsDegrees", "frameSpread",
        "linesUsed", "lineCorrection", "framesAnalysed", "framesInfo", "modelFile",
        "modelSha256", "estimatorVersion", "linesData", "overscan")}


previous = analysis_snapshot(ac)
ac["analysisFrameCount"].setValue(2)
ac["analysisModelSha256"].setValue("f" * 64)
try:
    nuke.execute(ac, 1, 1, views=[nuke.views()[0]])
except RuntimeError:
    pass
check("incomplete analysis preserves result and provenance", analysis_snapshot(ac) == previous)

ac["analysisFrameCount"].setValue(1)
ac["analysisModelFile"].setValue(os.path.join(OUT, "nonexistent-model.pt"))
ac["analysisModelSha256"].setValue("e" * 64)
try:
    nuke.execute(ac, 1, 1, views=[nuke.views()[0]])
except RuntimeError:
    pass
check("failed model load preserves result and provenance", analysis_snapshot(ac) == previous)
ac["analyze"].execute()
check("analysis recovers after failure", ac["modelSha256"].value() == previous["modelSha256"])
check("line refinement is exported", ac.metadata("lens/lines_used") == int(ac["linesUsed"].value()) and
      abs(ac.metadata("lens/line_correction_px") - ac["lineCorrection"].value()) < 1e-9)

# STMaps against Foundry's LensDistortion nodes
syn = read(SYN)
an = nuke.nodes.AnyCalib(inputs=[syn], writeMaps=True)
an["analyze"].execute()
W, H = syn.format().width(), syn.format().height()
d1, d2 = an["distortion1"].value(), an["distortion2"].value()
cx, cy = an["distortionCenter"].value(0), an["distortionCenter"].value(1)
print("synthetic: d1=%.5f d2=%.5f centre=(%.5f, %.5f) lines=%d overscan=%s" % (
    d1, d2, cx, cy, an["linesUsed"].value(),
    [round(an["overscan"].value(i), 1) for i in range(4)]))
check("forward/backward channels present",
      all(c in an.channels() for c in ("forward.u", "forward.v", "backward.u", "backward.v")))


def ld2():
    """Foundry's LensDistortion (LensDistortion2), configured as CameraTracker does."""
    ld = nuke.nodes.LensDistortion2(inputs=[syn])
    ld["distortionModelPreset"].setValue("NukeX Classic")
    ld["output"].setValue("STMap")
    ld["distortionDenominator0"].setValue(d1)
    ld["distortionDenominator1"].setValue(d2)
    ld["centre"].setValue([cx, cy])
    return ld


def legacy(invert):
    ld = nuke.nodes.LensDistortion(inputs=[syn])
    ld["outputType"].setValue("Displacement")
    ld["distortion1"].setValue(d1)
    ld["distortion2"].setValue(d2)
    ld["distortionCenter"].setValue([cx, cy])
    ld["invertDistortion"].setValue(invert)
    return ld


def max_err(ours, layer, theirs, their_layer, points, w=None, h=None):
    w, h = w or W, h or H
    err, n = 0.0, 0
    for (x, y) in points:
        try:
            gu, gv = sample_uv(theirs, their_layer, x, y)
        except ValueError:  # outside their (cropped) bbox
            continue
        fu, fv = sample_uv(ours, layer, x, y)
        n += 1
        err = max(err, abs(fu - gu) * w, abs(fv - gv) * h)
    return err, n


grid = make_grid(W, H)
L2 = ld2()
e, n = max_err(an, "backward", L2, "backward", grid)
check("backward.u/v == LensDistortion2 redistort map", e < 0.01 and n > 100, "max %.5f px over %d" % (e, n))
e, n = max_err(an, "forward", L2, "forward", grid)
check("forward.u/v == LensDistortion2 undistort map (within its own approximation)", e < 0.15 and n > 100,
      "max %.5f px over %d" % (e, n))
e, _, info = bbox_diff(an, L2)
check("bbox == LensDistortion2 STMap Auto bbox", e <= 1, info)
e, n = max_err(an, "forward", legacy(True), "forward", grid)
check("forward.u/v == legacy LensDistortion, invertDistortion on", e < 0.01, "max %.5f px over %d" % (e, n))
e, n = max_err(an, "backward", legacy(False), "forward", grid)
check("backward.u/v == legacy LensDistortion, invertDistortion off", e < 0.01, "max %.5f px over %d" % (e, n))

ox, oy = -0.5 * an["overscan"].value(0), -0.5 * an["overscan"].value(1)
e, n = max_err(an, "forward", legacy(True), "forward", [(ox, oy)])
check("undistort map defined in overscan", n == 1 and e < 0.01, "at (%.1f, %.1f): %.5f px" % (ox, oy, e))

# Pincushion: LensDistortion2 writes (-1, -1) at some off-plate pixels, we only where
# there's no inverse.
an["distortion1"].setValue(0.1)
d1_saved, d1 = d1, 0.1
L2b = ld2()
agree_err, marked_ld2, marked_ours, n = 0.0, 0, 0, 0
for (x, y) in grid:
    ours, theirs = sample_uv(an, "forward", x, y), sample_uv(L2b, "forward", x, y)
    marked_ld2 += theirs == (-1.0, -1.0)
    marked_ours += ours == (-1.0, -1.0)
    if theirs != (-1.0, -1.0) and ours != (-1.0, -1.0):
        n += 1
        agree_err = max(agree_err, abs(ours[0] - theirs[0]) * W, abs(ours[1] - theirs[1]) * H)
check("pincushion: forward.u/v == LensDistortion2 wherever it has a value", agree_err < 0.15 and n > 50,
      "max %.4f px over %d; (-1,-1) marked by LensDistortion2 %d, by us %d (known difference)" % (
          agree_err, n, marked_ld2, marked_ours))
e, _, info = bbox_diff(an, L2b)
check("pincushion: bbox == LensDistortion2 STMap Auto bbox", e <= 1, info)
an["distortion1"].setValue(d1_saved)
d1 = d1_saved

# nuke.sample is nearest-pixel, so interpolate the backward map for the round trip.
def bilinear(node, layer, x, y):
    x0, y0 = math.floor(x - 0.5), math.floor(y - 0.5)
    tx, ty = x - 0.5 - x0, y - 0.5 - y0
    acc = [0.0, 0.0]
    for (dx, dy, w) in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)),
                        (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
        u, v = sample_uv(node, layer, x0 + dx + 0.5, y0 + dy + 0.5)
        acc[0] += w * u
        acc[1] += w * v
    return acc


rt = 0.0
for (x, y) in grid[::3]:
    fu, fv = sample_uv(an, "forward", x, y)
    bu, bv = bilinear(an, "backward", fu * W, fv * H)
    rt = max(rt, math.hypot(bu * W - x, bv * H - y))
check("forward -> backward round trip", rt < 0.02, "max %.4f px" % rt)

stm = nuke.nodes.STMap(inputs=[syn, an])
stm["uv"].setValue("forward")
ldi = ld2()
ldi["output"].setValue("Undistort")
diff, moved = 0.0, 0.0
for (x, y) in grid[1::4]:
    for c in ("rgba.red", "rgba.green"):
        a, b = nuke.sample(stm, c, x, y), nuke.sample(ldi, c, x, y)
        diff = max(diff, abs(a - b))
        moved = max(moved, abs(a - nuke.sample(syn, c, x, y)))
check("STMap(forward) ~ LensDistortion2 Undistort", diff < 0.05 and moved > 0.05,
      "max diff %.4f (vs untouched plate %.4f)" % (diff, moved))

idn = nuke.nodes.AnyCalib(inputs=[syn], writeMaps=True)
u, v = sample_uv(idn, "forward", 100.5, 200.5)
check("identity maps with zero distortion", abs(u * W - 100.5) < 1e-3 and abs(v * H - 200.5) < 1e-3,
      "(%.4f, %.4f)" % (u * W, v * H))

pW, pH = 2000, 1000
nuke.addFormat("%d %d 2 anycalib_pa2" % (pW, pH))
pa_src = nuke.nodes.Constant(format="anycalib_pa2")
pa_ac = nuke.nodes.AnyCalib(inputs=[pa_src], writeMaps=True)
pa_ld = nuke.nodes.LensDistortion2(inputs=[pa_src])
pa_ld["distortionModelPreset"].setValue("NukeX Classic")
pa_ld["output"].setValue("STMap")
for node, names in ((pa_ac, ("distortion1", "distortion2", "distortionCenter")),
                    (pa_ld, ("distortionDenominator0", "distortionDenominator1", "centre"))):
    node[names[0]].setValue(-0.1)
    node[names[1]].setValue(0.02)
    node[names[2]].setValue([0.05, -0.03])
pa_grid = make_grid(pW, pH)
e, n = max_err(pa_ac, "backward", pa_ld, "backward", pa_grid, pW, pH)
check("pixel aspect 2: backward.u/v == LensDistortion2", e < 0.01 and n > 100, "max %.5f px over %d" % (e, n))
e, n = max_err(pa_ac, "forward", pa_ld, "forward", pa_grid, pW, pH)
check("pixel aspect 2: forward.u/v == LensDistortion2 (within its own approximation)", e < 0.15 and n > 100,
      "max %.5f px over %d" % (e, n))
# LensDistortion2's bbox misses this lens's extremes mid-edge.
_, contains, info = bbox_diff(pa_ac, pa_ld)
check("pixel aspect 2: bbox covers LensDistortion2 STMap Auto bbox", contains, info)

md = an.metadata()
want = ["lens/k1", "lens/k2", "lens/centre", "lens/model", "lens/model_sha256", "lens/estimator_version"]
check("lens metadata on the stream", all(k in md for k in want),
      json.dumps({k: md.get(k) for k in want}, default=str)[:300])
exr = os.path.join(OUT, "meta.exr")
w = nuke.nodes.Write(inputs=[an], file=exr, file_type="exr")
w["metadata"].setValue("all metadata")
w["channels"].setValue("all")
nuke.execute(w, 1, 1)
rb = nuke.nodes.Read(file=exr)
rmd = rb.metadata()
lens_keys = sorted(k for k in rmd if "lens/" in k)
check("lens metadata in the EXR header", any(k.endswith("lens/k1") for k in lens_keys) and
      any(k.endswith("lens/estimator_version") for k in lens_keys), ", ".join(lens_keys))
check("STMap channels written to EXR",
      all(c in rb.channels() for c in ("forward.u", "forward.v", "backward.u", "backward.v")))

before = set(n.name() for n in nuke.allNodes("LensDistortion2"))
an["createUndistort"].execute()
an["createRedistort"].execute()
new = [n for n in nuke.allNodes("LensDistortion2") if n.name() not in before]
ok = len(new) == 2
for n in new:
    ok = ok and abs(n["distortionDenominator0"].value() - d1) < 1e-12
    ok = ok and abs(n["distortionDenominator1"].value() - d2) < 1e-12
    ok = ok and abs(n["centre"].value(0) - cx) < 1e-12 and abs(n["centre"].value(1) - cy) < 1e-12
    ok = ok and n["distortionModelPreset"].value() == "NukeX Classic"
outs = sorted(n["output"].value() for n in new)
und = [n for n in new if n["output"].value() == "Undistort"]
check("Create LensDistortion buttons", ok and outs == ["Redistort", "Undistort"] and und and und[0].input(0) is syn,
      "created %s" % [(n.name(), n["output"].value()) for n in new])
diff = 0.0
for (x, y) in grid[2::5]:
    diff = max(diff, abs(nuke.sample(und[0], "rgba.red", x, y) - nuke.sample(ldi, "rgba.red", x, y)))
check("created node == LensDistortion2 Undistort with our values", diff < 1e-6, "max diff %.2e" % diff)

nk = os.path.join(OUT, "persist.nk")
saved = dict(d1=d1, rms=an["rmsDegrees"].value(), name=an.name(),
             lines=len(an["linesData"].value()))
nuke.scriptSaveAs(nk, overwrite=1)
nuke.scriptClear()
nuke.scriptOpen(nk)
a2 = nuke.toNode(saved["name"])
check("results persist in the script",
      abs(a2["distortion1"].value() - saved["d1"]) < 1e-9 and abs(a2["rmsDegrees"].value() - saved["rms"]) < 1e-9
      and len(a2["linesData"].value()) == saved["lines"] and saved["lines"] > 100,
      "linesData %d bytes" % saved["lines"])

# The reloaded script reads the plate differently, so compare against a fresh fit.
syn = read(SYN)
free = nuke.nodes.AnyCalib(inputs=[syn], writeMaps=True)
free["analyze"].execute()
k_free = free["distortion1"].value()

lc = nuke.nodes.AnyCalib(inputs=[syn], lockCentre=True)
lc["analyze"].execute()
check("lock centre: centre stays at 0, same lens",
      lc["distortionCenter"].value(0) == 0 and lc["distortionCenter"].value(1) == 0
      and abs(lc["distortion1"].value() - k_free) < 2e-3,
      "%.6f vs %.6f" % (lc["distortion1"].value(), k_free))

nr = nuke.nodes.AnyCalib(inputs=[syn], refineLines=False, writeMaps=True)
nr["analyze"].execute()
moved, _ = max_err(free, "backward", nr, "backward", [(x, y) for x in (0.5, W - 0.5) for y in (0.5, H - 0.5)])
check("refine with lines off: no lines used, the correction is the difference",
      nr["linesUsed"].value() == 0 and nr["lineCorrection"].value() == 0 and free["linesUsed"].value() >= 2
      and abs(moved - free["lineCorrection"].value()) < 0.5 * free["lineCorrection"].value() + 0.1,
      "%d lines, correction %.2f px, corners move %.2f px" % (
          free["linesUsed"].value(), free["lineCorrection"].value(), moved))

bars = nuke.nodes.Rectangle()
bars["area"].setValue([0, 0, W, int(0.12 * H)])
bars = nuke.nodes.Rectangle(inputs=[bars])
bars["area"].setValue([0, int(0.88 * H), W, H])
ms = nuke.nodes.AnyCalib(inputs=[syn, bars], ignore="Mask Alpha", writeMaps=True)
ms["analyze"].execute()
ys = [float(v) for l in ms["linesData"].value().splitlines() if l.startswith("S") for v in l.split()[2::2]]
shift, _ = max_err(ms, "forward", free, "forward", [(x, y) for x in (0.5, W - 0.5) for y in (0.5, H - 0.5)])
check("mask: no lines in letterbox bars, lens within a few px",
      ys and min(ys) >= 0.12 * H and max(ys) <= 0.88 * H and shift < 5,
      "%d line points, y %.0f-%.0f of %d; corners move %.2f px" % (
          len(ys), min(ys) if ys else -1, max(ys) if ys else -1, H, shift))

dark = nuke.nodes.AnyCalib(inputs=[nuke.nodes.Multiply(inputs=[syn], value=0.25)])
dark["analyze"].execute()
check("exposure doesn't change the estimate", abs(dark["distortion1"].value() - k_free) < 1e-4,
      "%.6f vs %.6f" % (dark["distortion1"].value(), k_free))

nuke.root()["colorManagement"].setValue("OCIO")
nuke.root()["OCIO_config"].setValue("fn-nuke_cg-config-v2.2.0_aces-v1.3_ocio-v2.4")
aces = read(SYN)
aces["colorspace"].setValue("sRGB Encoded Rec.709 (sRGB)")
ac2 = nuke.nodes.AnyCalib(inputs=[aces])
ac2["analyze"].execute()
m = [ac2["inputMatrix"].value(i) for i in range(9)]
check("ACES project: ACEScg matrix, same estimate",
      abs(m[0] - 1.70505) < 1e-4 and abs(ac2["distortion1"].value() - d1) < 1e-3,
      "m00 %.5f, %.6f vs %.6f" % (m[0], ac2["distortion1"].value(), d1))

failed = [r for r in results if not r[1]]
print("\n%d/%d checks passed" % (len(results) - len(failed), len(results)))
sys.exit(1 if failed else 0)
