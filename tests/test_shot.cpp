// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include "ShotAnalysis.h"

using namespace anycalib;

static int failures = 0;
static void Check(const char* name, bool ok) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++failures;
}

static FrameResult Frame(double number, size_t samples = 1000) {
  FrameResult f;
  f.frame = number;
  f.lens.W = 1920;
  f.lens.H = 1080;
  for (size_t i = 0; i < samples; ++i)
    f.samples.push_back({float(i), 0.5f, 0, 0, 1});
  return f;
}

int main() {
  {
    ShotAnalysis shot(4, 1000);
    Check("incomplete shot cannot produce a pooled fit", !shot.complete() && shot.pooledSamples().empty());
    bool added = true;
    for (int i = 3; i >= 0; --i) added = shot.addFrame(Frame(i)) && added;
    size_t retained = 0;
    for (const FrameResult& f : shot.frames()) retained += f.samples.size();
    Check("ray storage obeys the shot budget before pooling", added && retained == 1000);
    Check("complete shot is ordered and pooled", shot.complete() && shot.frames().front().frame == 0 &&
          shot.frames().back().frame == 3 && shot.pooledSamples().size() == 1000);
    shot.fail("cancelled");
    Check("failure prevents publishing even a complete set of frames",
          !shot.complete() && shot.pooledSamples().empty());
    Check("new analysis clears the previous failure", ShotAnalysis(1).error().empty());
  }
  for (bool pixelAspect : {false, true}) {
    ShotAnalysis shot(2);
    shot.addFrame(Frame(1));
    FrameResult second = Frame(2);
    if (pixelAspect) second.lens.pa = 2;
    else second.lens.W = 2048;
    Check(pixelAspect ? "pixel aspect changes fail the shot" : "format changes fail the shot",
          !shot.addFrame(second) && !shot.error().empty() && shot.pooledSamples().empty());
  }
  {
    ShotAnalysis shot(2);
    shot.addFrame(Frame(1));
    Check("duplicate frames (including stereo views) fail the shot", !shot.addFrame(Frame(1)));
    Check("empty frame is not silently skipped", !ShotAnalysis(1).addFrame(Frame(1, 0)));
    Check("invalid sample budget is rejected", !ShotAnalysis(100, 100).error().empty());
    Check("unplanned execution is rejected", !ShotAnalysis().error().empty());
    ShotAnalysis masked(2, 1000);
    masked.addFrame(Frame(1, 25));
    masked.addFrame(Frame(2));
    Check("masked frames do not duplicate evidence to fill the budget",
          masked.complete() && masked.pooledSamples().size() == 525);
  }
  {
    FrameResult a = Frame(1), b = Frame(2);
    a.lens.k1 = -0.03;
    b.lens = a.lens;
    Check("one lens and repeated identical lenses have zero spread",
          FrameSpread({a}) == 0 && FrameSpread({a, b}) == 0);
    const double radius2 = 1 + std::pow(1080.0 / 1920, 2);
    b.lens.k1 = -a.lens.k1 / (1 + 2 * a.lens.k1 * radius2);
    const double shift = CornerShiftPx(a.lens);
    Check("opposite distortions have equal corner magnitudes in this regression",
          std::fabs(shift - CornerShiftPx(b.lens)) < 1e-8);
    Check("map spread detects opposite distortion despite equal corner magnitudes",
          std::fabs(FrameSpread({a, b}) - std::sqrt(2.0) * shift) < 1e-8);
    b.lens = a.lens;
    a.lens.cx = 0.1;
    b.lens.cx = -0.1;
    Check("map spread detects centre disagreement",
          std::fabs(CornerShiftPx(a.lens) - CornerShiftPx(b.lens)) < 1e-8 &&
          FrameSpread({a, b}) > 1);
    a.lens.k1 = -2;
    Check("undefined maps cannot look stable", !std::isfinite(FrameSpread({a, b})));
  }
  return failures ? 1 : 0;
}
