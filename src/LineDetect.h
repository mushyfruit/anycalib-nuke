// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
//
// Long, smooth edges in the plate: candidate images of straight lines.
#pragma once

#include <cstdint>
#include <vector>

#include "LensModel.h"

namespace anycalib {

struct Segment {
  std::vector<float> xy;  // polyline, plate pixels, y down
};

// gray: top row first. Longest first, 24 points each.
std::vector<Segment> DetectSegments(const float* gray, int w, int h, int maxSegments = 48);

// m's k1 and k2, adjusted so the segments come out straight. Needs two that distortion bends.
struct LineRefinement {
  NukeLD lens;
  int lines = 0;  // segments that constrain the distortion
};
LineRefinement RefineWithLines(const std::vector<const Segment*>& segs, const NukeLD& m);

// One segment as the overlay draws it, in Nuke coordinates (y up).
struct OverlayLine {
  std::vector<double> plate;      // x y x y ...
  std::vector<double> predicted;  // the best-fit straight line, distorted by the estimate
  bool straight = false;          // straight under the estimate
};
OverlayLine BuildOverlayLine(const Segment& s, const NukeLD& m);

}  // namespace anycalib
