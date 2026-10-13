// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
//
// Long, smooth edges in the plate: candidate images of straight lines. Used for the
// viewer overlay and a plumb-line check of the estimate that doesn't involve the network.
#pragma once

#include <cstdint>
#include <vector>

#include "LensModel.h"

namespace anycalib {

struct Segment {
  std::vector<float> xy;  // polyline, plate pixels, y down
};

// gray: top row first. Nonzero `exclude` pixels are ignored. Longest first, 24 points each.
std::vector<Segment> DetectSegments(const float* gray, int w, int h, int maxSegments = 48,
                                    const uint8_t* exclude = nullptr);

// The scale alpha in [0, 2] of the estimated distortion at which the segments are
// straightest. ~1 corroborates the estimate, ~0 means they're straight in the plate.
// Not `informative` when the lines barely react to alpha.
struct PlumbLine {
  double alpha = 1;
  bool informative = false;
};
PlumbLine PlumbLineScale(const std::vector<const Segment*>& segs, const NukeLD& m);

// One segment as the overlay draws it, in Nuke coordinates (y up).
struct OverlayLine {
  std::vector<double> plate;      // x y x y ...
  std::vector<double> predicted;  // the best-fit straight line, distorted by the estimate
  bool straight = false;          // straight under the estimate
};
OverlayLine BuildOverlayLine(const Segment& s, const NukeLD& m);

}  // namespace anycalib
