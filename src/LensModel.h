// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
//
// Foundry's "NukeX Classic" model. Pixel coordinates are continuous, with pixel centres at
// +0.5.
#pragma once

#include <cmath>

namespace anycalib {

// n = ((p - c) * (pa, 1)) / s, s = max(W, H) / 2, Nuke pixels (y up).
// Undistorted n_u = n_d / (1 + k1 r_d^2 + k2 r_d^4). Negative k1 is barrel.
struct NukeLD {
  double k1 = 0, k2 = 0;  // distortion1/2 = LensDistortion2 distortionDenominator0/1
  double cx = 0, cy = 0;  // distortionCenter = LensDistortion2 centre, in units of s
  int W = 0, H = 0;
  double pa = 1;          // pixel aspect

  double s() const { return 0.5 * (W > H ? W : H); }
  double centreX() const { return 0.5 * W + cx * s() / pa; }
  double centreY() const { return 0.5 * H + cy * s(); }
  bool isIdentity() const { return k1 == 0 && k2 == 0; }

  // False where the denominator isn't positive.
  bool undistort(double xd, double yd, double& xu, double& yu) const;
  // False beyond the invertible range, where the result is clamped.
  bool distort(double xu, double yu, double& xd, double& yd) const;
  bool undistortRadius(double rd, double& ru) const;
  bool distortRadius(double ru, double& rd) const;
};

struct Extent { double x = 0, y = 0, r = 0, t = 0; };

// Sampled along the whole boundary: LensDistortion2's corners and midpoints miss extremes.
Extent UndistortedExtent(const NukeLD& m, int samplesPerEdge = 257);

// Bbox of the plate pixels whose undistorted position lies in the format, limited to
// |n_x|, |n_y| <= 1 as LensDistortion2 limits it.
Extent RedistortedExtent(const NukeLD& m, int samplesPerEdge = 257);

// LensDistortion2's Auto bbox: rounded outwards, plus 3 px.
struct PixelBox { int x, y, r, t; };
inline PixelBox AutoBBox(const Extent& e) {
  return {int(std::floor(e.x)) - 3, int(std::floor(e.y)) - 3, int(std::ceil(e.r)) + 3,
          int(std::ceil(e.t)) + 3};
}

}  // namespace anycalib
