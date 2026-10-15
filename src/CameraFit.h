// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
//
// Fits Foundry's lens model to AnyCalib's ray field.
#pragma once

#include <string>
#include <vector>

#include "LensModel.h"

namespace anycalib {

// A plate pixel (y down) and its unit ray (x right, y down, z forward).
struct RaySample {
  float u, v;
  float rx, ry, rz;
};

struct FitStats {
  double rmsDeg = 0;  // RMS angular residual
};

// AnyCalib's linear fit of its division camera (Nuke's model up to normalisation), refined
// on ray angles. focalPx is in square pixels; fixCentre keeps the centre at (0, 0).
bool FitNukeLD(const std::vector<RaySample>& s, int W, int H, double pa, NukeLD& out,
               double& focalPx, FitStats* stats, std::string* err, bool fixCentre = false);

// At most `maxN` evenly spaced samples.
std::vector<RaySample> Thin(const std::vector<RaySample>& s, size_t maxN);

// Largest distance (px) between a plate corner and its undistorted position.
double CornerShiftPx(const NukeLD& m);

}  // namespace anycalib
