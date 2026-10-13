// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
//
// Camera fits to AnyCalib's ray field. The OpenCV fit follows AnyCalib's `radial`
// calibrator; FitNukeLD fits Foundry's model to the same rays directly instead of
// converting.
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

bool FitOpenCVRadial(const std::vector<RaySample>& s, OpenCVRadial& out, FitStats* stats,
                     std::string* err);

// Nuke k1, k2, centre and a pinhole focal length (square pixels), initialised from an
// OpenCV fit. fixCentre keeps the centre at Nuke's (0, 0), the middle of the format.
bool FitNukeLD(const std::vector<RaySample>& s, int W, int H, double pa, const OpenCVRadial& init,
               NukeLD& out, double& focalPx, FitStats* stats, std::string* err,
               bool fixCentre = false);

// At most `maxN` evenly spaced samples.
std::vector<RaySample> Thin(const std::vector<RaySample>& s, size_t maxN);

// Largest distance (px) between a plate corner and its undistorted position.
double CornerShiftPx(const NukeLD& m);

}  // namespace anycalib
