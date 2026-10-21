// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
//
// The exported AnyCalib network (image [1,3,H,W] -> unit rays [1,H*W,3]) on Nuke's libtorch.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "CameraFit.h"

namespace anycalib {

enum class Device { Auto, CPU };

class AnyCalibEngine {
 public:
  virtual ~AnyCalibEngine() = default;

  // rgb: interleaved sRGB in [0,1], top row first. Appends a sample per network pixel.
  virtual bool Infer(const float* rgb, int w, int h, double pixelAspect,
                     std::vector<RaySample>& out, std::string* err) = 0;
};

// Auto tries CUDA, then the CPU.
std::unique_ptr<AnyCalibEngine> LoadEngine(const std::string& modelPath, Device device,
                                           std::string* err);

}  // namespace anycalib
