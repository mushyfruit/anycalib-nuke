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

enum class Device { Auto, CPU, CUDA };

class AnyCalibEngine {
 public:
  virtual ~AnyCalibEngine() = default;

  virtual const std::string& device() const = 0;  // "CUDA" or "CPU"

  // rgb: interleaved, display-referred sRGB in [0,1], w x h, top row first. Appends a
  // sample per network pixel (about 102400), in plate pixels. Non-square pixels are
  // desqueezed for the network.
  virtual bool Infer(const float* rgb, int w, int h, double pixelAspect,
                     std::vector<RaySample>& out, std::string* err) = 0;
};

// Auto tries CUDA, then the CPU.
std::unique_ptr<AnyCalibEngine> LoadEngine(const std::string& modelPath, Device device,
                                           std::string* err);

}  // namespace anycalib
