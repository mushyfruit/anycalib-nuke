// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>

#include "CameraFit.h"
#include "LineDetect.h"

namespace anycalib {

struct FrameResult {
  double frame = 0;
  NukeLD lens;  // this frame's own fit, for FrameSpread
  std::vector<Segment> segs;
  std::vector<RaySample> samples;
};

// The analysed frames of one lens and format, holding at most sampleBudget rays. A failed or
// incomplete shot gives no samples.
class ShotAnalysis {
 public:
  explicit ShotAnalysis(size_t expectedFrames = 0, size_t sampleBudget = 100000);
  bool checkFormat(int W, int H, double pa);
  bool addFrame(FrameResult frame);
  bool complete() const;
  void fail(const std::string& reason);
  const std::string& error() const { return error_; }
  const std::vector<FrameResult>& frames() const { return frames_; }
  std::vector<RaySample> pooledSamples() const;

 private:
  size_t expectedFrames_, samplesPerFrame_;
  std::vector<FrameResult> frames_;
  std::string error_;
};

// Largest standard deviation of undistorted positions over a 9 x 9 grid; infinite where a
// frame's map is undefined.
double FrameSpread(const std::vector<FrameResult>& frames);
std::vector<const Segment*> AllSegments(const std::vector<FrameResult>& frames);

}  // namespace anycalib
