// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
#include "ShotAnalysis.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace anycalib {

ShotAnalysis::ShotAnalysis(size_t expectedFrames, size_t sampleBudget)
    : expectedFrames_(expectedFrames),
      samplesPerFrame_(expectedFrames ? sampleBudget / expectedFrames : 0) {
  if (!expectedFrames || samplesPerFrame_ < 20)
    fail("invalid analysis frame count for the ray sample budget");
}

void ShotAnalysis::fail(const std::string& reason) {
  if (error_.empty()) error_ = reason;
}

bool ShotAnalysis::checkFormat(int W, int H, double pa) {
  if (!error_.empty()) return false;
  if (W < 32 || H < 32 || !(pa > 0) || !std::isfinite(pa)) {
    fail("invalid input format (minimum size is 32 x 32)");
  } else if (!frames_.empty()) {
    const NukeLD& first = frames_.front().lens;
    if (W != first.W || H != first.H || pa != first.pa)
      fail("input format or pixel aspect changed; analyze each format separately");
  }
  return error_.empty();
}

bool ShotAnalysis::addFrame(FrameResult frame) {
  if (!checkFormat(frame.lens.W, frame.lens.H, frame.lens.pa)) return false;
  if (!std::isfinite(frame.frame) || frames_.size() >= expectedFrames_) {
    fail("unexpected analysis frame");
  } else if (std::any_of(frames_.begin(), frames_.end(), [&](const FrameResult& f) {
               return f.frame == frame.frame;
             })) {
    fail("duplicate analysis frame; analyze one view at a time");
  } else if (frame.samples.size() < 20) {
    fail("too few unmasked rays in an analysis frame");
  } else if (!std::isfinite(frame.lens.k1) || !std::isfinite(frame.lens.k2) ||
             !std::isfinite(frame.lens.cx) || !std::isfinite(frame.lens.cy)) {
    fail("non-finite per-frame lens estimate");
  }
  if (!error_.empty()) return false;
  // Thinned now, so a long shot never holds more than the budget.
  frame.samples = Thin(frame.samples, samplesPerFrame_);
  frames_.push_back(std::move(frame));
  std::sort(frames_.begin(), frames_.end(),
            [](const FrameResult& a, const FrameResult& b) { return a.frame < b.frame; });
  return true;
}

bool ShotAnalysis::complete() const {
  return error_.empty() && expectedFrames_ > 0 && frames_.size() == expectedFrames_;
}

std::vector<RaySample> ShotAnalysis::pooledSamples() const {
  if (!complete()) return {};
  std::vector<RaySample> pooled;
  for (const FrameResult& f : frames_)
    pooled.insert(pooled.end(), f.samples.begin(), f.samples.end());
  return pooled;
}

double FrameSpread(const std::vector<FrameResult>& frames) {
  if (frames.empty()) return 0;
  const NukeLD& format = frames.front().lens;
  double worst = 0;
  for (int j = 0; j <= 8; ++j)
    for (int i = 0; i <= 8; ++i) {
      const double x = format.W * i / 8.0, y = format.H * j / 8.0;
      double mx = 0, my = 0, sum = 0;
      size_t n = 0;
      for (const FrameResult& f : frames) {
        double xu, yu;
        if (!f.lens.undistort(x, y, xu, yu) || !std::isfinite(xu) || !std::isfinite(yu))
          return std::numeric_limits<double>::infinity();
        ++n;
        const double dx = xu - mx, dy = yu - my;
        mx += dx / n;
        my += dy / n;
        sum += dx * (xu - mx) + dy * (yu - my);
      }
      if (n > 1) worst = std::max(worst, std::sqrt(std::max(0.0, sum / (n - 1))));
    }
  return worst;
}

std::vector<const Segment*> AllSegments(const std::vector<FrameResult>& frames) {
  std::vector<const Segment*> segs;
  for (const FrameResult& f : frames)
    for (const Segment& s : f.segs) segs.push_back(&s);
  return segs;
}

}  // namespace anycalib
