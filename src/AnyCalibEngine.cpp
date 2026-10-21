// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
#include "AnyCalibEngine.h"

#include <algorithm>
#include <cmath>

#include <torch/cuda.h>
#include <torch/script.h>

namespace anycalib {

namespace {

// AnyCalib.predict's input: a centre crop at H/W clamped to [0.5, 2], sized to about
// `pixels` px in multiples of the 14 px patch.
struct Input {
  int w, h;           // network input
  int x0, y0, cw, ch;  // crop window in the plate
};

Input NetworkInput(int w, int h, double pixels) {
  const double ar = std::clamp(double(h) / w, 0.5, 2.0);
  const double tw = std::sqrt(pixels / ar);
  Input in{14 * int(std::lround(tw / 14)), 14 * int(std::lround(ar * tw / 14)), 0, 0, w, h};
  const double target = double(in.w) / in.h;
  if (double(w) / h > target) {
    const int crop = int(std::lround(w - h * target));
    in.x0 = crop / 2;
    in.cw = w - crop;
  } else {
    const int crop = int(std::lround(h - w / target));
    in.y0 = crop / 2;
    in.ch = h - crop;
  }
  return in;
}

// Non-square pixels: the crop is sized on the desqueezed plate.
Input NetworkInput(int w, int h, double pa, double pixels) {
  Input in = NetworkInput(std::max(1, int(std::lround(w * pa))), h, pixels);
  in.x0 = std::min(w - 1, int(std::lround(in.x0 / pa)));
  in.cw = std::max(1, std::min(w - in.x0, int(std::lround(in.cw / pa))));
  return in;
}

// One axis of PyTorch's antialiased bicubic resize, as AnyCalib.predict uses: another
// filter shifts the predicted distortion.
struct Taps {
  int first;
  std::vector<double> w;
};

double Cubic(double x) {
  return x < 1 ? (1.5 * x - 2.5) * x * x + 1
       : x < 2 ? ((-0.5 * x + 2.5) * x - 4) * x + 2
               : 0;
}

std::vector<Taps> ResizeTaps(int in, int out) {
  const double scale = double(in) / out;
  const double widen = std::max(scale, 1.0);
  std::vector<Taps> taps(out);
  for (int i = 0; i < out; ++i) {
    const double centre = scale * (i + 0.5);
    const int lo = std::max(int(centre - 2 * widen + 0.5), 0);
    const int hi = std::min(int(centre + 2 * widen + 0.5), in);
    double sum = 0;
    taps[i].first = lo;
    for (int j = lo; j < hi; ++j) {
      const double v = Cubic(std::fabs(j + 0.5 - centre) / widen);
      taps[i].w.push_back(v);
      sum += v;
    }
    for (double& v : taps[i].w) v = sum != 0 ? v / sum : 0;
  }
  return taps;
}

// Weighted sum of the RGB pixels from `p`, `step` floats apart.
void Blend(const Taps& t, const float* p, size_t step, double out[3]) {
  out[0] = out[1] = out[2] = 0;
  for (double w : t.w) {
    for (int c = 0; c < 3; ++c) out[c] += w * p[c];
    p += step;
  }
}

// The crop, resized to the network's planar [3, h, w] input and clamped as AnyCalib does.
std::vector<float> Resize(const float* rgb, int stride, const Input& in) {
  const std::vector<Taps> tx = ResizeTaps(in.cw, in.w), ty = ResizeTaps(in.ch, in.h);
  std::vector<float> rows(size_t(in.ch) * in.w * 3);
  for (int y = 0; y < in.ch; ++y)
    for (int x = 0; x < in.w; ++x) {
      double v[3];
      Blend(tx[x], rgb + (size_t(in.y0 + y) * stride + in.x0 + tx[x].first) * 3, 3, v);
      for (int c = 0; c < 3; ++c) rows[(size_t(y) * in.w + x) * 3 + c] = float(v[c]);
    }
  const size_t plane = size_t(in.w) * in.h;
  std::vector<float> out(3 * plane);
  for (int y = 0; y < in.h; ++y)
    for (int x = 0; x < in.w; ++x) {
      double v[3];
      Blend(ty[y], &rows[(size_t(ty[y].first) * in.w + x) * 3], size_t(in.w) * 3, v);
      for (int c = 0; c < 3; ++c)
        out[c * plane + size_t(y) * in.w + x] = float(std::clamp(v[c], 0.0, 1.0));
    }
  return out;
}

void AppendSamples(const float* r, const Input& in, std::vector<RaySample>& out) {
  const double sx = double(in.cw) / in.w, sy = double(in.ch) / in.h;
  for (int py = 0; py < in.h; ++py)
    for (int px = 0; px < in.w; ++px, r += 3)
      if (std::isfinite(r[0]) && std::isfinite(r[1]) && std::isfinite(r[2]))
        out.push_back({float(in.x0 + (px + 0.5) * sx), float(in.y0 + (py + 0.5) * sy), r[0], r[1],
                       r[2]});
}

std::string ErrorText(const std::exception& e) {
  if (auto* c = dynamic_cast<const c10::Error*>(&e)) return c->what_without_backtrace();
  return e.what();
}

// Set by export_anycalib.py --resolution; older exports are AnyCalib's 102400.
double InputPixels(const torch::jit::Module& m) {
  return m.hasattr("resolution") ? double(m.attr("resolution").toTensor().item<int64_t>()) : 102400;
}

class TorchEngine final : public AnyCalibEngine {
 public:
  TorchEngine(torch::jit::Module module, torch::Device device)
      : module_(std::move(module)), device_(device), pixels_(InputPixels(module_)) {}

  bool Infer(const float* rgb, int w, int h, double pa, std::vector<RaySample>& out,
             std::string* err) override {
    const Input in = NetworkInput(w, h, pa, pixels_);
    std::vector<float> image = Resize(rgb, w, in);
    torch::Tensor rays;
    if (!Forward(image, in, rays, err)) return false;
    AppendSamples(rays.data_ptr<float>(), in, out);
    return true;
  }

 private:
  bool Forward(std::vector<float>& image, const Input& in, torch::Tensor& rays,
               std::string* err) {
    try {
      c10::InferenceMode guard;
      torch::Tensor x = torch::from_blob(image.data(), {1, 3, in.h, in.w}).to(device_);
      rays = module_.forward({x}).toTensor().to(torch::kCPU, torch::kFloat).contiguous();
    } catch (const std::exception& e) {
      if (err) *err = "AnyCalib inference failed: " + ErrorText(e);
      return false;
    }
    if (rays.numel() != int64_t(3) * in.w * in.h) {
      if (err) *err = "unexpected AnyCalib output size (expected [1, H*W, 3])";
      return false;
    }
    return true;
  }

  torch::jit::Module module_;
  torch::Device device_;
  double pixels_;
};

std::unique_ptr<AnyCalibEngine> TryLoad(const std::string& path, torch::Device device,
                                        std::string& why) {
  if (device.is_cuda() && !torch::cuda::is_available()) {
    why = "CUDA is not available";
    return nullptr;
  }
  try {
    torch::jit::Module m = torch::jit::load(path, device);
    m.eval();
    return std::make_unique<TorchEngine>(std::move(m), device);
  } catch (const std::exception& e) {
    why = std::string(device.is_cuda() ? "CUDA" : "CPU") + " model load failed: " + ErrorText(e);
    return nullptr;
  }
}

}  // namespace

std::unique_ptr<AnyCalibEngine> LoadEngine(const std::string& modelPath, Device device,
                                           std::string* err) {
  std::string why;
  if (device == Device::Auto)
    if (auto e = TryLoad(modelPath, torch::Device(torch::kCUDA, 0), why)) return e;
  if (auto e = TryLoad(modelPath, torch::Device(torch::kCPU), why)) return e;
  if (err) *err = why;
  return nullptr;
}

}  // namespace anycalib
