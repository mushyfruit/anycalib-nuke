// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
//
// Runs the plugin's engine + fits on PPM or PFM images, outside Nuke.
//   anycalib_cli <model.pt> <device auto|cpu|cuda> <img.ppm|pfm> [...]
// Prints one JSON object per image. Used by tests/compare_reference.py.
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "AnyCalibEngine.h"
#include "CameraFit.h"

using namespace anycalib;

// 8-bit PPM, or little-endian float PFM (rows bottom to top, values as they are).
static bool ReadImage(const std::string& path, int& w, int& h, std::vector<float>& rgb) {
  std::ifstream f(path, std::ios::binary);
  std::string magic;
  double maxv;
  f >> magic;
  auto skip = [&] {
    while (f.peek() == '#' || std::isspace(f.peek())) {
      if (f.peek() == '#') { std::string l; std::getline(f, l); } else f.get();
    }
  };
  skip(); f >> w; skip(); f >> h; skip(); f >> maxv; f.get();
  if (w <= 0 || h <= 0) return false;
  rgb.resize(size_t(w) * h * 3);
  if (magic == "PF" && maxv < 0) {
    const size_t row = size_t(w) * 3;
    for (int y = h - 1; y >= 0; --y)
      f.read(reinterpret_cast<char*>(rgb.data() + y * row), row * sizeof(float));
    return bool(f);
  }
  if (magic != "P6" || maxv != 255) return false;
  std::vector<unsigned char> b(rgb.size());
  f.read(reinterpret_cast<char*>(b.data()), b.size());
  if (!f) return false;
  for (size_t i = 0; i < b.size(); ++i) rgb[i] = b[i] / 255.0f;
  return true;
}

static void Print(const char* name, int W, int H, const OpenCVRadial& cv, const NukeLD& nk,
                  double focal, const FitStats& st, double ms) {
  std::printf(
      "{\"image\": \"%s\", \"width\": %d, \"height\": %d, "
      "\"opencv\": {\"fx\": %.4f, \"fy\": %.4f, \"cx\": %.4f, \"cy\": %.4f, \"k1\": %.6f, \"k2\": %.6f}, "
      "\"nuke\": {\"distortion1\": %.6f, \"distortion2\": %.6f, \"centre\": [%.6f, %.6f], \"focal\": %.4f}, "
      "\"rms_deg\": %.4f, \"corner_shift_px\": %.3f, \"ms\": %.1f}\n",
      name, W, H, cv.fx, cv.fy, cv.cx, cv.cy, cv.k1, cv.k2, nk.k1, nk.k2, nk.cx, nk.cy, focal,
      st.rmsDeg, CornerShiftPx(nk), ms);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s model.pt auto|cpu|cuda img.ppm [...]\n", argv[0]);
    return 2;
  }
  const std::string dev = argv[2];
  std::string err;
  auto eng = LoadEngine(argv[1], dev == "cpu" ? Device::CPU : dev == "cuda" ? Device::CUDA : Device::Auto,
                        &err);
  if (!eng) { std::fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
  std::fprintf(stderr, "device: %s\n", eng->device().c_str());

  for (int i = 3; i < argc; ++i) {
    int W, H;
    std::vector<float> rgb;
    if (!ReadImage(argv[i], W, H, rgb)) { std::fprintf(stderr, "bad image %s\n", argv[i]); return 1; }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<RaySample> s;
    if (!eng->Infer(rgb.data(), W, H, 1.0, s, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    OpenCVRadial cv;
    NukeLD nk;
    double focal;
    FitStats st;
    if (!FitOpenCVRadial(s, cv, nullptr, &err) || !FitNukeLD(s, W, H, 1.0, cv, nk, focal, &st, &err)) {
      std::fprintf(stderr, "fit failed: %s\n", err.c_str());
      return 1;
    }
    Print(argv[i], W, H, cv, nk, focal, st, ms);
  }
  return 0;
}
