// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
//
// The engine, fit and line refinement outside Nuke, one JSON line per image:
//   anycalib_cli <model.pt> auto|cpu|cuda [--lock-centre] <img.ppm|pfm> [...]
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "AnyCalibEngine.h"
#include "CameraFit.h"
#include "LineDetect.h"

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

static void Print(const char* name, int W, int H, const NukeLD& nk, double focal,
                  const FitStats& st, const LineRefinement& lr, double ms) {
  std::printf(
      "{\"image\": \"%s\", \"width\": %d, \"height\": %d, "
      "\"nuke\": {\"distortion1\": %.6f, \"distortion2\": %.6f, \"centre\": [%.6f, %.6f], \"focal\": %.4f}, "
      "\"refined\": {\"distortion1\": %.6f, \"distortion2\": %.6f, \"lines\": %d}, "
      "\"rms_deg\": %.4f, \"corner_shift_px\": %.3f, \"ms\": %.1f}\n",
      name, W, H, nk.k1, nk.k2, nk.cx, nk.cy, focal, lr.lens.k1, lr.lens.k2, lr.lines, st.rmsDeg,
      CornerShiftPx(nk), ms);
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

  int first = 3;
  const bool lockCentre = argc > 3 && std::string(argv[3]) == "--lock-centre";
  if (lockCentre) ++first;
  for (int i = first; i < argc; ++i) {
    int W, H;
    std::vector<float> rgb;
    if (!ReadImage(argv[i], W, H, rgb)) { std::fprintf(stderr, "bad image %s\n", argv[i]); return 1; }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<RaySample> s;
    if (!eng->Infer(rgb.data(), W, H, 1.0, s, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    NukeLD nk;
    double focal;
    FitStats st;
    if (!FitNukeLD(s, W, H, 1.0, nk, focal, &st, &err, lockCentre)) {
      std::fprintf(stderr, "fit failed: %s\n", err.c_str());
      return 1;
    }
    // Lines on the luminance, as the node finds them.
    std::vector<float> gray(size_t(W) * H);
    for (size_t p = 0; p < gray.size(); ++p)
      gray[p] = 0.2126f * rgb[3 * p] + 0.7152f * rgb[3 * p + 1] + 0.0722f * rgb[3 * p + 2];
    const std::vector<Segment> segs = DetectSegments(gray.data(), W, H);
    std::vector<const Segment*> ptrs;
    for (const Segment& sg : segs) ptrs.push_back(&sg);
    Print(argv[i], W, H, nk, focal, st, RefineWithLines(ptrs, nk), ms);
  }
  return 0;
}
