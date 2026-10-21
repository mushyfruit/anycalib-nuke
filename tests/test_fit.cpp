// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "CameraFit.h"
#include "LensModel.h"
#include "LineDetect.h"

using namespace anycalib;

static int failures = 0;
static void Check(const char* what, bool ok, const std::string& detail) {
  std::printf("%s  %s  %s\n", ok ? "PASS" : "FAIL", what, detail.c_str());
  if (!ok) ++failures;
}
static std::string F(const char* fmt, double a, double b = 0, double c = 0) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), fmt, a, b, c);
  return buf;
}

// Rays for a 64x36 grid of plate pixels seen through `pixelToRay`.
template <class Fn>
static std::vector<RaySample> Synth(int W, int H, Fn pixelToRay) {
  std::vector<RaySample> s;
  for (int j = 0; j < 36; ++j)
    for (int i = 0; i < 64; ++i) {
      double u = (i + 0.5) * W / 64.0, v = (j + 0.5) * H / 36.0, r[3];
      pixelToRay(u, v, r);
      double n = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
      s.push_back({float(u), float(v), float(r[0] / n), float(r[1] / n), float(r[2] / n)});
    }
  return s;
}

int main() {
  for (double pa : {1.0, 2.0}) {
    NukeLD m;
    m.W = 1920;
    m.H = 1080;
    m.k1 = -0.12;
    m.k2 = 0.02;
    m.cx = 0.03;
    m.cy = -0.02;  // barrel
    m.pa = pa;
    double worst = 0;
    for (int j = 0; j <= 20; ++j)
      for (int i = 0; i <= 20; ++i) {
        double x = 1920.0 * i / 20, y = 1080.0 * j / 20, xu, yu, xd, yd;
        m.undistort(x, y, xu, yu);
        m.distort(xu, yu, xd, yd);
        worst = std::max(worst, std::hypot(xd - x, yd - y));
      }
    Check(pa == 1.0 ? "NukeLD distort(undistort(p)) == p"
                    : "NukeLD distort(undistort(p)) == p, pixel aspect 2",
          worst < 1e-6, F("max %.2e px", worst));
  }
  // LensDistortion2 backward.u/v with non-square pixels, rendered in Nuke 17.0v3.
  {
    struct Px {
      double x, y, u, v;
    };
    struct Fmt {
      int W, H;
      double pa, k1, k2, cx, cy;
      std::vector<Px> px;
    };
    const Fmt fmts[] = {{2000,
                         1000,
                         2.0,
                         -0.1,
                         0.02,
                         0.05,
                         -0.03,
                         {{60.5, 30.5, -29.0992, -10.3282},
                          {1940.5, 970.5, 2042.9745, 1026.5224},
                          {1100.5, 600.5, 1100.7996, 601.0177}}},
                        {1920,
                         1080,
                         0.9,
                         -0.12,
                         0.02,
                         0.02,
                         0.01,
                         {{57.5, 32.5, -48.7974, -26.9981},
                          {1862.5, 1047.5, 1956.1473, 1100.4150},
                          {1056.5, 648.5, 1056.6404, 648.6848}}}};
    double worst = 0;
    for (const Fmt& f : fmts) {
      NukeLD m;
      m.W = f.W;
      m.H = f.H;
      m.pa = f.pa;
      m.k1 = f.k1;
      m.k2 = f.k2;
      m.cx = f.cx;
      m.cy = f.cy;
      for (const Px& p : f.px) {
        double xu, yu;
        m.undistort(p.x, p.y, xu, yu);
        worst = std::max(worst, std::hypot(xu - p.u, yu - p.v));
      }
    }
    Check("NukeLD == LensDistortion2 with non-square pixels", worst < 1e-3,
          F("max %.2e px", worst));
  }
  // LensDistortion2's Auto bboxes (Undistort, Redistort), rendered in Nuke 17.0v3.
  // Its Redistort box comes from its approximate inverse, so allow 1 px.
  {
    struct Bx {
      int W, H;
      double pa, k1, k2, cx, cy;
      PixelBox u, r;
    };
    const Bx cases[] = {
        {1920, 1080, 1, -0.1, 0, 0, 0, {-149, -85, 2069, 1165}, {77, 13, 1843, 1067}},
        {1920, 1080, 1, 0.1, 0, 0, 0, {84, 13, 1836, 1067}, {-3, -80, 1923, 1160}},
        {1920, 1080, 1, -0.25, 0.02, 0.05, -0.03, {-480, -231, 2282, 1353}, {174, 28, 1784, 1041}},
        {1920, 1080, 1, -0.3, 0.03, 0.04, 0.02, {-579, -326, 2379, 1371}, {192, 43, 1761, 1045}},
        {1080, 1920, 1, 0.1, 0, 0.05, 0.08, {18, 105, 1071, 1854}, {-102, -3, 1158, 1923}},
        {2000, 1000, 2, 0.1, 0, 0.05, -0.03, {300, 7, 1735, 989}, {497, -69, 1503, 1082}}};
    int worst = 0;
    for (const Bx& c : cases) {
      NukeLD m;
      m.W = c.W;
      m.H = c.H;
      m.pa = c.pa;
      m.k1 = c.k1;
      m.k2 = c.k2;
      m.cx = c.cx;
      m.cy = c.cy;
      const PixelBox u = AutoBBox(UndistortedExtent(m)), r = AutoBBox(RedistortedExtent(m));
      for (auto [a, b] : {std::pair{u, c.u}, std::pair{r, c.r}})
        worst = std::max({worst, std::abs(a.x - b.x), std::abs(a.y - b.y), std::abs(a.r - b.r),
                          std::abs(a.t - b.t)});
    }
    Check("Auto bboxes == LensDistortion2", worst <= 1, F("max %.0f px", worst));
  }

  struct Case {
    int W, H;
    double pa, f, k1, k2, cx, cy;
  };
  for (Case c : {Case{1920, 1080, 1.0, 1500, -0.08, 0.01, 0.01, -0.02},     // barrel
                 Case{4096, 2160, 1.0, 1800, -0.25, 0.02, -0.03, 0.04},     // strong barrel, 4K
                 Case{1080, 1920, 1.0, 1300, 0.05, -0.01, 0.0, 0.0},        // pincushion, portrait
                 Case{1828, 1556, 2.0, 1600, -0.06, 0.01, 0.02, -0.01}}) {  // 2x anamorphic
    NukeLD gt;
    gt.W = c.W;
    gt.H = c.H;
    gt.pa = c.pa;
    gt.k1 = c.k1;
    gt.k2 = c.k2;
    gt.cx = c.cx;
    gt.cy = c.cy;
    const double S = gt.s();
    const double cxImg = gt.centreX(), cyImg = c.H - gt.centreY();
    auto s = Synth(c.W, c.H, [&](double u, double v, double* r) {
      double xu, yu;
      gt.undistort(u, c.H - v, xu, yu);  // Nuke is y-up
      r[0] = (xu - cxImg) * c.pa / c.f;
      r[1] = ((c.H - yu) - cyImg) / c.f;
      r[2] = 1;
    });
    NukeLD fit;
    double f = 0;
    std::string err;
    bool ok = FitNukeLD(s, c.W, c.H, c.pa, fit, f, nullptr, &err);
    const double shiftErr = std::fabs(CornerShiftPx(fit) - CornerShiftPx(gt));
    Check(("Nuke fit recovers LensDistortion " + std::to_string(c.W) + "x" + std::to_string(c.H) +
           (c.pa != 1.0 ? " pixel aspect " + std::to_string(int(c.pa)) : ""))
              .c_str(),
          ok && std::fabs(fit.k1 - gt.k1) < 1e-4 && std::fabs(fit.k2 - gt.k2) < 1e-4 &&
              std::fabs(fit.cx - gt.cx) < 1e-5 && std::fabs(fit.cy - gt.cy) < 1e-5 &&
              std::fabs(f - c.f) < 0.1 && shiftErr < 0.05,
          F("k1 %.6f (gt %.3f) f %.3f", fit.k1, gt.k1, f) + F(" centre %.6f %.6f", fit.cx, fit.cy) +
              (ok ? "" : " err=" + err) + F(" (s=%.0f)", S));
  }

  // Line refinement. Lines through the centre don't bend, so they change nothing.
  {
    NukeLD gt;
    gt.W = 1920;
    gt.H = 1080;
    gt.k1 = -0.05;
    gt.k2 = 0.005;
    gt.cx = 0.01;
    gt.cy = -0.02;
    auto line = [&](double x0, double y0, double x1, double y1) {  // undistorted, Nuke y up
      Segment s;
      for (int i = 0; i < 24; ++i) {
        double xd, yd;
        gt.distort(x0 + (x1 - x0) * i / 23, y0 + (y1 - y0) * i / 23, xd, yd);
        s.xy.push_back(float(xd));
        s.xy.push_back(float(gt.H - yd));
      }
      return s;
    };
    std::vector<Segment> grid, radial;
    for (double f : {0.1, 0.9}) grid.push_back(line(0.1 * gt.W, f * gt.H, 0.9 * gt.W, f * gt.H));
    for (double f : {0.08, 0.92}) grid.push_back(line(f * gt.W, 0.1 * gt.H, f * gt.W, 0.9 * gt.H));
    const double cx = gt.centreX(), cy = gt.centreY();
    for (double a : {0.3, 1.2, 2.0, 2.8})
      radial.push_back(line(cx + 300 * std::cos(a), cy + 300 * std::sin(a), cx + 520 * std::cos(a),
                            cy + 520 * std::sin(a)));
    auto ptrs = [](const std::vector<Segment>& v) {
      std::vector<const Segment*> p;
      for (const Segment& s : v) p.push_back(&s);
      return p;
    };
    NukeLD weak = gt;
    weak.k1 = -0.02;
    weak.k2 = 0;
    const LineRefinement r = RefineWithLines(ptrs(grid), weak);
    const double before = std::fabs(CornerShiftPx(weak) - CornerShiftPx(gt));
    const double after = std::fabs(CornerShiftPx(r.lens) - CornerShiftPx(gt));
    Check("lines correct an estimate that reads too little barrel",
          r.lines >= 2 && after < 0.2 * before,
          F("%.0f lines, corner shift error %.2f -> %.2f px", r.lines, before, after));
    NukeLD none = gt;
    none.k1 = none.k2 = 0;
    const LineRefinement rr = RefineWithLines(ptrs(radial), none);
    Check("radial lines leave the estimate alone",
          rr.lines == 0 && rr.lens.k1 == 0 && rr.lens.k2 == 0, F("%.0f lines", rr.lines));
  }

  std::printf("\n%s\n", failures ? "FAILED" : "all passed");
  return failures ? 1 : 0;
}
