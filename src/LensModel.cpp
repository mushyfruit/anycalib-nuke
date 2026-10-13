// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
#include "LensModel.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace anycalib {

bool NukeLD::undistortRadius(double rd, double& ru) const {
  const double r2 = rd * rd, D = 1 + k1 * r2 + k2 * r2 * r2;
  if (!(D > 0)) { ru = rd; return false; }
  ru = rd / D;
  return true;
}

bool NukeLD::distortRadius(double ru, double& rd) const {
  if (ru <= 0) { rd = 0; return true; }
  if (isIdentity()) { rd = ru; return true; }
  // g(r) = r / (1 + k1 r^2 + k2 r^4) maps distorted to undistorted radius;
  // invert it on the branch connected to r = 0.
  auto D = [&](double r) { double r2 = r * r; return 1 + k1 * r2 + k2 * r2 * r2; };
  auto g = [&](double r) { return r / D(r); };
  auto dg = [&](double r) {
    double r2 = r * r, d = D(r), dD = 2 * k1 * r + 4 * k2 * r2 * r;
    return (d - r * dD) / (d * d);
  };

  // Newton from the identity guess.
  double r = ru;
  for (int it = 0; it < 60; ++it) {
    if (D(r) <= 1e-9 || dg(r) <= 1e-12) break;
    double step = (g(r) - ru) / dg(r);
    double rn = r - step;
    if (rn <= 0) rn = 0.5 * r;
    r = rn;
    if (std::fabs(step) < 1e-13 * std::max(1.0, r)) {
      if (dg(r) > 0 && D(r) > 0) { rd = r; return true; }
      break;
    }
  }

  // Fallback: bracket the increasing branch [0, rmax] and bisect.
  double rmax = 0, h = 1e-3;
  while (rmax < 50 && D(rmax + h) > 1e-9 && dg(rmax + h) > 0) rmax += h;
  if (g(rmax) < ru) { rd = rmax; return false; }  // beyond the invertible range
  double lo = 0, hi = rmax;
  for (int it = 0; it < 200; ++it) {
    double mid = 0.5 * (lo + hi);
    (g(mid) < ru ? lo : hi) = mid;
  }
  rd = 0.5 * (lo + hi);
  return true;
}

bool NukeLD::undistort(double xd, double yd, double& xu, double& yu) const {
  const double S = s(), cX = centreX(), cY = centreY();
  const double nx = (xd - cX) * pa / S, ny = (yd - cY) / S;
  const double r2 = nx * nx + ny * ny, D = 1 + k1 * r2 + k2 * r2 * r2;
  const bool ok = D > 0;
  const double inv = ok ? 1.0 / D : 1.0;
  xu = nx * inv * S / pa + cX;
  yu = ny * inv * S + cY;
  return ok;
}

bool NukeLD::distort(double xu, double yu, double& xd, double& yd) const {
  const double S = s(), cX = centreX(), cY = centreY();
  const double nx = (xu - cX) * pa / S, ny = (yu - cY) / S;
  const double ru = std::sqrt(nx * nx + ny * ny);
  double rd = ru;
  const bool ok = distortRadius(ru, rd);
  const double sc = ru > 1e-15 ? rd / ru : 1.0;
  xd = nx * sc * S / pa + cX;
  yd = ny * sc * S + cY;
  return ok;
}

void OpenCVRadial::distort(double xu, double yu, double& xd, double& yd) const {
  const double x = (xu - cx) / fx, y = (yu - cy) / fy;
  const double r2 = x * x + y * y;
  const double g = 1 + k1 * r2 + k2 * r2 * r2;
  xd = x * g * fx + cx;
  yd = y * g * fy + cy;
}

bool OpenCVRadial::undistort(double xd, double yd, double& xu, double& yu) const {
  const double x0 = (xd - cx) / fx, y0 = (yd - cy) / fy;
  // As AnyCalib (cameras/radial.py newton_undistort_radii): 25 plain Newton
  // steps from r_u = r_d, converged when |f| <= 1e-5 at the last step. The
  // fit's handling of non-converged samples must match AnyCalib's.
  const double rd = std::sqrt(x0 * x0 + y0 * y0);
  double ru = rd, f = 0;
  for (int it = 0; it < 25 && rd > 0; ++it) {
    const double r2 = ru * ru;
    f = ru * (1 + k1 * r2 + k2 * r2 * r2) - rd;
    const double df = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2;
    if (df == 0 || !std::isfinite(df)) { f = 1; break; }
    ru -= f / df;
  }
  const double sc = rd > 1e-15 ? ru / rd : 1.0;
  xu = x0 * sc * fx + cx;
  yu = y0 * sc * fy + cy;
  return std::fabs(f) <= 1e-5 && std::isfinite(ru);
}

Extent UndistortedExtent(const NukeLD& m, int n) {
  Extent e{double(m.W), double(m.H), 0, 0};
  n = std::max(n, 2);
  auto add = [&](double x, double y) {
    double xu, yu;
    if (!m.undistort(x, y, xu, yu)) return;
    e.x = std::min(e.x, xu); e.r = std::max(e.r, xu);
    e.y = std::min(e.y, yu); e.t = std::max(e.t, yu);
  };
  for (int i = 0; i < n; ++i) {
    double t = double(i) / (n - 1);
    add(t * m.W, 0); add(t * m.W, m.H);
    add(0, t * m.H); add(m.W, t * m.H);
  }
  if (e.x > e.r) return Extent{0, 0, double(m.W), double(m.H)};  // no valid samples
  return e;
}

Extent RedistortedExtent(const NukeLD& m, int n) {
  // The distorted format clipped to the square: its bbox is on the format boundary, where
  // that crosses the square, or at the square's corners.
  const double hx = m.s() / m.pa, hy = m.s();
  const double x0 = 0.5 * m.W - hx, x1 = 0.5 * m.W + hx, y0 = 0.5 * m.H - hy, y1 = 0.5 * m.H + hy;
  auto inSquare = [&](double x, double y) { return x >= x0 && x <= x1 && y >= y0 && y <= y1; };
  // The format boundary at t in [0, 4] (bottom, right, top, left), distorted.
  auto boundary = [&](double t, double& x, double& y) {
    const int k = std::min(3, int(t));
    const double f = t - k;
    const double px = k == 0 ? f * m.W : k == 1 ? m.W : k == 2 ? (1 - f) * m.W : 0;
    const double py = k == 0 ? 0 : k == 1 ? f * m.H : k == 2 ? m.H : (1 - f) * m.H;
    return m.distort(px, py, x, y) && inSquare(x, y);
  };
  Extent e{x1, y1, x0, y0};
  auto add = [&](double x, double y) {
    e.x = std::min(e.x, x); e.r = std::max(e.r, x);
    e.y = std::min(e.y, y); e.t = std::max(e.t, y);
  };
  const int N = 4 * std::max(n, 2);
  double x, y;
  bool prev = boundary(0, x, y);
  for (int i = 0; i <= N; ++i) {
    const double t = 4.0 * i / N;
    const bool in = boundary(t, x, y);
    if (in) add(x, y);
    if (i > 0 && in != prev) {
      double a = 4.0 * (i - 1) / N, b = t;  // boundary(a) == prev, boundary(b) == in
      for (int it = 0; it < 50; ++it) {
        const double mid = 0.5 * (a + b);
        (boundary(mid, x, y) == prev ? a : b) = mid;
      }
      if (boundary(prev ? a : b, x, y)) add(x, y);
    }
    prev = in;
  }
  for (double cx : {x0, x1})
    for (double cy : {y0, y1}) {
      double xu, yu;
      if (m.undistort(cx, cy, xu, yu) && xu >= 0 && xu <= m.W && yu >= 0 && yu <= m.H) add(cx, cy);
    }
  if (e.x > e.r) return Extent{0, 0, double(m.W), double(m.H)};  // no valid samples
  return e;
}

namespace {
// OpenCV's undistorted radius for a distorted radius, both focal-normalised.
double OpenCVUndistortRadius(double k1, double k2, double rd) {
  double ru = rd;
  for (int it = 0; it < 50 && rd > 0; ++it) {
    double r2 = ru * ru, f = ru * (1 + k1 * r2 + k2 * r2 * r2) - rd;
    double df = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2;
    if (df <= 1e-9) break;
    ru -= f / df;
  }
  return ru;
}
}  // namespace

NukeLD OpenCVToNuke(const OpenCVRadial& cv, int W, int H, double pa, double* maxErrPx) {
  NukeLD nk;
  nk.W = W; nk.H = H; nk.pa = pa;
  const double S = nk.s();
  const double f = 0.5 * (cv.fx * pa + cv.fy);  // in square pixels (plate rows)
  nk.cx = (cv.cx - 0.5 * W) * pa / S;
  nk.cy = (0.5 * H - cv.cy) / S;  // image y-down -> Nuke y-up
  const double q = S / f;          // Nuke radius r -> OpenCV (focal-normalised) radius r * q

  // Distorted radius range the plate covers (to its farthest corner).
  double rmax = 0;
  const double corners[4][2] = {{0, 0}, {double(W), 0}, {0, double(H)}, {double(W), double(H)}};
  for (auto& c : corners) rmax = std::max(rmax, std::hypot((c[0] - cv.cx) * pa, c[1] - cv.cy) / S);

  // Target profile: ratio t(r_d) = r_u / r_d of the OpenCV lens, in Nuke units.
  const int N = 256;
  std::vector<double> R(N), T(N), Wt(N);
  for (int i = 0; i < N; ++i) {
    R[i] = rmax * (i + 0.5) / N;
    T[i] = OpenCVUndistortRadius(cv.k1, cv.k2, R[i] * q) / (q * R[i]);
    Wt[i] = R[i];  // area weighting
  }
  // Linear init: Nuke's ratio is 1/(1 + d1 r^2 + d2 r^4), so 1/t - 1 = d1 r^2 + d2 r^4.
  double A00 = 0, A01 = 0, A11 = 0, b0 = 0, b1 = 0;
  for (int i = 0; i < N; ++i) {
    if (!(T[i] > 1e-6) || !std::isfinite(T[i])) continue;
    double r2 = R[i] * R[i], r4 = r2 * r2, y = 1.0 / T[i] - 1.0;
    A00 += Wt[i] * r2 * r2; A01 += Wt[i] * r2 * r4; A11 += Wt[i] * r4 * r4;
    b0 += Wt[i] * r2 * y; b1 += Wt[i] * r4 * y;
  }
  double det = A00 * A11 - A01 * A01;
  double d1 = 0, d2 = 0;
  if (std::fabs(det) > 1e-30) { d1 = (A11 * b0 - A01 * b1) / det; d2 = (A00 * b1 - A01 * b0) / det; }

  // Gauss-Newton on the undistorted-radius error itself.
  for (int it = 0; it < 20; ++it) {
    double J00 = 0, J01 = 0, J11 = 0, g0 = 0, g1 = 0;
    for (int i = 0; i < N; ++i) {
      double r = R[i], r2 = r * r, r4 = r2 * r2, D = 1 + d1 * r2 + d2 * r4;
      if (D <= 0 || !std::isfinite(T[i])) continue;
      double e = r / D - r * T[i];
      double j0 = -r * r2 / (D * D), j1 = -r * r4 / (D * D);
      J00 += Wt[i] * j0 * j0; J01 += Wt[i] * j0 * j1; J11 += Wt[i] * j1 * j1;
      g0 += Wt[i] * j0 * e; g1 += Wt[i] * j1 * e;
    }
    double dt = J00 * J11 - J01 * J01;
    if (std::fabs(dt) < 1e-30) break;
    double s0 = -(J11 * g0 - J01 * g1) / dt, s1 = -(J00 * g1 - J01 * g0) / dt;
    d1 += s0; d2 += s1;
    if (std::fabs(s0) + std::fabs(s1) < 1e-14) break;
  }
  nk.k1 = d1; nk.k2 = d2;

  if (maxErrPx) {
    double m = 0;
    for (int i = 0; i < N; ++i) {
      double r = R[i], r2 = r * r, D = 1 + d1 * r2 + d2 * r2 * r2;
      if (D > 0 && std::isfinite(T[i])) m = std::max(m, std::fabs(r / D - r * T[i]) * S);
    }
    *maxErrPx = m;
  }
  return nk;
}

}  // namespace anycalib
