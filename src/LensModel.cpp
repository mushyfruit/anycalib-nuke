// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
#include "LensModel.h"

#include <algorithm>
#include <cmath>

namespace anycalib {

namespace {

double Denominator(const NukeLD& m, double r) {
  const double r2 = r * r;
  return 1 + m.k1 * r2 + m.k2 * r2 * r2;
}

double G(const NukeLD& m, double r) { return r / Denominator(m, r); }

double GSlope(const NukeLD& m, double r) {
  const double r2 = r * r, d = Denominator(m, r), dD = 2 * m.k1 * r + 4 * m.k2 * r2 * r;
  return (d - r * dD) / (d * d);
}

bool InvertByNewton(const NukeLD& m, double ru, double& rd) {
  double r = ru;
  for (int it = 0; it < 60; ++it) {
    if (Denominator(m, r) <= 1e-9 || GSlope(m, r) <= 1e-12) break;
    const double step = (G(m, r) - ru) / GSlope(m, r);
    double rn = r - step;
    if (rn <= 0) rn = 0.5 * r;
    r = rn;
    if (std::fabs(step) < 1e-13 * std::max(1.0, r)) {
      if (GSlope(m, r) > 0 && Denominator(m, r) > 0) { rd = r; return true; }
      break;
    }
  }
  return false;
}

// Fallback: bracket the increasing branch [0, rmax] and bisect.
bool InvertByBisection(const NukeLD& m, double ru, double& rd) {
  double rmax = 0;
  const double h = 1e-3;
  while (rmax < 50 && Denominator(m, rmax + h) > 1e-9 && GSlope(m, rmax + h) > 0) rmax += h;
  if (G(m, rmax) < ru) { rd = rmax; return false; }  // beyond the invertible range
  double lo = 0, hi = rmax;
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    (G(m, mid) < ru ? lo : hi) = mid;
  }
  rd = 0.5 * (lo + hi);
  return true;
}

}  // namespace

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

bool NukeLD::distortRadius(double ru, double& rd) const {
  if (ru <= 0) { rd = 0; return true; }
  if (isIdentity()) { rd = ru; return true; }
  // Inverts G on its branch through r = 0.
  return InvertByNewton(*this, ru, rd) || InvertByBisection(*this, ru, rd);
}

namespace {

void Grow(Extent& e, double x, double y) {
  e.x = std::min(e.x, x); e.r = std::max(e.r, x);
  e.y = std::min(e.y, y); e.t = std::max(e.t, y);
}

Extent OrFormat(const Extent& e, const NukeLD& m) {
  if (e.x > e.r) return Extent{0, 0, double(m.W), double(m.H)};  // no valid samples
  return e;
}

// The format boundary at t in [0, 4] (bottom, right, top, left).
void FormatBoundary(const NukeLD& m, double t, double& x, double& y) {
  const int k = std::min(3, int(t));
  const double f = t - k;
  switch (k) {
    case 0: x = f * m.W; y = 0; break;
    case 1: x = m.W; y = f * m.H; break;
    case 2: x = (1 - f) * m.W; y = m.H; break;
    default: x = 0; y = (1 - f) * m.H; break;
  }
}

}  // namespace

Extent UndistortedExtent(const NukeLD& m, int n) {
  Extent e{double(m.W), double(m.H), 0, 0};
  n = std::max(n, 2);
  auto add = [&](double x, double y) {
    double xu, yu;
    if (m.undistort(x, y, xu, yu)) Grow(e, xu, yu);
  };
  for (int i = 0; i < n; ++i) {
    const double t = double(i) / (n - 1);
    add(t * m.W, 0); add(t * m.W, m.H);
    add(0, t * m.H); add(m.W, t * m.H);
  }
  return OrFormat(e, m);
}

Extent RedistortedExtent(const NukeLD& m, int n) {
  // The distorted format clipped to the square: its bbox is on the format boundary, where
  // that crosses the square, or at the square's corners.
  const double hx = m.s() / m.pa, hy = m.s();
  const double x0 = 0.5 * m.W - hx, x1 = 0.5 * m.W + hx, y0 = 0.5 * m.H - hy, y1 = 0.5 * m.H + hy;
  auto inSquare = [&](double x, double y) { return x >= x0 && x <= x1 && y >= y0 && y <= y1; };
  auto boundary = [&](double t, double& x, double& y) {
    double px, py;
    FormatBoundary(m, t, px, py);
    return m.distort(px, py, x, y) && inSquare(x, y);
  };
  Extent e{x1, y1, x0, y0};
  const int N = 4 * std::max(n, 2);
  double x, y;
  bool prev = boundary(0, x, y);
  for (int i = 0; i <= N; ++i) {
    const double t = 4.0 * i / N;
    const bool in = boundary(t, x, y);
    if (in) Grow(e, x, y);
    if (i > 0 && in != prev) {
      double a = 4.0 * (i - 1) / N, b = t;  // boundary(a) == prev, boundary(b) == in
      for (int it = 0; it < 50; ++it) {
        const double mid = 0.5 * (a + b);
        (boundary(mid, x, y) == prev ? a : b) = mid;
      }
      if (boundary(prev ? a : b, x, y)) Grow(e, x, y);
    }
    prev = in;
  }
  for (double cx : {x0, x1})
    for (double cy : {y0, y1}) {
      double xu, yu;
      if (m.undistort(cx, cy, xu, yu) && xu >= 0 && xu <= m.W && yu >= 0 && yu <= m.H)
        Grow(e, cx, cy);
    }
  return OrFormat(e, m);
}

}  // namespace anycalib
