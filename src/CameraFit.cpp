// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
#include "CameraFit.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace anycalib {

namespace {

constexpr double kRadToDeg = 57.29577951308232;
constexpr int kMaxParams = 6;

// Normal equations A^T A x = A^T b, accumulated a row at a time (lower triangle only).
struct Normal {
  int n;
  double AtA[kMaxParams * kMaxParams] = {};
  double Atb[kMaxParams] = {};

  explicit Normal(int n) : n(n) {}

  void add(const double* a, double b) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) AtA[i * n + j] += a[i] * a[j];
      Atb[i] += a[i] * b;
    }
  }

  // Cholesky, with the diagonal scaled by 1 + damping.
  bool solve(double damping, double* x) const {
    double L[kMaxParams * kMaxParams];
    for (int i = 0; i < n; ++i)
      for (int j = 0; j <= i; ++j) {
        double v = AtA[i * n + j];
        if (i == j) v += damping * std::max(v, 1e-12);
        for (int k = 0; k < j; ++k) v -= L[i * n + k] * L[j * n + k];
        if (i == j) {
          if (!(v > 0)) return false;
          L[i * n + i] = std::sqrt(v);
        } else {
          L[i * n + j] = v / L[j * n + j];
        }
      }
    for (int i = 0; i < n; ++i) {  // L y = A^T b
      x[i] = Atb[i];
      for (int k = 0; k < i; ++k) x[i] -= L[i * n + k] * x[k];
      x[i] /= L[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {  // L^T x = y
      for (int k = i + 1; k < n; ++k) x[i] -= L[k * n + i] * x[k];
      x[i] /= L[i * n + i];
    }
    return true;
  }
};

// A unit ray and its tangent basis, as AnyCalib's Unit3.get_tangent_basis.
struct Observation {
  double o[3], e1[3], e2[3];
};

Observation Observe(const RaySample& s) {
  Observation b;
  const double n = std::sqrt(double(s.rx) * s.rx + double(s.ry) * s.ry + double(s.rz) * s.rz);
  b.o[0] = s.rx / n; b.o[1] = s.ry / n; b.o[2] = s.rz / n;
  const double h = std::hypot(b.o[0], b.o[2]);  // > 0 for rays in front of the camera
  b.e1[0] = b.o[2] / h; b.e1[1] = 0; b.e1[2] = -b.o[0] / h;
  b.e2[0] = b.o[1] * b.e1[2];
  b.e2[1] = b.o[2] * b.e1[0] - b.o[0] * b.e1[2];
  b.e2[2] = -b.o[1] * b.e1[0];
  return b;
}

// Log map of the predicted ray (x, y, 1) at the observed one (AnyCalib's res_tangent =
// "observed"): r = g t, g = atan2(|t|, c) / |t|. J = dr / d(x, y).
void Residual(const Observation& b, double x, double y, double r[2], double J[2][2]) {
  const double* e[2] = {b.e1, b.e2};
  const double t[2] = {e[0][0] * x + e[0][1] * y + e[0][2], e[1][0] * x + e[1][1] * y + e[1][2]};
  const double c = b.o[0] * x + b.o[1] * y + b.o[2];
  const double s = std::hypot(t[0], t[1]);
  const bool tiny = s < 1e-12 * c;  // g -> 1 / c
  const double g = tiny ? 1 / c : std::atan2(s, c) / s;
  for (int k = 0; k < 2; ++k) {
    double dg = -b.o[k] / (c * c);
    if (!tiny) {
      const double ds = (t[0] * e[0][k] + t[1] * e[1][k]) / s;
      const double dAngle = (c * ds - s * b.o[k]) / (s * s + c * c);
      dg = (dAngle - g * ds) / s;
    }
    for (int i = 0; i < 2; ++i) J[i][k] = g * e[i][k] + t[i] * dg;
  }
  r[0] = g * t[0];
  r[1] = g * t[1];
}

// A sample's predicted ray (x, y, 1) and its derivatives. Samples the model can't
// undistort are left out, as in AnyCalib.
struct Projection {
  double x, y, dx[kMaxParams], dy[kMaxParams];
};
using Model = std::function<bool(const RaySample&, const double* p, Projection&)>;

// Of kMaxParams parameters, the first np are fitted and the rest held.
struct Problem {
  std::vector<RaySample> samples;
  std::vector<Observation> obs;
  Model model;
  int np = 0;
};

bool Prepare(const std::vector<RaySample>& s, Problem& pb, std::string* err) {
  for (const RaySample& r : s)
    if (r.rz > 1e-6f) {
      pb.samples.push_back(r);
      pb.obs.push_back(Observe(r));
    }
  if (pb.samples.size() >= 20) return true;
  if (err) *err = "too few valid rays";
  return false;
}

// Mean squared angle (rad^2) and the Gauss-Newton normal equations.
struct Linearisation {
  Normal ne;
  double cost = std::numeric_limits<double>::infinity();
  size_t valid = 0;
};

Linearisation Linearise(const Problem& pb, const double* p) {
  Linearisation L{Normal(pb.np)};
  double sum = 0;
  for (size_t i = 0; i < pb.samples.size(); ++i) {
    Projection q;
    if (!pb.model(pb.samples[i], p, q)) continue;
    double r[2], J[2][2];
    Residual(pb.obs[i], q.x, q.y, r, J);
    if (!std::isfinite(r[0] + r[1])) continue;
    for (int k = 0; k < 2; ++k) {
      double row[kMaxParams];
      for (int j = 0; j < pb.np; ++j) row[j] = J[k][0] * q.dx[j] + J[k][1] * q.dy[j];
      L.ne.add(row, -r[k]);
    }
    sum += r[0] * r[0] + r[1] * r[1];
    ++L.valid;
  }
  if (L.valid) L.cost = sum / L.valid;
  return L;
}

double RmsDeg(const Problem& pb, const double* p) {
  const Linearisation L = Linearise(pb, p);
  return L.valid ? std::sqrt(L.cost) * kRadToDeg : 0;
}

// AnyCalib's refinement (optim/gauss_newton.py): undamped Gauss-Newton for a fixed number
// of iterations, kept only if it lowered the cost.
void GaussNewton(const Problem& pb, double* p, int iters) {
  double q[kMaxParams];
  std::copy(p, p + kMaxParams, q);
  Linearisation L = Linearise(pb, q);
  const double cost0 = L.cost;
  for (int it = 0; it < iters; ++it) {
    double d[kMaxParams];
    if (L.valid <= size_t(pb.np) || !L.ne.solve(0, d)) break;
    for (int j = 0; j < pb.np; ++j) q[j] += d[j];
    L = Linearise(pb, q);
  }
  if (L.cost < cost0) std::copy(q, q + pb.np, p);
}

// A step mustn't lose samples: pushing them out of the invertible range would look like a gain.
void LevenbergMarquardt(const Problem& pb, double* p, int iters) {
  Linearisation cur = Linearise(pb, p);
  if (cur.valid <= size_t(pb.np)) return;
  double lambda = 1e-4;
  for (int it = 0, fails = 0; it < iters && fails < 10;) {
    double d[kMaxParams], q[kMaxParams];
    std::copy(p, p + kMaxParams, q);
    if (cur.ne.solve(lambda, d)) {
      for (int j = 0; j < pb.np; ++j) q[j] += d[j];
      Linearisation next = Linearise(pb, q);
      if (next.cost < cur.cost && next.valid >= cur.valid) {
        const bool converged = cur.cost - next.cost < 1e-12 * cur.cost;
        std::copy(q, q + pb.np, p);
        cur = next;
        lambda = std::max(lambda / 10, 1e-12);
        ++it;
        fails = 0;
        if (converged) return;
        continue;
      }
    }
    lambda *= 10;
    ++fails;
  }
}

// AnyCalib's radial:2 camera, p = [fx, fy, cx, cy, k1, k2]. The undistorted point w solves
// w D(|w|^2) = w0, with w0 the distorted point and D = 1 + k1 r^2 + k2 r^4, so
// (D I + a w w^T) dw = dw0 - w dD|w, a = 2 dD/d(r^2), which Sherman-Morrison inverts.
bool ProjectOpenCV(const RaySample& s, const double* p, Projection& q) {
  const OpenCVRadial m{p[0], p[1], p[2], p[3], p[4], p[5]};
  double xu, yu;
  if (!m.undistort(s.u, s.v, xu, yu)) return false;
  q.x = (xu - m.cx) / m.fx;
  q.y = (yu - m.cy) / m.fy;
  const double x0 = (s.u - m.cx) / m.fx, y0 = (s.v - m.cy) / m.fy;
  const double r2 = q.x * q.x + q.y * q.y;
  const double D = 1 + m.k1 * r2 + m.k2 * r2 * r2;
  const double a = 2 * (m.k1 + 2 * m.k2 * r2);
  const double den = D + a * r2;  // 1 + 3 k1 r^2 + 5 k2 r^4
  const double b[6][2] = {{-x0 / m.fx, 0}, {0, -y0 / m.fy}, {-1 / m.fx, 0}, {0, -1 / m.fy},
                          {-q.x * r2, -q.y * r2}, {-q.x * r2 * r2, -q.y * r2 * r2}};
  for (int j = 0; j < 6; ++j) {
    const double wb = a * (q.x * b[j][0] + q.y * b[j][1]) / den;
    q.dx[j] = (b[j][0] - wb * q.x) / D;
    q.dy[j] = (b[j][1] - wb * q.y) / D;
  }
  return true;
}

// AnyCalib's linear fit (Radial.fit). With (x, y) a ray's pinhole projection and r its
// radius, u / fx - cx / fx - k1 x r^2 - k2 x r^4 = x is linear in
// [1/fx, 1/fy, cx/fx, cy/fy, k1, k2], and likewise for v. Pixel coordinates are divided
// by their maximum per axis for conditioning. Rays more than 85 degrees off-axis
// (AnyCalib's max_fov of 170) are left out: their projections explode and wreck the fit
// on wide lenses.
bool LinearFit(const std::vector<RaySample>& s, double p[6]) {
  double su = 1, sv = 1;
  for (const RaySample& r : s) {
    su = std::max(su, double(r.u));
    sv = std::max(sv, double(r.v));
  }
  const double cosMax = std::cos(85 / kRadToDeg);
  Normal ne(6);
  int used = 0;
  for (const RaySample& r : s) {
    if (r.rz < cosMax * std::sqrt(double(r.rx) * r.rx + double(r.ry) * r.ry + double(r.rz) * r.rz))
      continue;
    const double x = double(r.rx) / r.rz, y = double(r.ry) / r.rz, r2 = x * x + y * y;
    const double ax[6] = {r.u / su, 0, -1, 0, -x * r2, -x * r2 * r2};
    const double ay[6] = {0, r.v / sv, 0, -1, -y * r2, -y * r2 * r2};
    ne.add(ax, x);
    ne.add(ay, y);
    ++used;
  }
  double z[6];
  if (used < 20 || !ne.solve(0, z) || !(z[0] > 0 && z[1] > 0)) return false;
  const double fx = su / z[0], fy = sv / z[1];
  const double init[6] = {fx, fy, z[2] * fx, z[3] * fy, z[4], z[5]};
  std::copy(init, init + 6, p);
  return true;
}

}  // namespace

std::vector<RaySample> Thin(const std::vector<RaySample>& s, size_t maxN) {
  if (maxN == 0 || s.size() <= maxN) return s;
  std::vector<RaySample> out(maxN);
  for (size_t i = 0; i < maxN; ++i) out[i] = s[i * s.size() / maxN];
  return out;
}

bool FitOpenCVRadial(const std::vector<RaySample>& s, OpenCVRadial& out, FitStats* stats,
                     std::string* err) {
  Problem pb;
  if (!Prepare(s, pb, err)) return false;
  double p[kMaxParams];
  if (!LinearFit(pb.samples, p)) {
    if (err) *err = "linear fit failed";
    return false;
  }
  pb.model = ProjectOpenCV;
  pb.np = 6;
  GaussNewton(pb, p, 10);         // AnyCalib's own refinement
  LevenbergMarquardt(pb, p, 30);  // polish: only ever lowers the cost
  if (!(p[0] > 0 && p[1] > 0) || !std::isfinite(p[4]) || !std::isfinite(p[5])) {
    if (err) *err = "refinement diverged";
    return false;
  }
  out = OpenCVRadial{p[0], p[1], p[2], p[3], p[4], p[5]};
  if (stats) stats->rmsDeg = RmsDeg(pb, p);
  return true;
}

bool FitNukeLD(const std::vector<RaySample>& s, int W, int H, double pa, const OpenCVRadial& init,
               NukeLD& out, double& focalPx, FitStats* stats, std::string* err, bool fixCentre) {
  Problem pb;
  if (!Prepare(s, pb, err)) return false;
  const NukeLD nk0 = OpenCVToNuke(init, W, H, pa);
  const double S = nk0.s();
  // p = [f, k1, k2, cx, cy], the centre in image pixels (y down).
  double p[kMaxParams] = {0.5 * (init.fx * pa + init.fy), nk0.k1, nk0.k2,
                          fixCentre ? 0.5 * W : init.cx, fixCentre ? 0.5 * H : init.cy};
  pb.np = fixCentre ? 3 : 5;
  // n = (u - c) (pa, 1) / S, w = n S / (f D), D = 1 + k1 |n|^2 + k2 |n|^4.
  pb.model = [pa, S](const RaySample& r, const double* p, Projection& q) {
    const double nx = (r.u - p[3]) * pa / S, ny = (r.v - p[4]) / S;
    const double r2 = nx * nx + ny * ny;
    const double D = 1 + p[1] * r2 + p[2] * r2 * r2;
    if (!(D > 0)) return false;
    const double k = S / (p[0] * D);
    q.x = nx * k;
    q.y = ny * k;
    q.dx[0] = -q.x / p[0];
    q.dy[0] = -q.y / p[0];
    q.dx[1] = -q.x * r2 / D;
    q.dy[1] = -q.y * r2 / D;
    q.dx[2] = q.dx[1] * r2;
    q.dy[2] = q.dy[1] * r2;
    const double a = 2 * (p[1] + 2 * p[2] * r2) / D;  // dD / D = a n . dn
    const double dn[2][2] = {{-pa / S, 0}, {0, -1 / S}};
    for (int j = 0; j < 2; ++j) {
      const double dD = a * (nx * dn[j][0] + ny * dn[j][1]);
      q.dx[3 + j] = k * dn[j][0] - q.x * dD;
      q.dy[3 + j] = k * dn[j][1] - q.y * dD;
    }
    return true;
  };
  LevenbergMarquardt(pb, p, 30);
  if (!(p[0] > 0) || !std::isfinite(p[1]) || !std::isfinite(p[2])) {
    if (err) *err = "Nuke model refinement diverged";
    return false;
  }
  out.W = W; out.H = H; out.pa = pa;
  out.k1 = p[1]; out.k2 = p[2];
  out.cx = (p[3] - 0.5 * W) * pa / S;
  out.cy = (0.5 * H - p[4]) / S;  // image y-down -> Nuke y-up
  focalPx = p[0];
  if (stats) stats->rmsDeg = RmsDeg(pb, p);
  return true;
}

double CornerShiftPx(const NukeLD& m) {
  double best = 0;
  const double c[4][2] = {{0, 0}, {double(m.W), 0}, {0, double(m.H)}, {double(m.W), double(m.H)}};
  for (auto& q : c) {
    double xu, yu;
    m.undistort(q[0], q[1], xu, yu);
    best = std::max(best, std::hypot(xu - q[0], yu - q[1]));
  }
  return best;
}

}  // namespace anycalib
