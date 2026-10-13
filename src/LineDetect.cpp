// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
#include "LineDetect.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace anycalib {

namespace {

struct Img {
  int w = 0, h = 0;
  std::vector<float> d;
  Img() = default;
  Img(int w_, int h_) : w(w_), h(h_), d(size_t(w_) * h_, 0.f) {}
  float& at(int x, int y) { return d[size_t(y) * w + x]; }
  float at(int x, int y) const { return d[size_t(y) * w + x]; }
};

Img Blur(const Img& in) {
  static const float k[5] = {0.0545f, 0.2442f, 0.4026f, 0.2442f, 0.0545f};  // sigma ~1
  Img t(in.w, in.h), o(in.w, in.h);
  for (int y = 0; y < in.h; ++y)
    for (int x = 0; x < in.w; ++x) {
      float s = 0;
      for (int i = -2; i <= 2; ++i) s += k[i + 2] * in.at(std::clamp(x + i, 0, in.w - 1), y);
      t.at(x, y) = s;
    }
  for (int y = 0; y < in.h; ++y)
    for (int x = 0; x < in.w; ++x) {
      float s = 0;
      for (int i = -2; i <= 2; ++i) s += k[i + 2] * t.at(x, std::clamp(y + i, 0, in.h - 1));
      o.at(x, y) = s;
    }
  return o;
}

using Chain = std::vector<std::pair<float, float>>;

double PerpDist(const Chain& c, size_t a, size_t b, size_t i) {
  double x0 = c[a].first, y0 = c[a].second, x1 = c[b].first, y1 = c[b].second;
  double dx = x1 - x0, dy = y1 - y0, L = std::hypot(dx, dy);
  if (L < 1e-9) return std::hypot(c[i].first - x0, c[i].second - y0);
  return std::fabs((c[i].first - x0) * dy - (c[i].second - y0) * dx) / L;
}

// Splits at corners. The tolerance grows with length, so lens curvature doesn't split.
void Split(const Chain& c, size_t a, size_t b,
           std::vector<std::pair<size_t, size_t>>& out) {
  if (b <= a + 2) { out.push_back({a, b}); return; }
  double L = std::hypot(c[b].first - c[a].first, c[b].second - c[a].second);
  double best = 0;
  size_t bi = a;
  for (size_t i = a + 1; i < b; ++i) {
    double d = PerpDist(c, a, b, i);
    if (d > best) { best = d; bi = i; }
  }
  const double tol = std::max(2.0, 0.06 * L);
  if (best > tol) {
    Split(c, a, bi, out);
    Split(c, bi, b, out);
  } else {
    out.push_back({a, b});
  }
}

// Residual of a cubic fit y(t) in the chord frame of c[a..b]: the image of a
// straight line under radial distortion is smooth and well fitted by a low
// order curve; two lines meeting at a shallow kink are not. Returns the RMS
// and the index of the largest residual.
double CubicResidual(const Chain& c, size_t a, size_t b, size_t& worst) {
  const double x0 = c[a].first, y0 = c[a].second;
  const double dx = c[b].first - x0, dy = c[b].second - y0, L = std::hypot(dx, dy);
  worst = (a + b) / 2;
  if (L < 1e-9 || b < a + 6) return 0;
  const double ux = dx / L, uy = dy / L;
  const size_t n = b - a + 1;
  std::vector<double> T(n), Y(n);
  for (size_t i = 0; i < n; ++i) {
    double px = c[a + i].first - x0, py = c[a + i].second - y0;
    T[i] = (px * ux + py * uy) / L - 0.5;  // [-0.5, 0.5] for conditioning
    Y[i] = -px * uy + py * ux;
  }
  double A[16] = {0}, B[4] = {0};
  for (size_t i = 0; i < n; ++i) {
    double p[4] = {1, T[i], T[i] * T[i], T[i] * T[i] * T[i]};
    for (int r = 0; r < 4; ++r) {
      for (int q = 0; q < 4; ++q) A[r * 4 + q] += p[r] * p[q];
      B[r] += p[r] * Y[i];
    }
  }
  for (int col = 0; col < 4; ++col) {  // Gaussian elimination (SPD, no pivoting needed)
    if (std::fabs(A[col * 4 + col]) < 1e-12) return 0;
    for (int r = col + 1; r < 4; ++r) {
      double f = A[r * 4 + col] / A[col * 4 + col];
      for (int q = col; q < 4; ++q) A[r * 4 + q] -= f * A[col * 4 + q];
      B[r] -= f * B[col];
    }
  }
  double k[4];
  for (int r = 3; r >= 0; --r) {
    double v = B[r];
    for (int q = r + 1; q < 4; ++q) v -= A[r * 4 + q] * k[q];
    k[r] = v / A[r * 4 + r];
  }
  double ss = 0, best = -1;
  for (size_t i = 0; i < n; ++i) {
    double t = T[i], e = Y[i] - (k[0] + k[1] * t + k[2] * t * t + k[3] * t * t * t);
    ss += e * e;
    // interior points only, so a split always leaves two non-empty pieces
    if (i > 2 && i + 3 < n && std::fabs(e) > best) { best = std::fabs(e); worst = a + i; }
  }
  return std::sqrt(ss / n);
}

void SplitSmooth(const Chain& c, size_t a, size_t b, double minLen,
                 std::vector<std::pair<size_t, size_t>>& out) {
  double L = std::hypot(c[b].first - c[a].first, c[b].second - c[a].second);
  if (L < minLen) return;
  size_t worst;
  if (CubicResidual(c, a, b, worst) <= 0.25) { out.push_back({a, b}); return; }
  SplitSmooth(c, a, worst, minLen, out);
  SplitSmooth(c, worst, b, minLen, out);
}

// Centroid and scatter matrix of points x y x y ...
struct Scatter {
  double mx = 0, my = 0, sxx = 0, syy = 0, sxy = 0;
  explicit Scatter(const std::vector<double>& xy) {
    const size_t n = xy.size() / 2;
    for (size_t i = 0; i < n; ++i) { mx += xy[2 * i]; my += xy[2 * i + 1]; }
    mx /= n; my /= n;
    for (size_t i = 0; i < n; ++i) {
      double dx = xy[2 * i] - mx, dy = xy[2 * i + 1] - my;
      sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
  }
};

// RMS distance (px) of points from their total-least-squares line.
double LineRms(const std::vector<double>& xy) {
  const size_t n = xy.size() / 2;
  if (n < 3) return 0;
  const Scatter s(xy);
  // smallest eigenvalue of the scatter matrix = sum of squared normal distances
  double tr = s.sxx + s.syy, det = s.sxx * s.syy - s.sxy * s.sxy;
  double lmin = 0.5 * tr - std::sqrt(std::max(0.0, 0.25 * tr * tr - det));
  return std::sqrt(std::max(0.0, lmin) / n);
}

// Straightness (TLS RMS, px) of a plate segment (y down) after undistortion with `m`
// (Nuke y up; the RMS doesn't depend on which way y points).
double UndistortedRms(const Segment& s, const NukeLD& m) {
  const size_t n = s.xy.size() / 2;
  std::vector<double> und(2 * n);
  for (size_t i = 0; i < n; ++i)
    m.undistort(s.xy[2 * i], m.H - s.xy[2 * i + 1], und[2 * i], und[2 * i + 1]);
  return LineRms(und);
}

}  // namespace

PlumbLine PlumbLineScale(const std::vector<const Segment*>& segs, const NukeLD& m) {
  PlumbLine out;
  const int NA = 21;  // alpha = 0, 0.1, ..., 2
  const double minLen = 0.1 * std::max(m.W, m.H);
  std::vector<std::vector<double>> rms;  // [segment][alpha]
  for (const Segment* s : segs) {
    if (s->xy.size() < 6) continue;
    const size_t n = s->xy.size() / 2;
    if (std::hypot(s->xy[2 * n - 2] - s->xy[0], s->xy[2 * n - 1] - s->xy[1]) < minLen) continue;
    std::vector<double> r(NA);
    double best = 1e30;
    for (int a = 0; a < NA; ++a) {
      NukeLD ma = m;
      ma.k1 = m.k1 * a * 0.1;
      ma.k2 = m.k2 * a * 0.1;
      r[a] = UndistortedRms(*s, ma);
      best = std::min(best, r[a]);
    }
    if (best > 1.0) continue;  // not the image of a straight line under any alpha
    rms.push_back(std::move(r));
  }
  if (rms.size() < 4) return out;
  std::vector<double> e(NA, 0.0);
  for (const auto& r : rms)
    for (int a = 0; a < NA; ++a) e[a] += r[a] * r[a] / rms.size();
  int ia = int(std::min_element(e.begin(), e.end()) - e.begin());
  double alpha = 0.1 * ia;
  if (ia > 0 && ia < NA - 1) {  // parabolic refinement
    double den = e[ia - 1] - 2 * e[ia] + e[ia + 1];
    if (den > 1e-12) alpha += 0.1 * std::clamp(0.5 * (e[ia - 1] - e[ia + 1]) / den, -0.5, 0.5);
  }
  out.alpha = alpha;
  // Mean-square straightness gained vs the worse of alpha 0 / 1, in px^2.
  out.informative = std::max(e[0], e[10]) - e[ia] > 0.02;
  return out;
}

OverlayLine BuildOverlayLine(const Segment& s, const NukeLD& m) {
  OverlayLine o;
  const size_t n = s.xy.size() / 2;
  if (n < 2) return o;
  o.plate.resize(2 * n);
  std::vector<double> und(2 * n);
  for (size_t i = 0; i < n; ++i) {
    o.plate[2 * i] = s.xy[2 * i];
    o.plate[2 * i + 1] = m.H - s.xy[2 * i + 1];
    m.undistort(o.plate[2 * i], o.plate[2 * i + 1], und[2 * i], und[2 * i + 1]);
  }
  const double before = LineRms(o.plate), after = LineRms(und);
  o.straight = before < 0.75 || after < std::max(0.6, 0.5 * before);

  // Total-least-squares line through the undistorted points, mapped back into the plate.
  const Scatter sc(und);
  const double th = 0.5 * std::atan2(2 * sc.sxy, sc.sxx - sc.syy);
  const double dx = std::cos(th), dy = std::sin(th);
  double t0 = 1e30, t1 = -1e30;
  for (size_t i = 0; i < n; ++i) {
    double t = (und[2 * i] - sc.mx) * dx + (und[2 * i + 1] - sc.my) * dy;
    t0 = std::min(t0, t); t1 = std::max(t1, t);
  }
  for (int k = 0; k <= 24; ++k) {
    double t = t0 + (t1 - t0) * k / 24.0, xd, yd;
    m.distort(sc.mx + t * dx, sc.my + t * dy, xd, yd);
    o.predicted.push_back(xd);
    o.predicted.push_back(yd);
  }
  return o;
}

std::vector<Segment> DetectSegments(const float* gray, int W, int H, int maxSegments,
                                    const uint8_t* exclude) {
  std::vector<Segment> result;
  if (W < 16 || H < 16) return result;

  // downsample to <= 2048 on the long side (integer box factor)
  const int f = std::max(1, int(std::ceil(std::max(W, H) / 2048.0)));
  Img g(W / f, H / f);
  for (int y = 0; y < g.h; ++y)
    for (int x = 0; x < g.w; ++x) {
      double a = 0;
      for (int yy = 0; yy < f; ++yy)
        for (int xx = 0; xx < f; ++xx) a += gray[size_t(y * f + yy) * W + (x * f + xx)];
      g.at(x, y) = float(a / (f * f));
    }
  g = Blur(g);

  // Excluded pixels at this resolution, grown by 2 px so edges along the mask's border go too.
  std::vector<uint8_t> ex;
  if (exclude) {
    std::vector<uint8_t> e(size_t(g.w) * g.h, 0);
    for (int y = 0; y < g.h * f; ++y)
      for (int x = 0; x < g.w * f; ++x)
        if (exclude[size_t(y) * W + x]) e[size_t(y / f) * g.w + x / f] = 1;
    ex.assign(e.size(), 0);
    for (int y = 0; y < g.h; ++y)
      for (int x = 0; x < g.w; ++x)
        if (e[size_t(y) * g.w + x])
          for (int ny = std::max(0, y - 2); ny <= std::min(g.h - 1, y + 2); ++ny)
            for (int nx = std::max(0, x - 2); nx <= std::min(g.w - 1, x + 2); ++nx)
              ex[size_t(ny) * g.w + nx] = 1;
  }

  // Sobel + non-maximum suppression
  Img mag(g.w, g.h), nms(g.w, g.h);
  std::vector<uint8_t> dir(size_t(g.w) * g.h, 0);
  for (int y = 1; y < g.h - 1; ++y)
    for (int x = 1; x < g.w - 1; ++x) {
      float gx = (g.at(x + 1, y - 1) + 2 * g.at(x + 1, y) + g.at(x + 1, y + 1)) -
                 (g.at(x - 1, y - 1) + 2 * g.at(x - 1, y) + g.at(x - 1, y + 1));
      float gy = (g.at(x - 1, y + 1) + 2 * g.at(x, y + 1) + g.at(x + 1, y + 1)) -
                 (g.at(x - 1, y - 1) + 2 * g.at(x, y - 1) + g.at(x + 1, y - 1));
      mag.at(x, y) = std::hypot(gx, gy);
      double a = std::atan2(gy, gx);
      if (a < 0) a += M_PI;
      dir[size_t(y) * g.w + x] = uint8_t(int(std::lround(a / (M_PI / 4))) % 4);
    }
  // NMS along the gradient; the surviving maxima are located to sub-pixel
  // precision with a parabola through (a, m, b) — integer edge positions are
  // too noisy (~0.5 px) to see lens curvature on shorter lines.
  std::vector<float> strong;
  Img offX(g.w, g.h), offY(g.w, g.h);
  for (int y = 1; y < g.h - 1; ++y)
    for (int x = 1; x < g.w - 1; ++x) {
      float m = mag.at(x, y);
      if (m <= 0 || (!ex.empty() && ex[size_t(y) * g.w + x])) continue;
      float a, b;
      int sx, sy;  // step from a (behind) to b (ahead)
      switch (dir[size_t(y) * g.w + x]) {
        case 0: a = mag.at(x - 1, y); b = mag.at(x + 1, y); sx = 1; sy = 0; break;
        case 1: a = mag.at(x - 1, y - 1); b = mag.at(x + 1, y + 1); sx = 1; sy = 1; break;
        case 2: a = mag.at(x, y - 1); b = mag.at(x, y + 1); sx = 0; sy = 1; break;
        default: a = mag.at(x + 1, y - 1); b = mag.at(x - 1, y + 1); sx = -1; sy = 1; break;
      }
      if (m >= a && m > b) {
        nms.at(x, y) = m;
        strong.push_back(m);
        float den = a - 2 * m + b;
        float d = den < 0 ? std::clamp(0.5f * (a - b) / den, -0.5f, 0.5f) : 0.f;
        offX.at(x, y) = d * sx;
        offY.at(x, y) = d * sy;
      }
    }
  if (strong.size() < 100) return result;

  // hysteresis (adaptive thresholds)
  std::nth_element(strong.begin(), strong.begin() + strong.size() * 80 / 100, strong.end());
  const float hi = std::max(0.08f, strong[strong.size() * 80 / 100]);
  const float lo = 0.4f * hi;
  std::vector<uint8_t> edge(size_t(g.w) * g.h, 0);
  std::vector<std::pair<int, int>> stack;
  for (int y = 1; y < g.h - 1; ++y)
    for (int x = 1; x < g.w - 1; ++x)
      if (nms.at(x, y) >= hi && !edge[size_t(y) * g.w + x]) {
        edge[size_t(y) * g.w + x] = 1;
        stack.push_back({x, y});
        while (!stack.empty()) {
          auto [cx, cy] = stack.back();
          stack.pop_back();
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              int nx = cx + dx, ny = cy + dy;
              if (nx < 1 || ny < 1 || nx >= g.w - 1 || ny >= g.h - 1) continue;
              size_t k = size_t(ny) * g.w + nx;
              if (!edge[k] && nms.at(nx, ny) >= lo) { edge[k] = 1; stack.push_back({nx, ny}); }
            }
        }
      }

  // link edge pixels into chains (start at endpoints, then loops)
  auto isEdge = [&](int x, int y) {
    return x >= 0 && y >= 0 && x < g.w && y < g.h && edge[size_t(y) * g.w + x] == 1;
  };
  auto neighbours = [&](int x, int y) {
    int n = 0;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx)
        if ((dx || dy) && isEdge(x + dx, y + dy)) ++n;
    return n;
  };
  static const int ORDER[8][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1},
                                  {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
  std::vector<Chain> chains;
  auto trace = [&](int x, int y) {
    Chain c;
    edge[size_t(y) * g.w + x] = 2;
    c.push_back({x + offX.at(x, y), y + offY.at(x, y)});
    for (;;) {
      bool moved = false;
      for (auto& o : ORDER) {
        int nx = x + o[0], ny = y + o[1];
        if (isEdge(nx, ny)) {
          edge[size_t(ny) * g.w + nx] = 2;
          c.push_back({nx + offX.at(nx, ny), ny + offY.at(nx, ny)});
          x = nx; y = ny;
          moved = true;
          break;
        }
      }
      if (!moved) break;
    }
    if (c.size() >= 10) chains.push_back(std::move(c));
  };
  for (int pass = 0; pass < 2; ++pass)
    for (int y = 1; y < g.h - 1; ++y)
      for (int x = 1; x < g.w - 1; ++x)
        if (isEdge(x, y) && (pass == 1 || neighbours(x, y) == 1)) trace(x, y);

  // split at corners, keep long smooth segments
  const double Lmin = 0.06 * std::max(g.w, g.h);
  struct Cand { double len; Chain pts; };
  std::vector<Cand> cands;
  for (auto& c : chains) {
    std::vector<std::pair<size_t, size_t>> corners, pieces;
    Split(c, 0, c.size() - 1, corners);
    for (auto [a, b] : corners) SplitSmooth(c, a, b, Lmin, pieces);
    for (auto [a, b] : pieces) {
      double L = std::hypot(c[b].first - c[a].first, c[b].second - c[a].second);
      if (L < Lmin) continue;
      double path = 0;
      for (size_t i = a + 1; i <= b; ++i)
        path += std::hypot(c[i].first - c[i - 1].first, c[i].second - c[i - 1].second);
      if (path > 1.12 * L) continue;  // wiggly (foliage, texture), not a line image
      cands.push_back({L, Chain(c.begin() + a, c.begin() + b + 1)});
    }
  }
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.len > b.len; });
  if (int(cands.size()) > maxSegments) cands.resize(maxSegments);

  // resample to 24 points along arc length, back to plate pixels
  for (const Cand& c : cands) {
    std::vector<double> acc(c.pts.size(), 0.0);
    for (size_t i = 1; i < c.pts.size(); ++i)
      acc[i] = acc[i - 1] + std::hypot(c.pts[i].first - c.pts[i - 1].first,
                                       c.pts[i].second - c.pts[i - 1].second);
    const int K = 24;
    Segment s;
    size_t j = 0;
    for (int k = 0; k < K; ++k) {
      double t = acc.back() * k / (K - 1);
      while (j + 1 < acc.size() && acc[j + 1] < t) ++j;
      size_t j1 = std::min(j + 1, acc.size() - 1);
      double seg = acc[j1] - acc[j], a = seg > 0 ? (t - acc[j]) / seg : 0;
      double x = c.pts[j].first + a * (c.pts[j1].first - c.pts[j].first);
      double y = c.pts[j].second + a * (c.pts[j1].second - c.pts[j].second);
      s.xy.push_back(float((x + 0.5) * f));
      s.xy.push_back(float((y + 0.5) * f));
    }
    result.push_back(std::move(s));
  }
  return result;
}

}  // namespace anycalib
