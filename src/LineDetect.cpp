// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0
#include "LineDetect.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace anycalib {

namespace {

template <class T>
struct Grid {
  int w = 0, h = 0;
  std::vector<T> d;
  Grid() = default;
  Grid(int w_, int h_) : w(w_), h(h_), d(size_t(w_) * h_, T(0)) {}
  T& at(int x, int y) { return d[size_t(y) * w + x]; }
  T at(int x, int y) const { return d[size_t(y) * w + x]; }
};
using Img = Grid<float>;
using Mask = Grid<uint8_t>;

struct Pt {
  float x, y;
};
using Chain = std::vector<Pt>;
using Range = std::pair<size_t, size_t>;

double Dist(const Pt& a, const Pt& b) { return std::hypot(b.x - a.x, b.y - a.y); }

double PathLength(const Chain& c, size_t a, size_t b) {
  double path = 0;
  for (size_t i = a + 1; i <= b; ++i) path += Dist(c[i - 1], c[i]);
  return path;
}

Img Downsample(const float* gray, int W, int H, int f) {
  Img g(W / f, H / f);
  for (int y = 0; y < g.h; ++y)
    for (int x = 0; x < g.w; ++x) {
      double a = 0;
      for (int yy = 0; yy < f; ++yy)
        for (int xx = 0; xx < f; ++xx) a += gray[size_t(y * f + yy) * W + (x * f + xx)];
      g.at(x, y) = float(a / (f * f));
    }
  return g;
}

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

struct Gradient {
  Img mag;
  Mask dir;
};

Gradient Sobel(const Img& g) {
  Gradient gr{Img(g.w, g.h), Mask(g.w, g.h)};
  for (int y = 1; y < g.h - 1; ++y)
    for (int x = 1; x < g.w - 1; ++x) {
      float gx = (g.at(x + 1, y - 1) + 2 * g.at(x + 1, y) + g.at(x + 1, y + 1)) -
                 (g.at(x - 1, y - 1) + 2 * g.at(x - 1, y) + g.at(x - 1, y + 1));
      float gy = (g.at(x - 1, y + 1) + 2 * g.at(x, y + 1) + g.at(x + 1, y + 1)) -
                 (g.at(x - 1, y - 1) + 2 * g.at(x, y - 1) + g.at(x + 1, y - 1));
      gr.mag.at(x, y) = std::hypot(gx, gy);
      double a = std::atan2(gy, gx);
      if (a < 0) a += M_PI;
      gr.dir.at(x, y) = uint8_t(int(std::lround(a / (M_PI / 4))) % 4);
    }
  return gr;
}

struct Edges {
  Img nms, offX, offY;
  std::vector<float> strong;
};

// NMS, with sub-pixel maxima from a parabola: integer positions are too noisy for lens curvature.
Edges Suppress(const Gradient& gr) {
  const Img& mag = gr.mag;
  Edges e{Img(mag.w, mag.h), Img(mag.w, mag.h), Img(mag.w, mag.h), {}};
  for (int y = 1; y < mag.h - 1; ++y)
    for (int x = 1; x < mag.w - 1; ++x) {
      float m = mag.at(x, y);
      if (m <= 0) continue;
      float a, b;
      int sx, sy;  // step from a (behind) to b (ahead)
      switch (gr.dir.at(x, y)) {
        case 0: a = mag.at(x - 1, y); b = mag.at(x + 1, y); sx = 1; sy = 0; break;
        case 1: a = mag.at(x - 1, y - 1); b = mag.at(x + 1, y + 1); sx = 1; sy = 1; break;
        case 2: a = mag.at(x, y - 1); b = mag.at(x, y + 1); sx = 0; sy = 1; break;
        default: a = mag.at(x + 1, y - 1); b = mag.at(x - 1, y + 1); sx = -1; sy = 1; break;
      }
      if (m >= a && m > b) {
        e.nms.at(x, y) = m;
        e.strong.push_back(m);
        float den = a - 2 * m + b;
        float d = den < 0 ? std::clamp(0.5f * (a - b) / den, -0.5f, 0.5f) : 0.f;
        e.offX.at(x, y) = d * sx;
        e.offY.at(x, y) = d * sy;
      }
    }
  return e;
}

Mask Hysteresis(const Img& nms, std::vector<float>& strong) {
  std::nth_element(strong.begin(), strong.begin() + strong.size() * 80 / 100, strong.end());
  const float hi = std::max(0.08f, strong[strong.size() * 80 / 100]);
  const float lo = 0.4f * hi;
  Mask edge(nms.w, nms.h);
  std::vector<std::pair<int, int>> stack;
  for (int y = 1; y < nms.h - 1; ++y)
    for (int x = 1; x < nms.w - 1; ++x)
      if (nms.at(x, y) >= hi && !edge.at(x, y)) {
        edge.at(x, y) = 1;
        stack.push_back({x, y});
        while (!stack.empty()) {
          auto [cx, cy] = stack.back();
          stack.pop_back();
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              int nx = cx + dx, ny = cy + dy;
              if (nx < 1 || ny < 1 || nx >= nms.w - 1 || ny >= nms.h - 1) continue;
              if (!edge.at(nx, ny) && nms.at(nx, ny) >= lo) {
                edge.at(nx, ny) = 1;
                stack.push_back({nx, ny});
              }
            }
        }
      }
  return edge;
}

bool IsEdge(const Mask& edge, int x, int y) {
  return x >= 0 && y >= 0 && x < edge.w && y < edge.h && edge.at(x, y) == 1;
}

int EdgeNeighbours(const Mask& edge, int x, int y) {
  int n = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
      if ((dx || dy) && IsEdge(edge, x + dx, y + dy)) ++n;
  return n;
}

Chain Trace(Mask& edge, const Edges& e, int x, int y) {
  static const int ORDER[8][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1},
                                  {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
  Chain c;
  edge.at(x, y) = 2;
  c.push_back({x + e.offX.at(x, y), y + e.offY.at(x, y)});
  for (;;) {
    bool moved = false;
    for (auto& o : ORDER) {
      int nx = x + o[0], ny = y + o[1];
      if (IsEdge(edge, nx, ny)) {
        edge.at(nx, ny) = 2;
        c.push_back({nx + e.offX.at(nx, ny), ny + e.offY.at(nx, ny)});
        x = nx; y = ny;
        moved = true;
        break;
      }
    }
    if (!moved) break;
  }
  return c;
}

// Endpoints first, then loops.
std::vector<Chain> LinkChains(Mask edge, const Edges& e) {
  std::vector<Chain> chains;
  for (int pass = 0; pass < 2; ++pass)
    for (int y = 1; y < edge.h - 1; ++y)
      for (int x = 1; x < edge.w - 1; ++x)
        if (IsEdge(edge, x, y) && (pass == 1 || EdgeNeighbours(edge, x, y) == 1)) {
          Chain c = Trace(edge, e, x, y);
          if (c.size() >= 10) chains.push_back(std::move(c));
        }
  return chains;
}

double PerpDist(const Chain& c, size_t a, size_t b, size_t i) {
  double x0 = c[a].x, y0 = c[a].y, x1 = c[b].x, y1 = c[b].y;
  double dx = x1 - x0, dy = y1 - y0, L = std::hypot(dx, dy);
  if (L < 1e-9) return std::hypot(c[i].x - x0, c[i].y - y0);
  return std::fabs((c[i].x - x0) * dy - (c[i].y - y0) * dx) / L;
}

// Splits at corners. The tolerance grows with length, so lens curvature doesn't split.
void Split(const Chain& c, size_t a, size_t b, std::vector<Range>& out) {
  if (b <= a + 2) { out.push_back({a, b}); return; }
  double best = 0;
  size_t bi = a;
  for (size_t i = a + 1; i < b; ++i) {
    double d = PerpDist(c, a, b, i);
    if (d > best) { best = d; bi = i; }
  }
  const double tol = std::max(2.0, 0.06 * Dist(c[a], c[b]));
  if (best > tol) {
    Split(c, a, bi, out);
    Split(c, bi, b, out);
  } else {
    out.push_back({a, b});
  }
}

bool FitCubic(const std::vector<double>& T, const std::vector<double>& Y, double k[4]) {
  double A[16] = {0}, B[4] = {0};
  for (size_t i = 0; i < T.size(); ++i) {
    double p[4] = {1, T[i], T[i] * T[i], T[i] * T[i] * T[i]};
    for (int r = 0; r < 4; ++r) {
      for (int q = 0; q < 4; ++q) A[r * 4 + q] += p[r] * p[q];
      B[r] += p[r] * Y[i];
    }
  }
  for (int col = 0; col < 4; ++col) {  // Gaussian elimination (SPD, no pivoting needed)
    if (std::fabs(A[col * 4 + col]) < 1e-12) return false;
    for (int r = col + 1; r < 4; ++r) {
      double f = A[r * 4 + col] / A[col * 4 + col];
      for (int q = col; q < 4; ++q) A[r * 4 + q] -= f * A[col * 4 + q];
      B[r] -= f * B[col];
    }
  }
  for (int r = 3; r >= 0; --r) {
    double v = B[r];
    for (int q = r + 1; q < 4; ++q) v -= A[r * 4 + q] * k[q];
    k[r] = v / A[r * 4 + r];
  }
  return true;
}

// RMS of a cubic fit in the chord frame, and the worst interior point: a distorted line fits
// a cubic, a shallow kink doesn't.
double CubicResidual(const Chain& c, size_t a, size_t b, size_t& worst) {
  const double x0 = c[a].x, y0 = c[a].y;
  const double dx = c[b].x - x0, dy = c[b].y - y0, L = std::hypot(dx, dy);
  worst = (a + b) / 2;
  if (L < 1e-9 || b < a + 6) return 0;
  const double ux = dx / L, uy = dy / L;
  const size_t n = b - a + 1;
  std::vector<double> T(n), Y(n);
  for (size_t i = 0; i < n; ++i) {
    double px = c[a + i].x - x0, py = c[a + i].y - y0;
    T[i] = (px * ux + py * uy) / L - 0.5;  // [-0.5, 0.5] for conditioning
    Y[i] = -px * uy + py * ux;
  }
  double k[4];
  if (!FitCubic(T, Y, k)) return 0;
  double ss = 0, best = -1;
  for (size_t i = 0; i < n; ++i) {
    double t = T[i], e = Y[i] - (k[0] + k[1] * t + k[2] * t * t + k[3] * t * t * t);
    ss += e * e;
    // interior points only, so a split always leaves two non-empty pieces
    if (i > 2 && i + 3 < n && std::fabs(e) > best) { best = std::fabs(e); worst = a + i; }
  }
  return std::sqrt(ss / n);
}

void SplitSmooth(const Chain& c, size_t a, size_t b, double minLen, std::vector<Range>& out) {
  if (Dist(c[a], c[b]) < minLen) return;
  size_t worst;
  if (CubicResidual(c, a, b, worst) <= 0.25) { out.push_back({a, b}); return; }
  SplitSmooth(c, a, worst, minLen, out);
  SplitSmooth(c, worst, b, minLen, out);
}

struct Candidate {
  double len;
  Chain pts;
};

std::vector<Candidate> LineCandidates(const std::vector<Chain>& chains, double minLen) {
  std::vector<Candidate> cands;
  for (const Chain& c : chains) {
    std::vector<Range> corners, pieces;
    Split(c, 0, c.size() - 1, corners);
    for (auto [a, b] : corners) SplitSmooth(c, a, b, minLen, pieces);
    for (auto [a, b] : pieces) {
      const double L = Dist(c[a], c[b]);
      if (L < minLen) continue;
      if (PathLength(c, a, b) > 1.12 * L) continue;  // wiggly (foliage, texture), not a line image
      cands.push_back({L, Chain(c.begin() + a, c.begin() + b + 1)});
    }
  }
  return cands;
}

// 24 points along the arc, in plate pixels.
Segment Resample(const Chain& c, int f) {
  std::vector<double> acc(c.size(), 0.0);
  for (size_t i = 1; i < c.size(); ++i) acc[i] = acc[i - 1] + Dist(c[i - 1], c[i]);
  const int K = 24;
  Segment s;
  size_t j = 0;
  for (int k = 0; k < K; ++k) {
    double t = acc.back() * k / (K - 1);
    while (j + 1 < acc.size() && acc[j + 1] < t) ++j;
    size_t j1 = std::min(j + 1, acc.size() - 1);
    double seg = acc[j1] - acc[j], a = seg > 0 ? (t - acc[j]) / seg : 0;
    double x = c[j].x + a * (c[j1].x - c[j].x);
    double y = c[j].y + a * (c[j1].y - c[j].y);
    s.xy.push_back(float((x + 0.5) * f));
    s.xy.push_back(float((y + 0.5) * f));
  }
  return s;
}

}  // namespace

std::vector<Segment> DetectSegments(const float* gray, int W, int H, int maxSegments) {
  std::vector<Segment> result;
  if (W < 16 || H < 16) return result;

  const int f = std::max(1, int(std::ceil(std::max(W, H) / 2048.0)));
  const Img g = Blur(Downsample(gray, W, H, f));

  Edges e = Suppress(Sobel(g));
  if (e.strong.size() < 100) return result;

  const std::vector<Chain> chains = LinkChains(Hysteresis(e.nms, e.strong), e);
  std::vector<Candidate> cands = LineCandidates(chains, 0.06 * std::max(g.w, g.h));
  std::sort(cands.begin(), cands.end(),
            [](const Candidate& a, const Candidate& b) { return a.len > b.len; });
  if (int(cands.size()) > maxSegments) cands.resize(maxSegments);

  for (const Candidate& c : cands) result.push_back(Resample(c.pts, f));
  return result;
}

namespace {

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
  double angle() const { return 0.5 * std::atan2(2 * sxy, sxx - syy); }  // of the TLS line
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

// Total-least-squares line through the undistorted points, mapped back into the plate.
std::vector<double> PredictedLine(const std::vector<double>& und, const NukeLD& m) {
  const size_t n = und.size() / 2;
  const Scatter sc(und);
  const double th = sc.angle();
  const double dx = std::cos(th), dy = std::sin(th);
  double t0 = 1e30, t1 = -1e30;
  for (size_t i = 0; i < n; ++i) {
    double t = (und[2 * i] - sc.mx) * dx + (und[2 * i + 1] - sc.my) * dy;
    t0 = std::min(t0, t); t1 = std::max(t1, t);
  }
  std::vector<double> line;
  for (int k = 0; k <= 24; ++k) {
    double t = t0 + (t1 - t0) * k / 24.0, xd, yd;
    m.distort(sc.mx + t * dx, sc.my + t * dy, xd, yd);
    line.push_back(xd);
    line.push_back(yd);
  }
  return line;
}

// Lengths in px of a 1280-wide plate, the scale the constants were tuned at.
const double kRefPx = 640;
const double kMinChord = 150 / kRefPx;  // shorter segments bend too little to read
const double kMinOffAxis = 0.4;         // segment midpoint from the centre, normalised
const double kMaxResidual = 2.0;        // px from the segment's line: arcs aren't lines
const double kMinSensitivity = 0.03;    // px of bow per 0.01 of k1; radial lines barely react
const double kHuber = 0.3;              // px; larger bows are mostly curved scene edges
const double kPrior = 0.003;            // weight of the estimate, per px^2 of map change

struct Line {
  std::vector<double> xy;  // normalised, about the distortion centre
  double mx, my, ux, uy, len;
  double residual = 0;  // largest distance of a point from the line
  double weight = 0;
};

Line MakeLine(std::vector<double> xy) {
  Line l;
  const size_t n = xy.size() / 2;
  const Scatter sc(xy);
  const double th = sc.angle();
  l.mx = sc.mx;
  l.my = sc.my;
  l.ux = std::cos(th);
  l.uy = std::sin(th);
  l.len = std::hypot(xy[2 * n - 2] - xy[0], xy[2 * n - 1] - xy[1]);
  for (size_t i = 0; i < n; ++i) {
    const double d = (xy[2 * i + 1] - l.my) * l.ux - (xy[2 * i] - l.mx) * l.uy;
    l.residual = std::max(l.residual, std::fabs(d));
  }
  l.xy = std::move(xy);
  return l;
}

// Signed bow (px) of points about their total-least-squares line, from a parabola across
// their span; > 0 bulges away from the centre (barrel).
double Bow(const std::vector<double>& xy) {
  const size_t n = xy.size() / 2;
  const Scatter sc(xy);
  const double th = sc.angle();
  const double ux = std::cos(th), uy = std::sin(th);
  const double sn = sc.mx * -uy + sc.my * ux < 0 ? -1 : 1;
  std::vector<double> t(n), d(n);
  double t0 = 1e30, t1 = -1e30;
  for (size_t i = 0; i < n; ++i) {
    const double dx = xy[2 * i] - sc.mx, dy = xy[2 * i + 1] - sc.my;
    t[i] = dx * ux + dy * uy;
    d[i] = sn * (dy * ux - dx * uy);
    t0 = std::min(t0, t[i]);
    t1 = std::max(t1, t[i]);
  }
  const double mid = 0.5 * (t0 + t1), half = 0.5 * (t1 - t0);
  if (half <= 0) return 0;
  // d = a + b u + c u^2 with u in [-1, 1], so the bow at the ends is -c.
  double S[5] = {0}, B[3] = {0};
  for (size_t i = 0; i < n; ++i) {
    const double u = (t[i] - mid) / half;
    double p = 1;
    for (int k = 0; k < 5; ++k, p *= u) {
      S[k] += p;
      if (k < 3) B[k] += d[i] * p;
    }
  }
  const double det = S[0] * (S[2] * S[4] - S[3] * S[3]) - S[1] * (S[1] * S[4] - S[2] * S[3]) +
                     S[2] * (S[1] * S[3] - S[2] * S[2]);
  if (std::fabs(det) < 1e-12) return 0;
  const double c = (S[0] * (S[2] * B[2] - S[3] * B[1]) - S[1] * (S[1] * B[2] - S[2] * B[1]) +
                    B[0] * (S[1] * S[3] - S[2] * S[2])) / det;
  return -c * kRefPx;
}

// Bow after undistortion, scaled back to the plate by the chord ratio: a bow measured on the
// undistorted points shrinks with the image, which would favour lenses that shrink it.
double BowUnder(const Line& l, double k1, double k2) {
  std::vector<double> u(l.xy.size());
  for (size_t i = 0; i + 1 < u.size(); i += 2) {
    const double x = l.xy[i], y = l.xy[i + 1], r2 = x * x + y * y;
    const double D = 1 + k1 * r2 + k2 * r2 * r2;
    u[i] = x / D;
    u[i + 1] = y / D;
  }
  const size_t n = u.size();
  const double chord = std::hypot(u[n - 2] - u[0], u[n - 1] - u[1]);
  return chord > 0 ? Bow(u) * l.len / chord : 0;
}

// Bent pieces are halved, as a line-segment detector would, so arcs don't pass as lines.
void AddPieces(const std::vector<double>& xy, size_t a, size_t b, std::vector<Line>& out) {
  if (b < a + 5) return;
  Line l = MakeLine(std::vector<double>(xy.begin() + 2 * a, xy.begin() + 2 * b + 2));
  if (l.len < kMinChord) return;
  if (l.residual * kRefPx > kMaxResidual) {
    const size_t m = a + (b - a + 1) / 2;
    AddPieces(xy, a, m, out);
    AddPieces(xy, m, b, out);
    return;
  }
  const double cx = 0.5 * (xy[2 * a] + xy[2 * b]), cy = 0.5 * (xy[2 * a + 1] + xy[2 * b + 1]);
  if (std::hypot(cx, cy) >= kMinOffAxis) out.push_back(std::move(l));
}

// Long, off-axis, nearly straight pieces. A line seen twice (both sides of it, several frames)
// counts once.
std::vector<Line> CandidateLines(const std::vector<const Segment*>& segs, const NukeLD& m) {
  std::vector<Line> all;
  for (const Segment* s : segs) {
    const size_t n = s->xy.size() / 2;
    std::vector<double> xy(2 * n);
    for (size_t i = 0; i < n; ++i) {
      xy[2 * i] = (s->xy[2 * i] - m.centreX()) * m.pa / m.s();
      xy[2 * i + 1] = (m.H - s->xy[2 * i + 1] - m.centreY()) / m.s();
    }
    if (n) AddPieces(xy, 0, n - 1, all);
  }
  std::sort(all.begin(), all.end(), [](const Line& a, const Line& b) { return a.len > b.len; });
  std::vector<Line> kept;
  for (Line& l : all) {
    bool dup = false;
    for (const Line& k : kept) {
      const double dx = k.mx - l.mx, dy = k.my - l.my;
      if (std::fabs(l.ux * k.ux + l.uy * k.uy) > 0.999 &&
          std::fabs(l.ux * dy - l.uy * dx) < 5 / kRefPx &&
          std::fabs(dx * l.ux + dy * l.uy) < 0.35 * (l.len + k.len)) {
        dup = true;
        break;
      }
    }
    if (!dup) kept.push_back(std::move(l));
  }
  return kept;
}

}  // namespace

LineRefinement RefineWithLines(const std::vector<const Segment*>& segs, const NukeLD& m) {
  LineRefinement out{m, 0};
  std::vector<Line> lines;
  for (Line& l : CandidateLines(segs, m)) {
    const double sens = 5 * std::fabs(BowUnder(l, 0.001, 0) - BowUnder(l, -0.001, 0));
    if (sens < kMinSensitivity) continue;
    l.weight = std::min(1.0, sens / 0.3);
    lines.push_back(std::move(l));
  }
  out.lines = int(lines.size());
  if (lines.size() < 2) return out;

  const double p[2] = {m.k1, m.k2};
  // Prior metric: mean squared change of the undistortion map (px) per change of k, on a
  // 33 x 21 grid over the plate.
  double R[3] = {0, 0, 0};  // xx, xy, yy
  for (int j = 0; j < 21; ++j)
    for (int i = 0; i < 33; ++i) {
      const double x = ((0.5 + (m.W - 1.0) * i / 32) - m.centreX()) * m.pa / m.s();
      const double y = ((0.5 + (m.H - 1.0) * j / 20) - m.centreY()) / m.s();
      const double r2 = x * x + y * y, D = 1 + p[0] * r2 + p[1] * r2 * r2;
      for (double c : {x, y}) {
        const double a = c * r2 / (D * D) * kRefPx, b = a * r2;
        R[0] += a * a;
        R[1] += a * b;
        R[2] += b * b;
      }
    }
  for (double& r : R) r /= 33 * 21;

  // Keep the undistortion defined and monotonic out to the farthest corner.
  double r2max = 0;
  for (double X : {0.0, double(m.W)})
    for (double Y : {0.0, double(m.H)})
      r2max = std::max(r2max, std::pow((X - m.centreX()) * m.pa / m.s(), 2) +
                                  std::pow((Y - m.centreY()) / m.s(), 2));
  auto objective = [&](const double k[2]) -> double {
    for (int i = 0; i < 30; ++i) {
      const double r2 = r2max * i / 29;
      if (1 + k[0] * r2 + k[1] * r2 * r2 < 0.5 || 1 - k[0] * r2 - 3 * k[1] * r2 * r2 < 0.2)
        return INFINITY;
    }
    double e = 0;
    for (const Line& l : lines) {
      const double b = std::fabs(BowUnder(l, k[0], k[1]));
      e += l.weight * (b <= kHuber ? b * b : 2 * kHuber * b - kHuber * kHuber);
    }
    const double d0 = k[0] - p[0], d1 = k[1] - p[1];
    return e + kPrior * (R[0] * d0 * d0 + 2 * R[1] * d0 * d1 + R[2] * d1 * d1);
  };

  // Gauss-Newton on the Huber-weighted bows, with a backtracking line search.
  double k[2] = {p[0], p[1]};
  for (int it = 0; it < 6; ++it) {
    double A[3] = {kPrior * R[0], kPrior * R[1], kPrior * R[2]};
    const double d0 = k[0] - p[0], d1 = k[1] - p[1];
    double g[2] = {kPrior * (R[0] * d0 + R[1] * d1), kPrior * (R[1] * d0 + R[2] * d1)};
    const double h = 1e-4;
    for (const Line& l : lines) {
      const double b = BowUnder(l, k[0], k[1]);
      const double j0 = (BowUnder(l, k[0] + h, k[1]) - BowUnder(l, k[0] - h, k[1])) / (2 * h);
      const double j1 = (BowUnder(l, k[0], k[1] + h) - BowUnder(l, k[0], k[1] - h)) / (2 * h);
      const double w = l.weight * std::min(1.0, kHuber / std::max(std::fabs(b), 1e-8));
      A[0] += w * j0 * j0;
      A[1] += w * j0 * j1;
      A[2] += w * j1 * j1;
      g[0] += w * j0 * b;
      g[1] += w * j1 * b;
    }
    A[0] += 1e-8;
    A[2] += 1e-8;
    const double det = A[0] * A[2] - A[1] * A[1];
    double step[2] = {-(A[2] * g[0] - A[1] * g[1]) / det, -(A[0] * g[1] - A[1] * g[0]) / det};
    const double big = std::max(std::fabs(step[0]), std::fabs(step[1]));
    if (big > 0.05) { step[0] *= 0.05 / big; step[1] *= 0.05 / big; }
    const double e0 = objective(k);
    double next[2];
    bool moved = false;
    for (double sc = 1; sc > 0.06; sc *= 0.5) {
      next[0] = k[0] + sc * step[0];
      next[1] = k[1] + sc * step[1];
      if (objective(next) < e0) { moved = true; break; }
    }
    if (!moved) break;
    const double dk = std::hypot(next[0] - k[0], next[1] - k[1]);
    k[0] = next[0];
    k[1] = next[1];
    if (dk < 1e-6) break;
  }
  out.lens.k1 = k[0];
  out.lens.k2 = k[1];
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
  o.predicted = PredictedLine(und, m);
  return o;
}

}  // namespace anycalib
