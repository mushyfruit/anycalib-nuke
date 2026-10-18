// Copyright the anycalib-nuke authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "DDImage/Executable.h"
#include "DDImage/Interest.h"
#include "DDImage/Iop.h"
#include "DDImage/Knobs.h"
#include "DDImage/MetaData.h"
#include "DDImage/Row.h"
#include "DDImage/ViewerContext.h"
#include "DDImage/gl.h"

#include "AnyCalibEngine.h"
#include "CameraFit.h"
#include "LensModel.h"
#include "LineDetect.h"
#include "ShotAnalysis.h"

#include <torch/version.h>

using namespace DD::Image;
using namespace anycalib;

static const char* const CLASS = "AnyCalib";
static const char* const HELP =
    "AnyCalib " ANYCALIB_NUKE_VERSION "\n\n"
    "Estimates lens distortion from the plate with the AnyCalib network "
    "(github.com/javrtg/AnyCalib).\n"
    "  forward.u/v   undistort STMap (as LensDistortion2, output STMap)\n"
    "  backward.u/v  redistort STMap\n"
    "  lens/*        metadata\n\n"
    "distortion1/2 and distortionCenter are Foundry's NukeX Classic values, as "
    "CameraTracker uses them.";

namespace {

const ChannelSet kMapChannels(Mask_U | Mask_V | Mask_Backward_U | Mask_Backward_V);

const char* const kDeviceNames[] = {"auto (CUDA, else CPU)", "CPU", "CUDA", nullptr};
const Device kDevices[] = {Device::Auto, Device::CPU, Device::CUDA};

const char* const kMaskNames[] = {"None", "Source Alpha", "Source Inverted Alpha", "Mask Luminance",
                                  "Mask Inverted Luminance", "Mask Alpha", "Mask Inverted Alpha",
                                  nullptr};
enum MaskMode {
  kNoMask,
  kSourceAlpha,
  kSourceInvertedAlpha,
  kMaskLuminance,
  kMaskInvertedLuminance,
  kMaskAlpha,
  kMaskInvertedAlpha,
};

inline float SrgbOetf(float c) {
  c = std::min(std::max(c, 0.0f), 1.0f);
  return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

inline float Luma(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

std::vector<float> Luminance(const std::vector<float>& rgb) {
  std::vector<float> gray(rgb.size() / 3);
  for (size_t i = 0; i < gray.size(); ++i) gray[i] = Luma(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]);
  return gray;
}

NukeLD LensFor(const Format& fmt, double k1, double k2, double cx, double cy) {
  NukeLD m;
  m.W = fmt.width();
  m.H = fmt.height();
  m.pa = fmt.pixel_aspect();
  m.k1 = k1;
  m.k2 = k2;
  m.cx = cx;
  m.cy = cy;
  return m;
}

// Shared by every node, and leaked: destroying it at exit, after the CUDA driver, aborts Nuke.
std::mutex gEngineMu;
std::unique_ptr<AnyCalibEngine>& gEngine = *new std::unique_ptr<AnyCalibEngine>();
std::string gEngineKey;

AnyCalibEngine* GetEngine(const std::string& model, const std::string& digest, Device device,
                          std::string& err) {
  const std::string key = digest + "|" + std::to_string(int(device));
  if (gEngine && gEngineKey == key) return gEngine.get();
  gEngine.reset();  // free the old network before loading the new one
  gEngine = LoadEngine(model, device, &err);
  gEngineKey = key;
  return gEngine.get();
}

// Analysis state, on firstOp(): Nuke executes each frame on its own op.
struct Accumulator {
  ShotAnalysis shot;
  std::string device, digest, model;
  Device requestedDevice = Device::Auto;
  bool lockCentre = false, refineLines = true;
};

// The 90th percentile of up to ~1M unmasked pixels exposed to 0.8, as AnyCalib's training
// panoramas were, then the sRGB curve.
void ExposeToSrgb(std::vector<float>& img, const std::vector<uint8_t>& excluded) {
  const size_t n = img.size() / 3, step = std::max<size_t>(1, n >> 20);
  std::vector<float> s;
  s.reserve(3 * (n / step + 1));
  for (size_t i = 0; i < n; i += step)
    if (excluded.empty() || !excluded[i]) s.insert(s.end(), &img[3 * i], &img[3 * i + 3]);
  float k = 1;
  if (!s.empty()) {
    auto p90 = s.begin() + s.size() * 9 / 10;
    std::nth_element(s.begin(), p90, s.end());
    k = 0.8f / (*p90 + 1e-10f);
  }
  for (float& v : img) v = SrgbOetf(k * v);
}

// Largest distance (px) between a plate corner's undistorted positions under two lenses.
double CornerChange(const NukeLD& a, const NukeLD& b) {
  double d = 0;
  for (double x : {0.0, double(a.W)})
    for (double y : {0.0, double(a.H)}) {
      double ax, ay, bx, by;
      if (a.undistort(x, y, ax, ay) && b.undistort(x, y, bx, by))
        d = std::max(d, std::hypot(ax - bx, ay - by));
    }
  return d;
}

void DropExcluded(std::vector<RaySample>& samples, const std::vector<uint8_t>& excluded, int W,
                  int H) {
  if (excluded.empty()) return;
  auto masked = [&](const RaySample& s) {
    return excluded[size_t(std::min(H - 1, int(s.v))) * W + std::min(W - 1, int(s.u))] != 0;
  };
  samples.erase(std::remove_if(samples.begin(), samples.end(), masked), samples.end());
}

// linesData: "W <w> H <h>\nF <frame>\nS x y x y ...", plate pixels, y down.
std::string SerialiseLines(int W, int H, const std::vector<FrameResult>& frames, size_t stride) {
  std::ostringstream o;
  o << "W " << W << " H " << H << "\n";
  char b[32];
  for (size_t i = 0; i < frames.size(); i += stride) {
    const FrameResult& f = frames[i];
    o << "F " << f.frame << "\n";
    for (const Segment& s : f.segs) {
      o << "S";
      for (float v : s.xy) { std::snprintf(b, sizeof(b), " %.1f", v); o << b; }
      o << "\n";
    }
  }
  return o.str();
}

struct ParsedLines {
  int W = 0, H = 0;
  std::map<double, std::vector<Segment>> frames;
};

ParsedLines ParseLines(const char* text) {
  ParsedLines p;
  if (!text) return p;
  std::istringstream in(text);
  std::string line;
  std::vector<Segment>* cur = nullptr;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream ls(line.substr(1));
    if (line[0] == 'W') {
      std::string h;
      ls >> p.W >> h >> p.H;
    } else if (line[0] == 'F') {
      double f;
      ls >> f;
      cur = &p.frames[f];
    } else if (line[0] == 'S' && cur) {
      Segment s;
      float v;
      while (ls >> v) s.xy.push_back(v);
      if (s.xy.size() >= 6 && s.xy.size() % 2 == 0) cur->push_back(std::move(s));
    }
  }
  return p;
}

const auto& NearestFrame(const ParsedLines& p, double frame) {
  auto best = p.frames.begin();
  for (auto it = p.frames.begin(); it != p.frames.end(); ++it)
    if (std::fabs(it->first - frame) < std::fabs(best->first - frame)) best = it;
  return *best;
}

void DrawPolyline(const std::vector<double>& xy) {
  glBegin(GL_LINE_STRIP);
  for (size_t i = 0; i + 1 < xy.size(); i += 2) glVertex2d(xy[i], xy[i + 1]);
  glEnd();
}

void DrawOverlayLine(const OverlayLine& o) {
  const std::vector<double>& pts = o.plate;
  glLineWidth(2.0f);
  o.straight ? glColor3f(0.25f, 1.0f, 0.4f) : glColor3f(1.0f, 0.35f, 0.25f);
  DrawPolyline(pts);

  glLineWidth(1.0f);
  glColor3f(0.2f, 0.85f, 1.0f);
  DrawPolyline(o.predicted);

  glEnable(GL_LINE_STIPPLE);
  glLineStipple(2, 0xAAAA);
  glColor3f(0.6f, 0.6f, 0.6f);
  DrawPolyline({pts[0], pts[1], pts[pts.size() - 2], pts[pts.size() - 1]});
  glDisable(GL_LINE_STIPPLE);
}

}  // namespace

class AnyCalibOp : public Iop, public Executable {
  const char* _model = nullptr;
  int _device = 0;
  double _matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // working space -> linear Rec.709, row-major
  bool _useInputRange = true;
  int _first = 1, _last = 100;
  int _maxFrames = 24;
  int _ignore = kNoMask;
  bool _lockCentre = false;
  bool _refineLines = true;
  double _k1 = 0, _k2 = 0;
  double _centre[2] = {0, 0};
  double _focal = 0;
  double _rmsDeg = 0;
  double _frameSpread = 0;
  int _linesUsed = 0;
  double _lineCorrection = 0;
  int _framesAnalysed = 0;
  const char* _framesInfo = nullptr;
  const char* _estimatorVersion = nullptr;
  const char* _modelFile = nullptr;
  const char* _modelDigest = nullptr;
  // Pending request; provenance above is committed only after a complete analysis.
  const char* _analysisModel = nullptr;
  const char* _analysisDigest = nullptr;
  int _analysisFrameCount = 0;
  double _overscan[4] = {0, 0, 0, 0};  // left, bottom, right, top (px)
  bool _writeMaps = false;
  bool _showLines = true;
  const char* _lines = nullptr;

  NukeLD _lens;
  std::shared_ptr<Accumulator> _acc;
  MetaData::Bundle _meta;
  std::string _parsedFrom;
  ParsedLines _parsed;

 public:
  static const Description description;

  explicit AnyCalibOp(Node* node)
      : Iop(node), Executable(this), _acc(std::make_shared<Accumulator>()) {}

  const char* Class() const override { return CLASS; }
  const char* node_help() const override { return HELP; }
  int minimum_inputs() const override { return 1; }
  int maximum_inputs() const override { return 2; }
  const char* input_label(int n, char*) const override { return n == 0 ? "plate" : "mask"; }

  Executable* executable() override { return this; }
  bool isExecuteThreadSafe() const override { return false; }  // one shared network
  // Scanlines pass through as in a PixelIop, so top-down rendering needn't cache them here.
  OpHints opHints() const override { return OpHints::eChainable; }

  void knobs(Knob_Callback f) override;
  int knob_changed(Knob* k) override;

  void _validate(bool) override;
  void _request(int x, int y, int r, int t, ChannelMask channels, int count) override;
  void engine(int y, int x, int r, ChannelMask channels, Row& row) override;
  const MetaData::Bundle& _fetchMetaData(const char* key) override;

  void beginExecuting() override;
  void execute() override;
  void endExecuting() override;

  HandlesMode doAnyHandles(ViewerContext* ctx) override;
  void build_handles(ViewerContext* ctx) override;
  void draw_handle(ViewerContext* ctx) override;

 private:
  void updateOverscan();
  void storeOverscan(const NukeLD& m);
  void setKnob(const char* name, double v, int idx = -1) { knob(name)->set_value(v, idx); }
  void setText(const char* name, const std::string& v) { knob(name)->set_text(v.c_str()); }

  Accumulator& acc() { return *static_cast<AnyCalibOp*>(firstOp())->_acc; }
  template <class F>
  void readRows(Iop& in, int W, int H, const ChannelSet& chans, F onRow);
  std::vector<float> readPlate(Iop& in, int W, int H);
  std::vector<uint8_t> readMask(int W, int H);
  bool infer(const std::vector<float>& img, int W, int H, double pa,
             std::vector<RaySample>& samples);
  bool fitLens(const std::vector<RaySample>& samples, int W, int H, double pa, NukeLD& nk,
               double& focal, FitStats* stats, std::string& err);
};

static Op* build(Node* node) { return new AnyCalibOp(node); }
const Op::Description AnyCalibOp::description(CLASS, build);

// NDK code can't create nodes, so the buttons run python/anycalib_nuke.py.
static const char* const kAnalyzeScript =
    "import anycalib_nuke\nanycalib_nuke.analyze(nuke.thisNode())\n";
static const char* const kUndistortScript =
    "import anycalib_nuke\nanycalib_nuke.create_lens_distortion(nuke.thisNode(), True)\n";
static const char* const kRedistortScript =
    "import anycalib_nuke\nanycalib_nuke.create_lens_distortion(nuke.thisNode(), False)\n";

void AnyCalibOp::knobs(Knob_Callback f) {
  Tab_knob(f, "AnyCalib");

  Divider(f, "Analysis");
  // Analysis settings don't change the output, so they don't re-render.
  Bool_knob(f, &_useInputRange, "useInputRange", "use input frame range");
  SetFlags(f, Knob::STARTLINE | Knob::NO_RERENDER);
  Int_knob(f, &_first, "analysisFirst", "range");
  ClearFlags(f, Knob::STARTLINE);
  SetFlags(f, Knob::NO_RERENDER);
  Int_knob(f, &_last, "analysisLast", "");
  ClearFlags(f, Knob::STARTLINE);
  SetFlags(f, Knob::NO_RERENDER);
  Int_knob(f, &_maxFrames, "maxFrames", "max frames");
  SetFlags(f, Knob::NO_RERENDER);
  Tooltip(f, "Frames sampled evenly across the range and pooled into one fit. The lens is "
             "assumed fixed over the range.");
  Enumeration_knob(f, &_ignore, kMaskNames, "ignore", "mask");
  SetFlags(f, Knob::NO_RERENDER);
  Tooltip(f, "Regions to leave out of the analysis, such as moving objects, burn-ins or "
             "letterbox bars. White is left out, as with CameraTracker's mask.");
  Bool_knob(f, &_lockCentre, "lockCentre", "lock centre");
  SetFlags(f, Knob::STARTLINE | Knob::NO_RERENDER);
  Tooltip(f, "Keep distortion center at 0, the middle of the format, and fit only distortion 1 "
             "and 2. Steadier when the distortion is mild.");
  Bool_knob(f, &_refineLines, "refineLines", "refine with lines");
  SetFlags(f, Knob::NO_RERENDER);
  Tooltip(f, "Adjust distortion 1 and 2 so that long straight edges in the plate come out "
             "straight, staying close to the network's estimate. Without two usable edges the "
             "network's estimate is kept.");
  PyScript_knob(f, kAnalyzeScript, "analyze", "Analyze");
  SetFlags(f, Knob::STARTLINE);

  const auto resultFlags = Knob::READ_ONLY | Knob::NO_RERENDER | Knob::NO_ANIMATION;
  Divider(f, "Distortion");
  Double_knob(f, &_k1, "distortion1", "distortion 1");
  ClearFlags(f, Knob::SLIDER);
  Tooltip(f, "CameraTracker distortion1 / LensDistortion2 distortionDenominator0. "
             "Negative is barrel.");
  Double_knob(f, &_k2, "distortion2", "distortion 2");
  ClearFlags(f, Knob::SLIDER);
  Tooltip(f, "CameraTracker distortion2 / LensDistortion2 distortionDenominator1.");
  MultiFloat_knob(f, _centre, 2, "distortionCenter", "distortion center");
  ClearFlags(f, Knob::SLIDER);
  Tooltip(f, "CameraTracker distortionCenter / LensDistortion2 centre.");
  MultiFloat_knob(f, _overscan, 4, "overscan", "overscan (l b r t px)");
  SetFlags(f, resultFlags);
  ClearFlags(f, Knob::SLIDER);
  Tooltip(f, "How far the undistorted plate extends past the format: the overscan CG needs.");

  Divider(f, "Checks");
  Double_knob(f, &_rmsDeg, "rmsDegrees", "fit RMS (deg)");
  SetFlags(f, resultFlags);
  ClearFlags(f, Knob::SLIDER);
  Tooltip(f, "RMS angle between the network's rays and the fitted lens.");
  Double_knob(f, &_frameSpread, "frameSpread", "frame spread (px)");
  SetFlags(f, resultFlags);
  ClearFlags(f, Knob::SLIDER);
  Tooltip(f, "Largest standard deviation of undistorted positions on a grid across the plate. "
             "Includes displacement direction and centre changes. 0 with one frame.");
  Int_knob(f, &_linesUsed, "linesUsed", "lines used");
  SetFlags(f, resultFlags);
  Tooltip(f, "Edges that refined the estimate: long, nearly straight and away from the centre, "
             "where distortion bends them. Below 2, the network's estimate is used as it is.");
  Double_knob(f, &_lineCorrection, "lineCorrection", "line correction (px)");
  SetFlags(f, resultFlags);
  ClearFlags(f, Knob::SLIDER);
  Tooltip(f, "How far the lines moved the network's estimate: the largest change of a plate "
             "corner's undistorted position.");

  Divider(f, "Output");
  Bool_knob(f, &_writeMaps, "writeMaps", "write STMaps (forward.u/v undistort, backward.u/v redistort)");
  SetFlags(f, Knob::STARTLINE);
  Tooltip(f, "Also grows the bbox to cover the overscan, so leave it off when the node is inline "
             "and only the values or Create LensDistortion are needed.");
  Bool_knob(f, &_showLines, "showLines", "show detected lines");
  SetFlags(f, Knob::STARTLINE);
  Tooltip(f, "Green: straight under the estimate. Red: not. Cyan: the estimate's image of the "
             "best straight line. Grey: the chord.");

  Divider(f, "Export");
  PyScript_knob(f, kUndistortScript, "createUndistort", "Create LensDistortion (undistort)");
  SetFlags(f, Knob::STARTLINE);
  PyScript_knob(f, kRedistortScript, "createRedistort", "Create LensDistortion (redistort)");

  Tab_knob(f, "Advanced");
  File_knob(f, &_model, "model", "model override");
  SetFlags(f, Knob::NO_RERENDER);
  Tooltip(f, "TorchScript network from tools/export_anycalib.py. Empty: $ANYCALIB_MODEL_PATH, "
             "then models/anycalib_dist.pt next to the plugin.");
  Enumeration_knob(f, &_device, kDeviceNames, "device", "device");
  SetFlags(f, Knob::NO_RERENDER);

  // Saved with the analysis, not shown.
  MultiFloat_knob(f, _matrix, 9, "inputMatrix", INVISIBLE);
  SetFlags(f, Knob::NO_RERENDER);
  Double_knob(f, &_focal, "focalPx", INVISIBLE);
  Int_knob(f, &_framesAnalysed, "framesAnalysed", INVISIBLE);
  String_knob(f, &_framesInfo, "framesInfo", INVISIBLE);
  String_knob(f, &_estimatorVersion, "estimatorVersion", INVISIBLE);
  String_knob(f, &_modelFile, "modelFile", INVISIBLE);
  String_knob(f, &_modelDigest, "modelSha256", INVISIBLE);
  String_knob(f, &_analysisModel, "analysisModelFile", INVISIBLE);
  SetFlags(f, Knob::NO_RERENDER);
  String_knob(f, &_analysisDigest, "analysisModelSha256", INVISIBLE);
  SetFlags(f, Knob::NO_RERENDER);
  Int_knob(f, &_analysisFrameCount, "analysisFrameCount", INVISIBLE);
  SetFlags(f, Knob::NO_RERENDER);
  Multiline_String_knob(f, &_lines, "linesData", INVISIBLE);
  SetFlags(f, Knob::NO_RERENDER | Knob::NO_UNDO);
}

int AnyCalibOp::knob_changed(Knob* k) {
  if (k->is("showPanel") || k->is("inputChange") || k->is("useInputRange") ||
      k->is("distortion1") || k->is("distortion2") || k->is("distortionCenter")) {
    knob("analysisFirst")->enable(!_useInputRange);
    knob("analysisLast")->enable(!_useInputRange);
    updateOverscan();
    return 1;
  }
  return Iop::knob_changed(k);
}

void AnyCalibOp::updateOverscan() {
  Iop* in = dynamic_cast<Iop*>(input(0));
  if (!in) return;
  in->validate(false);
  const NukeLD m = LensFor(in->format(), knob("distortion1")->get_value(),
                           knob("distortion2")->get_value(), knob("distortionCenter")->get_value(0),
                           knob("distortionCenter")->get_value(1));
  if (m.W > 0 && m.H > 0) storeOverscan(m);
}

void AnyCalibOp::storeOverscan(const NukeLD& m) {
  const Extent e = UndistortedExtent(m);
  setKnob("overscan", std::max(0.0, -e.x), 0);
  setKnob("overscan", std::max(0.0, -e.y), 1);
  setKnob("overscan", std::max(0.0, e.r - m.W), 2);
  setKnob("overscan", std::max(0.0, e.t - m.H), 3);
}

void AnyCalibOp::_validate(bool) {
  copy_info();
  _lens = LensFor(input0().format(), _k1, _k2, _centre[0], _centre[1]);
  if (!_writeMaps) {
    set_out_channels(Mask_None);
    return;
  }
  ChannelSet out = input0().info().channels();
  out += kMapChannels;
  info_.channels(out);
  set_out_channels(kMapChannels);

  // LensDistortion2's Auto STMap bbox, also covering the whole format, which its
  // square can miss with non-square pixels.
  Box b(input0().info().box());
  if (_lens.W > 0 && _lens.H > 0) {
    const PixelBox u = AutoBBox(UndistortedExtent(_lens)), d = AutoBBox(RedistortedExtent(_lens));
    b.merge(u.x, u.y, u.r, u.t);
    b.merge(d.x, d.y, d.r, d.t);
    b.merge(0, 0, _lens.W, _lens.H);
  }
  info_.set(b);
}

void AnyCalibOp::_request(int x, int y, int r, int t, ChannelMask channels, int count) {
  ChannelSet in(channels);
  in -= kMapChannels;
  in &= input0().info().channels();
  input0().request(x, y, r, t, in, count);
}

void AnyCalibOp::engine(int y, int x, int r, ChannelMask channels, Row& row) {
  ChannelSet img(channels);
  img -= kMapChannels;
  if (img) row.get(input0(), y, x, r, img);
  if (!(channels & kMapChannels)) return;

  const int W = std::max(1, _lens.W), H = std::max(1, _lens.H);
  float* fu = channels.contains(Chan_U) ? row.writable(Chan_U) : nullptr;
  float* fv = channels.contains(Chan_V) ? row.writable(Chan_V) : nullptr;
  float* bu = channels.contains(Chan_Backward_U) ? row.writable(Chan_Backward_U) : nullptr;
  float* bv = channels.contains(Chan_Backward_V) ? row.writable(Chan_Backward_V) : nullptr;
  const double py = y + 0.5;
  // (-1, -1) where the mapping doesn't exist, as LensDistortion2 writes it.
  for (int X = x; X < r; ++X) {
    const double px = X + 0.5;
    if (fu || fv) {
      double sx, sy;
      const bool ok = _lens.distort(px, py, sx, sy);
      if (fu) fu[X] = ok ? float(sx / W) : -1.f;
      if (fv) fv[X] = ok ? float(sy / H) : -1.f;
    }
    if (bu || bv) {
      double sx, sy;
      const bool ok = _lens.undistort(px, py, sx, sy);
      if (bu) bu[X] = ok ? float(sx / W) : -1.f;
      if (bv) bv[X] = ok ? float(sy / H) : -1.f;
    }
  }
}

const MetaData::Bundle& AnyCalibOp::_fetchMetaData(const char* key) {
  _meta = input0().fetchMetaData(key);
  const bool estimated = _framesAnalysed > 0;
  _meta.setData("lens/model", "nukex_classic");
  _meta.setData("lens/k1", _k1);
  _meta.setData("lens/k2", _k2);
  _meta.setData("lens/centre", _centre, 2);
  const double centrePx[2] = {_lens.centreX(), _lens.centreY()};
  _meta.setData("lens/centre_px", centrePx, 2);
  _meta.setData("lens/source", estimated ? "AnyCalib" : "manual");
  if (estimated) {
    _meta.setData("lens/focal_px", _focal);
    _meta.setData("lens/fit_rms_deg", _rmsDeg);
    _meta.setData("lens/frame_spread_px", _frameSpread);
    _meta.setData("lens/lines_used", _linesUsed);
    _meta.setData("lens/line_correction_px", _lineCorrection);
    _meta.setData("lens/frames", _framesInfo ? _framesInfo : "");
    _meta.setData("lens/estimator", "AnyCalib");
    _meta.setData("lens/estimator_version", _estimatorVersion ? _estimatorVersion : "");
    _meta.setData("lens/model_sha256", _modelDigest ? _modelDigest : "");
  }
  _meta.setData("lens/overscan", _overscan, 4);
  return _meta;
}

void AnyCalibOp::beginExecuting() {
  Accumulator& a = acc();
  a = Accumulator();
  // Set by the Analyze button.
  a.shot = ShotAnalysis(size_t(std::max(0, _analysisFrameCount)));
  a.model = _analysisModel ? _analysisModel : "";
  a.digest = _analysisDigest ? _analysisDigest : "";
  a.requestedDevice = kDevices[_device];
  a.lockCentre = _lockCentre;
  a.refineLines = _refineLines;
  if (a.model.empty()) a.shot.fail("run the analysis with the Analyze button");
}

void AnyCalibOp::execute() {
  Accumulator& a = acc();
  if (!a.shot.error().empty()) { error("AnyCalib: %s", a.shot.error().c_str()); return; }
  const double frame = outputContext().frame();
  progressMessage("AnyCalib: reading frame %g", frame);
  progressFraction(0.0);

  Iop& in = input0();
  in.validate(true);
  const Format& fmt = in.format();
  const int W = fmt.width(), H = fmt.height();
  const double pa = fmt.pixel_aspect();
  if (!a.shot.checkFormat(W, H, pa)) {
    error("AnyCalib: %s", a.shot.error().c_str());
    return;
  }

  std::vector<float> img = readPlate(in, W, H);
  if (aborted()) return;
  const std::vector<uint8_t> excluded = readMask(W, H);
  if (aborted()) return;
  ExposeToSrgb(img, excluded);

  progressMessage("AnyCalib: network, frame %g", frame);
  progressFraction(0.35);
  FrameResult fr;
  fr.frame = frame;
  if (!infer(img, W, H, pa, fr.samples)) return;
  if (aborted()) return;
  DropExcluded(fr.samples, excluded, W, H);

  progressMessage("AnyCalib: fitting frame %g", frame);
  progressFraction(0.7);
  double focal;
  std::string err;
  if (!fitLens(Thin(fr.samples, 4096), W, H, pa, fr.lens, focal, nullptr, err)) {
    a.shot.fail("frame " + std::to_string(frame) + " fit failed: " + err);
    error("AnyCalib: %s", a.shot.error().c_str());
    return;
  }

  progressMessage("AnyCalib: lines, frame %g", frame);
  progressFraction(0.85);
  const std::vector<float> gray = Luminance(img);
  fr.segs = DetectSegments(gray.data(), W, H, 48, excluded.empty() ? nullptr : excluded.data());
  if (aborted()) return;
  progressFraction(1.0);

  if (!a.shot.addFrame(std::move(fr))) error("AnyCalib: %s", a.shot.error().c_str());
}

void AnyCalibOp::endExecuting() {
  Accumulator& a = acc();
  // Moved out, so every exit path frees the rays.
  ShotAnalysis shot = std::move(a.shot);
  if (aborted()) return;
  if (!shot.complete()) {
    error("AnyCalib: %s", shot.error().empty() ? "analysis incomplete; previous result kept"
                                               : shot.error().c_str());
    return;
  }
  const auto& frames = shot.frames();
  const size_t nF = frames.size();
  const NukeLD& format = frames.front().lens;

  NukeLD nk;
  double focal = 0;
  FitStats st;
  std::string err;
  if (!fitLens(shot.pooledSamples(), format.W, format.H, format.pa, nk, focal, &st, err)) {
    error("AnyCalib: pooled fit failed (%s)", err.c_str());
    return;
  }
  const double spread = FrameSpread(frames);
  if (!std::isfinite(spread)) {
    error("AnyCalib: a per-frame lens is undefined on the plate; previous result kept");
    return;
  }
  const NukeLD network = nk;
  int lines = 0;
  if (a.refineLines) {
    const LineRefinement lr = RefineWithLines(AllSegments(frames), nk);
    nk = lr.lens;
    lines = lr.lines;
  }

  setKnob("distortion1", nk.k1);
  setKnob("distortion2", nk.k2);
  setKnob("distortionCenter", nk.cx, 0);
  setKnob("distortionCenter", nk.cy, 1);
  setKnob("focalPx", focal);
  setKnob("rmsDegrees", st.rmsDeg);
  setKnob("frameSpread", spread);
  setKnob("linesUsed", lines);
  setKnob("lineCorrection", CornerChange(network, nk));
  setText("modelFile", a.model);
  setText("modelSha256", a.digest);
  setKnob("framesAnalysed", double(nF));

  std::ostringstream fi;
  fi << frames.front().frame << "-" << frames.back().frame << " (" << nF << " frames)";
  setText("framesInfo", fi.str());
  setText("estimatorVersion",
          "anycalib-nuke/" ANYCALIB_NUKE_VERSION " libtorch/" TORCH_VERSION " " + a.device);

  // Overlay lines for up to 32 evenly spaced frames, to bound the script size.
  setText("linesData", SerialiseLines(nk.W, nk.H, frames, std::max<size_t>(1, (nF + 31) / 32)));

  storeOverscan(nk);
}

template <class F>
void AnyCalibOp::readRows(Iop& in, int W, int H, const ChannelSet& chans, F onRow) {
  in.request(0, 0, W, H, chans, 1);
  Interest interest(in, 0, 0, W, H, chans, true);
  interest.unlock();
  Row row(0, W);
  for (int y = 0; y < H && !aborted(); ++y) {
    row.get(in, y, 0, W, chans);
    onRow(row, y);
  }
}

std::vector<float> AnyCalibOp::readPlate(Iop& in, int W, int H) {
  std::vector<float> img(size_t(W) * H * 3);
  readRows(in, W, H, Mask_RGB, [&](const Row& row, int y) {
    const float* R = row[Chan_Red];
    const float* G = row[Chan_Green];
    const float* B = row[Chan_Blue];
    float* d = img.data() + size_t(H - 1 - y) * W * 3;
    for (int X = 0; X < W; ++X, d += 3) {
      const float p[3] = {R[X], G[X], B[X]};
      for (int c = 0; c < 3; ++c)
        d[c] = float(_matrix[3 * c] * p[0] + _matrix[3 * c + 1] * p[1] + _matrix[3 * c + 2] * p[2]);
    }
    if ((y & 63) == 0) progressFraction(0.3 * y / H);
  });
  return img;
}

// Pixels the mask leaves out, top row first. Empty without a mask.
std::vector<uint8_t> AnyCalibOp::readMask(int W, int H) {
  Iop* m = nullptr;
  if (_ignore == kSourceAlpha || _ignore == kSourceInvertedAlpha)
    m = &input0();
  else if (_ignore != kNoMask && node_input(1))
    m = dynamic_cast<Iop*>(input(1));
  if (!m) return {};

  const bool luma = _ignore == kMaskLuminance || _ignore == kMaskInvertedLuminance;
  const bool invert = _ignore == kSourceInvertedAlpha || _ignore == kMaskInvertedLuminance ||
                      _ignore == kMaskInvertedAlpha;
  const ChannelSet chans = luma ? ChannelSet(Mask_RGB) : ChannelSet(Mask_Alpha);
  m->validate(true);
  std::vector<uint8_t> out(size_t(W) * H);
  readRows(*m, W, H, chans, [&](const Row& row, int y) {
    uint8_t* e = &out[size_t(H - 1 - y) * W];
    for (int X = 0; X < W; ++X) {
      const float v = luma ? Luma(row[Chan_Red][X], row[Chan_Green][X], row[Chan_Blue][X])
                           : row[Chan_Alpha][X];
      e[X] = (invert ? 1 - v : v) >= 0.5f;
    }
  });
  return out;
}

bool AnyCalibOp::infer(const std::vector<float>& img, int W, int H, double pa,
                       std::vector<RaySample>& samples) {
  Accumulator& a = acc();
  std::lock_guard<std::mutex> l(gEngineMu);
  std::string err;
  AnyCalibEngine* eng = GetEngine(a.model, a.digest, a.requestedDevice, err);
  if (!eng || !eng->Infer(img.data(), W, H, pa, samples, &err)) {
    a.shot.fail(err);
    error("AnyCalib: %s", err.c_str());
    return false;
  }
  a.device = eng->device();
  return true;
}

bool AnyCalibOp::fitLens(const std::vector<RaySample>& samples, int W, int H, double pa,
                         NukeLD& nk, double& focal, FitStats* stats, std::string& err) {
  return FitNukeLD(samples, W, H, pa, nk, focal, stats, &err, acc().lockCentre);
}

Op::HandlesMode AnyCalibOp::doAnyHandles(ViewerContext* ctx) {
  if (panel_visible() && knob("showLines")->get_value() != 0) return eHandlesCooked;
  return Iop::doAnyHandles(ctx);
}

void AnyCalibOp::build_handles(ViewerContext* ctx) {
  build_input_handles(ctx);
  if (ctx->transform_mode() != VIEWER_2D) return;
  build_knob_handles(ctx);
  // Not when the panel is open but the viewer shows another branch.
  if (_showLines && ctx->reallyConnected()) add_draw_handle(ctx);
}

void AnyCalibOp::draw_handle(ViewerContext* ctx) {
  if (!ctx->draw_lines() || !_showLines || !_lines || !*_lines) return;
  if (_parsedFrom != _lines) {
    _parsed = ParseLines(_lines);
    _parsedFrom = _lines;
  }
  if (_parsed.frames.empty() || _parsed.W != _lens.W || _parsed.H != _lens.H) return;

  const auto& [frame, segs] = NearestFrame(_parsed, outputContext().frame());
  glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT);
  int nStraight = 0;
  for (const Segment& s : segs) {
    const OverlayLine o = BuildOverlayLine(s, _lens);
    DrawOverlayLine(o);
    nStraight += o.straight;
  }
  glColor3f(1.0f, 1.0f, 1.0f);
  char label[256];
  std::snprintf(label, sizeof(label),
                "AnyCalib  frame %g  k1 %.4f  k2 %.4f  lines straight %d / %d", frame, _lens.k1,
                _lens.k2, nStraight, int(segs.size()));
  gl_text(label, 10.0, _lens.H + 12.0);
  glPopAttrib();
}
