# Test assets: sources and licences

All photos were normalised with the same steps: EXIF orientation applied, converted to
RGB, downscaled with Lanczos so the long side is at most 2048 px, metadata stripped, and
saved as JPEG quality 95 (4:4:4). Nothing else was changed (no crop, colour or geometry
edits). Uniform downscaling keeps the normalised distortion (k1, k2) and scales the focal
length and principal point. For CC BY / CC BY-SA images, these steps are the "changes made".

Reference intrinsics from AnyCalib's official pipeline are in `reference.json`. To
regenerate them, run `tools/make_reference.py`.

| File | Content | Camera (original EXIF) | Author | Licence | Source |
|---|---|---|---|---|---|
| `control_seo_iphonexs.jpg` | Street scene, **near-undistorted control** (portrait 1536x2048) | Apple iPhone XS, tele 6 mm f/2.4 (52 mm equiv.) | Sergei Gussev | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | https://www.flickr.com/photos/sergeigussev/49521010167/ (obtained via the AnyCalib demo Space: https://huggingface.co/spaces/javrtg/AnyCalib/blob/main/assets/seo.jpg, which credits this author and licence in `app.py`) |
| `gopro_hero7_metro_avtozavodskaya.jpg` | Metro platform with columns and platform edges; strong barrel | GoPro HERO7 Black, 3 mm (15 mm equiv.) | Svetlov Artem | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | https://commons.wikimedia.org/wiki/File:Moscow_metro_Avtozavodskaya_2019-07.jpg |
| `gopro_hero7_dinamo_building.jpg` | High-rise facade, looking up; curved verticals | GoPro HERO7 Black, 3 mm (15 mm equiv.) | Svetlov Artem | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | https://commons.wikimedia.org/wiki/File:Moscow_Dinamo_building_2019-07.jpg |
| `gopro_hero7_metro_oktyabrskaya.jpg` | Metro vestibule with railings and arches | GoPro HERO7 Black, 3 mm (15 mm equiv.) | Svetlov Artem | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | https://commons.wikimedia.org/wiki/File:Moscow_Metro_Ortyabrskaya_koltsevaya_vestibule_2019-07.jpg |
| `gopro_hero4_assembly_hall.jpg` | Arena interior; very strong (fisheye-like) barrel | GoPro HERO4 Silver, 3 mm (15 mm equiv.) | Chris Kuga | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | https://commons.wikimedia.org/wiki/File:Indiana_Assembly_Hall_interior_(16100187018).jpg (from Flickr) |
| `fisheye_kutschpferdestall_potsdam.jpg` | Vaulted hall; extreme fisheye. Stress case beyond what radial:2 can model | Panasonic DC-G9 (manual lens, focal not recorded) | Rolf Dietrich Brecher | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | https://commons.wikimedia.org/wiki/File:Kutschpferdestall_Potsdam_(30955382858).jpg (from Flickr) |
| `synthetic_barrel_k1-0.12_k2+0.02.png` | Synthetic street canyon with **known** radial distortion (see below) | n/a | generated | generated for this project, no third-party content | n/a |
| `synthetic_undistorted.png` | The same render with k1 = k2 = 0 (zero-distortion control) | n/a | generated | generated for this project, no third-party content | n/a |

The three HERO7 photos come from the same camera and photographer within two days of each
other, so their estimates should be broadly consistent.

## Synthetic image ground truth

- Size: 1920 x 1080
- Camera model: OpenCV radial / AnyCalib `radial:2`, intrinsics `[fx, fy, cx, cy, k1, k2]`:
  - `fx = 1400`, `fy = 1400`
  - `cx = 960`, `cy = 540`, using the pixel convention below
  - `k1 = -0.12`, `k2 = +0.02` (barrel); `p1 = p2 = k3 = 0`
- Projection: `x = X/Z`, `y = Y/Z`, `r^2 = x^2 + y^2`, `s = 1 + k1 r^2 + k2 r^4`, `u = fx * x * s + cx`, `v = fy * y * s + cy`.
- **Pixel convention:** (0,0) is the top-left *corner* of the top-left pixel, so pixel centres sit at +0.5. This is the AnyCalib convention and the one `AnyCalibEngine.cpp` uses. In OpenCV's centre-at-integer convention the same camera is `cx = 959.5`, `cy = 539.5`, with fx, fy, k1, k2 unchanged.
- Rendering: the scene is ray-cast through an ideal pinhole with the same fx, fy, cx, cy at 2x resolution, on a canvas with a 96 x 61 px margin, so barrel distortion pulls in real content instead of black borders. That canvas is warped with `cv2.remap` (INTER_CUBIC) through the inverse model: for each output pixel, Newton's method solves `r_u (1 + k1 r_u^2 + k2 r_u^4) = r_d`. The result is area-downsampled to 1920x1080 and Gaussian noise (sigma 0.006) is added. The model is monotonic over the frame: min `d r_d / d r_u` = 0.79.
- Camera pose (does not affect intrinsics): yaw -12 deg, pitch +4 deg (looking up), 1.6 m above the street.
- `synthetic_undistorted.png`: the identical render with `k1 = k2 = 0` (same fx, fy, cx, cy).

The stock `anycalib_dist` model does **not** recover this ground truth well on the
flat-shaded render. It estimates fx ~ 1080 and k1 ~ -0.084 for the barrel image, and fx ~
1023 with k1 ~ -0.023 for the undistorted one; see `reference.json`. So the C++ port should
be checked against `reference.json`. Use the ground truth only as a sanity anchor, for
example "barrel image is more barrel than the undistorted one".
