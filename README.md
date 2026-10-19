# dicgen — synthetic 2D DIC validation dataset generator

`dicgen` is a small, self-contained Python command-line application that generates a
**synthetic 2D Digital Image Correlation (DIC) dataset**: a grayscale speckle image
sequence of a specimen that deforms, together with the **exact, analytic ground-truth
displacement and strain fields** that are physically present in those images. Its
purpose is to provide a *known-truth* benchmark so that a 2D DIC solver can be scored
against the deformation that was actually baked into the pixels.

The deformation is a single affine transform (by default) applied *incrementally* over
the sequence, so image 0 is the undeformed reference and the last image carries the full
requested stretch/rotation/translation. The reference speckle pattern is a random field
of circular dots that is **never re-randomised**: the very same dots are advected by the
deformation of each image. Everything random comes from one seeded
`numpy.random.Generator`, there are no wall-clock timestamps, and the tool is fully
non-interactive.

## What you get

- A grayscale (8-bit PNG) image sequence `image0001.png … imageNNNN.png`, reference
  first.
- Exact analytic ground truth as CSV: per-image transform/strain parameters
  (`ground_truth.csv`) and the full displacement/strain/Jacobian on a sample grid
  (`ground_truth_grid.csv`).
- Per-image validity masks (`masks/mask####.png`, `255 = valid`, `0 = invalid`) for the
  regions of a deformed image that came from outside the reference.
- A **movie** (`sequence.mp4`, H.264) assembled from the image sequence and written
  next to the images, so the deformation can be played back at a glance. It needs the
  `ffmpeg` binary; pass `--no-movie` to skip it.
- A complete, reproducibility-grade `metadata.json` with every parameter **and a SHA-256
  digest of every generated file**.
- Optional `.npy` displacement field, and optional colour/arrow QA maps for a quick
  visual check.
- Deterministic output: the same seed and the same arguments produce byte-identical
  files.

## Table of contents

- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [The deformation model](#the-deformation-model)
- [Image synthesis](#image-synthesis)
- [Output files](#output-files)
- [Using the dataset from a DIC solver](#using-the-dataset-from-a-dic-solver)
- [Verifying a generated dataset (`dicgen.verify`)](#verifying-a-generated-dataset-dicgenverify)
- [Reproducibility and determinism](#reproducibility-and-determinism)
- [Performance](#performance)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Optional features and known limitations](#optional-features-and-known-limitations)
- [Example dataset](#example-dataset)

---

## Quick start

Requires Python 3.10+ with `numpy`, `scipy` and `pillow`. Writing the `sequence.mp4`
movie additionally needs the external **`ffmpeg`** binary on `PATH` (e.g.
`apt install ffmpeg` or `brew install ffmpeg`); without it the run still succeeds and
simply skips the movie, so you can also install it later.

```bash
# from the repository root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # numpy, scipy, pillow
pip install -e .                     # installs the `dicgen` console script
```

Minimal invocation — a 5-image, 512×512 sequence with a total 2% stretch in x:

```bash
python -m dicgen --output-dir out/demo -n 5 --width 512 --height 512 --strain-x 0.02 --seed 0
```

It prints a concise progress log and a summary such as:

```
dicgen 1.0.0: preparing out/demo
Created output directory out/demo
Speckle: 2153 dots, radii [3, 6] px, coverage 0.4989 (target 0.5, 2 calibration pass(es))
[1/5] image0001.png: t = 0.000000, valid area 100.00 %, grey levels 1-137
[2/5] image0002.png: t = 0.250000, valid area 100.00 %, grey levels 1-137
[3/5] image0003.png: t = 0.500000, valid area 100.00 %, grey levels 1-137
[4/5] image0004.png: t = 0.750000, valid area 100.00 %, grey levels 1-137
[5/5] image0005.png: t = 1.000000, valid area 100.00 %, grey levels 2-137
Movie sequence.mp4: 5 frames, 10 fps, 0.10 MB
Done in 0.43 s: 14 files in out/demo
Coverage 0.4989 (target 0.5), 2153 dots, mean valid area 100.00 %, max |u| 5.110 px, noise sigma 2 grey levels

Generation summary
------------------
  output directory : out/demo
  images           : 5 x 512x512 px
  files written    : 14
  dot coverage     : 0.4989 (target 0.5, 2153 dots)
  mean valid area  : 100.00 %
  max |u|          : 5.1100 px
  noise sigma      : 2 grey levels (multiplicative 0)
  seed             : 0
  movie            : sequence.mp4 (10 fps)
  elapsed          : 0.43 s
```

Both entry points are equivalent: `python -m dicgen …` and the installed console script
`dicgen …`. The dataset verifier is a separate command: `python -m dicgen.verify --dataset OUT`
(or `dicgen-verify --dataset OUT`) — see [Verifying a generated dataset](#verifying-a-generated-dataset-dicgenverify).

---

## CLI reference

The full, always-current list of options is `python -m dicgen --help` (or `dicgen
--help`). The most important options and their exact defaults:

| Option | Meaning | Default |
|---|---|---|
| `--output-dir` | Directory for the images and ground truth (created if missing) | **required** |
| `--width` | Image width in pixels | `1024` |
| `--height` | Image height in pixels | `1024` |
| `-n`, `--num-images` | Total number of images, including the undeformed reference (image 0) | `11` |
| `--strain-x` | Total affine engineering stretch in x over the whole sequence (`0.02` = 2%) | `0.0` |
| `--strain-y` | Total affine engineering stretch in y over the whole sequence | `0.0` |
| `--rotation` | Total rotation over the whole sequence, in degrees | `0.0` |
| `--translation-x` | Total rigid-body translation in x, in pixels | `0.0` |
| `--translation-y` | Total rigid-body translation in y, in pixels | `0.0` |
| `--deformation-mode` | `affine` or `sinusoid` (adds a non-homogeneous field) | `affine` |
| `--sin-amplitude-x` / `--sin-amplitude-y` | sinusoid mode: displacement amplitude in x / y, in pixels | `0.0` |
| `--sin-wavelength-x` / `--sin-wavelength-y` | sinusoid mode: wavelength in x / y, in pixels | `None` → image width / height |
| `--dot-radius-min` / `--dot-radius-max` | Range of speckle dot radii, in pixels | `3.0` / `6.0` |
| `--dot-coverage` | Target fraction of the image area covered by dots (`0 < c < 0.95`) | `0.5` |
| `--dot-profile` | `taper` or `gaussian` analytic dot profile | `taper` |
| `--dot-softness` | taper profile: edge softness in pixels (sub-pixel anti-aliasing) | `0.7` |
| `--dot-amplitude-min` / `--dot-amplitude-max` | Range of dot darkening relative to the background, in grey levels | `70.0` / `120.0` |
| `--background` | Background grey level of the speckle pattern | `128.0` |
| `--min-dot-distance` | Minimum distance between dot centres, in pixels (`0` = unconstrained) | `0.0` |
| `--coverage-tolerance` | Stop calibrating the dot count once the measured coverage is this close to the target | `0.01` |
| `--sampling` | `analytic` (exact continuous field) or `bicubic` (spline resample of the reference) | `analytic` |
| `--interpolation-order` | Spline order used by `--sampling bicubic` (`1`–`5`) | `3` |
| `--noise-sigma` | Std-dev of additive Gaussian grey-level noise (`0` = noise free) | `2.0` |
| `--multiplicative-noise-sigma` | Relative sigma of the optional multiplicative speckle noise (`0` = off) | `0.0` |
| `--seed` | RNG seed; the run is fully reproducible | `0` |
| `--save-ground-truth` / `--no-ground-truth` | Write / do not write the ground-truth CSVs | on |
| `--save-masks` / `--no-masks` | Write / do not write the validity masks | on |
| `--save-displacement-maps` | Write `qa/` colour maps of `\|u\|` and arrow overlays | off |
| `--save-displacement-npy` | Write `displacement_field.npy` (`(n, 2, height, width)` float32) | off |
| `--save-movie` / `--no-movie` | Write / do not write `sequence.mp4` from the image sequence (needs `ffmpeg`) | on |
| `--movie-fps` | Playback rate of `sequence.mp4`, in frames per second | `10.0` |
| `--gt-grid NX NY` | Sample grid of `ground_truth_grid.csv`, in points | `16 16` |
| `--overwrite` | Delete everything inside `--output-dir`, including files not generated by dicgen | off |
| `--dry-run` | Validate arguments, report what would be deleted, write nothing | off |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Logging verbosity | `INFO` |
| `-q`, `--quiet` / `-v`, `--verbose` | Only warnings+errors / per-image debug logging | off |

### `--output-dir` and `--overwrite` semantics

- If the directory does not exist it is created, including parents.
- If it exists, everything **that this tool could have generated** is deleted before new
  files are written (so stale images from a longer previous run disappear). Recognised
  content: `image####.png`, `ground_truth.csv`, `ground_truth_grid.csv`, `metadata.json`,
  `displacement_field.npy`, `sequence.mp4`, and the `masks/` and `qa/` sub-directories
  when they contain only tool-generated files.
- Anything else ("foreign content" — an unrelated file, a symlink, a directory with an
  unknown name) is left untouched and the run is **refused** with a clear message that
  lists it, unless `--overwrite` is passed. With `--overwrite`, every entry inside the
  directory is removed (symlinks are unlinked, never followed).
- The output directory itself is never deleted, and nothing outside it is ever touched.
  As a safety net the tool refuses the filesystem root, the home directory, and the
  current working directory as `--output-dir`.

### `--dry-run`

Validates all arguments, inspects the output directory and prints exactly what *would*
be deleted, then exits without creating or deleting anything (the summary reports "dry
run: nothing was written").

### Exit codes and validation

Arguments are validated up front and errors are actionable. Examples: `n >= 2`,
`width`/`height >= 8`, `--dot-radius-max <= 25%` of the shorter side,
`0 < --dot-coverage < 0.95`, `--background - --dot-amplitude-max >= 0`,
`--interpolation-order` in `[1, 5]`, `--gt-grid` at least `2 2`, and
`--noise-sigma <= 128`. Exit codes: `0` success, `2` argument/validation error
(including refused foreign content), `3` an OS/runtime error.

---

## The deformation model

### Coordinates and the image centre

Coordinates are stored as `(x, y)` pairs where **`x` is the column index** and **`y` is
the row index** — the usual image convention with **`y` pointing down**. The centre of
the pixel grid is

```
center = ((width - 1) / 2, (height - 1) / 2)
```

All deformations are centred on this point and are the identity for image 0. Because `y`
points down, a **positive rotation angle looks clockwise** on screen.

### Load stepping

For a sequence of `n` images (`n >= 2`), the load fraction of image `i` is

```
t_i = i / (n - 1),      i = 0, 1, …, n - 1
```

so `t_0 = 0` (undeformed reference) and `t_{n-1} = 1` (full deformation): the total
deformation is divided **equally** over the sequence.

### Affine model (default)

For image `i` the deformation gradient is

```
F_i = R(theta * t_i) @ diag(1 + e_x * t_i, 1 + e_y * t_i)
```

where `R(a) = [[cos a, -sin a], [sin a, cos a]]` is the standard 2D rotation matrix and
`e_x`, `e_y` are the **total engineering stretches** (`--strain-x`, `--strain-y`) and
`theta` the **total rotation** in degrees (`--rotation`).

The forward mapping from reference coordinates `X` (undeformed, image 0) to deformed
coordinates `x` (image `i`) is

```
x = center + F_i @ (X - center) + t_i * T
```

where `T` is the optional total rigid-body translation
(`--translation-x`, `--translation-y`, in pixels; default `(0, 0)`). The ground-truth
displacement is therefore

```
u(X) = x - X = (F_i - I) @ (X - center) + t_i * T
```

which is *linear* in `X`. Its gradient `du/dX = F_i - I` is constant, so the reported
(engineering, small-strain) strain field is uniform:

```
epsilon_xx = e_x * t_i
epsilon_yy = e_y * t_i
gamma_xy   = 0
```

### The polar decomposition is exact

`R(theta * t_i)` is orthogonal and `diag(1 + e_x t_i, 1 + e_y t_i)` is symmetric positive
definite, so the parameterisation **is its own polar decomposition**:

```
F_i = R(theta * t_i) @ U_i,     U_i = diag(1 + e_x t_i, 1 + e_y t_i)
```

Consequently `theta * t_i` is exactly the polar (rigid) rotation and `e_x t_i`, `e_y t_i`
are exactly the principal engineering stretches — for any magnitude of the parameters,
not just small ones.

### Strain measures (which strain is reported?)

Only **engineering / infinitesimal** strains are reported, and the `epsilon_xx`,
`epsilon_yy`, `gamma_xy` above are the strains of the *stretch factors of the
parameterisation* (equivalently, the strains along the principal axes of the polar
decomposition). They are exact at every load level. For completeness the CSVs also report
the components of the **small-strain tensor in the image frame**, `sym(F_i) - I` (columns
`eps_xx_img`, `eps_yy_img`, `gamma_xy_img`); these differ from the parameter strains by
`O(theta * e)`. Other finite-strain measures are readily recomputed from `F_i`:

- logarithmic (Hencky) principal strains: `ln(1 + e_x t_i)` and `ln(1 + e_y t_i)`;
- Green–Lagrange strain: `((1 + e)^2 - 1) / 2`.

For the small strains typical of DIC validation (≤2%) these differ from the reported
engineering strains only at the level of the last significant digit of the parameters.

### Optional sinusoidal model (`--deformation-mode sinusoid`)

This adds a non-homogeneous displacement field, switched on with the same load fraction
`t_i`:

```
u_sin_x(X) = A_x * sin(2*pi*(X_x - c_x) / L_x)
u_sin_y(X) = A_y * sin(2*pi*(X_y - c_y) / L_y)
```

with `c = center`, amplitudes `A_x`, `A_y` (`--sin-amplitude-x/y`, in pixels) and
wavelengths `L_x`, `L_y` (`--sin-wavelength-x/y`, defaulting to the image width /
height). The forward map becomes

```
x = center + F_i @ (X - center) + t_i * T + t_i * u_sin(X)
```

Its Jacobian is available analytically,
`J_i(X) = F_i + t_i * diag(du_sin_x/dX_x, du_sin_y/dX_y)`, and the inverse mapping
`x -> X` is obtained with a vectorised Newton iteration initialised with the affine
inverse. To keep the map monotone and invertible, the amplitudes are constrained:
`|A| <= 0.95 * L / (2*pi)`. In this mode the documented strain field
(`eps_xx`, `eps_yy`) is non-uniform and also varies sinusoidally; the exact values at
every grid point are in `ground_truth_grid.csv`.

---

## Image synthesis

### The reference speckle pattern

The reference image (image 0) is a random field of circular dots on a mid-grey
background:

- **Dot centres** are sampled uniformly over the image, optionally enforcing a minimum
  distance between centres (`--min-dot-distance`, a dart-throwing sampler on a uniform
  grid).
- **Radii** are sampled uniformly in `[--dot-radius-min, --dot-radius-max]`.
- **Amplitudes** (darkening relative to the background) are sampled uniformly in
  `[--dot-amplitude-min, --dot-amplitude-max]`, so the speckle is not perfectly bimodal.
- Every dot is rendered as a **smooth analytic intensity profile** `f(d)` in `[0, 1]`,
  where `d` is the distance from the dot centre:

  - **`taper` (default)** — plateau + raised-cosine edge, with softness
    `s = --dot-softness`:

    ```
    f(d) = 1                                             d <= r - s
    f(d) = 0.5 * (1 + cos(pi * (d - (r - s)) / (2 s)))   r - s < d < r + s
    f(d) = 0                                             d >= r + s
    ```

    `f(r) = 0.5` exactly, so the area where a dot darkens the image by more than half of
    its amplitude equals the geometric disc area `pi r^2`.

  - **`gaussian`** — `f(d) = exp(-0.5 * (d / sigma)^2)` with `sigma = r / 2`.

  The rendered pixel is `background - max_k(amplitude_k * f_k)`: overlapping dots combine
  with a **union / max** rule (a pixel is as dark as the darkest dot covering it). This
  is physically sensible for an opaque speckle pattern and exactly consistent with the
  union-area coverage model below.

### Choosing the number of dots (coverage model and calibration)

For randomly placed, freely overlapping discs of equal radius, the union area saturates
as `coverage = 1 - exp(-n * pi * r^2 / A)`. `dicgen` therefore *starts* from

```
n = -A * ln(1 - coverage) / (pi * r_rms^2)
```

where `A = width * height` and `r_rms` is the **RMS** radius of a radius drawn uniformly
from `[radius_min, radius_max]`, `r_rms = sqrt((a^2 + a*b + b^2) / 3)` (the union area is
proportional to `E[r^2]`, not to the arithmetic mean).

A short **calibration loop** (at most 4 passes) then re-renders the reference, measures
the *achieved* coverage, and corrects `n` using the measured rate
`k = -ln(1 - achieved) / n`; it stops as soon as the measured coverage is within
`--coverage-tolerance` of the target (default `0.01`) and keeps the best pass. The
achieved coverage is printed and recorded in `metadata.json`. Here "coverage" means the
fraction of pixels darkened by more than `0.5` grey levels. The calibration re-draws the
dot population each pass, so the number of random draws depends on the number of passes —
but the whole procedure is deterministic for a fixed seed and fixed arguments.

### Warping the same dots into the deformed images

The same dot set is advected by the deformation; it is never re-randomised. `dicgen`
supports two sampling methods:

- **`analytic` (default)** — the continuous dot field is evaluated directly at the
  inverse-mapped coordinates `X = F_i^{-1}(x)` of every deformed pixel. The image content
  therefore matches the analytic ground truth **exactly** (no interpolation error),
  because each dot only affects a small bounding box of the deformed image. This is the
  cleanest possible validation dataset.
- **`bicubic`** — the classical DIC-dataset recipe: rasterise the reference at the pixel
  grid once, then sample it at `X = F_i^{-1}(x)` with a spline of order
  `--interpolation-order` (default 3) via `scipy.ndimage.map_coordinates`. This
  reproduces the small interpolation smoothing/bias a real camera + DIC pipeline would
  see. The valid region is **shrunk by 2 px** so that no output pixel is built from a
  spline stencil that reaches outside the reference image.

### Out-of-bounds handling and masks

Samples whose inverse-mapped `X` lies outside `[0, W-1] x [0, H-1]` (or, for bicubic,
within 2 px of the border) are filled with the **background grey level** and recorded as
**invalid** in the mask. Each image `i` therefore has a matching
`masks/mask####.png` where `255 = valid` (correlate) and `0 = invalid` (no speckle data).

### Noise

After warping, independent Gaussian noise is added to every image:

```
I = I * (1 + N(0, s_mul)) + N(0, s_add)
```

where `s_add = --noise-sigma` (default `2.0` grey levels) and `s_mul =
--multiplicative-noise-sigma` (default `0.0`, i.e. off). Set `--noise-sigma 0` for
noise-free output. The noise is drawn from the run's single RNG, independently per image,
so it is reproducible.

### Clipping

All intermediate computations are floating point. Grey levels are **deliberately clipped
to `[0, 255]` and rounded to the nearest integer only when the 8-bit PNG is written**.

---

## Output files

Assuming the output directory is `OUT`, a successful run writes:

| Path | Written when | Content |
|---|---|---|
| `OUT/image0001.png … imageNNNN.png` | always | 8-bit grayscale PNG sequence, reference first; `pad = max(4, len(str(n)))` digits, so `image0001.png` for 11 images |
| `OUT/sequence.mp4` | `--save-movie` (on) | H.264 movie of the whole image sequence, played at `--movie-fps`; needs `ffmpeg` |
| `OUT/masks/mask0001.png …` | `--save-masks` (on) | validity mask, `255 = valid`, `0 = invalid` |
| `OUT/qa/disp_0001.png …` | `--save-displacement-maps` | viridis-like colour map of the displacement magnitude `\|u\|` with a white arrow overlay |
| `OUT/qa/arrows_0001.png …` | `--save-displacement-maps` | grey reference image with red displacement arrows |
| `OUT/ground_truth.csv` | `--save-ground-truth` (on) | one row per image (exact transform + strain parameters) |
| `OUT/ground_truth_grid.csv` | `--save-ground-truth` (on) | analytic ground truth on an `NX x NY` sample grid, all images |
| `OUT/displacement_field.npy` | `--save-displacement-npy` | float32 array `(n, 2, height, width)` |
| `OUT/metadata.json` | always | full run record + per-file SHA-256 manifest |

The `qa/` file index is padded the same way as the images (`disp_0001.png`).

### `ground_truth.csv` columns

One row per image, in image order (`IMAGE_COLUMNS` in `dicgen/groundtruth.py`):

```
image_index, image, t, F11, F12, F21, F22, tx_px, ty_px, e_x_t, e_y_t, theta_t_deg,
eps_xx, eps_yy, gamma_xy, strain_uniform, eps_xx_img_mean, eps_yy_img_mean,
gamma_xy_img_mean, F_lambda_max, F_det, u_max_px, u_mean_px, valid_fraction
```

| Column | Meaning |
|---|---|
| `image_index` | 0-based index `i` used by `t_i`; `0` is the reference (`image0001.png`) |
| `image` | file name in the output directory |
| `t` | load fraction `t_i = i / (n - 1)` |
| `F11, F12, F21, F22` | the deformation gradient `F_i` (row-major) |
| `tx_px, ty_px` | rigid-body translation contribution `t_i * T`, in pixels |
| `e_x_t, e_y_t` | principal engineering stretches at this load (`e_x * t`, `e_y * t`) |
| `theta_t_deg` | rotation at this load (`theta * t`), in degrees |
| `eps_xx, eps_yy, gamma_xy` | documented engineering strain convention (uniform for the affine model) |
| `strain_uniform` | `1` if the documented strain field is uniform, else `0` |
| `eps_xx_img_mean, eps_yy_img_mean, gamma_xy_img_mean` | mean of the image-frame small-strain tensor `sym(J) - I` over the sample grid |
| `F_lambda_max` | largest principal stretch (spectral norm of `F_i`) |
| `F_det` | determinant of `F_i` |
| `u_max_px, u_mean_px` | max and mean displacement magnitude over the pixel grid |
| `valid_fraction` | fraction of pixels of this image that come from inside the reference |

### `ground_truth_grid.csv` columns

`n * NX * NY` rows (one block of `NX * NY` rows per image), grid ordered with `y`
varying slowest (`GRID_COLUMNS` in `dicgen/groundtruth.py`):

```
image_index, image, t, X_px, Y_px, x_px, y_px, ux_px, uy_px,
eps_xx, eps_yy, gamma_xy, eps_xx_img, eps_yy_img, gamma_xy_img,
J11, J12, J21, J22, in_ref, in_def
```

| Column | Meaning |
|---|---|
| `X_px, Y_px` | reference (undeformed) coordinates of the grid point |
| `x_px, y_px` | deformed coordinates `x = transform(X, t)` |
| `ux_px, uy_px` | ground-truth displacement `u = x - X` |
| `eps_xx, eps_yy, gamma_xy` | documented engineering strain convention at the point |
| `eps_xx_img, eps_yy_img, gamma_xy_img` | image-frame small-strain tensor `sym(J) - I` |
| `J11, J12, J21, J22` | Jacobian `du/dX + I` at the grid point |
| `in_ref, in_def` | `1` if `X` (resp. `x`) lies inside the image, else `0` |

The sample grid is `linspace(0, W-1, NX) x linspace(0, H-1, NY)`, i.e. it includes the
image borders.

### `displacement_field.npy`

A `float32` array of shape **`(n, 2, height, width)`** where axis 1 indexes
`(u_x, u_y)` and axes 2/3 are pixel row `y` and column `x`. For a 1024×1024, 11-image
run this file is about 92 MB.

### `sequence.mp4`

The image sequence encoded as an H.264 movie (`libx264`, `yuv420p`, CRF 18) so the
deformation can be watched rather than flipped through. It is written next to the
images and plays back at `--movie-fps` (default `10`, i.e. a 11-image sequence lasts
about 1 s). The frames are piped straight to the system **`ffmpeg`** binary, so the
movie is built from exactly the 8-bit images on disk. Two details are worth knowing:

- **`ffmpeg` is an optional external dependency.** It is *not* a Python package. If it
  is not on `PATH`, `dicgen` prints a warning, records `"movie": {"written": false, …}`
  in `metadata.json` and finishes normally with the rest of the dataset intact — the
  movie is best-effort. Pass `--no-movie` to skip it deliberately.
- **Reproducibility.** The encoder is invoked with `-fflags +bitexact` /
  `-flags:v +bitexact` and no metadata, so the same images and the same `ffmpeg` build
  produce a byte-identical MP4 (asserted by the tests). H.264 is not bit-stable
  *between* `ffmpeg` versions, so the movie — unlike the images and CSVs — is only
  guaranteed reproducible on the machine that generated it.

Odd image sizes (which `yuv420p` cannot represent) are padded by one row/column of
black automatically. The file is listed in the `metadata.json` SHA-256 manifest like
every other generated file.

### `metadata.json`

A deterministic JSON document (sorted keys, 2-space indent) recording the entire run.
Top-level sections:

| Section | Content |
|---|---|
| `tool`, `version`, `metadata_format_version` | tool name, `1.0.0`, format version `1` |
| `file_naming`, `mask_convention`, `deterministic`, `wall_clock_timestamps` | conventions (`True`, `False`) |
| `arguments` | every `GenerationConfig` field as passed (including `output_dir`) |
| `sequence` | `num_images`, `image_size_px`, `filenames`, `filename_pattern`, `reference_image`, `load_fraction`, `deformed_coordinates`, `ground_truth_displacement`, `image_center_px` |
| `deformation` | the model (`mode`, total strains/rotation/translation, `image_center_px`, `load_stepping`; sinusoid parameters if applicable) |
| `speckle` | dot radius/coverage/profile/softness/amplitude/background/min-distance parameters |
| `dots` | realised dot statistics (`num_dots`, radius mean/std/min/max, `amplitude_mean`) |
| `coverage` | `num_dots`, `iterations`, `coverage_any`, `coverage_half_amplitude`, `coverage_target`, `coverage_error`, `coverage_history` |
| `noise` | additive and multiplicative sigmas, where/when applied, clipping note |
| `sampling` | method, interpolation order/library, out-of-bounds handling, valid-mask rule |
| `ground_truth` | which CSVs were written, grid size, the column lists, the strain convention, and the `.npy` info |
| `movie` | `written`, and when written the `filename`, `frames`, `fps`, `codec`, `pixel_format`, dimensions and `size_bytes` (or a `reason` when `ffmpeg` was unavailable) |
| `grey_level_statistics` | per-image `min`/`max`/`mean` grey level and `valid_fraction` |
| `summary` | `num_dots`, `coverage_achieved`, `coverage_target`, `valid_fraction_mean`, `displacement_max_px` |
| `files` | the **SHA-256 manifest**: a sorted list of `{path, size_bytes, sha256}` for every generated file **except `metadata.json` itself** |
| `notes` | human-readable notes |

---

## Using the dataset from a DIC solver

1. **Reference image.** Image 0 (`image0001.png`) is the undeformed reference. Correlate
   every image `i` against image 0.
2. **Compare like with like.** The measured displacement `u` and strain should be compared
   with `ground_truth.csv` (uniform/strain parameters) and `ground_truth_grid.csv` (full
   field, point by point). The reported strains in the `eps_xx`/`eps_yy`/`gamma_xy`
   columns are the **parameter strains of the affine (polar) decomposition** — the
   principal engineering stretches of `diag(1 + e_x t, 1 + e_y t)` — not the small-strain
   tensor components in the image frame (those are the `eps_*_img` columns).
3. **Respect the masks.** Use `masks/mask####.png` to exclude `0`-valued pixels: those
   came from outside the reference and carry no speckle information. With pure expansion
   the valid fraction can be 100%; contraction, rotation or translation can make it
   smaller.
4. **Mind the convention.** `x` is the column, `y` the row, `y` points down, and a
   positive rotation looks clockwise on screen.
5. **Suggested validation workflow:**
   - **Bias:** correlate a zero-deformation sequence (`--strain-x 0 --strain-y 0`) and
     check that the recovered `u` is zero to sub-pixel accuracy.
   - **Sub-pixel accuracy / convergence:** sweep small strains (e.g. 0.2–2%) and check
     that the recovered strain matches `e_x t_i` from `ground_truth.csv`, and that image 0
     gives `u ≡ 0`.
   - **Noise robustness:** sweep `--noise-sigma` (0 → 8) with a fixed seed and measure how
     the error grows.
   - **Interpolation bias:** compare `--sampling analytic` (bias-free) with
     `--sampling bicubic` to isolate the solver-side interpolation error.
   - **Non-uniform fields:** use `--deformation-mode sinusoid` to test spatial resolution
     against the pointwise values in `ground_truth_grid.csv`.

---

## Verifying a generated dataset (`dicgen.verify`)

A speckle pattern is *statistically self-similar*: every frame is "round dots on a grey
background" with the same coverage and contrast, so the eye cannot judge the deformation,
especially for an affine field centred on the image centre where the middle of the frame
does not move at all. Always check numerically — the package ships a verifier that reads
a generated directory (through its `metadata.json`) and answers three questions from the
pixel data alone:

```bash
python -m dicgen.verify --dataset examples/demo_512x512   # or: dicgen-verify --dataset ...
```

```
dicgen.verify 1.0.0 -- examples/demo_512x512
  5 images, 512x512 px, deformation = affine, sampling = analytic, noise sigma = 2
  model-aligned criterion: MAE after undoing the model <= max(3.0, 0.15 * |diff|) + 6.0 (noise); DIC |du| <= 0.100 px, DIC |dgrad| <= 0.0100

  img    t     |diff|  aligned   limit  in-frame   DIC |du|max   DIC |dgrad|max   ZNCC
  ------------------------------------------------------------------------------------
     1  0.000      0.00     0.00    9.00    100.0%             -               -       -
     2  0.250      8.43     3.35    9.00     99.3%        0.0226          0.0030   0.996
     3  0.500     14.29     3.34    9.00     98.8%        0.0271          0.0022   0.995
     4  0.750     19.53     3.32    9.00     98.3%        0.0322          0.0025   0.995
     5  1.000     24.08     3.32    9.61     97.9%        0.0230          0.0021   0.995

  PASSED: every image differs from the reference, matches the recorded deformation, and the local DIC recovers the analytic fields.
```

The transcript above is the committed `examples/demo_512x512` sequence, so it is reproducible
from a fresh clone.

* **`|diff|`** — mean `|image − reference|` in grey levels: shows that the frames *are*
  different (24 grey levels at full load for the 2 % stretch of the demo, against a noise
  level of 2).
* **`aligned`** — the same difference after pulling the image back onto the reference grid
  with the deformation recorded in `metadata.json`. Undoing the model must collapse the
  difference to the interpolation/noise floor; `limit` is the tolerance and `in-frame` the
  percentage of the reference grid whose material is still inside that image.
* **`DIC |du|max`, `DIC |dgrad|max`** — a first-order (affine) local DIC runs at seed points
  on the real pixels, starting from a *deliberately perturbed* guess, and the recovered
  displacement/gradient are compared with the analytic values (`ZNCC` = residual
  correlation). This is the strongest check and it stays valid at large strains.

Exit codes: `0` all checks passed, `1` a check failed, `2` the dataset could not be read.
`--json report.json` writes the full report (every seed measurement) for scripting.
Other options: `--last-only`, `--max-images N`, `--seed-points`, `--window`,
`--max-displacement-error`, `--max-strain-error`, `--max-aligned-error`.

### FAQ: "all my frames look the same"

Expected for a speckle dataset, and not a sign of a problem:

* the texture has no landmarks, so a 1 % change of the dot spacing is invisible when you
  flip between two frames;
* the deformation is centred on the image centre: the middle of the frame genuinely does
  not move, and the displacement grows linearly towards the edges;
* the raw frame-to-frame difference saturates once the displacement exceeds the dot
  spacing (~9 px with the defaults), so `|diff|` stops growing after a few frames even
  though the strain keeps accumulating.

Use `dicgen.verify`, the `qa/` maps (`--save-displacement-maps`) or
`ground_truth_grid.csv` instead of eyeballing. Note also that at large total strain the
*per-frame* increment is small (`total_strain / (n-1)`): 40 % over 40 images is only
1.03 % per step, so consecutive frames differ by ~5 px at the border. If you want large
increments between frames, use fewer images.

---

## Reproducibility and determinism

- A **single** `numpy.random.Generator` seeded with `--seed` supplies every random draw
  (dot centres/radii/amplitudes, calibration re-draws, per-image noise), and the order of
  the draws is fixed by the generation routine.
- **No wall-clock timestamps** are written anywhere (`metadata.json` records
  `"wall_clock_timestamps": false`).
- Therefore the same seed and the same arguments produce **byte-identical** files. (The
  output directory path is itself recorded in `metadata.json.arguments.output_dir`, so
  runs that write to *different* directories differ only in that one field; regenerate
  into the same path to get identical bytes.)

To verify, compare the file digests with the manifest, or re-hash:

```bash
# digests of every file in a run
find OUT -type f -print0 | sort -z | xargs -0 sha256sum

# the manifest recorded by the tool
python -c "import json; m=json.load(open('OUT/metadata.json')); \
print('\n'.join(f\"{f['sha256']}  {f['path']}\" for f in m['files']))"
```

A rerun with the same seed and arguments must reproduce every digest.

---

## Performance

Measured on this machine (a normal desktop/laptop, Python 3.14, NumPy 2.5, SciPy 1.18)
for a **1024×1024, 11-image** sequence with `--strain-x 0.02 --strain-y 0.01
--rotation 0.5` and default speckle/noise settings (8583 dots):

| Sampling | Wall time | Peak RSS |
|---|---|---|
| `analytic` | ≈ 3.1 s | ≈ 180 MiB |
| `bicubic` | ≈ 2.1 s | ≈ 180 MiB |

For reference, the 512×512, 5-image quick-start run takes about 0.4 s. The work is fully
vectorised (no per-pixel Python loops); the dot rendering only touches a small bounding
box per dot. Memory is dominated by the floating-point image grids and the
`displacement_field.npy` buffer (≈92 MB for 1024×1024 × 11 when that option is enabled).
Encoding the default `sequence.mp4` adds only ≈0.2 s (≈0.9 MB) to the 1024×1024, 11-image
run, since `ffmpeg` is fast and the frames are streamed straight to it.

---

## Tests

The test suite lives in `tests/` and is wired up through `pyproject.toml`
(`testpaths = ["tests"]`). From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # numpy, scipy, pillow, pytest
pip install -e .
pytest                                 # or: python -m pytest -v
```

A full run takes only a few seconds (204 tests, ≈7 s on the development machine; the
images used by the tests are small, 96–256 px). The modules are:

| File | Coverage |
|---|---|
| `tests/test_deformation.py` | load stepping, `F_i`/`x`/`u` formulas, polar decomposition, forward↔inverse round-trips (affine + sinusoid), validation errors |
| `tests/test_speckle.py` | dot sampling and determinism, radius/amplitude ranges, `taper`/`gaussian` profiles, union (max) rule, coverage model and calibration |
| `tests/test_imagegen.py` | pixel grid, validity masks, analytic-vs-bicubic agreement, out-of-bounds fill, noise statistics, 8-bit PNG output |
| `tests/test_groundtruth.py` | sample-grid layout, per-image/per-grid CSV contents against independently recomputed formulas, `.npy` layout, metadata stability, SHA-256 manifest |
| `tests/test_cleanup.py` | stale-file deletion, foreign-content refusal (file, sub-directory, symlink), `--overwrite`, `--dry-run`, safety guards |
| `tests/test_cli.py` | flag defaults and mapping, exit codes for invalid arguments, output toggles, determinism |
| `tests/test_integration.py` | the acceptance criteria below end to end, plus the metadata schema and manifest digests |
| `tests/test_verify.py` | the verifier: passes on correct datasets (including 40 % strain and sinusoid mode), fails on tampered/stale frames, CLI exit codes and JSON report |
| `tests/test_movie.py` | `sequence.mp4` encoding, byte-determinism, odd-size padding, metadata/manifest integration, cleanup recognition, `--no-movie` / `--movie-fps`, graceful handling of a missing `ffmpeg` (skipped when `ffmpeg` is absent) |

The suite is organised around the specification's acceptance criteria and covers:

1. **Undeformed reference** — for image 0 the ground-truth displacement is identically
   zero.
2. **Pure translation** — an image generated with a pure rigid-body translation matches a
   shifted reference to sub-pixel accuracy (a sanity check of the warping).
3. **Affine round-trip / recovery** — a small known affine deformation is recovered (at
   minimum, `F_i^{-1}(F_i(X)) == X` within tolerance, and the analytic displacement field
   is reproduced by a local correlation/optical-flow check).
4. **Determinism** — rerunning with the same seed and arguments yields identical files; a
   different seed yields different files.
5. **Output-directory cleanup** — stale generated files are removed and the requested
   file set remains, while foreign content is refused without `--overwrite`.

---

## Project layout

```
Synthetic_Data_Generation/
├── dicgen/
│   ├── __init__.py        # public API re-exports
│   ├── __main__.py        # `python -m dicgen`
│   ├── _version.py        # single source of the version string
│   ├── cleanup.py         # output-directory creation/cleanup + safety rules
│   ├── cli.py             # argparse CLI (`dicgen` console script)
│   ├── dataset.py         # orchestration: GenerationConfig, generate_dataset
│   ├── deformation.py     # affine + sinusoidal models, load stepping, GT fields
│   ├── groundtruth.py     # CSV/JSON/npy ground truth, metadata, sha256 manifest
│   ├── imagegen.py        # analytic/bicubic warping, noise, PNG & QA output
│   ├── movie.py           # sequence.mp4 encoding via ffmpeg
│   ├── speckle.py         # reference speckle field, coverage model & calibration
│   └── verify.py          # dataset verifier (`dicgen-verify` / `python -m dicgen.verify`)
├── tests/                 # pytest suite
├── examples/              # a small committed demo sequence (see below)
├── pyproject.toml         # packaging, console script, pytest config
├── requirements.txt       # numpy, scipy, pillow
├── requirements-dev.txt   # + pytest
├── LICENSE                # MIT
└── README.md
```

---

## Optional features and known limitations

### Present and supported

- **Sinusoidal (non-homogeneous) deformation** behind `--deformation-mode sinusoid`, with
  analytic Jacobian and a Newton-based inverse.
- **Rigid-body translation** separate from strain/rotation (`--translation-x/y`).
- **QA maps** (`--save-displacement-maps`): colour maps of `|u|` plus arrow overlays.
- **Specimen validity masks** (`masks/`, on by default) matching the
  `255 = valid / 0 = invalid` contract of a typical downstream DIC solver.
- **Full-grid `.npy` displacement field** (`--save-displacement-npy`).
- **An MP4 movie of the sequence** (`sequence.mp4`, on by default) so the deformation
  can be played back at a glance; it uses the external `ffmpeg` binary and degrades to a
  warning if that is missing.
- Two sampling methods (`analytic`, `bicubic`) and optional multiplicative noise.
- **Dataset verification** (`dicgen.verify` / `python -m dicgen.verify`): re-checks a
  generated directory against its own metadata with an image-alignment metric and a
  first-order local DIC recovery, and fails on stale/tampered frames.

### Known limitations

- **Strain reporting is small-strain / engineering.** The `eps_xx`, `eps_yy`,
  `gamma_xy` columns are the parameter (principal, engineering) strains; the
  image-frame small-strain tensor is also written (`eps_*_img`), but Hencky and
  Green–Lagrange strains must be recomputed from `F_i` if needed.
- **Rotation convention.** Because `y` points down, a positive `--rotation` appears
  clockwise on screen; the rigid rotation magnitude is exactly `theta * t_i`.
- **Bicubic border margin.** With `--sampling bicubic` the valid region is shrunk by 2 px
  to keep the spline stencil inside the reference; those border pixels are marked invalid.
- **The default model is affine only.** Spatial gradients other than the uniform affine
  field require `--deformation-mode sinusoid`.
- **Coverage is approximate.** The dot count is calibrated to within
  `--coverage-tolerance` (default `0.01`); the *achieved* coverage is reported, not
  guaranteed exactly.
- **Large strains change what is visible in a frame.** Because the affine field is centred
  on the image centre, a large stretch magnifies the central part of the reference into the
  whole frame: at `--strain-x 0.4` only ≈71 % of the reference width is still inside the
  last image (reported as "in-frame" by `dicgen.verify` and as `valid_fraction` in
  `ground_truth.csv`), and the outer material has moved out of the field of view. The
  ground truth remains exact everywhere the data exists; exclude the rest with the masks.
  Strains are reported as engineering strains of the parameterised stretch (see above).
- **The verifier needs `metadata.json`.** `dicgen.verify` reconstructs the deformation from
  the dataset's own metadata, so it cannot check a directory whose metadata was deleted or
  edited.
- **The movie needs `ffmpeg`.** `sequence.mp4` is encoded by the external `ffmpeg` binary
  (not a Python dependency). Without it the run still succeeds, with a warning and no
  movie; the movie is also only byte-reproducible on the same `ffmpeg` build, since
  H.264 encoding is not bit-stable across versions.

---

## Example dataset

A small demo sequence is committed under `examples/` so that the generator can be
inspected without running a full-size job: a 512×512, 5-image sequence (image 0 plus four
deformed images) with a 2% stretch in x, a −1% stretch in y and a 0.5° rotation. It is
reproduced exactly (byte-for-byte, including `sequence.mp4` on the same `ffmpeg` build) by:

```bash
python -m dicgen \
  --output-dir examples/demo_512x512 \
  --width 512 --height 512 -n 5 \
  --strain-x 0.02 --strain-y -0.01 --rotation 0.5 \
  --seed 0
```

The resulting directory contains `image0001.png … image0005.png`,
`masks/mask0001.png … mask0005.png`, `ground_truth.csv`, `ground_truth_grid.csv`,
`metadata.json` and `sequence.mp4` (14 files, ≈1.3 MB).