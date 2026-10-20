---
name: dicgen
description: Use when generating, inspecting, verifying, or modifying synthetic 2D DIC datasets in this repository — including running the dicgen CLI, regenerating examples/demo_512x512, interpreting ground_truth.csv / ground_truth_grid.csv / masks, or changing the deformation, speckle, sampling, noise, or output code.
---

# dicgen — synthetic 2D DIC dataset generator

`dicgen` generates a grayscale speckle image sequence with an exact analytic
ground truth (displacement, Jacobian, strain) and a SHA-256 manifest. The
repository README is the specification; this skill is the short operational
guide and lists the invariants that are easy to get wrong.

## Setup

Run everything from the repository root. Prefer the existing virtualenv:

```bash
.venv/bin/python -m dicgen --help
.venv/bin/python -m dicgen.verify --help
```

If `.venv` is missing or incomplete:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install -e .
```

Do not install into the system Python and do not commit `.venv/`.

## Generate a dataset

Write scratch output to a gitignored directory (`out/`, `data/`, or `output/`),
or under `/tmp`. `examples/*` is gitignored except the committed
`examples/demo_512x512/`.

```bash
# small, fast run
.venv/bin/python -m dicgen \
  --output-dir out/demo -n 5 --width 512 --height 512 \
  --strain-x 0.02 --seed 0

# full-size run with rotation and translation
.venv/bin/python -m dicgen \
  --output-dir out/run01 -n 11 --width 1024 --height 1024 \
  --strain-x 0.02 --strain-y 0.01 --rotation 0.5 \
  --translation-x 1.5 --seed 0
```

Useful variants:

- Noise sweep: `--noise-sigma 0` … `8` with the same `--seed`.
- Non-uniform field: `--deformation-mode sinusoid --sin-amplitude-x 4`.
- Bias-free vs interpolation-biased: `--sampling analytic` (default) vs
  `--sampling bicubic`.
- Extra outputs: `--save-displacement-npy`, `--save-displacement-maps`.
- Preview cleanup: `--dry-run` before a run that may overwrite an existing
  directory.

`--output-dir` is required. Existing recognised generated files are replaced;
foreign content causes the run to be refused unless `--overwrite` is passed.
Never point it at the repository root, the home directory, or the current
working directory. Exit codes: `0` success, `2` argument/validation error,
`3` OS/runtime error.

## Verify generated data

Always verify numerically; the speckle pattern is visually self-similar, so
"the frames look the same" is not evidence of a problem.

```bash
.venv/bin/python -m dicgen.verify --dataset out/demo
.venv/bin/python -m dicgen.verify --dataset out/demo --json out/report.json
```

The verifier uses the dataset's own `metadata.json`, checks image difference
before/after undoing the recorded deformation, and runs a local first-order DIC
recovery. It exits `0` on pass, `1` on a failed check, `2` if the dataset cannot
be read. It cannot verify a dataset whose metadata was deleted or edited.

## Invariants

- **Coordinates:** `(x, y)` means `(column, row)`; `y` points down. The image
  centre is `((width - 1) / 2, (height - 1) / 2)`. A positive `--rotation`
  therefore looks clockwise on screen.
- **Load stepping:** image index `i = 0 … n-1` has `t_i = i / (n - 1)`.
  `image0001.png` is the undeformed reference; the last image carries the full
  requested deformation. The deformation is divided equally over the sequence.
- **Affine model:** `F_i = R(theta*t_i) @ diag(1 + e_x*t_i, 1 + e_y*t_i)` and
  `x = center + F_i (X - center) + t_i*T`. `u = x - X`.
- **Strain columns:** `eps_xx`, `eps_yy`, `gamma_xy` are the parameter /
  principal engineering strains from the polar decomposition, not the
  image-frame tensor. The image-frame small-strain tensor components are
  `eps_xx_img`, `eps_yy_img`, `gamma_xy_img`. Do not relabel or "correct" them.
  Hencky and Green–Lagrange strains must be recomputed from `F_i`.
- **Masks:** `masks/mask####.png` uses `255 = valid` and `0 = invalid`.
  Invalid pixels came from outside the reference (or, for bicubic sampling,
  within 2 px of the border).
- **Sampling:** `analytic` evaluates the continuous dot field at the
  inverse-mapped coordinates and is bias-free. `bicubic` rasterises the
  reference once and resamples it, reproducing interpolation smoothing/bias.
- **Speckle:** the same dot set is advected through every frame; it is never
  re-randomised per image. Overlapping dots combine with a max/union rule.
- **Determinism:** every random draw comes from one seeded
  `numpy.random.Generator`, in a fixed order, and no wall-clock timestamps are
  written. Do not add timestamps, reorder draws, or introduce unseeded
  randomness. The same seed and arguments (and the same output path) must
  produce byte-identical files.
- **Manifest:** `metadata.json` records the arguments, model, realised speckle
  statistics, coverage calibration, per-image grey-level statistics, and a
  sorted SHA-256 manifest of every generated file except `metadata.json`
  itself. Keep it deterministic (sorted keys, 2-space indent).
- **Movie:** `sequence.mp4` is best-effort and requires the external `ffmpeg`
  binary. Without it, the run still succeeds with a warning and no movie.
  H.264 is not bit-stable across `ffmpeg` builds, so the movie is only
  byte-reproducible on the machine that generated it.

## Regenerate the committed example

Do not hand-edit files under `examples/`. Regenerate the demo with its exact
documented arguments:

```bash
.venv/bin/python -m dicgen \
  --output-dir examples/demo_512x512 \
  --width 512 --height 512 -n 5 \
  --strain-x 0.02 --strain-y -0.01 --rotation 0.5 \
  --seed 0
```

Then verify it and check the working tree diff.

## Modify or extend the package

- Read the relevant README section first; it is the behavioural specification.
- Keep the module layout: `deformation.py` (models), `speckle.py` (dots and
  coverage), `imagegen.py` (warping, noise, masks, PNG/QA), `groundtruth.py`
  (CSVs, metadata, manifest), `dataset.py` (orchestration), `cli.py`,
  `verify.py`, `movie.py`, `cleanup.py`.
- New public API belongs in `dicgen/__init__.py` with an explicit `__all__`
  entry.
- Add or update tests for any behaviour change, especially determinism,
  metadata schema/manifest, cleanup safety, and CLI exit codes.
- Run the suite before claiming success:

```bash
.venv/bin/python -m pytest
```

For changes to generated output, regenerate a scratch dataset and run
`dicgen.verify` as well. For changes affecting the demo, regenerate the demo
and verify it.

## Source of truth

- `README.md` — CLI reference, model formulas, file formats, conventions.
- `tests/` — executable acceptance criteria; `tests/test_integration.py` and
  `tests/test_verify.py` are the end-to-end checks.
- `dicgen/__init__.py` — public API surface.
- `metadata.json` — the record of what a particular run actually did.
