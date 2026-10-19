"""Shared helpers and fixtures for the ``dicgen`` test suite.

The measurement helpers are intentionally *independent* of the package's own
transform arithmetic wherever that is practical:

* :func:`estimate_translation` recovers a global translation between two images
  with an FFT cross-correlation refined by a locally upsampled DFT (sub-pixel
  accuracy well below 0.1 px);
* :func:`ncc_local_displacement` measures a local displacement by directly
  resampling the deformed image (``scipy.ndimage.map_coordinates``) and
  maximising a mean-subtracted normalised cross-correlation (NCC).

Both are standard DIC "measurement" tools written from scratch here, so a test
that compares them with the analytic ground truth genuinely checks the images
and the ground truth against each other rather than against themselves.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
from PIL import Image

from dicgen.dataset import GenerationConfig, generate_dataset

# --------------------------------------------------------------------------- #
# Small, fast, fully-specified generation configs
# --------------------------------------------------------------------------- #

#: Default geometry used by most tests: small images keep the suite fast while
#: still containing hundreds of speckle dots.
SMALL_SIZE = 128
SMALL_IMAGES = 4


def make_config(output_dir, **overrides) -> GenerationConfig:
    """Build a :class:`GenerationConfig` for a small, fast run.

    ``overrides`` are applied on top of sensible small defaults (128x128, four
    images, noise-free, seed 0) so each test only states what it cares about.
    """
    params = dict(
        width=SMALL_SIZE,
        height=SMALL_SIZE,
        num_images=SMALL_IMAGES,
        noise_sigma=0.0,
        seed=0,
        # Movie encoding needs the external `ffmpeg` binary and would slow every
        # test down; the movie feature has its own dedicated tests in
        # tests/test_movie.py, so the shared small config turns it off.
        save_movie=False,
    )
    params.update(overrides)
    return GenerationConfig(output_dir=Path(output_dir), **params)


@pytest.fixture
def generate(tmp_path):
    """Return a callable that generates a dataset into a fresh sub-directory.

    ``generate(name="run", **overrides) -> (output_dir, GenerationSummary)``.
    Each call uses its own directory so several runs can coexist in one test.
    """

    counter = {"n": 0}

    def _generate(name: Optional[str] = None, **overrides):
        if name is None:
            counter["n"] += 1
            name = f"run{counter['n']}"
        out = tmp_path / name
        config = make_config(out, **overrides)
        summary = generate_dataset(config)
        return out, summary

    return _generate


# --------------------------------------------------------------------------- #
# File / array helpers
# --------------------------------------------------------------------------- #


def read_gray(path) -> np.ndarray:
    """Open a PNG and return it as a 2-D ``uint8`` array."""
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def read_rgb(path) -> np.ndarray:
    """Open a colour PNG and return it as a ``(H, W, 3)`` ``uint8`` array."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def png_mode(path) -> str:
    with Image.open(path) as img:
        return img.mode


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root) -> Dict[str, str]:
    """Map every file below ``root`` to its SHA-256 digest (relative paths)."""
    root = Path(root)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def read_csv_rows(path) -> List[Dict[str, str]]:
    """Read a CSV into a list of dictionaries (all values stay strings)."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_header(path) -> List[str]:
    with open(path, newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def read_json(path):
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Independent sub-pixel measurement helpers
# --------------------------------------------------------------------------- #


def _subpixel_peak(ccf: np.ndarray, shape, span: float = 1.5, upsample: int = 100):
    """Refine the integer cross-correlation peak of ``ccf`` (2-D FFT) locally.

    Evaluates the inverse transform of ``ccf`` on a fine grid around the integer
    peak using a matrix-multiply DFT (thus upsampling the correlation surface by
    ``upsample``), which gives a smooth, bias-free peak location.
    """
    height, width = shape
    cc = np.fft.ifft2(ccf).real
    py, px = np.unravel_index(np.argmax(cc), cc.shape)
    dy = py if py <= height // 2 else py - height
    dx = px if px <= width // 2 else px - width

    fu = np.fft.fftfreq(width) * width
    fv = np.fft.fftfreq(height) * height
    sx = np.linspace(dx - span, dx + span, 2 * upsample + 1)
    sy = np.linspace(dy - span, dy + span, 2 * upsample + 1)
    phase_u = np.exp(2j * np.pi * np.outer(sx, fu) / width)  # (Sx, W)
    phase_v = np.exp(2j * np.pi * np.outer(sy, fv) / height)  # (Sy, H)
    surface = (phase_v @ ccf @ phase_u.T).real / (width * height)
    jy, jx = np.unravel_index(np.argmax(surface), surface.shape)
    return float(sx[jx]), float(sy[jy])


def estimate_translation(reference, image, *, margin: int = 0) -> np.ndarray:
    """Estimate the global translation of ``image`` relative to ``reference``.

    The returned vector is the shift ``u`` such that ``image(x) ~= reference(x -
    u)``, i.e. the analytic displacement ``u = x - X`` of the package (a feature
    at reference coordinate ``X`` appears at ``x = X + u``).

    A Hann window suppresses wraparound leakage and the peak is refined with a
    locally upsampled DFT, giving sub-0.05 px accuracy for these speckle images.
    """
    ref = np.asarray(reference, dtype=float)
    img = np.asarray(image, dtype=float)
    if margin:
        ref = ref[margin:-margin, margin:-margin]
        img = img[margin:-margin, margin:-margin]
    height, width = ref.shape
    win_y = np.hanning(height)[:, None]
    win_x = np.hanning(width)[None, :]
    ref = (ref - ref.mean()) * win_y * win_x
    img = (img - img.mean()) * win_y * win_x
    ccf = np.fft.fft2(img) * np.conj(np.fft.fft2(ref))
    sx, sy = _subpixel_peak(ccf, (height, width))
    return np.array([sx, sy], dtype=float)


def ncc_local_displacement(
    reference,
    image,
    point,
    guess=None,
    *,
    half: int = 25,
    coarse: float = 0.05,
    coarse_range: float = 1.5,
    fine: float = 0.01,
    fine_range: float = 0.1,
) -> np.ndarray:
    """Measure the local displacement at reference ``point`` via NCC.

    A ``(2*half+1)`` reference patch centred on ``point`` is matched against the
    deformed image resampled with bicubic interpolation; ``guess`` (the analytic
    displacement) only seeds the search, and the residual relative to it is the
    quantity of interest.  Returns the absolute displacement
    ``image_location - point``.
    """
    point = np.asarray(point, dtype=float)
    if guess is None:
        guess = np.zeros(2)
    guess = np.asarray(guess, dtype=float)
    x0, y0 = int(round(point[0])), int(round(point[1]))
    patch = np.asarray(reference, dtype=float)[y0 - half : y0 + half + 1,
                                               x0 - half : x0 + half + 1]
    offsets = np.arange(-half, half + 1, dtype=float)
    grid_x, grid_y = np.meshgrid(offsets, offsets)
    grid_x = grid_x.ravel()
    grid_y = grid_y.ravel()
    template = (patch - patch.mean()).ravel()
    template_norm = np.linalg.norm(template)

    def score(offsets_x: np.ndarray, offsets_y: np.ndarray) -> np.ndarray:
        cols = x0 + offsets_x[:, None] + grid_x[None, :]
        rows = y0 + offsets_y[:, None] + grid_y[None, :]
        import scipy.ndimage as ndi

        samples = ndi.map_coordinates(
            np.asarray(image, dtype=float),
            [rows.ravel(), cols.ravel()],
            order=3,
            mode="nearest",
        ).reshape(offsets_x.shape[0], -1)
        samples = samples - samples.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(samples, axis=1) * template_norm + 1e-12
        return (samples @ template) / denom

    grid = np.arange(-coarse_range, coarse_range + 1e-9, coarse)
    gx, gy = np.meshgrid(grid, grid)
    gx = gx.ravel() + guess[0]
    gy = gy.ravel() + guess[1]
    best = int(np.argmax(score(gx, gy)))
    bx, by = gx[best], gy[best]

    grid = np.arange(-fine_range, fine_range + 1e-9, fine)
    gx, gy = np.meshgrid(grid, grid)
    gx = gx.ravel() + bx
    gy = gy.ravel() + by
    best = int(np.argmax(score(gx, gy)))
    return np.array([gx[best], gy[best]])


# --------------------------------------------------------------------------- #
# White-box reproduction of the reference speckle (for image-0 checks)
# --------------------------------------------------------------------------- #


def expected_reference(config: GenerationConfig) -> np.ndarray:
    """Rebuild the exact float reference image that ``generate_dataset`` renders.

    It re-seeds a fresh generator with the same seed and replays the same
    dot-sampling/calibration procedure the package uses, so it is a faithful
    (if white-box) reconstruction of image 0 for the analytical-equality test.
    """
    from dicgen.imagegen import render_reference
    from dicgen.speckle import calibrate_dot_count

    speckle = config.speckle_params()
    rng = np.random.default_rng(config.seed)
    _, reference, _ = calibrate_dot_count(
        speckle, rng, lambda dots: render_reference(dots, speckle)
    )
    return reference