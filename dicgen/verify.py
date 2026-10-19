"""Verification of a *generated* dataset: do the pixels really contain the ground truth?

This module is the "second half" of the validation loop: it takes a directory
produced by :mod:`dicgen.dataset` and checks, from the pixel data alone, that

1. the frames actually differ from the reference (raw mean absolute difference),
2. the difference *is* the analytic deformation: pulling every image back onto
   the reference grid with the model recorded in ``metadata.json`` collapses the
   difference to the noise/interpolation floor (the "model-aligned" MAE),
3. a small first-order (affine) local DIC, started from a deliberately perturbed
   guess and run on the real pixels, recovers the analytic displacement and
   displacement gradient at a set of seed points.

Point 3 is the strongest check and it stays valid at large strains: a
first-order subset warp is exact for an affine deformation, so even a 40 %
stretch is recovered to a small fraction of a pixel.

Usage::

    python -m dicgen.verify --dataset examples/demo_512x512
    python -m dicgen.verify --dataset out/seq --json report.json

Exit code is 0 when every check passes and 1 otherwise, so it can be used in a
script that generates a dataset and then validates it.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

from ._version import __version__
from .deformation import load_fraction, make_model
from .imagegen import pixel_grid

__all__ = [
    "SeedCheck",
    "ImageCheck",
    "VerificationReport",
    "load_dataset_model",
    "verify_dataset",
    "format_report",
    "build_parser",
    "main",
]

LOGGER = logging.getLogger("dicgen.verify")

#: Mean grey-level difference below which two independent noise realisations are
#: considered identical (used only to report "no change" cases).
IDENTICAL_MAE = 0.05


@dataclass
class SeedCheck:
    """Result of one first-order local DIC measurement."""

    x: float
    y: float
    zncc: float
    u_gt: Tuple[float, float]
    u_measured: Tuple[float, float]
    u_error_px: float
    grad_gt: Tuple[float, float, float, float]
    grad_measured: Tuple[float, float, float, float]
    grad_error: float
    in_frame: bool = True
    note: str = ""


@dataclass
class ImageCheck:
    """Verification result for one deformed image."""

    index: int
    filename: str
    t: float
    mae_raw: float
    mae_aligned: float
    mae_aligned_limit: float
    aligned_fraction: float
    identical_to_reference: bool
    seeds: List[SeedCheck] = field(default_factory=list)
    u_error_max_px: float = float("nan")
    u_error_median_px: float = float("nan")
    strain_error_max: float = float("nan")
    strain_error_median: float = float("nan")
    zncc_median: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["seeds"] = [asdict(s) for s in self.seeds]
        return data


@dataclass
class VerificationReport:
    """Full verification report for a dataset directory."""

    dataset_dir: str
    tool: str = "dicgen.verify"
    version: str = __version__
    num_images: int = 0
    image_size: Tuple[int, int] = (0, 0)
    deformation: Dict[str, Any] = field(default_factory=dict)
    sampling: str = ""
    noise_sigma: float = 0.0
    thresholds: Dict[str, float] = field(default_factory=dict)
    images: List[ImageCheck] = field(default_factory=list)
    passed: bool = False
    warnings: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "dataset_dir": self.dataset_dir,
            "tool": self.tool,
            "version": self.version,
            "num_images": self.num_images,
            "image_size": list(self.image_size),
            "deformation": self.deformation,
            "sampling": self.sampling,
            "noise_sigma": self.noise_sigma,
            "thresholds": self.thresholds,
            "passed": self.passed,
            "warnings": self.warnings,
            "failures": self.failures,
            "images": [image.to_dict() for image in self.images],
        }
        return data


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


def load_dataset_model(dataset_dir: Path):
    """Rebuild the deformation model and the metadata of a generated dataset.

    Returns ``(metadata, model, filenames)``.
    """
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{metadata_path} not found: --dataset must point at a directory "
            "written by dicgen (it must contain metadata.json)"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sequence = metadata.get("sequence", {})
    args = metadata.get("arguments", {})
    deform = metadata.get("deformation", {})
    size = sequence.get("image_size_px") or [args.get("width"), args.get("height")]
    if not size or size[0] is None:
        raise ValueError("metadata.json does not record the image size")
    width, height = int(size[0]), int(size[1])
    filenames = sequence.get("filenames")
    if not filenames:
        num = int(sequence.get("num_images") or args.get("num_images") or 0)
        pad = max(4, len(str(num)))
        filenames = [f"image{i + 1:0{pad}d}.png" for i in range(num)]
    model = make_model(
        deform.get("mode", args.get("mode", "affine")),
        width,
        height,
        strain_x=deform.get("total_strain_x", args.get("strain_x", 0.0)),
        strain_y=deform.get("total_strain_y", args.get("strain_y", 0.0)),
        rotation_deg=deform.get("total_rotation_deg", args.get("rotation_deg", 0.0)),
        translation=(
            deform.get("total_translation_px", [0.0, 0.0])[0],
            deform.get("total_translation_px", [0.0, 0.0])[1],
        ),
        amplitude_x=deform.get("sin_amplitude_x_px", 0.0),
        amplitude_y=deform.get("sin_amplitude_y_px", 0.0),
        wavelength_x=deform.get("sin_wavelength_x_px"),
        wavelength_y=deform.get("sin_wavelength_y_px"),
    )
    return metadata, model, list(filenames)


def _read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L"), dtype=float)


# --------------------------------------------------------------------------- #
# First-order local DIC
# --------------------------------------------------------------------------- #


def _subset_offsets(half: int) -> Tuple[np.ndarray, np.ndarray]:
    ii, jj = np.meshgrid(
        np.arange(-half, half + 1, dtype=float),
        np.arange(-half, half + 1, dtype=float),
        indexing="xy",
    )
    return ii, jj


def measure_seed(
    reference: np.ndarray,
    deformed: np.ndarray,
    x0: np.ndarray,
    u_guess: np.ndarray,
    grad_guess: np.ndarray,
    *,
    half: int = 15,
    background: float = 128.0,
    perturb_px: float = 3.0,
    perturb_strain: float = 0.02,
    order: int = 3,
    prefilter: bool = False,
) -> Optional[SeedCheck]:
    """Recover ``u`` and ``du/dX`` at ``x0`` by first-order subset matching.

    The optimiser is deliberately started from a *perturbed* version of the
    analytic guess (``perturb_px`` pixels and ``perturb_strain`` added to both
    diagonal terms), so a successful recovery is a genuine measurement rather
    than a restatement of the ground truth.

    For ``order > 1`` the two images must already be spline-filtered
    (``scipy.ndimage.spline_filter(img, order)``) and ``prefilter`` left False:
    this function evaluates interpolation thousands of times, and re-filtering a
    full image on every evaluation is orders of magnitude slower.
    """
    height, width = reference.shape
    ii, jj = _subset_offsets(half)
    cx, cy = float(x0[0]), float(x0[1])
    if not (half <= cx < width - half and half <= cy < height - half):
        return None
    # Sample the reference subset on the continuous grid centred exactly on x0
    # (integer rounding here would introduce a systematic half-pixel offset).
    ref_sub = ndimage.map_coordinates(
        reference,
        [cy + jj, cx + ii],
        order=order,
        mode="constant",
        cval=background,
        prefilter=prefilter,
    )

    # perturbed initial guess (u, du/dx, du/dy, dv/dx, dv/dy)
    p0 = np.array(
        [
            u_guess[0] + perturb_px,
            u_guess[1] - perturb_px,
            grad_guess[0] + perturb_strain,
            grad_guess[1],
            grad_guess[2],
            grad_guess[3] + perturb_strain,
        ],
        dtype=float,
    )

    def deformed_subset(params: np.ndarray) -> np.ndarray:
        ux, uy, dux, duy, dvx, dvy = params
        xs = x0[0] + ux + (1.0 + dux) * ii + duy * jj
        ys = x0[1] + uy + dvx * ii + (1.0 + dvy) * jj
        return ndimage.map_coordinates(
            deformed,
            [ys, xs],
            order=order,
            mode="constant",
            cval=background,
            prefilter=prefilter,
        )

    def residual(params: np.ndarray) -> np.ndarray:
        return (deformed_subset(params) - ref_sub).ravel()

    try:
        solution = optimize.least_squares(
            residual,
            p0,
            method="lm",
            xtol=1e-10,
            ftol=1e-10,
            diff_step=1e-4,
        )
    except Exception as exc:  # pragma: no cover - numerical guard
        return SeedCheck(
            x=float(x0[0]),
            y=float(x0[1]),
            zncc=float("nan"),
            u_gt=(float(u_guess[0]), float(u_guess[1])),
            u_measured=(float("nan"), float("nan")),
            u_error_px=float("nan"),
            grad_gt=tuple(float(v) for v in grad_guess),
            grad_measured=(float("nan"),) * 4,
            grad_error=float("nan"),
            in_frame=False,
            note=f"optimiser failed: {exc}",
        )

    p = solution.x
    subset = deformed_subset(p)
    a = ref_sub - ref_sub.mean()
    b = subset - subset.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    zncc = float((a * b).sum() / denom) if denom > 0 else float("nan")
    grad_measured = (p[2], p[3], p[4], p[5])
    u_error = float(
        math.hypot(p[0] - u_guess[0], p[1] - u_guess[1])
    )
    grad_error = float(np.max(np.abs(np.asarray(grad_measured) - np.asarray(grad_guess))))
    return SeedCheck(
        x=float(x0[0]),
        y=float(x0[1]),
        zncc=zncc,
        u_gt=(float(u_guess[0]), float(u_guess[1])),
        u_measured=(float(p[0]), float(p[1])),
        u_error_px=u_error,
        grad_gt=tuple(float(v) for v in grad_guess),
        grad_measured=tuple(float(v) for v in grad_measured),
        grad_error=grad_error,
    )


def _seed_points(
    model,
    t: float,
    width: int,
    height: int,
    count: int,
    *,
    half: int,
    margin: float,
) -> List[np.ndarray]:
    """Seed points that stay inside the reference *and* whose deformed subset fits."""
    n = max(2, int(math.ceil(math.sqrt(count))))
    xs = np.linspace(margin + half, width - 1 - margin - half, n)
    ys = np.linspace(margin + half, height - 1 - margin - half, max(2, int(round(n * height / width))))
    pts = [np.array([x, y]) for y in ys for x in xs]
    kept = []
    for point in pts:
        deformed = model.transform(point[None, :], t)[0]
        if (
            deformed[0] - half - 2 >= 0
            and deformed[0] + half + 2 <= width - 1
            and deformed[1] - half - 2 >= 0
            and deformed[1] + half + 2 <= height - 1
        ):
            kept.append(point)
    # keep a spatially even subset if too many pass
    if len(kept) > count:
        step = len(kept) / float(count)
        kept = [kept[int(i * step)] for i in range(count)]
    return kept


# --------------------------------------------------------------------------- #
# Main verification routine
# --------------------------------------------------------------------------- #


def verify_dataset(
    dataset_dir: Path,
    *,
    seed_points: int = 12,
    window: int = 31,
    perturb_px: float = 3.0,
    perturb_strain: float = 0.02,
    max_images: Optional[int] = None,
    max_aligned_mae: Optional[float] = None,
    aligned_error_floor: float = 3.0,
    aligned_error_ratio: float = 0.15,
    max_displacement_error: float = 0.1,
    max_strain_error: float = 0.01,
    check_all_images: bool = True,
) -> VerificationReport:
    """Verify a generated dataset (see the module docstring).

    The "model-aligned" criterion is scale free, because the residual of
    resampling an image at sub-pixel/sub-strain offsets grows with the image
    detail (dot size vs. pixel size) and with the noise::

        aligned MAE  <=  max(aligned_error_floor,
                             aligned_error_ratio * raw MAE)  +  3 * noise_sigma

    i.e. undoing the recorded deformation must collapse the difference to the
    interpolation/noise floor, whatever the image size.  Pass ``max_aligned_mae``
    to use a single absolute tolerance instead.
    """
    dataset_dir = Path(dataset_dir)
    metadata, model, filenames = load_dataset_model(dataset_dir)
    width, height = model.width, model.height
    noise_sigma = float(metadata.get("noise", {}).get("additive_sigma_grey_levels", 0.0))
    sampling = str(metadata.get("sampling", {}).get("method", ""))
    noise_allowance = 3.0 * noise_sigma

    report = VerificationReport(
        dataset_dir=str(dataset_dir),
        num_images=len(filenames),
        image_size=(width, height),
        deformation=dict(metadata.get("deformation", {})),
        sampling=sampling,
        noise_sigma=noise_sigma,
        thresholds={
            "aligned_error_floor": float(aligned_error_floor),
            "aligned_error_ratio": float(aligned_error_ratio),
            "aligned_error_noise_allowance": float(noise_allowance),
            "max_aligned_mae": (
                None if max_aligned_mae is None else float(max_aligned_mae)
            ),
            "max_displacement_error_px": float(max_displacement_error),
            "max_strain_error": float(max_strain_error),
            "seed_points": int(seed_points),
            "window_px": int(window),
        },
    )

    background = float(metadata.get("speckle", {}).get("background_grey", 128.0))
    reference_path = dataset_dir / filenames[0]
    if not reference_path.is_file():
        raise FileNotFoundError(f"reference image {reference_path} not found")
    reference = _read_gray(reference_path)
    # Spline prefiltering is O(image); do it once per image instead of once per
    # DIC evaluation (that was the difference between seconds and minutes).
    reference_filtered = ndimage.spline_filter(reference, order=3, mode="mirror")
    grid = pixel_grid(width, height)
    max_window = max(9, (min(width, height) // 3) | 1)
    if int(window) > max_window:
        report.warnings.append(
            f"window {int(window)} px does not fit a {width}x{height} image; "
            f"using {max_window} px instead"
        )
        window = max_window
    half = int(window) // 2

    indices = range(len(filenames))
    if not check_all_images:
        indices = [len(filenames) - 1]
    if max_images is not None:
        indices = list(indices)[: int(max_images)]

    for i in indices:
        t = load_fraction(i, len(filenames))
        path = dataset_dir / filenames[i]
        if not path.is_file():
            report.failures.append(f"missing image file: {filenames[i]}")
            continue
        image = _read_gray(path)
        image_filtered = ndimage.spline_filter(image, order=3, mode="mirror")
        mae_raw = float(np.abs(image - reference).mean())

        # --- model-aligned comparison (pull the image back onto the reference) ---
        deformed_coords = model.transform(grid, t)
        inside = (
            (deformed_coords[..., 0] >= 0)
            & (deformed_coords[..., 0] <= width - 1)
            & (deformed_coords[..., 1] >= 0)
            & (deformed_coords[..., 1] <= height - 1)
        )
        pulled = ndimage.map_coordinates(
            image_filtered,
            [deformed_coords[..., 1], deformed_coords[..., 0]],
            order=3,
            mode="constant",
            cval=background,
            prefilter=False,
        )
        aligned_fraction = float(np.mean(inside))
        mae_aligned = (
            float(np.abs(reference - pulled)[inside].mean()) if inside.any() else float("nan")
        )
        mae_limit = (
            float(max_aligned_mae)
            if max_aligned_mae is not None
            else max(aligned_error_floor, aligned_error_ratio * mae_raw) + noise_allowance
        )

        check = ImageCheck(
            index=i,
            filename=filenames[i],
            t=t,
            mae_raw=mae_raw,
            mae_aligned=mae_aligned,
            mae_aligned_limit=mae_limit,
            aligned_fraction=aligned_fraction,
            identical_to_reference=bool(mae_raw <= IDENTICAL_MAE),
        )

        # --- first-order local DIC at seed points ---
        if i > 0 and not check.identical_to_reference:
            for point in _seed_points(
                model, t, width, height, seed_points, half=half, margin=2.0
            ):
                u_gt = model.displacement(point[None, :], t)[0]
                J = model.jacobian(point[None, :], t)[0]
                grad_gt = (J[0, 0] - 1.0, J[0, 1], J[1, 0], J[1, 1] - 1.0)
                seed = measure_seed(
                    reference_filtered,
                    image_filtered,
                    point,
                    u_gt,
                    grad_gt,
                    half=half,
                    background=background,
                    perturb_px=perturb_px,
                    perturb_strain=perturb_strain,
                    prefilter=False,
                )
                if seed is not None:
                    check.seeds.append(seed)
            if check.seeds:
                u_errors = [s.u_error_px for s in check.seeds]
                g_errors = [s.grad_error for s in check.seeds]
                znccs = [s.zncc for s in check.seeds]
                check.u_error_max_px = float(np.nanmax(u_errors))
                check.u_error_median_px = float(np.nanmedian(u_errors))
                check.strain_error_max = float(np.nanmax(g_errors))
                check.strain_error_median = float(np.nanmedian(g_errors))
                check.zncc_median = float(np.nanmedian(znccs))
            else:
                report.warnings.append(
                    f"{filenames[i]}: no seed point kept its subset inside the frame"
                )

        report.images.append(check)

        # --- pass/fail ---
        if check.identical_to_reference and i > 0:
            report.failures.append(
                f"{filenames[i]}: identical to the reference image (MAE {mae_raw:.3f}) "
                "although t = "
                f"{t:.3f} > 0"
            )
        if not math.isnan(mae_aligned) and mae_aligned > check.mae_aligned_limit:
            report.failures.append(
                f"{filenames[i]}: model-aligned mean error {mae_aligned:.2f} grey levels "
                f"exceeds the tolerance {check.mae_aligned_limit:.2f} -- the image does not "
                "match the recorded deformation"
            )
        if check.u_error_max_px == check.u_error_max_px:  # not NaN
            if check.u_error_max_px > max_displacement_error:
                report.failures.append(
                    f"{filenames[i]}: local DIC displacement error "
                    f"{check.u_error_max_px:.3f} px exceeds {max_displacement_error:.3f} px"
                )
        if check.strain_error_max == check.strain_error_max:
            if check.strain_error_max > max_strain_error:
                report.failures.append(
                    f"{filenames[i]}: local DIC gradient/strain error "
                    f"{check.strain_error_max:.4f} exceeds {max_strain_error:.4f}"
                )

    report.passed = not report.failures
    return report


def format_report(report: VerificationReport, *, verbose: bool = True) -> str:
    """Human-readable verification report."""
    lines: List[str] = []
    lines.append(f"dicgen.verify {report.version} -- {report.dataset_dir}")
    lines.append(
        f"  {report.num_images} images, {report.image_size[0]}x{report.image_size[1]} px, "
        f"deformation = {report.deformation.get('mode', '?')}, "
        f"sampling = {report.sampling or '?'}, noise sigma = {report.noise_sigma:g}"
    )
    lines.append(
        "  model-aligned criterion: MAE after undoing the model <= max("
        f"{report.thresholds.get('aligned_error_floor', float('nan')):.1f}, "
        f"{report.thresholds.get('aligned_error_ratio', float('nan')):.2f} * |diff|) + "
        f"{report.thresholds.get('aligned_error_noise_allowance', 0.0):.1f} (noise)"
        + (
            f"; DIC |du| <= {report.thresholds.get('max_displacement_error_px', float('nan')):.3f} px, "
            f"DIC |dgrad| <= {report.thresholds.get('max_strain_error', float('nan')):.4f}"
        )
    )
    lines.append("")
    header = (
        "  img    t     |diff|  aligned   limit  in-frame   DIC |du|max   DIC |dgrad|max   ZNCC"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for image in report.images:
        if image.seeds:
            lines.append(
                f"  {image.index + 1:4d} {image.t:6.3f} {image.mae_raw:9.2f} "
                f"{image.mae_aligned:8.2f} {image.mae_aligned_limit:7.2f} "
                f"{100 * image.aligned_fraction:8.1f}% "
                f"{image.u_error_max_px:13.4f} {image.strain_error_max:15.4f} "
                f"{image.zncc_median:7.3f}"
            )
        else:
            lines.append(
                f"  {image.index + 1:4d} {image.t:6.3f} {image.mae_raw:9.2f} "
                f"{image.mae_aligned:8.2f} {image.mae_aligned_limit:7.2f} "
                f"{100 * image.aligned_fraction:8.1f}% "
                f"{'-':>13} {'-':>15} {'-':>7}"
            )
        if verbose and image.seeds:
            worst = max(image.seeds, key=lambda s: s.u_error_px)
            lines.append(
                f"        worst seed at ({worst.x:.0f}, {worst.y:.0f}): "
                f"u_gt = ({worst.u_gt[0]:+.3f}, {worst.u_gt[1]:+.3f}) px, "
                f"u_measured = ({worst.u_measured[0]:+.3f}, {worst.u_measured[1]:+.3f}) px"
            )
    lines.append("")
    if report.warnings:
        lines.append("  warnings:")
        lines.extend(f"    - {w}" for w in report.warnings)
    if report.failures:
        lines.append("  FAILED:")
        lines.extend(f"    - {f}" for f in report.failures)
    else:
        lines.append(
            "  PASSED: every image differs from the reference, matches the recorded "
            "deformation, and the local DIC recovers the analytic fields."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dicgen.verify",
        description=(
            "Check that a dataset generated by dicgen really contains the deformation "
            "recorded in its metadata.json (model-aligned image comparison plus a "
            "first-order local DIC recovery at seed points)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, required=True, help="directory written by dicgen")
    parser.add_argument("--seed-points", type=int, default=12, help="number of DIC seed points per image")
    parser.add_argument("--window", type=int, default=31, help="subset size in pixels (odd-ish)")
    parser.add_argument("--perturb-px", type=float, default=3.0, help="DIC initial-guess offset in pixels")
    parser.add_argument(
        "--perturb-strain", type=float, default=0.02, help="DIC initial-guess strain error"
    )
    parser.add_argument("--max-images", type=int, default=None, help="only check the first N images")
    parser.add_argument(
        "--max-aligned-error",
        type=float,
        default=None,
        help="absolute tolerance for the model-aligned MAE (default: scale-free rule)",
    )
    parser.add_argument(
        "--aligned-error-floor",
        type=float,
        default=3.0,
        help="floor of the scale-free model-aligned tolerance, in grey levels",
    )
    parser.add_argument(
        "--aligned-error-ratio",
        type=float,
        default=0.15,
        help="model-aligned MAE must stay below this fraction of the raw difference",
    )
    parser.add_argument("--max-displacement-error", type=float, default=0.1, help="[px]")
    parser.add_argument("--max-strain-error", type=float, default=0.01, help="[-]")
    parser.add_argument("--last-only", action="store_true", help="only check the last image")
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format="%(message)s")
    try:
        report = verify_dataset(
            args.dataset,
            seed_points=args.seed_points,
            window=args.window,
            perturb_px=args.perturb_px,
            perturb_strain=args.perturb_strain,
            max_images=args.max_images,
            max_aligned_mae=args.max_aligned_error,
            aligned_error_floor=args.aligned_error_floor,
            aligned_error_ratio=args.aligned_error_ratio,
            max_displacement_error=args.max_displacement_error,
            max_strain_error=args.max_strain_error,
            check_all_images=not args.last_only,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(format_report(report, verbose=False))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not args.quiet:
            print(f"  JSON report written to {args.json}")
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())