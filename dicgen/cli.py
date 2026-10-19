"""Command-line interface of ``dicgen``.

Typical use::

    python -m dicgen --output-dir data/sequence1 -n 11 --strain-x 0.02

The CLI is fully non-interactive: it never asks questions, and it refuses to
delete anything it did not generate unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from ._version import __version__
from .cleanup import ForeignContentError
from .dataset import TOOL_NAME, GenerationConfig, GenerationSummary, generate_dataset
from .deformation import DeformationError

__all__ = ["build_parser", "config_from_args", "main"]

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_RUNTIME = 3


def build_parser() -> argparse.ArgumentParser:
    """Build the ``argparse`` parser (every option documented in ``--help``)."""
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description=(
            "Generate a synthetic 2D DIC validation dataset: a grayscale speckle "
            "image sequence plus the exact analytic ground-truth displacement and "
            "strain fields."
        ),
        epilog=(
            "Deformation model (default): t_i = i/(n-1); "
            "F_i = R(theta*t_i) @ diag(1+e_x*t_i, 1+e_y*t_i); "
            "x = center + F_i (X - center) + t_i*T; u(X) = x - X; "
            "eps_xx = e_x*t_i, eps_yy = e_y*t_i, gamma_xy = 0. "
            "Image 0 is the undeformed reference. "
            "The output directory is cleaned of previously generated files; "
            "unrelated content is never deleted without --overwrite."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{TOOL_NAME} {__version__}")

    g_geo = parser.add_argument_group("image geometry and sequence")
    g_geo.add_argument("--width", type=int, default=1024, help="image width in pixels")
    g_geo.add_argument("--height", type=int, default=1024, help="image height in pixels")
    g_geo.add_argument(
        "-n",
        "--num-images",
        type=int,
        default=11,
        help="total number of images, including the undeformed reference (image 0)",
    )
    g_geo.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for the images and ground truth (created if missing)",
    )

    g_def = parser.add_argument_group("deformation (totals over the whole sequence)")
    g_def.add_argument(
        "--strain-x",
        type=float,
        default=0.0,
        help="total affine engineering stretch in x (0.02 = 2%%)",
    )
    g_def.add_argument(
        "--strain-y", type=float, default=0.0, help="total affine engineering stretch in y"
    )
    g_def.add_argument(
        "--rotation", type=float, default=0.0, help="total rotation in degrees"
    )
    g_def.add_argument(
        "--translation-x",
        type=float,
        default=0.0,
        help="total rigid-body translation in x, in pixels",
    )
    g_def.add_argument(
        "--translation-y",
        type=float,
        default=0.0,
        help="total rigid-body translation in y, in pixels",
    )
    g_def.add_argument(
        "--deformation-mode",
        choices=("affine", "sinusoid"),
        default="affine",
        help="deformation model; 'sinusoid' adds a non-homogeneous field",
    )
    g_def.add_argument(
        "--sin-amplitude-x",
        type=float,
        default=0.0,
        help="sinusoid mode: displacement amplitude in x, in pixels",
    )
    g_def.add_argument(
        "--sin-amplitude-y",
        type=float,
        default=0.0,
        help="sinusoid mode: displacement amplitude in y, in pixels",
    )
    g_def.add_argument(
        "--sin-wavelength-x",
        type=float,
        default=None,
        help="sinusoid mode: wavelength in x, in pixels (default: image width)",
    )
    g_def.add_argument(
        "--sin-wavelength-y",
        type=float,
        default=None,
        help="sinusoid mode: wavelength in y, in pixels (default: image height)",
    )

    g_speckle = parser.add_argument_group("speckle pattern")
    g_speckle.add_argument(
        "--dot-radius-min", type=float, default=3.0, help="minimum dot radius, in pixels"
    )
    g_speckle.add_argument(
        "--dot-radius-max", type=float, default=6.0, help="maximum dot radius, in pixels"
    )
    g_speckle.add_argument(
        "--dot-coverage",
        type=float,
        default=0.5,
        help="target fraction of the image area covered by dots (0 < c < 0.95)",
    )
    g_speckle.add_argument(
        "--dot-profile",
        choices=("taper", "gaussian"),
        default="taper",
        help="analytic intensity profile of a dot",
    )
    g_speckle.add_argument(
        "--dot-softness",
        type=float,
        default=0.7,
        help="taper profile: edge softness in pixels (sub-pixel anti-aliasing)",
    )
    g_speckle.add_argument(
        "--dot-amplitude-min",
        type=float,
        default=70.0,
        help="minimum dot darkening relative to the background, in grey levels",
    )
    g_speckle.add_argument(
        "--dot-amplitude-max",
        type=float,
        default=120.0,
        help="maximum dot darkening relative to the background, in grey levels",
    )
    g_speckle.add_argument(
        "--background",
        type=float,
        default=128.0,
        help="background grey level of the speckle pattern (0-255)",
    )
    g_speckle.add_argument(
        "--min-dot-distance",
        type=float,
        default=0.0,
        help="minimum distance between dot centres in pixels (0 = unconstrained)",
    )
    g_speckle.add_argument(
        "--coverage-tolerance",
        type=float,
        default=0.01,
        help="stop calibrating the dot count once the measured coverage is this close",
    )

    g_img = parser.add_argument_group("sampling and noise")
    g_img.add_argument(
        "--sampling",
        choices=("analytic", "bicubic"),
        default="analytic",
        help="analytic = evaluate the continuous speckle field (exact); "
        "bicubic = resample a rasterised reference with a bicubic spline",
    )
    g_img.add_argument(
        "--interpolation-order",
        type=int,
        default=3,
        help="spline order used by --sampling bicubic (1-5)",
    )
    g_img.add_argument(
        "--noise-sigma",
        type=float,
        default=2.0,
        help="standard deviation of the additive Gaussian grey-level noise (0 = noise free)",
    )
    g_img.add_argument(
        "--multiplicative-noise-sigma",
        type=float,
        default=0.0,
        help="relative sigma of the optional multiplicative speckle noise (0 = off)",
    )
    g_img.add_argument(
        "--seed", type=int, default=0, help="RNG seed; the run is fully reproducible"
    )

    g_out = parser.add_argument_group("outputs")
    # Two mutually exclusive flags per toggle.  The "off" variant carries
    # default=SUPPRESS so that ArgumentDefaultsHelpFormatter does not print a
    # misleading "(default: False)"; the "on" variant defines the real default.
    g_out.add_argument(
        "--save-ground-truth",
        dest="save_ground_truth",
        action="store_true",
        default=True,
        help="write ground_truth.csv / ground_truth_grid.csv",
    )
    g_out.add_argument(
        "--no-ground-truth",
        dest="save_ground_truth",
        action="store_false",
        default=argparse.SUPPRESS,
        help="do not write the ground-truth CSVs",
    )
    g_out.add_argument(
        "--save-masks",
        dest="save_masks",
        action="store_true",
        default=True,
        help="write masks/mask####.png validity masks, 255 = valid (use with the "
        "mask convention of the downstream DIC solver)",
    )
    g_out.add_argument(
        "--no-masks",
        dest="save_masks",
        action="store_false",
        default=argparse.SUPPRESS,
        help="do not write validity masks",
    )
    g_out.add_argument(
        "--save-displacement-maps",
        dest="save_qa_maps",
        action="store_true",
        default=False,
        help="write qa/ colour maps of |u| and arrow overlays (visual QA)",
    )
    g_out.add_argument(
        "--save-displacement-npy",
        action="store_true",
        default=False,
        help="write displacement_field.npy: (n, 2, height, width) float32 analytic u",
    )
    g_out.add_argument(
        "--save-movie",
        dest="save_movie",
        action="store_true",
        default=True,
        help="write sequence.mp4 next to the images (requires ffmpeg on PATH)",
    )
    g_out.add_argument(
        "--no-movie",
        dest="save_movie",
        action="store_false",
        default=argparse.SUPPRESS,
        help="do not write the sequence.mp4 movie",
    )
    g_out.add_argument(
        "--movie-fps",
        type=float,
        default=10.0,
        help="playback rate of the sequence.mp4 movie, in frames per second",
    )
    g_out.add_argument(
        "--gt-grid",
        type=int,
        nargs=2,
        metavar=("NX", "NY"),
        default=(16, 16),
        help="sample grid of the ground_truth_grid.csv output, in points",
    )
    g_out.add_argument(
        "--overwrite",
        action="store_true",
        help="delete everything inside --output-dir, including files not generated by dicgen",
    )
    g_out.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the arguments and report what would be deleted, then exit",
    )

    g_log = parser.add_argument_group("logging")
    g_log.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="logging verbosity",
    )
    g_log.add_argument(
        "-q", "--quiet", action="store_true", help="only warnings and errors (implies --log-level WARNING)"
    )
    g_log.add_argument(
        "-v", "--verbose", action="store_true", help="per-image debug logging"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> GenerationConfig:
    """Map parsed arguments onto a :class:`~dicgen.dataset.GenerationConfig`."""
    nx, ny = args.gt_grid
    return GenerationConfig(
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        num_images=args.num_images,
        strain_x=args.strain_x,
        strain_y=args.strain_y,
        rotation_deg=args.rotation,
        translation_x=args.translation_x,
        translation_y=args.translation_y,
        mode=args.deformation_mode,
        sin_amplitude_x=args.sin_amplitude_x,
        sin_amplitude_y=args.sin_amplitude_y,
        sin_wavelength_x=args.sin_wavelength_x,
        sin_wavelength_y=args.sin_wavelength_y,
        dot_radius_min=args.dot_radius_min,
        dot_radius_max=args.dot_radius_max,
        dot_coverage=args.dot_coverage,
        dot_profile=args.dot_profile,
        dot_softness=args.dot_softness,
        dot_amplitude_min=args.dot_amplitude_min,
        dot_amplitude_max=args.dot_amplitude_max,
        background=args.background,
        min_dot_distance=args.min_dot_distance,
        coverage_tolerance=args.coverage_tolerance,
        sampling=args.sampling,
        interpolation_order=args.interpolation_order,
        noise_sigma=args.noise_sigma,
        noise_multiplicative_sigma=args.multiplicative_noise_sigma,
        seed=args.seed,
        save_ground_truth=args.save_ground_truth,
        save_masks=args.save_masks,
        save_qa_maps=args.save_qa_maps,
        save_displacement_npy=args.save_displacement_npy,
        save_movie=args.save_movie,
        movie_fps=args.movie_fps,
        gt_grid_nx=int(nx),
        gt_grid_ny=int(ny),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def _configure_logging(level_name: str, quiet: bool, verbose: bool) -> logging.Logger:
    """Log to stdout (so piping the report is easy) with a plain message format."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    if quiet:
        level = max(level, logging.WARNING)
    if verbose:
        level = logging.DEBUG
    logger = logging.getLogger("dicgen")
    logger.handlers.clear()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def _print_summary(summary: GenerationSummary) -> None:
    """Human-readable end-of-run report."""
    print("")
    print("Generation summary")
    print("------------------")
    print(f"  output directory : {summary.output_dir}")
    print(f"  images           : {summary.num_images} x {summary.image_size[0]}x{summary.image_size[1]} px")
    print(f"  files written    : {len(summary.files)}")
    if summary.wrote_files:
        print(f"  dot coverage     : {summary.coverage_achieved:.4f} "
              f"(target {summary.coverage_target:g}, {summary.num_dots} dots)")
        print(f"  mean valid area  : {100.0 * summary.valid_fraction_mean:.2f} %")
        print(f"  max |u|          : {summary.displacement_max_px:.4f} px")
        print(f"  noise sigma      : {summary.noise_sigma:g} grey levels "
              f"(multiplicative {summary.noise_multiplicative_sigma:g})")
        print(f"  seed             : {summary.seed}")
        if summary.movie_written:
            print(f"  movie            : {summary.movie_filename} "
                  f"({summary.movie_fps:g} fps)")
        elif summary.movie_filename is None:
            print(f"  movie            : not written (ffmpeg unavailable or disabled)")
        print(f"  elapsed          : {summary.elapsed_seconds:.2f} s")
    else:
        print("  (dry run: nothing was written)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logger = _configure_logging(args.log_level, args.quiet, args.verbose)
    try:
        config = config_from_args(args)
        summary = generate_dataset(config, logger=logger)
    except (ValueError, DeformationError, ForeignContentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
    _print_summary(summary)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())