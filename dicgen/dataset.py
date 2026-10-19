"""Run orchestration: turn a :class:`GenerationConfig` into files on disk.

The dataset is produced by

1. validating the configuration;
2. preparing (creating/cleaning) the output directory (:mod:`dicgen.cleanup`);
3. sampling and calibrating the reference speckle dots (:mod:`dicgen.speckle`);
4. rendering the undeformed reference image (image 0);
5. warping it for every load step (image ``i``) and adding noise;
6. writing the images, the validity masks, the optional QA maps and the analytic
   ground truth / metadata.

Everything that is random comes from a single ``numpy.random.Generator`` seeded
with ``--seed``, and the order of the random draws is fixed by this function, so
repeating a run with identical arguments yields byte-identical files.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import groundtruth
from .cleanup import CleanupPlan, prepare_output_dir
from .deformation import load_fraction, make_model
from .imagegen import (
    apply_noise,
    pixel_grid,
    qa_displacement_image,
    reference_with_arrows,
    render_reference,
    save_gray_png,
    save_mask_png,
    save_rgb_png,
    warp_image,
)
from .speckle import SpeckleParams, calibrate_dot_count
from .movie import (
    DEFAULT_MOVIE_FPS,
    MOVIE_FILENAME,
    MovieError,
    write_movie,
)
from ._version import __version__

__all__ = ["GenerationConfig", "GenerationSummary", "generate_dataset", "TOOL_NAME"]

TOOL_NAME = "dicgen"
LOGGER = logging.getLogger("dicgen")

#: Filename prefix/padding of the image sequence (``image0001.png``, ...).
IMAGE_PREFIX = "image"

#: Arrow length (px) of the longest QA arrow.
QA_ARROW_LENGTH_PX = 42.0


@dataclass
class GenerationConfig:
    """All parameters of a single dataset generation run.

    The defaults match the command-line defaults: a 1024x1024, 11-image, 50 %
    coverage sequence with sigma = 2 grey levels of additive noise.
    """

    output_dir: Path

    # --- image geometry and sequence -------------------------------------
    width: int = 1024
    height: int = 1024
    num_images: int = 11

    # --- deformation -------------------------------------------------------
    strain_x: float = 0.0
    strain_y: float = 0.0
    rotation_deg: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0
    mode: str = "affine"
    sin_amplitude_x: float = 0.0
    sin_amplitude_y: float = 0.0
    sin_wavelength_x: Optional[float] = None
    sin_wavelength_y: Optional[float] = None

    # --- speckle -----------------------------------------------------------
    dot_radius_min: float = 3.0
    dot_radius_max: float = 6.0
    dot_coverage: float = 0.5
    dot_profile: str = "taper"
    dot_softness: float = 0.7
    dot_amplitude_min: float = 70.0
    dot_amplitude_max: float = 120.0
    background: float = 128.0
    min_dot_distance: float = 0.0
    coverage_tolerance: float = 0.01

    # --- sampling and noise ------------------------------------------------
    sampling: str = "analytic"
    interpolation_order: int = 3
    noise_sigma: float = 2.0
    noise_multiplicative_sigma: float = 0.0

    # --- reproducibility ---------------------------------------------------
    seed: int = 0

    # --- ground truth and outputs -----------------------------------------
    save_ground_truth: bool = True
    save_masks: bool = True
    save_qa_maps: bool = False
    save_displacement_npy: bool = False
    gt_grid_nx: int = 16
    gt_grid_ny: int = 16

    # --- movie -------------------------------------------------------------
    save_movie: bool = True
    movie_fps: float = DEFAULT_MOVIE_FPS

    # --- housekeeping ------------------------------------------------------
    overwrite: bool = False
    dry_run: bool = False

    # -------------------------------------------------------------- validation
    def validate(self) -> None:
        """Validate every parameter; raises :class:`ValueError` with a fix hint."""
        if self.width < 8 or self.height < 8:
            raise ValueError(
                f"--width/--height must be at least 8x8 px, got "
                f"{self.width}x{self.height}"
            )
        if self.num_images < 2:
            raise ValueError(
                f"--num-images must be >= 2 (image 0 is the undeformed reference "
                f"and t_i = i/(n-1)), got {self.num_images}"
            )
        if not np.isfinite([self.strain_x, self.strain_y, self.rotation_deg]).all():
            raise ValueError("--strain-x, --strain-y and --rotation must be finite")
        for name, value in (
            ("--noise-sigma", self.noise_sigma),
            ("--multiplicative-noise-sigma", self.noise_multiplicative_sigma),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.noise_sigma > 128.0:
            raise ValueError(
                f"--noise-sigma = {self.noise_sigma} is unreasonably large "
                "(grey levels are 0-255); keep it below ~128"
            )
        if self.sampling not in ("analytic", "bicubic"):
            raise ValueError(
                f"--sampling must be 'analytic' or 'bicubic', got {self.sampling!r}"
            )
        if not 1 <= self.interpolation_order <= 5:
            raise ValueError(
                f"--interpolation-order must lie in [1, 5], got "
                f"{self.interpolation_order}"
            )
        if self.gt_grid_nx < 2 or self.gt_grid_ny < 2:
            raise ValueError(
                f"--gt-grid must be at least 2x2 points, got "
                f"{self.gt_grid_nx}x{self.gt_grid_ny}"
            )
        if not np.isfinite(self.movie_fps) or self.movie_fps <= 0.0:
            raise ValueError(
                f"--movie-fps must be a positive number, got {self.movie_fps}"
            )
        if not isinstance(self.seed, (int, np.integer)):
            raise ValueError(f"--seed must be an integer, got {self.seed!r}")
        # The speckle and deformation models validate themselves; build them once
        # here so that a bad parameter fails before anything is written.
        self.speckle_params().validate()
        self.deformation_model()

    # ----------------------------------------------------------- sub-builders
    def speckle_params(self) -> SpeckleParams:
        """The :class:`~dicgen.speckle.SpeckleParams` described by this config."""
        return SpeckleParams(
            width=self.width,
            height=self.height,
            radius_min=self.dot_radius_min,
            radius_max=self.dot_radius_max,
            coverage=self.dot_coverage,
            background=self.background,
            amplitude_min=self.dot_amplitude_min,
            amplitude_max=self.dot_amplitude_max,
            profile=self.dot_profile,
            softness=self.dot_softness,
            min_distance=self.min_dot_distance,
            coverage_tolerance=self.coverage_tolerance,
        )

    def deformation_model(self):
        """The deformation model described by this config."""
        return make_model(
            self.mode,
            self.width,
            self.height,
            strain_x=self.strain_x,
            strain_y=self.strain_y,
            rotation_deg=self.rotation_deg,
            translation=(self.translation_x, self.translation_y),
            amplitude_x=self.sin_amplitude_x,
            amplitude_y=self.sin_amplitude_y,
            wavelength_x=self.sin_wavelength_x,
            wavelength_y=self.sin_wavelength_y,
        )

    def filenames(self) -> List[str]:
        """Unambiguous, zero-padded image file names (reference image first)."""
        pad = max(4, len(str(self.num_images)))
        return [
            f"{IMAGE_PREFIX}{i + 1:0{pad}d}.png" for i in range(self.num_images)
        ]

    def to_arguments_dict(self) -> Dict[str, Any]:
        """Canonical record of the arguments (goes into ``metadata.json``)."""
        data = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            data[f.name] = str(value) if f.name == "output_dir" else value
        return data


@dataclass
class GenerationSummary:
    """Everything worth reporting about a finished (or dry-run) generation."""

    output_dir: Path
    wrote_files: bool
    num_images: int
    image_size: Sequence[int]
    filenames: List[str]
    num_dots: int
    coverage_target: float
    coverage_achieved: float
    coverage_half_amplitude: Optional[float]
    coverage_iterations: int
    noise_sigma: float
    noise_multiplicative_sigma: float
    seed: int
    valid_fraction_mean: float
    valid_fractions: List[float]
    displacement_max_px: float
    files: List[str]
    elapsed_seconds: float
    deletion_plan: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    metadata_path: Optional[str] = None
    deformation: Dict[str, Any] = field(default_factory=dict)
    movie_filename: Optional[str] = None
    movie_fps: float = DEFAULT_MOVIE_FPS
    movie_written: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


def _log(message: str) -> None:  # pragma: no cover - thin logging wrapper
    LOGGER.info(message)


def generate_dataset(
    config: GenerationConfig,
    *,
    logger: Optional[logging.Logger] = None,
    announce=None,
) -> GenerationSummary:
    """Generate the complete dataset described by ``config``.

    ``announce`` is the callable used for the always-visible user messages
    (most importantly the exact list of items deleted from the output
    directory); it defaults to :func:`print`.  Pass ``lambda message: None`` to
    silence it when using this function as a library.

    Returns a :class:`GenerationSummary`.  Raises :class:`ValueError` for
    invalid parameters and
    :class:`~dicgen.cleanup.ForeignContentError` if the output directory holds
    content that would have to be deleted without ``--overwrite``.
    """
    start = time.perf_counter()
    _log = (logger or LOGGER).info

    config.validate()
    output_dir = Path(config.output_dir)
    _log(f"{TOOL_NAME} {__version__}: preparing {output_dir}")

    plan: CleanupPlan = prepare_output_dir(
        output_dir,
        overwrite=config.overwrite,
        dry_run=config.dry_run,
        announce=announce,
    )
    if config.dry_run:
        _log("--dry-run: no images or ground truth were written")
        return GenerationSummary(
            output_dir=output_dir,
            wrote_files=False,
            num_images=config.num_images,
            image_size=(config.width, config.height),
            filenames=config.filenames(),
            num_dots=0,
            coverage_target=config.dot_coverage,
            coverage_achieved=0.0,
            coverage_half_amplitude=None,
            coverage_iterations=0,
            noise_sigma=config.noise_sigma,
            noise_multiplicative_sigma=config.noise_multiplicative_sigma,
            seed=config.seed,
            valid_fraction_mean=0.0,
            valid_fractions=[],
            displacement_max_px=0.0,
            files=[],
            elapsed_seconds=time.perf_counter() - start,
            deletion_plan=plan.relative_deletions,
            deformation=config.deformation_model().to_dict(),
            movie_fps=config.movie_fps,
        )

    rng = np.random.default_rng(config.seed)
    speckle = config.speckle_params()
    speckle.validate()
    model = config.deformation_model()
    filenames = config.filenames()

    # --- reference speckle (dots) with coverage calibration -----------------
    dots, reference, coverage = calibrate_dot_count(
        speckle, rng, lambda d: render_reference(d, speckle)
    )
    _log(
        f"Speckle: {coverage['num_dots']} dots, radii "
        f"[{speckle.radius_min:g}, {speckle.radius_max:g}] px, coverage "
        f"{coverage['coverage_any']:.4f} (target {speckle.coverage:g}, "
        f"{coverage['iterations']} calibration pass(es))"
    )

    # --- fixed QA scaling over the whole sequence ---------------------------
    qa_vmax = 0.0
    qa_arrow_scale = 0.0
    if config.save_qa_maps:
        coarse = pixel_grid(config.width, config.height)[::4, ::4]
        u_final = model.displacement(coarse, 1.0)
        qa_vmax = float(np.max(np.hypot(u_final[..., 0], u_final[..., 1])))
        qa_vmax = max(qa_vmax, 1e-9)
        qa_arrow_scale = QA_ARROW_LENGTH_PX / qa_vmax if qa_vmax > 0 else 0.0

    masks_dir = output_dir / "masks"
    qa_dir = output_dir / "qa"
    if config.save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)
    if config.save_qa_maps:
        qa_dir.mkdir(parents=True, exist_ok=True)

    files: List[str] = []
    valid_fractions: List[float] = []
    grey_stats: List[Dict[str, float]] = []

    # --- image loop ---------------------------------------------------------
    for i, filename in enumerate(filenames):
        t = load_fraction(i, config.num_images)
        if i == 0:
            warped = reference.copy()
            valid = np.ones((config.height, config.width), dtype=bool)
        else:
            warped, valid = warp_image(
                config.sampling,
                model,
                t,
                dots,
                speckle,
                reference,
                interp_order=config.interpolation_order,
            )
        noisy = apply_noise(
            warped,
            rng,
            sigma_additive=config.noise_sigma,
            sigma_multiplicative=config.noise_multiplicative_sigma,
        )
        save_gray_png(noisy, output_dir / filename)
        files.append(filename)
        valid_fractions.append(float(np.mean(valid)))
        grey_stats.append(
            {
                "image": filename,
                "min": float(noisy.min()),
                "max": float(noisy.max()),
                "mean": float(noisy.mean()),
                "valid_fraction": valid_fractions[-1],
            }
        )

        if config.save_masks:
            mask_name = Path("masks") / f"mask{i + 1:0{max(4, len(str(config.num_images)))}d}.png"
            save_mask_png(valid, output_dir / mask_name)
            files.append(str(mask_name))

        if config.save_qa_maps:
            pad = max(4, len(str(config.num_images)))
            u = model.displacement(pixel_grid(config.width, config.height), t)
            ux, uy = u[..., 0], u[..., 1]
            disp_name = Path("qa") / f"disp_{i + 1:0{pad}d}.png"
            save_rgb_png(qa_displacement_image(ux, uy, qa_vmax, arrow_scale=qa_arrow_scale), output_dir / disp_name)
            files.append(str(disp_name))
            arrows_name = Path("qa") / f"arrows_{i + 1:0{pad}d}.png"
            save_rgb_png(
                reference_with_arrows(reference, ux, uy, arrow_scale=qa_arrow_scale),
                output_dir / arrows_name,
            )
            files.append(str(arrows_name))

        _log(
            f"[{i + 1}/{config.num_images}] {filename}: t = {t:.6f}, "
            f"valid area {100.0 * valid_fractions[-1]:.2f} %, "
            f"grey levels {grey_stats[-1]['min']:.0f}-{grey_stats[-1]['max']:.0f}"
        )

    # --- movie (built from the images just written) -------------------------
    movie_info: Dict[str, Any] = {"written": False, "fps": config.movie_fps}
    movie_filename: Optional[str] = None
    if config.save_movie:
        movie_path = output_dir / MOVIE_FILENAME
        try:
            info = write_movie(
                [output_dir / name for name in filenames],
                movie_path,
                fps=config.movie_fps,
                width=config.width,
                height=config.height,
            )
        except MovieError as exc:
            _log(f"WARNING: movie not written: {exc}")
            movie_info["reason"] = str(exc)
        else:
            movie_filename = MOVIE_FILENAME
            files.append(MOVIE_FILENAME)
            movie_info = {"written": True, **info}
            _log(
                f"Movie {MOVIE_FILENAME}: {info['frames']} frames, "
                f"{info['fps']:g} fps, {info['size_bytes'] / 1e6:.2f} MB"
            )

    # --- ground truth -------------------------------------------------------
    grid = groundtruth.sample_grid(
        config.width, config.height, config.gt_grid_nx, config.gt_grid_ny
    )
    records = groundtruth.image_records(
        model, config.num_images, filenames, valid_fractions, grid=grid
    )
    gt_info: Dict[str, Any] = {"written": bool(config.save_ground_truth)}
    if config.save_ground_truth:
        gt_csv = output_dir / "ground_truth.csv"
        groundtruth.write_ground_truth_csv(records, gt_csv)
        files.append(gt_csv.name)
        grid_csv = output_dir / "ground_truth_grid.csv"
        groundtruth.write_grid_csv(
            groundtruth.grid_records(model, config.num_images, filenames, grid), grid_csv
        )
        files.append(grid_csv.name)
        gt_info.update(
            {
                "per_image_csv": gt_csv.name,
                "grid_csv": grid_csv.name,
                "grid_points": [config.gt_grid_nx, config.gt_grid_ny],
                "columns_image": list(groundtruth.IMAGE_COLUMNS),
                "columns_grid": list(groundtruth.GRID_COLUMNS),
                "strain_convention": (
                    "engineering / small strain: eps_xx = e_x * t, eps_yy = e_y * t, "
                    "gamma_xy = 0 (parameter strains, uniform for the affine model); "
                    "eps_*_img are the components of sym(J) - I in the image frame"
                ),
                "displacement_field_npy": None,
            }
        )

    if config.save_displacement_npy:
        npy_path = output_dir / "displacement_field.npy"
        gt_info["displacement_field_npy"] = groundtruth.save_displacement_field(
            model, config.num_images, npy_path
        )
        files.append(npy_path.name)

    # --- metadata (also records the digests of everything just written) -----
    displacement_max = max(
        (float(r["u_max_px"]) for r in records), default=0.0
    )
    metadata = groundtruth.build_metadata(
        tool=TOOL_NAME,
        version=__version__,
        arguments=config.to_arguments_dict(),
        sequence={
            "num_images": config.num_images,
            "image_size_px": [config.width, config.height],
            "filenames": filenames,
            "filename_pattern": f"{IMAGE_PREFIX}%0{max(4, len(str(config.num_images)))}d.png",
            "reference_image": filenames[0],
            "load_fraction": "t_i = i / (n - 1)",
            "deformed_coordinates": "x = center + F_i (X - center) + t_i * T",
            "ground_truth_displacement": "u(X) = x - X",
            "image_center_px": [float(model.center[0]), float(model.center[1])],
        },
        model=model.to_dict(),
        speckle=speckle.to_dict(),
        dots=dots.to_dict(),
        coverage=coverage,
        noise={
            "additive_sigma_grey_levels": config.noise_sigma,
            "multiplicative_sigma": config.noise_multiplicative_sigma,
            "applied_to": "every image, independently, after warping",
            "clipping": "grey levels clipped to [0, 255] and rounded when writing the 8-bit PNG",
        },
        sampling={
            "method": config.sampling,
            "interpolation_order": (
                config.interpolation_order if config.sampling == "bicubic" else None
            ),
            "interpolation_library": (
                "scipy.ndimage.map_coordinates" if config.sampling == "bicubic" else None
            ),
            "out_of_bounds": "filled with the background grey level and flagged invalid in the mask",
            "valid_mask": "X inside [0, W-1] x [0, H-1]; bicubic mode also shrinks by 2 px of stencil",
        },
        ground_truth=gt_info,
        movie=movie_info,
        files=[],  # filled below, metadata is written last
        notes=[
            "This directory is fully regenerated by dicgen; unrelated files are never touched.",
            "Same seed + same arguments produce byte-identical files.",
            "Displacement/strain values are analytic ground truth, not measured.",
        ],
    )
    metadata["grey_level_statistics"] = grey_stats
    metadata["summary"] = {
        "num_dots": coverage["num_dots"],
        "coverage_achieved": coverage["coverage_any"],
        "coverage_target": speckle.coverage,
        "valid_fraction_mean": float(np.mean(valid_fractions)),
        "displacement_max_px": displacement_max,
    }
    relative_files = sorted(files)
    metadata["files"] = groundtruth.build_file_manifest(output_dir, relative_files)
    metadata_path = output_dir / "metadata.json"
    groundtruth.write_json(metadata_path, metadata)
    files.append(metadata_path.name)

    summary = GenerationSummary(
        output_dir=output_dir,
        wrote_files=True,
        num_images=config.num_images,
        image_size=(config.width, config.height),
        filenames=filenames,
        num_dots=coverage["num_dots"],
        coverage_target=speckle.coverage,
        coverage_achieved=coverage["coverage_any"],
        coverage_half_amplitude=coverage.get("coverage_half_amplitude"),
        coverage_iterations=coverage["iterations"],
        noise_sigma=config.noise_sigma,
        noise_multiplicative_sigma=config.noise_multiplicative_sigma,
        seed=config.seed,
        valid_fraction_mean=float(np.mean(valid_fractions)),
        valid_fractions=valid_fractions,
        displacement_max_px=displacement_max,
        files=sorted(files),
        elapsed_seconds=time.perf_counter() - start,
        deletion_plan=plan.relative_deletions,
        deleted=[str(p.relative_to(output_dir)) for p in plan.deleted],
        metadata_path=str(metadata_path),
        deformation=model.to_dict(),
        movie_filename=movie_filename,
        movie_fps=config.movie_fps,
        movie_written=movie_filename is not None,
    )

    _log(
        f"Done in {summary.elapsed_seconds:.2f} s: {len(summary.files)} files in "
        f"{output_dir}"
    )
    _log(
        f"Coverage {summary.coverage_achieved:.4f} (target {summary.coverage_target:g}), "
        f"{summary.num_dots} dots, mean valid area "
        f"{100.0 * summary.valid_fraction_mean:.2f} %, "
        f"max |u| {summary.displacement_max_px:.3f} px, noise sigma "
        f"{summary.noise_sigma:g} grey levels"
    )
    return summary