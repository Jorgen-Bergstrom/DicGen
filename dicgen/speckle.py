"""Reference speckle pattern generation for the synthetic DIC dataset generator.

The reference image (image 0) is a random field of circular dots:

* dot centres are sampled uniformly over the image (optionally enforcing a
  minimum distance between centres, which makes the pattern look less clumped);
* each dot has a radius sampled uniformly in ``[radius_min, radius_max]``;
* each dot has a peak amplitude (darkening relative to the background) sampled
  uniformly in ``[amplitude_min, amplitude_max]`` so that the speckle is not
  perfectly bimodal;
* every dot is rendered as a **smooth analytic intensity profile** rather than a
  hard-edged disc, so that warped images stay anti-aliased and sub-pixel
  accurate.  Two profiles are provided:

  ``taper`` (default)
      A "plateau + raised-cosine edge" disc::

          f(d) = 1                                        d <= r - s
          f(d) = 0.5 * (1 + cos(pi * (d - (r - s)) / (2 s)))   r - s < d < r + s
          f(d) = 0                                        d >= r + s

      with ``s = --dot-softness`` (in pixels, default 0.7).  ``f(r) = 0.5``
      exactly, so the area where the dot darkens the image by more than half of
      its amplitude equals the geometric disc area ``pi r^2``.

  ``gaussian``
      ``f(d) = exp(-0.5 * (d / sigma)^2)`` with ``sigma = r / 2``.

The number of dots is chosen to aim for the requested ``--dot-coverage``.  For
randomly placed (freely overlapping) discs of equal radius the union area
saturates as ``1 - exp(-n pi r^2 / A)``, so::

    n = -A * ln(1 - coverage) / (pi * r_mean^2)

A short calibration loop then re-renders the reference and corrects ``n`` so
that the *measured* coverage hits the target (see :func:`calibrate_dot_count`).
All random draws come from the single ``numpy.random.Generator`` passed in, so
the whole procedure is exactly reproducible for a given seed.

The dots are the *only* source of image texture: deformed images reuse the very
same dot set (advected by the deformation), they are never re-randomised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "SpeckleParams",
    "DotSet",
    "coverage_to_dot_count",
    "rms_radius",
    "generate_dots",
    "profile_values",
    "rasterise_dots",
    "measure_coverage",
    "calibrate_dot_count",
]

# Contribution (in grey levels) above which a pixel counts as "covered".
COVERAGE_THRESHOLD = 0.5


@dataclass
class SpeckleParams:
    """Parameters of the reference speckle pattern."""

    width: int
    height: int
    radius_min: float = 3.0
    radius_max: float = 6.0
    coverage: float = 0.5
    background: float = 128.0
    amplitude_min: float = 70.0
    amplitude_max: float = 120.0
    profile: str = "taper"
    softness: float = 0.7
    min_distance: float = 0.0
    coverage_tolerance: float = 0.01
    max_calibration_iterations: int = 4

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        self.radius_min = float(self.radius_min)
        self.radius_max = float(self.radius_max)
        self.coverage = float(self.coverage)
        self.background = float(self.background)
        self.amplitude_min = float(self.amplitude_min)
        self.amplitude_max = float(self.amplitude_max)
        self.profile = str(self.profile).lower()
        self.softness = float(self.softness)
        self.min_distance = float(self.min_distance)
        self.coverage_tolerance = float(self.coverage_tolerance)
        self.max_calibration_iterations = int(self.max_calibration_iterations)

    # ------------------------------------------------------------- validation
    def validate(self) -> None:
        """Raise :class:`ValueError` with an actionable message if invalid."""
        if self.width < 8 or self.height < 8:
            raise ValueError(
                f"image size must be at least 8x8 pixels, got {self.width}x{self.height}"
            )
        if not (0.0 < self.radius_min <= self.radius_max):
            raise ValueError(
                "dot radii must satisfy 0 < --dot-radius-min <= --dot-radius-max, "
                f"got {self.radius_min} and {self.radius_max}"
            )
        if self.radius_max > 0.25 * min(self.width, self.height):
            raise ValueError(
                f"--dot-radius-max = {self.radius_max} is too large for a "
                f"{self.width}x{self.height} image (keep it <= 25% of the shorter side)"
            )
        if not (0.0 < self.coverage < 0.95):
            raise ValueError(
                f"--dot-coverage must lie in (0, 0.95), got {self.coverage}"
            )
        if not (0.0 <= self.amplitude_min <= self.amplitude_max):
            raise ValueError(
                "dot amplitudes must satisfy 0 <= --dot-amplitude-min <= "
                f"--dot-amplitude-max, got {self.amplitude_min} and {self.amplitude_max}"
            )
        if not (0.0 <= self.background <= 255.0):
            raise ValueError(
                f"--background must lie in [0, 255], got {self.background}"
            )
        if self.background - self.amplitude_max < 0.0:
            raise ValueError(
                "--background - --dot-amplitude-max must stay >= 0 so that dot "
                "centres cannot go negative; got background = "
                f"{self.background}, amplitude_max = {self.amplitude_max}"
            )
        if self.profile not in ("taper", "gaussian"):
            raise ValueError(
                f"--dot-profile must be 'taper' or 'gaussian', got {self.profile!r}"
            )
        if self.softness < 0.05:
            raise ValueError(
                f"--dot-softness must be at least 0.05 px, got {self.softness}"
            )
        if self.min_distance < 0.0:
            raise ValueError(
                f"--min-dot-distance must be >= 0, got {self.min_distance}"
            )
        if not (0.0 <= self.coverage_tolerance < 0.5):
            raise ValueError(
                "--coverage-tolerance must lie in [0, 0.5), got "
                f"{self.coverage_tolerance}"
            )

    # ------------------------------------------------------------- reporting
    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "dot_radius_min_px": self.radius_min,
            "dot_radius_max_px": self.radius_max,
            "dot_coverage_target": self.coverage,
            "dot_coverage_tolerance": self.coverage_tolerance,
            "dot_profile": self.profile,
            "dot_softness_px": self.softness,
            "min_dot_distance_px": self.min_distance,
            "background_grey": self.background,
            "dot_amplitude_min": self.amplitude_min,
            "dot_amplitude_max": self.amplitude_max,
        }


@dataclass
class DotSet:
    """The reference speckle dot population."""

    centres: np.ndarray  # (N, 2) float, (x, y) in reference pixels
    radii: np.ndarray  # (N,) float, px
    amplitudes: np.ndarray  # (N,) float, grey levels of darkening

    def __post_init__(self) -> None:
        self.centres = np.asarray(self.centres, dtype=float).reshape(-1, 2)
        self.radii = np.asarray(self.radii, dtype=float).reshape(-1)
        self.amplitudes = np.asarray(self.amplitudes, dtype=float).reshape(-1)
        n = len(self.radii)
        if len(self.centres) != n or len(self.amplitudes) != n:
            raise ValueError("DotSet arrays must have the same length")

    def __len__(self) -> int:
        return int(self.radii.size)

    @property
    def mean_radius(self) -> float:
        return float(np.mean(self.radii)) if len(self) else 0.0

    @property
    def total_disc_area(self) -> float:
        """Sum of the individual disc areas in px^2 (overlaps counted twice)."""
        return float(np.sum(math.pi * self.radii**2))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_dots": len(self),
            "radius_mean_px": float(np.mean(self.radii)) if len(self) else 0.0,
            "radius_std_px": float(np.std(self.radii)) if len(self) else 0.0,
            "radius_min_px": float(np.min(self.radii)) if len(self) else 0.0,
            "radius_max_px": float(np.max(self.radii)) if len(self) else 0.0,
            "amplitude_mean": float(np.mean(self.amplitudes)) if len(self) else 0.0,
        }


# --------------------------------------------------------------------- helpers
def coverage_to_dot_count(
    coverage: float, width: int, height: int, radius_rms: float
) -> int:
    """Number of dots that realises ``coverage`` for freely overlapping discs.

    Uses the saturation model ``coverage = 1 - exp(-n pi r^2 / A)`` for randomly
    placed discs -- ``n = -A ln(1 - coverage) / (pi r_rms^2)`` -- where
    ``r_rms`` is the RMS (not arithmetic mean) radius, because the union area is
    proportional to ``E[r^2]``.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must lie in (0, 1), got {coverage}")
    if radius_rms <= 0.0:
        raise ValueError("radius_rms must be positive")
    area = float(width) * float(height)
    return max(1, int(round(-area * math.log1p(-coverage) / (math.pi * radius_rms**2))))


def rms_radius(radius_min: float, radius_max: float) -> float:
    """RMS of a radius drawn uniformly from ``[radius_min, radius_max]``."""
    a, b = float(radius_min), float(radius_max)
    return math.sqrt(max((a * a + a * b + b * b) / 3.0, 0.0))


def _sample_min_distance(
    rng: np.random.Generator,
    count: int,
    width: int,
    height: int,
    min_distance: float,
) -> np.ndarray:
    """Uniform dot centres with a minimum-distance constraint (dart throwing).

    Uses a uniform cell grid of side ``min_distance`` so that only the 3x3
    neighbouring cells have to be tested for each candidate.  Raises
    :class:`ValueError` if the packing density is unreachable.
    """
    if count <= 0:
        return np.zeros((0, 2))
    cell = float(min_distance)
    nx = max(1, int(math.ceil(width / cell)))
    ny = max(1, int(math.ceil(height / cell)))
    grid_n = nx * ny
    # Estimated maximum number of centres (hexagonal packing), used only to fail
    # fast with a clear message instead of looping forever.
    max_packable = 0.9069 * (width * height) / (math.pi * (min_distance / 2.0) ** 2)
    if count > max_packable:
        raise ValueError(
            f"cannot place {count} dot centres with a minimum distance of "
            f"{min_distance} px in a {width}x{height} image (at most about "
            f"{int(max_packable)} are geometrically possible); lower "
            "--min-dot-distance or --dot-coverage"
        )

    cell_of: Dict[int, list] = {}
    accepted = np.empty((count, 2), dtype=float)
    n_acc = 0
    batch = max(1024, 4 * count)
    exhausted = 0
    while n_acc < count and exhausted < 200:
        candidates = np.column_stack(
            (rng.uniform(0.0, width - 1.0, batch), rng.uniform(0.0, height - 1.0, batch))
        )
        for p in candidates:
            if n_acc >= count:
                break
            ci = min(int(p[0] / cell), nx - 1)
            cj = min(int(p[1] / cell), ny - 1)
            ok = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    key = (ci + di) * ny + (cj + dj)
                    neighbours = cell_of.get(key)
                    if not neighbours:
                        continue
                    arr = accepted[np.asarray(neighbours)]
                    if np.any(np.hypot(arr[:, 0] - p[0], arr[:, 1] - p[1]) < min_distance):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                accepted[n_acc] = p
                key = ci * ny + cj
                cell_of.setdefault(key, []).append(n_acc)
                n_acc += 1
        exhausted += 1
        # Grow the batch if acceptance is slow, but never blow up memory.
        batch = min(int(batch * 1.5) + 1, 4_000_000)
    if n_acc < count:
        raise ValueError(
            f"could only place {n_acc} of the requested {count} dot centres with a "
            f"minimum distance of {min_distance} px; lower --min-dot-distance or "
            "--dot-coverage"
        )
    return accepted


def generate_dots(
    params: SpeckleParams, rng: np.random.Generator, num_dots: Optional[int] = None
) -> DotSet:
    """Sample the random reference dot population.

    Parameters
    ----------
    params:
        Speckle parameters (validated here).
    rng:
        The single numpy random generator of the run (guarantees reproducibility).
    num_dots:
        Override for the number of dots (used by the coverage calibration loop).
    """
    params.validate()
    if num_dots is None:
        num_dots = coverage_to_dot_count(
            params.coverage,
            params.width,
            params.height,
            rms_radius(params.radius_min, params.radius_max),
        )
    num_dots = int(num_dots)
    if num_dots <= 0:
        raise ValueError(f"num_dots must be positive, got {num_dots}")

    if params.min_distance > 0.0:
        centres = _sample_min_distance(
            rng, num_dots, params.width, params.height, params.min_distance
        )
    else:
        centres = np.column_stack(
            (
                rng.uniform(0.0, params.width - 1.0, num_dots),
                rng.uniform(0.0, params.height - 1.0, num_dots),
            )
        )
    radii = rng.uniform(params.radius_min, params.radius_max, num_dots)
    amplitudes = rng.uniform(params.amplitude_min, params.amplitude_max, num_dots)
    return DotSet(centres=centres, radii=radii, amplitudes=amplitudes)


def profile_values(
    distance: np.ndarray, radius: np.ndarray | float, softness: float, profile: str
) -> np.ndarray:
    """Analytic dot intensity profile ``f(d)``, in ``[0, 1]``.

    ``f`` is the fraction of the dot amplitude that darkens the pixel; it is 1
    at the dot centre and 0 outside the support of radius ``r + s`` (``taper``)
    or asymptotically 0 (``gaussian``).
    """
    d = np.asarray(distance, dtype=float)
    if profile == "gaussian":
        sigma = np.maximum(np.asarray(radius, dtype=float) / 2.0, 1e-6)
        return np.exp(-0.5 * (d / sigma) ** 2)
    # smooth, C1 tapered disc
    inner = np.maximum(np.asarray(radius, dtype=float) - softness, 0.0)
    outer = np.asarray(radius, dtype=float) + softness
    f = np.zeros_like(d)
    plateau = d <= inner
    edge = (d > inner) & (d < outer)
    f[plateau] = 1.0
    if np.any(edge):
        f[edge] = 0.5 * (
            1.0 + np.cos(math.pi * (d[edge] - inner) / (2.0 * softness))
        )
    return f


def support_radius(radius: float, softness: float, profile: str) -> float:
    """Radius beyond which the profile is (numerically) zero."""
    if profile == "gaussian":
        return radius * 4.0
    return radius + softness


def rasterise_dots(
    coords: np.ndarray,
    centres_in_grid: np.ndarray,
    dots: DotSet,
    background: float,
    *,
    profile: str = "taper",
    softness: float = 0.7,
    radius_scale: float = 1.0,
    image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluate the continuous speckle field on a grid of sample points.

    Parameters
    ----------
    coords:
        ``(H, W, 2)`` array of **reference-frame** coordinates ``(x, y)`` at
        which the speckle field is evaluated.  For the reference image this is
        simply the pixel grid; for a deformed image it is the *inverse-mapped*
        pixel grid (``X = F^{-1}(x)``), which is exactly how an image is warped
        while keeping the ground truth exact.
    centres_in_grid:
        ``(N, 2)`` positions of the dot centres *expressed in the sampling grid
        frame* (for the reference image: the dot centres themselves; for a
        deformed image: the forward-mapped dot centres).  Used only to restrict
        the per-dot evaluation to a small bounding box: this is a pure
        performance optimisation and does not change the result, because the
        profile vanishes outside ``radius + softness``.
    dots:
        The dot population being rendered.
    background:
        Background grey level.
    profile, softness:
        Dot intensity profile and its edge softness.
    radius_scale:
        Safety factor applied to the dot support radius when computing the
        bounding box (equals the spectral norm of the deformation gradient for
        deformed images, so that a stretched dot is never clipped).
    image:
        Optional pre-allocated scratch array of shape ``(H, W)`` (modified in
        place).  It only avoids an allocation; its contents are overwritten.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` float array with the *unclipped* grey levels,
        ``background - max_k(amplitude_k * profile_k)`` (overlapping dots combine
        with a union/max rule, see below).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape (H, W, 2), got {coords.shape}")
    height, width = coords.shape[0], coords.shape[1]
    centres_in_grid = np.asarray(centres_in_grid, dtype=float).reshape(-1, 2)
    if image is None:
        darkening = np.zeros((height, width), dtype=float)
    else:
        darkening = image
        if darkening.shape != (height, width):
            raise ValueError(
                f"image has shape {darkening.shape}, expected {(height, width)}"
            )
        darkening[...] = 0.0

    for k in range(len(dots)):
        r = float(dots.radii[k])
        amp = float(dots.amplitudes[k])
        support = support_radius(r, softness, profile) * max(radius_scale, 1.0) + 1.0
        cx, cy = float(centres_in_grid[k, 0]), float(centres_in_grid[k, 1])
        x0 = int(math.floor(cx - support))
        x1 = int(math.ceil(cx + support)) + 1
        y0 = int(math.floor(cy - support))
        y1 = int(math.ceil(cy + support)) + 1
        if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
            continue
        x0, x1 = max(x0, 0), min(x1, width)
        y0, y1 = max(y0, 0), min(y1, height)
        sub = coords[y0:y1, x0:x1]
        dx = sub[..., 0] - float(dots.centres[k, 0])
        dy = sub[..., 1] - float(dots.centres[k, 1])
        dist = np.sqrt(dx * dx + dy * dy)
        limit = support_radius(r, softness, profile)
        if profile != "gaussian":
            limit *= max(radius_scale, 1.0)
        mask = dist <= limit
        if not mask.any():
            continue
        values = np.zeros_like(dist)
        values[mask] = profile_values(dist[mask], r, softness, profile)
        # Overlapping dots combine with a MAX (union) rule: a pixel is as dark as
        # the darkest dot covering it, which is both physically sensible for an
        # opaque speckle pattern and exactly consistent with the union-area
        # coverage model used to choose the number of dots.
        region = darkening[y0:y1, x0:x1]
        np.maximum(region, amp * values, out=region)
    return np.asarray(background, dtype=float) - darkening


def measure_coverage(
    reference: np.ndarray, background: float, amplitude_mean: float | None = None
) -> Dict[str, float]:
    """Measure how much of the reference image is covered by speckle.

    Returns a dict with

    ``coverage_any``
        fraction of pixels darkened by more than 0.5 grey levels (this is the
        "geometric" coverage of the dots and is the value compared with
        ``--dot-coverage``);
    ``coverage_half_amplitude``
        fraction of pixels darkened by more than half of the mean dot amplitude
        (only when ``amplitude_mean`` is given) -- a stricter, "solid dot" area.
    """
    ink = float(background) - np.asarray(reference, dtype=float)
    out = {"coverage_any": float(np.mean(ink > COVERAGE_THRESHOLD))}
    if amplitude_mean is not None and amplitude_mean > 0:
        out["coverage_half_amplitude"] = float(
            np.mean(ink > 0.5 * float(amplitude_mean))
        )
    return out


def calibrate_dot_count(
    params: SpeckleParams,
    rng: np.random.Generator,
    render,
    *,
    num_dots: Optional[int] = None,
) -> Tuple[DotSet, np.ndarray, Dict[str, Any]]:
    """Generate dots and calibrate their number so the measured coverage matches.

    Parameters
    ----------
    params:
        Speckle parameters.
    rng:
        The run's random generator.  Note that the calibration re-draws the dot
        population on every iteration, so the number of random draws depends on
        the number of iterations -- but the whole procedure is deterministic for
        a fixed seed and a fixed set of command-line arguments.
    render:
        Callable ``render(dots) -> reference image (H, W) float array``.
    num_dots:
        Optional starting guess.  Defaults to the analytic estimate.

    Returns
    -------
    (dots, reference, info)
        The final dot set, the rendered reference image, and a dictionary with
        ``num_dots``, ``iterations``, ``coverage_any``,
        ``coverage_half_amplitude``, ``coverage_target``, ``coverage_history``.
    """
    params.validate()
    if num_dots is None:
        num_dots = coverage_to_dot_count(
            params.coverage,
            params.width,
            params.height,
            rms_radius(params.radius_min, params.radius_max),
        )
    count = int(num_dots)
    history = []
    best = None
    for iteration in range(max(1, params.max_calibration_iterations)):
        dots = generate_dots(params, rng, count)
        reference = render(dots)
        stats = measure_coverage(reference, params.background, dots.amplitudes.mean() if len(dots) else None)
        achieved = stats["coverage_any"]
        history.append({"iteration": iteration, "num_dots": count, "coverage_any": achieved})
        error = achieved - params.coverage
        if best is None or abs(error) < abs(best[0]):
            best = (error, dots, reference, stats, count)
        if abs(error) <= params.coverage_tolerance:
            break
        if achieved >= 0.95 or achieved <= 1e-6:
            break  # nothing sensible to solve for any more
        # The union-area model gives coverage = 1 - exp(-k n); solve for the
        # next n with the measured k = -ln(1 - achieved) / n.
        ratio = math.log1p(-params.coverage) / math.log1p(-min(achieved, 0.949999))
        candidate = int(round(count * ratio))
        if candidate == count:
            candidate = count + max(1, count // 20)
        count = max(1, min(candidate, 40_000_000))
    error, dots, reference, stats, count = best
    info = {
        "num_dots": count,
        "iterations": len(history),
        "coverage_any": stats["coverage_any"],
        "coverage_half_amplitude": stats.get("coverage_half_amplitude"),
        "coverage_target": params.coverage,
        "coverage_error": stats["coverage_any"] - params.coverage,
        "coverage_history": history,
    }
    return dots, reference, info