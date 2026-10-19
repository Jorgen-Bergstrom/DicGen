"""Tests for :mod:`dicgen.speckle`: dot sampling, analytic profiles, coverage."""

from __future__ import annotations

import numpy as np
import pytest

from dicgen.speckle import (
    COVERAGE_THRESHOLD,
    DotSet,
    SpeckleParams,
    calibrate_dot_count,
    coverage_to_dot_count,
    generate_dots,
    measure_coverage,
    profile_values,
    rasterise_dots,
    rms_radius,
    support_radius,
)


def small_params(**overrides) -> SpeckleParams:
    params = dict(width=128, height=128, radius_min=3.0, radius_max=6.0, coverage=0.5)
    params.update(overrides)
    return SpeckleParams(**params)


# --------------------------------------------------------------------------- #
# Dot-count model and helpers
# --------------------------------------------------------------------------- #


def test_rms_radius_matches_uniform_rms():
    # For U[a, b]: E[r^2] = (a^2 + ab + b^2)/3.
    a, b = 2.0, 5.0
    assert rms_radius(a, b) == pytest.approx(np.sqrt((a * a + a * b + b * b) / 3.0))


def test_coverage_to_dot_count_monotone_and_errors():
    n_low = coverage_to_dot_count(0.2, 256, 256, 4.0)
    n_high = coverage_to_dot_count(0.7, 256, 256, 4.0)
    assert 0 < n_low < n_high
    with pytest.raises(ValueError):
        coverage_to_dot_count(0.0, 128, 128, 4.0)
    with pytest.raises(ValueError):
        coverage_to_dot_count(1.0, 128, 128, 4.0)
    with pytest.raises(ValueError):
        coverage_to_dot_count(0.5, 128, 128, 0.0)


@pytest.mark.parametrize("profile", ["taper", "gaussian"])
def test_profile_is_unit_at_centre_and_decreases(profile):
    d = np.linspace(0.0, 20.0, 400)
    values = profile_values(d, 4.0, 0.7, profile)
    assert values[0] == pytest.approx(1.0)
    assert np.all(np.diff(values) <= 1e-12)
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_taper_profile_half_value_at_radius_and_compact_support():
    radius, softness = 4.0, 0.7
    assert profile_values(np.array([radius]), radius, softness, "taper")[0] == pytest.approx(0.5)
    # Zero at/after the support radius r + s, unit on the plateau d <= r - s.
    beyond = profile_values(np.array([radius + softness + 1e-6]), radius, softness, "taper")
    assert beyond[0] == pytest.approx(0.0)
    assert profile_values(np.array([radius - softness]), radius, softness, "taper")[0] == pytest.approx(1.0)


def test_support_radius():
    assert support_radius(4.0, 0.7, "taper") == pytest.approx(4.7)
    assert support_radius(4.0, 0.7, "gaussian") == pytest.approx(16.0)


# --------------------------------------------------------------------------- #
# Dot sampling and determinism
# --------------------------------------------------------------------------- #


def test_generate_dots_shapes_and_ranges():
    params = small_params()
    dots = generate_dots(params, np.random.default_rng(0), num_dots=40)
    assert isinstance(dots, DotSet)
    assert len(dots) == 40
    assert dots.centres.shape == (40, 2)
    assert dots.radii.shape == (40,)
    assert dots.amplitudes.shape == (40,)
    assert np.all((dots.radii >= params.radius_min) & (dots.radii <= params.radius_max))
    assert np.all((dots.amplitudes >= params.amplitude_min) & (dots.amplitudes <= params.amplitude_max))
    assert np.all((dots.centres[:, 0] >= 0) & (dots.centres[:, 0] <= params.width - 1))
    assert np.all((dots.centres[:, 1] >= 0) & (dots.centres[:, 1] <= params.height - 1))


def test_generate_dots_is_deterministic_and_seed_dependent():
    params = small_params()
    a = generate_dots(params, np.random.default_rng(0), num_dots=50)
    b = generate_dots(params, np.random.default_rng(0), num_dots=50)
    c = generate_dots(params, np.random.default_rng(1), num_dots=50)
    assert np.array_equal(a.centres, b.centres)
    assert np.array_equal(a.radii, b.radii)
    assert not np.array_equal(a.centres, c.centres)


def test_generate_dots_min_distance_constraint():
    params = small_params(min_distance=8.0, coverage=0.2)
    dots = generate_dots(params, np.random.default_rng(0), num_dots=20)
    centres = dots.centres
    # No two centres closer than the requested minimum distance.
    for i in range(len(centres)):
        d = np.hypot(centres[:, 0] - centres[i, 0], centres[:, 1] - centres[i, 1])
        d[i] = np.inf
        assert d.min() >= 8.0 - 1e-9


def test_dotset_properties_and_to_dict():
    dots = DotSet(
        centres=np.array([[1.0, 2.0], [3.0, 4.0]]),
        radii=np.array([2.0, 4.0]),
        amplitudes=np.array([10.0, 20.0]),
    )
    assert len(dots) == 2
    assert dots.mean_radius == pytest.approx(3.0)
    assert dots.total_disc_area == pytest.approx(np.pi * (4.0 + 16.0))
    d = dots.to_dict()
    assert d["num_dots"] == 2
    assert d["radius_min_px"] == pytest.approx(2.0)
    assert d["radius_max_px"] == pytest.approx(4.0)


def test_dotset_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        DotSet(centres=np.zeros((2, 2)), radii=np.zeros(3), amplitudes=np.zeros(2))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides",
    [
        dict(width=4),
        dict(radius_min=0.0),
        dict(radius_min=5.0, radius_max=3.0),
        dict(radius_max=100.0),  # too large for a 128 px image
        dict(coverage=0.0),
        dict(coverage=1.0),
        dict(amplitude_min=-1.0),
        dict(amplitude_min=200.0, amplitude_max=10.0),
        dict(background=300.0),
        dict(background=10.0, amplitude_max=120.0),  # background - amplitude < 0
        dict(profile="box"),
        dict(softness=0.0),
        dict(min_distance=-1.0),
        dict(coverage_tolerance=0.9),
    ],
)
def test_speckle_params_validation_errors(overrides):
    with pytest.raises(ValueError):
        small_params(**overrides).validate()


def test_speckle_params_validate_accepts_defaults():
    small_params().validate()


# --------------------------------------------------------------------------- #
# Rasterisation and coverage measurement
# --------------------------------------------------------------------------- #


def test_rasterise_dots_uses_max_union_rule():
    """A pixel covered by two dots is darkened by the stronger one only."""
    params = small_params(profile="taper", softness=0.7)
    # Two coincident dots with different amplitudes -> the darker one wins.
    dots = DotSet(
        centres=np.array([[0.0, 0.0], [0.0, 0.0]]),
        radii=np.array([4.0, 4.0]),
        amplitudes=np.array([30.0, 90.0]),
    )
    coords = np.zeros((1, 1, 2))
    coords[0, 0] = [0.0, 0.0]
    image = rasterise_dots(coords, dots.centres, dots, background=128.0,
                           profile=params.profile, softness=params.softness)
    assert image[0, 0] == pytest.approx(128.0 - 90.0)


def test_rasterise_dots_shape_checks():
    dots = generate_dots(small_params(), np.random.default_rng(0), num_dots=5)
    with pytest.raises(ValueError):
        rasterise_dots(np.zeros((4, 4)), dots.centres, dots, 128.0)
    coords = np.zeros((10, 10, 2))
    with pytest.raises(ValueError):
        rasterise_dots(coords, dots.centres, dots, 128.0, image=np.zeros((3, 3)))


def test_rasterise_dots_background_where_no_dot():
    dots = DotSet(centres=np.array([[10.0, 10.0]]), radii=np.array([3.0]),
                  amplitudes=np.array([80.0]))
    coords = np.zeros((32, 32, 2))
    coords[..., 0] = np.arange(32)[None, :]
    coords[..., 1] = np.arange(32)[:, None]
    image = rasterise_dots(coords, dots.centres, dots, background=200.0,
                           profile="taper", softness=0.7)
    assert image[30, 30] == pytest.approx(200.0)  # far from the only dot
    assert image[10, 10] == pytest.approx(120.0)  # dot centre


def test_measure_coverage_counts_darkened_pixels():
    reference = np.full((10, 10), 128.0)
    reference[0, :6] = 120.0  # darkened by 8 grey levels
    stats = measure_coverage(reference, background=128.0, amplitude_mean=40.0)
    assert stats["coverage_any"] == pytest.approx(0.06)
    # 8 > 0.5*40 = 20 is false, so the half-amplitude coverage is zero.
    assert stats["coverage_half_amplitude"] == pytest.approx(0.0)
    assert COVERAGE_THRESHOLD == 0.5


def test_calibrate_dot_count_is_deterministic_and_hits_target():
    params = small_params(coverage=0.45, coverage_tolerance=0.01)
    render = lambda dots: rasterise_dots(  # noqa: E731
        _grid(params), dots.centres, dots, params.background,
        profile=params.profile, softness=params.softness,
    )
    dots_a, ref_a, info_a = calibrate_dot_count(params, np.random.default_rng(0), render)
    dots_b, ref_b, info_b = calibrate_dot_count(params, np.random.default_rng(0), render)
    assert np.array_equal(dots_a.centres, dots_b.centres)
    assert np.array_equal(ref_a, ref_b)
    assert info_a["num_dots"] == info_b["num_dots"]
    assert abs(info_a["coverage_any"] - params.coverage) <= 0.03
    assert info_a["iterations"] >= 1


def _grid(params: SpeckleParams) -> np.ndarray:
    xs = np.arange(params.width, dtype=float)
    ys = np.arange(params.height, dtype=float)
    grid = np.empty((params.height, params.width, 2))
    grid[..., 0] = xs[None, :]
    grid[..., 1] = ys[:, None]
    return grid