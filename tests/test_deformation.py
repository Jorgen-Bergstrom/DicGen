"""Tests for :mod:`dicgen.deformation`: the analytic affine/sinusoid models.

These tests check the *documented* mathematics independently: expected matrices,
displacements and inverse mappings are recomputed from the formulas in the
module docstring using plain numpy, then compared with the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from dicgen.deformation import (
    AffineModel,
    DeformationError,
    DeformationModel,
    SinusoidalModel,
    load_fraction,
    make_model,
    rotation_matrix,
)

# --------------------------------------------------------------------------- #
# Load stepping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [2, 3, 5, 11])
def test_load_fraction_endpoints(n):
    """t_0 = 0 (undeformed reference) and t_{n-1} = 1 (full deformation)."""
    assert load_fraction(0, n) == 0.0
    assert load_fraction(n - 1, n) == 1.0
    # Uniform stepping in between.
    fractions = [load_fraction(i, n) for i in range(n)]
    assert fractions == pytest.approx(np.linspace(0.0, 1.0, n))


def test_load_fraction_rejects_single_image_and_out_of_range():
    with pytest.raises(DeformationError):
        load_fraction(0, 1)
    with pytest.raises(DeformationError):
        load_fraction(3, 3)
    with pytest.raises(DeformationError):
        load_fraction(-1, 3)


# --------------------------------------------------------------------------- #
# Rotation matrix
# --------------------------------------------------------------------------- #


def test_rotation_matrix_90_degrees():
    R = rotation_matrix(90.0)
    assert R == pytest.approx(np.array([[0.0, -1.0], [1.0, 0.0]]), abs=1e-12)
    # Rotation matrices are orthogonal with determinant +1.
    assert R @ R.T == pytest.approx(np.eye(2), abs=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_rotation_matrix_vectorised_shapes():
    angles = np.array([0.0, 90.0, 180.0])
    R = rotation_matrix(angles)
    assert R.shape == (3, 2, 2)
    assert R[1] == pytest.approx(rotation_matrix(90.0), abs=1e-12)


# --------------------------------------------------------------------------- #
# Affine quantities and the deformation convention
# --------------------------------------------------------------------------- #


def _manual_F(model: DeformationModel, t: float) -> np.ndarray:
    """Recompute F_i = R(theta t) diag(1 + e_x t, 1 + e_y t) from the formula."""
    theta = np.deg2rad(model.rotation_deg * t)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R @ np.diag([1.0 + model.strain_x * t, 1.0 + model.strain_y * t])


def test_affine_matrix_matches_documented_formula():
    model = AffineModel(192, 128, strain_x=0.03, strain_y=-0.01, rotation_deg=2.5)
    for t in (0.0, 0.25, 1.0):
        assert model.affine_matrix(t) == pytest.approx(_manual_F(model, t), abs=1e-12)
        assert model.stretch(t) == pytest.approx(
            [1.0 + 0.03 * t, 1.0 - 0.01 * t], abs=1e-12
        )


def test_image_center_is_pixel_grid_center():
    model = AffineModel(129, 77)
    # (width-1)/2, (height-1)/2 -> integer pixels are exactly the centre.
    assert model.center == pytest.approx([64.0, 38.0])


def test_transform_and_displacement_about_centre():
    """x = c + F (X - c) + t T and u(X) = x - X, recomputed by hand."""
    model = AffineModel(
        160, 120, strain_x=0.02, strain_y=0.01, rotation_deg=3.0,
        translation=(1.5, -0.75),
    )
    X = np.array([[10.0, 20.0], [100.0, 90.0], [159.0, 5.0]])
    for t in (0.0, 0.4, 1.0):
        F = _manual_F(model, t)
        offset = t * np.array([1.5, -0.75])
        c = model.center
        expected_x = (X - c) @ F.T + c + offset
        assert model.transform(X, t) == pytest.approx(expected_x, abs=1e-12)
        assert model.displacement(X, t) == pytest.approx(expected_x - X, abs=1e-12)


def test_reference_is_undeformed():
    """Image 0 (t = 0) must have F = I, zero offset and u == 0 everywhere."""
    model = AffineModel(128, 128, strain_x=0.05, strain_y=-0.02, rotation_deg=5.0,
                        translation=(3.0, 4.0))
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 127, size=(50, 2))
    assert model.affine_matrix(0.0) == pytest.approx(np.eye(2), abs=1e-15)
    assert model.affine_offset(0.0) == pytest.approx([0.0, 0.0], abs=1e-15)
    assert model.transform(X, 0.0) == pytest.approx(X, abs=1e-12)
    assert model.displacement(X, 0.0) == pytest.approx(np.zeros_like(X), abs=1e-12)
    assert model.inverse_transform(X, 0.0) == pytest.approx(X, abs=1e-12)


def test_strain_spec_is_uniform_and_uses_parameter_strains():
    model = AffineModel(128, 128, strain_x=0.04, strain_y=-0.01, rotation_deg=4.0)
    X = np.random.default_rng(1).uniform(0, 128, size=(7, 3, 2))
    for t in (0.0, 0.5, 1.0):
        spec = model.strain_spec(X, t)
        assert spec["eps_xx"] == pytest.approx(np.full(X.shape[:-1], 0.04 * t))
        assert spec["eps_yy"] == pytest.approx(np.full(X.shape[:-1], -0.01 * t))
        assert spec["gamma_xy"] == pytest.approx(np.zeros(X.shape[:-1]))


def test_strain_image_frame_is_sym_part_of_F_minus_I():
    model = AffineModel(128, 128, strain_x=0.02, strain_y=0.01, rotation_deg=2.0)
    X = np.array([[3.0, 4.0], [100.0, 80.0]])
    img = model.strain_image_frame(X, 1.0)
    F = _manual_F(model, 1.0)
    assert img["eps_xx_img"] == pytest.approx(np.full(2, F[0, 0] - 1.0))
    assert img["eps_yy_img"] == pytest.approx(np.full(2, F[1, 1] - 1.0))
    assert img["gamma_xy_img"] == pytest.approx(np.full(2, F[0, 1] + F[1, 0]))


def test_jacobian_equals_affine_matrix_for_affine_model():
    model = AffineModel(96, 64, strain_x=0.03, rotation_deg=1.0)
    X = np.random.default_rng(2).uniform(0, 60, size=(5, 2))
    J = model.jacobian(X, 0.7)
    assert J.shape == (5, 2, 2)
    assert np.allclose(J, model.affine_matrix(0.7))


def test_parameters_at_and_to_dict():
    model = AffineModel(100, 80, strain_x=0.02, strain_y=0.03, rotation_deg=1.5,
                        translation=(2.0, -1.0))
    params = model.parameters_at(0.5)
    assert params == pytest.approx(
        {"t": 0.5, "e_x_t": 0.01, "e_y_t": 0.015, "theta_t_deg": 0.75,
         "translation_x_t": 1.0, "translation_y_t": -0.5}
    )
    d = model.to_dict()
    assert d["mode"] == "affine"
    assert d["load_stepping"] == "t_i = i / (n - 1)"
    assert d["total_strain_x"] == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# Forward / backward round trip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(strain_x=0.02, strain_y=0.02, rotation_deg=0.0),
        dict(strain_x=0.05, strain_y=-0.03, rotation_deg=3.0),
        dict(strain_x=0.0, strain_y=0.0, rotation_deg=10.0),
        dict(strain_x=0.0, strain_y=0.0, rotation_deg=0.0, translation=(4.0, -2.5)),
        dict(strain_x=0.01, strain_y=-0.02, rotation_deg=1.5, translation=(2.0, 3.0)),
    ],
)
def test_affine_round_trip(kwargs):
    """F_i^{-1}(F_i(X)) == X for random points, several parameter sets."""
    model = AffineModel(160, 120, **kwargs)
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 120.0, size=(200, 2))
    for t in (0.25, 0.5, 1.0):
        x = model.transform(X, t)
        back = model.inverse_transform(x, t)
        assert back == pytest.approx(X, abs=1e-9, rel=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(amplitude_x=5.0, amplitude_y=-3.0),
        dict(amplitude_x=4.0, amplitude_y=4.0, strain_x=0.01, rotation_deg=1.0),
        dict(amplitude_x=0.0, amplitude_y=6.0, translation=(1.5, -1.0)),
    ],
)
def test_sinusoid_round_trip(kwargs):
    """The vectorised Newton inverse must invert the sinusoid map too."""
    model = SinusoidalModel(160, 128, wavelength_x=100.0, wavelength_y=90.0, **kwargs)
    rng = np.random.default_rng(7)
    X = rng.uniform(5.0, 110.0, size=(150, 2))
    for t in (0.3, 0.6, 1.0):
        x = model.transform(X, t)
        back = model.inverse_transform(x, t)
        assert back == pytest.approx(X, abs=1e-7, rel=0)
        # And the inverse really inverts the forward map.
        assert model.transform(back, t) == pytest.approx(x, abs=1e-6)


def test_sinusoid_matches_documented_formula():
    model = SinusoidalModel(
        200, 160, amplitude_x=6.0, amplitude_y=-4.0,
        wavelength_x=200.0, wavelength_y=160.0, strain_x=0.01, rotation_deg=1.0,
    )
    X = np.array([[20.0, 30.0], [180.0, 140.0]])
    c = model.center
    t = 0.5
    kx = 2 * np.pi / 200.0
    ky = 2 * np.pi / 160.0
    u_sin = np.stack(
        [
            6.0 * np.sin(kx * (X[:, 0] - c[0])),
            -4.0 * np.sin(ky * (X[:, 1] - c[1])),
        ],
        axis=1,
    )
    expected = (X - c) @ _manual_F(model, t).T + c + t * u_sin
    assert model.transform(X, t) == pytest.approx(expected, abs=1e-12)


def test_sinusoid_strain_spec_includes_cosine_term():
    model = SinusoidalModel(200, 160, amplitude_x=6.0, amplitude_y=0.0,
                            wavelength_x=200.0, wavelength_y=160.0)
    X = np.array([[0.0, 50.0], [200.0, 90.0]])
    spec = model.strain_spec(X, 1.0)
    kx = 2 * np.pi / 200.0
    expected = kx * 6.0 * np.cos(kx * (X[:, 0] - model.center[0]))
    assert spec["eps_xx"] == pytest.approx(expected, abs=1e-12)
    assert spec["gamma_xy"] == pytest.approx(np.zeros(2))


def test_sinusoid_to_dict_contains_extra_fields():
    model = SinusoidalModel(100, 100, amplitude_x=3.0, amplitude_y=2.0)
    d = model.to_dict()
    assert d["mode"] == "sinusoid"
    assert d["sin_amplitude_x_px"] == pytest.approx(3.0)
    assert d["sin_wavelength_x_px"] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Factory and validation errors
# --------------------------------------------------------------------------- #


def test_make_model_dispatch_and_unknown_mode():
    assert isinstance(make_model("affine", 64, 64), AffineModel)
    assert isinstance(make_model("SINUSOID", 64, 64), SinusoidalModel)
    with pytest.raises(DeformationError):
        make_model("warp", 64, 64)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(width=4, height=64),  # too small
        dict(strain_x=-1.0),  # collapses the image
        dict(strain_y=-1.5),
        dict(strain_x=float("nan")),
        dict(translation=(float("inf"), 0.0)),
    ],
)
def test_affine_validation_errors(kwargs):
    with pytest.raises(DeformationError):
        AffineModel(**{"width": 64, "height": 64, **kwargs})


def test_sinusoid_amplitude_limit():
    with pytest.raises(DeformationError):
        # amplitude must stay below 0.95 * wavelength / (2 pi)
        SinusoidalModel(64, 64, amplitude_x=100.0, wavelength_x=64.0)
    with pytest.raises(DeformationError):
        SinusoidalModel(64, 64, wavelength_x=-5.0)
    # A legal amplitude is accepted.
    SinusoidalModel(64, 64, amplitude_x=5.0, wavelength_x=64.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(wavelength_x=0.0),  # 0 is a real (invalid) wavelength, not "unset"
        dict(wavelength_y=0.0),
        dict(wavelength_x=float("inf")),
        dict(amplitude_x=float("nan")),
        dict(amplitude_y=float("-inf")),
    ],
)
def test_sinusoid_validation_errors(kwargs):
    with pytest.raises(DeformationError):
        SinusoidalModel(64, 64, **kwargs)


def test_sinusoid_default_wavelengths_are_the_image_size():
    model = SinusoidalModel(64, 48, amplitude_x=2.0)
    assert model.wavelength_x == 64.0
    assert model.wavelength_y == 48.0