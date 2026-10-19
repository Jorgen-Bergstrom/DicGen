"""Tests for :mod:`dicgen.imagegen`: warping, noise, masks and PNG output."""

from __future__ import annotations

import numpy as np
import pytest

from dicgen.deformation import AffineModel
from dicgen.imagegen import (
    apply_noise,
    pixel_grid,
    render_reference,
    save_gray_png,
    save_mask_png,
    save_rgb_png,
    spectral_norm_2x2,
    to_uint8,
    valid_fraction,
    valid_mask,
    warp_analytic,
    warp_bicubic,
    warp_image,
)
from dicgen.speckle import SpeckleParams, calibrate_dot_count


@pytest.fixture(scope="module")
def speckle():
    """A modest, deterministic speckle population shared by these tests."""
    params = SpeckleParams(width=160, height=128, radius_min=3.0, radius_max=6.0,
                           coverage=0.4)
    rng = np.random.default_rng(0)
    dots, reference, info = calibrate_dot_count(
        params, rng, lambda d: render_reference(d, params)
    )
    return params, dots, reference, info


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def test_pixel_grid_layout_and_values():
    grid = pixel_grid(4, 3)
    assert grid.shape == (3, 4, 2)
    # x increases along columns, y along rows.
    assert grid[0, :, 0] == pytest.approx([0, 1, 2, 3])
    assert grid[:, 0, 1] == pytest.approx([0, 1, 2])
    assert grid[2, 3, 0] == pytest.approx(3.0)
    assert grid[2, 3, 1] == pytest.approx(2.0)


def test_valid_mask_boundaries():
    coords = np.array([
        [0.0, 0.0], [9.0, 5.0], [9.0001, 5.0], [-1e-9, 5.0], [5.0, 9.0], [5.0, -0.5],
    ])
    mask = valid_mask(coords, width=10, height=10)
    assert mask.tolist() == [True, True, False, False, True, False]


def test_valid_fraction():
    assert valid_fraction(np.ones((4, 4), dtype=bool)) == pytest.approx(1.0)
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True
    assert valid_fraction(mask) == pytest.approx(0.25)


def test_spectral_norm_of_known_matrices():
    identity = np.eye(2)
    assert spectral_norm_2x2(identity) == pytest.approx(1.0)
    diagonal = np.diag([2.0, 0.5])
    assert spectral_norm_2x2(diagonal) == pytest.approx(2.0)
    stacked = np.stack([identity, diagonal])
    assert spectral_norm_2x2(stacked) == pytest.approx([1.0, 2.0])


# --------------------------------------------------------------------------- #
# uint8 conversion and noise
# --------------------------------------------------------------------------- #


def test_to_uint8_rounds_and_clips():
    values = np.array([-4.6, 0.4, 127.5, 254.6, 400.0])
    assert to_uint8(values).tolist() == [0, 0, 128, 255, 255]
    assert to_uint8(values).dtype == np.uint8


def test_apply_noise_zero_is_identity():
    image = np.arange(12.0).reshape(3, 4)
    out = apply_noise(image, np.random.default_rng(0), sigma_additive=0.0)
    assert np.array_equal(out, image)


def test_apply_noise_standard_deviation_matches_sigma():
    image = np.full((80, 80), 128.0)
    noisy = apply_noise(image, np.random.default_rng(3), sigma_additive=10.0)
    # 80x80 = 6400 samples -> the sample std of N(0, 10) is within a few percent.
    assert np.std(noisy - image) == pytest.approx(10.0, rel=0.05)


def test_apply_noise_multiplicative():
    image = np.full((60, 60), 100.0)
    noisy = apply_noise(image, np.random.default_rng(4), sigma_multiplicative=0.1)
    assert np.std(noisy / image - 1.0) == pytest.approx(0.1, rel=0.1)


# --------------------------------------------------------------------------- #
# Analytic warping
# --------------------------------------------------------------------------- #


def test_identity_warp_equals_render_reference(speckle):
    """At t = 0 the analytic warp is exactly the rendered reference."""
    params, dots, reference, _ = speckle
    model = AffineModel(params.width, params.height, strain_x=0.05, rotation_deg=2.0)
    image, valid = warp_analytic(model, 0.0, dots, params)
    assert np.array_equal(image, reference)
    assert valid.all()


def test_warp_analytic_uses_background_outside_domain(speckle):
    params, dots, reference, _ = speckle
    # A large translation pushes content out of the reference domain.
    model = AffineModel(params.width, params.height, translation=(90.0, 0.0))
    image, valid = warp_analytic(model, 1.0, dots, params)
    assert not valid.all()
    assert np.allclose(image[~valid], params.background)
    assert np.all((image >= 0.0) & (image <= 255.0))


def test_warp_bicubic_shrinks_valid_region_by_stencil_margin(speckle):
    params, dots, reference, _ = speckle
    model = AffineModel(params.width, params.height)
    image, valid = warp_bicubic(reference, model, 0.0, params, margin=2)
    # Identity warp: the outermost 2-pixel border is excluded by the margin.
    expected = np.ones((params.height, params.width), dtype=bool)
    expected[:2, :] = expected[-2:, :] = False
    expected[:, :2] = expected[:, -2:] = False
    assert np.array_equal(valid, expected)
    assert np.allclose(image[~valid], params.background)


def test_analytic_and_bicubic_agree_on_valid_interior(speckle):
    """The two sampling methods must describe the same deformed texture."""
    params, dots, reference, _ = speckle
    model = AffineModel(params.width, params.height, strain_x=0.02, strain_y=-0.01,
                        rotation_deg=1.0, translation=(0.3, -0.2))
    analytic, valid_a = warp_analytic(model, 1.0, dots, params)
    bicubic, valid_b = warp_bicubic(reference, model, 1.0, params)
    both = valid_a & valid_b
    diff = analytic[both] - bicubic[both]
    # Interpolation of a rasterised speckle differs at dot edges, but the bulk of
    # the image agrees to a couple of grey levels.
    assert np.mean(np.abs(diff)) < 3.0
    assert np.percentile(np.abs(diff), 99) < 20.0


def test_warp_image_dispatch_and_unknown_method(speckle):
    params, dots, reference, _ = speckle
    model = AffineModel(params.width, params.height)
    img_a, _ = warp_image("analytic", model, 0.0, dots, params, reference)
    assert np.array_equal(img_a, reference)
    with pytest.raises(ValueError):
        warp_image("nearest", model, 0.0, dots, params, reference)


# --------------------------------------------------------------------------- #
# PNG output
# --------------------------------------------------------------------------- #


def test_save_gray_png_is_8bit_grayscale(tmp_path):
    from PIL import Image

    image = np.linspace(-10, 300, 64).reshape(8, 8)
    path = tmp_path / "gray.png"
    save_gray_png(image, path)
    with Image.open(path) as img:
        assert img.mode == "L"
        data = np.asarray(img)
    assert data.dtype == np.uint8
    assert data.min() == 0 and data.max() == 255


def test_save_mask_png_convention(tmp_path):
    from PIL import Image

    mask = np.array([[True, False], [False, True]])
    path = tmp_path / "mask.png"
    save_mask_png(mask, path)
    with Image.open(path) as img:
        assert img.mode == "L"
        data = np.asarray(img)
    assert data.tolist() == [[255, 0], [0, 255]]


def test_save_rgb_png(tmp_path):
    from PIL import Image

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 1] = 200
    path = tmp_path / "color.png"
    save_rgb_png(rgb, path)
    with Image.open(path) as img:
        assert img.mode == "RGB"
        assert np.asarray(img)[0, 0].tolist() == [0, 200, 0]