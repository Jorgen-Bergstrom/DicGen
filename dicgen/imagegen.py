"""Image synthesis: warping (sampling), noise, and file output.

Two sampling methods are available for the deformed images:

``analytic`` (default)
    The reference speckle is *not* rasterised and re-interpolated; instead the
    continuous dot field is evaluated directly at the inverse-mapped
    coordinates ``X = F_i^{-1}(x)`` of every deformed pixel.  The image content
    therefore matches the analytic ground truth *exactly* (no interpolation
    error), which is the cleanest possible validation dataset.  Cost is the same
    order as rasterising the reference, because each dot only affects a small
    bounding box of the deformed image.

``bicubic``
    The classical DIC-dataset recipe: rasterise the reference at the pixel grid,
    then sample it at ``X = F_i^{-1}(x)`` with a bicubic spline
    (``scipy.ndimage.map_coordinates``, order 3).  This reproduces the small
    interpolation smoothing/bias a real camera + DIC pipeline would see.  The
    valid region is shrunk by 2 px so that no pixel is generated from a spline
    stencil that reaches outside the reference image.

Both methods honour the same conventions:

* out-of-domain samples (``X`` outside ``[0, W-1] x [0, H-1]``) are filled with
  the background grey level and recorded as invalid in the exported mask;
* noise is added *after* warping, per image, from the run's single RNG;
* grey levels are clipped to ``[0, 255]`` and rounded to integers when the
  8-bit PNG is written.  Clipping happens deliberately and only at that last
  step; all intermediate computations are float.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .speckle import DotSet, SpeckleParams, rasterise_dots

__all__ = [
    "pixel_grid",
    "render_reference",
    "spectral_norm_2x2",
    "valid_mask",
    "warp_analytic",
    "warp_bicubic",
    "warp_image",
    "apply_noise",
    "to_uint8",
    "save_gray_png",
    "save_mask_png",
    "save_rgb_png",
    "displacement_colour_map",
    "draw_quiver",
]

VIRIDIS_CONTROL_POINTS = np.array(
    [
        [68, 1, 84],
        [72, 40, 120],
        [62, 74, 137],
        [49, 104, 142],
        [38, 130, 142],
        [31, 158, 137],
        [53, 183, 121],
        [109, 205, 89],
        [145, 220, 80],
        [190, 229, 60],
        [253, 231, 37],
    ],
    dtype=float,
)


def pixel_grid(width: int, height: int, dtype=np.float64) -> np.ndarray:
    """Pixel-centre coordinates as ``(height, width, 2)`` ``(x, y)`` array."""
    xs = np.arange(width, dtype=dtype)
    ys = np.arange(height, dtype=dtype)
    grid = np.empty((height, width, 2), dtype=dtype)
    grid[..., 0] = xs[None, :]
    grid[..., 1] = ys[:, None]
    return grid


def render_reference(
    dots: DotSet,
    params: SpeckleParams,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render the undeformed reference speckle field (image 0) as a float array."""
    coords = pixel_grid(params.width, params.height)
    return rasterise_dots(
        coords,
        dots.centres,
        dots,
        params.background,
        profile=params.profile,
        softness=params.softness,
        radius_scale=1.0,
        image=out,
    )


def spectral_norm_2x2(matrices: np.ndarray) -> np.ndarray:
    """Largest singular value of each 2x2 matrix (vectorised, closed form)."""
    m = np.asarray(matrices, dtype=float)
    a, b = m[..., 0, 0], m[..., 0, 1]
    c, d = m[..., 1, 0], m[..., 1, 1]
    # singular values of [[a,b],[c,d]]: sqrt of eigenvalues of M^T M
    s = a * a + b * b + c * c + d * d
    det = np.abs(a * d - b * c)
    disc = np.maximum(s * s - 4.0 * det * det, 0.0)
    return np.sqrt(np.maximum(0.5 * (s + np.sqrt(disc)), 0.0))


def valid_mask(coords: np.ndarray, width: int, height: int) -> np.ndarray:
    """Points inside the reference image domain ``[0, W-1] x [0, H-1]``."""
    c = np.asarray(coords, dtype=float)
    return (
        (c[..., 0] >= 0.0)
        & (c[..., 0] <= width - 1.0)
        & (c[..., 1] >= 0.0)
        & (c[..., 1] <= height - 1.0)
    )


def warp_analytic(
    model,
    t: float,
    dots: DotSet,
    params: SpeckleParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp the continuous speckle field analytically to load fraction ``t``.

    Returns ``(image, valid)`` with ``image`` the unclipped float grey levels
    (background where invalid) and ``valid`` the boolean domain mask.
    """
    coords = pixel_grid(params.width, params.height)
    X = model.inverse_transform(coords, t)
    valid = valid_mask(X, params.width, params.height)

    if len(dots):
        J = model.jacobian(dots.centres, t)
        radius_scale = float(np.max(spectral_norm_2x2(J))) if J.size else 1.0
        centres_in_grid = model.transform(dots.centres, t)
    else:  # pragma: no cover - dots are always non-empty in practice
        radius_scale = 1.0
        centres_in_grid = dots.centres
    image = rasterise_dots(
        X,
        centres_in_grid,
        dots,
        params.background,
        profile=params.profile,
        softness=params.softness,
        radius_scale=radius_scale,
    )
    image[~valid] = params.background
    return image, valid


def warp_bicubic(
    reference: np.ndarray,
    model,
    t: float,
    params: SpeckleParams,
    *,
    order: int = 3,
    margin: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp the rasterised reference with a spline interpolation of ``order``.

    ``margin`` shrinks the valid region by that many pixels (the half-width of
    the interpolation stencil), so that no output pixel is built from reference
    samples outside the image.
    """
    coords = pixel_grid(params.width, params.height)
    X = model.inverse_transform(coords, t)
    image = ndimage.map_coordinates(
        np.asarray(reference, dtype=float),
        [X[..., 1], X[..., 0]],
        order=int(order),
        mode="constant",
        cval=float(params.background),
        prefilter=True,
    )
    valid = valid_mask(X, params.width, params.height)
    if margin > 0 and order > 1:
        valid &= (
            (X[..., 0] >= margin)
            & (X[..., 0] <= params.width - 1 - margin)
            & (X[..., 1] >= margin)
            & (X[..., 1] <= params.height - 1 - margin)
        )
    image[~valid] = params.background
    return image, valid


def warp_image(
    method: str,
    model,
    t: float,
    dots: DotSet,
    params: SpeckleParams,
    reference: np.ndarray,
    *,
    interp_order: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to :func:`warp_analytic` or :func:`warp_bicubic`."""
    method = str(method).lower()
    if method == "analytic":
        return warp_analytic(model, t, dots, params)
    if method == "bicubic":
        return warp_bicubic(reference, model, t, params, order=interp_order)
    raise ValueError(f"unknown sampling method {method!r}; expected 'analytic' or 'bicubic'")


def apply_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma_additive: float = 0.0,
    sigma_multiplicative: float = 0.0,
) -> np.ndarray:
    """Add camera noise: ``I = I * (1 + N(0, s_mul)) + N(0, s_add)``.

    The input array is never modified.  With both sigmas equal to 0 the input is
    returned unchanged (upcast to float); otherwise a new array is returned.
    Nothing is clipped here -- clipping happens in :func:`to_uint8`.
    """
    out = np.asarray(image, dtype=float)
    if sigma_multiplicative > 0.0:
        out = out * (1.0 + rng.normal(0.0, sigma_multiplicative, out.shape))
    if sigma_additive > 0.0:
        out = out + rng.normal(0.0, sigma_additive, out.shape)
    return out


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Clip to ``[0, 255]``, round to nearest integer and cast to ``uint8``."""
    return np.clip(np.rint(np.asarray(image, dtype=float)), 0.0, 255.0).astype(np.uint8)


def save_gray_png(array: np.ndarray, path) -> None:
    """Write an 8-bit grayscale PNG (mode ``L``), byte-reproducibly."""
    Image.fromarray(to_uint8(array), mode="L").save(path, format="PNG", optimize=False)


def save_mask_png(mask: np.ndarray, path) -> None:
    """Write a validity mask as a PNG: 255 = valid (correlate), 0 = invalid."""
    data = np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)
    Image.fromarray(data, mode="L").save(path, format="PNG", optimize=False)


def save_rgb_png(array: np.ndarray, path) -> None:
    """Write an 8-bit RGB PNG."""
    data = np.asarray(array)
    if data.dtype != np.uint8:
        data = to_uint8(data)
    Image.fromarray(data, mode="RGB").save(path, format="PNG", optimize=False)


def _colour_lut(points: np.ndarray = VIRIDIS_CONTROL_POINTS, size: int = 256) -> np.ndarray:
    """Linear-interpolated RGB lookup table through the control points."""
    t = np.linspace(0.0, 1.0, size)
    ctrl_t = np.linspace(0.0, 1.0, len(points))
    lut = np.empty((size, 3), dtype=np.uint8)
    for channel in range(3):
        lut[:, channel] = np.clip(np.rint(np.interp(t, ctrl_t, points[:, channel])), 0, 255)
    return lut


def displacement_colour_map(
    ux: np.ndarray, uy: np.ndarray, vmax: float, *, vmin: float = 0.0
) -> np.ndarray:
    """Map the displacement magnitude to RGB with a viridis-like palette."""
    magnitude = np.hypot(np.asarray(ux, dtype=float), np.asarray(uy, dtype=float))
    span = max(float(vmax) - float(vmin), 1e-12)
    idx = np.clip(np.rint((magnitude - vmin) / span * 255.0), 0, 255).astype(np.uint8)
    return _colour_lut()[idx]


def _draw_line(rgb: np.ndarray, p0, p1, colour, thickness: int = 2) -> None:
    """Draw a straight line segment (used for the QA quiver arrows)."""
    height, width = rgb.shape[:2]
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    xs = np.rint(np.linspace(p0[0], p1[0], n)).astype(int)
    ys = np.rint(np.linspace(p0[1], p1[1], n)).astype(int)
    half = thickness // 2
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            xi = np.clip(xs + dx, 0, width - 1)
            yi = np.clip(ys + dy, 0, height - 1)
            rgb[yi, xi] = colour


def draw_quiver(
    rgb: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    *,
    stride: int = 48,
    scale: float = 1.0,
    colour: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    shift: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Overlay displacement arrows on an RGB image (visual QA only).

    ``ux``/``uy`` are the displacement fields in pixels; only every ``stride``-th
    pixel is drawn, and arrows are scaled by ``scale`` pixels per pixel of
    displacement.
    """
    out = np.asarray(rgb).copy()
    ux = np.asarray(ux, dtype=float)
    uy = np.asarray(uy, dtype=float)
    height, width = ux.shape
    for y in range(stride // 2, height, stride):
        for x in range(stride // 2, width, stride):
            dx, dy = ux[y, x] * scale, uy[y, x] * scale
            if dx == 0.0 and dy == 0.0:
                continue
            tail = (x + shift[0], y + shift[1])
            head = (tail[0] + dx, tail[1] + dy)
            _draw_line(out, tail, head, colour, thickness)
            angle = math.atan2(dy, dx)
            for delta in (math.radians(150.0), math.radians(-150.0)):
                hx = head[0] + 0.25 * math.hypot(dx, dy) * math.cos(angle + delta)
                hy = head[1] + 0.25 * math.hypot(dx, dy) * math.sin(angle + delta)
                _draw_line(out, head, (hx, hy), colour, thickness)
    return out


def qa_displacement_image(
    ux: np.ndarray,
    uy: np.ndarray,
    vmax: float,
    *,
    stride: int = 48,
    arrow_scale: float = 1.0,
) -> np.ndarray:
    """Colour map of ``|u|`` with a white arrow overlay (QA PNG content)."""
    rgb = displacement_colour_map(ux, uy, vmax)
    return draw_quiver(rgb, ux, uy, stride=stride, scale=arrow_scale)


def reference_with_arrows(
    reference: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    *,
    stride: int = 48,
    arrow_scale: float = 1.0,
) -> np.ndarray:
    """Grey reference image with red arrows, for an at-a-glance QA check."""
    rgb = np.repeat(to_uint8(reference)[..., None], 3, axis=2)
    return draw_quiver(rgb, ux, uy, stride=stride, scale=arrow_scale, colour=(255, 40, 40))


def valid_fraction(mask: np.ndarray) -> float:
    """Fraction of pixels flagged valid in ``mask``."""
    m = np.asarray(mask)
    return float(np.mean(m)) if m.size else 0.0


def image_limits(image: np.ndarray) -> Dict[str, float]:
    """Min/max/mean grey levels of a float image (used for the run summary)."""
    arr = np.asarray(image, dtype=float)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }