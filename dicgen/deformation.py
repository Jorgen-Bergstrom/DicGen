"""Deformation models for the synthetic 2D DIC dataset generator.

This module defines the *exact* deformation that is baked into the generated
images, and the analytic ground-truth fields that come with them.

Conventions used everywhere in this package
-------------------------------------------
* Coordinates are stored as ``(x, y)`` pairs, where ``x`` is the **column**
  index and ``y`` is the **row** index of the image (the usual "image
  coordinates" with ``y`` pointing down).  Arrays of points have shape
  ``(..., 2)`` with the last axis ordered ``(x, y)``.
* The image centre is ``I_center = ((width - 1) / 2, (height - 1) / 2)``,
  i.e. the centre of the pixel grid.
* All deformations are *centred* on ``I_center`` and are zero for image 0.

Load stepping (default)
-----------------------
For a sequence of ``n`` images (``n >= 2``) the load fraction of image ``i`` is::

    t_i = i / (n - 1),      i = 0, 1, ..., n - 1

so ``t_0 = 0`` (undeformed reference) and ``t_{n-1} = 1`` (full deformation).
The total deformation is therefore divided **equally** over the sequence.

Affine model (default, ``mode="affine"``)
-----------------------------------------
For image ``i`` the deformation gradient is::

    F_i = R(theta * t_i) @ diag(1 + e_x * t_i, 1 + e_y * t_i)

where ``R(a)`` is the standard 2D rotation matrix::

    R(a) = [[cos a, -sin a],
            [sin a,  cos a]]

and ``e_x``, ``e_y`` are the **total engineering stretches** requested on the
command line (``--strain-x``, ``--strain-y``) and ``theta`` the total rotation
(``--rotation``, in degrees).  Because ``R`` is orthogonal and ``diag(...)`` is
symmetric positive definite, the above ``F_i`` *is* its own polar decomposition::

    F_i = R(theta * t_i) @ U_i,     U_i = diag(1 + e_x t_i, 1 + e_y t_i)

so ``theta * t_i`` is exactly the polar (rigid) rotation and ``e_x t_i`` /
``e_y t_i`` are exactly the principal engineering stretches, for any magnitude
of the parameters.

The forward mapping from reference coordinates ``X`` (undeformed, image 0) to
deformed coordinates ``x`` (image ``i``) is::

    x = I_center + F_i @ (X - I_center) + t_i * T

where ``T`` is the optional total rigid-body translation
(``--translation-x`` / ``--translation-y``, in pixels, default ``(0, 0)``).

The ground-truth displacement field is therefore::

    u(X) = x - X = (F_i - I) @ (X - I_center) + t_i * T

which is *linear* in ``X``; its gradient ``du/dX = F_i - I`` is constant, so the
reported (engineering, small-strain) strain field is uniform::

    epsilon_xx = e_x * t_i
    epsilon_yy = e_y * t_i
    gamma_xy   = 0

The reported ``epsilon_*`` values are the strains of the *stretch* factors of
the parameterisation (equivalently, the strains along the principal axes of the
polar decomposition); they are exact for every load level.  For completeness the
ground truth also reports the components of the *small-strain tensor in the
image frame*, ``sym(F_i) - I`` (columns ``eps_xx_img``, ``eps_yy_img``,
``gamma_xy_img``); these differ from the parameter strains by ``O(theta * e)``.

Optional sinusoidal model (``mode="sinusoid"``)
-----------------------------------------------
Adds a non-homogeneous displacement field that is switched on together with the
affine part (same ``t_i``)::

    u_sin_x(X) = A_x * sin(2*pi*(X_x - c_x) / L_x)
    u_sin_y(X) = A_y * sin(2*pi*(X_y - c_y) / L_y)

so that::

    x = I_center + F_i @ (X - I_center) + t_i * T + t_i * u_sin(X)

The Jacobian of this map is ``J_i(X) = F_i + t_i * diag(du_sin_x/dX_x,
du_sin_y/dX_y)``, which is available analytically; the inverse mapping
``x -> X`` is obtained with a vectorised Newton iteration (the affine part is
used as the initial guess).

Documentation of the finite-strain interpretation
-------------------------------------------------
Only *engineering* (infinitesimal) strains are reported.  The exact logarithmic
(Hencky) strains of the principal stretches are ``ln(1 + e_x t_i)`` and
``ln(1 + e_y t_i)``; the Green-Lagrange strain is ``((1+e)^2 - 1) / 2``.  For
the small strains typical of DIC validation (<= 2%) the differences are at the
level of the last significant digit of the parameters, but a user who needs a
different strain measure can recompute it exactly from ``F_i`` (written to
``ground_truth.csv``) and the Jacobian fields written on the sample grid.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import numpy as np

__all__ = [
    "DeformationError",
    "DeformationModel",
    "AffineModel",
    "SinusoidalModel",
    "make_model",
    "load_fraction",
    "rotation_matrix",
]


class DeformationError(ValueError):
    """Raised when the deformation model is invalid or cannot be inverted."""


def load_fraction(index: int, num_images: int) -> float:
    """Return the load fraction ``t_i = i / (n - 1)`` for image ``index``.

    ``t_0 = 0`` (undeformed reference) and ``t_{n-1} = 1`` (full deformation).

    Parameters
    ----------
    index:
        Image index ``i`` (``0 <= i < num_images``).
    num_images:
        Total number of images ``n >= 2``.

    Raises
    ------
    DeformationError
        If ``num_images < 2`` or the index is out of range.
    """
    if num_images < 2:
        raise DeformationError(
            "at least 2 images are required to define the load stepping "
            f"t_i = i / (n - 1); got n = {num_images}"
        )
    if not 0 <= index < num_images:
        raise DeformationError(
            f"image index {index} out of range for n = {num_images}"
        )
    return index / (num_images - 1)


def rotation_matrix(angle_deg: float | np.ndarray) -> np.ndarray:
    """Standard 2D rotation matrix ``R(a)`` for an angle in **degrees**.

    Returns an array of shape ``(2, 2)`` (scalar input) or
    ``(..., 2, 2)`` (array input).
    """
    a = np.deg2rad(np.asarray(angle_deg, dtype=float))
    c, s = np.cos(a), np.sin(a)
    if c.ndim == 0:
        return np.array([[c, -s], [s, c]], dtype=float)
    out = np.empty(c.shape + (2, 2), dtype=float)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    return out


class DeformationModel:
    """2D affine deformation, centred on the image centre, zero at ``t = 0``.

    Parameters
    ----------
    width, height:
        Image size in pixels.  The centre is ``((width-1)/2, (height-1)/2)``.
    strain_x, strain_y:
        **Total** engineering stretches over the whole sequence (``0.02`` = 2 %).
    rotation_deg:
        **Total** rotation over the whole sequence, in degrees.
    translation:
        Optional **total** rigid-body translation ``(Tx, Ty)`` in pixels, applied
        linearly with ``t`` in addition to the affine stretch/rotation.
    """

    mode = "affine"

    def __init__(
        self,
        width: int,
        height: int,
        strain_x: float = 0.0,
        strain_y: float = 0.0,
        rotation_deg: float = 0.0,
        translation: Sequence[float] = (0.0, 0.0),
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.center = np.array(
            ((self.width - 1) / 2.0, (self.height - 1) / 2.0), dtype=float
        )
        self.strain_x = float(strain_x)
        self.strain_y = float(strain_y)
        self.rotation_deg = float(rotation_deg)
        self.translation = np.asarray(translation, dtype=float).reshape(2)
        self._validate()

    # ------------------------------------------------------------------ setup
    def _validate(self) -> None:
        if self.width < 8 or self.height < 8:
            raise DeformationError(
                f"image size must be at least 8x8 pixels, got "
                f"{self.width}x{self.height}"
            )
        for name, value in (
            ("strain_x", self.strain_x),
            ("strain_y", self.strain_y),
            ("rotation_deg", self.rotation_deg),
            ("translation_x", self.translation[0]),
            ("translation_y", self.translation[1]),
        ):
            if not math.isfinite(value):
                raise DeformationError(f"{name} must be finite, got {value!r}")
        for name, value in (("strain_x", self.strain_x), ("strain_y", self.strain_y)):
            if value <= -1.0:
                raise DeformationError(
                    f"{name} = {value} collapses or inverts the image "
                    "(the stretch 1 + e must stay positive)"
                )

    # ------------------------------------------------------- affine quantities
    def stretch(self, t: float) -> np.ndarray:
        """Principal stretch factors ``(1 + e_x t, 1 + e_y t)`` at load ``t``."""
        return np.array(
            (1.0 + self.strain_x * t, 1.0 + self.strain_y * t), dtype=float
        )

    def affine_matrix(self, t: float) -> np.ndarray:
        """The deformation gradient ``F(t) = R(theta t) @ diag(1 + e_x t, 1 + e_y t)``."""
        return rotation_matrix(self.rotation_deg * t) @ np.diag(self.stretch(t))

    def affine_offset(self, t: float) -> np.ndarray:
        """Rigid-body translation contribution ``t * T`` (pixels)."""
        return t * self.translation

    # ------------------------------------------------------------ public API
    def jacobian(self, X: np.ndarray, t: float) -> np.ndarray:
        """Deformation gradient ``d x / d X`` at reference points ``X``.

        ``X`` has shape ``(..., 2)``; the result has shape ``(..., 2, 2)``.
        For the affine model the result is constant in space.
        """
        X = np.asarray(X, dtype=float)
        F = self.affine_matrix(t)
        out = np.empty(X.shape[:-1] + (2, 2), dtype=float)
        out[...] = F
        return out

    def transform(self, X: np.ndarray, t: float) -> np.ndarray:
        """Forward map: reference coordinates ``X`` -> deformed coordinates ``x``."""
        X = np.asarray(X, dtype=float)
        F = self.affine_matrix(t)
        return (X - self.center) @ F.T + self.center + self.affine_offset(t)

    def displacement(self, X: np.ndarray, t: float) -> np.ndarray:
        """Ground-truth displacement ``u(X) = x(X) - X`` (pixels)."""
        X = np.asarray(X, dtype=float)
        return self.transform(X, t) - X

    def inverse_transform(self, x: np.ndarray, t: float) -> np.ndarray:
        """Backward map: deformed coordinates ``x`` -> reference coordinates ``X``."""
        x = np.asarray(x, dtype=float)
        F = self.affine_matrix(t)
        try:
            Finv = np.linalg.inv(F)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded by _validate
            raise DeformationError(f"deformation gradient is singular: {F!r}") from exc
        return (x - self.center - self.affine_offset(t)) @ Finv.T + self.center

    # ---------------------------------------------------------- strain fields
    def strain_spec(self, X: np.ndarray, t: float) -> Dict[str, np.ndarray]:
        """The *documented* strain convention (uniform for the affine model).

        Returns ``eps_xx``, ``eps_yy``, ``gamma_xy`` broadcast to the shape of
        ``X[..., 0]``.
        """
        X = np.asarray(X, dtype=float)
        shape = X.shape[:-1]
        return {
            "eps_xx": np.full(shape, self.strain_x * t),
            "eps_yy": np.full(shape, self.strain_y * t),
            "gamma_xy": np.zeros(shape),
        }

    def strain_image_frame(self, X: np.ndarray, t: float) -> Dict[str, np.ndarray]:
        """Small-strain tensor in the image frame from ``J = F + t * d u_sin / dX``.

        ``eps_xx_img = J11 - 1``, ``eps_yy_img = J22 - 1`` and the engineering
        shear ``gamma_xy_img = J12 + J21``.  For a pure stretch/rotation this is
        the symmetric part of ``F - I``.
        """
        J = self.jacobian(X, t)
        return {
            "eps_xx_img": J[..., 0, 0] - 1.0,
            "eps_yy_img": J[..., 1, 1] - 1.0,
            "gamma_xy_img": J[..., 0, 1] + J[..., 1, 0],
        }

    # ------------------------------------------------------------- reporting
    def parameters_at(self, t: float) -> Dict[str, float]:
        """Scalar parameters of the model at load fraction ``t``."""
        return {
            "t": float(t),
            "e_x_t": self.strain_x * t,
            "e_y_t": self.strain_y * t,
            "theta_t_deg": self.rotation_deg * t,
            "translation_x_t": self.translation[0] * t,
            "translation_y_t": self.translation[1] * t,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable description of the model (for ``metadata.json``)."""
        return {
            "mode": self.mode,
            "total_strain_x": self.strain_x,
            "total_strain_y": self.strain_y,
            "total_rotation_deg": self.rotation_deg,
            "total_translation_px": [float(self.translation[0]), float(self.translation[1])],
            "image_center_px": [float(self.center[0]), float(self.center[1])],
            "load_stepping": "t_i = i / (n - 1)",
        }


class SinusoidalModel(DeformationModel):
    """Affine deformation plus a sinusoidal (non-homogeneous) displacement field.

    See the module docstring for the exact formulas.  The extra field is
    ``u_sin`` (documented above) switched on with the same load fraction ``t``.
    """

    mode = "sinusoid"

    def __init__(
        self,
        width: int,
        height: int,
        strain_x: float = 0.0,
        strain_y: float = 0.0,
        rotation_deg: float = 0.0,
        translation: Sequence[float] = (0.0, 0.0),
        amplitude_x: float = 0.0,
        amplitude_y: float = 0.0,
        wavelength_x: float | None = None,
        wavelength_y: float | None = None,
        newton_tol: float = 1e-10,
        newton_max_iter: int = 30,
    ) -> None:
        super().__init__(width, height, strain_x, strain_y, rotation_deg, translation)
        self.amplitude_x = float(amplitude_x)
        self.amplitude_y = float(amplitude_y)
        self.wavelength_x = float(wavelength_x if wavelength_x is not None else width)
        self.wavelength_y = float(wavelength_y if wavelength_y is not None else height)
        self.newton_tol = float(newton_tol)
        self.newton_max_iter = int(newton_max_iter)
        for name, value in (
            ("--sin-amplitude-x", self.amplitude_x),
            ("--sin-amplitude-y", self.amplitude_y),
            ("--sin-wavelength-x", self.wavelength_x),
            ("--sin-wavelength-y", self.wavelength_y),
        ):
            if not math.isfinite(value):
                raise DeformationError(f"{name} must be finite, got {value!r}")
        if self.wavelength_x <= 0 or self.wavelength_y <= 0:
            raise DeformationError("sinusoidal wavelengths must be positive")
        # Keep the map monotone (and Newton well behaved) in the 1D sub-maps.
        for name, amp, lam in (
            ("--sin-amplitude-x", self.amplitude_x, self.wavelength_x),
            ("--sin-amplitude-y", self.amplitude_y, self.wavelength_y),
        ):
            limit = 0.95 * lam / (2.0 * math.pi)
            if abs(amp) > limit:
                raise DeformationError(
                    f"{name}={amp} is too large for a wavelength of {lam} px; "
                    f"keep |amplitude| <= {limit:.3f} px so that the mapping stays "
                    "monotone and invertible"
                )

    # ------------------------------------------------------------- sinusoid
    def sinusoid_displacement(self, X: np.ndarray, t: float) -> np.ndarray:
        """The extra non-homogeneous displacement ``u_sin(X)`` (not yet scaled by ``t``)."""
        X = np.asarray(X, dtype=float)
        arg_x = 2.0 * math.pi * (X[..., 0] - self.center[0]) / self.wavelength_x
        arg_y = 2.0 * math.pi * (X[..., 1] - self.center[1]) / self.wavelength_y
        out = np.empty_like(X)
        out[..., 0] = self.amplitude_x * np.sin(arg_x)
        out[..., 1] = self.amplitude_y * np.sin(arg_y)
        return out

    def _sinusoid_slopes(self, X: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        arg_x = 2.0 * math.pi * (X[..., 0] - self.center[0]) / self.wavelength_x
        arg_y = 2.0 * math.pi * (X[..., 1] - self.center[1]) / self.wavelength_y
        kx = 2.0 * math.pi / self.wavelength_x
        ky = 2.0 * math.pi / self.wavelength_y
        return (
            self.amplitude_x * kx * np.cos(arg_x),
            self.amplitude_y * ky * np.cos(arg_y),
        )

    def jacobian(self, X: np.ndarray, t: float) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        J = super().jacobian(X, t)
        if t != 0.0:
            J[..., 0, 0] += t * self.amplitude_x * (2.0 * math.pi / self.wavelength_x) * np.cos(
                2.0 * math.pi * (X[..., 0] - self.center[0]) / self.wavelength_x
            )
            J[..., 1, 1] += t * self.amplitude_y * (2.0 * math.pi / self.wavelength_y) * np.cos(
                2.0 * math.pi * (X[..., 1] - self.center[1]) / self.wavelength_y
            )
        return J

    def transform(self, X: np.ndarray, t: float) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return super().transform(X, t) + t * self.sinusoid_displacement(X, t)

    def inverse_transform(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        X = super().inverse_transform(x, t)  # affine initial guess
        if t == 0.0 or (self.amplitude_x == 0.0 and self.amplitude_y == 0.0):
            return X
        for _ in range(self.newton_max_iter):
            residual = self.transform(X, t) - x
            scale = float(np.max(np.abs(residual))) if residual.size else 0.0
            if scale <= self.newton_tol:
                return X
            J = self.jacobian(X, t)
            det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
            if np.any(np.abs(det) < 1e-12):
                raise DeformationError(
                    "the sinusoidal deformation gradient became singular while "
                    "inverting the mapping; reduce the sinusoidal amplitudes"
                )
            rx, ry = residual[..., 0], residual[..., 1]
            dX = np.empty_like(residual)
            dX[..., 0] = (J[..., 1, 1] * rx - J[..., 0, 1] * ry) / det
            dX[..., 1] = (-J[..., 1, 0] * rx + J[..., 0, 0] * ry) / det
            X = X - dX
        residual = float(np.max(np.abs(self.transform(X, t) - x))) if X.size else 0.0
        raise DeformationError(
            "Newton iteration failed to invert the sinusoidal mapping "
            f"(residual {residual:.3e} px after {self.newton_max_iter} iterations)"
        )

    def strain_spec(self, X: np.ndarray, t: float) -> Dict[str, np.ndarray]:
        X = np.asarray(X, dtype=float)
        kx = 2.0 * math.pi / self.wavelength_x
        ky = 2.0 * math.pi / self.wavelength_y
        return {
            "eps_xx": self.strain_x * t
            + t * self.amplitude_x * kx * np.cos(kx * (X[..., 0] - self.center[0])),
            "eps_yy": self.strain_y * t
            + t * self.amplitude_y * ky * np.cos(ky * (X[..., 1] - self.center[1])),
            "gamma_xy": np.zeros(X.shape[:-1]),
        }

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "mode": self.mode,
                "sin_amplitude_x_px": self.amplitude_x,
                "sin_amplitude_y_px": self.amplitude_y,
                "sin_wavelength_x_px": self.wavelength_x,
                "sin_wavelength_y_px": self.wavelength_y,
            }
        )
        return d


def make_model(
    mode: str,
    width: int,
    height: int,
    *,
    strain_x: float = 0.0,
    strain_y: float = 0.0,
    rotation_deg: float = 0.0,
    translation: Sequence[float] = (0.0, 0.0),
    amplitude_x: float = 0.0,
    amplitude_y: float = 0.0,
    wavelength_x: float | None = None,
    wavelength_y: float | None = None,
) -> DeformationModel:
    """Factory building the deformation model requested by ``mode``."""
    mode = str(mode).lower()
    if mode == "affine":
        return AffineModel(
            width, height, strain_x, strain_y, rotation_deg, translation
        )
    if mode == "sinusoid":
        return SinusoidalModel(
            width,
            height,
            strain_x,
            strain_y,
            rotation_deg,
            translation,
            amplitude_x=amplitude_x,
            amplitude_y=amplitude_y,
            wavelength_x=wavelength_x,
            wavelength_y=wavelength_y,
        )
    raise DeformationError(
        f"unknown deformation mode {mode!r}; expected 'affine' or 'sinusoid'"
    )


class AffineModel(DeformationModel):
    """Explicit name for the default affine model (see :class:`DeformationModel`)."""

    mode = "affine"