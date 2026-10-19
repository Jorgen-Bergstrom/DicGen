"""Ground-truth export: CSV fields, full-grid displacement, masks and metadata.

Everything written here is *analytic* (computed from the deformation model), not
measured from the images, so it can be used directly to score a DIC solver.

Files produced in the output directory (unless disabled by a command-line flag):

``ground_truth.csv``
    One row per image with the exact transform and strain parameters.  See
    :data:`IMAGE_COLUMNS`.
``ground_truth_grid.csv``
    The same information evaluated on a regular sample grid: full displacement
    ``(ux, uy)``, image-frame strain and the Jacobian at every grid point.
    See :data:`GRID_COLUMNS`.
``displacement_field.npy`` (optional, ``--save-displacement-npy``)
    ``float32`` array of shape ``(n, 2, height, width)``, axis 1 = ``u_x``,
    ``u_y``, axes 2/3 = pixel ``(row=y, col=x)``.
``metadata.json``
    Every parameter needed to reproduce the run, plus per-file SHA-256 digests.

No wall-clock timestamps are written anywhere, so repeating a run with the same
arguments produces byte-identical output.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .imagegen import pixel_grid, valid_mask

__all__ = [
    "IMAGE_COLUMNS",
    "GRID_COLUMNS",
    "sample_grid",
    "image_records",
    "write_ground_truth_csv",
    "grid_records",
    "write_grid_csv",
    "save_displacement_field",
    "sha256_file",
    "build_file_manifest",
    "build_metadata",
    "write_json",
]

IMAGE_COLUMNS: Sequence[str] = (
    "image_index",  # 0-based index i used by t_i; 0 == reference == image0001.png
    "image",  # file name in the output directory
    "t",  # load fraction t_i = i / (n - 1)
    "F11",
    "F12",
    "F21",
    "F22",  # deformation gradient F_i, x = c + F (X - c) + t * T
    "tx_px",
    "ty_px",  # rigid-body translation t * T (pixels)
    "e_x_t",
    "e_y_t",
    "theta_t_deg",  # affine parameters at load t
    "eps_xx",
    "eps_yy",
    "gamma_xy",  # documented (uniform for affine) engineering strains
    "strain_uniform",  # 1 if the spec strain field is uniform, else 0
    "eps_xx_img_mean",
    "eps_yy_img_mean",
    "gamma_xy_img_mean",  # mean of sym(J)-I over the sample grid
    "F_lambda_max",  # largest principal stretch (spectral norm)
    "F_det",
    "u_max_px",
    "u_mean_px",  # displacement magnitude statistics over the pixel grid
    "valid_fraction",  # fraction of pixels coming from inside the reference
)

GRID_COLUMNS: Sequence[str] = (
    "image_index",
    "image",
    "t",
    "X_px",
    "Y_px",  # reference (undeformed) coordinates
    "x_px",
    "y_px",  # deformed coordinates x = transform(X, t)
    "ux_px",
    "uy_px",  # ground-truth displacement u = x - X
    "eps_xx",
    "eps_yy",
    "gamma_xy",  # documented engineering strain convention
    "eps_xx_img",
    "eps_yy_img",
    "gamma_xy_img",  # small-strain tensor in the image frame, sym(J) - I
    "J11",
    "J12",
    "J21",
    "J22",  # Jacobian du/dX + I at the grid point
    "in_ref",
    "in_def",  # 1 if X (resp. x) lies inside the image
)


def sample_grid(width: int, height: int, nx: int, ny: int) -> np.ndarray:
    """Regular sample grid of ``(x, y)`` points including the image borders.

    Rows are ordered with ``y`` varying slowest (row-major), so the CSV reads
    like the image.
    """
    if nx < 2 or ny < 2:
        raise ValueError(f"sample grid must be at least 2x2 points, got {nx}x{ny}")
    xs = np.linspace(0.0, float(width - 1), int(nx))
    ys = np.linspace(0.0, float(height - 1), int(ny))
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack((gx, gy), axis=-1).reshape(-1, 2)


def _fmt(value: float) -> str:
    """Compact, round-trippable fixed formatting for CSV numbers."""
    return f"{float(value):.12g}"


def _displacement_stats(model, t: float, width: int, height: int) -> Dict[str, float]:
    """Max/mean displacement magnitude over the pixel grid (analytic)."""
    coords = pixel_grid(width, height)
    u = model.displacement(coords, t)
    magnitude = np.hypot(u[..., 0], u[..., 1])
    return {
        "u_max_px": float(magnitude.max()),
        "u_mean_px": float(magnitude.mean()),
    }


def image_records(
    model,
    num_images: int,
    filenames: Sequence[str],
    valid_fractions: Optional[Sequence[float]] = None,
    *,
    grid: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Build the per-image ground-truth records (one dict per image)."""
    from .deformation import load_fraction

    if len(filenames) != num_images:
        raise ValueError(
            f"expected {num_images} file names, got {len(filenames)}"
        )
    if valid_fractions is None:
        valid_fractions = [1.0] * num_images
    if grid is None:
        grid = sample_grid(model.width, model.height, 2, 2)
    records: List[Dict[str, Any]] = []
    for i in range(num_images):
        t = load_fraction(i, num_images)
        F = model.affine_matrix(t)
        d = model.affine_offset(t)
        params = model.parameters_at(t)
        spec = model.strain_spec(grid, t)
        image_strain = model.strain_image_frame(grid, t)
        stats = _displacement_stats(model, t, model.width, model.height)
        spec_uniform = bool(
            np.allclose(spec["eps_xx"], spec["eps_xx"].flat[0])
            and np.allclose(spec["eps_yy"], spec["eps_yy"].flat[0])
            and np.allclose(spec["gamma_xy"], 0.0)
        )
        record = {
            "image_index": i,
            "image": filenames[i],
            "t": t,
            "F11": F[0, 0],
            "F12": F[0, 1],
            "F21": F[1, 0],
            "F22": F[1, 1],
            "tx_px": d[0],
            "ty_px": d[1],
            "e_x_t": params["e_x_t"],
            "e_y_t": params["e_y_t"],
            "theta_t_deg": params["theta_t_deg"],
            "eps_xx": float(np.mean(spec["eps_xx"])),
            "eps_yy": float(np.mean(spec["eps_yy"])),
            "gamma_xy": float(np.mean(spec["gamma_xy"])),
            "strain_uniform": 1 if spec_uniform else 0,
            "eps_xx_img_mean": float(np.mean(image_strain["eps_xx_img"])),
            "eps_yy_img_mean": float(np.mean(image_strain["eps_yy_img"])),
            "gamma_xy_img_mean": float(np.mean(image_strain["gamma_xy_img"])),
            "F_lambda_max": float(np.linalg.norm(F, ord=2)),
            "F_det": float(np.linalg.det(F)),
            "valid_fraction": float(valid_fractions[i]),
            **stats,
        }
        records.append(record)
    return records


def write_ground_truth_csv(records: Iterable[Dict[str, Any]], path) -> None:
    """Write the per-image ground-truth CSV (see :data:`IMAGE_COLUMNS`)."""
    rows = list(records)
    if not rows:
        raise ValueError("no ground-truth records to write")
    missing = [c for c in IMAGE_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"ground-truth records are missing columns: {missing}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(IMAGE_COLUMNS)
        for record in rows:
            writer.writerow(
                [
                    _fmt(record[c]) if isinstance(record[c], (int, float, np.floating, np.integer)) else record[c]
                    for c in IMAGE_COLUMNS
                ]
            )


def grid_records(
    model,
    num_images: int,
    filenames: Sequence[str],
    grid: np.ndarray,
) -> List[Dict[str, float]]:
    """Analytic ground truth evaluated on a sample grid, for every image."""
    from .deformation import load_fraction

    rows: List[Dict[str, float]] = []
    for i in range(num_images):
        t = load_fraction(i, num_images)
        X = np.asarray(grid, dtype=float)
        x = model.transform(X, t)
        u = x - X
        spec = model.strain_spec(X, t)
        image_strain = model.strain_image_frame(X, t)
        J = model.jacobian(X, t)
        in_ref = valid_mask(X, model.width, model.height)
        in_def = valid_mask(x, model.width, model.height)
        for k in range(X.shape[0]):
            rows.append(
                {
                    "image_index": i,
                    "image": filenames[i],
                    "t": t,
                    "X_px": X[k, 0],
                    "Y_px": X[k, 1],
                    "x_px": x[k, 0],
                    "y_px": x[k, 1],
                    "ux_px": u[k, 0],
                    "uy_px": u[k, 1],
                    "eps_xx": spec["eps_xx"][k],
                    "eps_yy": spec["eps_yy"][k],
                    "gamma_xy": spec["gamma_xy"][k],
                    "eps_xx_img": image_strain["eps_xx_img"][k],
                    "eps_yy_img": image_strain["eps_yy_img"][k],
                    "gamma_xy_img": image_strain["gamma_xy_img"][k],
                    "J11": J[k, 0, 0],
                    "J12": J[k, 0, 1],
                    "J21": J[k, 1, 0],
                    "J22": J[k, 1, 1],
                    "in_ref": int(bool(in_ref[k])),
                    "in_def": int(bool(in_def[k])),
                }
            )
    return rows


def write_grid_csv(rows: Iterable[Dict[str, Any]], path) -> None:
    """Write the sample-grid ground-truth CSV (see :data:`GRID_COLUMNS`)."""
    data = list(rows)
    if not data:
        raise ValueError("no grid ground-truth rows to write")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(GRID_COLUMNS)
        for row in data:
            writer.writerow(
                [
                    _fmt(row[c]) if isinstance(row[c], (int, float, np.floating, np.integer)) else row[c]
                    for c in GRID_COLUMNS
                ]
            )


def save_displacement_field(model, num_images: int, path) -> Dict[str, Any]:
    """Save the analytic displacement field on the full pixel grid as ``.npy``.

    Shape ``(n, 2, height, width)`` ``float32``; axis 1 indexes ``(u_x, u_y)``.
    """
    from .deformation import load_fraction

    coords = pixel_grid(model.width, model.height)
    field = np.empty((num_images, 2, model.height, model.width), dtype=np.float32)
    for i in range(num_images):
        t = load_fraction(i, num_images)
        u = model.displacement(coords, t)
        field[i, 0] = u[..., 0].astype(np.float32)
        field[i, 1] = u[..., 1].astype(np.float32)
    np.save(path, field)
    return {
        "path": str(path),
        "shape": list(field.shape),
        "dtype": "float32",
        "layout": "(image_index, [ux, uy], row=y, col=x)",
    }


def sha256_file(path) -> str:
    """SHA-256 digest of a file, used for the run manifest."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_manifest(output_dir, relative_paths: Iterable[str]) -> List[Dict[str, Any]]:
    """List of ``{path, size_bytes, sha256}`` for every generated file."""
    root = Path(output_dir)
    manifest = []
    for rel in sorted(relative_paths):
        path = root / rel
        manifest.append(
            {
                "path": str(rel),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return manifest


def build_metadata(
    *,
    tool: str,
    version: str,
    arguments: Dict[str, Any],
    sequence: Dict[str, Any],
    model: Dict[str, Any],
    speckle: Dict[str, Any],
    dots: Dict[str, Any],
    coverage: Dict[str, Any],
    noise: Dict[str, Any],
    sampling: Dict[str, Any],
    ground_truth: Dict[str, Any],
    files: List[Dict[str, Any]],
    movie: Optional[Dict[str, Any]] = None,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble the ``metadata.json`` payload (fully reproducible run record)."""
    return {
        "tool": tool,
        "version": version,
        "metadata_format_version": 1,
        "file_naming": "image%0*d.png, 1-based, image 0 (undeformed reference) first",
        "mask_convention": "mask*.png: 255 = valid (correlate), 0 = invalid (no speckle data)",
        "deterministic": True,
        "wall_clock_timestamps": False,
        "arguments": arguments,
        "sequence": sequence,
        "deformation": model,
        "speckle": speckle,
        "dots": dots,
        "coverage": coverage,
        "noise": noise,
        "sampling": sampling,
        "ground_truth": ground_truth,
        "movie": movie if movie is not None else {"written": False},
        "files": files,
        "notes": list(notes or []),
    }


def write_json(path, payload: Dict[str, Any]) -> None:
    """Write JSON deterministically (sorted keys, fixed indentation, trailing LF)."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")