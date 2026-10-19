"""Tests for :mod:`dicgen.groundtruth`: CSVs, .npy, JSON and the manifest."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from dicgen.deformation import AffineModel, SinusoidalModel
from dicgen.groundtruth import (
    GRID_COLUMNS,
    IMAGE_COLUMNS,
    build_file_manifest,
    build_metadata,
    grid_records,
    image_records,
    sample_grid,
    save_displacement_field,
    sha256_file,
    write_grid_csv,
    write_ground_truth_csv,
    write_json,
)


# --------------------------------------------------------------------------- #
# Sample grid
# --------------------------------------------------------------------------- #


def test_sample_grid_layout_and_endpoints():
    grid = sample_grid(10, 6, nx=4, ny=3)
    assert grid.shape == (12, 2)
    # y varies slowest: the first `nx` points share the smallest Y.
    assert grid[:4, 1] == pytest.approx(np.zeros(4))
    assert grid[:4, 0] == pytest.approx(np.linspace(0.0, 9.0, 4))
    assert grid[-4:, 1] == pytest.approx(np.full(4, 5.0))
    # Includes the borders exactly.
    assert grid[:, 0].min() == pytest.approx(0.0)
    assert grid[:, 0].max() == pytest.approx(9.0)
    assert grid[:, 1].max() == pytest.approx(5.0)


def test_sample_grid_rejects_tiny_grid():
    with pytest.raises(ValueError):
        sample_grid(10, 10, 1, 4)


# --------------------------------------------------------------------------- #
# Per-image records
# --------------------------------------------------------------------------- #


def _affine():
    return AffineModel(64, 48, strain_x=0.02, strain_y=-0.01, rotation_deg=1.0,
                       translation=(1.5, -0.5))


def test_image_records_reference_is_undeformed_and_last_is_full_load():
    model = _affine()
    n = 5
    names = [f"image{i + 1:04d}.png" for i in range(n)]
    records = image_records(model, n, names)
    first, last = records[0], records[-1]
    assert first["t"] == 0.0
    assert first["F11"] == pytest.approx(1.0) and first["F22"] == pytest.approx(1.0)
    assert first["F12"] == pytest.approx(0.0) and first["F21"] == pytest.approx(0.0)
    assert first["u_max_px"] == pytest.approx(0.0)
    assert first["eps_xx"] == pytest.approx(0.0)
    # The final image carries the full requested deformation.
    assert last["t"] == 1.0
    assert last["e_x_t"] == pytest.approx(0.02)
    assert last["e_y_t"] == pytest.approx(-0.01)
    assert last["theta_t_deg"] == pytest.approx(1.0)
    assert last["tx_px"] == pytest.approx(1.5)
    assert last["strain_uniform"] == 1


def test_image_records_translation_matches_t_times_total():
    model = AffineModel(64, 64, translation=(2.5, -1.25))
    n = 5
    names = [f"image{i + 1:04d}.png" for i in range(n)]
    records = image_records(model, n, names)
    for i, rec in enumerate(records):
        t = i / (n - 1)
        assert rec["t"] == pytest.approx(t)
        assert rec["tx_px"] == pytest.approx(2.5 * t)
        assert rec["ty_px"] == pytest.approx(-1.25 * t)
        # A pure translation has a constant displacement magnitude |t*T|.
        assert rec["u_max_px"] == pytest.approx(np.hypot(2.5, 1.25) * t, abs=1e-9)
        assert rec["u_mean_px"] == pytest.approx(rec["u_max_px"], abs=1e-9)


def test_write_ground_truth_csv_header_and_round_trip(tmp_path):
    model = _affine()
    names = ["image0001.png", "image0002.png"]
    records = image_records(model, 2, names)
    path = tmp_path / "gt.csv"
    write_ground_truth_csv(records, path)
    from conftest import csv_header, read_csv_rows

    assert tuple(csv_header(path)) == tuple(IMAGE_COLUMNS)
    rows = read_csv_rows(path)
    assert len(rows) == 2
    # The CSV values round-trip to the same floats.
    assert float(rows[0]["u_max_px"]) == pytest.approx(records[0]["u_max_px"])
    assert float(rows[1]["e_x_t"]) == pytest.approx(0.02)


def test_write_ground_truth_csv_rejects_missing_columns(tmp_path):
    with pytest.raises(ValueError):
        write_ground_truth_csv([{"image_index": 0}], tmp_path / "gt.csv")


# --------------------------------------------------------------------------- #
# Grid records: independent recomputation of the analytic formulas
# --------------------------------------------------------------------------- #


def test_grid_records_match_analytic_formulas_for_affine():
    model = _affine()
    n = 4
    names = [f"image{i + 1:04d}.png" for i in range(n)]
    grid = sample_grid(model.width, model.height, nx=4, ny=3)
    rows = grid_records(model, n, names, grid)
    assert len(rows) == n * len(grid)
    c = model.center
    for row in rows:
        i = row["image_index"]
        t = i / (n - 1)
        assert row["t"] == pytest.approx(t)
        X = np.array([row["X_px"], row["Y_px"]])
        # u(X) = (F - I) (X - c) + t T  computed from the documented formula.
        theta = np.deg2rad(1.0 * t)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        F = R @ np.diag([1 + 0.02 * t, 1 - 0.01 * t])
        expected_u = (F - np.eye(2)) @ (X - c) + t * np.array([1.5, -0.5])
        assert row["ux_px"] == pytest.approx(expected_u[0], abs=1e-9)
        assert row["uy_px"] == pytest.approx(expected_u[1], abs=1e-9)
        assert row["x_px"] == pytest.approx(X[0] + expected_u[0], abs=1e-9)
        assert row["y_px"] == pytest.approx(X[1] + expected_u[1], abs=1e-9)
        # Uniform parameter strain and Jacobian = F.
        assert row["eps_xx"] == pytest.approx(0.02 * t)
        assert row["eps_yy"] == pytest.approx(-0.01 * t)
        assert row["gamma_xy"] == pytest.approx(0.0)
        assert row["J11"] == pytest.approx(F[0, 0])
        assert row["J12"] == pytest.approx(F[0, 1])
        assert row["J21"] == pytest.approx(F[1, 0])
        assert row["J22"] == pytest.approx(F[1, 1])
        assert row["in_ref"] == 1  # grid points are inside by construction


def test_grid_records_sinusoid_uses_nonuniform_strain():
    model = SinusoidalModel(64, 64, amplitude_x=4.0, amplitude_y=0.0,
                            wavelength_x=64.0, wavelength_y=64.0)
    grid = sample_grid(64, 64, nx=5, ny=5)
    rows = grid_records(model, 2, ["image0001.png", "image0002.png"], grid)
    last = [r for r in rows if r["image_index"] == 1]
    eps = np.array([r["eps_xx"] for r in last])
    # The sinusoidal strain field is not uniform.
    assert not np.allclose(eps, eps[0])
    kx = 2 * np.pi / 64.0
    for r in last:
        expected = 4.0 * kx * np.cos(kx * (r["X_px"] - model.center[0]))
        assert r["eps_xx"] == pytest.approx(expected, abs=1e-9)


def test_write_grid_csv_header(tmp_path):
    model = _affine()
    grid = sample_grid(64, 48, 2, 2)
    rows = grid_records(model, 2, ["image0001.png", "image0002.png"], grid)
    path = tmp_path / "grid.csv"
    write_grid_csv(rows, path)
    import csv as _csv

    with open(path, newline="") as handle:
        header = next(_csv.reader(handle))
    assert tuple(header) == tuple(GRID_COLUMNS)


# --------------------------------------------------------------------------- #
# Displacement .npy
# --------------------------------------------------------------------------- #


def test_save_displacement_field_shape_and_values(tmp_path):
    model = _affine()
    n = 3
    path = tmp_path / "disp.npy"
    info = save_displacement_field(model, n, path)
    field = np.load(path)
    assert field.shape == (n, 2, model.height, model.width)
    assert field.dtype == np.float32
    assert info["dtype"] == "float32"
    for i in range(n):
        t = i / (n - 1)
        from dicgen.imagegen import pixel_grid

        expected = model.displacement(pixel_grid(model.width, model.height), t)
        assert field[i, 0] == pytest.approx(expected[..., 0].astype(np.float32), abs=0)
        assert field[i, 1] == pytest.approx(expected[..., 1].astype(np.float32), abs=0)


# --------------------------------------------------------------------------- #
# Metadata / JSON / manifest
# --------------------------------------------------------------------------- #


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"hello dicgen" * 100)
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_file_manifest(tmp_path):
    (tmp_path / "a.txt").write_text("aaa")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("bbbb")
    manifest = build_file_manifest(tmp_path, ["a.txt", "sub/b.txt"])
    assert [m["path"] for m in manifest] == ["a.txt", "sub/b.txt"]
    assert manifest[0]["size_bytes"] == 3
    assert manifest[1]["sha256"] == hashlib.sha256(b"bbbb").hexdigest()


def _metadata_payload():
    return build_metadata(
        tool="dicgen",
        version="1.0.0",
        arguments={"seed": 0},
        sequence={"num_images": 2},
        model={"mode": "affine"},
        speckle={"coverage": 0.5},
        dots={"num_dots": 10},
        coverage={"num_dots": 10},
        noise={"additive_sigma_grey_levels": 0.0},
        sampling={"method": "analytic"},
        ground_truth={"written": True},
        files=[],
    )


def test_build_metadata_required_fields():
    payload = _metadata_payload()
    for key in (
        "tool", "version", "metadata_format_version", "file_naming", "mask_convention",
        "deterministic", "wall_clock_timestamps", "arguments", "sequence",
        "deformation", "speckle", "dots", "coverage", "noise", "sampling",
        "ground_truth", "files", "notes",
    ):
        assert key in payload
    assert payload["deterministic"] is True
    assert payload["wall_clock_timestamps"] is False
    assert payload["metadata_format_version"] == 1


def test_write_json_is_deterministic_and_sorted(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_json(a, {"b": 1, "a": 2})
    write_json(b, {"a": 2, "b": 1})  # different insertion order, same content
    assert a.read_bytes() == b.read_bytes()
    text = a.read_text()
    assert text.endswith("\n")
    assert text.index('"a"') < text.index('"b"')  # sorted keys


def test_write_json_cannot_emit_nan(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "nan.json", {"bad": float("nan")})