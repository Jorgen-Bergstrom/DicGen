"""End-to-end acceptance tests: images, ground truth, determinism, cleanup.

This module checks the behaviours required by the task specification against
datasets produced by the public :func:`dicgen.dataset.generate_dataset` API.  The
image-based displacement checks use the independent measurement helpers from
``conftest`` (FFT cross-correlation + local NCC), not the package's transform
code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import (
    estimate_translation,
    expected_reference,
    hash_tree,
    ncc_local_displacement,
    read_csv_rows,
    read_gray,
    read_json,
    read_rgb,
    sha256_file,
)
from dicgen.cleanup import ForeignContentError
from dicgen.dataset import GenerationConfig, generate_dataset
from dicgen.imagegen import to_uint8


# --------------------------------------------------------------------------- #
# 1. The reference image (i = 0) is undeformed
# --------------------------------------------------------------------------- #


def test_reference_image_is_undeformed(generate):
    out, _ = generate(strain_x=0.03, strain_y=-0.02, rotation_deg=2.0,
                      translation_x=1.5, translation_y=-0.75, num_images=4)
    rows = read_csv_rows(out / "ground_truth.csv")
    first = rows[0]
    assert float(first["t"]) == 0.0
    assert float(first["image_index"]) == 0
    assert first["image"] == "image0001.png"
    # Ground-truth displacement is identically zero.
    assert float(first["u_max_px"]) == pytest.approx(0.0, abs=1e-9)
    assert float(first["u_mean_px"]) == pytest.approx(0.0, abs=1e-9)
    # F is the identity and the strain is zero.
    assert float(first["F11"]) == pytest.approx(1.0)
    assert float(first["F22"]) == pytest.approx(1.0)
    assert float(first["F12"]) == 0.0 and float(first["F21"]) == 0.0
    assert float(first["eps_xx"]) == 0.0 and float(first["eps_yy"]) == 0.0
    assert float(first["valid_fraction"]) == pytest.approx(1.0)


def test_reference_grid_displacement_is_zero(generate):
    out, _ = generate(strain_x=0.02, rotation_deg=1.0, num_images=3)
    rows = [r for r in read_csv_rows(out / "ground_truth_grid.csv")
            if r["image_index"] == "0"]
    assert rows  # the grid has points
    assert all(
        float(r["ux_px"]) == pytest.approx(0.0, abs=1e-9)
        and float(r["uy_px"]) == pytest.approx(0.0, abs=1e-9)
        for r in rows
    )


def test_reference_validity_mask_is_all_valid(generate):
    out, _ = generate(strain_x=0.02, num_images=3)
    mask = read_gray(out / "masks" / "mask0001.png")
    assert np.all(mask == 255)


def test_noise_free_reference_equals_analytic_render(generate):
    """`--noise-sigma 0` image 0 is exactly the analytic reference, rounded."""
    out, summary = generate(
        width=128, height=128, num_images=3, noise_sigma=0.0, seed=11,
        strain_x=0.02, rotation_deg=1.0,
    )
    config = GenerationConfig(
        output_dir=out, width=128, height=128, num_images=3, noise_sigma=0.0,
        seed=11, strain_x=0.02, rotation_deg=1.0,
    )
    expected = to_uint8(expected_reference(config))
    actual = read_gray(out / "image0001.png")
    assert np.array_equal(actual, expected)


# --------------------------------------------------------------------------- #
# 2. Pure translation: sub-pixel recovery and analytic ground truth
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("translation", [(2.37, -1.62), (0.5, 0.5), (-3.1, 2.9)])
def test_pure_translation_is_recovered_subpixel(generate, translation):
    tx, ty = translation
    out, _ = generate(
        width=192, height=160, num_images=2, noise_sigma=0.0,
        translation_x=tx, translation_y=ty, seed=0,
    )
    ref = read_gray(out / "image0001.png")
    moved = read_gray(out / "image0002.png")
    # n = 2 -> t_1 = 1 -> image 2 is displaced by the full translation.
    measured = estimate_translation(ref, moved, margin=24)
    assert measured == pytest.approx([tx, ty], abs=0.1)


def test_translation_ground_truth_row_equals_translation_times_t(generate):
    tx, ty = 2.0, -1.5
    out, _ = generate(width=160, height=128, num_images=5, noise_sigma=0.0,
                      translation_x=tx, translation_y=ty, seed=0)
    rows = read_csv_rows(out / "ground_truth.csv")
    assert float(rows[0]["t"]) == 0.0
    assert float(rows[-1]["t"]) == 1.0
    for row in rows:
        t = float(row["t"])
        assert float(row["tx_px"]) == pytest.approx(tx * t, abs=1e-9)
        assert float(row["ty_px"]) == pytest.approx(ty * t, abs=1e-9)
        # Constant displacement field for a pure translation.
        assert float(row["u_max_px"]) == pytest.approx(np.hypot(tx, ty) * t, abs=1e-9)


def test_partial_translation_between_second_and_third_images(generate):
    """load stepping: image 2 of 3 carries half of the total translation."""
    tx, ty = 2.0, -1.5
    out, _ = generate(width=192, height=160, num_images=3, noise_sigma=0.0,
                      translation_x=tx, translation_y=ty, seed=0)
    ref = read_gray(out / "image0001.png")
    second = read_gray(out / "image0002.png")
    measured = estimate_translation(ref, second, margin=24)
    assert measured == pytest.approx([0.5 * tx, 0.5 * ty], abs=0.1)


# --------------------------------------------------------------------------- #
# 3. A small affine deformation is recovered by local NCC
# --------------------------------------------------------------------------- #


def test_small_affine_deformation_recovered_by_local_ncc(generate):
    params = dict(strain_x=0.008, strain_y=-0.004, rotation_deg=0.5,
                  translation_x=0.7, translation_y=-0.3)
    out, _ = generate(width=256, height=256, num_images=2, noise_sigma=0.0,
                      seed=0, **params)
    ref = read_gray(out / "image0001.png")
    deformed = read_gray(out / "image0002.png")

    config = GenerationConfig(output_dir=out, width=256, height=256, num_images=2,
                              noise_sigma=0.0, seed=0, **params)
    model = config.deformation_model()

    points = [(64, 64), (128, 128), (192, 160), (96, 176), (176, 96)]
    for point in points:
        p = np.array(point, dtype=float)
        analytic = model.displacement(p, 1.0)
        measured = ncc_local_displacement(ref, deformed, p, guess=analytic, half=25)
        residual = measured - analytic
        assert np.all(np.abs(residual) < 0.1), (point, measured, analytic)


# --------------------------------------------------------------------------- #
# 4. Determinism and metadata stability
# --------------------------------------------------------------------------- #


def test_same_seed_is_byte_identical_except_metadata(generate):
    kwargs = dict(strain_x=-0.04, rotation_deg=1.5, noise_sigma=0.0, seed=3, num_images=4)
    a, _ = generate("a", **kwargs)
    b, _ = generate("b", **kwargs)
    hashes_a = hash_tree(a)
    hashes_b = hash_tree(b)
    assert set(hashes_a) == set(hashes_b)
    differing = [name for name in hashes_a if hashes_a[name] != hashes_b[name]]
    assert differing == ["metadata.json"]
    # ... and metadata agrees everywhere except the recorded output path.
    meta_a = read_json(a / "metadata.json")
    meta_b = read_json(b / "metadata.json")
    meta_a["arguments"].pop("output_dir")
    meta_b["arguments"].pop("output_dir")
    assert meta_a == meta_b


# Documented, intentional exception to strict cross-directory byte-identity:
# requirement #4 of the task asks for two runs "into two different directories"
# to be reproducible.  Every image, mask and CSV is byte-identical, but
# metadata.json records the absolute output path in `arguments.output_dir`
# (provenance), so the two files differ in exactly that one string.  Re-running
# into the *same* directory is byte-identical everywhere (test below).
def test_cross_directory_runs_differ_only_in_the_recorded_output_path(generate):
    kwargs = dict(strain_x=-0.04, noise_sigma=0.0, seed=3, num_images=4)
    a, _ = generate("a", **kwargs)
    b, _ = generate("b", **kwargs)
    hashes_a, hashes_b = hash_tree(a), hash_tree(b)
    differing = {name for name in hashes_a if hashes_a[name] != hashes_b[name]}
    assert differing <= {"metadata.json"}, differing
    meta_a = read_json(a / "metadata.json")
    meta_b = read_json(b / "metadata.json")
    assert meta_a["arguments"].pop("output_dir") == str(a)
    assert meta_b["arguments"].pop("output_dir") == str(b)
    assert meta_a == meta_b


def test_rerun_into_same_directory_is_fully_byte_identical(tmp_path):
    """Re-running with identical arguments into the same directory changes nothing."""
    from conftest import make_config

    out = tmp_path / "out"
    generate_dataset(make_config(out, strain_x=-0.03, noise_sigma=0.0, seed=2,
                                 num_images=4))
    first = hash_tree(out)
    generate_dataset(make_config(out, strain_x=-0.03, noise_sigma=0.0, seed=2,
                                 num_images=4))
    assert hash_tree(out) == first


def test_different_seed_changes_the_images_and_metadata(generate):
    common = dict(strain_x=0.02, noise_sigma=0.0, num_images=3)
    a, _ = generate("a", seed=0, **common)
    b, _ = generate("b", seed=1, **common)
    assert not np.array_equal(read_gray(a / "image0001.png"), read_gray(b / "image0001.png"))
    assert read_json(a / "metadata.json")["dots"] != read_json(b / "metadata.json")["dots"]


def test_metadata_schema_and_file_manifest(generate):
    out, summary = generate(width=128, height=128, num_images=3, noise_sigma=0.0, seed=0)
    meta = read_json(out / "metadata.json")
    required = {
        "tool", "version", "metadata_format_version", "file_naming",
        "mask_convention", "deterministic", "wall_clock_timestamps", "arguments",
        "sequence", "deformation", "speckle", "dots", "coverage", "noise",
        "sampling", "ground_truth", "movie", "files", "notes",
        "grey_level_statistics", "summary",
    }
    assert required <= set(meta)
    assert meta["tool"] == "dicgen"
    assert meta["deterministic"] is True
    assert meta["metadata_format_version"] == 1

    # sequence / naming
    assert meta["sequence"]["filenames"][0] == "image0001.png"
    assert meta["sequence"]["reference_image"] == "image0001.png"
    assert meta["arguments"]["width"] == 128
    assert meta["arguments"]["output_dir"] == str(out)

    # the manifest lists every generated file except metadata.json itself, with
    # a correct digest, size and relative path
    manifest = {entry["path"]: entry for entry in meta["files"]}
    on_disk = {
        str(p.relative_to(out)) for p in out.rglob("*")
        if p.is_file() and str(p.relative_to(out)) != "metadata.json"
    }
    assert set(manifest) == on_disk
    for rel, entry in manifest.items():
        assert entry["sha256"] == sha256_file(out / rel)
        assert entry["size_bytes"] == (out / rel).stat().st_size
    # summary block is present and consistent
    assert meta["summary"]["num_dots"] == summary.num_dots
    assert "coverage_achieved" in meta["summary"]


# --------------------------------------------------------------------------- #
# 5. Output-directory cleanup
# --------------------------------------------------------------------------- #


def test_stale_and_foreign_content_aborts_run_and_preserves_foreign(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    stale = out / "image9999.png"
    stale.write_bytes(b"stale")
    foreign = out / "important.dat"
    foreign.write_text("do not delete")

    config = GenerationConfig(output_dir=out, width=96, height=96, num_images=2,
                              noise_sigma=0.0)
    with pytest.raises(ForeignContentError):
        generate_dataset(config)
    assert foreign.read_text() == "do not delete"
    assert stale.read_bytes() == b"stale"  # nothing deleted on refusal
    assert out.is_dir()


def test_overwrite_removes_foreign_content_and_writes_only_expected_set(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "image9999.png").write_bytes(b"stale")
    (out / "important.dat").write_text("obsolete")
    config = GenerationConfig(output_dir=out, width=96, height=96, num_images=2,
                              noise_sigma=0.0, overwrite=True, save_movie=False)
    generate_dataset(config)
    assert not (out / "image9999.png").exists()
    assert not (out / "important.dat").exists()
    assert out.is_dir()  # the directory itself survives
    on_disk = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert on_disk == {
        "image0001.png", "image0002.png",
        "masks/mask0001.png", "masks/mask0002.png",
        "ground_truth.csv", "ground_truth_grid.csv", "metadata.json",
    }


def test_missing_output_directory_is_created_including_parents(tmp_path):
    out = tmp_path / "fresh" / "deeper" / "out"
    config = GenerationConfig(output_dir=out, width=96, height=96, num_images=2,
                              noise_sigma=0.0, save_movie=False)
    generate_dataset(config)
    assert (out / "image0001.png").is_file()
    assert (out / "metadata.json").is_file()


def test_dry_run_generate_writes_nothing(tmp_path):
    out = tmp_path / "out"
    config = GenerationConfig(output_dir=out, width=96, height=96, num_images=2,
                              noise_sigma=0.0, dry_run=True)
    summary = generate_dataset(config)
    assert summary.wrote_files is False
    assert not out.exists()


# --------------------------------------------------------------------------- #
# 6. Additional quantitative checks
# --------------------------------------------------------------------------- #


def test_noise_sigma_is_measured_close_to_request(tmp_path):
    common = dict(width=128, height=128, num_images=3, seed=5, strain_x=0.01)
    clean = GenerationConfig(output_dir=tmp_path / "clean", noise_sigma=0.0,
                             save_movie=False, **common)
    noisy = GenerationConfig(output_dir=tmp_path / "noisy", noise_sigma=12.0,
                             save_movie=False, **common)
    generate_dataset(clean)
    generate_dataset(noisy)
    a = read_gray(tmp_path / "clean" / "image0001.png").astype(float)
    b = read_gray(tmp_path / "noisy" / "image0001.png").astype(float)
    assert not np.array_equal(a, b)
    # Restrict to mid-grey pixels so deliberate clipping at 0/255 does not bias
    # the standard deviation.
    mid = (a > 30) & (a < 225)
    measured = np.std((b - a)[mid])
    assert measured == pytest.approx(12.0, rel=0.15)


def test_reported_coverage_matches_measured_coverage(generate):
    """The coverage reported in metadata is what the reference image really shows."""
    from dicgen.speckle import measure_coverage

    out, summary = generate(width=160, height=160, num_images=2, dot_coverage=0.5,
                            coverage_tolerance=0.005, noise_sigma=0.0, seed=0)
    reference = read_gray(out / "image0001.png").astype(float)
    measured = measure_coverage(reference, background=128.0)["coverage_any"]
    # The reported value is measured on the float render; the PNG is 8-bit, so a
    # small tolerance absorbs grey-level rounding near the 0.5-level threshold.
    assert measured == pytest.approx(summary.coverage_achieved, abs=0.005)


def test_dot_coverage_is_close_to_target(generate):
    out, summary = generate(width=192, height=192, num_images=2, dot_coverage=0.4,
                            coverage_tolerance=0.005, noise_sigma=0.0, seed=0)
    assert summary.coverage_achieved == pytest.approx(0.4, abs=0.03)
    meta = read_json(out / "metadata.json")
    assert meta["coverage"]["coverage_target"] == pytest.approx(0.4)
    assert meta["coverage"]["coverage_any"] == pytest.approx(summary.coverage_achieved)


def test_validity_mask_and_fraction_decrease_with_compression(generate):
    out, summary = generate(width=160, height=160, num_images=5, noise_sigma=0.0,
                            strain_x=-0.06, seed=0)
    assert np.all(read_gray(out / "masks" / "mask0001.png") == 255)
    fractions = summary.valid_fractions
    assert fractions[0] == pytest.approx(1.0)
    # Monotonically non-increasing, and strictly smaller at full load.
    assert all(b <= a + 1e-12 for a, b in zip(fractions, fractions[1:]))
    assert fractions[-1] < 1.0
    last_mask = read_gray(out / "masks" / f"mask{len(fractions):04d}.png")
    assert np.any(last_mask == 0)
    assert np.all((last_mask == 0) | (last_mask == 255))


def test_grid_csv_matches_analytic_formulas(generate):
    """Recompute u(X), the Jacobian and the strain from the documented formulas."""
    params = dict(strain_x=0.02, strain_y=-0.01, rotation_deg=1.5,
                  translation_x=1.0, translation_y=-0.5)
    out, _ = generate(width=128, height=128, num_images=4, noise_sigma=0.0,
                      seed=0, **params)
    config = GenerationConfig(output_dir=out, width=128, height=128, num_images=4,
                              noise_sigma=0.0, seed=0, **params)
    model = config.deformation_model()
    rows = read_csv_rows(out / "ground_truth_grid.csv")
    assert rows
    for row in rows:
        i = int(row["image_index"])
        t = i / (4 - 1)
        assert float(row["t"]) == pytest.approx(t, abs=1e-12)
        X = np.array([float(row["X_px"]), float(row["Y_px"])])
        expected_u = model.displacement(X, t)
        assert float(row["ux_px"]) == pytest.approx(expected_u[0], abs=1e-9)
        assert float(row["uy_px"]) == pytest.approx(expected_u[1], abs=1e-9)
        # Constant Jacobian = F_i for the affine model.
        F = model.affine_matrix(t)
        assert float(row["J11"]) == pytest.approx(F[0, 0], abs=1e-9)
        assert float(row["J12"]) == pytest.approx(F[0, 1], abs=1e-9)
        assert float(row["J21"]) == pytest.approx(F[1, 0], abs=1e-9)
        assert float(row["J22"]) == pytest.approx(F[1, 1], abs=1e-9)
        # Documented uniform engineering strains.
        assert float(row["eps_xx"]) == pytest.approx(params["strain_x"] * t, abs=1e-12)
        assert float(row["eps_yy"]) == pytest.approx(params["strain_y"] * t, abs=1e-12)
        assert float(row["gamma_xy"]) == pytest.approx(0.0, abs=1e-12)


def test_file_naming_and_png_format(generate):
    out, summary = generate(width=96, height=96, num_images=3, noise_sigma=0.0, seed=0)
    from conftest import png_mode

    assert summary.filenames == ["image0001.png", "image0002.png", "image0003.png"]
    for name in summary.filenames:
        path = out / name
        assert path.is_file()
        assert png_mode(path) == "L"  # 8-bit grayscale
        img = read_gray(path)
        assert img.shape == (96, 96)
    for i in range(1, 4):
        mask = out / "masks" / f"mask{i:04d}.png"
        assert mask.is_file()
        assert png_mode(mask) == "L"


def test_sinusoid_mode_runs_and_ground_truth_is_nonuniform(generate):
    out, _ = generate(width=128, height=128, num_images=3, noise_sigma=0.0, seed=0,
                      mode="sinusoid", sin_amplitude_x=5.0, sin_amplitude_y=3.0,
                      strain_x=0.01)
    meta = read_json(out / "metadata.json")
    assert meta["deformation"]["mode"] == "sinusoid"
    rows = [r for r in read_csv_rows(out / "ground_truth_grid.csv")
            if r["image_index"] == "2"]
    eps = np.array([float(r["eps_xx"]) for r in rows])
    assert np.ptp(eps) > 1e-3  # the sinusoidal strain field varies in space


def test_optional_qa_maps_and_displacement_npy(generate):
    out, _ = generate(width=96, height=96, num_images=3, noise_sigma=0.0, seed=0,
                      strain_x=0.02, save_qa_maps=True, save_displacement_npy=True)
    assert (out / "qa" / "disp_0001.png").is_file()
    assert (out / "qa" / "arrows_0001.png").is_file()
    assert read_rgb(out / "qa" / "disp_0001.png").shape == (96, 96, 3)
    field = np.load(out / "displacement_field.npy")
    assert field.shape == (3, 2, 96, 96)
    assert field.dtype == np.float32
    # Displacement of the reference image is exactly zero.
    assert np.all(field[0] == 0.0)