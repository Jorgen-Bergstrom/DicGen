"""Tests for :mod:`dicgen.cli`: argument parsing, exit codes and orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from dicgen.cli import EXIT_OK, EXIT_USAGE, build_parser, config_from_args, main


def _small_argv(output_dir, *extra):
    return [
        "--output-dir", str(output_dir),
        "--width", "96", "--height", "96", "-n", "3",
        "--noise-sigma", "0", "--seed", "0",
        # Movie encoding needs the external `ffmpeg` binary; it is covered by
        # tests/test_movie.py, so the CLI tests use the image-only file set.
        "--no-movie",
        *extra,
    ]


# --------------------------------------------------------------------------- #
# Parser / config mapping
# --------------------------------------------------------------------------- #


def test_parser_defaults():
    args = build_parser().parse_args(["--output-dir", "out"])
    assert args.width == 1024
    assert args.height == 1024
    assert args.num_images == 11
    assert args.strain_x == 0.0
    assert args.noise_sigma == 2.0
    assert args.sampling == "analytic"
    assert args.seed == 0
    assert args.save_ground_truth is True
    assert args.save_masks is True
    assert args.save_movie is True
    assert args.movie_fps == 10.0
    assert args.overwrite is False


def test_config_from_args_maps_deformation_and_outputs(tmp_path):
    args = build_parser().parse_args([
        "--output-dir", str(tmp_path / "out"),
        "--width", "128", "--height", "64", "-n", "5",
        "--strain-x", "0.02", "--strain-y", "-0.01", "--rotation", "2.5",
        "--translation-x", "3.0", "--translation-y", "-1.0",
        "--noise-sigma", "1.5", "--seed", "7",
        "--gt-grid", "8", "6", "--no-ground-truth", "--no-masks",
        "--sampling", "bicubic", "--overwrite", "--dry-run",
        "--no-movie", "--movie-fps", "24",
    ])
    config = config_from_args(args)
    assert config.output_dir == tmp_path / "out"
    assert (config.width, config.height, config.num_images) == (128, 64, 5)
    assert config.strain_x == 0.02 and config.strain_y == -0.01
    assert config.rotation_deg == 2.5
    assert config.translation_x == 3.0 and config.translation_y == -1.0
    assert config.noise_sigma == 1.5
    assert config.seed == 7
    assert (config.gt_grid_nx, config.gt_grid_ny) == (8, 6)
    assert config.save_ground_truth is False
    assert config.save_masks is False
    assert config.save_movie is False
    assert config.movie_fps == 24.0
    assert config.sampling == "bicubic"
    assert config.overwrite is True
    assert config.dry_run is True


def test_parser_rejects_unknown_deformation_mode(tmp_path):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--output-dir", str(tmp_path), "--deformation-mode", "warp"]
        )


def test_parser_requires_output_dir():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


# --------------------------------------------------------------------------- #
# Successful runs
# --------------------------------------------------------------------------- #


def test_main_success_writes_expected_files(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(_small_argv(out)) == EXIT_OK
    expected = {
        "image0001.png", "image0002.png", "image0003.png",
        "masks/mask0001.png", "masks/mask0002.png", "masks/mask0003.png",
        "ground_truth.csv", "ground_truth_grid.csv", "metadata.json",
    }
    written = {
        str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()
    }
    assert written == expected
    capsys.readouterr()  # drain the summary output


def test_main_no_ground_truth_omits_csvs(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(_small_argv(out, "--no-ground-truth")) == EXIT_OK
    assert not (out / "ground_truth.csv").exists()
    assert not (out / "ground_truth_grid.csv").exists()
    assert (out / "metadata.json").exists()
    capsys.readouterr()


def test_main_without_masks(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(_small_argv(out, "--no-masks")) == EXIT_OK
    assert not (out / "masks").exists()
    capsys.readouterr()


def test_main_dry_run_writes_nothing(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(_small_argv(out, "--dry-run")) == EXIT_OK
    assert not out.exists()
    capsys.readouterr()


def test_main_prints_summary(tmp_path, capsys):
    out = tmp_path / "out"
    main(_small_argv(out))
    captured = capsys.readouterr()
    assert "output directory" in captured.out
    assert str(out) in captured.out


def test_quiet_mode_still_reports_deletions(tmp_path, capsys):
    """--quiet suppresses progress chatter but never the deletion report."""
    out = tmp_path / "out"
    assert main(_small_argv(out) + ["-q"]) == EXIT_OK
    capsys.readouterr()
    assert main(_small_argv(out) + ["-q"]) == EXIT_OK
    report = capsys.readouterr().out
    assert "Deleting" in report
    assert "image0001.png" in report


def test_main_is_deterministic_for_same_arguments(tmp_path, capsys):
    from conftest import hash_tree

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    assert main(_small_argv(out_a)) == EXIT_OK
    assert main(_small_argv(out_b)) == EXIT_OK
    hashes_a = hash_tree(out_a)
    hashes_b = hash_tree(out_b)
    # Everything except metadata.json (which records the output path) matches.
    assert {k: v for k, v in hashes_a.items() if k != "metadata.json"} == {
        k: v for k, v in hashes_b.items() if k != "metadata.json"
    }
    capsys.readouterr()


# --------------------------------------------------------------------------- #
# Validation: documented EXIT_USAGE (2) for bad parameters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("overrides", [
    dict(num_images=1),
    dict(dot_radius_min=-1.0),
    dict(dot_coverage=1.0),
    dict(noise_sigma=-1.0),
])
def test_generation_config_validate_raises_value_error(tmp_path, overrides):
    """The config-level validators raise ValueError (the CLI turns it into 2)."""
    from dicgen.dataset import GenerationConfig

    config = GenerationConfig(output_dir=tmp_path / "out", width=96, height=96,
                              **overrides)
    with pytest.raises(ValueError):
        config.validate()


def test_generate_dataset_validates_before_creating_anything(tmp_path):
    from dicgen.dataset import GenerationConfig, generate_dataset

    out = tmp_path / "out"
    config = GenerationConfig(output_dir=out, width=96, height=96, num_images=1)
    with pytest.raises(ValueError):
        generate_dataset(config)
    assert not out.exists()  # nothing is created for an invalid configuration


def test_main_returns_usage_error_for_foreign_content(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keepme.txt").write_text("precious")
    assert main(_small_argv(out)) == EXIT_USAGE
    assert (out / "keepme.txt").exists()
    assert "error:" in capsys.readouterr().err


def test_main_overwrite_removes_foreign_content(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keepme.txt").write_text("obsolete")
    assert main(_small_argv(out, "--overwrite")) == EXIT_OK
    assert not (out / "keepme.txt").exists()
    assert (out / "metadata.json").is_file()
    capsys.readouterr()


@pytest.mark.parametrize(
    "extra",
    [
        ["-n", "1"],                       # n must be >= 2
        ["--dot-radius-min", "-1.0"],      # negative radius
        ["--dot-coverage", "1.0"],         # coverage must be < 0.95
        ["--noise-sigma", "-1.0"],         # negative noise sigma
        ["--width", "4"],                  # image too small
        ["--dot-radius-max", "100"],       # radius too large for the image
        ["--sampling", "analytic", "--interpolation-order", "9"],
    ],
)
def test_main_returns_usage_error_for_bad_parameters(tmp_path, capsys, extra):
    out = tmp_path / "out"
    assert main(_small_argv(out, *extra)) == EXIT_USAGE
    assert not out.exists()  # argument validation happens before any writing
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert captured.out == ""