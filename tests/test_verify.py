"""Tests for :mod:`dicgen.verify` (the dataset verifier).

The verifier is the "does the dataset really contain the ground truth?" tool, so
these tests check that it

* passes on correct datasets, including at large strain (40 % engineering
  stretch) where the deformation is far outside the small-strain regime;
* recovers the analytic displacement/gradient from the pixels at the seed points
  (the recovered values are compared with ``model.displacement``/``jacobian``);
* FAILS on tampered datasets (an image carrying a different deformation, or a
  copy of the reference image);
* reports the documented CLI exit codes and can write a JSON report.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import make_config, read_gray, read_json
from dicgen.dataset import generate_dataset
from dicgen.verify import build_parser, format_report, load_dataset_model, main, verify_dataset


def _generate(tmp_path, name, **overrides):
    out = tmp_path / name
    generate_dataset(make_config(out, **overrides))
    return out


# --------------------------------------------------------------------------- #
# Correct datasets pass
# --------------------------------------------------------------------------- #


def test_verify_passes_on_a_small_strain_dataset(tmp_path):
    out = _generate(
        tmp_path, "good", width=256, height=256, num_images=3, strain_x=0.02, rotation_deg=0.5
    )
    report = verify_dataset(out)
    assert report.passed, report.failures
    assert report.num_images == 3
    # The reference frame has no DIC seeds; every deformed frame does.
    assert report.images[0].seeds == []
    for check in report.images[1:]:
        assert check.seeds, "no seed point was checked"
        assert not check.identical_to_reference
        assert check.mae_raw > 1.0, "the frames should not look identical"
        assert check.mae_aligned < check.mae_aligned_limit, "model-aligned error should collapse"
        # undoing the model must remove most of the difference
        assert check.mae_aligned < 0.4 * check.mae_raw
        assert check.u_error_max_px < 0.1
        assert check.strain_error_max < 0.01
        assert check.zncc_median > 0.99
    # the last image carries the full deformation, so the collapse is dramatic
    last = report.images[-1]
    assert last.mae_aligned < 0.25 * last.mae_raw


@pytest.mark.parametrize("strain", [0.05, 0.4])
def test_verify_recovers_displacement_and_strain_at_large_and_small_strain(tmp_path, strain):
    """40 % stretch is far beyond small strain: the first-order DIC must still work."""
    out = _generate(tmp_path, "big", width=256, height=256, num_images=3, strain_x=strain)
    report = verify_dataset(out)
    assert report.passed, report.failures
    check = report.images[-1]
    assert check.t == pytest.approx(1.0)
    assert check.u_error_max_px < 0.1
    # undoing the deformation explains the frames: the aligned difference is a
    # tiny fraction of the raw difference (sub-percent for the 40 % case)
    assert check.mae_aligned < 0.05 * check.mae_raw
    # the measured values really are the analytic ones
    for seed in check.seeds:
        assert seed.u_measured == pytest.approx(seed.u_gt, abs=0.1)
        # pure x-stretch: du/dx = strain, everything else zero
        assert seed.grad_measured[0] == pytest.approx(strain, abs=0.01)
        assert seed.grad_measured[1] == pytest.approx(0.0, abs=0.01)
        assert seed.grad_measured[3] == pytest.approx(0.0, abs=0.01)
    # max |u| at the image border is large for the 40 % case
    assert check.mae_raw > 20.0


def test_verify_handles_the_sinusoid_mode(tmp_path):
    out = _generate(
        tmp_path,
        "sin",
        width=256,
        height=256,
        num_images=3,
        strain_x=0.01,
        mode="sinusoid",
        sin_amplitude_x=2.0,
        sin_amplitude_y=1.0,
    )
    report = verify_dataset(out)
    assert report.passed, report.failures
    assert report.deformation["mode"] == "sinusoid"


def test_verify_works_with_noise_and_scales_the_mae_tolerance(tmp_path):
    out = _generate(
        tmp_path, "noisy", width=256, height=256, num_images=3, strain_x=0.02, noise_sigma=3.0
    )
    report = verify_dataset(out)
    assert report.passed, report.failures
    # the default model-aligned tolerance must account for the noise level
    assert report.thresholds["aligned_error_noise_allowance"] == pytest.approx(9.0)
    for check in report.images[1:]:
        assert check.mae_aligned < 3.0 * 3.0


# --------------------------------------------------------------------------- #
# Tampered datasets fail
# --------------------------------------------------------------------------- #


def test_verify_detects_an_image_with_the_wrong_deformation(tmp_path):
    good = _generate(tmp_path, "good", width=256, height=256, num_images=3, strain_x=0.02)
    other = _generate(tmp_path, "other", width=256, height=256, num_images=3, strain_x=0.3)
    shutil.copyfile(other / "image0003.png", good / "image0003.png")
    report = verify_dataset(good)
    assert not report.passed
    assert any("image0003.png" in failure for failure in report.failures)
    assert report.images[-1].mae_aligned > report.images[-1].mae_aligned_limit


def test_verify_detects_a_frame_that_was_not_deformed(tmp_path):
    out = _generate(tmp_path, "stale", width=256, height=256, num_images=3, strain_x=0.02)
    shutil.copyfile(out / "image0001.png", out / "image0003.png")
    report = verify_dataset(out)
    assert not report.passed
    assert report.images[-1].identical_to_reference
    assert any("identical to the reference" in failure for failure in report.failures)


def test_verify_detects_a_reference_frame_that_is_not_undeformed(tmp_path):
    """If image 0 is replaced by a deformed frame, the model-aligned MAE blows up."""
    out = _generate(tmp_path, "swapped", width=256, height=256, num_images=3, strain_x=0.02)
    shutil.copyfile(out / "image0003.png", out / "image0001.png")
    report = verify_dataset(out)
    assert not report.passed


# --------------------------------------------------------------------------- #
# Model reconstruction, report formatting, CLI
# --------------------------------------------------------------------------- #


def test_load_dataset_model_rebuilds_the_generator_parameters(tmp_path):
    out = _generate(
        tmp_path,
        "meta",
        width=192,
        height=160,
        num_images=3,
        strain_x=0.03,
        strain_y=-0.02,
        rotation_deg=0.25,
        translation_x=1.5,
        seed=7,
    )
    metadata, model, filenames = load_dataset_model(out)
    assert (model.width, model.height) == (192, 160)
    assert model.strain_x == pytest.approx(0.03)
    assert model.strain_y == pytest.approx(-0.02)
    assert model.rotation_deg == pytest.approx(0.25)
    assert model.translation[0] == pytest.approx(1.5)
    assert filenames == metadata["sequence"]["filenames"]


def test_load_dataset_model_requires_metadata_json(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_dataset_model(empty)


def test_format_report_mentions_pass_and_the_images(tmp_path):
    out = _generate(tmp_path, "report", width=192, height=192, num_images=3, strain_x=0.02)
    text = format_report(verify_dataset(out))
    assert "image" in text.lower()
    assert "PASSED" in text
    assert "3 images" in text


def test_cli_exit_codes_and_json_report(tmp_path, capsys):
    good = _generate(tmp_path, "good", width=192, height=192, num_images=3, strain_x=0.02)
    assert main(["--dataset", str(good)]) == 0
    capsys.readouterr()

    bad = _generate(tmp_path, "bad", width=192, height=192, num_images=3, strain_x=0.02)
    shutil.copyfile(bad / "image0001.png", bad / "image0002.png")
    assert main(["--dataset", str(bad), "--quiet"]) == 1

    report_path = tmp_path / "report.json"
    assert main(["--dataset", str(good), "--quiet", "--json", str(report_path)]) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["tool"] == "dicgen.verify"
    assert len(payload["images"]) == 3
    assert payload["images"][-1]["seeds"], "the JSON report should carry the DIC measurements"
    assert payload["thresholds"]["max_displacement_error_px"] == pytest.approx(0.1)
    capsys.readouterr()


def test_cli_reports_a_missing_dataset_with_exit_code_2(tmp_path, capsys):
    assert main(["--dataset", str(tmp_path / "nope")]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_parser_defaults():
    args = build_parser().parse_args(["--dataset", "d"])
    assert args.seed_points == 12
    assert args.window == 31
    assert args.max_displacement_error == pytest.approx(0.1)
    assert args.max_strain_error == pytest.approx(0.01)
    assert args.last_only is False


def test_verify_last_only_option(tmp_path):
    out = _generate(tmp_path, "last", width=192, height=192, num_images=4, strain_x=0.02)
    report = verify_dataset(out, check_all_images=False)
    assert len(report.images) == 1
    assert report.images[0].index == 3


def test_verify_flags_a_stale_image_that_kept_the_reference_pattern(tmp_path):
    """A frame generated with the *reference* pattern but no deformation is caught."""
    out = _generate(tmp_path, "noise_only", width=192, height=192, num_images=3, strain_x=0.02)
    # replace image 3 with pure background + noise (no speckle at all)
    rng = np.random.default_rng(0)
    noise = np.rint(128 + rng.normal(0, 2, size=(192, 192))).clip(0, 255).astype(np.uint8)
    from PIL import Image

    Image.fromarray(noise, mode="L").save(out / "image0003.png")
    report = verify_dataset(out)
    assert not report.passed
    check = report.images[-1]
    assert check.mae_raw > 20.0
    assert check.zncc_median < 0.5