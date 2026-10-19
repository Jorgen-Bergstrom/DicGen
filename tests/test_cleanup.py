"""Tests for :mod:`dicgen.cleanup`: output-directory preparation and safety."""

from __future__ import annotations

from pathlib import Path

import pytest

from dicgen.cleanup import (
    ForeignContentError,
    prepare_output_dir,
    scan_output_dir,
)


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #


def test_scan_missing_directory_is_a_no_op(tmp_path):
    plan = scan_output_dir(tmp_path / "does-not-exist")
    assert plan.existed is False
    assert plan.to_delete == []
    assert plan.foreign == []


def test_scan_classifies_generated_content(tmp_path):
    for name in ("image0001.png", "image9999.png", "ground_truth.csv",
                 "ground_truth_grid.csv", "metadata.json", "displacement_field.npy",
                 "sequence.mp4"):
        _write(tmp_path / name)
    _write(tmp_path / "masks" / "mask0001.png")
    _write(tmp_path / "qa" / "disp_0001.png")
    plan = scan_output_dir(tmp_path)
    assert plan.existed is True
    assert sorted(p.name for p in plan.to_delete) == sorted(
        ["image0001.png", "image9999.png", "ground_truth.csv",
         "ground_truth_grid.csv", "metadata.json", "displacement_field.npy",
         "sequence.mp4", "masks", "qa"]
    )
    assert plan.foreign == []


def test_scan_flags_foreign_file_and_raises(tmp_path):
    stale = _write(tmp_path / "image9999.png")
    foreign = _write(tmp_path / "notes.txt", "please keep me")
    with pytest.raises(ForeignContentError) as excinfo:
        scan_output_dir(tmp_path)
    assert "notes.txt" in str(excinfo.value)
    # The scan itself must not delete anything.
    assert stale.exists() and foreign.exists()


def test_scan_unknown_subdirectory_is_foreign(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(ForeignContentError):
        scan_output_dir(tmp_path)


def test_scan_generated_subdir_with_unknown_file_is_foreign(tmp_path):
    _write(tmp_path / "masks" / "mask0001.png")
    _write(tmp_path / "masks" / "readme.txt")
    with pytest.raises(ForeignContentError):
        scan_output_dir(tmp_path)


def test_scan_symlink_is_foreign(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    target = _write(tmp_path / "target.txt", "outside data")
    (output / "link.txt").symlink_to(target)
    with pytest.raises(ForeignContentError):
        scan_output_dir(output)


def test_scan_raises_if_output_dir_is_a_file(tmp_path):
    path = _write(tmp_path / "not-a-dir")
    with pytest.raises(ForeignContentError):
        scan_output_dir(path)


def test_scan_overwrite_marks_everything_for_deletion(tmp_path):
    _write(tmp_path / "image0001.png")
    _write(tmp_path / "foreign.bin")
    plan = scan_output_dir(tmp_path, overwrite=True)
    assert plan.refused is False
    assert sorted(p.name for p in plan.to_delete) == ["foreign.bin", "image0001.png"]


# --------------------------------------------------------------------------- #
# Preparing (creating / cleaning) the directory
# --------------------------------------------------------------------------- #


def test_prepare_creates_missing_directory_with_parents(tmp_path):
    target = tmp_path / "a" / "b" / "out"
    plan = prepare_output_dir(target)
    assert target.is_dir()
    assert plan.existed is False  # it was created, not pre-existing
    assert plan.deleted == []


def test_prepare_deletes_stale_generated_files_but_keeps_directory(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    _write(output / "image9999.png")
    _write(output / "ground_truth.csv")
    plan = prepare_output_dir(output)
    assert output.is_dir()  # the directory itself is never removed
    assert not (output / "image9999.png").exists()
    assert not (output / "ground_truth.csv").exists()
    assert sorted(plan.relative_deletions) == ["ground_truth.csv", "image9999.png"]


def test_prepare_refuses_foreign_content_and_leaves_it_untouched(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    foreign = _write(output / "important.dat", "do not delete")
    stale = _write(output / "image9999.png")
    with pytest.raises(ForeignContentError):
        prepare_output_dir(output)
    assert foreign.exists()
    assert stale.exists()  # nothing is deleted when the run is refused
    assert output.is_dir()


def test_prepare_overwrite_removes_foreign_content(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    _write(output / "important.dat")
    _write(output / "image0001.png")
    plan = prepare_output_dir(output, overwrite=True)
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert plan.deleted  # something was actually deleted
    assert sorted(plan.relative_deletions) == ["image0001.png", "important.dat"]


def test_prepare_overwrite_unlinks_symlink_without_following_it(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    target = _write(tmp_path / "target.txt", "outside data")
    (output / "link.txt").symlink_to(target)
    prepare_output_dir(output, overwrite=True)
    assert not (output / "link.txt").exists()  # symlink removed
    assert target.exists()  # target untouched


def test_prepare_dry_run_changes_nothing(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    stale = _write(output / "image0001.png")
    plan = prepare_output_dir(output, dry_run=True)
    assert stale.exists()
    assert output.is_dir()
    assert plan.relative_deletions == ["image0001.png"]
    assert plan.deleted == []


def test_prepare_dry_run_does_not_create_missing_directory(tmp_path):
    target = tmp_path / "never-created"
    prepare_output_dir(target, dry_run=True)
    assert not target.exists()


def test_prepare_announces_exactly_what_will_be_deleted(tmp_path, capsys):
    """The specification requires the tool to print exactly what will be deleted."""
    output = tmp_path / "out"
    output.mkdir()
    _write(output / "image0003.png")
    _write(output / "ground_truth.csv")
    prepare_output_dir(output)
    captured = capsys.readouterr().out
    assert "Deleting 2 previously generated item(s)" in captured
    assert "image0003.png" in captured
    assert "ground_truth.csv" in captured


def test_prepare_announce_callback_can_be_customised(tmp_path):
    """Library users can redirect the user-facing messages."""
    output = tmp_path / "out"
    output.mkdir()
    _write(output / "image0001.png")
    messages = []
    prepare_output_dir(output, announce=messages.append)
    assert any("image0001.png" in message for message in messages)
    # ... and the callback can be silenced entirely.
    _write(output / "image0002.png")
    prepare_output_dir(output, announce=lambda message: None)


# --------------------------------------------------------------------------- #
# Safety guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dangerous", [Path("/"), Path.home()])
def test_prepare_refuses_dangerous_output_dirs(dangerous):
    with pytest.raises(ForeignContentError):
        # dry_run keeps the guard test harmless even if the guard ever changed.
        prepare_output_dir(dangerous, dry_run=True)


def test_prepare_refuses_current_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ForeignContentError):
        prepare_output_dir(tmp_path, dry_run=True)