"""Tests for :mod:`dicgen.movie`: MP4 generation from the image sequence.

Movie encoding shells out to the external ``ffmpeg`` binary, so the tests that
actually encode are skipped when it is not installed; the pure-Python error paths
and the metadata/cleanup integration still run everywhere.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import make_config, read_json, sha256_file
from dicgen.cleanup import scan_output_dir
from dicgen.dataset import GenerationConfig, generate_dataset
from dicgen.movie import (
    MOVIE_FILENAME,
    MovieError,
    ffmpeg_available,
    write_movie,
)

requires_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(), reason="ffmpeg is not installed"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_frames(tmp_path: Path, *, count: int, width: int, height: int):
    """Write ``count`` deterministic grayscale PNG frames; return their paths."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        rng = np.random.default_rng(100 + i)
        data = (rng.random((height, width)) * 200.0 + 20.0).astype(np.uint8)
        path = tmp_path / f"image{i + 1:04d}.png"
        Image.fromarray(data, mode="L").save(path, format="PNG")
        paths.append(path)
    return paths


def _decode_gray(movie_path: Path, width: int, height: int) -> np.ndarray:
    """Decode an MP4 back to a ``(n, height, width)`` uint8 array with ffmpeg."""
    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-v", "error",
        "-i", str(movie_path),
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    frame_size = width * height
    frames = len(result.stdout) // frame_size
    return np.frombuffer(
        result.stdout[: frames * frame_size], dtype=np.uint8
    ).reshape(frames, height, width)


# --------------------------------------------------------------------------- #
# write_movie: error paths (no ffmpeg required)
# --------------------------------------------------------------------------- #


def test_write_movie_requires_at_least_one_image(tmp_path):
    with pytest.raises(MovieError):
        write_movie([], tmp_path / "empty.mp4")


def test_write_movie_reports_missing_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr("dicgen.movie.ffmpeg_available", lambda: False)
    paths = _write_frames(tmp_path, count=2, width=32, height=32)
    with pytest.raises(MovieError, match="ffmpeg"):
        write_movie(paths, tmp_path / "movie.mp4")


def test_write_movie_rejects_inconsistent_frame_sizes(tmp_path):
    if not ffmpeg_available():
        pytest.skip("ffmpeg is not installed")
    a = _write_frames(tmp_path / "a", count=1, width=32, height=32)
    b = _write_frames(tmp_path / "b", count=1, width=48, height=32)
    with pytest.raises(MovieError):
        write_movie(a + b, tmp_path / "movie.mp4")


# --------------------------------------------------------------------------- #
# write_movie: real encoding
# --------------------------------------------------------------------------- #


@requires_ffmpeg
def test_write_movie_produces_a_playable_mp4(tmp_path):
    paths = _write_frames(tmp_path, count=4, width=64, height=48)
    info = write_movie(paths, tmp_path / "movie.mp4", fps=8.0)

    movie = tmp_path / "movie.mp4"
    assert movie.is_file()
    assert movie.stat().st_size > 0
    assert movie.read_bytes()[4:8] == b"ftyp"  # ISO base media file signature
    assert info["frames"] == 4
    assert info["fps"] == 8.0
    assert (info["width"], info["height"]) == (64, 48)
    assert info["codec"] == "h264"

    decoded = _decode_gray(movie, 64, 48)
    assert decoded.shape[0] == 4
    # H.264 is lossy, but at crf 18 the frames must match closely.
    for original, restored in zip(paths, decoded):
        source = np.asarray(Image.open(original), dtype=np.float64)
        assert np.mean(np.abs(source - restored)) < 8.0


@requires_ffmpeg
def test_write_movie_pads_odd_dimensions(tmp_path):
    paths = _write_frames(tmp_path, count=3, width=33, height=27)
    info = write_movie(paths, tmp_path / "odd.mp4")
    # yuv420p requires even dimensions; the encoder pads by one row/column.
    assert info["width"] == 33 and info["height"] == 27
    assert (tmp_path / "odd.mp4").is_file()


@requires_ffmpeg
def test_write_movie_is_byte_deterministic(tmp_path):
    paths = _write_frames(tmp_path, count=3, width=48, height=48)
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    write_movie(paths, first)
    write_movie(paths, second)
    assert sha256_file(first) == sha256_file(second)


# --------------------------------------------------------------------------- #
# Integration with generate_dataset
# --------------------------------------------------------------------------- #


@requires_ffmpeg
def test_generate_dataset_writes_movie_by_default(generate):
    out, summary = generate(save_movie=True, num_images=4, strain_x=0.02, seed=0)
    movie = out / MOVIE_FILENAME
    assert movie.is_file()
    assert movie.read_bytes()[4:8] == b"ftyp"
    assert summary.movie_written is True
    assert summary.movie_filename == MOVIE_FILENAME
    assert MOVIE_FILENAME in summary.files

    meta = read_json(out / "metadata.json")
    assert meta["movie"]["written"] is True
    assert meta["movie"]["frames"] == 4
    assert meta["movie"]["fps"] == 10.0
    # the movie is part of the SHA-256 manifest
    manifest = {entry["path"]: entry for entry in meta["files"]}
    assert MOVIE_FILENAME in manifest
    assert manifest[MOVIE_FILENAME]["sha256"] == sha256_file(movie)


@requires_ffmpeg
def test_movie_is_deterministic_across_runs_into_different_dirs(tmp_path):
    common = dict(width=96, height=96, num_images=3, noise_sigma=0.0, seed=4,
                  save_movie=True)
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_dataset(GenerationConfig(output_dir=a, **common))
    generate_dataset(GenerationConfig(output_dir=b, **common))
    # Unlike metadata.json (which records the output path), the movie depends
    # only on the images, so the two encodings are byte-identical.
    assert sha256_file(a / MOVIE_FILENAME) == sha256_file(b / MOVIE_FILENAME)


def test_generate_dataset_without_movie_omits_it(generate):
    out, summary = generate(save_movie=False, num_images=3)
    assert not (out / MOVIE_FILENAME).exists()
    assert summary.movie_written is False
    assert summary.movie_filename is None
    meta = read_json(out / "metadata.json")
    assert meta["movie"]["written"] is False
    assert all(entry["path"] != MOVIE_FILENAME for entry in meta["files"])


@requires_ffmpeg
def test_generate_dataset_honours_movie_fps(generate):
    out, summary = generate(save_movie=True, movie_fps=25.0, num_images=3)
    meta = read_json(out / "metadata.json")
    assert meta["movie"]["fps"] == 25.0
    assert summary.movie_fps == 25.0


def test_movie_fps_must_be_positive(tmp_path):
    config = GenerationConfig(output_dir=tmp_path / "out", width=64, height=64,
                              num_images=2, movie_fps=0.0)
    with pytest.raises(ValueError):
        config.validate()


@requires_ffmpeg
def test_stale_movie_is_cleaned_up_as_generated_content(generate):
    """A previous movie is recognised and removed on the next run."""
    out, _ = generate(save_movie=True, num_images=3)
    assert (out / MOVIE_FILENAME).is_file()
    # A scan must classify the movie as generated (not foreign) content.
    plan = scan_output_dir(out)
    assert any(p.name == MOVIE_FILENAME for p in plan.to_delete)
    assert plan.foreign == []


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@requires_ffmpeg
def test_cli_writes_movie_by_default(tmp_path, capsys):
    from dicgen.cli import EXIT_OK, main

    out = tmp_path / "out"
    argv = [
        "--output-dir", str(out),
        "--width", "96", "--height", "96", "-n", "3",
        "--noise-sigma", "0", "--seed", "0",
    ]
    assert main(argv) == EXIT_OK
    assert (out / MOVIE_FILENAME).is_file()
    report = capsys.readouterr().out
    assert MOVIE_FILENAME in report


def test_cli_no_movie_flag(tmp_path, capsys):
    from dicgen.cli import EXIT_OK, main

    out = tmp_path / "out"
    argv = [
        "--output-dir", str(out),
        "--width", "96", "--height", "96", "-n", "3",
        "--noise-sigma", "0", "--seed", "0", "--no-movie",
    ]
    assert main(argv) == EXIT_OK
    assert not (out / MOVIE_FILENAME).exists()
    capsys.readouterr()


def test_cli_rejects_non_positive_movie_fps(tmp_path, capsys):
    from dicgen.cli import EXIT_USAGE, main

    out = tmp_path / "out"
    argv = [
        "--output-dir", str(out),
        "--width", "96", "--height", "96", "-n", "3",
        "--movie-fps", "0",
    ]
    assert main(argv) == EXIT_USAGE
    assert not out.exists()
    capsys.readouterr()
