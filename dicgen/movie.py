"""Movie generation: encode the image sequence into an MP4 file.

The generated frames are piped to the system ``ffmpeg`` binary as raw 8-bit
grayscale video, so no temporary files or shell quoting are involved and the
movie is written next to the images it was made from.

The encoder is configured to be **byte-reproducible**: given the same images and
the same ``ffmpeg`` build, the same arguments produce the same MP4.  The
``-fflags +bitexact`` / ``-flags:v +bitexact`` flags suppress the encoder version
string and the MP4 creation time, which are the usual sources of run-to-run
differences.  (Reproducibility across *different* ``ffmpeg`` versions is not
guaranteed -- the H.264 encoder is not bit-stable between releases.)

``ffmpeg`` is an external, optional dependency: :func:`ffmpeg_available` reports
whether it can be found, and :func:`write_movie` raises :class:`MovieError` when
it is missing or the encode fails.  Callers that want the movie to be
best-effort (the dataset generator does) should catch :class:`MovieError`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

__all__ = [
    "MovieError",
    "MOVIE_FILENAME",
    "DEFAULT_MOVIE_FPS",
    "ffmpeg_available",
    "write_movie",
]

LOGGER = logging.getLogger("dicgen")

#: Name of the movie written next to the image sequence.
MOVIE_FILENAME = "sequence.mp4"

#: Default playback rate of the movie, in frames per second.
DEFAULT_MOVIE_FPS = 10.0

#: Constant-rate-factor quality (lower = better/larger); 18 is visually lossless.
DEFAULT_MOVIE_CRF = 18

#: x264 speed/efficiency preset.
DEFAULT_MOVIE_PRESET = "medium"


class MovieError(RuntimeError):
    """Raised when the MP4 movie could not be produced."""


def ffmpeg_available() -> bool:
    """True if an ``ffmpeg`` executable can be found on ``PATH``."""
    return shutil.which("ffmpeg") is not None


def _even_size_filter(width: int, height: int) -> Optional[str]:
    """Pad filter making the dimensions even, or ``None`` if already even.

    H.264 with the widely playable ``yuv420p`` chroma format requires even width
    and height, so an odd-sized image is padded by one row/column of black.
    """
    if width % 2 == 0 and height % 2 == 0:
        return None
    return "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=black"


def _read_gray_frame(path: Path) -> np.ndarray:
    """Decode a PNG to a 2-D ``uint8`` grayscale array."""
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)
    except OSError as exc:  # unreadable / missing / not an image
        raise MovieError(f"cannot read image {path}: {exc}") from exc


def write_movie(
    image_paths: Sequence[Path],
    output_path: Path,
    *,
    fps: float = DEFAULT_MOVIE_FPS,
    width: Optional[int] = None,
    height: Optional[int] = None,
    crf: int = DEFAULT_MOVIE_CRF,
    preset: str = DEFAULT_MOVIE_PRESET,
) -> Dict[str, Any]:
    """Encode ``image_paths`` (in order) into an H.264 MP4 at ``output_path``.

    ``image_paths`` are read as 8-bit grayscale frames and piped to ``ffmpeg``;
    the frames must all share the same size (taken from the first image unless
    ``width``/``height`` are given).  ``fps`` is the playback rate.

    Returns a small dictionary describing the movie (file name, frame count,
    size, codec, ...).  Raises :class:`MovieError` if ``ffmpeg`` is unavailable,
    an image cannot be read, the frames are not all the same size, or the encode
    fails.
    """
    paths: List[Path] = [Path(p) for p in image_paths]
    if not paths:
        raise MovieError("no images to encode")
    output_path = Path(output_path)
    if not ffmpeg_available():
        raise MovieError(
            "ffmpeg was not found on PATH; install ffmpeg to write the movie "
            "(e.g. 'apt install ffmpeg' or 'brew install ffmpeg'), or pass "
            "--no-movie"
        )

    first = _read_gray_frame(paths[0])
    frame_height, frame_width = first.shape
    if width is None:
        width = frame_width
    if height is None:
        height = frame_height
    if (frame_width, frame_height) != (width, height):
        raise MovieError(
            f"image {paths[0]} is {frame_width}x{frame_height} px, expected "
            f"{width}x{height} px"
        )

    command: List[str] = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        # raw 8-bit grayscale frames arriving on stdin
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{float(fps):g}",
        "-i",
        "pipe:0",
    ]
    even_filter = _even_size_filter(width, height)
    if even_filter is not None:
        command += ["-vf", even_filter]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        # byte-reproducible: no encoder version string, no creation time
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-map_metadata",
        "-1",
        str(output_path),
    ]

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:  # pragma: no cover - ffmpeg_available() already checked
        raise MovieError(f"could not start ffmpeg: {exc}") from exc

    # Drain stderr in a background thread so a chatty encoder can never deadlock
    # the frame writer on a full pipe buffer.
    stderr_lines: List[bytes] = []

    def _drain_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_lines.append(line)

    drain = threading.Thread(target=_drain_stderr, daemon=True)
    drain.start()

    frames = 0
    try:
        for path in paths:
            frame = _read_gray_frame(path)
            if frame.shape != (height, width):
                raise MovieError(
                    f"image {path} is {frame.shape[1]}x{frame.shape[0]} px, "
                    f"expected {width}x{height} px"
                )
            process.stdin.write(frame.tobytes())
            frames += 1
        process.stdin.close()
        returncode = process.wait()
    except BrokenPipeError as exc:
        process.kill()
        process.wait()
        drain.join(timeout=5.0)
        message = b"".join(stderr_lines).decode("utf-8", "replace").strip()
        raise MovieError(f"ffmpeg stopped early: {message or exc}") from exc
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            # e.g. a MovieError raised while feeding frames: do not leave a
            # half-written encoder behind.
            process.kill()
            process.wait()
        drain.join(timeout=5.0)

    if returncode != 0:
        message = b"".join(stderr_lines).decode("utf-8", "replace").strip()
        raise MovieError(f"ffmpeg failed (exit code {returncode}): {message}")

    LOGGER.debug("wrote movie %s (%d frames)", output_path, frames)
    return {
        "filename": output_path.name,
        "frames": frames,
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "codec": "h264",
        "container": "mp4",
        "pixel_format": "yuv420p",
        "size_bytes": int(output_path.stat().st_size),
    }
