"""``dicgen`` -- synthetic 2D Digital Image Correlation (DIC) dataset generator.

The package generates a deterministic grayscale image sequence of a speckle
pattern deforming (by default) with a single 2D affine transform applied
incrementally over the sequence, together with the *exact* analytic ground-truth
displacement and strain fields.

Main entry point::

    python -m dicgen --output-dir out/ -n 11 --strain-x 0.02

See ``README.md`` for the deformation convention, the output file layout and
worked examples.
"""

from ._version import __version__
from .deformation import (
    AffineModel,
    DeformationError,
    DeformationModel,
    SinusoidalModel,
    load_fraction,
    make_model,
    rotation_matrix,
)
from .imagegen import (
    apply_noise,
    pixel_grid,
    save_gray_png,
    save_mask_png,
    warp_image,
    to_uint8,
    valid_mask,
)
from .speckle import DotSet, SpeckleParams, generate_dots, rasterise_dots
from .movie import (
    DEFAULT_MOVIE_FPS,
    MOVIE_FILENAME,
    MovieError,
    ffmpeg_available,
    write_movie,
)
from .dataset import GenerationConfig, GenerationSummary, generate_dataset

__all__ = [
    "__version__",
    "AffineModel",
    "DeformationError",
    "DeformationModel",
    "SinusoidalModel",
    "load_fraction",
    "make_model",
    "rotation_matrix",
    "apply_noise",
    "pixel_grid",
    "save_gray_png",
    "save_mask_png",
    "warp_image",
    "to_uint8",
    "valid_mask",
    "DotSet",
    "SpeckleParams",
    "generate_dots",
    "rasterise_dots",
    "DEFAULT_MOVIE_FPS",
    "MOVIE_FILENAME",
    "MovieError",
    "ffmpeg_available",
    "write_movie",
    "GenerationConfig",
    "GenerationSummary",
    "generate_dataset",
]