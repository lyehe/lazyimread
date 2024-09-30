"""GUI-based file loading for LazyImRead."""

import tkinter as tk
from logging import getLogger
from pathlib import Path
from tkinter import filedialog

import numpy as np

from .lazyimread import (
    LazyImReadError,
    LoadOptions,
    MetadataSaveOption,
    imread,
    imsave,
    load_options,
)

# Set up logging
logger = getLogger(__name__)


def gpath(options: LoadOptions | None = None) -> tuple[Path, LoadOptions | None]:
    """Open a file dialog for selecting an input file.

    :param options: LoadOptions instance with loading parameters
    :return: Tuple of (selected file path, options)
    """
    logger.info("Opening file dialog for GUI-based loading")
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
        title="Select file to load",
        filetypes=[
            (
                "All supported files",
                "*.tif *.tiff *.h5 *.hdf5 *.zarr *.png *.jpg *.jpeg *.bmp "
                "*.mov *.mp4 *.avi *.webm *.mkv",
            ),
            ("TIFF files", "*.tif *.tiff"),
            ("HDF5 files", "*.h5 *.hdf5"),
            ("Zarr files", "*.zarr"),
            ("Image files", "*.png *.jpg *.jpeg *.bmp"),
            ("Video files", "*.mov *.mp4 *.avi *.webm *.mkv"),
            ("All files", "*.*"),
        ],
    )

    if not file_path:
        logger.warning("No file selected")
        raise LazyImReadError("No file selected")

    logger.info(f"File selected: {file_path}")
    return Path(file_path), options


def gload(options: LoadOptions | None = None) -> tuple[np.ndarray, str, dict | None]:
    """Load data using a GUI file selector and then call imread.

    :param options: LoadOptions instance with loading parameters
    :return: Tuple of (data, dimension order, metadata)
    """
    file_path, options = gpath(options)
    return imread(file_path, options)


def gset() -> LoadOptions:
    """Open a GUI to load LoadOptions from a YAML file.

    :return: LoadOptions instance
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
        title="Select LoadOptions YAML file",
        filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
    )

    if not file_path:
        logger.warning("No LoadOptions file selected")
        raise LazyImReadError("No LoadOptions file selected")

    return load_options(Path(file_path))


def gsetload() -> tuple[np.ndarray, str, dict | None]:
    """Load data using a GUI file selector and LoadOptions from a YAML file.

    :return: Tuple of (data, dimension order, metadata)
    """
    try:
        options = gset()
        return gload(options)
    except LazyImReadError as err:
        logger.exception("Error loading data with GUI")
        raise LazyImReadError("Error loading data with GUI") from err


def gsave(
    data: np.ndarray,
    dim_order: str,
    save_metadata: MetadataSaveOption = MetadataSaveOption.DONT_SAVE,
    metadata: dict | None = None,
) -> None:
    """Save data using a GUI file selector and then call imsave.

    :param data: numpy array to save
    :param dim_order: Dimension order of the data
    :param save_metadata: Whether to save metadata
    :param metadata: Optional metadata to save
    """
    logger.info("Opening file dialog for GUI-based saving")
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.asksaveasfilename(
        title="Save file as",
        filetypes=[
            ("TIFF files", "*.tif *.tiff"),
            ("HDF5 files", "*.h5 *.hdf5"),
            ("Zarr files", "*.zarr"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*"),
        ],
        defaultextension=".tif",
    )

    if not file_path:
        logger.warning("No file selected for saving")
        raise LazyImReadError("No file selected for saving")

    output_path = Path(file_path)
    logger.info(f"Saving file to: {output_path}")

    try:
        imsave(data, output_path, dim_order, save_metadata, metadata)
        logger.info(f"File successfully saved to {output_path}")
    except Exception as err:
        logger.exception(f"Error saving file: {err}")
        raise LazyImReadError(f"Error saving file: {err}") from err
