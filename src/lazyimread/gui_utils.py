"""GUI-based file loading for LazyImRead."""

import tkinter as tk
from logging import getLogger
from pathlib import Path
from tkinter import filedialog

import numpy as np

from .lazyimread import (
    LazyImReadError,
    LoadOptions,
    imread,
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
    root.attributes("-topmost", "True")  # Make the dialog appear on top

    file_path = filedialog.askopenfilename(
        title="Select file to load",
        filetypes=[
            (
                "All supported files",
                "*.tif *.tiff *.h5 *.hdf5 *.png *.jpg *.jpeg *.bmp "
                "*.mov *.mp4 *.avi *.webm *.mkv",
            ),
            ("TIFF files", "*.tif *.tiff"),
            ("HDF5 files", "*.h5 *.hdf5"),
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


def gdir(options: LoadOptions | None = None) -> tuple[Path, LoadOptions | None]:
    """Open a directory dialog for selecting an input folder or Zarr store.

    :param options: LoadOptions instance with loading parameters
    :return: Tuple of (selected folder path, options)
    """
    logger.info("Opening directory dialog for GUI-based loading")
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", "True")  # Make the dialog appear on top

    folder_path = filedialog.askdirectory(
        title="Select folder or Zarr store to load",
    )
    if not folder_path:
        logger.warning("No folder selected")
        raise LazyImReadError("No folder selected")

    logger.info(f"Folder selected: {folder_path}")
    return Path(folder_path), options


def gload(options: LoadOptions | None = None) -> np.ndarray:
    """Load data using a GUI file selector and then call imread.

    :param options: LoadOptions instance with loading parameters
    :return: Tuple of (data, dimension order, metadata)
    """
    file_path, options = gpath(options)
    return imread(file_path, options)


def gdirload(options: LoadOptions | None = None) -> np.ndarray:
    """Load data using a GUI directory selector and then call imread.

    :param options: LoadOptions instance with loading parameters
    :return: Tuple of (data, dimension order, metadata)
    """
    folder_path, options = gdir(options)
    return imread(folder_path, options)


def gset() -> LoadOptions:
    """Open a GUI to load LoadOptions from a YAML file.

    :return: LoadOptions instance
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", "True")  # Make the dialog appear on top

    file_path = filedialog.askopenfilename(
        title="Select LoadOptions YAML file",
        filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
    )
    if not file_path:
        logger.warning("No LoadOptions file selected")
        raise LazyImReadError("No LoadOptions file selected")

    return load_options(Path(file_path))


def gsetload() -> np.ndarray:
    """Load data using a GUI file selector and LoadOptions from a YAML file.

    :return: Tuple of (data, dimension order, metadata)
    """
    try:
        options = gset()
        return gload(options)
    except LazyImReadError as err:
        logger.exception("Error loading data with GUI")
        raise LazyImReadError("Error loading data with GUI") from err


def gsetdirload() -> np.ndarray:
    """Load data using a GUI directory selector and LoadOptions from a YAML file.

    :return: Tuple of (data, dimension order, metadata)
    """
    try:
        options = gset()
        return gdirload(options)
    except LazyImReadError as err:
        logger.exception("Error loading data with GUI")
        raise LazyImReadError("Error loading data with GUI") from err
