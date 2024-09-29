"""Utility functions for saving data."""

from collections.abc import Callable
from enum import Enum, auto
from json import dump
from logging import getLogger
from pathlib import Path

import numpy as np
import zarr
from h5py import File
from tifffile import imwrite

# Set up logging
logger = getLogger(__name__)


class MetadataSaveOption(Enum):
    """Enum for metadata saving options."""

    SAVE = auto()
    DONT_SAVE = auto()


def save_tiff(
    data: np.ndarray,
    output_path: Path,
    dim_order: str,
    save_metadata: MetadataSaveOption = MetadataSaveOption.DONT_SAVE,
    metadata: dict | None = None,
) -> None:
    """Save data as a TIFF file.

    :param data: numpy array to save
    :param output_path: Path to save the TIFF file
    :param dim_order: Dimension order of the data
    :param save_metadata: Enum indicating whether to save metadata
    :param metadata: Optional metadata to save
    """
    logger.info(f"Saving TIFF file to {output_path}")
    imwrite(str(output_path), data, metadata={"axes": dim_order})

    if save_metadata == MetadataSaveOption.SAVE and metadata:
        metadata_path = output_path.with_suffix(".json")
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, "w") as f:
            dump(metadata, f, indent=2)


def save_hdf5(
    data: np.ndarray,
    output_path: Path,
    dataset_name: str = "data",
    dim_order: str = "",
    save_metadata: MetadataSaveOption = MetadataSaveOption.DONT_SAVE,
    metadata: dict | None = None,
) -> None:
    """Save data as an HDF5 file.

    :param data: numpy array to save
    :param output_path: Path to save the HDF5 file
    :param dataset_name: Name of the dataset in the HDF5 file
    :param dim_order: Dimension order of the data
    :param save_metadata: Enum indicating whether to save metadata
    :param metadata: Optional metadata to save
    """
    logger.info(f"Saving HDF5 file to {output_path}")
    with File(output_path, "w") as f:
        dataset = f.create_dataset(dataset_name, data=data)
        if dim_order:
            dataset.attrs["dim_order"] = dim_order

        if save_metadata == MetadataSaveOption.SAVE and metadata:
            for key, value in metadata.items():
                dataset.attrs[key] = value


def save_zarr(
    data: np.ndarray,
    output_path: Path,
    group_name: str = "data",
    dim_order: str = "",
    save_metadata: MetadataSaveOption = MetadataSaveOption.DONT_SAVE,
    metadata: dict | None = None,
) -> None:
    """Save data as a Zarr file.

    :param data: numpy array to save
    :param output_path: Path to save the Zarr file
    :param group_name: Name of the group in the Zarr file
    :param dim_order: Dimension order of the data
    :param save_metadata: Enum indicating whether to save metadata
    :param metadata: Optional metadata to save
    """
    logger.info(f"Saving Zarr file to {output_path}")
    root = zarr.open(str(output_path), mode="w")
    dataset = root.create_dataset(group_name, data=data) if isinstance(root, zarr.Group) else None
    if dataset:
        if dim_order:
            dataset.attrs["dim_order"] = dim_order
        if save_metadata == MetadataSaveOption.SAVE and metadata:
            for key, value in metadata.items():
                dataset.attrs[key] = value


def save_folder(
    data: np.ndarray,
    output_path: Path,
    save_metadata: MetadataSaveOption = MetadataSaveOption.DONT_SAVE,
    metadata: dict | None = None,
) -> None:
    """Save data as a folder of images.

    :param data: numpy array to save
    :param output_path: Path to save the folder of images
    :param save_metadata: Enum indicating whether to save metadata
    :param metadata: Optional metadata to save
    """
    logger.info(f"Saving folder of images to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(data):
        img_path = output_path / f"{i:04d}.tiff"
        imwrite(str(img_path), img)

    if save_metadata == MetadataSaveOption.SAVE and metadata:
        metadata_path = output_path / "metadata.json"
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, "w") as f:
            dump(metadata, f, indent=2)


class SaveFactory:
    """Factory class for creating appropriate saver functions."""

    @staticmethod
    def get_saver(
        output_path: Path,
    ) -> Callable[..., None]:
        """Get the appropriate saver based on the output path.

        :param output_path: Path to save the output
        :return: Appropriate saver function
        """
        logger.debug(f"Getting saver for {output_path}")
        if output_path.suffix.lower() in (".tif", ".tiff"):
            logger.info("Using TIFF saver")
            return save_tiff
        elif output_path.suffix.lower() in (".h5", ".hdf5"):
            logger.info("Using HDF5 saver")
            return save_hdf5
        elif output_path.suffix.lower() == ".zarr":
            logger.info("Using Zarr saver")
            return save_zarr
        else:
            logger.info("Using folder saver")
            return save_folder
