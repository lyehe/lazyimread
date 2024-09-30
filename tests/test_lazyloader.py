"""Test suite for the lazyloader module."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from lazyimread import LoadOptions, imset, lazyload
from lazyimread.dummy_data_generator import generate_test_data


@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory):
    """Generate test data and return path to test directory."""
    test_dir = tmp_path_factory.mktemp("test_data")
    generate_test_data(test_dir)
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


def test_load_2d_tiff(test_data_dir: Path) -> None:
    """Test loading a 2D TIFF file."""
    file_path = test_data_dir / "test_2d_XY_128x128.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (128, 128)
    assert str(dim_order) == "XY"


def test_load_3d_tiff_time_series(test_data_dir: Path) -> None:
    """Test loading a 3D TIFF file with a time series."""
    file_path = test_data_dir / "test_3d_TXY_50x128x128.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 128, 128)
    assert str(dim_order) in ["TXY", "ZXY"]


def test_load_3d_tiff_rgb(test_data_dir: Path) -> None:
    """Test loading a 3D TIFF file with RGB data."""
    file_path = test_data_dir / "test_3d_XYC_128x128x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (128, 128, 3)
    assert str(dim_order) == "XYC"


def test_load_4d_tiff(test_data_dir: Path) -> None:
    """Test loading a 4D TIFF file."""
    file_path = test_data_dir / "test_4d_TXYC_50x128x128x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 128, 128, 3)
    assert str(dim_order) in ["TXYC", "ZXYC"]


def test_load_2d_hdf5(test_data_dir: Path) -> None:
    """Test loading a 2D HDF5 file."""
    file_path = test_data_dir / "test_2d_XY_128x128.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (128, 128)
    assert str(dim_order) == "XY"


def test_load_3d_hdf5_time_series(test_data_dir: Path) -> None:
    """Test loading a 3D HDF5 file with a time series."""
    file_path = test_data_dir / "test_3d_TXY_50x128x128.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 128, 128)
    assert str(dim_order) in ["TXY", "ZXY"]


def test_load_3d_hdf5_rgb(test_data_dir: Path) -> None:
    """Test loading a 3D HDF5 file with RGB data."""
    file_path = test_data_dir / "test_3d_XYC_128x128x3.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (128, 128, 3)
    assert str(dim_order) == "XYC"


def test_load_4d_hdf5(test_data_dir: Path) -> None:
    """Test loading a 4D HDF5 file."""
    file_path = test_data_dir / "test_4d_TXYC_50x128x128x3.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 128, 128, 3)
    assert str(dim_order) in ["TXYC", "ZXYC"]


def test_load_video(test_data_dir: Path) -> None:
    """Test loading a video file."""
    file_path = test_data_dir / "test_4d_TXYC_50x128x128x3.avi"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 128, 128, 3)
    assert str(dim_order) == "TXYC"


def test_load_with_options(test_data_dir: Path) -> None:
    """Test loading a 4D TIFF file with options."""
    file_path = test_data_dir / "test_4d_TXYC_50x128x128x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    options = imset(t_range=(10, 30), y_range=(10, 110), x_range=(10, 110))
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (20, 100, 100, 3)
    assert dim_order in ["TXYC", "ZXYC"]


def test_load_partial_video(test_data_dir: Path) -> None:
    """Test loading a partial video file."""
    file_path = test_data_dir / "test_4d_TXYC_50x128x128x3.avi"
    assert file_path.exists(), f"File not found: {file_path}"
    options = imset(t_range=(5, 25), y_range=(10, 110), x_range=(10, 110))
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (20, 100, 100, 3)
    assert dim_order == "TXYC"


def test_load_5d_tiff(test_data_dir: Path) -> None:
    """Test loading a 5D TIFF file."""
    file_path = test_data_dir / "test_5d_TZCYX_50x50x128x128x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 50, 128, 128, 3)
    assert dim_order == "TZXYC"


def test_load_5d_hdf5(test_data_dir: Path) -> None:
    """Test loading a 5D HDF5 file."""
    file_path = test_data_dir / "test_5d_TZCYX_50x50x128x128x3.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 50, 128, 128, 3)
    assert dim_order == "TZXYC"


def test_load_3d_tiff_z_stack(test_data_dir: Path) -> None:
    """Test loading a 3D TIFF file with a Z stack."""
    file_path = test_data_dir / "test_3d_ZXY_50x128x128.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 128, 128)
    assert dim_order in ["ZXY", "TXY"]


def test_load_4d_tiff_z_stack_time_series(test_data_dir: Path) -> None:
    """Test loading a 4D TIFF file with a Z stack and a time series."""
    file_path = test_data_dir / "test_4d_TZXY_50x50x128x128.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 50, 128, 128)
    assert dim_order == "TZXY"


def test_load_4d_tiff_z_stack_rgb(test_data_dir: Path) -> None:
    """Test loading a 4D TIFF file with a Z stack and RGB data."""
    file_path = test_data_dir / "test_4d_ZXYC_50x128x128x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 128, 128, 3)
    assert dim_order in ["ZXYC", "TXYC"]


def test_load_with_options_5d(test_data_dir: Path) -> None:
    """Test loading a 5D TIFF file with options."""
    file_path = test_data_dir / "test_5d_TZCYX_50x50x128x128x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    options = imset(t_range=(5, 15), z_range=(2, 8), y_range=(10, 110), x_range=(10, 110))
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (10, 6, 100, 100, 3)
    assert dim_order == "TZXYC"


def test_load_2d_grayscale_video(test_data_dir: Path) -> None:
    """Test loading a 2D grayscale video file."""
    file_path = test_data_dir / "test_3d_TXY_50x128x128_grayscale.avi"
    assert file_path.exists(), f"File not found: {file_path}"
    options = imset(target_order="TXY")
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 128, 128)
    assert dim_order == "TXY"


def test_load_partial_5d_hdf5(test_data_dir: Path) -> None:
    """Test loading a partial 5D HDF5 file."""
    file_path = test_data_dir / "test_5d_TZCYX_50x50x128x128x3.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    options = imset(
        t_range=(5, 15),
        z_range=(2, 8),
        y_range=(10, 110),
        x_range=(10, 110),
        c_range=(0, 2),
    )
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (10, 6, 100, 100, 2)
    assert dim_order == "TZXYC"


def test_load_2d_zarr(test_data_dir: Path) -> None:
    """Test loading a 2D Zarr file."""
    file_path = test_data_dir / "test_2d_XY_128x128.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (128, 128)
    assert dim_order == "XY"


def test_load_3d_zarr_time_series(test_data_dir: Path) -> None:
    """Test loading a 3D Zarr file with a time series."""
    file_path = test_data_dir / "test_3d_TXY_50x128x128.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 128, 128)
    assert dim_order in ["TXY", "ZXY"]


def test_load_4d_zarr(test_data_dir: Path) -> None:
    """Test loading a 4D Zarr file."""
    file_path = test_data_dir / "test_4d_TXYC_50x128x128x3.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 128, 128, 3)
    assert dim_order in ["TXYC", "ZXYC"]


def test_load_5d_zarr(test_data_dir: Path) -> None:
    """Test loading a 5D Zarr file."""
    file_path = test_data_dir / "test_5d_TZCYX_50x50x128x128x3.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 50, 128, 128, 3)
    assert dim_order == "TZXYC"


def test_load_3d_zarr_z_stack(test_data_dir: Path) -> None:
    """Test loading a 3D Zarr file with a Z stack."""
    file_path = test_data_dir / "test_3d_ZXY_50x128x128.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 128, 128)
    assert dim_order in ["ZXY", "TXY"]


def test_load_4d_zarr_z_stack_time_series(test_data_dir: Path) -> None:
    """Test loading a 4D Zarr file with a Z stack and a time series."""
    file_path = test_data_dir / "test_4d_TZXY_50x50x128x128.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 50, 128, 128)
    assert dim_order == "TZXY"


def test_load_4d_zarr_z_stack_rgb(test_data_dir: Path) -> None:
    """Test loading a 4D Zarr file with a Z stack and RGB data."""
    file_path = test_data_dir / "test_4d_ZXYC_50x128x128x3.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 128, 128, 3)
    assert dim_order in ["ZXYC", "TXYC"]


def test_load_zarr_with_options(test_data_dir: Path) -> None:
    """Test loading a 5D Zarr file with options."""
    file_path = test_data_dir / "test_5d_TZCYX_50x50x128x128x3.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    options = imset(
        t_range=(5, 15),
        z_range=(2, 8),
        y_range=(10, 110),
        x_range=(10, 110),
        c_range=(0, 2),
    )
    data, dim_order, metadata = lazyload(file_path, options)
    assert isinstance(data, np.ndarray)
    assert data.shape == (10, 6, 100, 100, 2)
    assert dim_order == "TZXYC"


def test_load_2d_grayscale_image_folder(test_data_dir: Path) -> None:
    """Test loading a 2D grayscale image folder."""
    folder_path = test_data_dir / "test_2d_XY_128x128_folder"
    assert folder_path.exists(), f"Folder not found: {folder_path}"
    options = imset(target_order="TXY")
    data, dim_order, metadata = lazyload(folder_path, options)
    assert data.shape == (50, 128, 128)
    assert dim_order == "TXY"
    assert len(os.listdir(folder_path)) == 50


def test_load_3d_rgb_image_folder(test_data_dir: Path) -> None:
    """Test loading a 3D RGB image folder."""
    folder_path = test_data_dir / "test_3d_XYC_128x128x3_folder"
    assert folder_path.exists(), f"Folder not found: {folder_path}"
    options = imset(target_order="TXYC")
    data, dim_order, metadata = lazyload(folder_path, options)
    assert data.shape == (50, 128, 128, 3)
    assert dim_order == "TXYC"
    assert len(os.listdir(folder_path)) == 50


def test_load_partial_2d_grayscale_image_folder(test_data_dir: Path) -> None:
    """Test loading a partial 2D grayscale image folder."""
    folder_path = test_data_dir / "test_2d_XY_128x128_folder"
    assert folder_path.exists(), f"Folder not found: {folder_path}"
    options = imset(t_range=(10, 30), y_range=(10, 110), x_range=(10, 110), target_order="TXY")
    data, dim_order, metadata = lazyload(folder_path, options)
    assert data.shape == (20, 100, 100)
    assert dim_order == "TXY"


def test_load_partial_3d_rgb_image_folder(test_data_dir: Path) -> None:
    """Test loading a partial 3D RGB image folder."""
    folder_path = test_data_dir / "test_3d_XYC_128x128x3_folder"
    assert folder_path.exists(), f"Folder not found: {folder_path}"
    options = imset(
        t_range=(10, 30),
        y_range=(10, 110),
        x_range=(10, 110),
        c_range=(1, 3),
        target_order="TXYC",
    )
    data, dim_order, metadata = lazyload(folder_path, options)
    assert data.shape == (20, 100, 100, 2)
    assert dim_order == "TXYC"


def test_load_hdf5_multi_dataset(test_data_dir: Path) -> None:
    """Test loading a multi-dataset HDF5 file."""
    file_path = test_data_dir / "multi_dataset.h5"
    assert file_path.exists(), f"File not found: {file_path}"

    # Test loading the first dataset using integer selector
    options = LoadOptions(dataset=0)
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 128, 128)
    assert dim_order in ["TXY", "ZXY"]

    # Test loading the second dataset using integer selector
    options = LoadOptions(dataset=1)
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 128, 128, 3)
    assert dim_order in ["TXYC", "ZXYC"]

    # Test loading a dataset from a group using string selectors
    options = LoadOptions(group="group1", dataset="dataset3")
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (10, 10, 10, 10)
    assert dim_order == "TZXY"

    # Test loading a large dataset from a group using string selectors
    options = LoadOptions(group="group1", dataset="dataset4")
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 50, 128, 128, 3)
    assert dim_order == "TZXYC"


def test_load_zarr_multi_dataset(test_data_dir: Path) -> None:
    """Test loading a multi-dataset Zarr file."""
    file_path = test_data_dir / "multi_dataset.zarr"
    assert file_path.exists(), f"File not found: {file_path}"

    # Test loading the first dataset using integer selector
    options = LoadOptions(dataset=0)
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 128, 128)
    assert dim_order in ["TXY", "ZXY"]

    # Test loading the second dataset using integer selector
    options = LoadOptions(dataset=1)
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 128, 128, 3)
    assert dim_order in ["TXYC", "ZXYC"]

    # Test loading a dataset from a group using string selectors
    options = LoadOptions(group="group1", dataset="dataset3")
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (10, 10, 10, 10)
    assert dim_order == "TZXY"

    # Test loading a large dataset from a group using string selectors
    options = LoadOptions(group="group2", dataset="dataset6")
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 50, 128, 128, 3)
    assert dim_order == "TZXYC"


def test_load_partial_hdf5_multi_dataset(test_data_dir: Path) -> None:
    """Test loading a partial multi-dataset HDF5 file."""
    file_path = test_data_dir / "multi_dataset.h5"
    assert file_path.exists(), f"File not found: {file_path}"

    options = imset(
        group="group1",
        dataset="dataset4",
        t_range=(5, 15),
        z_range=(2, 8),
        y_range=(10, 110),
        x_range=(10, 110),
        c_range=(0, 2),
    )
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (10, 6, 100, 100, 2)
    assert dim_order == "TZXYC"


def test_load_partial_zarr_multi_dataset(test_data_dir: Path) -> None:
    """Test loading a partial multi-dataset Zarr file."""
    file_path = test_data_dir / "multi_dataset.zarr"
    assert file_path.exists(), f"File not found: {file_path}"

    options = imset(
        group="group2",
        dataset="dataset6",
        t_range=(5, 15),
        z_range=(2, 8),
        y_range=(10, 110),
        x_range=(10, 110),
        c_range=(0, 2),
    )
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (10, 6, 100, 100, 2)
    assert dim_order == "TZXYC"
