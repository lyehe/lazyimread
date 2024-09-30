"""Test the asyncio capabilities."""

import asyncio
import shutil
from pathlib import Path

import numpy as np
import pytest

from lazyimread import (
    LoadOptions,
    async_imread,
    async_lazyload,
    async_load,
    configure_load_options,
)
from tests.dummy_data_generator import generate_test_data


@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory):
    """Generate test data and return path to test directory."""
    test_dir = tmp_path_factory.mktemp("test_data")
    generate_test_data(test_dir)
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.mark.asyncio()
async def test_async_load_2d_tiff(test_data_dir: Path) -> None:
    """Test asynchronous loading of a 2D TIFF file."""
    file_path = test_data_dir / "test_2d_XY_128x128.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = await async_lazyload(file_path)
    assert data.shape == (128, 128)
    assert dim_order == "XY"


@pytest.mark.asyncio()
async def test_async_load_3d_tiff_time_series(test_data_dir: Path) -> None:
    """Test asynchronous loading of a 3D TIFF time series."""
    file_path = test_data_dir / "test_3d_TXY_50x128x128.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = await async_load(file_path)
    assert data.shape == (50, 128, 128)
    assert dim_order in ["TXY", "ZXY"]


@pytest.mark.asyncio()
async def test_async_load_4d_tiff(test_data_dir: Path) -> None:
    """Test asynchronous loading of a 4D TIFF file."""
    file_path = test_data_dir / "test_4d_TXYC_50x128x128x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = await async_imread(file_path)
    assert data.shape == (50, 128, 128, 3)
    assert dim_order in ["TXYC", "ZXYC"]


@pytest.mark.asyncio()
async def test_async_load_with_options(test_data_dir: Path) -> None:
    """Test asynchronous loading with specific load options."""
    file_path = test_data_dir / "test_4d_TXYC_50x128x128x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    options = configure_load_options(t_range=(10, 30), y_range=(10, 110), x_range=(10, 110))
    data, dim_order, metadata = await async_lazyload(file_path, options)
    assert data.shape == (20, 100, 100, 3)
    assert dim_order in ["TXYC", "ZXYC"]


@pytest.mark.asyncio()
async def test_async_load_5d_hdf5(test_data_dir: Path) -> None:
    """Test asynchronous loading of a 5D HDF5 file."""
    file_path = test_data_dir / "test_5d_TZCYX_50x50x128x128x3.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = await async_load(file_path)
    assert data.shape == (50, 50, 128, 128, 3)
    assert dim_order == "TZXYC"


@pytest.mark.asyncio()
async def test_async_load_zarr(test_data_dir: Path) -> None:
    """Test asynchronous loading of a Zarr file."""
    file_path = test_data_dir / "test_4d_TXYC_50x128x128x3.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = await async_imread(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 128, 128, 3)
    assert dim_order in ["TXYC", "ZXYC"]


@pytest.mark.asyncio()
async def test_async_load_video(test_data_dir: Path) -> None:
    """Test asynchronous loading of a video file."""
    file_path = test_data_dir / "test_4d_TXYC_50x128x128x3.avi"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = await async_lazyload(file_path)
    assert data.shape == (50, 128, 128, 3)
    assert dim_order == "TXYC"


@pytest.mark.asyncio()
async def test_async_load_image_folder(test_data_dir: Path) -> None:
    """Test asynchronous loading of an image folder."""
    folder_path = test_data_dir / "test_2d_XY_128x128_folder"
    assert folder_path.exists(), f"Folder not found: {folder_path}"
    options = configure_load_options(target_order="TXY")
    data, dim_order, metadata = await async_load(folder_path, options)
    assert data.shape == (50, 128, 128)
    assert dim_order == "TXY"


@pytest.mark.asyncio()
async def test_async_load_partial_5d_hdf5(test_data_dir: Path) -> None:
    """Test asynchronous loading of a partial 5D HDF5 file."""
    file_path = test_data_dir / "test_5d_TZCYX_50x50x128x128x3.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    options = configure_load_options(
        t_range=(5, 15),
        z_range=(2, 8),
        y_range=(10, 110),
        x_range=(10, 110),
        c_range=(0, 2),
    )
    data, dim_order, metadata = await async_imread(file_path, options)
    assert data.shape == (10, 6, 100, 100, 2)
    assert dim_order == "TZXYC"


@pytest.mark.asyncio()
async def test_async_load_hdf5_multi_dataset(test_data_dir: Path) -> None:
    """Test asynchronous loading of a multi-dataset HDF5 file."""
    file_path = test_data_dir / "multi_dataset.h5"
    assert file_path.exists(), f"File not found: {file_path}"

    options = LoadOptions(group="group1", dataset="dataset4")
    data, dim_order, metadata = await async_lazyload(file_path, options)
    assert data.shape == (50, 50, 128, 128, 3)
    assert dim_order == "TZXYC"


@pytest.mark.asyncio()
async def test_async_load_zarr_multi_dataset(test_data_dir: Path) -> None:
    """Test asynchronous loading of a multi-dataset Zarr file."""
    file_path = test_data_dir / "multi_dataset.zarr"
    assert file_path.exists(), f"File not found: {file_path}"

    options = LoadOptions(group="group2", dataset="dataset6")
    data, dim_order, metadata = await async_load(file_path, options)
    assert data.shape == (50, 50, 128, 128, 3)
    assert dim_order == "TZXYC"


@pytest.mark.asyncio()
async def test_async_load_multiple_files_concurrently(test_data_dir: Path) -> None:
    """Test asynchronous loading of multiple files concurrently."""
    file_paths = [
        test_data_dir / "test_2d_XY_128x128.tiff",
        test_data_dir / "test_3d_TXY_50x128x128.h5",
        test_data_dir / "test_4d_TXYC_50x128x128x3.zarr",
        test_data_dir / "test_5d_TZCYX_50x50x128x128x3.tiff",
    ]

    async def load_file(file_path):
        """Load a single file asynchronously."""
        return await async_lazyload(file_path)

    results = await asyncio.gather(*[load_file(path) for path in file_paths])

    assert len(results) == 4
    for data, dim_order, _ in results:
        assert isinstance(data, np.ndarray)
        assert isinstance(dim_order, str)
