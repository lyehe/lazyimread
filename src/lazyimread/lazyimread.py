"""Lazy image reading library."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, TypeVar

import cv2
import numpy as np
import yaml
from h5py import Dataset, File
from h5py import Group as H5pyGroup
from tifffile import TiffFile
from xmltodict import parse
from zarr import Array
from zarr import Group as ZarrGroup
from zarr import open as zarr_open

from .dimension_utils import predict_dimension_order, rearrange_dimensions
from .saving_utils import MetadataSaveOption, SaveFactory

# Set up logging
logger = getLogger(__name__)

# Define complex types
DimensionOrder = Literal["T", "Z", "X", "Y", "C"]
RangeType = int | tuple[int, int]
DimensionRangeType = dict[DimensionOrder, RangeType]
DatasetType = str | int | None
GroupType = str | int | None
FilePathType = str | Path
DataType = TypeVar("DataType", np.ndarray, Dataset, Array)


@dataclass
class LoadOptions:
    """Options for loading data, including ranges, dataset, and group.

    :param ranges: dictionary of dimension ranges
    :param dataset: Dataset name or index
    :param group: Group name or index
    :param dim_order: Input dimension order
    :param target_order: Target dimension order
    """

    ranges: dict = field(default_factory=dict)
    dataset: DatasetType | None = None
    group: GroupType | None = None
    dim_order: str | None = None
    target_order: str | None = None


def imset(
    t_range: RangeType | None = None,
    z_range: RangeType | None = None,
    x_range: RangeType | None = None,
    y_range: RangeType | None = None,
    c_range: RangeType | None = None,
    dataset: DatasetType | None = None,
    group: GroupType | None = None,
    dim_order: str | None = None,
    target_order: str | None = None,
) -> LoadOptions:
    """Create a LoadOptions instance with the given parameters.

    :param t_range: Range for the T dimension
    :param z_range: Range for the Z dimension
    :param x_range: Range for the X dimension
    :param y_range: Range for the Y dimension
    :param c_range: Range for the C dimension
    :param dataset: Dataset name or index
    :param group: Group name or index
    :param dim_order: Input dimension order
    :param target_order: Target dimension order
    :return: LoadOptions instance
    """
    ranges = {
        dim: range_val
        for dim, range_val in zip(
            "TZXYC", (t_range, z_range, x_range, y_range, c_range), strict=True
        )
        if range_val is not None
    }

    logger.debug(f"Creating LoadOptions with ranges={ranges}, dataset={dataset}, group={group}")
    return LoadOptions(
        ranges=ranges,
        dataset=dataset,
        group=group,
        dim_order=dim_order,
        target_order=target_order,
    )


def save_options(options: LoadOptions, file_path: Path) -> None:
    """Save LoadOptions to a YAML file.

    :param options: LoadOptions instance to save
    :param file_path: Path to save the YAML file
    """
    try:
        data = {
            "ranges": options.ranges,
            "dataset": options.dataset,
            "group": options.group,
            "dim_order": options.dim_order,
            "target_order": options.target_order,
        }
        with file_path.open("w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
        logger.info(f"LoadOptions saved to {file_path}")
    except Exception as err:
        logger.exception(f"Error saving LoadOptions: {err}")
        raise LazyImReadError(f"Error saving LoadOptions: {err}") from err


def load_options(file_path: Path) -> LoadOptions:
    """Load LoadOptions from a YAML file.

    :param file_path: Path to the YAML file
    :return: LoadOptions instance
    """
    try:
        with file_path.open("r") as f:
            data = yaml.safe_load(f)
        logger.info(f"LoadOptions loaded from {file_path}")
        return LoadOptions(**data)
    except Exception as err:
        logger.exception(f"Error loading LoadOptions: {err}")
        raise LazyImReadError(f"Error loading LoadOptions: {err}") from err


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(
        self, file_path: Path, options: LoadOptions
    ) -> tuple[np.ndarray, str | None, dict | None]:
        """Load data from a file.

        :param file_path: Path to the file to load
        :param options: LoadOptions instance with loading parameters
        :return: Tuple of (data, dimension order, metadata)
        """

    @staticmethod
    def _process_range(range_val: RangeType | None, max_val: int) -> tuple[int, int]:
        """Process and validate a range value."""
        if isinstance(range_val, int):
            return 0, min(range_val, max_val)
        if isinstance(range_val, tuple) and len(range_val) == 2:
            start, end = range_val
            return max(0, start), min(end, max_val)
        return 0, max_val

    def _calculate_slices(self, shape: tuple[int, ...], options: LoadOptions) -> tuple[slice, ...]:
        """Calculate slices for all dimensions based on the shape and options."""
        slices = []
        dim_order = options.dim_order or predict_dimension_order(shape)

        for i, dim in enumerate(dim_order):
            if dim in options.ranges:
                start, end = self._process_range(options.ranges[dim], shape[i])
                slices.append(slice(start, end))
            else:
                slices.append(slice(None))

        logger.debug(f"Calculated slices: {slices}")
        return tuple(slices)


class ImageFolderLoader(DataLoader):
    """Loader for image folders."""

    supported_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".jp2",
        ".j2k",
        ".webp",
        ".pbm",
        ".pgm",
        ".ppm",
        ".sr",
        ".ras",
    )

    def load(
        self, folder_path: Path, options: LoadOptions
    ) -> tuple[np.ndarray, str | None, dict | None]:
        """Load images from a folder.

        :param folder_path: Path to the folder containing images
        :param options: LoadOptions object containing loading parameters
        :return: Tuple of (data, dimension order, metadata)
        """
        logger.info(f"Loading images from folder: {folder_path}")
        image_files = sorted(
            [f for f in folder_path.iterdir() if f.suffix.lower() in self.supported_extensions]
        )
        if not image_files:
            logger.error(f"No image files found in folder: {folder_path}")
            raise ValueError(f"No image files found in folder: {folder_path}")

        try:
            first_image = cv2.imread(str(image_files[0]))
            if first_image is None:
                raise OSError(f"Unable to open first image file: {image_files[0]}")

            is_color = len(first_image.shape) == 3 and first_image.shape[2] == 3
            shape = (len(image_files),) + first_image.shape
            dim_order = "TXYC" if is_color else "TXY"

            slices = self._calculate_slices(shape, options)
            t_slice, y_slice, x_slice = slices[:3]

            t_start, t_stop = t_slice.start or 0, t_slice.stop or len(image_files)
            y_start, y_stop = y_slice.start or 0, y_slice.stop or first_image.shape[0]
            x_start, x_stop = x_slice.start or 0, x_slice.stop or first_image.shape[1]

            if is_color:
                c_slice = slices[3]
                c_start, c_stop = c_slice.start or 0, c_slice.stop or first_image.shape[2]
                data_shape = (
                    t_stop - t_start,
                    y_stop - y_start,
                    x_stop - x_start,
                    c_stop - c_start,
                )
            else:
                data_shape = (t_stop - t_start, y_stop - y_start, x_stop - x_start)

            data = np.empty(data_shape, dtype=np.uint8)

            for i, img_file in enumerate(image_files[t_start:t_stop]):
                logger.debug(f"Loading image: {img_file}")
                img = cv2.imread(str(img_file))
                if img is None:
                    raise OSError(f"Unable to open image file: {img_file}")
                if is_color:
                    data[i] = img[y_start:y_stop, x_start:x_stop, c_start:c_stop]
                else:
                    data[i] = img[y_start:y_stop, x_start:x_stop]

            logger.info(f"Loaded {len(data)} images")

            # Convert to grayscale if target order is TXY
            if options.target_order == "TXY" and is_color:
                logger.info("Converting color images to grayscale")
                data = np.mean(data, axis=-1).astype(np.uint8)
                dim_order = "TXY"

            return data, dim_order, None
        except Exception as err:
            logger.exception(f"Error loading images from folder: {err}")
            raise LazyImReadError(f"Error loading images from folder: {err}") from err


class VideoLoader(DataLoader):
    """Loader for video files."""

    def load(
        self, video_path: Path, options: LoadOptions
    ) -> tuple[np.ndarray, str, dict[str, Any] | None]:
        """Load frames from a video file.

        :param video_path: Path to the video file
        :param options: LoadOptions object containing loading parameters
        :return: Tuple of (data, dimension order, metadata)
        """
        logger.info(f"Loading video from: {video_path}")
        is_grayscale: bool = options.target_order == "TXY"

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise OSError(f"Failed to open video file: {video_path}")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("Failed to read the first frame of the video.")

            shape = (frame_count,) + first_frame.shape
            slices = self._calculate_slices(shape, options)

            t_slice, y_slice, x_slice, *c_slice = slices + (slice(None),)
            t_start = t_slice.start or 0
            t_stop = t_slice.stop or frame_count
            y_start, y_stop = y_slice.start or 0, y_slice.stop or first_frame.shape[0]
            x_start, x_stop = x_slice.start or 0, x_slice.stop or first_frame.shape[1]

            data_shape = (t_stop - t_start, y_stop - y_start, x_stop - x_start)
            if not is_grayscale:
                c_start, c_stop = c_slice[0].start or 0, c_slice[0].stop or first_frame.shape[2]
                data_shape += (c_stop - c_start,)

            data = np.empty(data_shape, dtype=np.uint8)
            dim_order = "TXY" if is_grayscale else "TXYC"

            cap.set(cv2.CAP_PROP_POS_FRAMES, t_start)
            for i in range(t_stop - t_start):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Reached end of video at frame {i + t_start}")
                    data = data[:i]
                    break
                processed_frame = frame[y_start:y_stop, x_start:x_stop]
                if is_grayscale:
                    data[i] = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                else:
                    data[i] = processed_frame[:, :, c_start:c_stop]

            logger.info(f"Loaded {len(data)} frames from video")
            return data, dim_order, None
        except Exception as err:
            logger.exception(f"Error loading video: {err}")
            raise LazyImReadError(f"Error loading video: {err}") from err
        finally:
            if "cap" in locals():
                cap.release()


class TiffLoader(DataLoader):
    """Loader for TIFF files."""

    def load(
        self, file_path: Path, options: LoadOptions
    ) -> tuple[np.ndarray, str | None, dict | None]:
        """Load data from a TIFF or OME-TIFF file.

        :param file_path: Path to the TIFF file
        :param options: LoadOptions object containing loading parameters
        :return: Tuple of (data, dimension order, metadata)
        """
        logger.info(f"Loading TIFF file: {file_path}")
        try:
            with TiffFile(str(file_path)) as tif:
                return (
                    self._load_ome_tiff(tif, options)
                    if tif.is_ome
                    else self._load_regular_tiff(tif, options)
                )
        except Exception as err:
            logger.exception(f"Error loading TIFF file: {err}")
            raise LazyImReadError(f"Error loading TIFF file: {err}") from err

    def _load_regular_tiff(
        self, tif: TiffFile, options: LoadOptions
    ) -> tuple[np.ndarray, str, dict | None]:
        """Load data from a regular TIFF file."""
        metadata = {**tif.imagej_metadata} if tif.imagej_metadata else {}
        metadata.update({tag.name: tag.value for tag in tif.pages[0].tags})

        dim_order = predict_dimension_order(tif.series[0].shape)
        logger.debug(f"Predicted dimension order for TIFF: {dim_order}")

        slices = self._calculate_slices(tif.series[0].shape, options)
        data = tif.asarray()[slices]

        # Handle grayscale videos
        if data.ndim == 3 and data.shape[-1] == 3:
            if np.all(data[:, :, 0] == data[:, :, 1]) and np.all(data[:, :, 1] == data[:, :, 2]):
                logger.info("Detected grayscale video, removing color dimension")
                data = data[:, :, 0]
                dim_order = dim_order.replace("C", "")

        return data, dim_order, metadata

    def _load_ome_tiff(
        self, tif: TiffFile, options: LoadOptions
    ) -> tuple[np.ndarray, str | None, dict | None]:
        """Load data from an OME-TIFF file."""
        metadata = tif.ome_metadata

        dim_order = predict_dimension_order(tif.series[0].shape)
        if isinstance(metadata, dict) and "Image" in metadata:
            image_metadata = metadata.get("Image")
            if isinstance(image_metadata, dict):
                ome_dim_order = image_metadata.get("DimensionOrder", "")
                dim_order = "".join(
                    char for char in ome_dim_order if char in "TZXYC"[: len(tif.series[0].shape)]
                )

        logger.debug(f"Dimension order: {dim_order}")

        slices = self._calculate_slices(tif.series[0].shape, options)
        data = tif.asarray()[slices]

        # Handle grayscale videos
        if data.ndim == 3 and data.shape[-1] == 3:
            if np.all(data[:, :, 0] == data[:, :, 1]) and np.all(data[:, :, 1] == data[:, :, 2]):
                logger.info("Detected grayscale video, removing color dimension")
                data = data[:, :, 0]
                dim_order = dim_order.replace("C", "")

        metadata = parse(metadata) if isinstance(metadata, str) else metadata

        return data, dim_order, metadata


class HDF5Loader(DataLoader):
    """Loader for HDF5 files."""

    def load(self, file_path: Path, options: LoadOptions) -> tuple[np.ndarray, str, dict | None]:
        """Load data from an HDF5 file.

        :param file_path: Path to the HDF5 file
        :param options: LoadOptions object containing loading parameters
        :return: Tuple of (data, dimension order, metadata)
        """
        logger.info(f"Loading HDF5 file: {file_path}")
        try:
            with File(file_path, "r") as f:
                data, dim_order, metadata = self._get_data(f, options)
                slices = self._calculate_slices(data.shape, options)
                data = data[slices]
        except OSError as err:
            logger.exception(f"Error opening HDF5 file: {err}")
            raise LazyImReadError(f"Failed to open HDF5 file: {err}") from err
        except Exception as err:
            logger.exception(f"Unexpected error while loading HDF5 file: {err}")
            raise LazyImReadError(f"Error loading HDF5 file: {err}") from err
        finally:
            logger.debug("Finished HDF5 file loading operation")

        return data, dim_order, metadata

    def _get_data(
        self, root: File, options: LoadOptions
    ) -> tuple[np.ndarray, str, dict[str, Any] | None]:
        """Get the data array from an HDF5 file."""
        try:
            group = (
                root
                if options.group is None
                else (
                    self._get_group_by_index(root, options.group)
                    if isinstance(options.group, int)
                    else self._get_group_by_name(root, options.group)
                )
            )

            dataset = (
                self._get_first_dataset(group)
                if options.dataset is None
                else (
                    self._get_dataset_by_index(group, options.dataset)
                    if isinstance(options.dataset, int)
                    else self._get_dataset_by_name(group, options.dataset)
                )
            )

            data = dataset[:]
            metadata = dict(dataset.attrs)
            dim_order = str(metadata.get("dim_order", predict_dimension_order(data.shape)))

            return data, dim_order, metadata
        except Exception as err:
            logger.exception(f"Error getting data from HDF5 file: {err}")
            raise LazyImReadError(f"Error getting data from HDF5 file: {err}") from err

    def _get_group_by_index(self, file: File, index: int) -> H5pyGroup:
        """Get a group from the HDF5 file by index."""
        groups = [v for k, v in file.items() if isinstance(v, H5pyGroup)]
        if index < 0 or index >= len(groups):
            logger.error(f"Group index {index} is out of range")
            raise GroupNotFoundError(f"Group index {index} is out of range")
        return groups[index]

    def _get_group_by_name(self, file: File, name: str) -> H5pyGroup:
        """Get a group from the HDF5 file by name."""
        if name not in file:
            logger.error(f"Group '{name}' not found in file")
            raise GroupNotFoundError(f"Group '{name}' not found in file")
        group = file[name]
        if not isinstance(group, H5pyGroup):
            logger.error(f"'{name}' is not a Group, but {type(group)}")
            raise GroupNotFoundError(f"'{name}' is not a Group, but {type(group)}")
        return group

    def _get_first_dataset(self, group: File | H5pyGroup) -> Dataset:
        """Get the first dataset found in the group."""
        for name, item in group.items():
            if isinstance(item, Dataset):
                logger.debug(f"Using first dataset found: {name}")
                return item
        logger.error("No dataset found in the HDF5 file")
        raise DatasetNotFoundError("No dataset found in the HDF5 file")

    def _get_dataset_by_index(self, group: File | H5pyGroup, index: int) -> Dataset:
        """Get a dataset from the group by index."""
        datasets = [v for k, v in group.items() if isinstance(v, Dataset)]
        if index < 0 or index >= len(datasets):
            logger.error(f"Dataset index {index} is out of range")
            raise DatasetNotFoundError(f"Dataset index {index} is out of range")
        return datasets[index]

    def _get_dataset_by_name(self, group: File | H5pyGroup, name: str) -> Dataset:
        """Get a dataset from the group by name."""
        if name not in group:
            logger.error(f"Dataset '{name}' not found in group")
            raise DatasetNotFoundError(f"Dataset '{name}' not found in group")
        dataset = group[name]
        if not isinstance(dataset, Dataset):
            logger.error(f"'{name}' is not a Dataset, but {type(dataset)}")
            raise DatasetNotFoundError(f"'{name}' is not a Dataset, but {type(dataset)}")
        return dataset


class ZarrLoader(DataLoader):
    """Loader for Zarr files."""

    def load(
        self, file_path: Path, options: LoadOptions
    ) -> tuple[np.ndarray, str | None, dict[str, Any] | None]:
        """Load data from a Zarr file.

        :param file_path: Path to the Zarr file
        :param options: LoadOptions object containing loading parameters
        :return: Tuple of (data, dimension order, metadata)
        """
        logger.info(f"Loading Zarr file: {file_path}")
        try:
            root = zarr_open(str(file_path), mode="r")
            data_array = self._get_data(root, options)
            metadata = dict(data_array.attrs)
            dim_order = metadata.get("dim_order") or predict_dimension_order(data_array.shape)
            dim_order = "".join(char.upper() for char in dim_order if char.upper() in "TZXYC")

            logger.debug(f"Dimension order: {dim_order}")

            slices = self._calculate_slices(data_array.shape, options)
            data = data_array[slices]

            return data, dim_order, metadata or None
        except Exception as err:
            logger.exception(f"Error loading Zarr file: {err}")
            raise LazyImReadError(f"Error loading Zarr file: {err}") from err

    def _get_data(self, root: ZarrGroup | Array, options: LoadOptions) -> Array:
        """Get the data array from a Zarr file."""
        try:
            if isinstance(root, Array):
                return root
            group = self._get_group(root, options.group)
            return self._get_dataset(group, options.dataset)
        except Exception as err:
            logger.exception(f"Error getting data from Zarr file: {err}")
            raise LazyImReadError(f"Error getting data from Zarr file: {err}") from err

    def _get_group(self, root: ZarrGroup, group: GroupType | None) -> ZarrGroup:
        """Get a group from the Zarr file."""
        if group is None:
            return root
        return (
            self._get_group_by_index(root, group)
            if isinstance(group, int)
            else self._get_group_by_name(root, group)
        )

    def _get_group_by_index(self, root: ZarrGroup, index: int) -> ZarrGroup:
        """Get a group from the Zarr file by index."""
        group_names = list(root.group_keys())
        if 0 <= index < len(group_names):
            group_name = group_names[index]
            return root[group_name]  # type: ignore
        logger.error(f"Group index {index} is out of range")
        raise ValueError(f"Group index {index} is out of range")

    def _get_group_by_name(self, root: ZarrGroup, name: str) -> ZarrGroup:
        """Get a group from the Zarr file by name."""
        if name not in root:
            logger.error(f"Group '{name}' not found in the Zarr file.")
            raise ValueError(f"Group '{name}' not found in the Zarr file.")
        output = root[name]
        if not isinstance(output, ZarrGroup):
            logger.error(f"'{name}' is not a Group, but {type(output)}")
            raise ValueError(f"'{name}' is not a Group, but {type(output)}")
        return output

    def _get_dataset(self, group: ZarrGroup, dataset: DatasetType | None) -> Array:
        """Get a dataset from the Zarr group."""
        if isinstance(dataset, int):
            arrays = list(group.arrays())
            if 0 <= dataset < len(arrays):
                output = arrays[dataset][1]
                if not isinstance(output, Array):
                    logger.error(f"Array at index {dataset} is not an Array, but {type(output)}")
                    raise ValueError(
                        f"Array at index {dataset} is not an Array, but {type(output)}"
                    )
                return output
            logger.error(f"Dataset index {dataset} is out of range")
            raise ValueError(f"Dataset index {dataset} is out of range")
        if dataset is None:
            for name, value in group.arrays():
                if isinstance(value, Array):
                    logger.debug(f"Using first array found: {name}")
                    return value
            logger.error("No array found in Zarr group")
            raise ValueError("No array found in Zarr group")
        if dataset not in group or not isinstance(group[dataset], Array):
            logger.error(f"Dataset '{dataset}' not found in the Zarr group.")
            raise ValueError(f"Dataset '{dataset}' not found in the Zarr group.")
        output = group[dataset]
        if not isinstance(output, Array):
            logger.error(f"'{dataset}' is not an Array, but {type(output)}")
            raise ValueError(f"'{dataset}' is not an Array, but {type(output)}")
        return output


class DataLoaderFactory:
    """Factory class for creating appropriate DataLoader instances."""

    @staticmethod
    def get_loader(file_path: Path) -> DataLoader:
        """Get the appropriate loader based on the file path.

        :param file_path: Path to the file or directory
        :return: Appropriate DataLoader instance
        """
        logger.debug(f"Getting loader for {file_path}")

        if file_path.suffix.lower() == ".zarr" or DataLoaderFactory._is_zarr_directory(file_path):
            logger.info("Using ZarrLoader")
            return ZarrLoader()
        elif file_path.is_dir():
            logger.info("Using ImageFolderLoader")
            return ImageFolderLoader()
        elif file_path.suffix.lower() in (".mov", ".mp4", ".avi", ".webm", ".mkv"):
            logger.info("Using VideoLoader")
            return VideoLoader()
        elif file_path.suffix.lower() in (".tif", ".tiff"):
            logger.info("Using TiffLoader")
            return TiffLoader()
        elif file_path.suffix.lower() in (".h5", ".hdf5"):
            logger.info("Using HDF5Loader")
            return HDF5Loader()
        else:
            logger.error(f"Unsupported input format: {file_path}")
            raise FileFormatError(f"Unsupported input format: {file_path}")

    @staticmethod
    def _is_zarr_directory(path: Path) -> bool:
        """Check if the given path is a Zarr directory.

        :param path: Path to check
        :return: True if the path is a Zarr directory, False otherwise
        """
        return path.is_dir() and any(f.name in (".zarray", ".zgroup") for f in path.iterdir())


def lazyload(
    input_path: Path,
    options: LoadOptions | None = None,
) -> tuple[np.ndarray, str, dict | None]:
    """Main function. Load input data from various file formats.

    :param input_path: Path to the input file
    :param options: LoadOptions instance with loading parameters
    :return: Tuple of (data, dimension order, metadata)
    """
    logger.info(f"Loading input from {input_path}")
    options = options or LoadOptions()

    try:
        loader = DataLoaderFactory.get_loader(input_path)
        data, current_order, metadata = loader.load(input_path, options)

        current_order = options.dim_order or current_order or predict_dimension_order(data)
        logger.debug(f"Current dimension order: {current_order}")

        if options.target_order:
            logger.debug(f"Rearranging dimensions from {current_order} to {options.target_order}")
            data, final_order = rearrange_dimensions(data, current_order, options.target_order)
        else:
            final_order = current_order

        return data, str(final_order), metadata
    except FileFormatError as err:
        logger.exception(f"Unsupported file format: {input_path}")
        raise FileFormatError(f"Unsupported file format: {input_path}") from err
    except ValueError as err:
        logger.exception(f"Error loading data: {err}")
        raise LazyImReadError(f"Error loading data: {err}") from err
    except Exception as err:
        logger.exception(f"Unexpected error during data loading: {err}")
        raise LazyImReadError(f"Unexpected error during data loading: {err}") from err


def load(
    input_path: Path,
    options: LoadOptions | None = None,
) -> tuple[np.ndarray, str, dict | None]:
    """Alias for lazyload function, providing a shorter name for convenience.

    :param input_path: Path to the input file
    :param options: LoadOptions instance with loading parameters
    :return: Tuple of (data, dimension order, metadata)
    """
    return lazyload(input_path, options)


def imread(
    input_path: Path,
    options: LoadOptions | None = None,
) -> np.ndarray:
    """Alias for lazyload function, mimicking the common imread function name.

    This allows for easier transition from other image reading libraries.

    :param input_path: Path to the input file
    :param options: LoadOptions instance with loading parameters
    :return: Numpy array containing the loaded image data
    """
    data, _, _ = lazyload(input_path, options)
    return data


def imsave(
    data: np.ndarray,
    output_path: Path,
    dim_order: str,
    save_metadata: MetadataSaveOption = MetadataSaveOption.DONT_SAVE,
    metadata: dict | None = None,
) -> None:
    """Save data to various file formats.

    :param data: numpy array to save
    :param output_path: Path to save the output
    :param dim_order: Dimension order of the data
    :param save_metadata: Whether to save metadata
    :param metadata: Optional metadata to save
    """
    saver = SaveFactory.get_saver(output_path)
    saver(data, output_path, dim_order, save_metadata, metadata)


def imwrite(
    data: np.ndarray,
    output_path: Path,
    dim_order: str,
    save_metadata: MetadataSaveOption = MetadataSaveOption.DONT_SAVE,
    metadata: dict | None = None,
) -> None:
    """Alias for imsave function, mimicking the common imwrite function name.

    :param data: numpy array to save
    :param output_path: Path to save the output
    :param dim_order: Dimension order of the data
    :param save_metadata: Whether to save metadata
    :param metadata: Optional metadata to save
    """
    return imsave(data, output_path, dim_order, save_metadata, metadata)


def save(
    data: np.ndarray,
    output_path: Path,
    dim_order: str,
    save_metadata: MetadataSaveOption = MetadataSaveOption.DONT_SAVE,
    metadata: dict | None = None,
) -> None:
    """Alias for imsave function, providing a shorter name for convenience.

    :param data: numpy array to save
    :param output_path: Path to save the output
    :param dim_order: Dimension order of the data
    :param save_metadata: Whether to save metadata
    :param metadata: Optional metadata to save
    """
    return imsave(data, output_path, dim_order, save_metadata, metadata)


class LazyImReadError(Exception):
    """Base exception class for LazyImRead errors."""


class FileFormatError(LazyImReadError):
    """Raised when an unsupported file format is encountered."""


class DatasetNotFoundError(LazyImReadError):
    """Raised when a specified dataset is not found."""


class GroupNotFoundError(LazyImReadError):
    """Raised when a specified group is not found."""


class DimensionOrderError(LazyImReadError):
    """Raised when there's an issue with dimension ordering."""


class LoadOptionsError(LazyImReadError):
    """Raised when there's an issue with load options."""
