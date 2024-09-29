# LazyImRead

LazyImRead is a Python library for lazy loading and processing of various image file formats, including TIFF, HDF5, Zarr, and video files. It provides efficient handling of large multi-dimensional image datasets with minimal memory footprint.

![CI](https://github.com/your-username/lazyimread/workflows/CI/badge.svg)
[![PyPI version](https://badge.fury.io/py/lazyimread.svg)](https://badge.fury.io/py/lazyimread)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)
[![Python Versions](https://img.shields.io/pypi/pyversions/lazyimread.svg)](https://pypi.org/project/lazyimread/)

## Features:

- Lazy loading of large image datasets
- Support for common scientific image formats: TIFF, HDF5, Zarr, and video files
- Automatic dimension order detection and rearrangement
- Configurable partial loading of datasets
- Easy-to-use API for loading and saving multi-dimensional image data
- Memory-efficient processing of large datasets

## Installation:

You can install LazyImRead using pip:

```bash
pip install lazyimread

# Install from GitHub
pip install git+https://github.com/your-username/lazyimread.git
```

For development installation, clone the repository and install in editable mode:

```bash
git clone https://github.com/your-username/lazyimread.git
cd lazyimread
pip install -e .
```

## Usage Examples:

### 1. Basic loading:

```python
from lazyimread import load

data, dim_order, metadata = load('path/to/your/file.tiff')
```

### 2. Configuring load options:

```python
from lazyimread import configure_load_options, load

options = configure_load_options(t_range=(0, 10), z_range=(5, 15), target_order='TZYXC')
data, dim_order, metadata = load('path/to/your/file.h5', options)
```

### 3. Rearranging dimensions:

```python
from lazyimread import load, rearrange_dimensions

data, dim_order, metadata = load('path/to/your/file.zarr')
rearranged_data, new_order = rearrange_dimensions(data, dim_order, 'TZCYX')
```

### 4. Saving data:

```python
from lazyimread import save_tiff
save_tiff(data, 'output.tiff', dim_order='TZCYX')
```

## Key Features:

LazyImRead offers several key features for working with multi-dimensional image data:

1. **Partial loading**: Load only the necessary portions of large datasets, saving memory and processing time.
2. **Automatic format detection**: The library automatically detects the input format and uses the appropriate loader.
3. **Dimension order handling**: Easily rearrange and work with different dimension orders (e.g., TZCYX, XYZCT).
4. **Lazy evaluation**: Operations are performed only when needed, allowing for efficient processing of large datasets.
5. **Common file format support**: Work seamlessly with popular scientific image formats used in microscopy and biomedical imaging.

LazyImRead simplifies the process of working with complex multi-dimensional image data, making it an invaluable tool for researchers and developers in fields such as microscopy, medical imaging, and computer vision.
