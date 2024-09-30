# Lazyimread

![CI](https://github.com/lyehe/lazyimread/workflows/CI/badge.svg)
[![PyPI version](https://badge.fury.io/py/lazyimread.svg)](https://badge.fury.io/py/lazyimread)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![Python Versions](https://img.shields.io/pypi/pyversions/lazyimread.svg)](https://pypi.org/project/lazyimread/)

ND image data loaders generally works great, but you often need to install them for specific data types and projects. Then, you have to make sure the data is in the desired dimentional order (for each type of data). I don't know about you, but I wasted a lot of 10 minutes on this type of tasks.

Lazyimread is a Python library that simplifies working with large, multi-dimensional image datasets. Using a single function call (e.g., `load` & `imread`), it can handle importing of various image file formats such as TIFF, HDF5, Zarr, image sequences, and video files without writing boilerplate code for each format. It handles 2-5D TZXYC data with a consistent API and some automation for automatic dimension order detection and rearrangement. It also includes several simple, boilerplate saving interfaces.

Whether you're dealing with microscopy data, satellite imagery, or video analysis, Lazyimread can significantly streamline your workflow and make handling complex image datasets more intuitive and efficient.

## Features:

- Using `imread`-like syntax to load all supported file formats
- Automatically detects file type and dimension order
- Configurable partial loading of datasets
- Asynchronous loading interface for queued tasks

## Warning:

Lazyimread is designed to handle common image and video formats and setups. It may not be suitable for specialized data formats and arrangements. This early development can be buggy, and the syntax of XYZCT can vary in different fields. Use at your own risk.

## Installation:

You can install LazyImRead using pip (pending release):

```bash
pip install lazyimread
```

or from GitHub:

```bash
pip install git+https://github.com/lyehe/lazyimread.git
```

For development installation, clone the repository and install in editable mode:

```bash
git clone https://github.com/lyehe/lazyimread.git
cd lazyimread
pip install -e .
```

Or feel free to copy and paste the code into your project / data analysis pipeline.

## Usage Examples:

You can find more examples in the [examples.ipynb](examples.ipynb).

### 1. Basic loading:

```python
from lazyimread import load, imread
from lazyimread import lazyload as ll

# All the same
data, dim_order, metadata = load('path/to/your/file.tiff')
data = imread('path/to/your/file.zarr') # Ignore dimension order and metadata
data, dim_order, metadata = ll('path/to/your/folder') # Folder with image files
```

### 2. Configuring load options:

Only required portions of the data are loaded to the memory.

```python
from lazyimread import imset, imread

# The loader will only load the frames between t=0-10 and z=5-15 and skip the rest
options = imset(t_range=(0, 10), z_range=(5, 15), target_order='TZYXC')
data, dim_order, metadata = imread('path/to/your/file.h5', options)
```

### 3. Rearranging dimensions:

You can rearrange the dimensions of the data to match your needs while loading.

```python
from lazyimread import load, rearrange_dimensions

# The default dimension order is TZYXC, but we can rearrange it to TCZXY
data, dim_order, metadata = load('path/to/your/file.zarr')
rearranged_data, new_order = rearrange_dimensions(data, dim_order, 'TCZYX')
```

### 4. Saving data:

This saves the data back to a file with minimal configuration.

```python
from lazyimread import save_tiff

# Save the data back to a TIFF file
save_tiff(data, 'output.tiff', dim_order='TZXYC')
```

### 5. Asynchronous loading:

This is useful for loading large datasets asynchronously while performing other tasks.

```python
from lazyimread import aload

data, dim_order, metadata = aload('path/to/your/file.tiff')
```

### 6. GUI loading:

This comes handy when you want to load data interactively.

```python
from lazyimread import gload, gdirload

data, dim_order, metadata = gload() # GUI file selector (for single files)
data, dim_order, metadata = gdirload() # GUI directory selector (for folder and zarr store)
```

### License:

This project is licensed under the CC0 1.0 Universal (CC0 1.0) Public Domain Dedication.
