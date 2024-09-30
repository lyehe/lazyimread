"""Asynchronous loading functionality."""

import asyncio
from logging import getLogger
from pathlib import Path

import numpy as np

from .lazyimread import (
    DataLoaderFactory,
    FileFormatError,
    LazyImReadError,
    LoadOptions,
    predict_dimension_order,
    rearrange_dimensions,
)

logger = getLogger(__name__)


async def aload(
    input_path: Path,
    options: LoadOptions | None = None,
) -> tuple[np.ndarray, str, dict | None]:
    """Asynchronously load input data from various file formats.

    :param input_path: Path to the input file
    :param options: LoadOptions instance with loading parameters
    :return: Tuple of (data, dimension order, metadata)
    """
    logger.info(f"Asynchronously loading input from {input_path}")
    options = options or LoadOptions()

    try:
        loader = DataLoaderFactory.get_loader(input_path)
        data, current_order, metadata = await asyncio.to_thread(loader.load, input_path, options)
    except FileFormatError:
        raise
    except Exception as err:
        logger.exception(f"Error loading data: {err}")
        raise LazyImReadError(f"Error loading data: {err}") from err

    if options.dim_order:
        current_order = options.dim_order
    elif not current_order:
        current_order = predict_dimension_order(data)
        logger.info(f"Predicted input dimension order: {current_order}")

    if options.target_order:
        logger.info(f"Rearranging dimensions from {current_order} to {options.target_order}")
        try:
            data, final_order = await asyncio.to_thread(
                rearrange_dimensions, data, current_order, options.target_order
            )
        except Exception as err:
            logger.exception(f"Error rearranging dimensions: {err}")
            raise LazyImReadError(f"Error rearranging dimensions: {err}") from err
    else:
        final_order = current_order

    return data, str(final_order), metadata
