"""Utility functions for handling image data dimensions."""

from dataclasses import dataclass
from enum import Enum, auto
from logging import getLogger

import numpy as np

logger = getLogger(__name__)


class ReturnOrder(Enum):
    """Enum class to specify the return order of data processing functions.

    :param DATA_ONLY: Return only the processed data.
    :param DATA_AND_ORDER: Return both the processed data and its dimension order.
    """

    DATA_ONLY = auto()
    DATA_AND_ORDER = auto()


class DefaultDimensionOrder(Enum):
    """Enum class to represent various dimension orders for image data.

    :param TZXYC: Time, Z-stack, X, Y, Channel
    :param ZXYC: Z-stack, X, Y, Channel
    :param TXYC: Time, X, Y, Channel
    :param TXY: Time, X, Y
    :param ZXY: Z-stack, X, Y
    :param XYC: X, Y, Channel
    :param XY: X, Y
    """

    TZXYC = "TZXYC"
    ZXYC = "ZXYC"
    TXYC = "TXYC"
    TXY = "TXY"
    ZXY = "ZXY"
    XYC = "XYC"
    XY = "XY"

    def __str__(self) -> str:
        """Return the string representation of the enum value."""
        return self.value


@dataclass
class DimensionRule:
    """Class to represent a dimension rule for image data."""

    max_channels: int = 32
    min_xy: int = 1024
    min_time: int = 33
    min_z: int = 64


def translate_dimension_names(order: str) -> str:
    """Translate F, D, W, H to T, Z, X, Y respectively.

    Ensure the output contains only one each of X, Y, Z, C, T.

    :param order: Input dimension order string
    :return: Translated dimension order string
    """
    translation = {"F": "T", "D": "Z", "W": "X", "H": "Y"}
    seen: set[str] = set()
    translated: list[str] = []

    for char in order:
        new_char = translation.get(char, char)
        if new_char in "XYZCT":
            if new_char in seen:
                logger.error(f"Duplicate dimension '{new_char}' found in order string")
                raise ValueError(f"Duplicate dimension '{new_char}' found in order string")
            seen.add(new_char)
            translated.append(new_char)

    return "".join(translated)


def predict_dimension_order(
    data: np.ndarray | tuple[int, ...], rule: DimensionRule | None = None
) -> str:
    """Predict the original dimension order based on array shape.

    This function uses the following rules to predict the dimension order:
    - 2D: Always assumed to be XY
    - 3D:
        - If last dimension <= max_channels, assumed to be XYC
        - Else if first dimension >= min_time, assumed to be TXY
        - Otherwise, assumed to be ZXY
    - 4D:
        - If last dimension <= max_channels:
            - If first dimension >= min_time, assumed to be TXYC
            - Otherwise, assumed to be ZXYC
        - Else, assumed to be TZXY
    - 5D: Always assumed to be TZXYC

    If your data does not fit these rules, you can use the `rearrange_dimensions`
    function to rearrange the dimensions into the desired order.

    :param data: Input data as numpy array or tuple of shape
    :param rule: DimensionRule object with threshold values, if None, default values are used
    :return: Predicted dimension order as a string
    :raises ValueError: If the number of dimensions is not supported (not 2-5)
    """
    shape = data.shape if isinstance(data, np.ndarray) else data
    dims = len(shape)
    rule = rule or DimensionRule()

    if dims == 2:
        return "XY"
    elif dims == 3:
        if shape[-1] <= rule.max_channels:
            logger.debug("Predicting dimension order: XYC")
            return "XYC"
        elif shape[0] >= rule.min_time:
            logger.debug("Predicting dimension order: TXY")
            return "TXY"
        else:
            logger.debug("Predicting dimension order: ZXY")
            return "ZXY"
    elif dims == 4:
        if shape[-1] <= rule.max_channels:
            if shape[0] >= rule.min_time:
                logger.debug("Predicting dimension order: TXYC")
                return "TXYC"
            else:
                logger.debug("Predicting dimension order: ZXYC")
                return "ZXYC"
        else:
            logger.debug("Predicting dimension order: TZXY")
            return "TZXY"
    elif dims == 5:
        logger.debug("Predicting dimension order: TZXYC")
        return "TZXYC"
    else:
        logger.error(f"Unsupported number of dimensions: {dims}")
        raise ValueError(f"Unsupported number of dimensions: {dims}")


def rearrange_dimensions(
    data: np.ndarray,
    current_order: str,
    target_order: str | None = None,
    return_order: ReturnOrder = ReturnOrder.DATA_AND_ORDER,
) -> np.ndarray | tuple[np.ndarray, str]:
    """Rearrange the dimensions of the input data to the desired order.

    :param data: Input data array
    :param current_order: Current dimension order of the data
    :param target_order: Desired dimension order. If None, use default order
    :param return_order: Enum specifying whether to return only data or data and order
    :return: Rearranged data array and final order (if return_order is DATA_AND_ORDER)
    """
    current_dims = data.ndim
    logger.debug(f"The shape of the input data is {data.shape}")

    if target_order is None:
        target_order = DefaultDimensionOrder[current_order].value
        logger.debug(f"No target order specified, using default order: {target_order}")
    else:
        target_order = translate_dimension_names(target_order)

    if len(target_order) != current_dims:
        logger.error(
            f"Target order '{target_order}' does not match input dimensions {current_dims}"
        )
        raise ValueError(
            f"Target order '{target_order}' does not match input dimensions {current_dims}"
        )

    logger.debug(f"The current order is {current_order} and the target is {target_order}")
    transpose_axes = [current_order.index(dim) for dim in target_order]
    logger.debug(f"Transposing axes: {transpose_axes}")

    rearranged_data = np.transpose(data, transpose_axes)
    logger.debug(f"The shape of the rearranged data is {rearranged_data.shape}")

    return (
        (rearranged_data, target_order)
        if return_order == ReturnOrder.DATA_AND_ORDER
        else rearranged_data
    )
