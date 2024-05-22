import numpy as np
import pandas as pd
from typing import Union, Tuple


def sphere_project(point: Union[np.ndarray, pd.DataFrame, Tuple[float, float, float]]) -> np.ndarray:
    """
    Project a set of points on a sphere onto a two-dimensional plane.

    Parameters:
        point (Union[np.ndarray, pd.DataFrame, Tuple[float, float, float]]): The points to be projected. It can be a NumPy array, a Pandas DataFrame, or a tuple of three floats.

    Returns:
        np.ndarray: The projected points on the two-dimensional plane. The array has shape (n, 2), where n is the number of input points.

    Raises:
        ValueError: If the input point has a shape other than (n, 3).

    """
    if isinstance(point, pd.DataFrame):
        point = point.values
    elif isinstance(point, tuple):
        point = np.array(point)

    if point.size == 0:
        return np.empty((0, 2))

    point = np.atleast_2d(point)
    if point.shape[1] != 3:
        raise ValueError("Input point must have shape (n, 3)")

    norm = np.linalg.norm(point, axis=1)
    theta = np.arccos(point[:, 2] / norm)
    phi = np.arctan2(point[:, 1], point[:, 0])

    return np.c_[theta, phi]


def calc_direction(start: Union[np.ndarray, pd.DataFrame, Tuple[float, float, float]],
                   end: Union[np.ndarray, pd.DataFrame, Tuple[float, float, float]]) -> np.ndarray:
    """
    Calculate the direction vector from the start point to the end point.

    Parameters:
        start (Union[np.ndarray, pd.DataFrame, Tuple[float, float, float]]): The starting point. It can be a NumPy array, a Pandas DataFrame, or a tuple of three floats.
        end (Union[np.ndarray, pd.DataFrame, Tuple[float, float, float]]): The ending point. It can be a NumPy array, a Pandas DataFrame, or a tuple of three floats.

    Returns:
        np.ndarray: The direction vector from the start point to the end point. The direction vector is a NumPy array with the same shape as the input points.

    Raises:
        ValueError: If the start and end points have different shapes.

    """
    if isinstance(start, (pd.DataFrame, tuple)):
        start = np.array(start)
    if isinstance(end, (pd.DataFrame, tuple)):
        end = np.array(end)
    if start.shape != end.shape:
        raise ValueError("Start and end must have the same shape.")

    direction = end - start
    direction_sum = np.sum(direction)
    direction_norm = np.linalg.norm(direction)

    direction[np.isclose(direction_sum, 0)] = 0
    direction[~np.isclose(direction_sum, 0)] /= direction_norm

    return direction
