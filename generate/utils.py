import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union


def distance(x0: float, y0: float, z0: float,
             x1: float, y1: float, z1: float) -> float:
    """
    Calculate the Euclidean distance between two points in 3D space.

    Parameters:
        x0 (float): The x-coordinate of the first point.
        y0 (float): The y-coordinate of the first point.
        z0 (float): The z-coordinate of the first point.
        x1 (float): The x-coordinate of the second point.
        y1 (float): The y-coordinate of the second point.
        z1 (float): The z-coordinate of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """

    return np.square(x0 - x1) + np.square(y0 - y1) + np.square(z0 - z1)


def sphere_uniform(points_count: int,
                   theta_range: Tuple[float, float],
                   phi_range: Tuple[float, float],
                   grad: bool = True,
                   radius: float = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate points uniformly distributed on a sphere.

    This function generates points uniformly distributed on a sphere. The sphere is centered at the origin and
    has a given radius. The points are evenly distributed in the spherical coordinates (theta, phi).

    Parameters:
        points_count (int): The number of points to generate.
        theta_range (Tuple[float, float]): The range of theta values in radians. (0, pi)
        phi_range (Tuple[float, float]): The range of phi values in radians. (0, 2pi)
        grad (bool, optional): Whether to convert the input ranges from degrees to radians. Defaults to True.
        radius (float, optional): The radius of the sphere. Defaults to 15.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The x, y, z coordinates of the generated points.
    """

    theta_min, theta_max = theta_range
    phi_min, phi_max = phi_range

    if grad:
        theta_min = np.deg2rad(theta_min)
        theta_max = np.deg2rad(theta_max)
        phi_min = np.deg2rad(phi_min)
        phi_max = np.deg2rad(phi_max)

    assert 0 <= theta_min < theta_max <= 2 * np.pi
    assert 0 <= phi_min < phi_max <= np.pi
    assert radius > 0
    assert points_count > 0

    theta = np.random.uniform(theta_min, theta_max, points_count)
    cos_phi = np.random.uniform(np.cos(phi_min), np.cos(phi_max), points_count)

    x = np.sqrt(1 - np.square(cos_phi)) * np.cos(theta) * radius
    y = np.sqrt(1 - np.square(cos_phi)) * np.sin(theta) * radius
    z = cos_phi * radius

    return np.c_[x, y, z]


def cube_uniform(points_count: int,
                 x_range: Tuple[float, float],
                 y_range: Tuple[float, float],
                 z_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a uniform distribution of points within a cubic volume.

    Parameters:
        points_count (int): The number of points to generate.
        x_range (Tuple[float, float]): The range of x-coordinates for the points.
        y_range (Tuple[float, float]): The range of y-coordinates for the points.
        z_range (Tuple[float, float]): The range of z-coordinates for the points.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three numpy arrays representing
        the x, y, and z coordinates of the generated points.
    """

    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    assert x_min < x_max
    assert y_min < y_max
    assert z_min < z_max

    x = np.random.uniform(x_min, x_max, points_count)
    y = np.random.uniform(y_min, y_max, points_count)
    z = np.random.uniform(z_min, z_max, points_count)

    return np.c_[x, y, z]


def visualise_data(data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                   x: np.ndarray = None, y: np.ndarray = None, z: np.ndarray = None,
                   divide_start_end: bool = True, divider_mask: np.array = None) -> None:
    """
    Visualize the given data points in a 3D scatter plot.

    Parameters:
        data (np.ndarray, optional): The data points to be visualized. It should be a 2D array with shape (n, 3), where n is the number of data points and each row represents the x, y, and z coordinates of a data point. Defaults to None.
        x (np.ndarray, optional): The x coordinates of the data points. It should be a 1D array with shape (n) where n is the number of data points. Defaults to None.
        y (np.ndarray, optional): The y coordinates of the data points. It should be a 1D array with shape (n) where n is the number of data points. Defaults to None.
        z (np.ndarray, optional): The z coordinates of the data points. It should be a 1D array with shape (n) where n is the number of data points. Defaults to None.
        divide_start_end (bool, optional): Whether to divide the data points into two categories based on the sign of the z-coordinate. Defaults to True.
        divider_mask (np.array, optional): The mask used to divide the data points. It should be a 1D array with shape (n) where n is the number of data points. Defaults to None.

    Returns:
        None

    This function visualizes the given data points in a 3D scatter plot. It takes in the data points as either a 2D array or individual x, y, and z coordinates. If the data array is provided, it extracts the x, y, and z coordinates from it. The function then creates a 3D scatter plot using matplotlib and sets the labels for the x, y, and z axes. It also adds a fake bounding box to the plot for visualization purposes. The plot is displayed using plt.show().
    """

    if x is None or y is None or z is None:
        if isinstance(data, np.ndarray):
            assert data.shape[1] == 3
            x, y, z = data[:, 0], data[:, 1], data[:, 2]
        elif isinstance(data, tuple) and len(data) == 3:
            x, y, z = data
        else:
            raise ValueError("Invalid data type")

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    if divide_start_end:
        if divider_mask is None:
            divider_mask = (z <= 5)
        ax.scatter(x, y, z, c=divider_mask, cmap='coolwarm')
    else:
        ax.scatter(x, y, z)

    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    ax.set_zlabel("Z, м")
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
    x_b = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
    y_b = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
    z_b = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(x_b, y_b, z_b):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()
