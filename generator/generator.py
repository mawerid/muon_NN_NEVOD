import os
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
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The x, y, and z coordinates of the generated points.
    """

    theta_min, theta_max = theta_range
    phi_min, phi_max = phi_range

    if grad:
        theta_min = np.deg2rad(theta_min)
        theta_max = np.deg2rad(theta_max)
        phi_min = np.deg2rad(phi_min)
        phi_max = np.deg2rad(phi_max)

    assert 0 <= theta_min < theta_max <= np.pi
    assert 0 <= phi_min < phi_max <= 2 * np.pi
    assert radius > 0
    assert points_count > 0

    phi = np.random.uniform(phi_min, phi_max, points_count)
    theta = np.random.uniform(np.cos(theta_min), np.cos(theta_max), points_count)

    x = np.sqrt(1 - np.square(theta)) * np.cos(phi) * radius
    y = np.sqrt(1 - np.square(theta)) * np.sin(phi) * radius
    z = theta * radius

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
                   divide_start_end: bool = True) -> None:
    """
    Visualize the given data points in a 3D scatter plot.

    Parameters:
        data (np.ndarray, optional): The data points to be visualized. It should be a 2D array with shape (n, 3), where n is the number of data points and each row represents the x, y, and z coordinates of a data point. Defaults to None.
        x (np.ndarray, optional): The x coordinates of the data points. It should be a 1D array with shape (n) where n is the number of data points. Defaults to None.
        y (np.ndarray, optional): The y coordinates of the data points. It should be a 1D array with shape (n) where n is the number of data points. Defaults to None.
        z (np.ndarray, optional): The z coordinates of the data points. It should be a 1D array with shape (n) where n is the number of data points. Defaults to None.
        divide_start_end (bool, optional): Whether to divide the data points into two categories based on the sign of the z-coordinate. Defaults to True.

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
        c = (z <= 0)
        ax.scatter(x, y, z, c=c, cmap='coolwarm')
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


class Generator:
    def __init__(self, track_count: int, energy_range: Tuple[float, float], run_number: int) -> None:
        """
        Initializes a new instance of the Generator class.

        Parameters:
            track_count (int): The number of tracks to generate. Must be greater than 0.
            energy_range (Tuple[float, float]): The range of energies for the tracks. Must be a tuple with the minimum and maximum energy values. The minimum energy value must be greater than 0 and less than the maximum energy value.
            run_number (int): The run number for the generator. Must be greater than 0.

        Returns:
            None
        """

        assert track_count > 0
        assert run_number > 0
        assert 0 < energy_range[0] < energy_range[1]

        self.track_count = track_count
        self.energy_range = energy_range
        self.run_number = run_number
        self.energy = None
        self.data_start = None
        self.data_end = None

    def energy_uniform(self) -> np.ndarray:
        """
        Generates an array of uniformly distributed random numbers from the energy range specified in the Generator instance.

        Returns:
            np.ndarray: An array of uniformly distributed random numbers with shape (track_count,) from the energy range specified in the Generator instance.

        Raises:
            ValueError: If the energy range is not set.
            ValueError: If the energy range is invalid (minimum energy is not greater than 0 or maximum energy is not greater than minimum energy).
        """
        if self.energy_range is None:
            raise ValueError("Energy range is not set.")

        energy_min, energy_max = self.energy_range

        if not (0 < energy_min < energy_max):
            raise ValueError("Invalid energy range.")

        return np.random.uniform(energy_min, energy_max, self.track_count)

    def energy_differential(self,
                            theta: float, phi: float) -> float:
        pass

    def generate_sphere(self,
                        theta_range: Tuple[float, float],
                        phi_range: Tuple[float, float],
                        radius: float = 15) -> Tuple[np.ndarray, np.ndarray]:

        pass

    def generate_sct(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate track from SCT by randomly distributing points within a set of cubic volumes.

        This function generates a scattering cross-section by randomly distributing points within a set of cubic volumes. The cubic volumes are defined by their size and position in the x, y, and z directions. The function takes no parameters, but relies on the instance variables of the class to determine the number of points to generate, the size and position of the cubic volumes, and the range of track counts.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays representing the start and end positions of the generated tracks.
        """

        self.data_start = np.zeros((0, 3))
        self.data_end = np.zeros((0, 3))

        cnt_size_x = 0.2  # 0.23
        cnt_size_y = 0.4  # 0.63
        cnt_size_z = 0.02  # 0.055

        cnt_pos_z_up = 9.0  # 4.5 + 4.5  # - 0.3
        cnt_pos_z_down = 0.0  # -4.5 + 4.5  # - 0.3

        points_num = self.track_count // 1600
        points = np.zeros((0, 3))

        for j in range(5):
            for i in range(4):
                cnt_pos_x = -3 - cnt_size_x / 2 + 2 * i + 4.5
                cnt_pos_y = -7.375 - cnt_size_y / 2 + 2.5 * j + 13

                upper_points = cube_uniform(points_num, (cnt_pos_x, cnt_pos_x + cnt_size_x),
                                            (cnt_pos_y, cnt_pos_y + cnt_size_y),
                                            (cnt_pos_z_up, cnt_pos_z_up + cnt_size_z))
                lower_points = cube_uniform(points_num, (cnt_pos_x, cnt_pos_x + cnt_size_x),
                                            (cnt_pos_y, cnt_pos_y + cnt_size_y),
                                            (cnt_pos_z_down, cnt_pos_z_down + cnt_size_z))

                self.data_start = np.vstack((self.data_start, upper_points))
                self.data_end = np.vstack((self.data_end, lower_points))
                points = np.vstack((points, upper_points, lower_points))

        for j in range(4):
            for i in range(5):
                cnt_pos_x = -4 - cnt_size_x / 2 + 2 * i + 4.5
                cnt_pos_y = -6.125 - cnt_size_y / 2 + 2.5 * j + 13

                upper_points = cube_uniform(points_num, (cnt_pos_x, cnt_pos_x + cnt_size_x),
                                            (cnt_pos_y, cnt_pos_y + cnt_size_y),
                                            (cnt_pos_z_up, cnt_pos_z_up + cnt_size_z))
                lower_points = cube_uniform(points_num, (cnt_pos_x, cnt_pos_x + cnt_size_x),
                                            (cnt_pos_y, cnt_pos_y + cnt_size_y),
                                            (cnt_pos_z_down, cnt_pos_z_down + cnt_size_z))

                self.data_start = np.vstack((self.data_start, upper_points))
                self.data_end = np.vstack((self.data_end, lower_points))
                points = np.vstack((points, upper_points, lower_points))

        # Uncomment the line below if you want to visualize the generated points
        visualise_data((points[:, 0], points[:, 1], points[:, 2]))

        return self.data_start, self.data_end

    def generate_decor(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def save_data(self, save_energy: bool = True) -> None:
        """
        Save the data to a file.

        Parameters:
            save_energy (bool): Flag to indicate whether to save energy data. Default is True.

        Returns:
            None
        """

        file_name = f"../data_sim/OTDCR_{self.run_number}.txt"

        with open(file_name, 'w') as file:
            i = 0

            for start in self.data_start:
                for end in self.data_end:
                    line = f"{self.run_number}\t{i}\t"
                    if save_energy:
                        line += f"{np.round(self.energy[i], 5)}\t"
                    line += f"{np.round(start[0], 5)}\t{np.round(start[1], 5)}\t{np.round(start[2], 5)}\t"
                    line += f"{np.round(end[0], 5)}\t{np.round(end[1], 5)}\t{np.round(end[2], 5)}\n"
                    file.write(line)
                    i += 1

    def load_data(self, file_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from a file and parse it into a NumPy array.

        Args:
            file_name (str): The path to the file containing the data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays. The first array represents the data start,
            with shape (n_tracks, 3), and the second array represents the data end, with shape (n_tracks, 3).

        Raises:
            FileNotFoundError: If the file does not exist.
        """

        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File {file_name} does not exist.")

        data_array = np.loadtxt(file_name, delimiter='\t')

        self.run_number = data_array[0][0]
        self.track_count = data_array.shape[0]

        if data_array.shape[1] == 8:
            self.energy = None
            data_array = data_array[:, 2:]
        elif data_array.shape[1] == 9:
            self.energy = data_array[:, 1]
            data_array = data_array[:, 2:]
        else:
            raise ValueError("Invalid data shape.")

        data_start = data_array[:, :3]
        data_end = data_array[:, 3:]

        return data_start, data_end


if __name__ == "__main__":
    gen = Generator(1600, (90, 100), 1)
    gen.generate_sct()
    gen.save_data(save_energy=False)

    # visualise_data(
    #     sphere_uniform(1000, (0, 180), (0, 360)))
    # cube_uniform(1000, (0, 360), (0, 180), (0, 90)))
