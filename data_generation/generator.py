# import libraries
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

# define some constants
num_pts = 5000
rad = 15


# define functions
def generate_data(size: int) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    generate evenly distributed 3D data on sphere (by Golden section)
    :param size: amount of points to generate
    :return: 3 np arrays (x, y ,z) respectively
    """
    indices = np.arange(0, size, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / size)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    return np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)


def visualise_data(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """
    Visualise data
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :return: None
    """
    plt.figure().add_subplot(111, projection='3d').scatter(x, y, z)
    plt.show()


def distance(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> float:
    """
    Calculate distance between two points
    :param x0: x coordinate of current point
    :param y0: y coordinate of current point
    :param z0: z coordinate of current point
    :param x1: x coordinate of current point
    :param y1: y coordinate of current point
    :param z1: z coordinate of current point
    :return: distance between points
    """
    return (x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2


def save_data(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """
    Save data as txt file
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :return: None
    """
    prefix = "100\t"
    min_distance = 450.0
    event_count = 1
    with open("OTDCR_100.txt", "w") as file:
        for i in range(num_pts):
            np.random.seed(13)
            count = 10
            while count > 0:
                index = np.random.randint(0, 5000)
                dist = distance(x[i], y[i], z[i], x[index], y[index], z[index])
                if dist > min_distance:
                    file.write(
                        prefix + str(event_count) + '\t' + str(round(x[i], 5)) + '\t' + str(
                            round(y[i], 5)) + '\t' + str(round(z[i], 5)) + '\t' + str(round(x[index], 5)) + '\t' + str(
                            round(y[index], 5)) + '\t' + str(round(z[index], 5)) + '\n')
                    count -= 1
                    event_count += 1
                    file.write(
                        prefix + str(event_count) + '\t' + str(round(x[index], 5)) + '\t' + str(
                            round(y[index], 5)) + '\t' + str(round(z[index], 5)) + '\t' + str(
                            round(x[i], 5)) + '\t' + str(round(y[i], 5)) + '\t' + str(round(z[i], 5)) + '\n')
                    event_count += 1


def main() -> int:
    x, y, z = generate_data(num_pts)
    x, y, z = rad * x, rad * y, rad * z

    visualise_data(x, y, z)
    save_data(x, y, z)

    return 0


if __name__ == "__main__":
    main()
