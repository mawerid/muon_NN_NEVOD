# import libraries
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Any

# define some constants
num_pts = 1
rad = 15


def generate_points_below(num_points: int,
                          max_theta_degrees: float = 55) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    max_theta = np.deg2rad(max_theta_degrees)

    num = num_points * int(max_theta_degrees) * 360

    # theta = np.random.uniform(max_theta, np.pi / 2, num_points)
    theta = np.random.uniform(0, max_theta, num)
    phi = np.random.uniform(0, 2 * np.pi, num)

    plt.figure().add_subplot().hist(np.rad2deg(theta), bins=50)
    plt.show()

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return x, y, z


def generate_points_above(num_points: int,
                          max_theta_degrees: float = 55) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    max_theta = np.deg2rad(max_theta_degrees)

    theta = np.arccos(np.sqrt(np.random.uniform(np.square(np.cos(max_theta)),
                                                1, num_points)))

    phi = np.random.uniform(0, 2 * np.pi, num_points)

    plt.figure().add_subplot().hist(np.rad2deg(theta), bins=50)
    plt.show()

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return x, y, z


def generate_random_points_sct(n, x_range, y_range, z_range):
    """
    Generate n random points in a volume with x, y, z coordinates and size.

    Parameters:
    - n: Number of points to generate.
    - x_range: Tuple (x_min, x_max) specifying the range of x coordinates.
    - y_range: Tuple (y_min, y_max) specifying the range of y coordinates.
    - z_range: Tuple (z_min, z_max) specifying the range of z coordinates.

    Returns:
    - points: Numpy array of shape (n, 3) where each row represents a point with (x, y, z).
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    x_values = np.random.uniform(x_min, x_max, n)
    y_values = np.random.uniform(y_min, y_max, n)
    z_values = np.random.uniform(z_min, z_max, n)

    points = np.column_stack((x_values, y_values, z_values))

    return points


def generate_data_sct(points_num: int) -> [np.ndarray, np.ndarray]:
    cnt_size_x = 0.23
    cnt_size_y = 0.63
    cnt_size_z = 0.055

    cnt_pos_z_up = 4.5 + 4.5  # - 0.3
    cnt_pos_z_down = -4.5 + 4.5  # - 0.3

    point_pairs = []
    upper = []
    lower = []
    points = np.array([])
    points = np.reshape(points, (0, 3))

    for j in range(5):
        for i in range(4):
            cnt_pos_x = -3 - cnt_size_x / 2 + 2 * i + 4.5
            cnt_pos_y = -7.375 - cnt_size_y / 2 + 2.5 * j + 13
            upper_points = generate_random_points_sct(points_num, (cnt_pos_x, cnt_pos_x + cnt_size_x),
                                                      (cnt_pos_y, cnt_pos_y + cnt_size_y),
                                                      (cnt_pos_z_up, cnt_pos_z_up + cnt_size_z))
            lower_points = generate_random_points_sct(points_num, (cnt_pos_x, cnt_pos_x + cnt_size_x),
                                                      (cnt_pos_y, cnt_pos_y + cnt_size_y),
                                                      (cnt_pos_z_down, cnt_pos_z_down + cnt_size_z))
            upper.append(upper_points)
            lower.append(lower_points)

            points = np.vstack((points, upper_points))
            points = np.vstack((points, lower_points))

            cnt_pos_x = -4 - cnt_size_x / 2 + 2 * i + 4.5
            cnt_pos_y = -6.125 - cnt_size_y / 2 + 2.5 * j + 13
            upper_points = generate_random_points_sct(points_num, (cnt_pos_x, cnt_pos_x + cnt_size_x),
                                                      (cnt_pos_y, cnt_pos_y + cnt_size_y),
                                                      (cnt_pos_z_up, cnt_pos_z_up + cnt_size_z))
            lower_points = generate_random_points_sct(points_num, (cnt_pos_x, cnt_pos_x + cnt_size_x),
                                                      (cnt_pos_y, cnt_pos_y + cnt_size_y),
                                                      (cnt_pos_z_down, cnt_pos_z_down + cnt_size_z))
            upper.append(upper_points)
            lower.append(lower_points)

            points = np.vstack((points, upper_points))
            points = np.vstack((points, lower_points))

    # print(points.shape)

    for up in upper:
        for low in lower:
            point_pairs.append((up, low))

    # return np.array(point_pairs)
    return point_pairs


# define functions
def generate_data(size: int) -> tuple[Any, Any, Any]:
    """
    generate evenly distributed 3D dataset on sphere (by Golden section)
    :param size: amount of points to generate
    :return: 3 np arrays (x, y ,z) respectively
    """
    indices = np.arange(0, size, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / size)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    return np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)


def visualise_data(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """
    Visualise dataset
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :return: None
    """
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    c = (z <= 0)
    ax.scatter(x, y, z, c=c, cmap='coolwarm')
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    ax.set_zlabel("Z, м")
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()


def load_data(file_path):
    """
    Load dataset from a text file into a NumPy array.

    Parameters:
    - file_path: Path to the text file containing the dataset.

    Returns:
    - data_array: NumPy array containing the loaded dataset.
    """
    data_array = np.loadtxt(file_path)

    data_array = data_array[:, 2:]

    data_array = np.vstack((data_array[:, :3], data_array[:, 3:]))

    return data_array


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
    Save dataset as txt file
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
        file.close()


def save_data_sct(pairs: np.ndarray) -> None:
    prefix = "200\t"
    event_count = 1
    with open("../data_sim/OTDCR_200.txt", "w") as file:
        for pair in pairs:
            for up, low in zip(pair[0], pair[1]):
                file.write(
                    prefix + str(event_count) + '\t' + str(round(up[0], 5)) + '\t' + str(
                        round(up[1], 5)) + '\t' + str(round(up[2], 5)) + '\t' + str(round(low[0], 5)) + '\t' + str(
                        round(low[1], 5)) + '\t' + str(round(low[2], 5)) + '\n')
                event_count += 1
        file.close()


def main() -> int:
    # x_below, y_below, z_below = generate_points_below(num_pts)
    # x_above, y_above, z_above = generate_points_above(num_pts)  

    # x, y, z = generate_points_below(num_pts)
    # x, y, z = generate_points_above(num_pts)
    # x, y, z = generate_data(num_pts)
    # x, y, z = rad * x, rad * y, rad * z
    # x, y, z = np.vstack((x, -x)), np.vstack((y, -y)), np.vstack((z, -z))
    # x -= 3.5
    # y -= 13
    # z -= 4.5 - 0.3

    points_pair = generate_data_sct(100)
    print(points_pair[:10])
    # points_pair += 2 * np.array([4.5, 13, 4.2])
    print(points_pair[:10])
    # print(points_pair.shape)

    # file_path = "OTDCR_935.txt"
    # loaded_data = load_data(file_path)
    # loaded_data = np.vstack((points_pair, loaded_data))
    # visualise_data(loaded_data[:, 0], loaded_data[:, 1], loaded_data[:, 2])

    # visualise_data(points_pair[:, 0], points_pair[:, 1], points_pair[:, 2])

    save_data_sct(points_pair)

    # visualise_data(x, y, z)

    # save_data(x, y, z)

    return 0


if __name__ == "__main__":
    main()
