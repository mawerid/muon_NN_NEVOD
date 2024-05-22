from generate.generator import Generator
import numpy as np
from typing import Tuple
from generate.utils import visualise_data, cube_uniform


class Decor(Generator):
    def __init__(self, track_count: int, run_number: int, energy_range: Tuple[float, float] = (80, 100)) -> None:
        """
        Initializes a new instance of the Sct generator class.

        Parameters:
            track_count (int): The number of tracks to generate. Must be greater than 0.
            run_number (int): The run number for the generator. Must be greater than 0.
            energy_range (Tuple[float, float]): The range of energies for the tracks. Must be a tuple with the minimum and maximum energy values. The minimum energy value must be greater than 0 and less than the maximum energy value.

        Returns:
            None
        """
        super().__init__(track_count, run_number, energy_range)

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate track from DECOR by randomly distributing points within a set of cubic volumes.

        This function generates a scattering cross-section by randomly distributing points within a set of cubic volumes. The cubic volumes are defined by their size and position in the x, y, and z directions. The function takes no parameters, but relies on the instance variables of the class to determine the number of points to generate, the size and position of the cubic volumes, and the range of track counts.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays representing the start and end positions of the generated tracks.
        """

        self.data_start = np.zeros((0, 3))
        self.data_end = np.zeros((0, 3))

        sm_size_x = 3120.0 / 1000.0
        sm_size_y = 480.0 / 1000.0
        sm_size_z = 2708.0 / 1000.0

        shift_SMx = 0 + sm_size_x / 2.0 * 1000.0
        shift_SMy = 0 + sm_size_y / 2.0 * 1000.0
        shift_SMz = 0 + sm_size_z / 2.0 * 1000.0

        sm_pos_x = ((np.array([2310.38, 6727.38, 6748.62, 2348.62]) - shift_SMx) / 1000.0).tolist()
        sm_pos_y = ((np.array([-1011, -1011, 27022, 27022]) - shift_SMy) / 1000.0).tolist()
        sm_pos_z = ((np.array([3926.62, 3923.62, 3916.62, 3919.62]) - shift_SMz) / 1000.0).tolist()

        sm_indexes = np.array([[2, 3], [2, 3], [0, 1], [0, 1]])

        points_num = self.track_count // 8
        points = np.zeros((0, 3))

        for i in range(4):
            for j in range(2):
                sm_pos_x_start = sm_pos_x[i]
                sm_pos_y_start = sm_pos_y[i]
                sm_pos_z_start = sm_pos_z[i]

                start_points = cube_uniform(points_num, (sm_pos_x_start, sm_pos_x_start + sm_size_x),
                                            (sm_pos_y_start, sm_pos_y_start + sm_size_y),
                                            (sm_pos_z_start, sm_pos_z_start + sm_size_z))

                sm_pos_x_end = sm_pos_x[sm_indexes[i][j]]
                sm_pos_y_end = sm_pos_y[sm_indexes[i][j]]
                sm_pos_z_end = sm_pos_z[sm_indexes[i][j]]

                end_points = cube_uniform(points_num, (sm_pos_x_end, sm_pos_x_end + sm_size_x),
                                          (sm_pos_y_end, sm_pos_y_end + sm_size_y),
                                          (sm_pos_z_end, sm_pos_z_end + sm_size_z))

                self.data_start = np.vstack((self.data_start, start_points))
                self.data_end = np.vstack((self.data_end, end_points))
                points = np.vstack((points, start_points, end_points))

        self.energy = self.generate_energy_uniform()

        # Uncomment the line below if you want to visualize the generated points
        visualise_data((points[:, 0], points[:, 1], points[:, 2]))

        return self.data_start, self.data_end

    def save(self, save_energy: bool = True) -> None:
        """
        Save the data to a file.

        Parameters:
            save_energy (bool): Flag to indicate whether to save energy data. Default is True.

        Returns:
            None
        """

        file_name = f"../data_sim/OTDCR_{self.run_number}.txt"

        with open(file_name, 'w') as file:
            for i in range(self.track_count):
                line = f"{self.run_number}\t{i}\t"
                if save_energy:
                    line += f"{np.round(self.energy[i], 5)}\t"
                line += f"{np.round(self.data_start[i][0], 5)}\t{np.round(self.data_start[i][1], 5)}\t{np.round(self.data_start[i][2], 5)}\t"
                line += f"{np.round(self.data_end[i][0], 5)}\t{np.round(self.data_end[i][1], 5)}\t{np.round(self.data_end[i][2], 5)}\n"
                file.write(line)
