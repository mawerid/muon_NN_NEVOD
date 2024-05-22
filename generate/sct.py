from generate.generator import Generator
import numpy as np
from typing import Tuple
from generate.utils import visualise_data, cube_uniform


class Sct(Generator):
    def __init__(self, track_count: int, run_number: int, energy_range: Tuple[float, float] = (1, 20)) -> None:
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

        points_num = np.sqrt(self.track_count // 1600).astype(int)
        print(points_num)
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
            i = 0

            print(self.data_start.shape)
            print(self.data_end.shape)
            print(self.energy.shape)

            for start in self.data_start:
                for end in self.data_end:
                    line = f"{self.run_number}\t{i}\t"
                    if save_energy:
                        line += f"{np.round(self.energy[i], 5)}\t"
                    line += f"{np.round(start[0], 5)}\t{np.round(start[1], 5)}\t{np.round(start[2], 5)}\t"
                    line += f"{np.round(end[0], 5)}\t{np.round(end[1], 5)}\t{np.round(end[2], 5)}\n"
                    file.write(line)
                    i += 1
