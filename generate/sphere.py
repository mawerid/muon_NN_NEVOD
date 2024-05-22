from generate.generator import Generator
import numpy as np
from typing import Tuple
from generate.utils import visualise_data, sphere_uniform


class Sphere(Generator):
    def __init__(self, track_count: int, run_number: int, energy_range: Tuple[float, float] = (1, 100)) -> None:
        """
        Initializes a new instance of the Sphere generator class.

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
        Generate track from sphere by randomly distributing points within a set of cubic volumes.

        This function generates a scattering cross-section by randomly distributing points within a set of cubic volumes. The cubic volumes are defined by their size and position in the x, y, and z directions. The function takes no parameters, but relies on the instance variables of the class to determine the number of points to generate, the size and position of the cubic volumes, and the range of track counts.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays representing the start and end positions of the generated tracks.
        """

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

            for i in range(self.track_count):
                line = f"{self.run_number}\t{i}\t"
                if save_energy:
                    line += f"{np.round(self.energy[i], 5)}\t"
                line += f"{np.round(self.data_start[i][0], 5)}\t{np.round(self.data_start[i][1], 5)}\t{np.round(self.data_start[i][2], 5)}\t"
                line += f"{np.round(self.data_end[i][0], 5)}\t{np.round(self.data_end[i][1], 5)}\t{np.round(self.data_end[i][2], 5)}\n"
                file.write(line)

