from generate.generator import Generator
import numpy as np
from typing import Tuple
from generate.utils import visualise_data, sphere_uniform, distance


class Sphere(Generator):
    def __init__(self, track_count: int, run_number: int, energy_range: Tuple[float, float] = (1, 100),
                 theta_range: Tuple[float, float] = (0, np.pi),
                 phi_range: Tuple[float, float] = (0, 2 * np.pi)) -> None:
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
        assert 0 <= theta_range[0] < theta_range[1] <= np.pi
        assert 0 <= phi_range[0] < phi_range[1] <= 2 * np.pi
        self.theta_range = theta_range
        self.phi_range = phi_range

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:

        self.data_start = np.zeros((0, 3))
        self.data_end = np.zeros((0, 3))

        radius = 16.0

        points = sphere_uniform(self.track_count, self.theta_range, self.phi_range, grad=False, radius=radius)

        self.data_start = np.array(points)

        end_theta_range = (np.pi - self.theta_range[1], np.pi - self.theta_range[0])
        end_phi_range = (2 * np.pi - self.phi_range[1], 2 * np.pi - self.phi_range[0])

        min_range = np.sqrt(2) * np.sqrt(1.0 - np.cos(120 / 180 * np.pi)) * radius

        for start in self.data_start:
            while True:
                end = sphere_uniform(1, end_theta_range, end_phi_range, grad=False,
                                     radius=radius)

                if distance(start[0], start[1], start[2], end[0][0], end[0][1], end[0][2]) >= min_range:
                    self.data_end = np.append(self.data_end, end, axis=0)
                    points = np.append(points, end, axis=0)
                    break

        shift_x = 4.5
        shift_y = 13.0
        shift_z = 4.5 - 0.3

        self.data_start[:, 0] += shift_x
        self.data_start[:, 1] += shift_y
        self.data_start[:, 2] += shift_z

        self.data_end[:, 0] += shift_x
        self.data_end[:, 1] += shift_y
        self.data_end[:, 2] += shift_z

        points[:, 0] += shift_x
        points[:, 1] += shift_y
        points[:, 2] += shift_z

        self.energy = self.generate_energy_uniform()

        # Uncomment the line below if you want to visualize the generated points
        visualise_data((points[:, 0], points[:, 1], points[:, 2]))

        return self.data_start, self.data_end

    def generate_energy(self, thetas: np.ndarray) -> np.ndarray:
        """
        Generate the energy values for a given array of theta values.

        Parameters:
            thetas (np.ndarray): The array of theta values.

        Returns:
            np.ndarray: The array of energy values corresponding to the theta values.

        Raises:
            ValueError: If any of the theta values are outside the range specified by `self.theta_range`.
        """
        if thetas.min() < self.theta_range[0] or thetas.max() > self.theta_range[1]:
            raise ValueError("Invalid theta value.")

        energy = np.empty_like(thetas, dtype=float)

        mask1 = thetas < 60
        mask2 = thetas >= 60

        energy[mask1] = np.random.uniform(low=80, high=100, size=mask1.sum())
        energy[mask2] = np.random.uniform(low=1, high=20, size=mask2.sum())

        return energy

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
