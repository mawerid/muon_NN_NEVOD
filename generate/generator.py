import numpy as np
import os
from typing import Tuple
from abc import ABC, abstractmethod

from generate.utils import visualise_data


class Generator(ABC):
    def __init__(self, track_count: int, run_number: int, energy_range: Tuple[float, float]) -> None:
        """
        Initializes a new instance of the Generator class.

        Parameters:
            track_count (int): The number of tracks to generate. Must be greater than 0.
            run_number (int): The run number for the generator. Must be greater than 0.
            energy_range (Tuple[float, float]): The range of energies for the tracks. Must be a tuple with the minimum and maximum energy values. The minimum energy value must be greater than 0 and less than the maximum energy value.

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

    @abstractmethod
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def generate_energy_uniform(self) -> np.ndarray:
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

    def generate_energy_differential(self,
                                     theta: float, phi: float) -> float:
        pass

    @abstractmethod
    def save(self, save_energy: bool = True) -> None:
        pass

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
        self.track_count += data_array.shape[0]

        if data_array.shape[1] == 8:
            self.energy = None
            data_array = data_array[:, 2:]
        elif data_array.shape[1] == 9:
            self.energy = data_array[:, 1]
            data_array = data_array[:, 2:]
        else:
            raise ValueError("Invalid data shape.")

        self.data_start = np.r_[self.data_start, data_array[:, :3]]
        self.data_end = np.r_[self.data_end, data_array[:, 3:]]

        visualise_data(np.r_[self.data_start, self.data_end])

        data_start = data_array[:, :3]
        data_end = data_array[:, 3:]

        return data_start, data_end
