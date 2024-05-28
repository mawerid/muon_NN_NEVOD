import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# - Подготовить класс датасета для работы с несколькими файлами (с единым форматом входным)


class CwdDataset(Dataset):
    def __init__(self, data_dir: str, answer_dir: str, transform=None) -> None:
        self.data_dir = data_dir
        self.answer_dir = answer_dir
        self.transform = transform

        # List all data files and sort them to ensure they match answer files
        self.data_files = sorted(
            [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))],
            key=lambda f: int(f.split('_')[-1].split('.')[0])
        )
        self.answer_files = sorted(
            [f for f in os.listdir(answer_dir) if os.path.isfile(os.path.join(answer_dir, f))],
            key=lambda f: int(f.split('_')[-1].split('.')[0])
        )

        # Ensure the number of data and answer files are the same
        assert len(self.data_files) == len(self.answer_files), "Mismatch between data and answer files"

        # Determine the total length by parsing the last file's end_index
        self.total_length = int(self.data_files[-1].split('_')[-1].split('.')[0])

        print(f"Data files: {self.data_files}")
        print(f"Answer files: {self.answer_files}")
        print(f"Total length: {self.total_length}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find the file corresponding to the index
        for file_name in self.data_files:
            start_idx, end_idx = map(int, file_name.split('_')[1:3])
            if start_idx <= idx <= end_idx:
                data_path = os.path.join(self.data_dir, file_name)
                answer_path = os.path.join(self.answer_dir, file_name)

                # Load the data and the corresponding answer
                data = pd.read_csv(data_path)
                answer = pd.read_csv(answer_path)

                # Apply any transformations if specified
                if self.transform:
                    data = self.transform(data)

                # Get the specific row for the given index
                data_row = data.iloc[idx - start_idx]
                answer_row = answer.iloc[idx - start_idx]

                return torch.tensor(data_row.values), torch.tensor(answer_row.values)

        raise IndexError("Index out of range")


# Example usage:
# data_dir = 'path/to/data'
# answer_dir = 'path/to/answer'
# dataset = CustomDataset(data_dir, answer_dir)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


if __name__ == "__main__":
    dataset = CwdDataset('../dataset/data_sim/OTDCR_1.dat')
    print(dataset[0])
