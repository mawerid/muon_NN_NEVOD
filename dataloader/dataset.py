import torch
from torch.utils.data import DataLoader, Dataset


def angles_from_sct(up_number: int, down_number: int):
    pass


class CwdDataset(Dataset):
    def __init__(self, data_path: str):
        if data_path.endswith('.csv'):
            self.data = torch.read_csv(data_path)
        elif data_path.endswith('.dat'):
            self.data = torch.loadtxt(data_path)
        else:
            raise ValueError("Unsupported file format. Only CSV and DAT files are supported.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":
    dataset = CwdDataset('../data_sim/OTDCR_1.dat')
    print(dataset[0])
