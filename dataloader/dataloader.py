import torch
from torch.utils.data import DataLoader, Dataset


class CwdDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":
    print("hello")
