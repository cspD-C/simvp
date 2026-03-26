import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    def __init__(self, x_data, y_data):
        super().__init__()
        self.x_data = (x_data + 1) / 2
        self.y_data = (y_data + 1) / 2
        self.mean = 0
        self.std = 1

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, index):
        x = torch.tensor(self.x_data[index, ::]).float()
        y = torch.tensor(self.y_data[index, ::]).float()
        return x, y


def build_dataset(data_root, split="train"):
    dataset_path = os.path.join(data_root, "taxibj/dataset.npz")
    dataset = np.load(dataset_path)

    if split == "train":
        x_data, y_data = dataset["X_train"], dataset["Y_train"]
    else:
        x_data, y_data = dataset["X_test"], dataset["Y_test"]
    return TrafficDataset(x_data=x_data, y_data=y_data)
