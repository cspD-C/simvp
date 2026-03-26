import gzip
import os
import random

import numpy as np
import torch
import torch.utils.data as data


def load_mnist(root):
    path = os.path.join(root, "moving_mnist/train-images-idx3-ubyte.gz")
    with gzip.open(path, "rb") as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root):
    path = os.path.join(root, "moving_mnist/mnist_test_seq.npy")
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class MovingMNIST(data.Dataset):
    def __init__(
        self,
        root,
        is_train=True,
        n_frames_input=10,
        n_frames_output=10,
        num_objects=(2,),
    ):
        super().__init__()
        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1
        self.mean = 0
        self.std = 1

    def __len__(self):
        return self.length

    def get_random_trajectory(self, seq_length):
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        data_arr = np.zeros(
            (self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32
        )
        for _ in range(num_digits):
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                data_arr[i, top:bottom, left:right] = np.maximum(
                    data_arr[i, top:bottom, left:right], digit_image
                )

        return data_arr[..., np.newaxis]

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            num_digits = random.choice(self.num_objects)
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        r = 1
        w = int(64 / r)
        images = (
            images.reshape((length, w, r, w, r))
            .transpose(0, 2, 4, 1, 3)
            .reshape((length, r * r, w, w))
        )

        x = images[: self.n_frames_input]
        y = images[self.n_frames_input : length] if self.n_frames_output > 0 else []
        y = torch.from_numpy(y / 255.0).contiguous().float()
        x = torch.from_numpy(x / 255.0).contiguous().float()
        return x, y


def build_dataset(data_root, split="train", in_frames=10, out_frames=10):
    is_train = split == "train"
    return MovingMNIST(
        root=data_root,
        is_train=is_train,
        n_frames_input=in_frames,
        n_frames_output=out_frames,
        num_objects=(2,),
    )
