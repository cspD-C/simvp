import torch


def build_loss(loss_name):
    if loss_name == "mse":
        return torch.nn.MSELoss()
    raise ValueError(f"Unsupported loss: {loss_name}")
