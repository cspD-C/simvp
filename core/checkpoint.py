import os

import torch


class BestCheckpointSaver:
    def __init__(self, mode="min", delta=0.0, verbose=False):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.best_value = None

    def _is_better(self, value):
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value <= (self.best_value - self.delta)
        return value >= (self.best_value + self.delta)

    def update(self, value, model, ckpt_path):
        if not self._is_better(value):
            return False
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        if self.verbose and self.best_value is not None:
            print(
                f"Checkpoint improved ({self.best_value:.6f} -> {value:.6f}), saved to {ckpt_path}"
            )
        self.best_value = value
        return True


def save_training_state(path, model, optimizer=None, scheduler=None, epoch=None, args=None, metrics=None):
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if args is not None:
        state["args"] = vars(args) if hasattr(args, "__dict__") else args
    if metrics is not None:
        state["metrics"] = metrics
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_model_weights(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    return checkpoint
