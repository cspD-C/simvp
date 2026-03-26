import os

from core.checkpoint import BestCheckpointSaver


class Recorder:
    def __init__(self, verbose=False, delta=0):
        self.saver = BestCheckpointSaver(mode="min", delta=delta, verbose=verbose)

    def __call__(self, val_loss, model, path):
        ckpt_path = os.path.join(path, "checkpoint.pth")
        self.saver.update(val_loss, model, ckpt_path)
