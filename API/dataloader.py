from types import SimpleNamespace

from datasets import build_dataloaders


def load_data(dataname, batch_size, val_batch_size, data_root, num_workers, **kwargs):
    args = SimpleNamespace(
        dataname=dataname,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        data_root=data_root,
        num_workers=num_workers,
        in_frames=kwargs.get("in_frames", 10),
        out_frames=kwargs.get("out_frames", 10),
    )
    train_loader, val_loader, test_loader = build_dataloaders(args)
    mean = getattr(train_loader.dataset, "mean", 0)
    std = getattr(train_loader.dataset, "std", 1)
    return train_loader, val_loader, test_loader, mean, std
