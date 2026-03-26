import torch

from .moving_mnist import build_dataset as build_mmnist_dataset
from .taxibj import build_dataset as build_taxibj_dataset


def build_dataset(dataname, data_root, split="train", in_frames=10, out_frames=10):
    if dataname == "mmnist":
        return build_mmnist_dataset(
            data_root=data_root,
            split=split,
            in_frames=in_frames,
            out_frames=out_frames,
        )
    if dataname == "taxibj":
        return build_taxibj_dataset(data_root=data_root, split=split)
    raise ValueError(f"Unsupported dataset: {dataname}")


def build_dataloaders(args):
    train_set = build_dataset(
        dataname=args.dataname,
        data_root=args.data_root,
        split="train",
        in_frames=args.in_frames,
        out_frames=args.out_frames,
    )

    # Keep legacy behavior: taxibj does not provide a dedicated validation split.
    if args.dataname == "taxibj":
        val_set = None
        test_set = build_dataset(
            dataname=args.dataname,
            data_root=args.data_root,
            split="test",
            in_frames=args.in_frames,
            out_frames=args.out_frames,
        )
    else:
        val_set = build_dataset(
            dataname=args.dataname,
            data_root=args.data_root,
            split="val",
            in_frames=args.in_frames,
            out_frames=args.out_frames,
        )
        test_set = build_dataset(
            dataname=args.dataname,
            data_root=args.data_root,
            split="test",
            in_frames=args.in_frames,
            out_frames=args.out_frames,
        )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    val_loader = None
    if val_set is not None:
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.val_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader
