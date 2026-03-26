import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    if value in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def create_parser():
    parser = argparse.ArgumentParser("SimVP training and evaluation")

    # Runtime.
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--use_gpu", default=True, type=str2bool)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)

    # Output / experiment.
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--res_dir", default="./results", type=str)
    parser.add_argument("--ex_name", default="Debug", type=str)

    # Data.
    parser.add_argument("--dataname", default="mmnist",
                        choices=["mmnist", "taxibj"])
    parser.add_argument("--data_root", default="./data/", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--val_batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--in_frames", default=10, type=int)
    parser.add_argument("--out_frames", default=10, type=int)

    # Model.
    parser.add_argument("--model", default="simvp",
                        choices=["simvp"], type=str)
    parser.add_argument("--in_shape", default=[10, 1, 64, 64],
                        type=int, nargs="*")
    parser.add_argument("--hid_S", default=64, type=int)
    parser.add_argument("--hid_T", default=256, type=int)
    parser.add_argument("--N_S", default=4, type=int)
    parser.add_argument("--N_T", default=8, type=int)
    parser.add_argument("--groups", default=4, type=int)

    # Optimization / training.
    parser.add_argument("--epochs", default=51, type=int)
    parser.add_argument(
        "--max_epochs",
        default=200,
        type=int,
        help="Safety cap when --epochs=0 (early-stop mode).",
    )
    parser.add_argument(
        "--min_epochs",
        default=5,
        type=int,
        help="Minimum epochs before early-stop can trigger.",
    )
    parser.add_argument(
        "--early_stop_patience",
        default=10,
        type=int,
        help="Stop when validation has no improvement for N epochs.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        default=1e-4,
        type=float,
        help="Minimum validation-loss improvement to reset patience.",
    )
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--log_step", default=1, type=int)
    parser.add_argument(
        "--max_steps_per_epoch",
        default=0,
        type=int,
        help="0 means full epoch.",
    )
    parser.add_argument("--max_val_samples", default=1000, type=int)
    parser.add_argument("--save_every", default=100, type=int)
    parser.add_argument("--loss", default="mse", choices=["mse"], type=str)

    # Evaluation / resume.
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--save_outputs", action="store_true")

    return parser
