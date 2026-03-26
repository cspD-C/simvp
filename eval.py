import os

from config import create_parser
from core.checkpoint import load_model_weights
from core.utils import print_log
from engine import test_model
from exp import Exp


def run(args):
    exp = Exp(args)
    ckpt_path = args.ckpt or os.path.join(exp.checkpoints_path, "checkpoint.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_model_weights(exp.model, ckpt_path, exp.device)
    save_dir = os.path.join(exp.path, "results", args.ex_name, "sv") if args.save_outputs else None
    metrics = test_model(
        model=exp.model,
        test_loader=exp.test_loader,
        device=exp.device,
        save_dir=save_dir,
    )
    print_log(
        "eval mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}".format(
            metrics["mse"], metrics["mae"], metrics["ssim"], metrics["psnr"]
        )
    )
    return metrics


def main():
    parser = create_parser()
    parser.set_defaults(eval_only=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
