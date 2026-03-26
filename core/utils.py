import json
import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def resolve_output_dir(args):
    if getattr(args, "output_dir", ""):
        return args.output_dir
    return os.path.join(args.res_dir, args.ex_name)


def setup_logging(log_file):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(message)s",
    )


def print_log(message):
    print(message)
    logging.info(message)


def format_namespace(namespace):
    configs = namespace.__dict__
    message = ""
    for key, value in configs.items():
        message += f"\n{key}:\t{value}\t"
    return message


def dump_args(args, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


def resolve_device(args):
    if args.use_gpu and torch.cuda.is_available() and args.device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        return torch.device("cuda:0")
    return torch.device("cpu")
