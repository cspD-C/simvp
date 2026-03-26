from config import create_parser
from train import run


def main():
    args = create_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
