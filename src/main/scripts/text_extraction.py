from argparse import ArgumentParser, Namespace
from pathlib import Path

from paddleocr import PaddleOCR

from src.main.lib.print_banner import print_banner


class Arguments(Namespace):
    image_path: Path
    output_dir: Path
    dbnet_model_path: Path


def main():

    # Parse command line arguments
    parser = ArgumentParser(description="Extract texts from an image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(namespace=Arguments())

    # Validate arguments
    if not args.image_path.is_file():
        raise ValueError(f"Video path {args.image_path} is not a valid file.")
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    if not args.output_dir.is_dir():
        raise ValueError(f"Output {args.output_dir} is not a directory.")

    # TODO


if __name__ == "__main__":
    main()
