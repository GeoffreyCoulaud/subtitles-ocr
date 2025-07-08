from argparse import ArgumentParser, Namespace
from pathlib import Path

import cv2

from src.main.lib.print_banner import print_banner


class Arguments(Namespace):
    image_path: Path
    output_dir: Path
    dbnet_model_path: Path


def main():

    # Parse command line arguments
    parser = ArgumentParser(
        description="Extract frames from a video file and rename them with frame number and timestamp."
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "dbnet_model_path",
        type=Path,
        help="Path to the pre-trained DBnet model file.",
    )
    args = parser.parse_args(namespace=Arguments())

    # Validate arguments
    if not args.image_path.is_file():
        raise ValueError(f"Video path {args.image_path} is not a valid file.")
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    if not args.output_dir.is_dir():
        raise ValueError(f"Output {args.output_dir} is not a directory.")

    # Load the image and the pre-trained DBnet model
    net = cv2.dnn.TextDetectionModel_DB(model=str(args.dbnet_model_path))
    image = cv2.imread(str(args.image_path))
    if image is None:
        raise ValueError(f"Could not read image from {args.image_path}")

    # Detect the text areas in the image using DBnet
    zones = net.detectTextRectangles(frame=image)
    if not zones:
        print_banner("No text detected in the image")

    # - Take each zone
    # - get it's xy bounds
    # - crop the image to the bounds
    # - mask out the outside of the zone in the bounds
    # - save the cropped image to the output directory

    # TODO - How to save the bounds/box coordinates properly?
    # TODO - Try direct text recognition with PaddleOCR v5 multi-lingual model (in another script)
    raise NotImplementedError("Text detection output processing is not implemented yet")


if __name__ == "__main__":
    main()
