from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
from typing import TypedDict

import cv2

from src.main.lib.print_banner import print_banner
from src.main.lib.rotated_rectangle import RotatedRectangleDto


class Arguments(Namespace):
    image_path: Path
    output_dir: Path
    dbnet_model_path: Path


class Output(TypedDict):
    source_image_path: str
    rectangles: list[RotatedRectangleDto]


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
    print_banner("Loading DBnet")
    net = cv2.dnn.TextDetectionModel_DB(model=str(args.dbnet_model_path))
    image = cv2.imread(str(args.image_path))
    if image is None:
        raise ValueError(f"Could not read image from {args.image_path}")

    # Detect the text areas in the image using DBnet
    print_banner("Detecting text areas")
    results = zip(*net.detectTextRectangles(frame=image))

    # Remove results with low confidence
    print_banner("Pruning results by confidence")
    CONFIDENCE_THRESHOLD = 0.8
    pruned_results = [
        rect for rect, confidence in results if confidence >= CONFIDENCE_THRESHOLD
    ]

    # Output the results
    print_banner("Saving results")
    output = Output(
        source_image_path=str(args.image_path),
        rectangles=[
            RotatedRectangleDto(
                center=rect.center,  # type: ignore
                size=rect.size,  # type: ignore
                angle=rect.angle,  # type: ignore
            )
            for rect in pruned_results
        ],
    )
    output_file_path = args.output_dir / args.image_path.with_suffix(".json").name
    with open(output_file_path, "w") as output_file:
        json.dump(output, output_file, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
