from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
from typing import TypedDict

from cv2 import RotatedRect, minAreaRect
import numpy as np

from src.main.lib.display_image_with_rects import display_image_with_rects
from src.main.lib.print_banner import print_banner
from src.main.lib.rotated_rectangle import RotatedRectangleDto


class Arguments(Namespace):
    input_path: Path
    output_dir: Path
    distance_threshold: float
    angle_threshold: float
    debug_display: bool


class Input(TypedDict):
    source_image_path: str
    rectangles: list[RotatedRectangleDto]


class Output(TypedDict):
    source_image_path: str
    rectangles: list[RotatedRectangleDto]


def main():

    # Parse command line arguments
    parser = ArgumentParser(
        description="Merge cv2 rotated rectangles by proximity and angle similarity."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=10.0,
        help="Max distance threshold for merging rectangles",
    )
    parser.add_argument(
        "--angle_threshold",
        type=float,
        default=5.0,
        help="Max angle threshold for merging rectangles",
    )
    parser.add_argument(
        "--debug-display",
        action="store_true",
        help="Display an image with detected text areas for debugging purposes.",
    )
    args = parser.parse_args(namespace=Arguments())

    # Validate arguments
    if not args.input_path.is_file():
        raise ValueError(f"Json path {args.input_path} is not a valid file.")
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    if not args.output_dir.is_dir():
        raise ValueError(f"Output {args.output_dir} is not a directory.")

    # Load and parse the input JSON file
    with open(args.input_path, "r") as input_file:
        data: Input = json.load(input_file)
        rectangles = [
            RotatedRect(
                center=(dto["center"][0], dto["center"][1]),
                size=(dto["size"][0], dto["size"][1]),
                angle=dto["angle"],
            )
            for dto in data["rectangles"]
        ]

    # Try to consolidate the results by merging close rectangles of similar angles
    print_banner("Consolidating results")
    groups = group_rectangles(
        rectangles=rectangles,
        distance_threshold=args.distance_threshold,
        angle_threshold=args.angle_threshold,
    )

    # Merge groups of rectangle to a single rectangle
    print_banner("Merging groups of rectangles")
    merged_groups = [combine_rectangles(group) for group in groups]

    # Save the consolidated rectangles to the output directory
    print_banner("Saving results")
    output = Output(
        source_image_path=str(data["source_image_path"]),
        rectangles=[
            RotatedRectangleDto(
                center=rect.center,  # type: ignore
                size=rect.size,  # type: ignore
                angle=rect.angle,
            )
            for rect in merged_groups
        ],
    )
    output_file_path = args.output_dir / f"{args.input_path.stem}.json"
    with open(output_file_path, "w") as output_file:
        json.dump(output, output_file, indent=4, sort_keys=True)

    # If debug display is enabled, show the image with detected text areas
    if args.debug_display:
        print_banner("Displaying grouped text areas")
        display_image_with_rects(Path(data["source_image_path"]), merged_groups)


def group_rectangles(
    rectangles: list[RotatedRect],
    distance_threshold: float,
    angle_threshold: float,
) -> list[list[RotatedRect]]:
    """Groups rotated rectangles based on spatial proximity and angle similarity."""

    rect_count: int = len(rectangles)
    visited: list[bool] = [False] * rect_count
    groups = list[list[RotatedRect]]()

    for i in range(rect_count):
        if visited[i]:
            continue

        group: list[int] = [i]
        queue: list[int] = [i]
        visited[i] = True

        while queue:
            current: int = queue.pop()
            for j in range(rect_count):
                if visited[j]:
                    continue
                if is_similar(
                    rect1=rectangles[current],
                    rect2=rectangles[j],
                    angle_threshold=angle_threshold,
                    distance_threshold=distance_threshold,
                ):
                    visited[j] = True
                    queue.append(j)
                    group.append(j)

        grouped_rects: list[RotatedRect] = [rectangles[i] for i in group]
        groups.append(grouped_rects)

    return groups


def is_similar(
    rect1: RotatedRect,
    rect2: RotatedRect,
    angle_threshold: float,
    distance_threshold: float,
) -> bool:
    """
    Check if two rectangles are similar based on their distances and angles.
    """

    # Get the closest point pair
    min_distance = float("inf")
    for p1 in rect1.points():
        for p2 in rect2.points():
            d = distance(p1, p2)  # type: ignore
            if d < min_distance:
                min_distance = d

    # Normalize angles to be in the range [-180, 180)
    rect1.angle = normalize_angle(rect1.angle)
    rect2.angle = normalize_angle(rect2.angle)

    # check if the distance is small enough and the angles are similar
    return (
        min_distance < distance_threshold
        and abs(rect1.angle - rect2.angle) < angle_threshold
    )


def normalize_angle(angle: float) -> float:
    """Normalize an angle to be in the range [-180, 180)."""
    return (angle + 180) % 360 - 180


def distance(v1: tuple[float, ...], v2: tuple[float, ...]) -> float:
    """Compute the distance between two N dimensional vectors."""
    return sum(map(lambda pair: ((pair[0] - pair[1]) ** 2), zip(v1, v2))) ** 0.5


def combine_rectangles(rectangles: list[RotatedRect]) -> RotatedRect:
    """
    Combine a list of rectangles into a single rectangle encompassing them.
    """
    raw = minAreaRect(
        np.array(
            [point for rect in rectangles for point in rect.points()],
            dtype=np.float32,
        )
    )

    return RotatedRect(center=raw[0], size=raw[1], angle=raw[2])


if __name__ == "__main__":
    main()
