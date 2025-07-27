from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
from typing import TypedDict

from cv2 import RotatedRect, minAreaRect
import numpy as np

from src.main.lib.print_banner import print_banner
from src.main.lib.rotated_rectangle import RotatedRectangleDto


class Arguments(Namespace):
    input_path: Path
    output_dir: Path


class Output(TypedDict):
    source_image_path: str
    rectangle: RotatedRectangleDto


def main():

    # Parse command line arguments
    parser = ArgumentParser(description="")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_dir", type=Path)
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
        data = json.load(input_file)
        rotated_rectangles = [
            RotatedRect(
                center=(dto["center"][0], dto["center"][1]),
                size=(dto["size"][0], dto["size"][1]),
                angle=dto["angle"],
            )
            for dto in data["rectangles"]
        ]

    # Try to consolidate the results by merging close rectangles of similar angles
    print_banner("Consolidating results")
    DISTANCE_THRESHOLD = 10
    groups: list[list[RotatedRect]] = []
    for rect in rotated_rectangles:
        found_group = False
        for group in groups:
            # Check if the rectangle is close enough to any rectangle in the group
            if any(distance_function(rect, r) < DISTANCE_THRESHOLD for r in group):  # type: ignore
                group.append(rect)  # type: ignore
                found_group = True
                break
        if not found_group:
            groups.append([rect])  # type: ignore

    # Merge groups of rectangle to a single rectangle
    print_banner("Merging groups of rectangles")
    merged_groups = [
        (combine_rectangles(group) if len(group) > 1 else group[0]) for group in groups
    ]

    # Save the consolidated rectangles to the output directory
    print_banner("Saving results")
    for i, rect in enumerate(merged_groups):

        # Build the output structure
        output = Output(
            source_image_path=str(args.image_path),
            rectangle=RotatedRectangleDto(
                center=rect.center,  # type: ignore
                size=rect.size,  # type: ignore
                angle=rect.angle,
            ),
        )

        # Save the output as a JSON file
        output_file_path = args.output_dir / f"{args.input_path.stem}_group_{i}.json"
        with open(output_file_path, "w") as output_file:
            json.dump(output, output_file, indent=4, sort_keys=True)


def distance_function(
    rect1: RotatedRect,
    rect2: RotatedRect,
):
    """
    Compute the clustering distance between two rectangles.
    The base distance is the smallest pixel distance between edges.
    Similar angles keep the pixel as is, while different angles make the distance skyrocket, no matter the pixel distance.
    """

    # Get the closest point pair
    rect1_min_point = rect1.center
    rect2_min_point = rect2.center
    min_distance = float("inf")
    for p1 in rect1.points():
        for p2 in rect2.points():
            d = distance(p1, p2)  # type: ignore
            if d < min_distance:
                min_distance = d
                rect1_min_point = p1
                rect2_min_point = p2

    # Normalize angles to be in the range [-180, 180)
    rect1.angle = normalize_angle(rect1.angle)
    rect2.angle = normalize_angle(rect2.angle)

    # Make x,y,angle a 3D vector
    ANGLE_BIAS = 2  # 5° should be about the same as 10px distance
    rect1_vector = (
        rect1_min_point[0],
        rect1_min_point[1],
        rect1.angle / 180 * ANGLE_BIAS,
    )
    rect2_vector = (
        rect2_min_point[0],
        rect2_min_point[1],
        rect2.angle / 180 * ANGLE_BIAS,
    )

    # Compute the distance between the two vectors
    return distance(rect1_vector, rect2_vector)


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
    return minAreaRect(
        np.array(
            [point for rect in rectangles for point in rect.points()],
            dtype=np.float32,
        )
    )  # type: ignore


if __name__ == "__main__":
    main()
