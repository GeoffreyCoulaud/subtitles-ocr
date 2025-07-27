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
    parser.add_argument(
        "--hardware", type=str, default="cpu", choices=["cpu", "gpu", "xpu"]
    )
    args = parser.parse_args(namespace=Arguments())

    # Validate arguments
    if not args.image_path.is_file():
        raise ValueError(f"Video path {args.image_path} is not a valid file.")
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    if not args.output_dir.is_dir():
        raise ValueError(f"Output {args.output_dir} is not a directory.")

    # Load PaddleOCR
    print_banner("Starting text extraction with PaddleOCR")
    paddle_args = {
        # Specify French recognition model with the lang parameter
        "lang": "fr",
        # PP-OCRv5 server-side text detection model with higher accuracy, suitable for deployment on high-performance servers
        "text_detection_model_name": "PP-OCRv5_server_det",
        # PP-OCRv5_rec is a next-generation text recognition model.
        # It aims to efficiently and accurately support the recognition of four major languages
        # (Simplified Chinese, Traditional Chinese, English, Japanese)
        # as well as complex text scenarios such as handwriting, vertical text, pinyin, and rare characters using a single model.
        # While maintaining recognition performance, it balances inference speed and model robustness, providing efficient and accurate technical support for document understanding in various scenarios.
        "text_recognition_model_name": "PP-OCRv5_server_rec",
        # Disable document specific features
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    paddle_args["device"] = args.hardware
    ocr = PaddleOCR(**paddle_args)
    results = ocr.predict(input=str(args.image_path))

    # Save the OCR results
    print_banner("Saving OCR results")
    for i, result in enumerate(results):
        result.save_to_json(args.output_dir / f"result_{i}.json")


if __name__ == "__main__":
    main()
