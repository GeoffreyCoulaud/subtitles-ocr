import argparse
import tempfile
from pathlib import Path
from queue import Queue
from services.crop_service import CropService
from services.image_extraction_service import ImageExtractionService
from services.preprocessing_service import PreprocessingService
from services.ocr_service import OCRService, OcrResult
from services.subtitle_service import SubtitleService, SubtitleEntry
import threading


def format_time(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(subs: list[SubtitleEntry], output_srt: Path) -> None:
    with output_srt.open("w", encoding="utf-8") as f:
        for idx, entry in enumerate(subs, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_time(entry.start)} --> {format_time(entry.end)}\n")
            f.write(entry.text + "\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract hardsubs from a video into an SRT file."
    )
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument("output", type=Path, help="Output SRT file")
    parser.add_argument(
        "--crop-height",
        type=int,
        default=100,
        help="Height in pixels of subtitle region to crop",
    )
    parser.add_argument(
        "--y-pos",
        type=int,
        default=380,
        help="Y offset for cropping the subtitle region",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="Frame extraction rate (frames per second)",
    )
    parser.add_argument(
        "--lang", type=str, default="fra", help="Tesseract language code"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cropped = tmpdir / "cropped.mp4"
        frames = tmpdir / "frames"

        # Queues for each service
        crop_in: Queue = Queue()
        crop_out: Queue = Queue()
        extract_in: Queue = Queue()
        extract_out: Queue = Queue()
        preprocess_in: Queue = Queue()
        preprocess_out: Queue = Queue()
        ocr_in: Queue = Queue()
        ocr_out: Queue = Queue()
        sub_in: Queue = Queue()
        sub_out: Queue = Queue()

        # Instantiate service classes
        crop_service = CropService(crop_in, crop_out, args.crop_height, args.y_pos)
        extract_service = ImageExtractionService(extract_in, extract_out, args.fps)
        preprocess_service = PreprocessingService(preprocess_in, preprocess_out)
        ocr_service = OCRService(ocr_in, ocr_out, args.lang)
        subtitle_service = SubtitleService(sub_in, sub_out)

        # Start service threads
        crop_thread = threading.Thread(target=crop_service.run)
        extract_thread = threading.Thread(target=extract_service.run)
        preprocess_thread = threading.Thread(target=preprocess_service.run)
        ocr_thread = threading.Thread(target=ocr_service.run)
        sub_thread = threading.Thread(target=subtitle_service.run)
        for t in [
            crop_thread,
            extract_thread,
            preprocess_thread,
            ocr_thread,
            sub_thread,
        ]:
            t.start()

        # Pipeline: crop -> extract -> preprocess -> ocr -> subtitle
        crop_in.put((args.input, cropped))
        crop_in.put(None)
        count = {"crop": 0, "extract": 0, "preprocess": 0, "ocr": 0}

        # Crop
        while True:
            out = crop_out.get()
            if out is None:
                extract_in.put(None)
                break
            extract_in.put((cropped, frames))
            count["crop"] += 1
            print(f"Crop done: {count['crop']}")

        # Extract
        while True:
            out = extract_out.get()
            if out is None:
                preprocess_in.put(None)
                break
            for frame in frames.iterdir():
                if frame.name.endswith(".jpg"):
                    preprocess_in.put(str(frame))
                    count["extract"] += 1
                    print(f"Frames extracted: {count['extract']}")

        # Preprocess
        while True:
            out = preprocess_out.get()
            if out is None:
                ocr_in.put(None)
                break
            ocr_in.put(out)
            count["preprocess"] += 1
            print(f"Preprocessed: {count['preprocess']}")

        # OCR
        while True:
            out = ocr_out.get()
            if out is None:
                sub_in.put(None)
                break
            sub_in.put(out)
            count["ocr"] += 1
            print(f"OCR done: {count['ocr']}")

        # Subtitle consolidation
        subs = sub_out.get()
        print(f"Writing SRT to {args.output}...")
        write_srt(subs, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
