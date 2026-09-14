from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess_lungs import LungCropper


def main():
    parser = argparse.ArgumentParser(description="Crop detected lung regions and resize images")
    parser.add_argument("--input", required=True, help="Input image root")
    parser.add_argument("--output", required=True, help="Output image root")
    parser.add_argument("--model-id", default="detection-lungs/25")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    cropper = LungCropper(model_id=args.model_id, image_size=args.image_size)
    success, skipped = cropper.run(args.input, args.output)
    print(f"Processed={success}, skipped={skipped}")


if __name__ == "__main__":
    main()
