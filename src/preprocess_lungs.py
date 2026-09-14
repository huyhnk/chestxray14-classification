from __future__ import annotations

import os
import time
from pathlib import Path

from PIL import Image


class LungCropper:
    """Crop the union of detected lung bounding boxes, then resize the crop."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "detection-lungs/25",
        api_url: str = "https://serverless.roboflow.com",
        image_size: int = 224,
        request_delay: float = 0.1,
    ):
        api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("Set ROBOFLOW_API_KEY before running lung preprocessing.")

        try:
            from inference_sdk import InferenceHTTPClient
        except ImportError as exc:
            raise RuntimeError(
                "inference-sdk is required for lung cropping. Install requirements-preprocess.txt"
            ) from exc

        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.model_id = model_id
        self.image_size = image_size
        self.request_delay = request_delay

    def predict(self, image_path: str | Path):
        time.sleep(self.request_delay)
        result = self.client.infer(str(image_path), model_id=self.model_id)
        return result.get("predictions", [])

    def crop_one(self, image_path: str | Path, output_path: str | Path) -> bool:
        predictions = self.predict(image_path)
        if not predictions:
            return False

        image = Image.open(image_path)
        width_img, height_img = image.size
        boxes = []
        for pred in predictions:
            x, y = pred["x"], pred["y"]
            w, h = pred["width"], pred["height"]
            left = max(0, int(x - w / 2))
            top = max(0, int(y - h / 2))
            right = min(width_img, int(x + w / 2))
            bottom = min(height_img, int(y + h / 2))
            boxes.append((left, top, right, bottom))

        left = min(b[0] for b in boxes)
        top = min(b[1] for b in boxes)
        right = max(b[2] for b in boxes)
        bottom = max(b[3] for b in boxes)
        if right <= left or bottom <= top:
            return False

        crop = image.crop((left, top, right, bottom)).resize((self.image_size, self.image_size))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)
        return True

    def run(self, input_root: str | Path, output_root: str | Path):
        input_root = Path(input_root)
        output_root = Path(output_root)
        image_paths = [
            p for p in input_root.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ]
        success, skipped = 0, 0
        for image_path in image_paths:
            relative = image_path.relative_to(input_root)
            if self.crop_one(image_path, output_root / relative):
                success += 1
            else:
                skipped += 1
                print(f"No valid lung detection: {relative}")
        return success, skipped
