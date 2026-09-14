from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import DISEASES, IMAGENET_MEAN, IMAGENET_STD


def build_image_index(image_root: str | Path) -> Dict[str, str]:
    """Map image filename -> absolute path by recursively scanning image_root."""
    image_root = Path(image_root)
    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    mapping: Dict[str, str] = {}
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for path in image_root.rglob(pattern):
            mapping[path.name] = str(path)

    if not mapping:
        raise RuntimeError(f"No images found under {image_root}")
    return mapping


def label_to_vector(label_string: str) -> list[int]:
    labels = set(str(label_string).split("|"))
    if labels == {"No Finding"}:
        return [0] * len(DISEASES)
    return [int(disease in labels) for disease in DISEASES]


def load_metadata(csv_path: str | Path, image_root: str | Path) -> tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(csv_path)
    required = {"Image Index", "Finding Labels", "Patient ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Metadata is missing required columns: " + ", ".join(sorted(missing))
        )

    image_index = build_image_index(image_root)
    df = df[df["Image Index"].isin(image_index)].copy()
    if df.empty:
        raise RuntimeError("No metadata rows match image files under image_root.")
    return df.reset_index(drop=True), image_index


def patient_level_split(
    df: pd.DataFrame,
    test_size: float = 0.10,
    val_size: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by Patient ID to avoid image-level patient leakage."""
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be > 0 and sum to < 1.")

    patients = df["Patient ID"].dropna().unique()
    train_val_patients, test_patients = train_test_split(
        patients, test_size=test_size, random_state=seed
    )
    relative_val = val_size / (1.0 - test_size)
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=relative_val, random_state=seed
    )

    train_df = df[df["Patient ID"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["Patient ID"].isin(val_patients)].reset_index(drop=True)
    test_df = df[df["Patient ID"].isin(test_patients)].reset_index(drop=True)
    return train_df, val_df, test_df


def make_transforms(image_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


class ChestXray14Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_index: Dict[str, str], transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_index = image_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        image_path = self.image_index[row["Image Index"]]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(label_to_vector(row["Finding Labels"]), dtype=torch.float32)
        return image, target
