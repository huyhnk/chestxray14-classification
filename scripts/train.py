from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import DISEASES
from src.data import ChestXray14Dataset, load_metadata, make_transforms, patient_level_split
from src.engine import predict, train_one_epoch
from src.losses import FocalLoss
from src.metrics import compute_metrics, fit_thresholds
from src.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ChestX-ray14 multi-label classifier")
    parser.add_argument("--csv", required=True, help="Path to Data_Entry_2017.csv")
    parser.add_argument("--images", required=True, help="Root containing image folders")
    parser.add_argument("--output", default="outputs/run1")
    parser.add_argument("--model", default="densenet121", choices=["densenet121", "resnet18", "resnet101", "resnet152"])
    parser.add_argument("--loss", default="focal", choices=["focal", "bce"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df, image_index = load_metadata(args.csv, args.images)
    train_df, val_df, test_df = patient_level_split(
        df, test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )
    print(f"Images: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(
        "Patients: "
        f"train={train_df['Patient ID'].nunique()}, "
        f"val={val_df['Patient ID'].nunique()}, "
        f"test={test_df['Patient ID'].nunique()}"
    )

    train_transform, eval_transform = make_transforms(args.image_size)
    train_loader = make_loader(
        ChestXray14Dataset(train_df, image_index, train_transform),
        args.batch_size, True, args.workers, device,
    )
    val_loader = make_loader(
        ChestXray14Dataset(val_df, image_index, eval_transform),
        args.batch_size, False, args.workers, device,
    )
    test_loader = make_loader(
        ChestXray14Dataset(test_df, image_index, eval_transform),
        args.batch_size, False, args.workers, device,
    )

    model = build_model(args.model, len(DISEASES), pretrained=not args.no_pretrained).to(device)
    criterion = FocalLoss() if args.loss == "focal" else nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=1, factor=0.1
    )

    best_auc = -math.inf
    patience_counter = 0
    checkpoint_path = output_dir / "best_model.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_val, p_val = predict(model, val_loader, criterion, device)
        thresholds = fit_thresholds(y_val, p_val)
        val_metrics = compute_metrics(y_val, p_val, thresholds)
        scheduler.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{k: v for k, v in val_metrics.items() if not k.startswith("per_class")},
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_AUROC={val_metrics['macro_auroc']:.4f} | "
            f"val_F1={val_metrics['macro_f1']:.4f}"
        )

        current_auc = val_metrics["macro_auroc"]
        if not math.isnan(current_auc) and current_auc > best_auc:
            best_auc = current_auc
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "labels": DISEASES,
                    "thresholds": thresholds.tolist(),
                    "image_size": args.image_size,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if not checkpoint_path.exists():
        raise RuntimeError("No valid checkpoint was produced.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float32)
    test_loss, y_test, p_test = predict(model, test_loader, criterion, device)
    test_metrics = compute_metrics(y_test, p_test, thresholds)
    results = {"test_loss": test_loss, **test_metrics, "labels": DISEASES}

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Test macro AUROC={test_metrics['macro_auroc']:.4f} | macro F1={test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
