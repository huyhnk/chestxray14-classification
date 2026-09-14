from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def predict(model, loader, criterion, device):
    model.eval()
    losses, targets_all, probs_all = [], [], []
    for images, targets in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.sigmoid(logits)

        losses.append(loss.item())
        targets_all.append(targets.cpu().numpy())
        probs_all.append(probs.cpu().numpy())

    return (
        float(np.mean(losses)) if losses else float("nan"),
        np.concatenate(targets_all, axis=0),
        np.concatenate(probs_all, axis=0),
    )
