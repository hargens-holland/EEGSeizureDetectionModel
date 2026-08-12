"""Training and validation loops shared by both CNN training scripts."""

import numpy as np
import torch

from metrics import compute_metrics


def train_epoch(model, loader, criterion, optimizer, device):
    """Run one training pass. Returns (mean loss, metrics)."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    epoch_loss = running_loss / max(len(loader), 1)
    # No AUC here — argmax predictions alone can't produce a ranking score.
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return epoch_loss, metrics


def validate_epoch(model, loader, criterion, device):
    """Run one evaluation pass. Returns (mean loss, metrics including AUC)."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())

    epoch_loss = running_loss / max(len(loader), 1)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores),
    )
    return epoch_loss, metrics


def random_split_indices(n, seed, train_ratio=0.7, val_ratio=0.15):
    """Shuffle window indices into train/val/test partitions.

    Note: this splits windows, not patients. See the README's Limitations
    section for what that means for the reported numbers.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    return indices[:train_end], indices[train_end:val_end], indices[val_end:]
