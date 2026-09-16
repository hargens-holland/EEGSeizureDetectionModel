"""Training, evaluation and splitting helpers shared by every entry point.

Two things live here that the individual scripts should never reimplement:

- ``split_indices`` — the train/val/test partition. Recording-level by default,
  so no EDF file contributes windows to more than one split.
- ``choose_threshold`` / ``evaluate_at_threshold`` — operating-point selection
  on the validation split, then evaluation on test at that fixed threshold.
"""

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

    This is the *leaky* split: overlapping neighbours of a test window can sit
    in train. Kept so the original course result can be reproduced; use
    ``split_indices(..., split="recording")`` for an honest evaluation.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def recording_split_indices(recording_ids, recording_has_seizure, seed,
                            train_ratio=0.7, val_ratio=0.15):
    """Partition windows so that each recording lands in exactly one split.

    Recordings containing a seizure are rare (a handful per patient), so they
    are shuffled and allocated separately from seizure-free recordings. That
    keeps every split from ending up with zero positive windows, which would
    make sensitivity undefined.

    Args:
        recording_ids: Per-window recording index, shape (n_windows,).
        recording_has_seizure: Per-recording bool, indexed by recording id.
        seed: RNG seed.

    Returns:
        (train_idx, val_idx, test_idx) window index arrays.
    """
    rng = np.random.RandomState(seed)
    recording_ids = np.asarray(recording_ids)
    has_seizure = np.asarray(recording_has_seizure, dtype=bool)

    present = np.unique(recording_ids)
    seizure_recs = rng.permutation(present[has_seizure[present]])
    clean_recs = rng.permutation(present[~has_seizure[present]])

    def allocate(recs):
        n = len(recs)
        train_end = int(round(train_ratio * n))
        val_end = int(round((train_ratio + val_ratio) * n))
        # With very few recordings the rounding can starve val or test; make
        # sure each split gets at least one when there are enough to go round.
        if n >= 3:
            train_end = min(max(train_end, 1), n - 2)
            val_end = min(max(val_end, train_end + 1), n - 1)
        return recs[:train_end], recs[train_end:val_end], recs[val_end:]

    parts = [allocate(seizure_recs), allocate(clean_recs)]
    splits = []
    for i in range(3):
        recs = np.concatenate([parts[0][i], parts[1][i]])
        splits.append(np.flatnonzero(np.isin(recording_ids, recs)))

    return tuple(splits)


def split_indices(dataset, seed, split="recording", train_ratio=0.7, val_ratio=0.15):
    """Dispatch to the requested split strategy for a ``CHBMITDataset``."""
    if split == "window":
        return random_split_indices(len(dataset), seed, train_ratio, val_ratio)
    if split == "recording":
        return recording_split_indices(
            dataset.recording_ids, dataset.recording_has_seizure, seed, train_ratio, val_ratio
        )
    raise ValueError(f"Unknown split strategy: {split!r}")


def describe_split(name, dataset, idx):
    """One-line summary of a split: window count, seizure windows, recordings."""
    labels = dataset.labels[idx]
    n_recs = len(np.unique(dataset.recording_ids[idx])) if len(idx) else 0
    print(f"{name:<6} {len(idx):>7} windows  {int(labels.sum()):>5} seizure  {n_recs:>3} recordings")


def collect_scores(model, loader, device):
    """Return (labels, seizure-class probabilities) over the whole loader."""
    model.eval()
    labels, scores = [], []

    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.to(device)), dim=1)[:, 1]
            scores.extend(probs.cpu().numpy())
            labels.extend(y.numpy())

    return np.array(labels), np.array(scores)


def false_alarms_per_hour(fp, n_windows, stride_seconds):
    """False positives normalised by the hours of EEG the windows span."""
    hours = n_windows * stride_seconds / 3600.0
    return fp / hours if hours > 0 else float("nan")


def evaluate_at_threshold(labels, scores, threshold, stride_seconds):
    """Metrics for ``scores >= threshold``, including false alarms per hour."""
    preds = (scores >= threshold).astype(int)
    metrics = compute_metrics(labels, preds, scores)
    fp = int(((preds == 1) & (labels == 0)).sum())
    metrics["false_alarms_per_hour"] = false_alarms_per_hour(fp, len(labels), stride_seconds)
    metrics["threshold"] = float(threshold)
    return metrics


def choose_threshold(labels, scores, stride_seconds, fa_budget=5.0, steps=199):
    """Pick an operating point on the validation split.

    Returns the threshold with the highest sensitivity whose false-alarm rate
    stays within ``fa_budget`` per hour. If no threshold qualifies (or the
    split has no seizure windows), falls back to maximising Youden's J
    (sensitivity + specificity - 1).
    """
    thresholds = np.linspace(0, 1, steps + 2)[1:-1]
    rows = [evaluate_at_threshold(labels, scores, t, stride_seconds) for t in thresholds]

    viable = [r for r in rows if r["false_alarms_per_hour"] <= fa_budget]
    if viable and (labels == 1).any():
        best = max(viable, key=lambda r: (r["sensitivity"], r["specificity"]))
        return best["threshold"], "max sensitivity at <= {:g} FA/hr".format(fa_budget)

    best = max(rows, key=lambda r: r["sensitivity"] + r["specificity"])
    return best["threshold"], "Youden's J (no threshold met the FA budget)"


def report_operating_point(model, val_loader, test_loader, device, stride_seconds, fa_budget):
    """Choose a threshold on val, evaluate on test, and print both views."""
    val_labels, val_scores = collect_scores(model, val_loader, device)
    test_labels, test_scores = collect_scores(model, test_loader, device)

    threshold, rule = choose_threshold(val_labels, val_scores, stride_seconds, fa_budget)
    default = evaluate_at_threshold(test_labels, test_scores, 0.5, stride_seconds)
    chosen = evaluate_at_threshold(test_labels, test_scores, threshold, stride_seconds)

    print(f"Test AUC:                {chosen['auc']:.4f}")
    print(f"\nAt the default 0.5 threshold (argmax):")
    print(f"  Sensitivity: {default['sensitivity']:.3f}  Specificity: {default['specificity']:.3f}  "
          f"FA/hr: {default['false_alarms_per_hour']:.2f}")
    print(f"\nAt threshold {threshold:.3f}, chosen on validation by {rule}:")
    print(f"  Sensitivity: {chosen['sensitivity']:.3f}  Specificity: {chosen['specificity']:.3f}  "
          f"FA/hr: {chosen['false_alarms_per_hour']:.2f}  Accuracy: {chosen['accuracy']:.4f}")

    return chosen
