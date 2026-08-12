"""Sweep the decision threshold of a trained checkpoint.

The training scripts classify with ``argmax``, which is equivalent to a fixed
0.5 cutoff on the seizure probability. Under heavy class imbalance that cutoff
is often never crossed, giving sensitivity 0.000 even when the model ranks
seizure windows well (high AUC). This script evaluates every threshold instead
and reports the sensitivity / false-alarm tradeoff, so an operating point can be
chosen deliberately.

Requires a trained checkpoint, which is not committed to this repository.

Example:
    python src/threshold_sweep.py --data ./data --checkpoint cnn2d.pth --model 2d
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from data_loader import CHBMITDataset, SpectrogramDataset
from models import CNN1D, CNN2D
from train_utils import random_split_indices


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep the decision threshold of a trained model")
    parser.add_argument("--data", type=Path, required=True, help="Path to CHB-MIT data directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .pth checkpoint")
    parser.add_argument("--model", choices=["1d", "2d"], required=True, help="Which architecture")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42, help="Must match the training seed")
    parser.add_argument("--steps", type=int, default=99, help="Number of thresholds to try")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV output path")

    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-files-per-patient", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)

    return parser.parse_args()


def load_model(kind, checkpoint_path, device):
    """Load either architecture from a raw state_dict or a wrapped checkpoint."""
    model = CNN1D(in_channels=23, num_classes=2) if kind == "1d" else CNN2D(num_classes=2)

    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob

    model.load_state_dict(state)
    return model.to(device).eval()


def collect_scores(model, loader, device):
    """Return (labels, seizure-class probabilities) over the whole loader."""
    labels, scores = [], []

    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.to(device)), dim=1)[:, 1]
            scores.extend(probs.cpu().numpy())
            labels.extend(y.numpy())

    return np.array(labels), np.array(scores)


def sweep(labels, scores, thresholds, stride_seconds):
    """Evaluate every threshold. Returns a list of per-threshold metric dicts."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    # Each window advances the recording by `stride` seconds, so the test split
    # represents this many hours of EEG.
    hours = len(labels) * stride_seconds / 3600.0

    rows = []
    for t in thresholds:
        pred = (scores >= t).astype(int)

        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())

        rows.append({
            "threshold": t,
            "sensitivity": tp / n_pos if n_pos else 0.0,
            "specificity": (n_neg - fp) / n_neg if n_neg else 0.0,
            "false_alarms_per_hour": fp / hours if hours else float("nan"),
            "tp": tp,
            "fp": fp,
        })

    return rows


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dataset = CHBMITDataset(
        args.data,
        max_patients=args.max_patients,
        max_files_per_patient=args.max_files_per_patient,
        max_windows=args.max_windows,
    )
    if len(base_dataset) == 0:
        print("ERROR: Dataset loaded 0 windows.")
        return

    dataset = base_dataset if args.model == "1d" else SpectrogramDataset(base_dataset)

    # Reuse the training split so the test set is genuinely held out.
    _, _, test_idx = random_split_indices(len(dataset), args.seed)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    model = load_model(args.model, args.checkpoint, device)
    labels, scores = collect_scores(model, test_loader, device)

    n_pos = int((labels == 1).sum())
    print(f"\nTest windows: {len(labels)}  (seizure: {n_pos}, non-seizure: {len(labels) - n_pos})")
    if n_pos == 0:
        print("ERROR: No seizure windows in the test split — sensitivity is undefined.")
        return

    print(f"AUC: {roc_auc_score(labels, scores):.4f}")

    stride_seconds = base_dataset.window_size * (1 - base_dataset.overlap)
    thresholds = np.linspace(0, 1, args.steps + 2)[1:-1]
    rows = sweep(labels, scores, thresholds, stride_seconds)

    print(f"\n{'Thresh':>8}  {'Sens':>7}  {'Spec':>7}  {'FA/hr':>9}")
    print("-" * 38)
    for r in rows[::max(len(rows) // 20, 1)]:
        print(f"{r['threshold']:>8.3f}  {r['sensitivity']:>7.3f}  "
              f"{r['specificity']:>7.3f}  {r['false_alarms_per_hour']:>9.2f}")

    # A reasonable default operating point: the highest sensitivity reachable
    # while keeping false alarms tolerable for a monitoring setting.
    for budget in (1.0, 5.0, 10.0):
        viable = [r for r in rows if r["false_alarms_per_hour"] <= budget]
        if viable:
            best = max(viable, key=lambda r: r["sensitivity"])
            print(f"\nAt <= {budget:g} false alarms/hour: sensitivity {best['sensitivity']:.3f} "
                  f"at threshold {best['threshold']:.3f}")
        else:
            print(f"\nAt <= {budget:g} false alarms/hour: no threshold qualifies")

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
