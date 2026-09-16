"""Sweep the decision threshold of a trained checkpoint.

The training scripts classify with ``argmax``, which is equivalent to a fixed
0.5 cutoff on the seizure probability. Under heavy class imbalance that cutoff
is often never crossed, giving sensitivity 0.000 even when the model ranks
seizure windows well (high AUC).

This script prints the full sensitivity / false-alarm curve on the test split,
but the operating point it recommends is chosen on the *validation* split —
tuning a threshold on test and then reporting test numbers is a form of
test-set leakage.

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
from train_utils import (choose_threshold, collect_scores, describe_split,
                         evaluate_at_threshold, split_indices)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep the decision threshold of a trained model")
    parser.add_argument("--data", type=Path, required=True, help="Path to CHB-MIT data directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .pth checkpoint")
    parser.add_argument("--model", choices=["1d", "2d"], required=True, help="Which architecture")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42, help="Must match the training seed")
    parser.add_argument("--split", choices=["recording", "window"], default="recording",
                        help="Must match the split used in training")
    parser.add_argument("--fa-budget", type=float, default=5.0,
                        help="Max false alarms per hour when choosing the threshold on validation")
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

    # Reuse the training split (same seed and strategy) so test stays held out.
    _, val_idx, test_idx = split_indices(base_dataset, args.seed, args.split)
    describe_split("Val", base_dataset, val_idx)
    describe_split("Test", base_dataset, test_idx)

    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    model = load_model(args.model, args.checkpoint, device)
    val_labels, val_scores = collect_scores(model, val_loader, device)
    test_labels, test_scores = collect_scores(model, test_loader, device)

    if (test_labels == 1).sum() == 0:
        print("ERROR: No seizure windows in the test split — sensitivity is undefined.")
        return

    print(f"\nTest AUC: {roc_auc_score(test_labels, test_scores):.4f}")

    stride = base_dataset.stride_seconds
    thresholds = np.linspace(0, 1, args.steps + 2)[1:-1]
    rows = [evaluate_at_threshold(test_labels, test_scores, t, stride) for t in thresholds]

    print(f"\nTest-split curve:")
    print(f"{'Thresh':>8}  {'Sens':>7}  {'Spec':>7}  {'FA/hr':>9}")
    print("-" * 38)
    for r in rows[::max(len(rows) // 20, 1)]:
        print(f"{r['threshold']:>8.3f}  {r['sensitivity']:>7.3f}  "
              f"{r['specificity']:>7.3f}  {r['false_alarms_per_hour']:>9.2f}")

    threshold, rule = choose_threshold(val_labels, val_scores, stride, args.fa_budget)
    chosen = evaluate_at_threshold(test_labels, test_scores, threshold, stride)
    print(f"\nOperating point chosen on validation ({rule}): threshold {threshold:.3f}")
    print(f"  Test sensitivity {chosen['sensitivity']:.3f}  specificity {chosen['specificity']:.3f}  "
          f"FA/hr {chosen['false_alarms_per_hour']:.2f}")

    if args.out:
        fields = ["threshold", "sensitivity", "specificity", "false_alarms_per_hour", "accuracy"]
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
