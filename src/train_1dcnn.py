"""Train the 1D CNN on raw multi-channel EEG windows.

Example:
    python src/train_1dcnn.py --data ./data --epochs 10 --use-class-weights
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data_loader import CHBMITDataset
from models import CNN1D
from train_utils import (describe_split, report_operating_point, split_indices,
                         train_epoch, validate_epoch)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 1D CNN on raw EEG windows (temporal conv across channels)"
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to CHB-MIT data directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-model", type=str, default="cnn1d.pth")
    parser.add_argument("--seed", type=int, default=42)

    # Caps for keeping development runs tractable on the full dataset.
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-files-per-patient", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)

    parser.add_argument(
        "--split",
        choices=["recording", "window"],
        default="recording",
        help="'recording' holds out whole EDF files (default). 'window' is the "
             "original leaky random-window split, kept for comparison.",
    )
    parser.add_argument(
        "--fa-budget",
        type=float,
        default=5.0,
        help="Max false alarms per hour when choosing the threshold on validation",
    )

    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use class-weighted loss (recommended for seizure imbalance)",
    )
    return parser.parse_args()


def build_criterion(base_dataset, train_indices, use_class_weights, device):
    """Cross-entropy loss, optionally reweighted to offset class imbalance."""
    if not use_class_weights:
        return nn.CrossEntropyLoss()

    train_labels = np.array([int(base_dataset[i][1]) for i in train_indices])

    num_pos = int((train_labels == 1).sum())
    num_neg = int((train_labels == 0).sum())
    print(f"Train class counts -> pos: {num_pos}, neg: {num_neg}")

    w_pos = num_neg / max(num_pos, 1)
    weights = torch.tensor([1.0, w_pos], dtype=torch.float32).to(device)
    print(f"Using class weights: [neg=1.0, pos={w_pos:.4f}]")

    return nn.CrossEntropyLoss(weight=weights)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading CHB-MIT dataset...")
    base_dataset = CHBMITDataset(
        args.data,
        max_patients=args.max_patients,
        max_files_per_patient=args.max_files_per_patient,
        max_windows=args.max_windows,
    )

    if len(base_dataset) == 0:
        print("ERROR: Dataset loaded 0 windows.")
        return

    train_idx, val_idx, test_idx = split_indices(base_dataset, args.seed, args.split)
    print(f"\nSplit strategy: {args.split}")
    describe_split("Train", base_dataset, train_idx)
    describe_split("Val", base_dataset, val_idx)
    describe_split("Test", base_dataset, test_idx)

    train_loader = DataLoader(Subset(base_dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(base_dataset, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(base_dataset, test_idx), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    print("\nCreating 1D CNN model...")
    model = CNN1D(in_channels=23, num_classes=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = build_criterion(base_dataset, train_idx, args.use_class_weights, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = -1.0
    best_epoch = 0

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(args.epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_metrics['accuracy']:.4f}, "
              f"Val AUC: {val_metrics['auc']:.4f}")

        # Select on AUC rather than accuracy — accuracy is ~1.0 for a model
        # that never predicts a seizure, so it can't rank checkpoints.
        val_auc = val_metrics["auc"] if not np.isnan(val_metrics["auc"]) else -1.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.save_model)
            print(f"  Saved best model weights (Val AUC: {best_val_auc:.4f})")

        print()

    if best_epoch > 0 and os.path.exists(args.save_model):
        print(f"\nLoading best model weights from epoch {best_epoch} (Val AUC: {best_val_auc:.4f})")
        model.load_state_dict(torch.load(args.save_model, map_location=device))
    else:
        print("\nNo best checkpoint saved; using final weights.")

    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    report_operating_point(
        model, val_loader, test_loader, device,
        stride_seconds=base_dataset.stride_seconds,
        fa_budget=args.fa_budget,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
