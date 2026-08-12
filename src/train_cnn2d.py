"""Train the 2D CNN on STFT spectrograms of EEG windows.

Example:
    python src/train_cnn2d.py --data ./data --epochs 30
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data_loader import CHBMITDataset, SpectrogramDataset
from models import CNN2D
from train_utils import random_split_indices, train_epoch, validate_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2D CNN on EEG spectrograms")
    parser.add_argument("--data", type=Path, required=True, help="Path to CHB-MIT data directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save-model", type=str, default="cnn2d.pth")
    parser.add_argument("--nperseg", type=int, default=64, help="STFT segment length")
    parser.add_argument("--noverlap", type=int, default=32, help="STFT overlap")
    parser.add_argument("--seed", type=int, default=42)

    # Caps for keeping development runs tractable on the full dataset.
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-files-per-patient", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not args.data.exists() or not args.data.is_dir():
        print(f"ERROR: Data directory does not exist: {args.data}")
        return

    print("\nLoading CHB-MIT dataset...")
    base_dataset = CHBMITDataset(
        args.data,
        max_patients=args.max_patients,
        max_files_per_patient=args.max_files_per_patient,
        max_windows=args.max_windows,
    )

    if len(base_dataset) == 0:
        print("ERROR: Dataset loaded 0 windows. Check that --data points at a "
              "directory of patient folders (chb01/, chb02/, ...) containing .edf files.")
        return

    print("Creating spectrogram dataset...")
    dataset = SpectrogramDataset(base_dataset, nperseg=args.nperseg, noverlap=args.noverlap)

    train_idx, val_idx, test_idx = random_split_indices(len(dataset), args.seed)
    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples:   {len(val_idx)}")
    print(f"Test samples:  {len(test_idx)}")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=pin)

    print("\nCreating 2D CNN model...")
    model = CNN2D(num_classes=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = 0.0
    best_epoch = 0

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(args.epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val AUC: {val_metrics['auc']:.4f}")

        # Select on AUC rather than accuracy — accuracy is ~1.0 for a model
        # that never predicts a seizure, so it can't rank checkpoints.
        val_auc = val_metrics['auc'] if not np.isnan(val_metrics['auc']) else 0.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_metrics': val_metrics,
            }, args.save_model)
            print(f"  Saved best model (Val AUC: {val_auc:.4f})")

        print()

    if best_epoch > 0 and os.path.exists(args.save_model):
        print(f"\nLoaded best model from epoch {best_epoch}")
        checkpoint = torch.load(args.save_model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("\nNo best model checkpoint was saved (AUC was NaN every epoch).")
        print("Using the final epoch model weights for test evaluation.")

    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    _, test_metrics = validate_epoch(model, test_loader, criterion, device)

    print(f"Test Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Test Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"Test Specificity: {test_metrics['specificity']:.4f}")
    print(f"Test AUC:         {test_metrics['auc']:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
