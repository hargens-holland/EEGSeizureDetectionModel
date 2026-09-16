"""Classical ML baselines on flattened EEG windows.

These exist as a comparison point for the CNNs: they show what happens when a
model without temporal or spectral structure meets a heavily imbalanced dataset.

Uses the same recording-level split and validation-chosen threshold as the CNN
scripts, so the numbers are directly comparable.

Example:
    python src/baseline_models.py --data ./data --model all
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from data_loader import CHBMITDataset
from metrics import compute_metrics
from train_utils import choose_threshold, describe_split, evaluate_at_threshold, split_indices


SplitData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def prepare_dataset(dataset: CHBMITDataset, split: str, seed: int) -> SplitData:
    """Flatten windows and partition them with the shared split helper."""
    X = dataset.windows.astype(np.float32)
    y = dataset.labels.astype(np.int64)
    X = X.reshape(len(y), -1)

    train_idx, val_idx, test_idx = split_indices(dataset, seed, split)
    print(f"\nSplit strategy: {split}")
    describe_split("Train", dataset, train_idx)
    describe_split("Val", dataset, val_idx)
    describe_split("Test", dataset, test_idx)

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def get_prediction_scores(model, X: np.ndarray) -> np.ndarray | None:
    """Return probability or decision scores when available."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def train_and_evaluate(
    name: str,
    model,
    use_scaler: bool,
    splits: SplitData,
    stride_seconds: float,
    fa_budget: float,
) -> Dict[str, float | str]:
    """Fit the model, choose a threshold on val, and evaluate on test."""
    X_train, y_train, X_val, y_val, X_test, y_test = splits

    scaler = StandardScaler() if use_scaler else None

    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)

    val_scores = get_prediction_scores(model, X_val)
    test_scores = get_prediction_scores(model, X_test)

    # Default operating point: whatever ``predict`` does (0.5 on probabilities).
    default = compute_metrics(y_test, model.predict(X_test), test_scores)

    # Chosen operating point: threshold picked on validation, applied to test.
    threshold, rule = choose_threshold(y_val, val_scores, stride_seconds, fa_budget)
    chosen = evaluate_at_threshold(y_test, test_scores, threshold, stride_seconds)

    return {
        "model": name,
        "test_auc": chosen["auc"],
        "default_sensitivity": default["sensitivity"],
        "default_specificity": default["specificity"],
        "threshold": threshold,
        "threshold_rule": rule,
        "sensitivity": chosen["sensitivity"],
        "specificity": chosen["specificity"],
        "false_alarms_per_hour": chosen["false_alarms_per_hour"],
        "accuracy": chosen["accuracy"],
    }


def build_models() -> Dict[str, Tuple[object, bool]]:
    """Create baseline models and indicate whether scaling is required."""
    models: Dict[str, Tuple[object, bool]] = {}

    # Logistic Regression
    models["logistic_regression"] = (
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        True,
    )

    # SVM
    models["svm"] = (
        SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            gamma="scale",
        ),
        True,
    )

    # Random Forest
    models["random_forest"] = (
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        False,
    )

    return models


def main():
    parser = argparse.ArgumentParser(description="Train baseline ML models on EEG dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to root directory containing dataset",
    )
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
    # Model selection
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "svm", "random_forest", "all"],
        default="all",
        help="Which model to train (default: all baselines).",
    )
    args = parser.parse_args()

    dataset = CHBMITDataset(
        args.data,
        max_patients=args.max_patients,
        max_files_per_patient=args.max_files_per_patient,
        max_windows=args.max_windows,
    )
    if len(dataset) == 0:
        print("ERROR: Dataset loaded 0 windows.")
        return

    splits = prepare_dataset(dataset, args.split, args.seed)
    models = build_models()

    selected_models = (
        models.items()
        if args.model == "all"
        else [(args.model, models[args.model])]
    )

    for name, (model, use_scaler) in selected_models:
        print(f"\nTraining {name} baseline...")
        r = train_and_evaluate(name, model, use_scaler, splits, dataset.stride_seconds, args.fa_budget)

        print(f"{name}")
        print(f"  Test AUC:                {r['test_auc']:.4f}")
        print(f"  Default threshold:       Sensitivity {r['default_sensitivity']:.3f}  "
              f"Specificity {r['default_specificity']:.3f}")
        print(f"  Threshold {r['threshold']:.3f} ({r['threshold_rule']}):")
        print(f"    Sensitivity {r['sensitivity']:.3f}  Specificity {r['specificity']:.3f}  "
              f"FA/hr {r['false_alarms_per_hour']:.2f}  Accuracy {r['accuracy']:.4f}")


if __name__ == "__main__":
    main()

