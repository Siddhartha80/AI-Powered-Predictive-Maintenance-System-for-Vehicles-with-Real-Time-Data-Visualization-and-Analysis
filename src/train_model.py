"""Train and evaluate the engine-condition Gradient Boosting classifier.

Reproduces the model developed in ``notebooks/Data_Preprocessing.ipynb``:
same engineered feature, same 60/40 split, same hyperparameters.

Usage (from the repository root):

    python src/train_model.py
    python src/train_model.py --output models/model.pkl --roc-plot images/roc_curve.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
TARGET = "Engine Condition"


def load_data(path: Path) -> pd.DataFrame:
    """Load the sensor data and add the engineered temperature feature."""
    data = pd.read_csv(path)
    data["Temperature_difference"] = data["Coolant temp"] - data["lub oil temp"]
    return data


def build_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        max_features="sqrt",
        min_samples_leaf=5,
        min_samples_split=2,
        subsample=0.8,
    )


def save_roc_plot(y_true, y_score, auc: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(fpr, tpr, color="#1f6feb", linewidth=2, label=f"Gradient Boosting (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve: engine condition classifier (test set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "Data" / "engine_data.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "model.pkl",
                        help="Where to save the trained model (pickle).")
    parser.add_argument("--roc-plot", type=Path, default=None,
                        help="Optional path for a ROC curve PNG.")
    parser.add_argument("--test-size", type=float, default=0.4)
    args = parser.parse_args()

    data = load_data(args.data)
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    model = build_model().fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)

    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:  {auc:.4f}\n")
    print(classification_report(y_test, y_pred, digits=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {args.output}")

    if args.roc_plot:
        save_roc_plot(y_test, y_score, auc, args.roc_plot)
        print(f"ROC plot saved to {args.roc_plot}")


if __name__ == "__main__":
    main()
