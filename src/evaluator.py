"""
evaluator.py
-------------
Evaluates trained models on the test set using several simple,
well-known classification metrics, and selects the best-performing
model automatically.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelEvaluator:
    """
    Compares multiple trained models on a held-out test set and
    reports Accuracy, Precision, Recall and F1-score (macro-averaged,
    since this is a multi-class problem with several crop classes).
    """

    def __init__(self):
        self.results_: pd.DataFrame | None = None

    def evaluate_all(self, trained_models: dict, X_test, y_test) -> pd.DataFrame:
        """
        Args:
            trained_models: {model_name: trained sklearn estimator}
            X_test, y_test: scaled/encoded test data

        Returns:
            pd.DataFrame of metrics, sorted best-to-worst by accuracy.
        """
        records = []
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)

            record = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "F1-score (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
            }
            records.append(record)

        self.results_ = (
            pd.DataFrame(records)
            .sort_values(by="Accuracy", ascending=False)
            .reset_index(drop=True)
        )
        return self.results_

    def get_best_model_name(self) -> str:
        """Returns the name of the highest-accuracy model."""
        if self.results_ is None or self.results_.empty:
            raise RuntimeError("Call evaluate_all() before requesting the best model.")
        return self.results_.iloc[0]["Model"]

    def print_report(self):
        """Pretty-prints the comparison table to the console."""
        if self.results_ is None:
            raise RuntimeError("Call evaluate_all() before printing a report.")

        print("\n" + "=" * 70)
        print("MODEL COMPARISON REPORT")
        print("=" * 70)
        print(self.results_.round(4).to_string(index=False))
        print("=" * 70)
        print(f"Best model: {self.get_best_model_name()} "
              f"(Accuracy = {self.results_.iloc[0]['Accuracy']:.4f})")
        print("=" * 70 + "\n")

    def save_report(self, path: str):
        """Saves the comparison table as a CSV for later reference."""
        if self.results_ is None:
            raise RuntimeError("Call evaluate_all() before saving a report.")
        self.results_.to_csv(path, index=False)
