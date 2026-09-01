"""
data_loader.py
---------------
Handles loading the crop dataset from disk, generating a synthetic
one if none exists yet, and splitting it into train/test sets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH, FEATURE_NAMES, TARGET_NAME, RANDOM_STATE, TEST_SIZE,
)
from src.dataset_generator import DatasetGenerator


class DataLoader:
    """
    Responsible for producing a clean train/test split of the
    crop recommendation dataset.

    If a dataset already exists at `dataset_path`, it is loaded
    directly (this is where you'd plug in a real-world CSV such
    as Kaggle's Crop_recommendation.csv). Otherwise, a synthetic
    dataset is generated automatically so the project works
    out-of-the-box.
    """

    def __init__(self, dataset_path: str = DATASET_PATH):
        self.dataset_path = dataset_path
        self.dataframe: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Loads the dataset from disk, generating it first if needed."""
        if not os.path.exists(self.dataset_path):
            print(f"[DataLoader] No dataset found at '{self.dataset_path}'.")
            print("[DataLoader] Generating a synthetic agricultural dataset...")
            generator = DatasetGenerator()
            self.dataframe = generator.save(self.dataset_path)
            print(f"[DataLoader] Synthetic dataset saved to '{self.dataset_path}' "
                  f"({len(self.dataframe)} rows).")
        else:
            self.dataframe = pd.read_csv(self.dataset_path)
            print(f"[DataLoader] Loaded existing dataset from '{self.dataset_path}' "
                  f"({len(self.dataframe)} rows).")

        self._validate_columns()
        return self.dataframe

    def _validate_columns(self):
        """Ensures the loaded dataset has the columns the project expects."""
        missing = [c for c in FEATURE_NAMES + [TARGET_NAME] if c not in self.dataframe.columns]
        if missing:
            raise ValueError(
                f"Dataset at '{self.dataset_path}' is missing required columns: {missing}. "
                f"Expected columns: {FEATURE_NAMES + [TARGET_NAME]}"
            )

    def get_train_test_split(self, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
        """
        Splits the loaded dataset into features/labels for training and testing.

        Returns:
            X_train, X_test, y_train, y_test (all pandas objects)
        """
        if self.dataframe is None:
            self.load()

        X = self.dataframe[FEATURE_NAMES]
        y = self.dataframe[TARGET_NAME]

        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,  # keeps class balance consistent across train/test
        )
