"""
dataset_generator.py
---------------------
Generates a realistic, synthetic agricultural dataset for crop
recommendation when a real dataset (e.g. the popular Kaggle
"Crop Recommendation Dataset") is not available locally.

Each crop is modelled as a normal (Gaussian) distribution over the
7 input features, using mean/std values that reflect realistic
agronomic conditions for that crop (e.g. rice needs high rainfall
and humidity, while grapes prefer a drier climate).

This keeps the project fully self-contained and runnable offline,
while still producing data that behaves like real sensor/soil data
-- which is exactly what makes the ML comparison meaningful.

NOTE: If you have access to a real dataset (for example the Kaggle
"Crop_recommendation.csv" file), simply place it at data/crop_data.csv
with the same column names and DatasetGenerator will not be needed --
DataLoader will use the real file automatically.
"""

import os
import numpy as np
import pandas as pd

from src.config import FEATURE_NAMES, TARGET_NAME, CROPS, RANDOM_STATE


class DatasetGenerator:
    """
    Builds a synthetic but agronomically realistic crop dataset.

    Attributes:
        samples_per_crop (int): number of rows to generate per crop.
        random_state (int): seed for reproducibility.
    """

    def __init__(self, samples_per_crop: int = 120, random_state: int = RANDOM_STATE):
        self.samples_per_crop = samples_per_crop
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._profiles = self._build_crop_profiles()

    def _build_crop_profiles(self) -> dict:
        """
        Defines (mean, std) pairs for each feature, per crop.

        Format: {crop_name: {feature: (mean, std)}}
        Values are approximations based on typical agronomic
        requirements and are only meant to create *separable*,
        realistic-looking synthetic data for demonstration purposes.
        """
        # feature order: N, P, K, temperature, humidity, ph, rainfall
        raw_profiles = {
            "rice":        [(80, 15), (45, 10), (40, 10), (24, 3), (82, 5), (6.4, 0.5), (230, 40)],
            "maize":       [(75, 15), (45, 10), (20, 8),  (23, 4), (62, 8), (6.2, 0.5), (85, 25)],
            "chickpea":    [(40, 10), (60, 10), (80, 12), (19, 3), (17, 5), (7.2, 0.4), (70, 20)],
            "kidneybeans": [(20, 8),  (65, 10), (20, 8),  (18, 3), (21, 5), (5.8, 0.4), (100, 25)],
            "pigeonpeas":  [(20, 8),  (65, 10), (20, 8),  (27, 4), (48, 10),(5.9, 0.5), (150, 35)],
            "mothbeans":   [(20, 8),  (48, 10), (20, 8),  (28, 3), (53, 8), (6.8, 0.5), (50, 15)],
            "mungbean":    [(20, 8),  (48, 10), (20, 8),  (28, 3), (85, 5), (6.7, 0.4), (48, 12)],
            "blackgram":   [(40, 10), (65, 10), (19, 8),  (29, 3), (65, 8), (7.1, 0.4), (68, 18)],
            "lentil":      [(19, 8),  (68, 10), (19, 8),  (24, 3), (65, 8), (6.9, 0.4), (46, 12)],
            "pomegranate": [(19, 8),  (18, 8),  (40, 10), (21, 3), (90, 4), (6.4, 0.4), (105, 20)],
            "banana":      [(100, 15),(82, 10), (50, 10), (27, 2), (80, 5), (6.0, 0.4), (105, 20)],
            "mango":       [(20, 8),  (27, 8),  (30, 8),  (31, 3), (50, 8), (5.8, 0.4), (95, 20)],
            "grapes":      [(19, 8),  (135, 10),(200, 10),(24, 3), (82, 5), (6.0, 0.4), (68, 15)],
            "watermelon":  [(100, 12),(17, 8),  (50, 10), (25, 3), (85, 5), (6.5, 0.4), (48, 12)],
            "muskmelon":   [(100, 12),(17, 8),  (50, 10), (28, 3), (92, 3), (6.4, 0.4), (24, 8)],
            "apple":       [(20, 8),  (135, 10),(200, 10),(22, 3), (92, 3), (6.0, 0.4), (112, 20)],
            "orange":      [(19, 8),  (16, 8),  (10, 5),  (22, 3), (92, 3), (6.9, 0.4), (110, 20)],
            "papaya":      [(50, 12), (58, 10), (50, 10), (33, 3), (92, 3), (6.7, 0.4), (145, 25)],
            "coconut":     [(21, 8),  (16, 8),  (30, 8),  (27, 2), (95, 3), (5.9, 0.4), (175, 25)],
            "cotton":      [(118, 12),(46, 10), (19, 8),  (24, 3), (80, 5), (6.9, 0.4), (80, 20)],
            "jute":        [(78, 12), (47, 10), (40, 10), (25, 2), (80, 5), (6.7, 0.4), (175, 25)],
            "coffee":      [(100, 12),(28, 8),  (30, 8),  (25, 3), (58, 8), (6.8, 0.4), (160, 25)],
        }
        return {
            crop: dict(zip(FEATURE_NAMES, values))
            for crop, values in raw_profiles.items()
            if crop in CROPS
        }

    def generate(self) -> pd.DataFrame:
        """
        Generates the full synthetic dataset as a pandas DataFrame.

        Returns:
            pd.DataFrame: columns = FEATURE_NAMES + [TARGET_NAME]
        """
        rows = []
        for crop, feature_stats in self._profiles.items():
            for _ in range(self.samples_per_crop):
                row = {}
                for feature in FEATURE_NAMES:
                    mean, std = feature_stats[feature]
                    value = self._rng.normal(mean, std)
                    row[feature] = value
                row[TARGET_NAME] = crop
                rows.append(row)

        df = pd.DataFrame(rows)

        # Clip values to physically sensible bounds
        # (a normal distribution can occasionally dip below 0, etc.)
        df["N"] = df["N"].clip(0, 150)
        df["P"] = df["P"].clip(0, 150)
        df["K"] = df["K"].clip(0, 210)
        df["temperature"] = df["temperature"].clip(-5, 55)
        df["humidity"] = df["humidity"].clip(0, 100)
        df["ph"] = df["ph"].clip(0, 14)
        df["rainfall"] = df["rainfall"].clip(0, 350)

        # Shuffle rows so crops aren't grouped together
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        return df

    def save(self, path: str) -> pd.DataFrame:
        """Generates the dataset and writes it to disk as a CSV."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.generate()
        df.to_csv(path, index=False)
        return df
