"""
recommender.py
----------------
The heart of the project: CropRecommendationSystem ties together
data loading, preprocessing, model training, evaluation and
prediction into one clean, easy-to-use class.

This is the class both main.py (training pipeline) and predict.py
(interactive prediction) build on top of.
"""

import os
import joblib
import pandas as pd

from src.config import (
    FEATURE_NAMES, MODEL_DIR, BEST_MODEL_PATH, SCALER_PATH,
    LABEL_ENCODER_PATH, METRICS_REPORT_PATH,
)
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.models import ModelTrainer
from src.evaluator import ModelEvaluator
from src.validator import InputValidator, ValidationError


class CropRecommendationSystem:
    """
    High-level façade for the whole crop recommendation pipeline.

    Typical usage (training):
        system = CropRecommendationSystem()
        system.run_training_pipeline()

    Typical usage (prediction, after training):
        system = CropRecommendationSystem()
        system.load_trained_system()
        crop = system.recommend_crop({"N": 90, "P": 42, ...})
    """

    def __init__(self, dataset_path: str | None = None):
        self.data_loader = DataLoader(dataset_path) if dataset_path else DataLoader()
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.best_model = None
        self.best_model_name = None

    # ------------------------------------------------------------------
    # Training pipeline
    # ------------------------------------------------------------------
    def run_training_pipeline(self):
        """
        Executes the full pipeline:
        load data -> preprocess -> train all models -> evaluate ->
        pick the best model -> save everything to disk.
        """
        # 1. Load & split data
        self.data_loader.load()
        X_train, X_test, y_train, y_test = self.data_loader.get_train_test_split()

        # 2. Preprocess (scale features, encode labels)
        X_train_scaled, y_train_encoded = self.preprocessor.fit_transform(X_train, y_train)
        X_test_scaled, y_test_encoded = self.preprocessor.transform(X_test, y_test)

        # 3. Train every candidate model
        trained_models = self.trainer.train_all(X_train_scaled, y_train_encoded)

        # 4. Evaluate and compare
        self.evaluator.evaluate_all(trained_models, X_test_scaled, y_test_encoded)
        self.evaluator.print_report()

        # 5. Select the best model
        self.best_model_name = self.evaluator.get_best_model_name()
        self.best_model = trained_models[self.best_model_name]

        # 6. Persist everything needed for future predictions
        self._save_artifacts()

        return self.evaluator.results_

    def _save_artifacts(self):
        """Saves the best model, the scaler and the label encoder to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.best_model, BEST_MODEL_PATH)
        self.preprocessor.save(SCALER_PATH, LABEL_ENCODER_PATH)
        self.evaluator.save_report(METRICS_REPORT_PATH)
        print(f"[CropRecommendationSystem] Best model ('{self.best_model_name}') "
              f"saved to '{BEST_MODEL_PATH}'.")
        print(f"[CropRecommendationSystem] Comparison report saved to '{METRICS_REPORT_PATH}'.")

    # ------------------------------------------------------------------
    # Prediction pipeline
    # ------------------------------------------------------------------
    def load_trained_system(self):
        """
        Loads a previously trained model + preprocessor from disk so
        the system is ready to make predictions without retraining.
        """
        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(
                "No trained model found. Please run 'python main.py' first "
                "to train and save a model."
            )

        self.best_model = joblib.load(BEST_MODEL_PATH)
        self.preprocessor = DataPreprocessor.load(SCALER_PATH, LABEL_ENCODER_PATH)
        print("[CropRecommendationSystem] Trained model and preprocessor loaded.")

    def recommend_crop(self, feature_values: dict) -> str:
        """
        Predicts the most suitable crop for the given soil/environmental
        readings.

        Args:
            feature_values: dict with keys N, P, K, temperature,
                             humidity, ph, rainfall.

        Returns:
            str: the recommended crop name.

        Raises:
            ValidationError: if the input values are invalid.
        """
        if self.best_model is None:
            raise RuntimeError(
                "No model is loaded. Call load_trained_system() or "
                "run_training_pipeline() first."
            )

        # Validate & clean the input before doing anything else
        clean_values = InputValidator.validate(feature_values)

        # Build a DataFrame with columns in the exact order the model expects
        input_df = pd.DataFrame([[clean_values[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

        # Scale using the SAME scaler that was fit during training
        input_scaled = self.preprocessor.transform(input_df)

        # Predict and decode back to a human-readable crop name
        encoded_prediction = self.best_model.predict(input_scaled)[0]
        crop_name = self.preprocessor.decode_label(encoded_prediction)
        return crop_name

    def recommend_crop_with_confidence(self, feature_values: dict, top_n: int = 3):
        """
        Same as recommend_crop(), but also returns the top-N most likely
        crops with their predicted probabilities, when the underlying
        model supports probability estimates (predict_proba).

        Returns:
            list[tuple[str, float]]: [(crop_name, probability), ...]
            sorted from most to least likely.
        """
        if self.best_model is None:
            raise RuntimeError(
                "No model is loaded. Call load_trained_system() or "
                "run_training_pipeline() first."
            )
        if not hasattr(self.best_model, "predict_proba"):
            raise AttributeError(
                f"The best model ({type(self.best_model).__name__}) does not "
                f"support probability estimates."
            )

        clean_values = InputValidator.validate(feature_values)
        input_df = pd.DataFrame([[clean_values[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
        input_scaled = self.preprocessor.transform(input_df)

        probabilities = self.best_model.predict_proba(input_scaled)[0]
        class_names = self.preprocessor.label_encoder.inverse_transform(
            range(len(probabilities))
        )

        ranked = sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
