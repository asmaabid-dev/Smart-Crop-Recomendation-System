"""
preprocessor.py
-----------------
Handles feature scaling and label encoding, and packages both
transformers together so they can be saved/loaded alongside a
trained model (this is essential: a model trained on scaled data
will make wrong predictions if you forget to scale new inputs
the same way).
"""

import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    """
    Wraps a StandardScaler (for the numeric features) and a
    LabelEncoder (for the crop-name target) into a single,
    convenient class.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._is_fitted = False

    def fit_transform(self, X_train, y_train):
        """Fits the scaler/encoder on training data and transforms it."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self._is_fitted = True
        return X_train_scaled, y_train_encoded

    def transform(self, X, y=None):
        """Transforms new data using the already-fitted scaler/encoder."""
        if not self._is_fitted:
            raise RuntimeError("DataPreprocessor must be fit before calling transform().")

        X_scaled = self.scaler.transform(X)
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        return X_scaled

    def decode_label(self, encoded_label):
        """Converts a numeric prediction back into a human-readable crop name."""
        return self.label_encoder.inverse_transform([encoded_label])[0]

    def save(self, scaler_path: str, encoder_path: str):
        """Persists the fitted scaler and encoder to disk."""
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)

    @classmethod
    def load(cls, scaler_path: str, encoder_path: str) -> "DataPreprocessor":
        """Rebuilds a DataPreprocessor from previously saved files."""
        preprocessor = cls()
        preprocessor.scaler = joblib.load(scaler_path)
        preprocessor.label_encoder = joblib.load(encoder_path)
        preprocessor._is_fitted = True
        return preprocessor
