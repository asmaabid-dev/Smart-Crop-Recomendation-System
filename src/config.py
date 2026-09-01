"""
config.py
----------
Central place for project-wide constants.

Keeping constants in one file (instead of scattering "magic numbers"
across the codebase) makes the project easier to read, maintain and
extend -- a good habit for any real-world Python project.
"""

# The 7 input features used by every model in this project.
# Order matters: it must match the column order of the dataset
# and the order in which we build the feature vector for predictions.
FEATURE_NAMES = [
    "N",            # Nitrogen content in soil (kg/ha)
    "P",            # Phosphorus content in soil (kg/ha)
    "K",            # Potassium content in soil (kg/ha)
    "temperature",  # Average temperature (degrees Celsius)
    "humidity",     # Relative humidity (%)
    "ph",           # Soil pH value
    "rainfall",     # Rainfall (mm)
]

TARGET_NAME = "label"  # Column holding the crop name

# Sensible real-world ranges for each feature.
# Used by InputValidator to reject nonsensical user input
# (e.g. a negative temperature in Kelvin-speak, or pH of 50).
FEATURE_RANGES = {
    "N": (0, 140),
    "P": (0, 145),
    "K": (0, 205),
    "temperature": (-5, 55),
    "humidity": (0, 100),
    "ph": (0, 14),
    "rainfall": (0, 350),
}

# List of crops the synthetic dataset is built around.
# These mirror commonly grown crops used in agricultural ML datasets.
CROPS = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas",
    "mothbeans", "mungbean", "blackgram", "lentil", "pomegranate",
    "banana", "mango", "grapes", "watermelon", "muskmelon",
    "apple", "orange", "papaya", "coconut", "cotton", "jute", "coffee",
]

# File paths (kept relative so the project runs the same way
# on any machine / any contributor's laptop).
DATASET_PATH = "data/crop_data.csv"
MODEL_DIR = "saved_models"
BEST_MODEL_PATH = f"{MODEL_DIR}/best_model.joblib"
SCALER_PATH = f"{MODEL_DIR}/scaler.joblib"
LABEL_ENCODER_PATH = f"{MODEL_DIR}/label_encoder.joblib"
METRICS_REPORT_PATH = f"{MODEL_DIR}/model_comparison_report.csv"

RANDOM_STATE = 42   # Fixed seed -> reproducible results
TEST_SIZE = 0.2     # 80% train / 20% test split
