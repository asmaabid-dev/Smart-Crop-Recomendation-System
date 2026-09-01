# 🌾 Smart Crop Recommendation System Using Machine Learning

A beginner-friendly, professionally structured Python project that recommends the
most suitable crop to grow based on soil and environmental conditions — built with
clean **Object-Oriented Programming (OOP)** and **scikit-learn**.

The system trains and compares **five** Machine Learning algorithms, automatically
selects the best-performing one, and lets the user enter soil/climate readings to
get an instant crop recommendation.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Machine Learning Models](#-machine-learning-models)
- [Installation](#-installation)
- [Usage](#-usage)
- [Sample Output](#-sample-output)
- [How It Works](#-how-it-works)
- [Testing](#-testing)
- [Possible Improvements](#-possible-improvements)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

Choosing the right crop for a given field depends on multiple soil and climate
factors — Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH,
and rainfall. This project uses supervised Machine Learning to learn the
relationship between these factors and the best-suited crop, then exposes that
knowledge through a simple command-line interface.

It is designed to be:
- **Readable** — every class has a single, clear responsibility.
- **Extensible** — adding a new ML model or feature takes minutes.
- **Educational** — ideal for a Computer Science student's resume/portfolio,
  demonstrating OOP design, the ML workflow, and software engineering practices
  (validation, testing, project structure) rather than just a single script.

---

## ✨ Features

- Clean **OOP architecture** — one class per responsibility (data loading,
  preprocessing, training, evaluation, validation, recommendation).
- Trains and compares **5 ML algorithms**: KNN, Decision Tree, Naive Bayes,
  Logistic Regression, and Random Forest.
- Evaluates each model with **Accuracy, Precision, Recall, and F1-score**.
- **Automatically selects and saves the best model** — no manual comparison needed.
- **Robust input validation** with clear, human-readable error messages.
- Works **out of the box** — generates a realistic synthetic agricultural dataset
  if no dataset is provided (or drop in a real dataset with the same columns).
- Interactive CLI for entering new readings and getting recommendations, including
  **confidence scores** for the top 3 most likely crops.
- Includes a small **automated test suite**.

---

## 🗂 Project Structure

```
smart_crop_recommendation/
│
├── main.py                     # Entry point: trains & compares all models
├── predict.py                  # Entry point: interactive crop prediction
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
├── .gitignore
│
├── src/                         # Core application package
│   ├── __init__.py
│   ├── config.py                # Constants: feature names, ranges, file paths
│   ├── dataset_generator.py     # Generates a realistic synthetic dataset
│   ├── data_loader.py           # Loads data & creates train/test splits
│   ├── preprocessor.py          # Feature scaling + label encoding
│   ├── models.py                # ML model definitions (ModelFactory, ModelTrainer)
│   ├── evaluator.py             # Model evaluation & comparison
│   ├── validator.py             # User input validation
│   └── recommender.py           # High-level CropRecommendationSystem class
│
├── data/
│   └── crop_data.csv            # Dataset (auto-generated on first run)
│
├── saved_models/                # Trained model + scaler + encoder (auto-generated)
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   └── model_comparison_report.csv
│
└── tests/
    └── test_system.py           # Basic automated tests
```

---

## 🌱 Dataset

The system expects 7 numeric input features and 1 target label:

| Feature       | Description                          | Typical Range |
|---------------|---------------------------------------|----------------|
| `N`           | Nitrogen content in soil (kg/ha)      | 0 – 140        |
| `P`           | Phosphorus content in soil (kg/ha)    | 0 – 145        |
| `K`           | Potassium content in soil (kg/ha)     | 0 – 205        |
| `temperature` | Average temperature (°C)              | -5 – 55        |
| `humidity`    | Relative humidity (%)                 | 0 – 100        |
| `ph`          | Soil pH value                         | 0 – 14         |
| `rainfall`    | Rainfall (mm)                         | 0 – 350        |
| `label`       | Target crop (e.g. rice, maize, ...)   | 22 crop types  |

**No dataset? No problem.** If `data/crop_data.csv` doesn't exist, `DatasetGenerator`
automatically creates a realistic synthetic dataset (~2,600 rows across 22 crops)
modeled on typical agronomic conditions, so the project runs immediately after
cloning.

**Want to use a real dataset?** Simply place a CSV with the same column names
(e.g. the popular Kaggle "Crop Recommendation Dataset") at `data/crop_data.csv` —
`DataLoader` will use it automatically instead of generating synthetic data.

---

## 🤖 Machine Learning Models

The following classifiers are trained and compared on identical data splits:

| Model                  | scikit-learn Class          |
|-------------------------|-------------------------------|
| K-Nearest Neighbors     | `KNeighborsClassifier`        |
| Decision Tree           | `DecisionTreeClassifier`      |
| Naive Bayes             | `GaussianNB`                  |
| Logistic Regression     | `LogisticRegression`          |
| Random Forest           | `RandomForestClassifier`      |

Each model is evaluated on a held-out test set (20% of the data) using:

- **Accuracy**
- **Precision (macro-averaged)**
- **Recall (macro-averaged)**
- **F1-score (macro-averaged)**

The model with the **highest accuracy** is automatically chosen, saved to
`saved_models/best_model.joblib`, and used for all future predictions.

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/smart-crop-recommendation.git
   cd smart-crop-recommendation
   ```

2. **(Recommended) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### 1. Train the models

```bash
python main.py
```

This will load/generate the dataset, train all 5 models, print a comparison
report, and save the best model to `saved_models/`.

### 2. Get a crop recommendation

```bash
python predict.py
```

You'll be prompted to enter N, P, K, temperature, humidity, pH, and rainfall
values. The system validates your input and returns the recommended crop along
with the top-3 most likely crops and their confidence scores.

### 3. Use it in your own code

```python
from src.recommender import CropRecommendationSystem

system = CropRecommendationSystem()
system.load_trained_system()

reading = {
    "N": 90, "P": 42, "K": 43,
    "temperature": 24, "humidity": 82,
    "ph": 6.5, "rainfall": 230,
}

print(system.recommend_crop(reading))
# -> "rice"
```

---

## 🖥 Sample Output

**Training (`python main.py`):**

```
======================================================================
MODEL COMPARISON REPORT
======================================================================
              Model  Accuracy  Precision (macro)  Recall (macro)  F1-score (macro)
        Naive Bayes    0.9337              0.9375           0.9337             0.9341
      Random Forest    0.9299              0.9359           0.9299             0.9298
Logistic Regression    0.9299              0.9331           0.9299             0.9299
K-Nearest Neighbors    0.9148              0.9186           0.9148             0.9149
      Decision Tree    0.8561              0.8643           0.8561             0.8563
======================================================================
Best model: Naive Bayes (Accuracy = 0.9337)
======================================================================
```

**Prediction (`python predict.py`):**

```
Recommended crop: RICE

Top 3 most likely crops:
  - rice            90.1% confidence
  - jute             9.9% confidence
  - coffee           0.0% confidence
```

*(Exact numbers vary slightly each run since the synthetic dataset is randomly
generated — a real dataset will give consistent, reproducible results.)*

---

## 🧠 How It Works

1. **`DataLoader`** loads the dataset (generating a synthetic one via
   `DatasetGenerator` if needed) and splits it into train/test sets.
2. **`DataPreprocessor`** scales numeric features with `StandardScaler` and
   encodes crop names with `LabelEncoder`.
3. **`ModelTrainer`** (using `ModelFactory`) trains all 5 candidate models on
   the same scaled training data.
4. **`ModelEvaluator`** scores every model on the test set and ranks them.
5. **`CropRecommendationSystem`** selects the top-ranked model, saves it (along
   with the fitted scaler/encoder) to disk with `joblib`.
6. At prediction time, **`InputValidator`** checks that user input is complete,
   numeric, and within realistic ranges before the saved model makes a
   prediction — preventing garbage-in-garbage-out errors.

This is a textbook example of a clean **ML pipeline**: *load → preprocess →
train → evaluate → select → persist → predict*, with each stage isolated in
its own class.

---

## 🧪 Testing

Run the included test suite to verify the core components:

```bash
python -m pytest tests/
# or simply:
python tests/test_system.py
```

Tests cover dataset generation, and input validation (accepting valid data,
rejecting out-of-range values, and rejecting incomplete input).

---

## 🚀 Possible Improvements

Ideas for extending this project further:

- Add a **web interface** (Flask/Streamlit) on top of `CropRecommendationSystem`.
- Add **hyperparameter tuning** (e.g. `GridSearchCV`) for each model.
- Add a **confusion matrix visualization** per model in `evaluator.py`.
- Support **multi-label recommendations** (e.g. top-3 crops instead of just one).
- Integrate a **real-time weather API** to auto-fill temperature/humidity/rainfall.

---

## 🛠 Tech Stack

- **Python 3.10+**
- **pandas** & **NumPy** — data handling
- **scikit-learn** — Machine Learning models & preprocessing
- **joblib** — model persistence

---

## 📄 License

This project is open-source and available for learning and portfolio purposes.
Feel free to fork, modify, and build upon it.
