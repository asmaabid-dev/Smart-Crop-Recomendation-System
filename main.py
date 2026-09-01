"""
main.py
--------
Entry point for TRAINING the Smart Crop Recommendation System.

Running this script will:
    1. Load (or generate) the agricultural dataset.
    2. Split it into training and testing sets.
    3. Train 5 different ML models: KNN, Decision Tree, Naive Bayes,
       Logistic Regression, and Random Forest.
    4. Evaluate each model using Accuracy, Precision, Recall and F1-score.
    5. Automatically select the best-performing model.
    6. Save the best model (plus the scaler & label encoder) to disk,
       ready to be used by predict.py.

Usage:
    python main.py
"""

from src.recommender import CropRecommendationSystem


def main():
    print("=" * 70)
    print("SMART CROP RECOMMENDATION SYSTEM - TRAINING PIPELINE")
    print("=" * 70)

    system = CropRecommendationSystem()
    system.run_training_pipeline()

    print("Training complete! Run 'python predict.py' to get crop recommendations.")


if __name__ == "__main__":
    main()
