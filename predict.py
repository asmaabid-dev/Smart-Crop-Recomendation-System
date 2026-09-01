"""
predict.py
-----------
Entry point for USING the trained Smart Crop Recommendation System.

Run this after main.py has trained and saved a model. It prompts
the user for soil and environmental readings, validates them, and
prints the recommended crop (plus the top-3 most likely crops with
their confidence, when supported by the chosen model).

Usage:
    python predict.py
"""

from src.recommender import CropRecommendationSystem
from src.validator import ValidationError
from src.config import FEATURE_NAMES, FEATURE_RANGES


FEATURE_PROMPTS = {
    "N": "Nitrogen content in soil (N, kg/ha)",
    "P": "Phosphorus content in soil (P, kg/ha)",
    "K": "Potassium content in soil (K, kg/ha)",
    "temperature": "Average temperature (Celsius)",
    "humidity": "Relative humidity (%)",
    "ph": "Soil pH value",
    "rainfall": "Rainfall (mm)",
}


def get_user_input() -> dict:
    """Interactively collects and returns raw feature values from the user."""
    print("\nPlease enter the following soil and environmental values:")
    values = {}
    for feature in FEATURE_NAMES:
        low, high = FEATURE_RANGES[feature]
        prompt = f"  {FEATURE_PROMPTS[feature]} [{low}-{high}]: "
        while True:
            raw = input(prompt).strip()
            try:
                values[feature] = float(raw)
                break
            except ValueError:
                print("    Please enter a valid number.")
    return values


def main():
    print("=" * 70)
    print("SMART CROP RECOMMENDATION SYSTEM - PREDICTION")
    print("=" * 70)

    system = CropRecommendationSystem()
    system.load_trained_system()

    while True:
        user_values = get_user_input()

        try:
            crop = system.recommend_crop(user_values)
            print(f"\nRecommended crop: {crop.upper()}")

            try:
                top_matches = system.recommend_crop_with_confidence(user_values, top_n=3)
                print("\nTop 3 most likely crops:")
                for name, probability in top_matches:
                    print(f"  - {name:<15} {probability * 100:.1f}% confidence")
            except AttributeError:
                pass  # Selected model doesn't support probability estimates

        except ValidationError as e:
            print(f"\nInput error: {e}")

        again = input("\nWould you like another recommendation? (y/n): ").strip().lower()
        if again != "y":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
