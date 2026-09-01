"""
test_system.py
----------------
Lightweight sanity tests for the Smart Crop Recommendation System.

These are simple, dependency-free tests (using plain `assert`
statements) intended to demonstrate good practice for a student
project. Run with:

    python -m pytest tests/
    (or simply: python tests/test_system.py)
"""

import sys
import os

# Allow running this file directly from the tests/ folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_generator import DatasetGenerator
from src.validator import InputValidator, ValidationError
from src.config import FEATURE_NAMES, TARGET_NAME


def test_dataset_generator_produces_expected_columns():
    df = DatasetGenerator(samples_per_crop=5).generate()
    for col in FEATURE_NAMES + [TARGET_NAME]:
        assert col in df.columns, f"Missing column: {col}"
    print("test_dataset_generator_produces_expected_columns: PASSED")


def test_dataset_generator_produces_correct_row_count():
    generator = DatasetGenerator(samples_per_crop=10)
    df = generator.generate()
    expected_rows = 10 * len(generator._profiles)
    assert len(df) == expected_rows
    print("test_dataset_generator_produces_correct_row_count: PASSED")


def test_validator_accepts_valid_input():
    valid_input = {
        "N": 90, "P": 42, "K": 43, "temperature": 20.8,
        "humidity": 82.0, "ph": 6.5, "rainfall": 202.9,
    }
    cleaned = InputValidator.validate(valid_input)
    assert cleaned["N"] == 90.0
    print("test_validator_accepts_valid_input: PASSED")


def test_validator_rejects_out_of_range_input():
    invalid_input = {
        "N": 90, "P": 42, "K": 43, "temperature": 20.8,
        "humidity": 82.0, "ph": 25.0,  # invalid: pH can't be 25
        "rainfall": 202.9,
    }
    try:
        InputValidator.validate(invalid_input)
        assert False, "Expected a ValidationError to be raised"
    except ValidationError:
        print("test_validator_rejects_out_of_range_input: PASSED")


def test_validator_rejects_missing_feature():
    incomplete_input = {"N": 90, "P": 42}  # missing several features
    try:
        InputValidator.validate(incomplete_input)
        assert False, "Expected a ValidationError to be raised"
    except ValidationError:
        print("test_validator_rejects_missing_feature: PASSED")


if __name__ == "__main__":
    test_dataset_generator_produces_expected_columns()
    test_dataset_generator_produces_correct_row_count()
    test_validator_accepts_valid_input()
    test_validator_rejects_out_of_range_input()
    test_validator_rejects_missing_feature()
    print("\nAll tests passed!")
