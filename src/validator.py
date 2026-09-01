"""
validator.py
-------------
Validates user-supplied soil/environmental readings before they are
fed into a trained model. Keeping validation separate from the
prediction logic follows the Single Responsibility Principle and
makes both classes easier to test and reuse.
"""

from src.config import FEATURE_NAMES, FEATURE_RANGES


class ValidationError(Exception):
    """Raised when user input fails validation."""
    pass


class InputValidator:
    """
    Validates a dictionary of feature values before prediction.

    Checks performed:
        1. All required features are present.
        2. Every value can be converted to a float.
        3. Every value falls within a realistic real-world range.
    """

    @staticmethod
    def validate(feature_values: dict) -> dict:
        """
        Args:
            feature_values: dict like {"N": 90, "P": 42, ...}

        Returns:
            dict: the same values, cleaned and cast to float.

        Raises:
            ValidationError: if any check fails, with a clear message.
        """
        missing = [f for f in FEATURE_NAMES if f not in feature_values]
        if missing:
            raise ValidationError(f"Missing required feature(s): {', '.join(missing)}")

        cleaned = {}
        for feature in FEATURE_NAMES:
            raw_value = feature_values[feature]
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                raise ValidationError(
                    f"'{feature}' must be a number, got: {raw_value!r}"
                )

            low, high = FEATURE_RANGES[feature]
            if not (low <= value <= high):
                raise ValidationError(
                    f"'{feature}' = {value} is outside the realistic range "
                    f"[{low}, {high}]. Please double-check this value."
                )

            cleaned[feature] = value

        return cleaned
