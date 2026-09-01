"""
models.py
----------
Defines the set of Machine Learning algorithms compared in this
project, and a ModelTrainer class that trains all of them in a
consistent, reusable way.

Adding a new algorithm to compare only requires adding one line
to ModelFactory.build_all() -- everything else (evaluation, best
model selection, saving) works automatically.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE


class ModelFactory:
    """
    Creates fresh, un-trained instances of each candidate model
    with sensible, beginner-friendly hyperparameters.
    """

    @staticmethod
    def build_all() -> dict:
        """
        Returns:
            dict: {model_name: un-trained sklearn estimator}
        """
        return {
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE
            ),
        }


class ModelTrainer:
    """
    Trains every model produced by ModelFactory on the same
    (scaled) training data, and keeps them ready for evaluation.
    """

    def __init__(self):
        self.models = ModelFactory.build_all()
        self.trained_models = {}

    def train_all(self, X_train, y_train) -> dict:
        """
        Fits every model on the training set.

        Returns:
            dict: {model_name: trained sklearn estimator}
        """
        for name, model in self.models.items():
            print(f"[ModelTrainer] Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model

        print(f"[ModelTrainer] Finished training {len(self.trained_models)} models.")
        return self.trained_models
