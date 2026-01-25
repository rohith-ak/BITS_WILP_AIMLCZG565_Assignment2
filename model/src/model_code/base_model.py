from abc import ABC, abstractmethod
import joblib
import numpy as np
from model.src.metrics_generation.metrics import calculate_metrics


class BaseModel(ABC):
    """Abstract base class for all ML models with class weight support."""

    def __init__(self):
        self.model = None
        self.use_class_weights = True  # Enable by default for imbalanced data

    @abstractmethod
    def _create_model(self):
        """Create and return the sklearn model instance."""
        pass

    def train(self, X_train, y_train):
        """Train the model with automatic class weight balancing."""
        if self.model is None:
            self.model = self._create_model()

        # Apply class weights if the model supports it
        if self.use_class_weights:
            if hasattr(self.model, 'class_weight'):
                # Models like Logistic Regression, SVM, Decision Tree
                try:
                    self.model.set_params(class_weight='balanced')
                    print(f"✅ Class weights enabled: balanced")
                except:
                    pass
            elif hasattr(self.model, 'scale_pos_weight'):
                # XGBoost
                unique, counts = np.unique(y_train, return_counts=True)
                if len(counts) == 2:
                    scale_pos_weight = counts[0] / counts[1]
                    try:
                        self.model.set_params(scale_pos_weight=scale_pos_weight)
                        print(f"⚖️ XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
                    except:
                        pass

        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

    def evaluate(self, X_test, y_test):
        """Evaluate model and return metrics."""
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        if y_prob is not None:
            y_prob = y_prob[:, 1]
        return calculate_metrics(y_test, y_pred, y_prob)

    def save_model(self, file_path):
        """Save model to disk."""
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        """Load model from disk."""
        self.model = joblib.load(file_path)
