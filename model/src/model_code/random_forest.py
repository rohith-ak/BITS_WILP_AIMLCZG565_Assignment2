from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    def _create_model(self):
        return RandomForestClassifier()

    def train(self, X_train, y_train):
        """Override to include randomized search."""
        if self.model is None:
            self.model = self._create_model()

        param_dist = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        random_search = RandomizedSearchCV(
            self.model,
            param_dist,
            n_iter=10,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
