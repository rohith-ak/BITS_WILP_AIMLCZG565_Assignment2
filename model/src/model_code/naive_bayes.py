from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel


class NaiveBayesModel(BaseModel):
    def _create_model(self):
        return GaussianNB()

    def train(self, X_train, y_train):
        """Override to include grid search."""
        if self.model is None:
            self.model = self._create_model()

        param_grid = {}
        grid_search = GridSearchCV(self.model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
