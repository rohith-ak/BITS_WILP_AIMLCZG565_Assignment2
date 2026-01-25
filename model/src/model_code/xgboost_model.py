from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel


class XGBoostModel(BaseModel):
    def _create_model(self):
        return XGBClassifier(eval_metric='logloss')

    def train(self, X_train, y_train):
        """Override to include grid search."""
        if self.model is None:
            self.model = self._create_model()

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            scoring='roc_auc',
            cv=5
        )

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
