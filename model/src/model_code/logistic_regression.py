from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def _create_model(self):
        return LogisticRegression(max_iter=1000)
