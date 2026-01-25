from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(self, params=None):
        super().__init__()
        self.params = params or {}

    def _create_model(self):
        return DecisionTreeClassifier(**self.params)
