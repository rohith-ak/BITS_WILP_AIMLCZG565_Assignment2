from sklearn.neighbors import KNeighborsClassifier
from .base_model import BaseModel


class KNNModel(BaseModel):
    def __init__(self, n_neighbors=5):
        super().__init__()
        self.n_neighbors = n_neighbors

    def _create_model(self):
        return KNeighborsClassifier(n_neighbors=self.n_neighbors)
