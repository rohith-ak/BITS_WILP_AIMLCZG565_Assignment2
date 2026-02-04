from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
# Computing classification metrics for model evaluation 

def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)

    return metrics


# Alias for compatibility
def compute_metrics(y_true, y_pred, y_prob=None):
    return calculate_metrics(y_true, y_pred, y_prob)


# Individual metric functions for test compatibility
def accuracy(y_pred, y_true):
    return accuracy_score(y_true, y_pred)


def precision(y_pred, y_true):
    return precision_score(y_true, y_pred)


def recall(y_pred, y_true):
    return recall_score(y_true, y_pred)
