# evaluate_model.py

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_accuracy(y_true, y_pred):
    """Evaluate model accuracy."""
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def evaluate_precision_recall(y_true, y_pred):
    """Evaluate precision, recall, and F1 score."""
    prec, recall, fscore, supp = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return prec, recall, fscore, supp

def evaluate_model(model, X_test, y_test):
    """Evaluate a machine learning model."""
    y_pred = model.predict(X_test)
    
    accuracy = evaluate_accuracy(y_test, y_pred)
    precision, recall, fscore, _ = evaluate_precision_recall(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
    }
