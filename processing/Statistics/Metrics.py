from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import cross_val_score, learning_curve
import numpy as np
import warnings
warnings.filterwarnings('ignore')
class Metrics:
    def __init__(self):
        pass
    
    def acuracy_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
    def mean_squared_error(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    def r2_score(self, y_true, y_pred):
        return r2_score(y_true, y_pred)
    
    def classification_report(self, y_true, y_pred):
        return classification_report(y_true, y_pred, output_dict=True)
    
    def cross_validation_score(self, model, X, y):
        return cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    def mean(self, scores, axis=None):
        return np.mean(scores, axis=axis)
    
    def learning_curve(self, model, X, y):
        return learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    
    def confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)