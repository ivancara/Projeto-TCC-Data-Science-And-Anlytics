import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from processing.Statistics.Metrics import Metrics
import pandas as pd
import numpy as np
class TrainingDepression:
    def __init__(self, dataSplitUtils, constantsManagement, fileUtils) -> None:
        data = fileUtils.readFile(';')
        self.fileUtils = fileUtils
        self.constantsManagement = constantsManagement
        
        self.X_train, self.X_test, self.y_train, self.y_test = dataSplitUtils.split_data(data)
        self.model = None

        self.target = dataSplitUtils.target
        self.features = dataSplitUtils.features
        self.metrics = Metrics()
    
    def scalar(self):
        # Escalar os dados
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
    
    def train(self):
        self.fileUtils.deleteFile(self.constantsManagement.MODEL_DEPRESSION_ANALYSIS_PATH)
        self.scalar()
        models = self.getModels()
        # Avaliar os modelos usando GridSearchCV e plotar gráficos de treino e teste
        results = {}
        max_best_score = 0
        for model_name, model_info in models.items():
            clf = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy')
            clf.fit(self.X_train, self.y_train)
            y_pred_train = clf.predict(self.X_train)
            y_pred_test = clf.predict(self.X_test)
            accuracy_train = self.metrics.acuracy_score(self.y_train, y_pred_train)
            accuracy_test = self.metrics.acuracy_score(self.y_test, y_pred_test)
            mse = self.metrics.mean_squared_error(self.y_test, y_pred_test)
            r2 = self.metrics.r2_score(self.y_test, y_pred_test)
            overfitting = accuracy_train - accuracy_test
            cross_val_scores = self.metrics.cross_validation_score(clf.best_estimator_, self.X_train, self.y_train)
            cross_val_mean = self.metrics.mean(cross_val_scores)
            best_estimator_score = accuracy_test + r2 - mse - overfitting + cross_val_mean
            results[model_name] = {
                'model_name': model_name,
                'best_params': clf.best_params_,
                'accuracy_train': accuracy_train,
                'accuracy_test': accuracy_test,
                'mse': mse,
                'r2': r2,
                'overfitting': overfitting,
                'cv_mean_acc': cross_val_mean,
                'classification_report': self.metrics.classification_report(self.y_test, y_pred_test),
                'best_estimator': best_estimator_score
            }
            if(best_estimator_score > max_best_score):
                max_best_score = best_estimator_score
                self.model = clf
        #parse results into a dataframe
        results = pd.DataFrame(results).T
        results = results.sort_values(by='best_estimator', ascending=False)
        self.fileUtils.writeDataframeFile(results, self.constantsManagement.RESULTS_DEPRESSION_ANALYSIS_PATH)
        self.fileUtils.saveModel(self.model, self.constantsManagement.MODEL_DEPRESSION_ANALYSIS_PATH)
        return results

    def getModels(self):
        # Definir os modelos e os hiperparâmetros para GridSearchCV
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [100, 200, 300]
                }
            },
            'Ridge Classifier': {
                'model': RidgeClassifier(),
                'params': {
                    'alpha': [0.01, 0.1, 1, 10, 100]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'Extra Trees': {
                'model': ExtraTreesClassifier(),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'SVM': {
                'model': SVC(),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(),
                'params': {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {}
            }
        }

        return models
    
    def sumary(self):
        self.fileUtils.loadModel(self.constantsManagement.MODEL_DEPRESSION_ANALYSIS_PATH)
        return self.model.summary()
    