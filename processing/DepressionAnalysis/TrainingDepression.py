import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, learning_curve, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import numpy as np
class TrainingDepression:
    def __init__(self, dataSplitUtils, constantsManagement, fileUtils) -> None:
        data = fileUtils.readFile(';')
        self.fileUtils = fileUtils
        self.constantsManagement = constantsManagement
        
        self.X_train, self.X_test, self.y_train, self.y_test = dataSplitUtils.split_data(data)
        self.model = None

        self.target = data.iloc[:, -1]
        self.features = data.iloc[:, :-1]
    
    def scalar(self):
        # Escalar os dados
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    def train(self):
        self.scalar()
        models = self.getModels()
        # Avaliar os modelos usando GridSearchCV e plotar gráficos de treino e teste
        results = {}
        for model_name, model_info in models.items():
            clf = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy')
            clf.fit(self.X_train, self.y_train)
            y_pred_train = clf.predict(self.X_train)
            y_pred_test = clf.predict(self.X_test)
            accuracy_train = accuracy_score(self.y_train, y_pred_train)
            accuracy_test = accuracy_score(self.y_test, y_pred_test)
            mse = mean_squared_error(self.y_test, y_pred_test)
            r2 = r2_score(self.y_test, y_pred_test)
            overfitting = accuracy_train - accuracy_test
            cross_val_scores = cross_val_score(clf.best_estimator_, self.X_train, self.y_train, cv=5, scoring='accuracy')
            cross_val_mean = np.mean(cross_val_scores)
            best_estimator_score = accuracy_test + r2 - mse - overfitting + cross_val_mean
            results[model_name] = {
                'best_params': clf.best_params_,
                'accuracy_train': accuracy_train,
                'accuracy_test': accuracy_test,
                'mse': mse,
                'r2': r2,
                'overfitting': overfitting,
                'cv_mean_acc': cross_val_mean,
                'classification_report': classification_report(self.y_test, y_pred_test, output_dict=True),
                'save_path': f'{model_name}_learning_curve.png',
                'best_estimator': best_estimator_score,
                'model': y_pred_train
            }
        self.model = self.selectBestModel(results)
        self.fileUtils.writeFile(self.model)
        self.model.save(self.constantsManagement.MODEL_DEPRESSION_ANALYSIS_PATH)
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
    
    def selectBestModel(self, results):
        best_model_name = max(results, key=lambda x: results[x]['best_estimator'])
        best_model_info = results[best_model_name]
        return best_model_info['model']
    
    def sumary(self):
        self.fileUtils.loadModelStatsModel(self.constantsManagement.MODEL_DEPRESSION_ANALYSIS_PATH)
        return self.model.summary()
    