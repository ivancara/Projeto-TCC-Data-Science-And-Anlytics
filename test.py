#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, make_scorer
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
import numpy as np
import io

#remover wargnings
import warnings
warnings.filterwarnings("ignore")
# Definir a paleta de cores do estilo Material Design
sns.set_palette("muted")

# Carregar o arquivo CSV
file_path = 'data/files/out_out_dados.csv'
df = pd.read_csv(file_path, sep=';')

# Substituir 'True' e 'False' por 1 e 0
df = df.replace({'True': 1, 'False': 0})

# Identificar colunas que contêm datas
date_columns = ['emocao','data_resposta','aceitou','estado','emocoes_conhecidas','descricao_lembranca_passado','emocoes_lembranca_passado','emocoes_lembranca_transformada','lembranca_atual_futuro','emocoes_lembranca_atual','emocoes_lembranca_atual_transformada_futuro','emocao_lembranca_passado','emocao_lembranca_transformada','emocao_lembranca_atual','emocao_lembranca_atual_transformada_futuro']

# Remover colunas de datas
df = df.drop(columns=date_columns)

# Calcular a correlação de Pearson
correlation_matrix = df.corr(method='pearson')

# Selecionar colunas com correlação mínima de 0.7 com a variável alvo
target_correlation = correlation_matrix.iloc[:, -1].abs()
selected_columns = target_correlation[target_correlation >= 0.7].index

# Filtrar o DataFrame para incluir apenas as colunas selecionadas
df_filtered = df[selected_columns]

correlation_matrix = df_filtered.corr(method='pearson')
# Plotar o heatmap da correlação de Pearson
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap da Correlação de Pearson')
plt.savefig('correlation_heatmap.png')
plt.show()

# Separar características e alvo
X = df_filtered.iloc[:, :-1]
y = df_filtered.iloc[:, -1]

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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

# Avaliar os modelos usando GridSearchCV e plotar gráficos de treino e teste
results = {}
for model_name, model_info in models.items():
    clf = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    overfitting = accuracy_train - accuracy_test
    cross_val_scores = cross_val_score(clf.best_estimator_, X_train, y_train, cv=5, scoring='accuracy')
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
        'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
        'save_path': f'{model_name}_learning_curve.png',
        'best_estimator': best_estimator_score
    }
    print(f"{model_name} Best Params: {clf.best_params_}")
    print(f"{model_name} Train Accuracy: {accuracy_train:.4f}")
    print(f"{model_name} Test Accuracy: {accuracy_test:.4f}")
    print(f"{model_name} MSE: {mse:.4f}")
    print(f"{model_name} R2: {r2:.4f}")
    print(f"{model_name} Overfitting: {overfitting:.4f}")
    print(f"{model_name} CV Mean Acc: {cross_val_mean:.4f}")
    print(f"{model_name} Best Estimator Score: {best_estimator_score:.4f}")
    print(classification_report(y_test, y_pred_test))

    # Plotar curva de aprendizado
    train_sizes, train_scores, test_scores = learning_curve(clf.best_estimator_, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, train_scores_mean, 'o-', color=sns.color_palette('muted')[0], label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color=sns.color_palette('muted')[1], label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title(f'{model_name} Learning Curve\nAccuracy: {accuracy_test:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}, Best Estimator: {best_estimator_score:.4f}')
    plt.legend(loc='best')
    plt.savefig(f'{model_name}_learning_curve.png')
    plt.show()

# Selecionar o melhor modelo
best_model_name = max(results, key=lambda x: results[x]['best_estimator'])
best_model_info = results[best_model_name]
print(f"Best Model: {best_model_name} with Test Accuracy: {best_model_info['accuracy_test']:.4f}, R²: {best_model_info['r2']:.4f}, MSE: {best_model_info['mse']:.4f}, Overfitting: {best_model_info['overfitting']:.4f}, CV Mean Acc: {best_model_info['cv_mean_acc']:.4f}, Best Estimator Score: {best_model_info['best_estimator']:.4f} and Params: {best_model_info['best_params']}")

# Criar o PDF com os resultados
pdf_file = 'model_results.pdf'
doc = SimpleDocTemplate(pdf_file, pagesize=letter)
elements = []

# Adicionar título
elements.append(Paragraph("Model Results", ParagraphStyle('Title', fontSize=16, spaceAfter=20)))

# Adicionar heatmap
elements.append(Image('correlation_heatmap.png', width=550, height=250))
elements.append(PageBreak())

# Adicionar gráficos de curvas de aprendizado
for model_name, model_info in results.items():
    learning_curve_image_path = f'{model_name}_learning_curve.png'
    elements.append(Paragraph(f"{model_name} Learning Curve", ParagraphStyle('Title', fontSize=14, spaceAfter=10)))
    elements.append(Image(learning_curve_image_path, width=550, height=250))
    elements.append(PageBreak())

# Adicionar detalhes do melhor modelo ao PDF
elements.append(Paragraph(f"Best Model: {best_model_name}", ParagraphStyle('Title', fontSize=14, spaceAfter=10)))
elements.append(Paragraph(f"Test Accuracy: {best_model_info['accuracy_test']:.4f}", ParagraphStyle('Normal', fontSize=12, spaceAfter=10)))
elements.append(Paragraph(f"R²: {best_model_info['r2']:.4f}", ParagraphStyle('Normal', fontSize=12, spaceAfter=10)))
elements.append(Paragraph(f"MSE: {best_model_info['mse']:.4f}", ParagraphStyle('Normal', fontSize=12, spaceAfter=10)))
elements.append(Paragraph(f"Overfitting: {best_model_info['overfitting']:.4f}", ParagraphStyle('Normal', fontSize=12, spaceAfter=10)))
elements.append(Paragraph(f"CV Mean Acc: {best_model_info['cv_mean_acc']:.4f}", ParagraphStyle('Normal', fontSize=12, spaceAfter=10)))
elements.append(Paragraph(f"Best Estimator Score: {best_model_info['best_estimator']:.4f}", ParagraphStyle('Normal', fontSize=12, spaceAfter=10)))
elements.append(Paragraph(f"Best Params: {best_model_info['best_params']}", ParagraphStyle('Normal', fontSize=12, spaceAfter=10)))
elements.append(PageBreak())

# Adicionar textos dos prints em uma tabela
data = [["Model", "Best Params", "Train Accuracy", "Test Accuracy", "Classification Report"]]
for model_name, model_info in results.items():
    data.append([
        model_name,
        str(model_info['best_params']),
        f"{model_info['accuracy_train']:.4f}",
        f"{model_info['accuracy_test']:.4f}",
        Paragraph(str(model_info['classification_report']), ParagraphStyle('Normal', fontSize=10))
    ])

# Adicionar tabela com MSE, R², acurácia e best_estimator
mse_r2_data = [["Model", "MSE", "R²", "Accuracy", "Overfitting", "Cross-Validator\n Mean Acc", "Best Estimator"]]
for model_name, model_info in results.items():
    mse_r2_data.append([
        model_name,
        f"{model_info['mse']:.4f}",
        f"{model_info['r2']:.4f}",
        f"{model_info['accuracy_test']:.4f}",
        f"{model_info['overfitting']:.4f}",
        f"{model_info['cv_mean_acc']:.4f}",
        f"{model_info['best_estimator']:.4f}"
    ])

# Salvar a tabela como imagem
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('tight')
ax.axis('off')
#aumente a altura das colunas de ax.table
table = ax.table(cellText=mse_r2_data,  colLabels=None, cellLoc='center', loc='center',)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
table[0,0].set_height(0.1)
table[0,1].set_height(0.1)
table[0,2].set_height(0.1)
table[0,3].set_height(0.1)
table[0,4].set_height(0.1)
table[0,5].set_height(0.1)
table[0,6].set_height(0.1)
plt.savefig('mse_r2_table.png')
plt.show()

# Adicionar imagem da tabela
elements.append(Image('mse_r2_table.png', width=550, height=250))
elements.append(PageBreak())

# Salvar o PDF
doc.build(elements)
print(f"PDF saved as {pdf_file}")