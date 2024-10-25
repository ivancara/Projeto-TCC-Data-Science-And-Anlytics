import statsmodels.formula.api as smf
import statsmodels.api as sm 
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,recall_score
class TrainingDepression:
    def __init__(self, dataSplitUtils, constantsManagement, fileUtils, target=None, features=[]) -> None:
        data = fileUtils.readFile(';')
        self.fileUtils = fileUtils
        self.constantsManagement = constantsManagement
        self.df_train, self.df_test = dataSplitUtils.split_data(data, size=constantsManagement.TRAIN_PERCENTAGE)
        self.df_val, self.df_test = dataSplitUtils.split_data(self.df_test, size=constantsManagement.TEST_PERCENTAGE)
        self.model = None
        self.target = target
        self.features = features

    def train(self):
        self.model = smf.glm(formula=f'{self.target} ~ {' + '.join(self.features)}'	
                                 , data=self.df_train
                                 , family=sm.families.Binomial())
        self.model = self.model.fit()
        self.df_train['possui_depressao_predicted'] = np.round(self.model.predict(),0).astype(int)
        self.fileUtils.writeFile(self.df_train)
        self.model.save(self.constantsManagement.MODEL_DEPRESSION_ANALYSIS_PATH)

    def sumary(self):
        self.fileUtils.loadModelStatsModel(self.constantsManagement.MODEL_DEPRESSION_ANALYSIS_PATH)
        return self.model.summary()
    
    def eval(self, cutoff):
        values = self.df_train['possui_depressao'].values
        observado =self.df_train['possui_depressao_predicted'].values
        predicao_binaria = []
        for item in values:
            if item < cutoff:
                predicao_binaria.append(0)
            else:
                predicao_binaria.append(1)
            
            
        sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
        especificidade = recall_score(observado, predicao_binaria, pos_label=0)
        acuracia = accuracy_score(observado, predicao_binaria)

        # Visualização dos principais indicadores desta matriz de confusão
        indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                    'Especificidade':[especificidade],
                                    'Acurácia':[acuracia]})
        return indicadores
