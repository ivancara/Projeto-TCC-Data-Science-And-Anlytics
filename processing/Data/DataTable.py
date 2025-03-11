from processing.Data.Dummy import Dummy
import pandas as pd
class DataTable:
    def __init__(self, fileUtils,constantsManagement, normalizeUtils, fileUtilsFinal, fileUtilsFeelings) -> None:
        self.normalizeUtils = normalizeUtils
        self.fileUtils=fileUtils
        self.constantsManagement = constantsManagement
        self.fileUtilsFinal=fileUtilsFinal
        self.fileUtilsFeelings=fileUtilsFeelings
        self.dataFrame = self.fileUtils.readFile()
        self.dataFrameFeeling = self.fileUtilsFeelings.readFile(';')    
        self.dummy = Dummy(self.dataFrame)
        self.dummyFeeling = Dummy(self.dataFrameFeeling)
        self.dataFrameFinal = pd.DataFrame()
        self.dummyFinal = Dummy(None)      
          
    def dummies(self):
        self.dummy.getDummy('genero')
        self.dummy.getDummy('faixa_etaria')
        self.dummy.getDummy('escolaridade')
        self.dummy.getDummy('estado_civil')
        self.dummy.getDummy('renda_familiar_mensal')
        self.dummy.getDummy('possui_depressao', applyMapping=self.normalizeUtils.Sim2Binary)
        self.dummy.getDummy('faz_terapia_regularmente', applyMapping=self.normalizeUtils.Sim2Binary)
        self.dummy.splitColumn('emocoes_conhecidas', ',| e | E ')
        self.dummy.getDummy('emocoes_conhecidas', applyMapping=self.normalizeUtils.normalizeString)
        self.dataFrame = self.dummy.data
        self.dummyFeeling.getDummy('tipo', applyMapping=self.normalizeUtils.dummyFeelingType)
        self.dummyFeeling.getDummy('emocao', applyMapping=self.normalizeUtils.normalizeString)
        
    def rename(self):
        columns = self.constantsManagement.REFACTOR_FIELDS_NAME
        self.dummy.data = self.dataFrame.rename(columns=columns)
        self.dummy.data = self.dummy.data[columns.values()]
        
    def mergeDataFrames(self):
        self.dataFrameFinal = pd.merge(self.dataFrame, self.dataFrameFeeling, how='right', left_on='emocoes_conhecidas', right_on='emocao', suffixes=('', '_conhecida'))
        self.dataFrameFinal = self.dataFrameFinal.drop_duplicates().reset_index(drop=True)
        self.dataFrameFinal = self.dataFrameFinal.fillna(0)
        self.dataFrameFinal = self.dataFrameFinal.drop(columns=['emocoes_conhecidas', 'emocao'])
        self.dataFrameFinal = self.dataFrameFinal.replace({True: 1, False: 0})
    def addDepressionAnalysisPredictedFields(self):
        self.dummyFinal.getDummy('possui_depressao_predicted', applyMapping=self.normalizeUtils.predictDepression(self.dataFrameFinal))     
    
    def writeDataTableIntoFile(self):
        self.rename()
        self.dummies()
        self.mergeDataFrames()
        self.dummyFinal = Dummy(self.dataFrameFinal)
        addDepressionAnalysisPredictedFields = self.fileUtils.hasFile(self.constantManagement.MODEL_DEPRESSION_ANALYSIS_PATH)
        if addDepressionAnalysisPredictedFields:
            self.addDepressionAnalysisPredictedFields()
        self.fileUtils.writeFile(self.dataFrame)
        self.fileUtilsFeelings.writeFile(self.dataFrameFeeling)
        self.fileUtilsFinal.writeFile(self.dataFrameFinal)
   