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
        self.dummy.getDummy('terapia', applyMapping=self.normalizeUtils.Sim2Binary)
        self.dummy.splitColumn('emocoes_conhecidas', ',| e | E ')
        self.dummy.getDummy('emocoes_conhecidas', applyMapping=self.normalizeUtils.normalizeString)
        self.dummy.splitColumn('emocoes_lembranca_passado', ';')
        self.dummy.getDummy('emocoes_lembranca_passado', applyMapping=self.normalizeUtils.normalizeString)
        self.dummy.splitColumn('emocoes_lembranca_transformada', ';')
        self.dummy.getDummy('emocoes_lembranca_transformada', applyMapping=self.normalizeUtils.normalizeString)
        self.dummy.splitColumn('emocoes_lembranca_atual', ';')
        self.dummy.getDummy('emocoes_lembranca_atual', applyMapping=self.normalizeUtils.normalizeString)
        self.dummy.splitColumn('emocoes_lembranca_atual_transformada_futuro', ';')
        self.dummy.getDummy('emocoes_lembranca_atual_transformada_futuro', applyMapping=self.normalizeUtils.normalizeString)

        self.dataFrame = self.dummy.data
        self.dummyFeeling.getDummy('tipo', applyMapping=self.normalizeUtils.dummyFeelingType)
        self.dummyFeeling.getDummy('emocao', applyMapping=self.normalizeUtils.normalizeString)
    def rename(self):
        columns = self.constantsManagement.REFACTOR_FIELDS_NAME
        self.dummy.data = self.dataFrame.rename(columns=columns)
    def mergeDataFrames(self):
        self.dataFrameFinal = pd.merge(self.dataFrame, self.dataFrameFeeling, how='right', left_on='emocoes_conhecidas', right_on='emocao', suffixes=('', '_conhecida'))
        self.dataFrameFinal = pd.merge(self.dataFrameFinal, self.dataFrameFeeling, how='right', left_on='emocoes_lembranca_passado', right_on='emocao', suffixes=('', '_lembranca_passado'))
        self.dataFrameFinal = pd.merge(self.dataFrameFinal, self.dataFrameFeeling, how='right', left_on='emocoes_lembranca_transformada', right_on='emocao', suffixes=('', '_lembranca_transformada'))
        self.dataFrameFinal = pd.merge(self.dataFrameFinal, self.dataFrameFeeling, how='right', left_on='emocoes_lembranca_atual', right_on='emocao', suffixes=('', '_lembranca_atual'))
        self.dataFrameFinal = pd.merge(self.dataFrameFinal, self.dataFrameFeeling, how='right', left_on='emocoes_lembranca_atual_transformada_futuro', right_on='emocao', suffixes=('', '_lembranca_atual_transformada_futuro'))
        self.dataFrameFinal = self.dataFrameFinal.drop_duplicates().reset_index(drop=True)
        self.dataFrameFinal = self.dataFrameFinal.fillna(0)
    def addTextAnalysisPredictedFields(self):
        self.dummyFinal.getDummy('lembranca_atual_futuro', 'lembranca_atual_futuro_predicted', applyMapping=self.normalizeUtils.predictFeelings)
        self.dummyFinal.getDummy('descricao_lembranca_passado', 'descricao_lembranca_passado_predicted', applyMapping=self.normalizeUtils.predictFeelings)
    def addDepressionAnalysisPredictedFields(self):
        self.dummyFinal.getDummy('possui_depressao_predicted', applyMapping=self.normalizeUtils.predictDepression(self.dataFrameFinal))     
    def writeDataTableIntoFile(self, addTexAnalysisPredictedFields=False, addDepressionAnalysisPredictedFields=False):
        self.rename()
        self.dummies()
        self.mergeDataFrames()
        self.dummyFinal = Dummy(self.dataFrameFinal)
        if addTexAnalysisPredictedFields:
            self.addTextAnalysisPredictedFields()
        if addDepressionAnalysisPredictedFields:
            self.addDepressionAnalysisPredictedFields()
        self.fileUtils.writeFile(self.dataFrame)
        self.fileUtilsFeelings.writeFile(self.dataFrameFeeling)
        self.fileUtilsFinal.writeFile(self.dataFrameFinal)
   