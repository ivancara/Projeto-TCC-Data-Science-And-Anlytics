from processamento.FileUtils import FileUtils
from processamento.Dummy import Dummy
from processamento.NormalizeUtils import NormalizeUtils
import pandas as pd

from utils.ConstantsManagement import ConstantsManagement

class DataTable:
    def __init__(self):
        self.normalizeUtils = NormalizeUtils()
        self.fileUtils=FileUtils()
        self.constantsManagement = ConstantsManagement()
        self.fileUtilsFinal=FileUtils(self.constantsManagement.WRANGLED_DATA_FINAL)
        self.fileUtilsFeelings=FileUtils(self.constantsManagement.EMOTIONS_FILE)
        self.dataFrame = self.fileUtils.readFile()
        self.dataFrameFeeling = self.fileUtilsFeelings.readFile(';')    
        self.dummy = Dummy(self.dataFrame)
        self.dummyFeeling = Dummy(self.dataFrameFeeling)
        self.dataFrameFinal = pd.DataFrame()
        
    
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
        columns = {
                        'Carimbo de data/hora':'data_resposta'
                        ,'Li o Termo acima e concordo':'aceitou'
                        ,'Qual  o seu gênero?':'genero'
                        ,'Qual é a sua faixa etária?':'faixa_etaria'
                        ,'Qual é o seu nível de escolaridade?':'escolaridade'
                        ,'Qual é o seu estado civil?':'estado_civil'
                        ,'Qual é a sua renda familiar mensal?':'renda_familiar_mensal'
                        ,'Qual estado que você reside?':'estado'
                        ,'Você ja foi diagnosticada(o) com depressão?':'possui_depressao'
                        ,'Você acredita que lembranças passadas podem refletir suas decisões atualmente?':'lembrancas_afetam_decisoes'
                        ,'Você acredita que lembranças da sua infância refletem suas decisões atualmente?':'lembrancas_infancia_afetam_decisoes'
                        ,'Você consegue identificar suas emoções?':'identifica_emocoes'
                        ,'Você acredita que a prática de atividades físicas podem ajudar no tratamento da saúde mental?':'atividade_fisica'
                        ,'Você faz terapia com um profissional regularmente?':'terapia'
                        ,'Cite até 3 emoções que você consiga lembrar (Separadas por vírgula \',\').':'emocoes_conhecidas'
                        ,'Descreva uma lembrança do seu passado que pode ter refletido em alguma decisão atual.':'descricao_lembranca_passado'
                        ,'Com base nas emoções listadas abaixo, qual delas se encaixa com a lembrança na época do ocorrido?':'emocoes_lembranca_passado'
                        ,'Com base nas emoções listadas abaixo, qual delas se encaixa com a lembrança atualmente?':'emocoes_lembranca_transformada'
                        ,'Descreva uma situação ocorrida atualmente que pode refletir em suas decisões futuras':'lembranca_atual_futuro'
                        ,'Com base nas emoções listadas abaixo, qual delas se encaixa com a situação no momento em que ocorreu?':'emocoes_lembranca_atual'
                        ,'Com base nas emoções listadas abaixo, qual delas você acredita que pode sentir no futuro quando se lembrar deste ocorrido?':'emocoes_lembranca_atual_transformada_futuro'
                        }
        self.dummy.data = self.dataFrame.rename(columns=columns)

    def mergeDataFrames(self):
        self.dataFrameFinal = pd.merge(self.dataFrame, self.dataFrameFeeling, how='right', left_on='emocoes_conhecidas', right_on='emocao', suffixes=('', '_conhecida'))
        self.dataFrameFinal = pd.merge(self.dataFrameFinal, self.dataFrameFeeling, how='right', left_on='emocoes_lembranca_passado', right_on='emocao', suffixes=('', '_lembranca_passado'))
        self.dataFrameFinal = pd.merge(self.dataFrameFinal, self.dataFrameFeeling, how='right', left_on='emocoes_lembranca_transformada', right_on='emocao', suffixes=('', '_lembranca_transformada'))
        self.dataFrameFinal = pd.merge(self.dataFrameFinal, self.dataFrameFeeling, how='right', left_on='emocoes_lembranca_atual', right_on='emocao', suffixes=('', '_lembranca_atual'))
        self.dataFrameFinal = pd.merge(self.dataFrameFinal, self.dataFrameFeeling, how='right', left_on='emocoes_lembranca_atual_transformada_futuro', right_on='emocao', suffixes=('', '_lembranca_atual_transformada_futuro'))
        self.dataFrameFinal = self.dataFrameFinal.drop(columns=['emocao', 'emocao_lembranca_passado', 'emocao_lembranca_transformada', 'emocao_lembranca_atual', 'emocao_lembranca_atual_transformada_futuro'])
        self.dataFrameFinal = self.dataFrameFinal.drop_duplicates().reset_index(drop=True)
        self.dataFrameFinal = self.dataFrameFinal.fillna(0)
        
    def writeDataTableIntoFile(self):
        self.rename()
        self.dummies()
        self.mergeDataFrames()
        self.fileUtils.writeFile(self.dataFrame)
        self.fileUtilsFeelings.writeFile(self.dataFrameFeeling)
        self.fileUtilsFinal.writeFile(self.dataFrameFinal)
   