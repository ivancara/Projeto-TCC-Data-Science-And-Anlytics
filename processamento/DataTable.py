from processamento.FileUtils import FileUtils
from processamento.Dummy import Dummy
class DataTable:
    def __init__(self):
        self.fileUtils=FileUtils()
        self.dataFrame = self.fileUtils.readFile()
        self.dummy = Dummy(self.dataFrame)
    
    def dummies(self):
        self.dummy.getDummy('genero')
        self.dummy.getDummy('faixa_etaria')
        self.dummy.getDummy('escolaridade')
        self.dummy.getDummy('estado_civil')
        self.dummy.getDummy('renda_familiar_mensal')
        self.dummy.getDummy('possui_depressao', applyMapping=self.Sim2Binary)
        self.dataFrame = self.dummy.data
        
        
    def rename(self):
        self.columns = {
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
                        ,'Cite até 3 emoções que você consiga lembrar':'emocoes_conhecidas'
                        ,'Descreva uma lembrança do seu passado que pode ter refletido em alguma decisão atual.':'descricao_lembranca_passado'
                        ,'Com base nas emoções listadas abaixo, qual delas se encaixa com a lembrança na época do ocorrido?':'emocoes_lembranca_passado'
                        ,'Com base nas emoções listadas abaixo, qual delas se encaixa com a lembrança atualmente?':'emocoes_lembranca_transformada'
                        ,'Descreva uma situação ocorrida atualmente que pode refletir em suas decisões futuras':'lembranca_atual_futuro'
                        ,'Com base nas emoções listadas abaixo, qual delas se encaixa com a situação no momento em que ocorreu?':'emocoes_lembranca_atual'
                        ,'Com base nas emoções listadas abaixo, qual delas você acredita que pode sentir no futuro quando se lembrar deste ocorrido?':'lembranca_atual_transformada_futuro'
                        }
        self.dummy.data = self.dataFrame.rename(columns=self.columns)
        
    def writeDataFrame(self):
        self.rename()
        self.dummies()
        self.fileUtils.writeFile(self.dataFrame)
    
    def Sim2Binary(self, text):
        if text == 'Sim':
            return 1
        elif text == 'Não':
            return 0
        else:
            return text