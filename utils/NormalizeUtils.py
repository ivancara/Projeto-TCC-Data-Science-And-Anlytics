import unicodedata
from processing.TextAnalysis.PredictFeelingAnalysis import PredictFeelingAnalysis
from processing.DepressionAnalysis.PredictDepression import PredictDepression
class NormalizeUtils():
    def __init__(self, constantsManagement, fileUtils, deviceUtils) -> None:
        self.predictFeelingAnalysis = PredictFeelingAnalysis(constantManagement=constantsManagement, fileUtils=fileUtils, deviceUtils=deviceUtils)
        self.PredictDepression = PredictDepression(constantsManagement=constantsManagement, fileUtils=fileUtils)
        pass
     
    def dummyFeelingType(self, text):
        if text == "positivo":
            return 2
        elif text == "negativo":
            return 0
        elif text == "neutro":
            return 1
        else: 
            return text
    
    def Sim2Binary(self, text):
        if text == 'Sim':
            return 1
        elif text == 'Não':
            return 0
        else:
            return text
        
    def normalizeString(self, text):
         # Unicode normalize transforma um caracter em seu equivalente em latin.
        nfkd = unicodedata.normalize('NFKD', text)
        palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])

        # Usa expressão regular para retornar a palavra apenas com números, letras e espaço
        return palavraSemAcento.upper().replace(' ','').replace('.','')
    
    def predictFeelings(self, text):
        predicted = self.predictFeelingAnalysis.predict(str(text))
        return predicted[0]
    
    def predictDepression(self, data):
        return self.PredictDepression.predict(data)