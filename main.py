import pandas as pd
from processing.TextAnalysis.TrainingFeelingAnalysis import TrainingFeelingAnalysis
from processing.DepressionAnalysis.TrainingDepression import TrainingDepression
from processing.Data.DataTable import DataTable
from utils.DataSplitUtils import DataSplitUtils
from utils.DeviceUtils import DeviceUtils
from utils.NormalizeUtils import NormalizeUtils
from utils.FileUtils import FileUtils
from utils.ConstantsManagement import ConstantsManagement
from model_statistics.SummaryDepressionAnalysis import SummaryDepressionAnalysis
from model_statistics.SummaryTextAnalysis import SummaryTextAnalysis
import warnings
warnings.filterwarnings('ignore')

class Main:
    def __init__(self) -> None:
        self.normalizeUtils, self.fileUtils, self.constantManagement, self.dataSplitUtils, self.deviceUtils = self.dependencies()
        pass
    def dependencies(self):
        deviceUtils = DeviceUtils()
        constantManagement = ConstantsManagement()
        fileUtils = FileUtils(deviceUtils=deviceUtils, constantsManagement=constantManagement)
        normalizeUtils = NormalizeUtils(fileUtils=fileUtils, constantsManagement=constantManagement, deviceUtils=deviceUtils)
        dataSplitUtils = DataSplitUtils(constantsManagement=constantManagement)
        return normalizeUtils, fileUtils, constantManagement, dataSplitUtils, deviceUtils
    def main(self): 
        
        while True:
            print("-"*34)
            print("1 - Training Feeling Analysis")
            print("2 - Predict Feeling Analysis")
            print("3 - Training Depression Analysis")
            print("4 - Predict Depression Analysis")
            print("5 - Statistics Feeling Analysis")
            print("6 - Statistics Depression Analysis")
            print("7 - Process Data")
            print("8 - Exit")
            print("-"*34)
            option = int(input("Choose an option: "))
            try:
                match option:
                    case 1:
                        text='lembranca_atual_futuro'
                        target='tipo_lembranca_atual'
                        fileUtilsWrangledDataFinal = FileUtils(fileName=self.constantManagement.WRANGLED_DATA_FINAL, deviceUtils=self.deviceUtils, constantsManagement=self.constantManagement)
                        fileUtilsTextStatisticsHistory = FileUtils(fileName=self.constantManagement.TEXT_STATISTICS_HISTORY, deviceUtils=self.deviceUtils, constantsManagement=self.constantManagement)
                        training = TrainingFeelingAnalysis(text=text, targets=target, deviceUtils=self.deviceUtils, constantsManagement=self.constantManagement, dataSplitUtils=self.dataSplitUtils, fileUtilsWrangledData=fileUtilsWrangledDataFinal, fileUtilsTextStatisticHistory=fileUtilsTextStatisticsHistory)
                        training.train()
                        pass
                    case 2:
                        predict = NormalizeUtils(fileUtils=self.fileUtils, constantsManagement=self.constantManagement)
                        description = input("Enter the some text: ")
                        print(predict.predictFeelingAnalysis(description))
                        pass
                    case 3:
                        target = 'possui_depressao'
                        features = ['lembranca_atual_futuro_predicted',
                                    'descricao_lembranca_passado_predicted',
                                    'terapia',
                                    'identifica_emocoes',
                                    'genero_Feminino',
                                    'genero_Masculino']
                        file = FileUtils(fileName=self.constantManagement.WRANGLED_DATA_FINAL, deviceUtils=self.deviceUtils, constantsManagement=self.constantManagement)
                        training = TrainingDepression(target=target, features=features, dataSplitUtils=self.dataSplitUtils, fileUtils=file, constantsManagement=self.constantManagement)
                        training.train()
                        pass
                    case 4:
                        description = input("Describe some history that enhance your decisions in future: ")
                        lembranca_atual_futuro_predicted = self.normalizeUtils.predictFeelings(description)
                        description = input("Remember some history that enhance your decisions today: ")
                        descricao_lembranca_passado_predicted = self.normalizeUtils.predictFeelings(description)
                        terapia = int(input("Do you have therapy? (1-yes or 0-no): "))
                        identifica_emocoes = int(input("Do you identify emotions? (1-yes or 0-no): "))
                        genero = int(input("Enter your Genre (Male: 0 or Female: 1): "))
                        data = pd.DataFrame({'lembranca_atual_futuro_predicted': [lembranca_atual_futuro_predicted],
                                            'descricao_lembranca_passado_predicted': [descricao_lembranca_passado_predicted],
                                            'terapia': [terapia],
                                            'identifica_emocoes': [identifica_emocoes],
                                            'genero_Feminino': [genero == 1],
                                            'genero_Masculino': [genero == 0]})
                        print(data)
                        self.normalizeUtils.predictDepression(data)
                        pass
                    case 5:
                        file = FileUtils(fileName='out_'+self.constantManagement.TEXT_STATISTICS_HISTORY, deviceUtils=self.deviceUtils, constantsManagement=self.constantManagement)
                        statistics = SummaryTextAnalysis(fileUtils=file)
                        statistics.get_summary()
                        pass
                    case 6:
                        statistics = SummaryDepressionAnalysis(fileUtils=self.fileUtils, constantsManagement=self.constantManagement)
                        statistics.get_summary()
                        pass
                    case 7:
                        fileUtilsFinal=FileUtils(fileName=self.constantManagement.DATA_FINAL, deviceUtils=self.deviceUtils, constantsManagement=self.constantManagement)
                        fileUtilsFeelings=FileUtils(fileName=self.constantManagement.EMOTIONS_FILE, deviceUtils=self.deviceUtils, constantsManagement=self.constantManagement)
                        data = DataTable(fileUtils=self.fileUtils, constantManagement=self.constantManagement, normalizeUtils=self.normalizeUtils, fileUtilsFinal=fileUtilsFinal, fileUtilsFeelings=fileUtilsFeelings)
                        addTextFeelingsAnalysis = self.fileUtils.hasFile(self.constantManagement.MODEL_FEELINGS_ANALYSIS_PATH)
                        addDepressionAnalysis = self.fileUtils.hasFile(self.constantManagement.MODEL_DEPRESSION_ANALYSIS_PATH) 
                        data.writeDataTableIntoFile(addTexAnalysisPredictedFields=addTextFeelingsAnalysis, addDepressionAnalysisPredictedFields=addDepressionAnalysis)
                        pass
                    case 8:
                        exit()
                    case _:
                        print("Invalid option")
                        continue
            except Exception as e:
                print(e)
                continue
if __name__ == "__main__":
    main = Main()
    main.main()