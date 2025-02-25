import pandas as pd
from processing.DepressionAnalysis.TrainingDepression import TrainingDepression
from processing.Data.DataTable import DataTable
from utils.DataSplitUtils import DataSplitUtils
from utils.DeviceUtils import DeviceUtils
from utils.NormalizeUtils import NormalizeUtils
from utils.FileUtils import FileUtils
from utils.ConstantsManagement import ConstantsManagement
from model_statistics.SummaryDepressionAnalysis import SummaryDepressionAnalysis
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
            print("1 - Process Data")
            print("2 - Training Depression Analysis")
            print("3 - Predict Depression Analysis")
 
            print("4 - Exit")
            print("-"*34)
            option = int(input("Choose an option: "))
            try:
                match option:
                    case 1:
                        fileUtilsFinal=FileUtils(fileName=self.constantManagement.DATA_FINAL, deviceUtils=self.deviceUtils, constantsManagement=self.constantManagement)
                        fileUtilsFeelings=FileUtils(fileName=self.constantManagement.EMOTIONS_FILE, deviceUtils=self.deviceUtils, constantsManagement=self.constantManagement)
                        data = DataTable(fileUtils=self.fileUtils, constantsManagement=self.constantManagement, normalizeUtils=self.normalizeUtils, fileUtilsFinal=fileUtilsFinal, fileUtilsFeelings=fileUtilsFeelings)
                        addDepressionAnalysis = False

                        try: 
                            addDepressionAnalysis = self.fileUtils.hasFile(self.constantManagement.MODEL_DEPRESSION_ANALYSIS_PATH) 
                        except Exception as e:
                            print('Depression analysis model not found')
                        data.writeDataTableIntoFile(addDepressionAnalysisPredictedFields=addDepressionAnalysis)
                        pass
                    case 2:
                        file = FileUtils(fileName=self.constantManagement.WRANGLED_DATA_FINAL, deviceUtils=self.deviceUtils, constantsManagement=self.constantManagement)
                        training = TrainingDepression(dataSplitUtils=self.dataSplitUtils, fileUtils=file, constantsManagement=self.constantManagement)
                        training.train()
                        pass
                    case 3:

                        pass
                    case 4:
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