import pandas as pd
from processing.DepressionAnalysis.TrainingDepression import TrainingDepression
from processing.Data.DataTable import DataTable
from processing.Statistics.Chart import Chart
from utils.DataSplitUtils import DataSplitUtils
from utils.NormalizeUtils import NormalizeUtils
from utils.FileUtils import FileUtils
from utils.ConstantsManagement import ConstantsManagement
from model_statistics.SummaryDepressionAnalysis import SummaryDepressionAnalysis

import warnings
warnings.filterwarnings('ignore')
class Main:
    def __init__(self) -> None:
        self.normalizeUtils, self.fileUtils, self.constantManagement, self.dataSplitUtils = self.dependencies()
        pass
    def dependencies(self):
        constantManagement = ConstantsManagement()
        fileUtils = FileUtils(constantsManagement=constantManagement)
        normalizeUtils = NormalizeUtils(fileUtils=fileUtils, constantsManagement=constantManagement)
        dataSplitUtils = DataSplitUtils(constantsManagement=constantManagement)
        return normalizeUtils, fileUtils, constantManagement, dataSplitUtils
    def main(self):       
        while True:
            print("-"*34)
            print("1 - Process Data")
            print("2 - Training Depression Analysis")
            print("3 - Model Statistics")
            print("4 - Exit")
            print("-"*34)
            option = int(input("Choose an option: "))
            try:
                match option:
                    case 1:
                        fileUtilsFinal=FileUtils(fileName=self.constantManagement.DATA_FINAL, constantsManagement=self.constantManagement)
                        fileUtilsFeelings=FileUtils(fileName=self.constantManagement.EMOTIONS_FILE, constantsManagement=self.constantManagement)
                        data = DataTable(fileUtils=self.fileUtils, constantsManagement=self.constantManagement, normalizeUtils=self.normalizeUtils, fileUtilsFinal=fileUtilsFinal, fileUtilsFeelings=fileUtilsFeelings)
                        data.writeDataTableIntoFile()
                        pass
                    case 2:
                        file = FileUtils(fileName=self.constantManagement.WRANGLED_DATA_FINAL, constantsManagement=self.constantManagement)
                        training = TrainingDepression(dataSplitUtils=self.dataSplitUtils, fileUtils=file, constantsManagement=self.constantManagement)
                        training.train()
                        pass
                    case 3:
                        chart = Chart(constantsManagement=self.constantManagement)
                        file = FileUtils(fileName=self.constantManagement.WRANGLED_DATA_FINAL, constantsManagement=self.constantManagement)
                        summary = SummaryDepressionAnalysis(fileUtils=file, constantsManagement=self.constantManagement, chart=chart)
                        summary.summary()
                        pass
                    case 4:
                        exit()
                        break
                    case _:
                        print("Invalid option")
                        continue
            except Exception as e:
                print(e)
                continue
if __name__ == "__main__":
    main = Main()
    main.main()