
import pandas as pd
import statsmodels.api as sm 
import torch
from pathlib import Path
class FileUtils():
    def __init__(self, deviceUtils, constantsManagement, fileName=None) -> None:
        self.deviceUtils = deviceUtils
        self.constantsManagement = constantsManagement
        self.path=self.constantsManagement.DIRECTORY_DATA
        if fileName == None: 
            self.fileName = self.constantsManagement.FILE_DATA
        else:
            self.fileName = fileName
        self.fileOriginal = self.path + self.fileName
        self.file_out = self.path + 'out_' + self.fileName
        self.sep = ','
        
    def readFile(self, sep=','):
        self.fileOriginal = self.path+self.fileName
        self.file_out = self.path + 'out_' + self.fileName
        file = self.fileOriginal
        return pd.read_csv(file, sep=sep, encoding='utf-8')
    
    def writeFile(self, dataFrame):
        return dataFrame.to_csv(self.file_out, sep=';', encoding='utf-8',index=False, header=True)
    
    def loadModelStatsModel(self, fileName):
        if self.hasFile(fileName):
            return sm.load(fileName)
        else:
            return None
    
    def loadTorchModel(self, fileName):
        if self.hasFile(fileName):
            return torch.load(fileName, map_location=self.deviceUtils.get_device())
        else:
            return None
        
    def hasFile(self, fileName):
        my_file = Path(fileName)
        if my_file.is_file():
            return True
        else:
            raise ValueError("File not found")