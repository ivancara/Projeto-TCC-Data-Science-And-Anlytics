
import pandas as pd
import statsmodels.api as sm 
import torch
from utils.ConstantsManagement import ConstantsManagement
from pathlib import Path
from utils.DeviceUtils import DeviceUtils
class FileUtils():
    def __init__(self, fileName=None):
        self.deviceUtils = DeviceUtils()
        self.constantsManagement = ConstantsManagement()
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
        my_file = Path(fileName)
        if my_file.is_file():
            return sm.load(fileName)
        else:
            return None
    
    def loadTorchModel(self, fileName):
        my_file = Path(fileName)
        if my_file.is_file():
            return torch.load(fileName, map_location=self.deviceUtils.get_device())
        else:
            return None