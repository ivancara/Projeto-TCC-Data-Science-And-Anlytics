
import pandas as pd

from utils.ConstantsManagement import ConstantsManagement

class FileUtils():
    def __init__(self, fileName=None):
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