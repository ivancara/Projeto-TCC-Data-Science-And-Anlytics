
import pandas as pd

class FileUtils():
    def __init__(self, fileName=None):
        self.path="./dados/"
        if fileName == None: 
            self.fileName = "dados.csv"
        else:
            self.fileName = fileName
        self.fileOriginal = self.path + self.fileName
        self.fileDataFrame = self.path + "wrangled_" + self.fileName
        self.sep = ','
        
    def readFile(self, sep=','):
        self.fileOriginal = self.path+self.fileName
        self.fileDataFrame = self.path + "wrangled_" + self.fileName
        file = self.fileOriginal
        return pd.read_csv(file, sep=sep, encoding='utf-8')
    
    def writeFile(self, dataFrame):
        return dataFrame.to_csv(self.fileDataFrame, sep=';', encoding='utf-8',index=False, header=True)