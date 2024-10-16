
import pandas as pd

class FileUtils():
    def __init__(self):
        self.path="./dados/"
        self.fileName = "dados.csv"
        self.fileOriginal = self.path + self.fileName
        self.fileDataFrame = self.path + "wrangled_" + self.fileName
        self.sep = ','
        
    def readFile(self, file=None, sep=','):
        if file == None:
            file = self.fileName
        self.fileOriginal = self.path+file
        self.fileDataFrame = self.path + "wrangled_" + file
        file = self.fileOriginal
        return pd.read_csv(file, sep=sep, encoding='utf-8')
    
    def writeFile(self, dataFrame):
        return dataFrame.to_csv(self.fileDataFrame, sep=';', encoding='utf-8',index=False, header=True)