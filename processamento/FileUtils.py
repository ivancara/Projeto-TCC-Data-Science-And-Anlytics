
import pandas as pd

class FileUtils():
    def __init__(self):
        self.path="./dados/"
        self.fileOriginal=self.path+"dados.csv"
        self.fileDataFrame=self.path+"dados_wrangled.csv"
        
    def readFile(self, file=None):
        if file == None:
            file = self.fileOriginal
        return pd.read_csv(file)
    
    def writeFile(self, dataFrame):
        return dataFrame.to_csv(self.fileDataFrame, sep=';', encoding='utf-8',index=False, header=True)