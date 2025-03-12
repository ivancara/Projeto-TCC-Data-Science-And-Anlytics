
import pandas as pd
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
class FileUtils():
    def __init__(self, constantsManagement, fileName=None) -> None:
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
    
    def readFileFromPath(self, sep=',', path=None):
        if path == None:
            path = self.fileOriginal
        return pd.read_csv(path, sep=sep, encoding='utf-8')
    
    def writeFile(self, dataFrame):
        file = pd.DataFrame(data=dataFrame.flatten())
        return file.to_csv(self.file_out, sep=';', encoding='utf-8',index=False, header=True)
    
    def writeDataframeFile(self, dataFrame, fileName):
        return dataFrame.to_csv(fileName, sep=';', encoding='utf-8',index=False, header=True)
    
    def saveModel(self, model, fileName):
        if not self.hasFile(fileName):
            return joblib.dump(model, self.constantsManagement.MODEL_DEPRESSION_ANALYSIS_PATH)
        else:
            return None

    def deleteFile(self, fileName):
        if self.hasFile(fileName):
            return Path(fileName).unlink()
        else:
            return    
    
    def loadModel(self, fileName):
        if self.hasFile(fileName):
            return joblib.load(fileName)
        else:
            return None
        
    def hasFile(self, fileName):
        my_file = Path(fileName)
        if my_file.is_file():
            return True
        else:
            return False
        