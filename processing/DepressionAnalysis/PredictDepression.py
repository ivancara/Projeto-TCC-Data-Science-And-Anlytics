
import numpy as np
from utils.ConstantsManagement import ConstantsManagement
from utils.FileUtils import FileUtils
class PredictDepression:
    def __init__(self):
        pass
    
    def loadModel(self):
        fileUtils = FileUtils()
        self.model = fileUtils.loadModelStatsModel(self.constantsManagement.MODEL_DEPRESSION_PATH)
        return self.model
    
    def predict(self, data):
        self.model = self.loadModel()
        predict = np.round(self.model.predict(data),0).astype(int)
        return predict