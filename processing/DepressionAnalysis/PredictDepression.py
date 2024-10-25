
import numpy as np
class PredictDepression:
    def __init__(self, constantsManagement, fileUtils) -> None:
        self.constantsManagement = constantsManagement
        self.fileUtils = fileUtils
        pass
    
    def loadModel(self):
        self.model = self.fileUtils.loadModelStatsModel(self.constantsManagement.MODEL_DEPRESSION_ANALYSIS_PATH)
        return self.model
    
    def predict(self, data):
        self.model = self.loadModel()
        predicted = self.model.predict(data)
        predict = np.round(predicted,0).astype(int)
        return predict