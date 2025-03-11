class SummaryDepressionAnalysis:
    def __init__(self, fileUtils, constantsManagement) -> None:
        constrants = constantsManagement
        fileUtils = fileUtils
        self.model = fileUtils.loadModel(constrants.MODEL_DEPRESSION_ANALYSIS_PATH)

    def get_summary(self):
        return self.model.summary()
    
    def get_confusion_matrix(self):
        return self.model