from utils.ConstantsManagement import ConstantsManagement
from utils.FileUtils import FileUtils
class SummaryDepressionAnalysis:
    def __init__(self):
        constrants = ConstantsManagement()
        fileUtils = FileUtils()
        self.model = fileUtils.loadModelStatsModel(constrants.MODEL_DEPRESSION_ANALYSIS_PATH)

    def get_summary(self):
        return self.model.summary()
    
    def get_confusion_matrix(self):
        return self.model