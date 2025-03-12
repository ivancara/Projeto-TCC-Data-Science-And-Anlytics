from utils.JsonUtils import JsonUtils
from processing.Statistics.Correlation import Correlation
import warnings
warnings.filterwarnings('ignore')
class SummaryDepressionAnalysis:
    def __init__(self, fileUtils, constantsManagement, chart) -> None:
        self.root_file = fileUtils.readFile(sep=';')
        self.file = fileUtils.readFileFromPath(path=constantsManagement.RESULTS_DEPRESSION_ANALYSIS_PATH,sep=';')
        self.chart = chart
        self.correlation = Correlation(constantsManagement=constantsManagement)
        self.jsonutils = JsonUtils()

    def summary(self):
        #Correlation Matrix
        self.chart.heatmap(self.correlation.getCorrelationMatrix(self.root_file), model_name='Correlation Matrix', title='Correlation Matrix', x_label='Features', y_label='Features', statistic_name='', figure_size=(50, 45))
        for key, value in self.file.iterrows():
            #Learning Curve
            data={}
            data['Cross-validation score'] = [self.jsonutils.stringToJson(value['lc_train_sizes']), self.jsonutils.stringToJson(value['lc_test_scores'])]
            data['Training score'] = [self.jsonutils.stringToJson(value['lc_train_sizes']), self.jsonutils.stringToJson(value['lc_train_scores'])]
            self.chart.plot(value['model_name'], title=f'{value['model_name']} Learning Curve\nAccuracy: {value['accuracy_test']:.4f}, MSE: {value['mse']:.4f}, R²: {value['r2']:.4f}, Best Estimator: {value['best_estimator']:.4f}', x_label='Training examples', y_label='Score', statistc_name='learning_curve', data=data)
            #Heatmap Confusion Matrix
            self.chart.heatmap(self.jsonutils.stringToJson(value['confusion_matrix']), model_name=value['model_name'], title=f'{value['model_name']} Classification Report', x_label='Predicted', y_label='Actual', statistic_name='confusion_matrix', figure_size=(10, 5))