from sklearn.model_selection import train_test_split
from processing.Statistics.Correlation import Correlation
import warnings
warnings.filterwarnings('ignore')
class DataSplitUtils:
    def __init__(self, constantsManagement) -> None:
        self.constantsManagement = constantsManagement
        self.random_state = constantsManagement.RANDOM_SEED
        self.correlation = Correlation(constantsManagement=constantsManagement)
        self.data = None
        self.features = None
        self.target = None
        pass

    def split_data(self, data, size=0.2):
        correlation_matrix = self.correlation.getCorrelationMatrix(data)
        
        selected_columns = correlation_matrix.columns[correlation_matrix.apply(lambda x: any((x >= 0.7) & (x.index != x.name)), axis=0)]

        df_filtered = data[selected_columns]
        self.features = df_filtered.iloc[:, :-1]
        self.target = data[self.constantsManagement.TARGET]
        X_train, X_test, y_train, y_test = train_test_split(self.features,self.target, test_size=size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test 