from sklearn.model_selection import train_test_split
from processing.Statistics.Correlation import Correlation
class DataSplitUtils:
    def __init__(self, constantsManagement) -> None:
        self.random_state = constantsManagement.RANDOM_SEED
        self.correlation = Correlation(constantsManagement=constantsManagement)
        self.data = None
        self.features = None
        self.target = None
        pass

    def split_data(self, data, size=0.2):
        correlation_matrix = self.correlation.getCorrelationMatrix(data)
        selected_columns = correlation_matrix[correlation_matrix >= 0.7].index
        df_filtered = data[selected_columns]
        self.features = df_filtered.iloc[:, :-1]
        self.target = df_filtered.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(self.features,self.target, test_size=size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test 