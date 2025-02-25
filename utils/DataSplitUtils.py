from sklearn.model_selection import train_test_split
from processing.Statistics.Correlation import Correlation
class DataSplitUtils:
    def __init__(self, constantsManagement) -> None:
        self.random_state = constantsManagement.RANDOM_SEED
        self.correlation = Correlation(constantsManagement=constantsManagement)
        pass

    def split_data(self, data, size=0.2):
        correlation_matrix = self.correlation.getCorrelationMatrix(data)
        target_correlation = correlation_matrix.iloc[:, -1].abs()
        selected_columns = target_correlation[target_correlation >= 0.7].index
        df_filtered = data[selected_columns]

        return train_test_split(df_filtered, test_size=size, random_state=self.random_state)