from sklearn.model_selection import train_test_split
class DataSplitUtils:
    def __init__(self, constantsManagement) -> None:
        self.random_state = constantsManagement.RANDOM_SEED
        pass

    def split_data(self, data, size=0.2):
        return train_test_split(data, test_size=size, random_state=self.random_state)