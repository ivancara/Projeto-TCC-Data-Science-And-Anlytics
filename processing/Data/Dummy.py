import pandas as pd
from pandas.api.types import is_numeric_dtype
class Dummy():

    def __init__(self, data) -> None:
        self.data = data
    
    def getDummy(self, column, newColumn=None,applyMapping=None, dtype='bool'):
        if applyMapping is not None:
            if newColumn is None:
                newColumn = column
            self.data[newColumn] = self.data[column].map(applyMapping, na_action=None)
            if is_numeric_dtype(self.data[column]):
                self.data[column] = self.data[column].astype('int64')
        else:
            if column in self.data:
               self.data = pd.get_dummies(self.data, columns = [column], dtype=dtype)


    def splitColumn(self, column, sep):
        self.data[column] = self.data[column].str.split(sep)
        self.data = self.data.explode(column=column)
        self.data = self.data.drop_duplicates().reset_index(drop=True)

