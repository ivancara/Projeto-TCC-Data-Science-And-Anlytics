import pandas as pd
from pandas.api.types import is_numeric_dtype
class Dummy():

    def __init__(self, data) -> None:
        self.data = data
    
    def getDummy(self, column, newColumn=None,applyMapping=None, dtype='bool'):
        date_columns = self.data.columns[self.data.apply(lambda col: col.astype(str).str.contains(r'\d{4}/\d{2}/\d{2} \d{1,2}:\d{2}:\d{2} [AP]M GMT[+-]\d').any())]
        #parse all date columns to datetime
        for date_column in date_columns:
            date_data = pd.to_datetime(self.data[date_column], errors='coerce')
            #create new columns with month, day and year
            self.data[date_column + '_year'] = date_data.dt.year
            self.data[date_column + '_month'] = date_data.dt.month
            self.data[date_column + '_day'] = date_data.dt.day
            
            
            self.data = self.data.drop(columns=[date_column])
            
        self.data = self.data.replace({'True': 1, 'False': 0})
        
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
