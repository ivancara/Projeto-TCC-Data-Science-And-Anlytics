import pandas as pd
class Dummy():

    def __init__(self, data) -> None:
        self.data = data
    
    def getDummy(self, column,applyMapping=None, dtype='bool'):
        if applyMapping is not None:
            self.data[column] = self.data[column].map(applyMapping, na_action=None)
        else:
            if column in self.data:
               self.data = pd.get_dummies(self.data, columns = [column], dtype=dtype)

    def normalize(self):
        self.data = self.data.str.normalize('NFC').str.encode('ascii', errors='ignore').str.decode('utf-8').str.upper()

    def splitColumn(self, column, sep):
        self.data[column] = self.data[column].str.split(sep)
        self.data = self.data.explode(column=column)

