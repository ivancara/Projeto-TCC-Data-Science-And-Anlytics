import pandas as pd
class Dummy():
    def __init__(self, data) -> None:
        self.data = data
    
    def getDummy(self, column, sameColumn=False,applyMapping=None, dtype='bool'):
        if applyMapping is not None:
            self.data = self.data.map(applyMapping, na_action=None)
        else:
            if sameColumn == True:
                self.data[column] = self.data[column].astype(dtype)
            else: 
               self.data = pd.get_dummies(self.data, columns = [column], dtype=dtype)
               if column in self.data:
                    self.data = self.data.drop(columns=[column])