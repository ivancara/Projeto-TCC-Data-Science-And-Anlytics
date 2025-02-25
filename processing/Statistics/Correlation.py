class Correlation:
    def __init__(self, constantsManagement):
        self.constantsManagement = constantsManagement

    def pearson(self):
        return self.data.corr(method='pearson')

    def spearman(self):
        return self.data.corr(method='spearman')

    def kendall(self):
        return self.data.corr(method='kendall')
    
    def getCorrelationMatrix(self, data):
        self.data = data
        method=self.constantsManagement.CORRELATION
        match method:
            case 'pearson':
                return self.pearson()
            case 'spearman':
                return self.spearman()
            case 'kendall':
                return self.kendall()
            case _:
                return self.pearson()
    