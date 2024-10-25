class SummaryTextAnalysis:
    def __init__(self, fileUtils) -> None:
        self.history = fileUtils.readFile(';')
        pass
    
    def get_summary(self):
        print(self.history.describe())
        data = self.history.tail(1)
        print('-'*100)
        print(f"Train Accuracy: {data['train_acc'].values[0]}")
        print(f"Train Loss: {data['train_loss'].values[0]}")
        print(f"Validation Accuracy: {data['val_acc'].values[0]}")
        print(f"Validation Loss: {data['val_loss'].values[0]}")
        print(f"Train R Squared: {data['train_r2'].values[0]}")
        print(f"Validation R Squared: {data['val_r2'].values[0]}")
        print('-'*100) 