
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from processamento.DataTable import DataTable
from processamento.FileUtils import FileUtils
from processamento.TextAnalysis.PredictFeelingAnalysis import PredictFeelingAnalysis
from processamento.TextAnalysis.TrainingFeelingAnalysis import TrainingFeelingAnalysis
import pandas as pd
import seaborn as sns

class Main:
    def __init__(self) -> None:
        self.dataTable = DataTable()
    def show_confusion_matrix(confusion_matrix):
        hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True sentiment')
        plt.xlabel('Predicted sentiment')


    def main(self):
      #self.dataTable.writeDataTableIntoFile()  
      #column_description = 'lembranca_atual_futuro'
      #column_target = 'tipo_lembranca_atual'
      #training = TrainingFeelingAnalysis(column_description, column_target)      
      #best_accuracy, history = training.train()
      #print(f'Best accuracy: {best_accuracy}')
      predictions = PredictFeelingAnalysis()   
      print('ganhei um cocô', predictions.predict('ganhei um cocô'))
        


        
if __name__ == "__main__":
    main = Main()
    main.main()