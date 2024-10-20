
from processamento.DataTable import DataTable
from processamento.TextAnalysis.TrainingFeelingAnalysis import TrainingFeelingAnalysis

class Main:
    class_names = ['negative', 'neutral', 'positive']

    def __init__(self) -> None:
        self.dataTable = DataTable()
    
    def main(self):
      self.dataTable.writeDataTableIntoFile()  
      column_description = 'descricao_lembranca_passado'
      column_target = 'tipo_lembranca_passado'
      training = TrainingFeelingAnalysis(self.class_names, column_description, column_target)
      training.train()

   
if __name__ == "__main__":
    main = Main()
    main.main()