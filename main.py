import pandas as pd
from processing.TextAnalysis.TrainingFeelingAnalysis import TrainingFeelingAnalysis
from processing.TextAnalysis.PredictFeelingAnalysis import PredictFeelingAnalysis
from processing.DepressionAnalysis.TrainingDepression import TrainingDepression
from processing.DepressionAnalysis.PredictDepression import PredictDepression
from processing.Data.DataTable import DataTable
from utils.FileUtils import FileUtils
from utils.ConstantsManagement import ConstantsManagement
class Main:
    def __init__(self) -> None:
        pass
    def main(self): 
        while True:
            print("-"*34)
            print("1 - Training Feeling Analysis")
            print("2 - Predict Feeling Analysis")
            print("3 - Training Depression Analysis")
            print("4 - Predict Depression Analysis")
            print("5 - Statistics Feeling Analysis")
            print("6 - Statistics Depression Analysis")
            print("7 - Process Data")
            print("8 - Exit")
            print("-"*34)
            option = int(input("Choose an option: "))
            match option:
                case 1:
                    text='lembranca_atual_futuro'
                    target='tipo_lembranca_atual'
                    training = TrainingFeelingAnalysis(text, target)
                    training.train()
                    pass
                case 2:
                    predict = PredictFeelingAnalysis()
                    description = input("Enter the some text: ")
                    print(predict.predict(description))
                    pass
                case 3:
                    target = 'possui_depressao'
                    fieldsToValidate = ['lembranca_atual_futuro_predicted',
                                 'descricao_lembranca_passado_predicted',
                                 'terapia',
                                 'identifica_emocoes',
                                 'genero_Feminino',
                                 'genero_Masculino']
                    training = TrainingDepression(target=target, fieldsToValidate=fieldsToValidate)
                    training.train()
                    pass
                case 4:
                    predict = PredictDepression()
                    predict = PredictFeelingAnalysis()
                    description = input("Describe some history that enhance your decisions in future: ")
                    lembranca_atual_futuro_predicted = predict.predict(description)
                    description = input("Remember some history that enhance your decisions today: ")
                    descricao_lembranca_passado_predicted = predict.predict(description)
                    terapia = int(input("Do you have therapy? (1-yes or 0-no): "))
                    identifica_emocoes = int(input("Do you identify emotions? (1-yes or 0-no): "))
                    genero = input("Enter your Genre (Male: 0 or Female: 1): ")
                    data = pd.DataFrame({'lembranca_atual_futuro_predicted': lembranca_atual_futuro_predicted,
                                            'descricao_lembranca_passado_predicted': descricao_lembranca_passado_predicted,
                                         'terapia': terapia,
                                         'identifica_emocoes': identifica_emocoes,
                                         'genero_Feminino': 1 if genero == 1 else 0,
                                         'genero_Masculino': 1 if genero == 0 else 0}, index=[0])
                    predict.predict(data)
                    pass
                case 5:
                    print("Statistics Feeling Analysis")
                    pass
                case 6:
                    print("Statistics Feeling Analysis")
                    pass
                case 7:
                    fileUtils = FileUtils()
                    constantManagement = ConstantsManagement()
                    data = DataTable()
                    addPredictedData = fileUtils.hasFile(constantManagement.MODEL_FEELINGS_ANALYSIS_PATH) \
                                and fileUtils.hasFile(constantManagement.MODEL_DEPRESSION_PATH)
                    data.writeDataTableIntoFile(addPredictedData)
                    pass
                case 8:
                    exit()
                case _:
                    print("Invalid option")
                    continue

if __name__ == "__main__":
    main = Main()
    main.main()