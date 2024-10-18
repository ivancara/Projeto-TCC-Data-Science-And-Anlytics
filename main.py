
import tqdm
from processamento.DataTable import DataTable
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from processamento.TextAnalysis.TrainingFeelingAnalysis import TrainingFeelingAnalysis
from processamento.TextAnalysis.SentimentClassifier import SentimentClassifier


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