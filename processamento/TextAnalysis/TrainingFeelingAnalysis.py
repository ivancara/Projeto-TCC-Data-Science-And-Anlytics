import torch
import torch.nn as nn
import numpy as np
from transformers import  BertTokenizer, AdamW, get_linear_schedule_with_warmup
from processamento.FileUtils import FileUtils
from processamento.TextAnalysis.DataLoaderSentimentAnalysis import DataLoaderSentimentAnalysis
from processamento.TextAnalysis.SentimentClassifier import SentimentClassifier
from utils.DeviceUtils import DeviceUtils
from utils.ConstantsManagement import ConstantsManagement
from collections import defaultdict
from sklearn.model_selection import train_test_split
class TrainingFeelingAnalysis:
    def __init__(self,  class_names, text, targets):     
        self.class_names = class_names
        self.text = text
        self.targets = targets
        self.deviceUtils = DeviceUtils()
        self.device = self.deviceUtils.get_device()
        self.loss_fn = nn.CrossEntropyLoss().to(self.device) 
        self.model = SentimentClassifier(len(self.class_names))
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=3e-5, correct_bias=False)
        self.constrantsManagement = ConstantsManagement()
        self.fileUtils = FileUtils(self.constrantsManagement.WRANGLED_DATA_FINAL)
        self.tokenizer = BertTokenizer.from_pretrained(self.constrantsManagement.PRE_TRAINED_MODEL_NAME)
        self.df_train, self.df_val, self.df_test = self.split_traininig_test(training_size=0.1, test_size=0.5)
        self.data_loader = DataLoaderSentimentAnalysis(self.tokenizer, self.constrantsManagement.MAX_LEN, self.constrantsManagement.BATCH_SIZE)
        
    def train_epoch(self,  n_examples ):
        self.model = self.model.train()
        train_data_loader = self.train_data_loader()
        correct_predictions, losses = self.correct_predictions_losses(optimize=True,data_loader= train_data_loader)
        return correct_predictions.double() / n_examples, np.mean(losses)
    
    def split_traininig_test(self, training_size, test_size):
        self.seeder()
        df = self.fileUtils.readFile(';')
        df_train, df_test = train_test_split(df, test_size=training_size, random_state=self.constrantsManagement.RANDOM_SEED)
        df_val, df_test = train_test_split(df_test, test_size=test_size, random_state=self.constrantsManagement.RANDOM_SEED)
        return df_train, df_val, df_test
    
    def correct_predictions_losses(self, optimize=False, data_loader=None):
      losses = []
      correct_predictions = 0
      
      for d in data_loader:
        outputs, targets = self.generate_model(d)

        if len(outputs.shape) == 2 and outputs.shape[1] == len(self.class_names):
            _, preds = torch.max(outputs, dim=1)
            loss = self.loss_fn(outputs, targets)
        else:
            raise ValueError(f"Unexpected output shape: {outputs.shape}")
        if optimize:
            scheduler = self.scheduler(data_loader=data_loader)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            scheduler.step()
            self.optimizer.zero_grad()
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        return correct_predictions, losses
    
    def generate_model(self, data):
        input_ids = data["input_ids"].to(self.device)
        attention_mask = data["attention_mask"].to(self.device)
        targets = data["targets"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
          )
        return outputs, targets
    
    def eval_model(self, n_examples):
      self.model = self.model.eval()
      val_data_loader = self.val_data_loader()
      losses = []
      correct_predictions = 0

      with torch.no_grad():
        correct_predictions, losses = self.correct_predictions_losses(optimize=False,data_loader=val_data_loader)

      return correct_predictions.double() / n_examples, np.mean(losses)
    
    def seeder(self):
        np.random.seed(self.constrantsManagement.RANDOM_SEED)
        torch.manual_seed(self.constrantsManagement.RANDOM_SEED)
    
    def scheduler(self, data_loader):
        total_steps = len(data_loader) * self.constrantsManagement.EPOCHS
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
    def val_data_loader(self):
        return self.data_loader.create_data_loader(text=self.df_val[self.text], targets=self.df_val[self.targets])
    def test_data_loader(self):
        return self.data_loader.create_data_loader(text=self.df_test[self.text], targets=self.df_test[self.targets])
    def train_data_loader(self):
        return self.data_loader.create_data_loader(text=self.df_train[self.text], targets=self.df_train[self.targets])
    def train(self):
        self.seeder()
 
        n_examples = len(self.df_train)
        history = defaultdict(list)
        best_accuracy = 0

        for epoch in range(self.constrantsManagement.EPOCHS):

            print(f'Epoch {epoch + 1}/{self.constrantsManagement.EPOCHS}')
            print('-' * 10)

            train_acc, train_loss = self.train_epoch(n_examples=n_examples)

            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval_model(n_examples=n_examples)

            print(f'Val loss {val_loss} accuracy {val_acc}')


            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(self.model.state_dict(), self.constrantsManagement.MODEL_FEELINGS_ANALYSIS_PATH)
                best_accuracy = val_acc
        return best_accuracy, history