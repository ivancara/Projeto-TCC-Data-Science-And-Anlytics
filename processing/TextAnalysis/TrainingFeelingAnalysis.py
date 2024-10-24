import torch
import torch.nn as nn
import numpy as np
from transformers import  BertTokenizer, AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
from torcheval.metrics import R2Score
from utils.FileUtils import FileUtils
from processing.TextAnalysis.DataLoaderSentimentAnalysis import DataLoaderSentimentAnalysis
from processing.TextAnalysis.SentimentClassifier import SentimentClassifier
from utils.DataSplitUtils import DataSplitUtils
from utils.DeviceUtils import DeviceUtils
from utils.ConstantsManagement import ConstantsManagement


class TrainingFeelingAnalysis:
    def __init__(self, text, targets):     
        self.metric = R2Score()
        self.text = text
        self.targets = targets
        self.deviceUtils = DeviceUtils()
        self.device = self.deviceUtils.get_device()
        self.loss_fn = nn.CrossEntropyLoss().to(self.device) 
        self.constantsManagement = ConstantsManagement()
        self.fileUtils = FileUtils(self.constantsManagement.WRANGLED_DATA_FINAL)
        self.class_names = self.constantsManagement.FEELINGS_ANALYSIS_CLASSES
        self.model = SentimentClassifier(len(self.class_names))
        self.model = self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.constantsManagement.PRE_TRAINED_MODEL_NAME)
        self.df_train, self.df_val, self.df_test = self.split_traininig_test(training_size=self.constantsManagement.TRAIN_PERCENTAGE, test_size=self.constantsManagement.TEST_PERCENTAGE)
        self.data_loader = DataLoaderSentimentAnalysis(self.tokenizer, self.constantsManagement.MAX_LEN, self.constantsManagement.BATCH_SIZE)
        self.dataUtils = DataSplitUtils()
    
    def optimizer(self):
        return AdamW(self.model.parameters(), lr=self.constantsManagement.LEARNING_RATE, correct_bias=False, no_deprecation_warning=True)
     
    def train_epoch(self,  n_examples ):
        self.model = self.model.train()
        train_data_loader = self.train_data_loader()
        correct_predictions, losses, r2 = self.correct_predictions_losses(optimize=True,data_loader= train_data_loader)
        return (correct_predictions.double()).to('cpu').item() / n_examples, np.mean(losses), r2
    
    def split_traininig_test(self, training_size, test_size):
        self.seeder()
        df = self.fileUtils.readFile(';')
        df_train, df_test = self.dataSplitUtils.split_data(df, size=training_size)
        df_val, df_test = self.dataSplitUtils.split_data(df_test, size=test_size)
        return df_train, df_val, df_test
    
    def correct_predictions_losses(self, optimize=False, data_loader=None):
      losses = []
      correct_predictions = 0
      optimizer = self.optimizer()
      scheduler = self.scheduler(data_loader=data_loader, optimizer=optimizer)
      
      for d in data_loader:
        outputs, targets = self.generate_model(d)

        if len(outputs.shape) == 2 and outputs.shape[1] == len(self.class_names):
            _, preds = torch.max(outputs, dim=1)
            loss = self.loss_fn(outputs, targets)
        else:
            raise ValueError(f"Unexpected output shape: {outputs.shape}")
        self.metric.update(preds.to('cpu'), targets.to('cpu'))
        r2 = self.metric.compute().to('cpu').item() 
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        if optimize == True:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        return correct_predictions, losses,r2
    
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
        correct_predictions, losses, r2 = self.correct_predictions_losses(optimize=False,data_loader=val_data_loader)

      return (correct_predictions.double() / n_examples).to('cpu').item(), np.mean(losses), r2
    
    def seeder(self):
        np.random.seed(self.constantsManagement.RANDOM_SEED)
        torch.manual_seed(self.constantsManagement.RANDOM_SEED)
    
    def scheduler(self, data_loader, optimizer=None):
        total_steps = len(data_loader) * self.constantsManagement.EPOCHS
        return get_linear_schedule_with_warmup(
            optimizer,
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
        best_r2 = -100

        for epoch in range(self.constantsManagement.EPOCHS):
            print(f'Epoch {epoch + 1}/{self.constantsManagement.EPOCHS} - Best Accuracy: {best_accuracy} - Best R2 {best_r2}')
            print('-' * 100)

            train_acc, train_loss, train_r2 = self.train_epoch(n_examples=n_examples)

            print(f'Train loss {train_loss} accuracy {train_acc}, train r2 {train_r2}')
            print('-' * 100)
            val_acc, val_loss, val_r2 = self.eval_model(n_examples=n_examples)

            print(f'Val loss {val_loss} accuracy {val_acc}, val r2 {val_r2}')
            print('-' * 100)

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)

            if val_r2 > best_r2 and best_r2 < 1:
                torch.save(self.model.state_dict(), self.constantsManagement.MODEL_FEELINGS_ANALYSIS_PATH)
                best_accuracy = val_acc
                best_r2 = val_r2
        return best_accuracy, history, best_r2