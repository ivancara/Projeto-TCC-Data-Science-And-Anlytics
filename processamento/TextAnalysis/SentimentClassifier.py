from torch import nn
from transformers import BertModel

from utils.ConstantsManagement import ConstantsManagement
class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.constantsManagement = ConstantsManagement()
        self.bert = BertModel.from_pretrained(self.constantsManagement.PRE_TRAINED_MODEL_NAME, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        #The last_hidden_state is a sequence of hidden states of the last layer of the model
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)