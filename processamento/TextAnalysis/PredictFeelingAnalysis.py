
import torch.nn.functional as F
import torch
from transformers import  BertTokenizer
from processamento.TextAnalysis.SentimentClassifier import SentimentClassifier
from utils.DeviceUtils import DeviceUtils
from utils.ConstantsManagement import ConstantsManagement
class PredictFeelingAnalysis:
    def __init__(self):
        self.device = DeviceUtils()
        self.constrantManagement = ConstantsManagement()
        self.class_names = self.constrantManagement.FEELINGS_ANALYSIS_CLASSES
        self.tokenizer = BertTokenizer.from_pretrained(self.constrantManagement.PRE_TRAINED_MODEL_NAME)
        self.classifier =SentimentClassifier(len(self.class_names))
        self.classifier.load_state_dict(torch.load(self.constrantManagement.MODEL_FEELINGS_ANALYSIS_PATH, map_location=self.device.get_device()))
        self.classifier = self.classifier.eval()
        self.classifier = self.classifier.to(self.device.get_device())
    
    def predict(self, text):

        encoded_text = \
            self.tokenizer.encode_plus( text,
                                        add_special_tokens=True,
                                        max_length = self.constrantManagement.MAX_LEN,
                                        truncation=True,
                                        return_token_type_ids=False,
                                        padding='max_length',
                                        pad_to_max_length=True,
                                        return_attention_mask=True,
                                        return_tensors='pt',)
        input_ids = encoded_text["input_ids"].to(self.device.get_device())
        attention_mask = encoded_text["attention_mask"].to(self.device.get_device())

        with torch.no_grad():
            #Getting the prediction
            probabilities = F.softmax(self.classifier(input_ids, attention_mask),
                            dim = 1)

        #Taking the most confident result
        confidence, predicted_class = torch.max(probabilities, dim = 1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()

        return(
            self.class_names[predicted_class],
            confidence,
            dict(zip(self.class_names, probabilities))
        )