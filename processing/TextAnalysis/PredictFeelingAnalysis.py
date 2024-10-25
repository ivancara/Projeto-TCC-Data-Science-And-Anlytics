
import torch.nn.functional as F
import torch
from transformers import  BertTokenizer
from processing.TextAnalysis.SentimentClassifier import SentimentClassifier
class PredictFeelingAnalysis:
    def __init__(self, constantManagement, fileUtils, deviceUtils) -> None:
        self.constantManagement = constantManagement
        self.device = deviceUtils
        self.fileUtils = fileUtils
        self.class_names = constantManagement.FEELINGS_ANALYSIS_CLASSES
        self.tokenizer = BertTokenizer.from_pretrained(constantManagement.PRE_TRAINED_MODEL_NAME)
        self.classifier = SentimentClassifier(len(self.class_names),constantsManagement=constantManagement)
        
    
    def predict(self, text):
        binaryModel = self.fileUtils.loadTorchModel(self.constantManagement.MODEL_FEELINGS_ANALYSIS_PATH)
        self.classifier = self.classifier.to(self.device.get_device())
        self.classifier.load_state_dict(binaryModel)
        self.classifier = self.classifier.eval()
        
        encoded_text = \
            self.tokenizer.encode_plus(text,
                                        add_special_tokens=True,
                                        max_length = self.constantManagement.MAX_LEN,
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
            predicted_class,
            self.class_names[predicted_class],
            confidence,
            dict(zip(self.class_names, probabilities))
        )