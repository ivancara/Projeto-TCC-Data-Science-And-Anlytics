import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')
class JsonUtils:
    def __init__(self):
        pass
    
    def stringToJson(self, string):
        return json.loads(string)
    
    
    def jsonToString(self, jsonData):
        return json.dumps(jsonData)