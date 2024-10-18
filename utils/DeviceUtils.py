import torch
class DeviceUtils:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
        print(f'Using {self.device_name} device')
    def get_device(self):
        return self.device