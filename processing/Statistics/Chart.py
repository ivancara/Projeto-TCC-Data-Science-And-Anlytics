import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np
class Chart:
    def __init__(self, constantsManagement) -> None:
        self.constantsManagement = constantsManagement


    def plot(self, model_name, title, x_label, y_label, statistc_name, data):
        plt.figure(figsize=(10, 5))
        for idx, (label, (x,y)) in enumerate(data.items()):
            plt.plot(x,y, 'o-', color=sns.color_palette('muted')[idx], label=label)
        plt.xlabel(x_label) 
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(loc='best')
        plt.savefig(f'{self.constantsManagement.RESULTS_PATH}{model_name}_{statistc_name}.png')
        
    def heatmap(self, data, model_name, title, x_label, y_label, statistic_name, figure_size):
        plt.figure(figsize=figure_size)
        sns.heatmap(data, annot=True, cmap='coolwarm')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(f'{self.constantsManagement.RESULTS_PATH}{model_name}_{statistic_name}.png')