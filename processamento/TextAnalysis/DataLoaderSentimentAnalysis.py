from torch.utils.data import Dataset, DataLoader
from processamento.TextAnalysis.FactDataset import FactDataset
class DataLoaderSentimentAnalysis:
    def __init__(self, tokenizer, max_len,batch_size):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
    def create_data_loader(self, text, targets):
        ds = FactDataset(
            facts=text.to_numpy(),
            targets=targets.to_numpy(),
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=4
        ) 