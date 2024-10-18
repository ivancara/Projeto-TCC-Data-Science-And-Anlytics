import torch
from torch.utils.data import Dataset
class FactDataset(Dataset):

  def __init__(self, facts, targets, tokenizer, max_len):
    self.facts = facts
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.facts)
  
  def __getitem__(self, item):
    facts = str(self.facts[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      facts,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    valor = {
        'facts_text': facts,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'targets': torch.tensor(target, dtype=torch.long)
      }

    return valor