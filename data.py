import torch
from torch.utils.data import Dataset

# Custom Dataset class handling the tokenization
class CustomDataset(Dataset):
    def __init__(self, data, targets, tokenizer, max_len):
        self.data = data  # Df containing filepaths and labels of imgs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data[index]
        encoding = self.tokenizer.encode_plus(sentence,
                                              max_length=self.max_len,
                                              add_special_tokens=True,
                                              padding='max_length',
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_token_type_ids=False,
                                              return_tensors='pt')

        return {
            'input_ids': torch.squeeze(encoding['input_ids'], dim=0),
            'attention_mask': torch.squeeze(encoding['attention_mask'], dim=0),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }