import torch
from torch.utils.data import Dataset
from setup import get_item


class HFDataset(Dataset):
    def __init__(self, ds_name, ds, n_label, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ds_name = ds_name
        self.ds = ds
        self.n_label = n_label
        self.texts, self.labels = self._data_formalize_()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _data_formalize_(self):
        texts = []
        labels = []

        for data in self.ds:
            doc_text, doc_label = get_item(self.ds_name, data)
            texts.append(doc_text)
            labels.append(doc_label)
        
        return texts, labels
    

class LLMDataset(Dataset):
    def __init__(self, ds, label_ls, max_len=128):
        self.max_len = max_len
        self.ds = ds
        self.label_ls = label_ls

    def __len__(self):
        return len(self.label_ls)

    def __getitem__(self, idx):
        labels = torch.as_tensor(self.label_ls[idx], dtype=torch.long)

        return {
            'input_ids': self.ds[idx]['input_ids'],
            'attention_mask': self.ds[idx]['attention_mask'].squeeze(0),
            'labels': labels
        }
    
