import os

import numpy as np
import torch
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer


class IMDBDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        # download data
        train_data = load_dataset("imdb", split="train")
        test_data = load_dataset("imdb", split="test")
        # tokenization
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        encode = lambda x: self.tokenizer(
            x,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt",
        )
        self.train_text_encoded = encode(train_data["text"])
        self.test_text_encoded = encode(test_data["text"])

    def setup(self, stage=None):
        train_data = load_dataset("imdb", split="train")
        test_data = load_dataset("imdb", split="test")

        train_input_ids = self.train_text_encoded["input_ids"]
        train_attention_mask = self.train_text_encoded["attention_mask"].logical_not()
        test_input_ids = self.test_text_encoded["input_ids"]
        test_attention_mask = self.test_text_encoded["attention_mask"].logical_not()
        train_labels = torch.tensor(train_data["label"])
        test_labels = torch.tensor(test_data["label"])

        # split train/val
        train_dataset = TensorDataset(
            train_input_ids, train_attention_mask, train_labels
        )
        indices = np.arange(len(train_dataset))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=True)
        self.train_dataset = TensorDataset(*train_dataset[train_idx])
        self.val_dataset = TensorDataset(*train_dataset[val_idx])
        self.test_dataset = TensorDataset(
            test_input_ids, test_attention_mask, test_labels
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )
