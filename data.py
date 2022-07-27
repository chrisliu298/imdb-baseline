import os

import numpy as np
import torch
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from torchtext import data, vocab


class TextDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
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

    def corrupt_train_labels(self, labels, corrupt_prob):
        true_labels = np.array(labels)
        labels = np.array(labels)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        labels[mask] = 1 - labels[mask]
        print(accuracy_score(true_labels, labels))
        return torch.tensor(labels)


class IMDBDataModule(TextDataModule):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = data.utils.get_tokenizer("spacy", language="en_core_web_sm")
        self.train_data = load_dataset("imdb", split="train")
        self.test_data = load_dataset("imdb", split="test")
        self.vocabulary = vocab.build_vocab_from_iterator(
            self._yield_tokens(self.train_data),
            specials=["<unk>"],
            max_tokens=self.config.vocab_size,
        )
        self.vocabulary.set_default_index(self.vocabulary["<unk>"])

        def text_pipeline(x):
            x = self.tokenizer(x)
            x = self.vocabulary(x)
            return torch.tensor(x)[-self.config.seq_len :]

        train_texts, test_texts = [], []
        train_labels, test_labels = [], []
        for sample in self.train_data:
            train_texts.append(text_pipeline(sample["text"]))
            train_labels.append(sample["label"])
        for sample in self.test_data:
            test_texts.append(text_pipeline(sample["text"]))
            test_labels.append(sample["label"])

        if self.config.label_noise > 0.0:
            train_labels = self.corrupt_train_labels(
                train_labels, self.config.label_noise
            )
        train_dataset = TensorDataset(
            pad_sequence(train_texts, batch_first=True),
            torch.tensor(train_labels),
        )
        indices = np.arange(len(train_dataset))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=True)
        self.train_dataset = TensorDataset(*train_dataset[train_idx])
        self.val_dataset = TensorDataset(*train_dataset[val_idx])
        self.test_dataset = TensorDataset(
            pad_sequence(test_texts, batch_first=True), torch.tensor(test_labels)
        )

    def _yield_tokens(self, data_iter):
        for sample in data_iter:
            yield self.tokenizer(sample["text"])
