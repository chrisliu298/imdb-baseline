import os

import numpy as np
import torch
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing, TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
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
        if self.config.vocab_size != None:  # train a customized tokenizer
            tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
            tokenizer.normalizer = normalizers.Sequence(
                [NFD(), Lowercase(), StripAccents()]
            )
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", 1),
                    ("[SEP]", 2),
                ],
            )
            trainer = WordPieceTrainer(
                vocab_size=self.config.vocab_size,
                special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                show_progress=False,
            )
            tokenizer.train_from_iterator(
                train_data["text"] + test_data["text"], trainer
            )
            tokenizer.enable_padding(length=self.config.max_seq_len)
            tokenizer.enable_truncation(max_length=self.config.max_seq_len)
            train_encodings_raw = tokenizer.encode_batch(train_data["text"])
            test_encodings_raw = tokenizer.encode_batch(test_data["text"])
            self.train_encodings = {
                "input_ids": torch.tensor([x.ids for x in train_encodings_raw]),
                "attention_mask": torch.tensor(
                    [x.attention_mask for x in train_encodings_raw]
                ),
            }
            self.test_encodings = {
                "input_ids": torch.tensor([x.ids for x in test_encodings_raw]),
                "attention_mask": torch.tensor(
                    [x.attention_mask for x in test_encodings_raw]
                ),
            }
            self.vocab_size = tokenizer.get_vocab_size()
        else:  # use a bert pre-trained tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            encode = lambda x: self.tokenizer(
                x,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_seq_len,
                return_tensors="pt",
            )
            self.train_encodings = encode(train_data["text"])
            self.test_encodings = encode(test_data["text"])
            self.vocab_size = self.tokenizer.vocab_size

    def setup(self, stage=None):
        # download dataset
        train_data = load_dataset("imdb", split="train")
        test_data = load_dataset("imdb", split="test")
        # extract input ids and attention masks (which are converted to BoolTensor format)
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
        train_input_ids = self.train_encodings["input_ids"]
        train_attention_mask = self.train_encodings["attention_mask"].logical_not()
        test_input_ids = self.test_encodings["input_ids"]
        test_attention_mask = self.test_encodings["attention_mask"].logical_not()
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
