import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from torchmetrics.functional import accuracy


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransformerEncoderModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.pos_embedding = PositionalEncoding(
            self.config.embedding_dim, max_len=self.config.max_seq_len
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.embedding_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.embedding_dim * 4,
            dropout=self.config.dropout,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.config.num_layers
        )
        self.fc = nn.Linear(self.config.embedding_dim, self.config.output_size)

    def forward(self, x, padding_mask=None):
        mask = nn.Transformer.generate_square_subsequent_mask(
            self.config.max_seq_len
        ).to(self.device)
        x = x.long()
        x = self.embedding(x) * math.sqrt(self.config.embedding_dim)
        x = x.permute(1, 0, 2)
        x = self.pos_embedding(x)
        if self.config.mask_type == "mask":
            padding_mask = None
        if self.config.mask_type == "padding_mask":
            mask = None
        if self.config.mask_type == "none":
            mask = padding_mask = None
        x = self.encoder(x, mask=mask, src_key_padding_mask=padding_mask)
        x = x[-1, :, :]  # use the last hidden state
        return self.fc(x)

    def evaluate(self, batch, stage=None):
        x, padding_mask, y = batch
        output = self(x, padding_mask)
        loss = F.cross_entropy(output, y)
        acc = accuracy(output.argmax(dim=1), y)
        if stage:
            self.log(f"{stage}_loss", loss, logger=True)
            self.log(f"{stage}_acc", acc, logger=True)
        return loss, acc

    def on_train_start(self):
        model_info = summary(self, input_size=(1, self.config.max_seq_len), verbose=0)
        self.log(
            "total_params",
            torch.tensor(model_info.total_params, dtype=torch.float32),
            logger=True,
        )

    def training_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "train")
        return {"loss": loss, "train_acc": acc}

    def training_epoch_end(self, outputs):
        acc = torch.stack([i["train_acc"] for i in outputs]).mean()
        loss = torch.stack([i["loss"] for i in outputs]).mean()
        self.log("avg_train_acc", acc, logger=True)
        self.log("avg_train_loss", loss, logger=True)

    def validation_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "val")
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        acc = torch.stack([i["val_acc"] for i in outputs]).mean()
        loss = torch.stack([i["val_loss"] for i in outputs]).mean()
        self.log("avg_val_acc", acc, logger=True)
        self.log("avg_val_loss", loss, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "test")
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        acc = torch.stack([i["test_acc"] for i in outputs]).mean()
        loss = torch.stack([i["test_loss"] for i in outputs]).mean()
        self.log("avg_test_acc", acc, logger=True)
        self.log("avg_test_loss", loss, logger=True)

    def configure_optimizers(self):
        opt = optim.Adam(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.wd
        )
        sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "epoch", "frequency": 1},
        }
