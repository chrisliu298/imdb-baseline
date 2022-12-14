import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from torchmetrics.functional import accuracy


class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_train_start(self):
        # log the number of parameters
        model_info = summary(self, input_size=(1, self.config.max_seq_len), verbose=0)
        self.log(
            "total_params",
            torch.tensor(model_info.total_params, dtype=torch.float32),
            logger=True,
        )
        # log data split sizes
        datamodule = self.trainer.datamodule
        self.log("train_size", float(len(datamodule.train_dataset)), logger=True)
        self.log("val_size", float(len(datamodule.val_dataset)), logger=True)
        self.log("test_size", float(len(datamodule.test_dataset)), logger=True)

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
        # sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 80], gamma=0.2)
        # return {
        #     "optimizer": opt,
        #     "lr_scheduler": {"scheduler": sch, "interval": "epoch", "frequency": 1},
        # }
        return opt


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.save_hyperparameters()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embedding = PositionalEncoding(
            config.embedding_dim, dropout=config.dropout, max_len=config.max_seq_len
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size,
            dropout=config.dropout,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=config.num_layers
        )
        self.fc = nn.Linear(config.embedding_dim, config.output_size)

    def forward(self, x, padding_mask=None):
        # generate src_mask
        mask = nn.Transformer.generate_square_subsequent_mask(
            self.config.max_seq_len
        ).to(self.device)
        out = x.long()
        out = self.embedding(out) * math.sqrt(self.config.embedding_dim)
        out = out.permute(1, 0, 2)
        out = self.pos_embedding(out)
        out = self.encoder(out, mask=mask, src_key_padding_mask=padding_mask)
        # use the last hidden state
        out = out[0, :]
        out = self.fc(out)
        return out

    def evaluate(self, batch, stage=None):
        x, padding_mask, y = batch
        output = self(x, padding_mask)
        loss = F.cross_entropy(output, y)
        acc = accuracy(output.argmax(dim=1), y)
        if stage:
            self.log(f"{stage}_loss", loss, logger=True)
            self.log(f"{stage}_acc", acc, logger=True)
        return loss, acc


class LSTM(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.save_hyperparameters()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout,
        )
        self.fc = nn.Linear(
            config.hidden_size * (2 if self.config.bidirectional else 1),
            config.output_size,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        out = x.long()
        out = self.embedding(out)
        out = out.permute(1, 0, 2)
        out, (h, _) = self.lstm(out, self.init_hidden(batch_size))
        h = torch.cat([h[-2], h[-1]], dim=-1) if self.config.bidirectional else h[-1]
        out = self.fc(h)
        return out

    def init_hidden(self, batch_size):
        return (
            torch.zeros(
                self.config.num_layers * (2 if self.config.bidirectional else 1),
                batch_size,
                self.config.hidden_size,
            ).to(self.device),
            torch.zeros(
                self.config.num_layers * (2 if self.config.bidirectional else 1),
                batch_size,
                self.config.hidden_size,
            ).to(self.device),
        )

    def evaluate(self, batch, stage=None):
        x, _, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        acc = accuracy(output.argmax(dim=1), y)
        if stage:
            self.log(f"{stage}_loss", loss, logger=True)
            self.log(f"{stage}_acc", acc, logger=True)
        return loss, acc
