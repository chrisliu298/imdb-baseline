import argparse
import logging
import os
import warnings

import wandb
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from data import IMDBDataModule
from models import LSTM, TransformerEncoder

MODELS = {"transformer": TransformerEncoder, "lstm": LSTM}
# Due to the issue described in
# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
# we need to disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--label_noise", type=float, default=0.0)
    # model
    parser.add_argument(
        "--model", type=str, default="transformer", choices=["transformer", "lstm"]
    )
    parser.add_argument("--output_size", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)  # transformer only
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bidirectional", action="store_true")  # lstm only
    # training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=200)
    # experiment
    parser.add_argument("--project_id", type=str, default="demo-imdb-transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    # convert arg namespace to easydict for easier access
    config = EasyDict(vars(parser.parse_args()))
    # seed for reproducibility
    seed_everything(config.seed)
    # mute everything
    if not config.verbose:
        os.environ["WANDB_SILENT"] = "True"
        warnings.filterwarnings("ignore")
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    # setup data module, model, and trainer
    datamodule = IMDBDataModule(config)
    # the vocab size is not deterministic in advance, so we need to assign it here
    config.vocab_size = datamodule.vocab_size
    model = MODELS[config.model](config)
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            filename="{epoch}_{avg_val_acc}",
            monitor="avg_val_acc",
            save_top_k=5,
            mode="max",
        )
    )
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    if not config.verbose:
        callbacks.append(TQDMProgressBar(refresh_rate=0))
    trainer = Trainer(
        gpus=-1,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=WandbLogger(
            offline=not config.wandb,
            project=config.project_id,
            entity="chrisliu298",
            config=config,
        ),
        profiler="simple",
    )
    # train
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, verbose=config.verbose)
    wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
