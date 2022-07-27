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
from models import TransformerEncoderModel

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
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--mask_type",
        type=str,
        default="both",
        choices=["both", "mask", "padding_mask", "none"],
    )
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
    # assign additional args
    config.dataset = config.project_id.split("-")[1]
    config.model = config.project_id.split("-")[2]
    config.output_size = 2
    # setup data module, model, and trainer
    datamodule = IMDBDataModule(config)
    datamodule.prepare_data()
    datamodule.setup()
    # the vocab size is not deterministic in advance, so we need to assign it here
    config.vocab_size = datamodule.vocab_size
    model = TransformerEncoderModel(config)
    callbacks = [
        ModelCheckpoint(
            filename="{epoch}_{avg_val_acc}",
            monitor="avg_val_acc",
            save_top_k=5,
            mode="max",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if not config.verbose:
        callbacks.append(TQDMProgressBar(refresh_rate=0))
    logger = WandbLogger(
        offline=not config.wandb,
        project=config.project_id,
        entity="chrisliu298",
        config=config,
    )
    trainer = Trainer(
        gpus=-1,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=logger,
    )
    wandb.log(
        {
            "train_size": len(datamodule.train_dataset),
            "val_size": len(datamodule.val_dataset),
            "test_size": len(datamodule.test_dataset),
        }
    )
    # train
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, verbose=config.verbose)
    wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
