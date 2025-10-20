import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import time
import numpy as np
import pytorch_lightning as pl
import hydra

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List
from pytorch_lightning import seed_everything, Callback
from model.model import CLIPModule
from model.datamodel import CLIPDataModule


def build_callbacks(cfg):
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping,
            mode="min"
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=str(cfg.save_model_path),
            filename="best_model_{epoch:03d}",
            save_top_k=1,
            mode="min"
        )
    ]
    return callbacks


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    start = time.time()
    seed = 512
    seed_everything(seed)
    callbacks: List[Callback] = build_callbacks(cfg)

    if cfg.load_pre_train == True:
        # Load checkpoint (if exist)
        ckpts = list(Path(cfg.save_model_path).glob('*.ckpt'))
        if len(ckpts) > 0:
            ckpt_epochs = np.array([
                int(ckpt.parts[-1].split('.')[0].split('_')[-1].split('=')[1])
                for ckpt in ckpts
            ])
            # Sort by epoch number and select the latest (maximum) checkpoint
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        else:
            ckpt = None
    else:
        ckpt = None

    model = CLIPModule(cfg)
    data = CLIPDataModule(cfg)
    # Configure the Trainer and add early-stopping and model-checkpoint callbacks (adjust as needed)
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        gpus=4 if os.environ.get("CUDA_VISIBLE_DEVICES", None) is not None and torch.cuda.is_available() else 0,
        accelerator='gpu',
        distributed_backend='ddp',
        deterministic=True,
        progress_bar_refresh_rate=20,
        callbacks=callbacks,
        # precision = 16,
        resume_from_checkpoint=ckpt

    )
    trainer.fit(model, data)

    end = time.time()
    with open(cfg.train_record_path, 'a+') as f:
        f.write(f"Training completed in {end - start:.2f} seconds\n")


if __name__ == '__main__':
    main()

