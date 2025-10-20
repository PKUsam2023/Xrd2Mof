import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pytorch_lightning as pl
import torch
import time
import numpy as np
import pandas as pd
import hydra
import re
from tqdm import tqdm
from copy import deepcopy
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning import seed_everything
from model.model import CLIPModule
from model.datamodel import CLIPDataModule


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    start = time.time()
    seed = 512
    seed_everything(seed)

    # Create model
    model = CLIPModule(cfg).to(cfg.device)
    
    # Load checkpoint (if exist)
    ckpts = list(Path(cfg.save_model_path).glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([
            int(ckpt.parts[-1].split('.')[0].split('_')[-1].split('=')[1])
            for ckpt in ckpts
        ])
        # select checkpoint
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        print("Choose the name of .ckpt file :", ckpt)
    else:
        ckpt = None

    # load data
    model = CLIPModule.load_from_checkpoint(ckpt)
    data = CLIPDataModule(cfg)
    data.setup(stage="test")

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        gpus=1 if os.environ.get("CUDA_VISIBLE_DEVICES", None) is not None and torch.cuda.is_available() else 0,
        accelerator='gpu',
        # distributed_backend='ddp',
        deterministic=True,
        progress_bar_refresh_rate=20
    )

    # get results
    trainer.test(model, data)
    test_results = model.all_test_results

    test_results_cpu = []
    for token, cif_emb, feature_emb in test_results:
        feature_emb = feature_emb.cpu().numpy()
        test_results_cpu.append({
        "mof_id": token,
        "feature_embedding": feature_emb
    })
    
    torch.save(test_results_cpu, cfg.output_feature_embedding_data)
    
    end = time.time()
    print(f"Test completed in {end - start:.2f} seconds\n")


if __name__ == '__main__':
    main()

