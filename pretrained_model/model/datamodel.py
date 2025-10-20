import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from model.dataset import CLIPDataset

def collate_fn(batch):
    """
    Custom collate_fn to pack a mini-batch into tensors or a PyG Batch.
    Tensors are used for standard inputs; Batch is used for graph data.

    Args:
        batch: A list where each element is the tuple returned by CLIPDataset.

    Returns:
        A tuple containing:
        - cif_lattice_batch: torch.Tensor
        - cif_atom_positions_batch: torch.Tensor
        - xrd_batch: torch.Tensor
        - metal_batch: torch.Tensor
        - linker_batch: torch_geometric.data.Batch
        - tokenizer_batch: list
    """
    cif_lattice_batch = torch.stack([item[0] for item in batch])
    cif_atom_positions_batch = torch.stack([item[1] for item in batch])
    xrd_batch = torch.stack([item[2] for item in batch])
    metal_batch = torch.stack([item[3] for item in batch])
    linker_batch = Batch.from_data_list([item[4] for item in batch])  # Pack into a PyG Batch
    tokenizer_batch = [item[5] for item in batch]  # Keep as a list

    return cif_lattice_batch,cif_atom_positions_batch, xrd_batch, metal_batch, linker_batch, tokenizer_batch


class CLIPDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        """
        Args:
            batch_size: Mini-batch size.
            num_workers: Number of worker processes for the DataLoader.
            data_path: Root directory of the data containing the .pt files.
        """
        super().__init__()
        self.cfg = cfg

        self.batch_size = self.cfg.batch_size
        self.num_workers = self.cfg.num_workers
        self.data_path = self.cfg.data_path

    def setup(self, stage=None):
        """
        Load datasets for different stages.
        If stage is 'fit' or None, load training and validation sets.
        If stage is 'test' or None, load the test set.
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = CLIPDataset(mode="train", cfg = self.cfg)
            self.valid_dataset = CLIPDataset(mode="valid", cfg = self.cfg)
        if stage == 'test' or stage is None:
            self.test_dataset = CLIPDataset(mode="test", cfg = self.cfg)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory = True,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory = True,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory = True,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )
    