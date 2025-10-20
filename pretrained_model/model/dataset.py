import os
import torch

from os.path import join
from torch.utils.data import Dataset

# cif_lattice[64, 3, 3], cif_atom_positions[64, 50, 3], xrd[64, 5250], metal[64, 65], linker[64, Data(x=[a, 37], edge_index=[2, b], edge_attr=[b, 10])]


class CLIPDataset(Dataset):
    def __init__(self, mode, cfg):
        """
        Args:
            mode (str): Dataset split, one of {'train', 'test', 'valid'}.
        """
        super().__init__()
        self.cfg = cfg
        self.data_path = self._get_data_path(mode)
        self.data = torch.load(self.data_path)


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.data)


    def __getitem__(self, index):
        """
        Load a sample by index and return CIF/XRD data along with an identifier.

        Args:
            index (int): Sample index.

        Returns:
            cif_lattice (torch.Tensor): Normalized CIF lattice tensor.
            cif_atom_positions (torch.Tensor): Normalized CIF atom-position tensor.
            xrd (torch.Tensor): Normalized XRD tensor.
            metal (torch.Tensor): Normalized metal-feature tensor.
            linker (torch_geometric.data.Data or compatible): Linker graph data.
            tokenizer (str): Filename identifier (prefix).
        """

        prefix = self.data[index]["prefix"]
        cif_lattice = self.data[index]["cif_lattice"]
        cif_atom_positions = self.data[index]["cif_atom_positions"]
        xrd = self.data[index]["xrd"]
        metal = self.data[index]["metal"]
        linker = self.data[index]["linker"]

        return cif_lattice, cif_atom_positions, xrd, metal, linker, prefix


    def _get_data_path(self, mode):
        if mode in ['train', 'valid']:
            return join(self.cfg.data_path, mode + ".pt")
        elif mode == 'test':
            return self.cfg.feature_extract_data
        else:
            raise ValueError(f"Invalid Mode: {mode}")
