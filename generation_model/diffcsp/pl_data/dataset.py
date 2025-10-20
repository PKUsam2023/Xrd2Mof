import hydra
import omegaconf
import torch
import pandas as pd
import numpy as np
from omegaconf import ValueNode
from torch.utils.data import Dataset
from torch_geometric.data import Data

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    preprocess_tensors, add_scaled_lattice_prop)


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, prop: ValueNode, graph_method: ValueNode, 
                 lattice_scale_method: ValueNode, save_path: ValueNode, tolerance: ValueNode,
                 use_space_group: ValueNode, use_pos_index: ValueNode,
                 **kwargs):
        super().__init__()
        self.save_path = save_path
        self.name = name
        self.prop = prop
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance

        self.preprocess(save_path)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def preprocess(self, save_path):
        self.cached_data = torch.load(save_path)

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        # scaler is set in DataModule set stage
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']
        feature_embedding = data_dict['feature_embedding']
        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            feature_embedding = torch.Tensor(feature_embedding),
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.save_path=})"


class TestMOFCrystDataset(Dataset):
    def __init__(self, name: ValueNode, lattice_scale_method: ValueNode, save_path: ValueNode,
                 **kwargs):
        super().__init__()
        self.save_path = save_path
        self.name = name
        self.lattice_scale_method = lattice_scale_method
        self.preprocess(save_path)
        # add_scaled_lattice_prop(self.cached_data, lattice_scale_method)

    def preprocess(self, save_path):
        self.cached_data = torch.load(save_path)

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        # scaler is set in DataModule set stage
        mof_id = data_dict['mof_id']
        num_atoms = data_dict['num_atoms']
        atom_types = data_dict['atom_types']
        feature_embedding = data_dict['feature_embedding']
        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            mof_id = mof_id,
            feature_embedding = torch.Tensor(feature_embedding),
            atom_types=torch.LongTensor(atom_types),
            num_atoms=num_atoms,
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )

        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.save_path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from diffcsp.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
