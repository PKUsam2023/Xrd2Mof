import time
import argparse
import torch
import copy
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch
from torch_geometric.data import Data

from eval_utils import load_model, lattices_to_params_shape, recommand_step_lr

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group


def diffusion(loader, model, num_evals, step_lr = 1e-5):
    all_data = []

    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()

        for eval_idx in range(num_evals):

            print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            outputs, traj = model.sample(batch, step_lr = step_lr)
            mof_id = outputs['mof_id']       # shape: [num_mof]
            num_atoms_tensor = outputs['num_atoms'].detach().cpu()       # shape: [num_mof]
            frac_coords_tensor = outputs['frac_coords'].detach().cpu()   # shape: [total_atoms, 3]
            atom_types_tensor = outputs['atom_types'].detach().cpu()     # shape: [total_atoms]
            lattices_tensor = outputs['lattices'].detach().cpu()           # shape: [num_mof, 3, 3]
            
            # Compute cumulative sum of num_atoms_tensor to determine slice indices
            # for each block in frac_coords and atom_types
            offsets = torch.cumsum(num_atoms_tensor, dim=0)
            num_mof = num_atoms_tensor.shape[0]
            
            # Iterate over each block
            for mof_index in range(num_mof):
                N = int(num_atoms_tensor[mof_index].item())
                start = 0 if mof_index == 0 else int(offsets[mof_index - 1].item())
                end = int(offsets[mof_index].item())
                
                # Extract the current block's fractional coordinates and atom types
                block_frac_coords = frac_coords_tensor[start:end, :]   # shape: [N, 3]
                block_atom_types = atom_types_tensor[start:end]          # shape: [N]
                block_lattice = lattices_tensor[mof_index, :, :]         # shape: [3, 3]
                # block_lengths = lengths_tensor[mof_index, :]
                # block_angles = angles_tensor[mof_index, :]

                # Construct a torch_geometric.data.Data object.
                # Optionally, lengths, angles, cell parameters could be computed here.
                data_obj = Data(
                    mof_id = f"{mof_id[mof_index]}_{eval_idx}",
                    edge_index = [],
                    frac_coords = block_frac_coords,
                    atom_types = block_atom_types,
                    num_atoms = torch.tensor(N),
                    num_components = torch.tensor(N),
                    lattice = block_lattice,
                    to_jimages = []
                )
                all_data.append(data_obj)

    return all_data


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=True)

    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate the diffusion model.')

    # You can modify it when train your own model
    dataset_type = 'CGmof_50'
    step_lr = recommand_step_lr['csp' if args.num_evals == 1 else 'csp_multi'][dataset_type]

    start_time = time.time()
    all_data = diffusion(test_loader, model, args.num_evals, step_lr)

    if args.label == '':
        diff_out_name = f'gen_csp_{args.num_evals}.pt'
    else:
        diff_out_name = f'gen_csp_{args.label}_num_eval_{args.num_evals}.pt'

    torch.save(all_data, Path(args.out_path) / diff_out_name)

    print(time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", required=True,
        help="Path to the trained model checkpoint (.ckpt)."
    )
    parser.add_argument(
        "--num_evals", default=5, type=int,
        help="Number of evaluation samples per input (default: 5)."
    )
    parser.add_argument(
        "--label", default="", type=str,
        help="Optional run label/tag for logging or output naming."
    )
    parser.add_argument('--out_path', required=True,
        help='Path to save results (e.g., /path/to/output.pt)'
    )
    args = parser.parse_args()

    main(args)
