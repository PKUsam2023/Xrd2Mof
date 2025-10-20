import torch
from tqdm import tqdm
import numpy as np
import argparse


def lattices_to_params_shape(lattices):

    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[...,i] = torch.clamp(torch.sum(lattices[...,j,:] * lattices[...,k,:], dim = -1) /
                            (lengths[...,j] * lengths[...,k]), -1., 1.)
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack(
        [
            lengths[:, 0] * sins[:, 1],
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 0] * coses[:, 1],
        ],
        dim=1,
    )
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [
            torch.zeros(lengths.size(0), device=lengths.device),
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 2],
        ],
        dim=1,
    )

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def create_data_from_pt(pt_file_path, bbs_file_path):
    """
    Load generated MOF list and building-block list, then for each MOF:
      - extract lattice params (lengths, angles, cell)
      - assemble each BB at its fractional coordinate
      - return the list of updated Data objects
    """
    gen_data_list = torch.load(pt_file_path, map_location=torch.device('cpu'))
    bbs_list = torch.load(bbs_file_path, map_location=torch.device('cpu'))
    print(f"Load data, mof num = {len(gen_data_list)}")

    mofs_list = []
    for data in tqdm(gen_data_list, desc = "Processing Gen MOFs:"):
        mof_id = data.mof_id
        prefix = "_".join(mof_id.split("_")[:-1])

        atom_types = data.atom_types
        frac_coords = data.frac_coords

        lattice = data.lattice.unsqueeze(0)
        lengths, angles = lattices_to_params_shape(lattice)
        cell = lattice_params_to_matrix_torch(lengths, angles).squeeze(0)


        query_metal_bb = None
        query_linker_bb = None
        for item in bbs_list:
            if item.get("mof_id") == prefix:
                bbs_dict = item.get("bbs", {})
                query_metal_bb = bbs_dict.get("metal")[0]
                query_linker_bb = bbs_dict.get("linker")[0]
                break
        if query_metal_bb is None or query_linker_bb is None:
            print(f"Warning: BB data not found for {prefix}, skipping")
            continue

        # Assemble a list of BBs matching atom_types
        total_bbs = []
        for i in atom_types:
            if i == 0 :
                total_bbs.append(query_metal_bb)
            else:
                total_bbs.append(query_linker_bb)
        
        ret_bbs = []
        for idx, bb in enumerate(total_bbs):
            ret_bb = bb.clone()
            ret_bb.centroid = frac_coords[idx]
            ret_bb.frac_coords = (
                ret_bb.frac_coords
                - ret_bb.frac_coords[ret_bb.is_anchor].mean(dim=0)
                + ret_bb.centroid
            )
            ret_bb.local_vectors = (
                ret_bb.frac_coords[ret_bb.is_anchor] - ret_bb.centroid
            )
            del ret_bb.feature
            ret_bbs.append(ret_bb)
        
        
        # Attach new attributes
        data.lengths=lengths
        data.angles=angles
        data.cell = cell
        data.bbs = ret_bbs

        mofs_list.append(data)

    return mofs_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process generated Single-MOF PT file and merge with BB data"
    )
    parser.add_argument(
        '--input_pt',
        type=str,
        required=True,
        help="Path to the generated MOF .pt file"
    )
    parser.add_argument(
        '--bbs_pt',
        type=str,
        required=True,
        help="Path to the building-block .pt file"
    )
    parser.add_argument(
        '--output_pt',
        type=str,
        required=True,
        help="Path where the processed output .pt will be saved"
    )

    args = parser.parse_args()

    # Run processing
    mofs_list = create_data_from_pt(args.input_pt, args.bbs_pt)
    print("Processing complete, saving output...")
    # Save as a dict under key "mofs"
    torch.save({"mofs": mofs_list}, args.output_pt)
    print(f"Saved processed MOFs to {args.output_pt}")
