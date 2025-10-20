import math
import numpy as np
import torch
from tqdm import tqdm
import argparse


def get_num_atoms_and_types(data_dict):
    """
    Compute the number of metal and linker atoms and their types based on valences and proportions.
    Returns a list of (num_atoms, atom_types_array) tuples for scaling factors 1 to 4,
    stopping if total atoms exceed 50.
    """
    # extract valences and proportions
    metal_valence    = data_dict["bbs"]["metal_valence"][0]
    linker_1_valence = data_dict["bbs"]["linker_valence"][0]
    linker_2_valence = data_dict["bbs"]["linker_valence"][1]
    linker_1_prop    = data_dict["bbs"]["linker_proportion"][0]
    linker_2_prop    = data_dict["bbs"]["linker_proportion"][1]

    # total linker valence weighted by proportions
    linker_valence = (
        linker_1_valence * linker_1_prop +
        linker_2_valence * linker_2_prop
    )

    # compute least common multiple of metal and linker valence
    gcd = math.gcd(metal_valence, linker_valence)
    multiple = abs(metal_valence * linker_valence) // gcd

    # number of metal and linker units for one repeat
    metal_num   = multiple // metal_valence
    linker_num  = multiple // linker_valence

    # split linker into two types according to proportions
    linker_1_num = int(linker_num * linker_1_prop)
    linker_2_num = int(linker_num * linker_2_prop)
    total_linker_num = linker_1_num + linker_2_num

    # total atoms per repeat
    base_num_atoms = metal_num + total_linker_num

    results = []
    for factor in range(1, 5):
        current_num_atoms = base_num_atoms * factor
        if current_num_atoms > 50:
            break
        # 0 for metal, 1 for linker
        atom_types = np.array(
            [0] * (metal_num * factor) +
            [1] * (total_linker_num * factor)
        )
        results.append((current_num_atoms, atom_types))

    return results


def create_data(data_dict):
    """
    Expand each MOF data_dict into multiple entries with different atom counts.
    """
    data_list = []
    mof_id = data_dict['mof_id']
    feature_embedding = data_dict['feature_embedding']
    bbs = data_dict['bbs']

    for i, (num_atoms, atom_types) in enumerate(
            get_num_atoms_and_types(data_dict)
    ):
        entry = {
            "mof_id": f"{mof_id}_{i}",
            "feature_embedding": feature_embedding,
            "atom_types": atom_types,
            "num_atoms": num_atoms,
            "num_nodes": num_atoms,
            "bbs": bbs  # used for batching in PyG
        }
        data_list.append(entry)

    return data_list


def process_pt_file(input_pt):
    """
    Load the input .pt file, process each entry to expand by atom counts,
    and return the combined list of processed entries.
    """
    processed = []
    data_list = torch.load(input_pt, map_location='cpu')
    for data_dict in tqdm(data_list, desc="Processing"):
        processed.extend(create_data(data_dict))
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a MOF .pt file to expand entries by atom count."
    )
    parser.add_argument(
        '--input_pt',
        type=str,
        required=True,
        help="Path to the input .pt file containing raw Dual-MOF data"
    )
    parser.add_argument(
        '--output_pt',
        type=str,
        required=True,
        help="Path to save the processed .pt file"
    )
    args = parser.parse_args()

    # process and save
    processed_data = process_pt_file(args.input_pt)
    torch.save(processed_data, args.output_pt)
    print(f"Processed data saved to {args.output_pt}")
