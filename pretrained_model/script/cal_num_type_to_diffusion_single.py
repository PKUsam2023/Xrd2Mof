import math
import numpy as np
import torch
import argparse


def get_num_atoms_and_types(data_dict):
    """
    Compute possible atom counts and corresponding type arrays.

    The function uses metallic and linker valences from data_dict["bbs"].
    It calculates the least common multiple (LCM) of metal_valence and linker_valence,
    then derives ratios metal_num and linker_num.
    For factors 1 through 4, it returns tuples of (total_atoms, atom_types_array)
    until total_atoms exceeds 50.
    """
    # Extract valences for metal and linker
    metal_valence = data_dict["bbs"]["metal_valence"][0]
    linker_valence = data_dict["bbs"]["linker_valence"][0]

    # Compute LCM via gcd
    multiple = abs(metal_valence * linker_valence) // math.gcd(metal_valence, linker_valence)

    # Determine counts
    metal_num = multiple // metal_valence
    linker_num = multiple // linker_valence

    # Total atoms per factor
    result = []
    for factor in range(1, 5):
        current_num_atoms = (metal_num + linker_num) * factor
        # Build atom_types: 0 for metal, 1 for linker
        current_atom_types = np.array([0] * metal_num * factor + [1] * linker_num * factor)
        if current_num_atoms > 50:
            break
        result.append((current_num_atoms, current_atom_types))

    return result


def create_data(data_dict):
    """
    Expand a single MOF entry into multiple configurations.

    For each (num_atoms, atom_types) pair, create a new dict with:
      - mof_id suffixed by index
      - feature_embedding
      - atom_types array
      - num_atoms and num_nodes
      - original bbs attribute
    Returns a list of such dicts.
    """
    data_list = []
    mof_id = data_dict['mof_id']
    feature_embedding = data_dict['feature_embedding']
    bbs = data_dict['bbs']
    configs = get_num_atoms_and_types(data_dict)
    for idx, (num_atoms, atom_types) in enumerate(configs):
        entry = {
            'mof_id': f"{mof_id}_{idx}",
            'feature_embedding': feature_embedding,
            'atom_types': atom_types,
            'num_atoms': num_atoms,
            'num_nodes': num_atoms,
            'bbs': bbs
        }
        data_list.append(entry)
    return data_list


def process_pt_file(input_pt):
    """
    Load a list of MOF dicts from input_pt, process each to expand atom configs,
    and flatten into a single list of entries.
    """
    processed = []
    data_list = torch.load(input_pt, map_location='cpu')
    for data_dict in data_list:
        processed.extend(create_data(data_dict))
    return processed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Expand MOF entries by atom count ratios.')
    parser.add_argument('--input_pt', type=str, required=True,
                        help='Path to input .pt file containing processed Single-MOF data list.')
    parser.add_argument('--output_pt', type=str, required=True,
                        help='Path to output .pt file for expanded data.')
    args = parser.parse_args()

    # Process and save
    expanded = process_pt_file(args.input_pt)
    torch.save(expanded, args.output_pt)
    print(f"Processed data saved to {args.output_pt}")
