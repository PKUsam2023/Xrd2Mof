import torch
import numpy as np
import torch.nn.functional as F
from pymatgen.core.periodic_table import Element
import pandas as pd
import argparse
import re
import os


def get_element_features(element_symbol, ratio):
    """
    Construct feature vector for a given element symbol and its atomic ratio.
    """
    element = Element(element_symbol)
    atomic_ratio = [ratio]
    atomic_number = [(element.Z - 3.0) / 41.0]
    atomic_mass = [(element.atomic_mass - 6.941) / 209.0]
    atomic_electronegativity = [element.X if element.X is not None else 0.0]
    atomic_period = [element.row / 6.0]
    atomic_group = [element.group / 14.0]
    special_feature = [
        int(element.is_metal),
        int(element.is_transition_metal),
        int(element.is_rare_earth_metal),
        int(element.is_metalloid)
    ]
    oxidation_states = list(element.oxidation_states)
    return np.array(
        atomic_ratio + atomic_number + atomic_mass + atomic_electronegativity +
        atomic_period + atomic_group + special_feature + oxidation_states,
        dtype=np.float32
    )


def elements_to_tensor(elements_dict):
    """
    Convert a dict of element_symbol -> ratio into a fixed-size feature tensor.
    Only metals or specified semimetals are included.
    Returns padded tensor (65 dims) and list of included symbols.
    """
    features = []
    symbols = []
    for sym, ratio in elements_dict.items():
        element = Element(sym)
        if element.is_metal or sym in ['Sb', 'Po']:
            symbols.append(sym)
            feat = get_element_features(sym, ratio)
            features.append(feat)
    if not features:
        raise ValueError("No valid metal elements found in row.")
    concatenated = np.concatenate(features)
    tensor = torch.tensor(concatenated, dtype=torch.float32)
    target_len = 65
    if tensor.numel() < target_len:
        pad_amt = target_len - tensor.numel()
        tensor = F.pad(tensor, (0, pad_amt), 'constant', 0)
    else:
        tensor = tensor[:target_len]
    return tensor, symbols


def main(input_csv, output_dir):
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        materials_name = str(row['Materials_name']).strip()
        raw = str(row['Metal_nodes']).strip()
        elem = [p for p in re.split(r'[;,]', raw) if p.strip()][0].strip()
        # split on comma or semicolon
        elements_dict = {elem: 1.0}
        tensor, syms = elements_to_tensor(elements_dict)
        metals_str = f"['{syms[0]}']"
        tensor_str = '[' + ', '.join(f'{str(x)}' for x in tensor.tolist()) + ']'
        results.append({
            'Materials_name': materials_name,
            'Metals': metals_str,
            'Metal_tensor': tensor_str
        })
    
    out_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "metal.csv")
    header = not os.path.isfile(output_csv)
    if os.path.exists(output_csv):
        os.remove(output_csv)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {len(results)} entries to {output_csv}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate metal feature tensors from a CSV of materials.'
    )
    parser.add_argument(
        '--input_csv', type=str, required=True,
        help='Path to input CSV with columns Materials_name and Metal_nodes.'
    )
    parser.add_argument(
        '--output_csv', type=str, required=True,
        help='Path for the output CSV file.'
    )
    args = parser.parse_args()
    main(args.input_csv, args.output_csv)
