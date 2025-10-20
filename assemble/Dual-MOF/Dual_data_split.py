import os
import math
import torch
import argparse
from tqdm import tqdm


def get_bbs_info(prefix, bbs_list):
    """
    Look up building-block info by MOF prefix in bbs_list.
    Each item in bbs_list is a dict containing keys:
      - "mof_id"
      - "bbs" (with subkeys "metal", "linker", "metal_valence", "linker_valence", "linker_proportion")
    Returns:
      metal_list, linker_list, metal_valence, linker_valence, linker_proportion
    or (None, None, None, None, None) if not found.
    """
    for item in bbs_list:
        if item.get("mof_id") == prefix:
            bbs_dict = item.get("bbs", {})
            return (
                bbs_dict.get("metal"),
                bbs_dict.get("linker"),
                bbs_dict.get("metal_valence"),
                bbs_dict.get("linker_valence"),
                bbs_dict.get("linker_proportion"),
            )
    return None, None, None, None, None


def split_and_save_pt_file(input_file, bbs_file_path, output_dir):
    """
    Split a .pt file (containing a list of dicts) into individual .pt files.
    Each entry is saved under its 'mof_id' name, distributed evenly into 10 subfolders.

    Args:
        input_file: path to the input .pt file (list of data dicts)
        bbs_file_path: path to the .pt file containing bbs_list for lookup
        output_dir: directory where per-item .pt files will be saved
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the main data list and the bbs lookup list
    data_list = torch.load(input_file, map_location="cpu")
    bbs_list = torch.load(bbs_file_path, map_location="cpu")

    total_items = len(data_list)
    # Determine approximate group size to split into 10 parts
    group_size = math.ceil(total_items / 10)

    # Create 10 numbered subfolders
    subfolder_paths = []
    for i in range(10):
        sub = os.path.join(output_dir, f"part_{i}")
        os.makedirs(sub, exist_ok=True)
        subfolder_paths.append(sub)

    # Iterate and save each item separately
    for idx, item in enumerate(tqdm(data_list, desc="Splitting items")):
        # Extract MOF ID
        try:
            mof_id = item.mof_id
        except AttributeError:
            print("Warning: item has no 'mof_id' attribute, skipping.")
            continue

        # Derive prefix by dropping final underscore-index
        prefix = "_".join(mof_id.split("_")[:-1])

        # Attach bbs info fields to the item
        (item.metal_list,
         item.linker_list,
         item.metal_valence,
         item.linker_valence,
         item.linker_proportion) = get_bbs_info(prefix, bbs_list)

        # Determine which subfolder this item belongs to
        group_idx = idx // group_size
        if group_idx >= 10:
            group_idx = 9
        target_folder = subfolder_paths[group_idx]

        # Construct and save the output .pt file
        out_path = os.path.join(target_folder, f"{mof_id}.pt")
        torch.save(item, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset .pt file into individual MOF files with bbs info attached."
    )
    parser.add_argument(
        "--input_pt",
        type=str,
        required=True,
        help="Path to the input .pt file (list of data dicts)."
    )
    parser.add_argument(
        "--bbs_pt",
        type=str,
        required=True,
        help="Path to the building-blocks .pt file for lookup."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where individual .pt files will be stored."
    )
    args = parser.parse_args()

    split_and_save_pt_file(args.input_pt, args.bbs_pt, args.output_dir)
    print("Finished splitting and saving individual .pt files.")
