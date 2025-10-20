import os
import torch
import argparse


def load_data_from_folder(input_folder, output_file):
    """
    Iterate over all .pt files in the specified folder,
    each file containing a single Data object,
    collect these objects into a list, and save to output_file.

    Args:
        input_folder (str): Path to the folder containing multiple .pt files.
        output_file (str): Path where the combined .pt file will be saved.
    """
    data_list = []

    # List all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.pt'):
            file_path = os.path.join(input_folder, filename)
            try:
                # Load the .pt file (assumed to contain one Data object)
                data_obj = torch.load(file_path, map_location=torch.device('cpu'))
                data_list.append(data_obj)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    # Wrap the collected list under key "mofs" and save
    updated_data = {"mofs": data_list}
    torch.save(updated_data, output_file)
    print(f"Loaded {len(data_list)} Data objects, saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine individual .pt files from a folder into a single .pt file."
    )
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='Path to the directory containing .pt files.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Destination path for the combined .pt file.'
    )
    args = parser.parse_args()

    load_data_from_folder(args.input_folder, args.output_file)
