import os
import math
import random
import argparse
import itertools
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.ewald import EwaldSummation

# will be set by argparse
OUTPUT_DIR = None


def check_lattice(lattice, max_length=120):
    """
    Return True if all lattice parameters a, b, c are below max_length.
    """
    mat = lattice.tolist()
    lattice_obj = Lattice(mat)
    a, b, c = lattice_obj.a, lattice_obj.b, lattice_obj.c
    return (a < max_length) and (b < max_length) and (c < max_length)


def lattices_to_params_shape(lattices):
    """
    From a batch of 3×3 lattice tensors, compute lengths and angles (in degrees).
    lengths: tensor of shape (N,3)
    angles:  tensor of shape (N,3)
    """
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


def process_mof(file_path):
    """
    Process one MOF .pt file:
      1. Load the Data object.
      2. Skip if lattice a,b,c exceed the threshold.
      3. Enumerate possible linker assignments to minimize Ewald energy.
      4. Update atom_types and bbs, then save back to OUTPUT_DIR.
    """
    mof = torch.load(file_path, map_location=torch.device('cpu'))

    # 1. Check lattice size
    lattice = mof.lattice
    if not check_lattice(lattice):
        return None
    
    # 2. Compute lengths, angles, and cell matrix
    lengths, angles = lattices_to_params_shape(lattice.unsqueeze(0))
    cell = lattice_params_to_matrix_torch(lengths, angles).squeeze(0)

    # 3. Gather BB and valence data from the mof object
    mof_id = mof.mof_id
    query_metal_bb = mof.query_metal_bb[0]
    metal_valence = mof.metal_valence[0]
    query_linker_bb1 = mof.query_linker_bb[0]
    query_linker_bb2 = mof.query_linker_bb[1]
    linker_valence1 = mof.linker_valence[0]
    linker_valence2 = mof.linker_valence[1]
    prop = mof.proportion
    frac_coords = mof.frac_coords

    # 4. Determine number of metals and linkers
    N = mof.num_atoms
    try:
        num_metals = (mof.atom_types == 0).sum().item()
    except Exception:
        num_metals = mof.atom_types.count(0)

    preliminary_atom_types = mof.atom_types[:]
    linker_indices = [i for i, v in enumerate(preliminary_atom_types) if v == 1]
    total_linkers = len(linker_indices)
    
    count_linker1 = int(round((prop[0] / (prop[0] + prop[1])) * total_linkers))
    count_linker2 = total_linkers - count_linker1

    best_energy = None
    best_linker_assignment = None

    # 5. Generate up to 10 combinations to avoid explosion
    n_linkers = len(linker_indices)
    total_combinations = math.comb(n_linkers, count_linker1)
    if total_combinations > 10:
        selected_assignments = set()
        while len(selected_assignments) < 10:
            assignment = tuple(sorted(random.sample(linker_indices, count_linker1)))
            selected_assignments.add(assignment)
        selected_assignments = list(selected_assignments)
        print(f"Combinations is {total_combinations}, random choose 10 combinations")
    else:
        selected_assignments = list(itertools.combinations(linker_indices, count_linker1))
        print(f"Combinations is {total_combinations}, return all combinations")

    for assignment in selected_assignments:
        temp_atom_types = []
        for i in range(N):
            if preliminary_atom_types[i] == 0:
                temp_atom_types.append(0)
            else:
                temp_atom_types.append(2)
        # 1 represent linker1
        for idx in assignment:
            temp_atom_types[idx] = 1
        
        species = []
        for v in temp_atom_types:
            if v == 0:
                species.append("Li")
            elif v == 1:
                species.append("F")
            else:
                species.append("Cl")
        structure = Structure(lattice, species, frac_coords)

        # map atom_types to species
        structure.add_oxidation_state_by_element({"Li": metal_valence, "F": -linker_valence1, "Cl": -linker_valence2})
        
        # 6. Find best assignment by Ewald energy
        try:
            ewald = EwaldSummation(structure)
            energy = ewald.total_energy
        except Exception as e:
            print(f"Error computing Ewald for MOF {mof_id} linker assignment {assignment}: {e}")
            continue

        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_linker_assignment = assignment
    
    if best_linker_assignment is None:
        print(f"MOF {mof_id}: No valid linker assignment found.")
        return None

    # 7. Apply best assignment to atom_types and bbs
    final_atom_types = []
    for i in range(N):
        if preliminary_atom_types[i] == 0:
            final_atom_types.append(0)
        else:
            if i in best_linker_assignment:
                final_atom_types.append(1)
            else:
                final_atom_types.append(2)
    
    if isinstance(mof.atom_types, torch.Tensor):
        mof.atom_types = torch.tensor(final_atom_types)
    else:
        mof.atom_types = final_atom_types

    total_bbs = []
    for i in range(N):
        if final_atom_types[i] == 0:
            total_bbs.append(query_metal_bb)
        elif final_atom_types[i] == 1:
            total_bbs.append(query_linker_bb1)
        else:
            total_bbs.append(query_linker_bb2)
    
    # rebuild bbs list
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

    mof.lengths=lengths
    mof.angles=angles
    mof.cell = cell
    mof.bbs = ret_bbs

    # Save updated MOF back to OUTPUT_DIR
    basename = os.path.basename(file_path)
    out_path = os.path.join(OUTPUT_DIR, basename)
    torch.save(mof, out_path)
    print(f"MOF {mof_id}: Best linker assignment = {best_linker_assignment}, Energy = {best_energy:.3f} eV")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parallel Ewald re-assignment for MOF linkers"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Directory containing individual MOF .pt files to process"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Directory where updated .pt files will be saved"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=min(20, os.cpu_count()),
        help="Number of parallel worker processes"
    )
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Gather all file paths
    files = [os.path.join(INPUT_DIR, f)
             for f in os.listdir(INPUT_DIR) if f.endswith('.pt')]

    # Process in parallel
    pool = Pool(args.workers)
    pool.map(process_mof, files)
    pool.close()
    pool.join()

    print(f"All updated files have been saved to {OUTPUT_DIR}")
