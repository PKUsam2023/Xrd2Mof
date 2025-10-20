import torch
import re
import ast
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from openbabel import pybel


def create_PCA_KDtree(file_path):
    """
    Load building-block vector space from .pt and build:
     - PCA reducer to 32 dimensions
     - KDTree for nearest-neighbor queries
    """
    bbs_list = torch.load(file_path, map_location='cpu')
    vectors = np.stack([bb.feature for bb in bbs_list])
    pca = PCA(n_components=32)
    lowdim = pca.fit_transform(vectors)
    tree = KDTree(lowdim)
    return tree, pca


def to_tensor(bits):
    """
    Build a 4096-dim binary fingerprint tensor from active bit indices.
    """
    x = torch.zeros(4096, dtype=torch.float32)
    x[bits] = 1
    return x


def smiles2vec(smiles, pca):
    """
    Convert a SMILES string to a low-dimensional vector via:
     1) ECFP4 fingerprint (4096 bits)
     2) PCA to 32 dims
    """
    mol = pybel.readstring('smi', smiles)
    fp = mol.calcfp('ecfp4')
    bits = [b - 1 for b in fp.bits]
    highdim = to_tensor(bits).numpy().reshape(1, -1)
    lowdim = pca.transform(highdim)
    return lowdim


def create_data_from_pt(pt_file, smi_csv, bbs_space_path, mapping_linker, tree, pca):
    """
    For each MOF in pt_file (list of dicts with key 'mof_id'):
      - Lookup row in smi_csv by Materials_name == mof_id
      - Extract Metal_nodes, Organic_linkers, Metal_valence, Linker_valence
      - If Linker_valence is missing/empty, match any nearest candidate
      - Query KDTree(k=10) for metal and each linker
      - Keep first candidate matching valence (or nearest if no valence)
      - Append 'bbs' field with dict of metal/linker candidates and valences
    Returns list of enriched MOF dicts.
    """
    # Load data sources
    test_data = torch.load(pt_file, map_location='cpu')
    smi_df = pd.read_csv(smi_csv)
    all_bbs = torch.load(bbs_space_path, map_location='cpu')
    mapping_linker_dict = torch.load(mapping_linker, map_location="cpu")

    enriched = []
    for entry in test_data:
        mof_id = entry.get('mof_id')
        row = smi_df[smi_df['Materials_name'] == mof_id]
        if row.empty:
            print(f"Warning: no SMILES row for {mof_id}")
            continue
        # parse columns
        metals = [s.strip() for s in str(row.iloc[0]['Metal_nodes']).split(',') if s.strip()]
        linkers = [str(x).strip() for x in ast.literal_eval(str(row.iloc[0]['Organic_linkers']).strip())]
        linker_prop = [int(v) for v in str(row.iloc[0]['Linker_proportion']).split(',') if v.strip().isdigit()]
        metal_vals = [int(v) for v in str(row.iloc[0]['Metal_valence']).split(',') if v.strip().isdigit()]
        linker_vals = [int(v) for v in str(row.iloc[0]['Linker_valence']).split(',') if v.strip().isdigit()]

        # select metal candidate
        metal_str = f'[{str(metals[0]).strip()}]'
        metal_val = metal_vals[0] if metal_vals else None
        m_vec = smiles2vec(metal_str, pca)
        dists, idxs = tree.query(m_vec, k=10)
        metal_cand = None
        for idx in idxs[0]:
            bb = all_bbs[idx].clone()
            bb.coord_num = (bb.atom_types == 2).sum().item()
            if bb.coord_num == metal_val:
                metal_cand = bb
                break
        if metal_cand is None:
            print(f"Warning: no metal block for {mof_id}")
            continue

        # select linker candidates
        linker_cands = []
        linker_val_list = []
        for i, lk in enumerate(linkers):
            val = linker_vals[i] if i < len(linker_vals) else None
            lk = mapping_linker_dict.get(lk, lk)
            lk = next((str(x).strip() for x in lk if str(x).strip()), "")
            l_vec = smiles2vec(lk, pca)
            dists, idxs = tree.query(l_vec, k=10)
            found = False
            for idx in idxs[0]:
                bb = all_bbs[idx].clone()
                bb.coord_num = (bb.atom_types == 2).sum().item()
                if val is None or bb.coord_num == val:
                    linker_cands.append(bb)
                    linker_val_list.append(bb.coord_num)
                    found = True
                    break
            if not found:
                print(f"Warning: no matched linker block for {mof_id} linker {lk}, place chack your linker valence (You can replace it to the number of [He] the {mof_id} have).")
                break
        if len(linker_cands) != len(linkers):
            continue

        entry['bbs'] = {
            'metal': [metal_cand],
            'linker': linker_cands,
            'linker_proportion': linker_prop,
            'metal_valence': [metal_val],
            'linker_valence': linker_val_list
        }
        enriched.append(entry)

    return enriched


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Match MOFs to building blocks via KDTree + valence filter.'
    )
    parser.add_argument('--pt_file', type=str, required=True,
                        help='Path to input .pt containing list of MOF dicts.')
    parser.add_argument('--smi_csv', type=str, required=True,
                        help='CSV with Materials_name, Metal_nodes, Organic_linkers, Metal_valence, Linker_valence.')
    parser.add_argument('--bbs_space', type=str, required=True,
                        help='Path to .pt file with building-block space list.')
    parser.add_argument('--map_linker_table', type=str, required=True,
                        help='Path to .pt file with standard linker smiles mapping table.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save processed MOF list .pt.')
    args = parser.parse_args()

    # build KDTree + PCA
    tree, pca = create_PCA_KDtree(args.bbs_space)
    print("KDTree and PCA created.")

    # process and enrich
    result = create_data_from_pt(args.pt_file, args.smi_csv, args.bbs_space, args.map_linker_table, tree, pca)
    print(f"Processed {len(result)} MOFs.")

    torch.save(result, args.output)
    print("Saved enriched MOFs to", args.output)
