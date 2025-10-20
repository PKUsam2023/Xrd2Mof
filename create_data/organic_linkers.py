import os
import argparse
import torch
import numpy as np
import pandas as pd
import re
import ast
from collections import defaultdict

# RDKit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# PyTorch Geometric
from torch_geometric.data import Data


def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    return [int(x == s) for s in permitted_list]


def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    atom_type_enc = [atom.GetAtomicNum()]
    #atom_type_enc = one_hot_encoding(atom.GetSymbol(), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]

    atomic_mass_scaled = [(atom.GetMass() - 10.812) / 116.092]
    vdw_radius_scaled = [(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6]
    covalent_radius_scaled = [(Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76]

    atom_feature_vector = (atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc +
                           hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc +
                           atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled)

    if use_chirality:
        chirality_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", 
                                                                    "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_enc

    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)

def get_bond_features(bond, use_stereochemistry=True):
    bond_type_enc = one_hot_encoding(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE, 
                                                          Chem.rdchem.BondType.DOUBLE, 
                                                          Chem.rdchem.BondType.TRIPLE, 
                                                          Chem.rdchem.BondType.AROMATIC])
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def create_graph_from_smiles(smiles):
    """
    Inputs:
    
    smiles = [smiles] ... a SMILES string

    Outputs:
    
    data_list = [G_1] ... a torch_geometric.data.Data object which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    
    if smiles:
        
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
            
        X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
    return Data(x = X, edge_index = E, edge_attr = EF)


def merge_graphs(graph_list):
    x_list, edge_index_list, edge_attr_list = [], [], []
    offset = 0
    for g in graph_list:
        x_list.append(g.x)
        edge_index_list.append(g.edge_index + offset)
        edge_attr_list.append(g.edge_attr)
        offset += g.x.size(0)
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def main(input_csv, output_dir):
    df = pd.read_csv(input_csv)
    grouped = defaultdict(list)
    # iterate rows
    for _, row in df.iterrows():
        name = str(row['Materials_name'])
        cell = row['Organic_linkers']
        if pd.isna(cell):
            smiles_list = []
        else:
            s = str(cell).strip()
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    smiles_list = [str(x).strip() for x in parsed if str(x).strip()]
                else:
                    smiles_list = [str(parsed).strip()] if str(parsed).strip() else []
            except Exception:
                smiles_list = [p.strip() for p in s.split(',') if p.strip()]

        for smi in smiles_list:
            g = create_graph_from_smiles(smi)
            if g is not None:
                grouped[name].append(g)
    # merge per material
    merged = {name: merge_graphs(gs) for name,gs in grouped.items()}
    # save dict to pt
    output_pt = os.path.join(output_dir, "linker.pt")
    torch.save(merged, output_pt)
    print(f"Saved {len(merged)} material graphs to {output_pt}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SMILES from CSV into merged PyG graphs.')
    parser.add_argument('--input_csv', type=str, required=True, help='CSV path with Materials_name and Organic_linkers columns')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output linkers(.pt) file')
    args = parser.parse_args()
    main(args.input_csv, args.output_dir)
