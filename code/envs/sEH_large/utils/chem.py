"""
Vendored from GFNOrg/gflownet (mols/utils/chem.py) trimmed to the symbols
needed for proxy evaluation (mol building + atom/bond featurization).

Notable changes from upstream:
  - `np.int` (deprecated) replaced with the standard `int`/`np.int64` dtypes
  - `torch_sparse.coalesce` swapped for `torch_geometric.utils.coalesce`
    so the environment runs without `torch_sparse` installed
  - dropping the docking / file-IO utilities that are not needed for the
    pretrained-proxy reward path used by the sEH_large env
"""

import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDConfig

import os
import pandas as pd

import torch
from torch_geometric.data import Data
try:
    from torch_geometric.utils import coalesce as _pyg_coalesce
except ImportError:  # pragma: no cover - very old torch_geometric
    _pyg_coalesce = None

rdBase.DisableLog("rdApp.error")

atomic_numbers = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
    "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
    "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
    "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46,
    "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55,
    "Ba": 56,
}


def onehot(arr, num_classes, dtype=np.int64):
    arr = np.asarray(arr, dtype=np.int64)
    assert len(arr.shape) == 1, "dims other than 1 not implemented"
    onehot_arr = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    onehot_arr[np.arange(arr.shape[0]), arr] = 1
    return onehot_arr


def mol_from_frag(jun_bonds, frags=None, frag_smis=None, optimize=False):
    """Joins 2 or more fragments into a single rdkit molecule."""
    jun_bonds = np.asarray(jun_bonds)
    if frags is not None:
        pass
    elif frag_smis is not None:
        frags = [Chem.MolFromSmiles(s) for s in frag_smis]
    else:
        raise ValueError("either frags or frag_smis must be provided")
    if len(frags) == 0:
        return None, None
    nfrags = len(frags)
    mol = frags[0]
    for i in np.arange(nfrags - 1) + 1:
        mol = Chem.CombineMols(mol, frags[i])

    frag_startidx = np.concatenate(
        [[0], np.cumsum([frag.GetNumAtoms() for frag in frags])], 0
    )[:-1]

    if jun_bonds.size == 0:
        mol_bonds = []
    else:
        mol_bonds = frag_startidx[jun_bonds[:, 0:2]] + jun_bonds[:, 2:4]

    emol = Chem.EditableMol(mol)
    [emol.AddBond(int(b[0]), int(b[1]), Chem.BondType.SINGLE) for b in mol_bonds]
    mol = emol.GetMol()
    atoms = list(mol.GetAtoms())

    def _pop_H(atom):
        nh = atom.GetNumExplicitHs()
        if nh > 0:
            atom.SetNumExplicitHs(nh - 1)

    [(_pop_H(atoms[b[0]]), _pop_H(atoms[b[1]])) for b in mol_bonds]
    Chem.SanitizeMol(mol)
    if optimize:
        Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        Chem.RemoveHs(mol)
    return mol, mol_bonds


_mpnn_feat_cache = [None]


def mpnn_feat(mol, ifcoord=True, panda_fmt=False, one_hot_atom=False, donor_features=False):
    """Compute the per-atom and per-bond featurisation expected by model_atom.MPNNet_v2."""
    atomtypes = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    bondtypes = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    natm = len(mol.GetAtoms())
    ntypes = len(atomtypes)
    nfeat = ntypes + 1 + 8
    if one_hot_atom:
        nfeat += len(atomic_numbers)
    atmfeat = np.zeros((natm, nfeat))

    for i, atom in enumerate(mol.GetAtoms()):
        type_idx = atomtypes.get(atom.GetSymbol(), 5)
        atmfeat[i, type_idx] = 1
        if one_hot_atom:
            atmfeat[i, ntypes + 9 + atom.GetAtomicNum() - 1] = 1
        else:
            atmfeat[i, ntypes + 1] = (atom.GetAtomicNum() % 16) / 2.0
        atmfeat[i, ntypes + 4] = atom.GetIsAromatic()
        hybridization = atom.GetHybridization()
        atmfeat[i, ntypes + 5] = hybridization == HybridizationType.SP
        atmfeat[i, ntypes + 6] = hybridization == HybridizationType.SP2
        atmfeat[i, ntypes + 7] = hybridization == HybridizationType.SP3
        atmfeat[i, ntypes + 8] = atom.GetTotalNumHs(includeNeighbors=True)

    if donor_features:
        if _mpnn_feat_cache[0] is None:
            fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
            factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
            _mpnn_feat_cache[0] = factory
        else:
            factory = _mpnn_feat_cache[0]
        feats = factory.GetFeaturesForMol(mol)
        for j in range(0, len(feats)):
            if feats[j].GetFamily() == "Donor":
                for k in feats[j].GetAtomIds():
                    atmfeat[k, ntypes + 3] = 1
            elif feats[j].GetFamily() == "Acceptor":
                for k in feats[j].GetAtomIds():
                    atmfeat[k, ntypes + 2] = 1

    if ifcoord:
        coord = np.asarray([mol.GetConformer(0).GetAtomPosition(j) for j in range(natm)])
    else:
        coord = None

    bond = np.asarray(
        [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
    )
    bondfeat = [bondtypes[bond.GetBondType()] for bond in mol.GetBonds()]
    bondfeat = onehot(bondfeat, num_classes=len(bondtypes))

    return atmfeat, coord, bond, bondfeat


def mol_to_graph_backend(atmfeat, coord, bond, bondfeat, props=None, data_cls=Data):
    """Convert atom/bond featurization into a torch_geometric Data object."""
    if props is None:
        props = {}
    natm = atmfeat.shape[0]
    atmfeat = torch.tensor(atmfeat, dtype=torch.float32)

    if bond.shape[0] > 0:
        edge_index = torch.tensor(
            np.concatenate([bond.T, np.flipud(bond.T)], axis=1), dtype=torch.int64
        )
        edge_attr = torch.tensor(
            np.concatenate([bondfeat, bondfeat], axis=0), dtype=torch.float32
        )
        if _pyg_coalesce is not None:
            edge_index, edge_attr = _pyg_coalesce(edge_index, edge_attr, num_nodes=natm)
        else:
            try:
                from torch_sparse import coalesce as _ts_coalesce
                edge_index, edge_attr = _ts_coalesce(edge_index, edge_attr, natm, natm)
            except ImportError as e:
                raise ImportError(
                    "Either torch_geometric>=2.0 (for torch_geometric.utils.coalesce) "
                    "or torch_sparse must be installed to featurize sEH_large molecules."
                ) from e
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.int64)
        edge_attr = torch.tensor(bondfeat, dtype=torch.float32)

    if coord is not None:
        coord = torch.tensor(coord, dtype=torch.float32)
        data = data_cls(x=atmfeat, pos=coord, edge_index=edge_index, edge_attr=edge_attr, **props)
    else:
        data = data_cls(x=atmfeat, edge_index=edge_index, edge_attr=edge_attr, **props)
    return data
