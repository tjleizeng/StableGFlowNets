"""
Vendored from GFNOrg/gflownet (mols/mol_mdp_ext.py) — only the bits needed
to (a) build a molecule from blocks and (b) prepare the PyG representation
expected by the pretrained sEH proxy.
"""

from collections import defaultdict
import numpy as np
from rdkit import Chem

from .utils.molMDP import BlockMoleculeData, MolMDP
from .utils import chem
from . import model_atom, model_block


class BlockMoleculeDataExtended(BlockMoleculeData):
    @property
    def mol(self):
        return chem.mol_from_frag(jun_bonds=self.jbonds, frags=self.blocks)[0]

    @property
    def smiles(self):
        return Chem.MolToSmiles(self.mol)

    def copy(self):
        o = BlockMoleculeDataExtended()
        o.blockidxs = list(self.blockidxs)
        o.blocks = list(self.blocks)
        o.slices = list(self.slices)
        o.numblocks = self.numblocks
        o.jbonds = list(self.jbonds)
        o.stems = list(self.stems)
        return o

    def as_dict(self):
        return {
            "blockidxs": self.blockidxs,
            "slices": self.slices,
            "numblocks": self.numblocks,
            "jbonds": self.jbonds,
            "stems": self.stems,
        }


class MolMDPExtended(MolMDP):
    def build_translation_table(self):
        """Build the symmetry mapping over duplicated blocks. Required only when
        computing parent transitions, which the env_base wrapper does not need
        for the linear-chain encoding, but we keep it because the upstream
        ``Proxy`` class calls it indirectly via ``post_init``."""
        from rdkit import Chem as _Chem  # local alias
        self.translation_table = {}
        for blockidx in range(len(self.block_mols)):
            atom_map = {}
            for j in range(len(self.block_mols)):
                if self.block_smi[blockidx] == self.block_smi[j]:
                    atom_map[self.block_rs[j][0]] = j
            self.translation_table[blockidx] = atom_map

        gold = _Chem.MolFromSmiles("[Au]")
        for blockidx in range(len(self.block_mols)):
            for j in self.block_rs[blockidx]:
                if j not in self.translation_table[blockidx]:
                    symmetric_duplicate = None
                    for atom, block_duplicate in self.translation_table[blockidx].items():
                        molA, _ = chem.mol_from_frag(
                            jun_bonds=[[0, 1, 0, j]], frags=[gold, self.block_mols[blockidx]]
                        )
                        molB, _ = chem.mol_from_frag(
                            jun_bonds=[[0, 1, 0, atom]], frags=[gold, self.block_mols[blockidx]]
                        )
                        if (
                            _Chem.MolToSmiles(molA) == _Chem.MolToSmiles(molB)
                            or molA.HasSubstructMatch(molB)
                        ):
                            symmetric_duplicate = block_duplicate
                            break
                    if symmetric_duplicate is None:
                        raise ValueError(
                            "block",
                            blockidx,
                            self.block_smi[blockidx],
                            "has no duplicate for atom",
                            j,
                            "in position 0, and no symmetrical correspondance",
                        )
                    self.translation_table[blockidx][j] = symmetric_duplicate

    def add_block_to(self, mol, block_idx, stem_idx=None, atmidx=None):
        """Out-of-place version of ``add_block``."""
        if mol.numblocks == 0:
            stem_idx = None
        new_mol = mol.copy()
        new_mol.add_block(
            block_idx,
            block=self.block_mols[block_idx],
            block_r=self.block_rs[block_idx],
            stem_idx=stem_idx,
            atmidx=atmidx,
        )
        return new_mol

    def reset(self):
        self.molecule = BlockMoleculeDataExtended()
        return None

    def post_init(self, device, repr_type, include_bonds=False, include_nblocks=False):
        self.device = device
        self.repr_type = repr_type
        self.max_num_atm = max(self.block_natm)
        self.true_block_set = sorted(set(self.block_smi))
        self.stem_type_offset = np.int32(
            [0]
            + list(
                np.cumsum(
                    [
                        max(self.block_rs[self.block_smi.index(i)]) + 1
                        for i in self.true_block_set
                    ]
                )
            )
        )
        self.num_stem_types = self.stem_type_offset[-1]
        self.true_blockidx = [self.true_block_set.index(i) for i in self.block_smi]
        self.num_true_blocks = len(self.true_block_set)
        self.include_nblocks = include_nblocks
        self.include_bonds = include_bonds
        self.molcache = {}

    def mols2batch(self, mols):
        if self.repr_type == "block_graph":
            return model_block.mols2batch(mols, self)
        elif self.repr_type == "atom_graph":
            return model_atom.mols2batch(mols, self)
        else:
            raise ValueError(
                f"sEH_large only supports repr_type 'block_graph' or 'atom_graph', got {self.repr_type!r}"
            )

    def mol2repr(self, mol=None):
        if mol is None:
            mol = self.molecule
        if self.repr_type == "block_graph":
            return model_block.mol2graph(mol, self, self.floatX)
        elif self.repr_type == "atom_graph":
            return model_atom.mol2graph(
                mol, self, self.floatX, bonds=self.include_bonds, nblocks=self.include_nblocks
            )
        else:
            raise ValueError(
                f"sEH_large only supports repr_type 'block_graph' or 'atom_graph', got {self.repr_type!r}"
            )
