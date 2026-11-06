"""Molecular mode-count metrics used by the NeurIPS 2021 sEH task.

The paper defines a Tanimoto-separated representative greedily: in sample
order, a molecule is a new representative only when its similarity is
strictly below 0.7 to every representative already retained.  The released
paper code uses RDKit's default topological ``RDKFingerprint`` for Tanimoto
comparisons.

This module expects its inputs to have already been filtered by the desired
reward threshold.  Keeping reward filtering in the environment makes all
three returned counts operate on exactly the same high-reward subset.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold


PAPER_TANIMOTO_SIMILARITY_THRESHOLD = 0.7
_TANIMOTO_BATCH_SIZE = 256


def _canonical_molecules(smiles: Iterable[str]) -> List[Tuple[str, Chem.Mol]]:
    """Parse and stable-deduplicate SMILES, retaining encounter order."""
    molecules: List[Tuple[str, Chem.Mol]] = []
    seen = set()

    for index, smi in enumerate(smiles):
        if not isinstance(smi, str):
            raise TypeError(
                "Expected a SMILES string at index "
                f"{index}, got {type(smi).__name__}"
            )
        if smi == "":
            # The empty string denotes the environment's initial state, not a
            # terminal molecule.
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {index}: {smi!r}")
        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
        if canonical_smi in seen:
            continue
        seen.add(canonical_smi)
        molecules.append((canonical_smi, mol))

    return molecules


def _is_tanimoto_separated(
    fingerprint,
    representative_fingerprints: List,
    threshold: float,
) -> bool:
    """Return whether ``fingerprint`` is below ``threshold`` to every rep."""
    # Chunking bounds the temporary similarity list and permits an early exit
    # as soon as a too-similar representative is found.
    for start in range(
        0,
        len(representative_fingerprints),
        _TANIMOTO_BATCH_SIZE,
    ):
        end = start + _TANIMOTO_BATCH_SIZE
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint,
            representative_fingerprints[start:end],
        )
        if any(similarity >= threshold for similarity in similarities):
            return False
    return True


def count_mode_metrics(
    smiles: Iterable[str],
    tanimoto_similarity_threshold: float = PAPER_TANIMOTO_SIMILARITY_THRESHOLD,
) -> Tuple[int, int, int]:
    """Count unique, Tanimoto-separated, and Bemis--Murcko modes.

    Args:
        smiles: High-reward molecule SMILES in encounter order.
        tanimoto_similarity_threshold: A new representative must have
            similarity strictly below this value to every prior
            representative.  The paper uses 0.7.

    Returns:
        ``(unique_molecules, tanimoto_separated_modes, murcko_scaffolds)``.

    Notes:
        RDKit represents the Murcko scaffold of an acyclic molecule as the
        empty SMILES string.  We retain that value as one scaffold bucket,
        matching a direct distinct-scaffold count.
    """
    threshold = float(tanimoto_similarity_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "tanimoto_similarity_threshold must be between 0 and 1, "
            f"got {threshold}"
        )

    molecules = _canonical_molecules(smiles)
    if not molecules:
        return (0, 0, 0)

    representative_fingerprints: List = []
    scaffolds = set()

    for _, mol in molecules:
        fingerprint = Chem.RDKFingerprint(mol)
        if _is_tanimoto_separated(
            fingerprint, representative_fingerprints, threshold
        ):
            representative_fingerprints.append(fingerprint)

        scaffolds.add(
            MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False,
            )
        )

    return (
        len(molecules),
        len(representative_fingerprints),
        len(scaffolds),
    )
