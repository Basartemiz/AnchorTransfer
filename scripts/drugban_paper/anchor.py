"""Drug-centric anchor retrieval for binary DTI via Tanimoto similarity.

Strategy: for a query drug, find the most similar drug in training positives,
then use one of its known binding proteins as the anchor protein.
Works on all splits (random, cold, cluster) since it only depends on drug similarity.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

log = logging.getLogger(__name__)

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def compute_morgan_fp(smiles: str):
    """Compute Morgan fingerprint. Returns None if SMILES invalid."""
    smi = smiles.split(" |")[0].strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return _MORGAN_GEN.GetFingerprint(mol)


class AnchorIndex:
    """Drug-centric anchor retrieval.

    For a query (drug_q, protein_q):
      1. Find drug_a = most Tanimoto-similar drug to drug_q among all training positives
         (excluding drug_q itself)
      2. Get protein_a = a protein that drug_a binds (Y=1), different from protein_q
      3. Anchor = protein_a

    No self-anchors: drug_a != drug_q AND protein_a != protein_q.
    """

    def __init__(
        self,
        train_smiles: list[str],
        train_proteins: list[str],
        train_labels: list[int],
    ):
        self._fp_cache: dict[str, object] = {}

        # Collect all unique positive drugs and their binding proteins
        self.drug_to_proteins: dict[str, set[str]] = defaultdict(set)
        for smi, prot, label in zip(train_smiles, train_proteins, train_labels):
            if label == 1:
                self.drug_to_proteins[smi].add(prot)

        # Build FP list for all positive drugs (for BulkTanimoto)
        self._pos_smiles: list[str] = []
        self._pos_fps: list = []
        for smi in tqdm(sorted(self.drug_to_proteins.keys()), desc="Building anchor FPs"):
            fp = self._get_fp(smi)
            if fp is not None:
                self._pos_smiles.append(smi)
                self._pos_fps.append(fp)

        log.info(
            f"AnchorIndex: {len(self._pos_smiles)} positive drugs, "
            f"{len(self.drug_to_proteins)} unique drugs with {sum(len(v) for v in self.drug_to_proteins.values())} bindings"
        )

    def _get_fp(self, smiles: str):
        if smiles not in self._fp_cache:
            self._fp_cache[smiles] = compute_morgan_fp(smiles)
        return self._fp_cache[smiles]

    def get_anchor(self, query_smiles: str, query_protein: str) -> tuple[str | None, str | None]:
        """Get anchor protein for a (drug, protein) query.

        Returns:
            (anchor_drug_smi, anchor_protein_seq) or (None, None) if no valid anchor.
        """
        query_fp = self._get_fp(query_smiles)
        if query_fp is None:
            return None, None

        # BulkTanimoto against all positive drugs
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, self._pos_fps)

        # Sort by similarity descending, try each until we find a valid anchor
        ranked = sorted(enumerate(sims), key=lambda x: -x[1])
        for idx, sim in ranked:
            anchor_smi = self._pos_smiles[idx]
            # No self-drug
            if anchor_smi == query_smiles:
                continue
            # Get a binding protein different from query protein
            bound_prots = self.drug_to_proteins.get(anchor_smi, set())
            for prot in bound_prots:
                if prot != query_protein:
                    return anchor_smi, prot

        return None, None

    def resolve_batch(
        self,
        smiles_list: list[str],
        protein_list: list[str],
        graph_cache: dict,
    ) -> dict[tuple[str, str], tuple[str | None, str]]:
        """Batch-resolve anchors for all (drug, protein) pairs.

        Returns dict mapping (smi, prot) -> (anchor_smi, anchor_prot_seq).
        anchor_smi is None if no valid anchor found.
        """
        results: dict[tuple[str, str], tuple[str | None, str]] = {}
        unique_pairs = list(set(zip(smiles_list, protein_list)))

        # Pre-compute query FPs
        new_smiles = set(s for s, _ in unique_pairs) - set(self._fp_cache.keys())
        if new_smiles:
            for smi in tqdm(sorted(new_smiles), desc="Computing query FPs"):
                self._get_fp(smi)

        for smi, prot in tqdm(unique_pairs, desc="Resolving anchors"):
            anchor_smi, anchor_prot = self.get_anchor(smi, prot)
            if anchor_smi is not None:
                results[(smi, prot)] = (anchor_smi, anchor_prot)
            else:
                results[(smi, prot)] = (None, "")

        n_found = sum(1 for v in results.values() if v[0] is not None)
        log.info(f"  Anchor resolution: {n_found}/{len(unique_pairs)} found")
        return results
