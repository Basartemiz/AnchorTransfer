"""ANM (Anisotropic Network Model) conformation generation via ProDy."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def generate_anm_conformations(
    ca_coords: np.ndarray,
    n_modes: int = 5,
    amplitudes: list[float] | None = None,
    cutoff: float = 15.0,
    n_conformations: int | None = None,
) -> np.ndarray:
    """Generate conformational ensemble via ANM perturbation.

    For each of the lowest-frequency normal modes, displace the structure
    along ±amplitude × mode_vector, producing diverse conformations that
    sample the protein's energy landscape around the input structure.

    Args:
        ca_coords: (N_residues, 3) Cα coordinates of the seed structure
        n_modes: number of lowest-frequency modes to sample
        amplitudes: RMSD-scaled displacement magnitudes (default [1.0, 2.0, 3.0])
        cutoff: ANM interaction distance cutoff in Ångströms
        n_conformations: optional number of conformations to randomly subsample

    Returns:
        (n_conformations, N_residues, 3) float32 array.
        n_conformations = n_modes * len(amplitudes) * 2 (± directions).
        Returns shape (0, N, 3) if ANM fails (e.g. protein too short).
    """
    if amplitudes is None:
        amplitudes = [1.0, 2.0, 3.0]

    n_atoms = ca_coords.shape[0]
    if n_atoms < 4:
        logger.warning("Protein has %d residues — too short for ANM", n_atoms)
        return np.empty((0, n_atoms, 3), dtype=np.float32)

    try:
        from prody import ANM, calcANM
    except ImportError:
        from prody import ANM

    try:
        anm = ANM("protein")
        anm.buildHessian(ca_coords, cutoff=cutoff)
        anm.calcModes(n_modes=n_modes)
    except Exception as e:
        logger.warning("ANM failed for %d-residue protein: %s", n_atoms, e)
        return np.empty((0, n_atoms, 3), dtype=np.float32)

    actual_modes = anm.numModes()
    if actual_modes == 0:
        logger.warning("ANM produced 0 modes for %d-residue protein", n_atoms)
        return np.empty((0, n_atoms, 3), dtype=np.float32)

    n_use = min(n_modes, actual_modes)
    conformations = []

    for mode_idx in range(n_use):
        mode_vector = anm.getEigvecs()[:, mode_idx]  # (3*N,)
        mode_3d = mode_vector.reshape(n_atoms, 3)

        for amp in amplitudes:
            # Scale mode vector to desired RMSD displacement
            current_rmsd = np.sqrt(np.mean(np.sum(mode_3d ** 2, axis=1)))
            if current_rmsd < 1e-10:
                continue
            scale = amp / current_rmsd

            conformations.append(ca_coords + scale * mode_3d)   # + direction
            conformations.append(ca_coords - scale * mode_3d)   # - direction

    if not conformations:
        return np.empty((0, n_atoms, 3), dtype=np.float32)

    result = np.array(conformations, dtype=np.float32)
    if n_conformations is not None and n_conformations < result.shape[0]:
        rng = np.random.default_rng()
        indices = rng.choice(result.shape[0], size=n_conformations, replace=False)
        result = result[indices]
    return result
