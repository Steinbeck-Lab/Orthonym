"""Coverage estimation from atom indices, for naming-time gating.

``estimate_parent_coverage`` reports what fraction of a molecule's heavy atoms
the named parent structure accounts for, WITHOUT requiring OPSIN or any
external process.  It is O(1) over data already available in the naming
pipeline, and it counts ATOMS -- it is a real fraction in [0, 1].

It is not, and must not be used as, a proof of correctness: covering the right
NUMBER of atoms is not covering the right atoms.  Constitution is decided by
``validation/atom_coverage.py`` (InChIKey skeleton), which is the only thing
that separates e.g. sarcosine from alanine -- they share a formula AND an
element multiset.

REMOVED 2026-08-02 (Task Z2): ``estimate_name_coverage_heuristic``, which
returned ``min(len(name) / heavy / 1.5, 1.0)``.  A name's character count is
not an estimate of how many atoms it names; the quantity is anti-correlated
with coverage (correct ``cholesterol`` 0.393, a name that INVENTS atoms 5.333),
and clamping it to 1.0 hid exactly the atom-gain case.  It had **zero callers**
in ``src/``, ``scripts/`` and ``eval/`` -- only its own unit tests -- and a
fresh-process spy over 40 molecules recorded **0** executions.  Nothing was
rewired: there was no consumer to rewire.  Do not reintroduce a name-length
estimator here; if atom indices are unavailable, the honest answer is that
coverage is unknown, not a number derived from the string.

Usage example::

    from orthonym.assembly.coverage_utils import estimate_parent_coverage

    coverage = estimate_parent_coverage(mol, parent_atom_indices)
    if coverage < 0.60 and mol.GetNumHeavyAtoms() > 10:
        return None  # fall through to next handler
"""


def estimate_parent_coverage(mol, parent_atom_indices: set) -> float:
    """Fraction of molecule's heavy atoms in the named parent structure.

    Args:
        mol: RDKit Mol object.
        parent_atom_indices: Set of atom indices in the named parent.

    Returns:
        Coverage ratio 0.0-1.0.
    """
    total_heavy = mol.GetNumHeavyAtoms()
    if total_heavy == 0:
        return 1.0
    return len(parent_atom_indices) / total_heavy
