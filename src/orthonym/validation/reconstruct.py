"""Wave 0: OPSIN-free structural verifier.

Two 0-wrong layers modeled on ``audit/reconstruction.py``:
  * ``has_unverifiable_atoms`` — the wildcard fail-close predicate (this task).
  * ``reconstruct_and_verify`` / ``verify_or_none`` — a sound-over-complete
    name->graph reconstructor + dual oracle (Tasks 2–4).

SOUNDNESS CONTRACT (load-bearing): the reconstructor NEVER falsely CONFIRMs, and
it rebuilds ONLY from name-level facts (``NameFacts``) that a producer derives
from NAME TOKENS — parent length, replacement/unsaturation locants, the
principal-group key, substituent NAMES. It must NEVER read the input graph to
populate a fact; that would make the compare circular and defeat the point.
On anything it does not model it returns ABSTAINED, never CONFIRMED.
"""
from __future__ import annotations

from rdkit import Chem


def has_unverifiable_atoms(mol) -> bool:
    """True iff *mol* carries an atom no oracle can verify.

    Wave 0 scope: a dummy/wildcard atom (atomic number 0, SMILES ``*``). Such an
    atom makes the input InChIKey uncomputable, so the SELF-01/OPSIN round-trip
    oracle has no reference and FAILS OPEN — a wildcard input then ships a WRONG
    molecule (``CC*`` -> ``ethane``). Callers must fail closed (abstain) when this
    returns True. Wave 0 is wildcard only, matching ``errors.classify_scope_limit``.
    """
    if mol is None:
        return False
    return any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
