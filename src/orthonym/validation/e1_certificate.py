"""v25 G1: E1 atom-coverage certificate.

Verifies a GeneralEngineResult's atom->token partition: every heavy atom of
the molecule is bound by EXACTLY ONE token, every token occurs in the
emitted name, and (G1 scope) the molecule is neutral. Pure Python +
RDKit -- no Java, no OPSIN. This is the load-bearing "no silent atom drop"
gate the legacy count-based validation/atom_coverage.py never was; that
legacy module is untouched and unrelated.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class E1Verdict:
    ok: bool
    reason: str


def verify_certificate(mol, result, allow_charged: bool = False) -> E1Verdict:
    """Atom-partition certificate for a GeneralEngineResult.

    v26 P5: ``allow_charged`` (set only under ``complete``) lifts the
    net-formal-charge refusal -- the charge is expressed as a
    ``-ylium``/``-ide``/``-uide``/``-ium`` suffix on an already-bound skeletal
    atom, so it introduces NO new atom and the partition is still complete.
    Default False -> byte-identical (the G1 neutral-scope guard still fires).
    """
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    seen = {}
    for binding in result.bindings:
        for idx in binding.atom_ids:
            if idx in seen:
                return E1Verdict(False, f"atom {idx} bound twice "
                                        f"({seen[idx]!r} and {binding.token!r})")
            seen[idx] = binding.token
    unbound = heavy - set(seen)
    if unbound:
        return E1Verdict(False, f"unbound heavy atoms: {sorted(unbound)}")
    phantom = set(seen) - heavy
    if phantom:
        return E1Verdict(False, f"bindings reference non-heavy/missing "
                                f"atoms: {sorted(phantom)}")
    for binding in result.bindings:
        if binding.token and binding.token not in result.name:
            return E1Verdict(False, f"token {binding.token!r} not in name")
    if not allow_charged and Chem.GetFormalCharge(mol) != 0:
        return E1Verdict(False, "net formal charge nonzero (G1 charge scope)")
    return E1Verdict(True, "ok")
