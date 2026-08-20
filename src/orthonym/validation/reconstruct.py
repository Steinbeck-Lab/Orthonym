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

from dataclasses import dataclass
from enum import Enum
from typing import Optional

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


class Verdict(str, Enum):
    CONFIRMED = "confirmed"
    ABSTAINED = "abstained"
    MISMATCH = "mismatch"
    ERROR = "error"


@dataclass(frozen=True)
class ReconResult:
    verdict: Verdict
    reason: str
    reconstructed_smiles: Optional[str] = None


@dataclass(frozen=True)
class NameFacts:
    parent_kind: str                 # "chain" | "carbocycle"
    parent_length: int
    replacements: tuple = ()         # ((locant, "O"|"N"|"S"|"P"), ...)
    unsaturations: tuple = ()        # ((locant, 2|3), ...)
    principal_group: Optional[tuple] = None   # (key, (locants,))
    substituents: tuple = ()         # ((name, locant), ...) NAME, never input SMILES
    indicated_h: tuple = ()
    net_charge: int = 0
    isotopes: bool = False


class _Abstain(Exception):
    """Raised by any unmodeled construct — converted to ABSTAINED. The rebuild
    abstains rather than guess, so it NEVER falsely CONFIRMs."""


def _normalize(mol) -> str:
    """Canonical SMILES with stereochemistry removed (constitution-only compare).

    Applied identically to BOTH the input and the rebuild. Wave 0 does NOT collapse
    charge separation — it is unneeded here: net-charge facts abstain pre-rebuild,
    and a charge-separated input vs a neutral rebuild differs in canonical SMILES,
    i.e. a safe MISMATCH (never a false CONFIRM). Returns '' if the mol cannot be
    canonicalized (the caller treats '' as a failed compare)."""
    if mol is None:
        return ""
    try:
        m = Chem.Mol(mol)
        Chem.RemoveStereochemistry(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return ""


def reconstruct_and_verify(name_facts: "NameFacts", input_mol) -> ReconResult:
    """Rebuild an RDKit graph from name-facts ALONE and compare constitution.

    CONFIRMED only on a full canonical-SMILES byte-match (stereo removed both
    sides); a constitution difference is MISMATCH; any unmodeled construct raises
    _Abstain -> ABSTAINED; a top-level exception -> ERROR. NEVER falsely CONFIRMs,
    NEVER raises."""
    try:
        if input_mol is None:
            return ReconResult(Verdict.ERROR, "input mol is None")
        if has_unverifiable_atoms(input_mol):
            return ReconResult(Verdict.ABSTAINED, "input has unverifiable atoms")
        if name_facts.net_charge != 0:
            return ReconResult(Verdict.ABSTAINED, "net charge unmodeled (Wave 0)")
        if name_facts.isotopes:
            return ReconResult(Verdict.ABSTAINED, "isotopes unmodeled (Wave 0)")
        if name_facts.indicated_h:
            return ReconResult(Verdict.ABSTAINED, "indicated_h unmodeled (Wave 0)")
        ref = _normalize(input_mol)
        if not ref:
            return ReconResult(Verdict.ABSTAINED, "input not canonicalizable")
        try:
            rebuilt = _build_from_facts(name_facts)
        except _Abstain as ab:
            return ReconResult(Verdict.ABSTAINED, str(ab))
        got = _normalize(rebuilt)
        if not got:
            return ReconResult(Verdict.ABSTAINED, "rebuild not canonicalizable")
        if got == ref:
            return ReconResult(Verdict.CONFIRMED, "constitution byte-match", got)
        return ReconResult(Verdict.MISMATCH, f"{got} != {ref}", got)
    except Exception as exc:
        return ReconResult(Verdict.ERROR, f"reconstruct raised: {exc!r}")


def _build_from_facts(f: "NameFacts"):
    """Rebuild parent skeleton + principal group. Raises _Abstain on anything
    unmodeled. Task 3 extends with replacements/unsaturations/substituents."""
    if f.parent_kind == "chain":
        if f.parent_length < 1:
            raise _Abstain("nonpositive chain length")
        rw = Chem.RWMol()
        idx_by_locant = {}
        for i in range(f.parent_length):
            j = rw.AddAtom(Chem.Atom(6))
            idx_by_locant[i + 1] = j
            if i > 0:
                rw.AddBond(idx_by_locant[i], j, Chem.BondType.SINGLE)
    elif f.parent_kind == "carbocycle":
        if f.parent_length < 3:
            raise _Abstain("ring too small")
        rw = Chem.RWMol()
        idx_by_locant = {}
        for i in range(f.parent_length):
            j = rw.AddAtom(Chem.Atom(6))
            idx_by_locant[i + 1] = j
            if i > 0:
                rw.AddBond(idx_by_locant[i], j, Chem.BondType.SINGLE)
        rw.AddBond(idx_by_locant[f.parent_length], idx_by_locant[1], Chem.BondType.SINGLE)
    else:
        raise _Abstain(f"unmodeled parent_kind {f.parent_kind!r}")

    if f.replacements or f.unsaturations or f.substituents:
        raise _Abstain("replacements/unsaturations/substituents unmodeled in Task 2")

    _apply_principal_group(rw, idx_by_locant, f.principal_group)

    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return m


def _apply_principal_group(rw, idx_by_locant, pg):
    if pg is None:
        return
    key, locants = pg
    if key == "ol":
        for L in locants:
            o = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(idx_by_locant[L], o, Chem.BondType.SINGLE)
    elif key == "one":
        for L in locants:
            o = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(idx_by_locant[L], o, Chem.BondType.DOUBLE)
    else:
        # oic_acid/amide/nitrile add a carbon that IUPAC counts inside the parent
        # length — modeling their carbon-counting is a later wave. Abstain to stay
        # sound-over-complete (never a false CONFIRM).
        raise _Abstain(f"unmodeled principal group {key!r}")
