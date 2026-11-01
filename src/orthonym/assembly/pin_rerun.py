"""The PIN tier's promotion re-run for a first-run name the default tier will decline (DT3 step A).

``Orthonym._name_with_pin_promotion`` names the molecule once, and names it again with the promoted
ring-substituent producers admitted (``rules.pin_vocabulary.promote_at_pin_tier``) when the first run
named nothing. A first run that names the molecule with a name the default tier's emission rule then
declines (``namer._default_tier_emits``: a general-engine name, a strict-path name that carries a
recorded non-PIN part, a name that fails a spelling check) used to end the wrapper at once, so the
re-run, which can answer with a name every piece of which a strict producer built
(``assembly.strict_ring_prefix``), was never tried for it: 'N-[2-(5-methoxy-1H-indol-3-yl)ethyl]
acetamide' (melatonin) was declined at the default tier while the same molecule without the methoxy group
is a ``pin_verified`` name there.

This module holds the three pieces of that path that do not belong in ``namer.py``:

*:func:`first_run_declined` -- the default tier's own decision on the first run's name, the same row and
  the same rule the caller's ``name`` applies to the name the wrapper returns, without its side effects;
*:func:`first_run_state` -- the per-call state of the first run that ``Orthonym._tier_row_and_pin_form``
  reads (dispatch class, winning offer, prefix-order fallback), restored when the re-run's name is not
  shipped, so the decision on the first run's name is the one it had before the re-run ran;
*:func:`rerun_class_guard` -- the classes for which a re-run name is not shipped here. "INTRODUCTION"
  (the Blue Book, sentence:50943): preferred IUPAC names are not identified for the compounds of
  Chapter (carbohydrates, natural products, lipids, amino acids and peptides), so a sugar-ring molecule, a
  natural-product scaffold, a lipid backbone or a peptide-like molecule keeps the first run's result; and a name that breaks a spelling rule of the PIN
  (``validation.pin_spelling``) is no PIN;
*:func:`rerun_name_certified` -- the certificate that lets a kept re-run name keep the PIN label (DT3 step C).
"""
from __future__ import annotations

from typing import Callable, Optional

from rdkit import Chem

#: the attributes of the first run that ``Orthonym._tier_row_and_pin_form`` reads after the run
_STATE = ("_last_dispatch_class", "_last_selected_offer", "_prefix_order_fallback_name", "_offers")

def first_run_declined(namer, smiles: str, name: str) -> bool:
    """True when the default tier's emission rule declines ``name``, the first run's name of ``smiles``.

    The row is the one ``Orthonym._tier_row_and_pin_form`` derives from the provenance of the first run
    and the rule is ``namer._default_tier_emits``: the pair ``Orthonym._default_tier_decision`` applies to
    the name the wrapper returns. A row that cannot be derived is no decline (the first run's name is
    returned as before)."""
    from .. import namer as _namer
    from ..metrics.provenance import get_provenance
    try:
        row, pin_form = namer._tier_row_and_pin_form(smiles, name, get_provenance())
        return not _namer._default_tier_emits(row, pin_form, smiles, namer._trivial_fallback)
    except Exception:  # noqa: BLE001 - the decision is information; the first run's name stands
        return False


def first_run_state(namer, named: bool) -> Callable[[], None]:
    """A function that puts the per-call state of the first run back on ``namer``; a no-op when the first
    run named nothing (``named`` False), where nothing reads that state."""
    if not named:
        return lambda: None
    saved = {attr: getattr(namer, attr, None) for attr in _STATE}

    def restore() -> None:
        for attr, value in saved.items():
            setattr(namer, attr, value)
    return restore


def rerun_class_guard(mol, name: str) -> Optional[str]:
    """The reason the re-run name ``name`` of ``mol`` is not shipped over a declined first run, or None.

    * a pyranose / furanose pattern (``assembly.strict_ring_prefix.carbohydrate_core``):
      "Carbohydrates" (the Blue Book) names carbohydrates with their own names, and
      "INTRODUCTION" (:50943) identifies no PIN for the compounds of Chapter;
    * a natural-product class of the engine's own router (``namer.classify_compound_class``: steroid,
      alkaloid, terpene, carbohydrate scaffolds): "NOMENCLATURE FOR NATURAL PRODUCTS BASED ON PARENT
      HYDRIDES, (ALKALOIDS, STEROIDS, TERPENES,...)" (:50952) sits under the same sentence of;
    * a lipid backbone (``perception.lipids.detect_lipid_backbone``: a glyceride, a phospholipid or a
      sphingoid): the same sentence of, and a glyceride named by prefixes is the method (2) name
      of "When anions are different, two methods are used" (:31831), whose PIN is the
      method (1) name (:31846);
    * a dipeptide unit (``rules.peptides.peptide_like``): "AMINO ACIDS AND PEPTIDES" (:54150) is a chapter whose
      compounds (:50943) gives no PIN, and the substitutive name of a peptide-like molecule is not promoted
      (DT3 lead decision 1);
    * a spelling failure of the PIN (``validation.pin_spelling.check_pin_spelling``), for example the
      stereodescriptor of a substituent at the front of a heteroatom parent,:44643)."""
    if mol is None:
        return "no structure"
    from ..namer import classify_compound_class
    from ..perception.lipids import detect_lipid_backbone
    from ..validation.pin_spelling import check_pin_spelling
    from .strict_ring_prefix import carbohydrate_core
    try:
        if carbohydrate_core(mol, range(mol.GetNumAtoms())):
            return "sugar ring (P-102.1.1)"
        natural_product = classify_compound_class(mol, Chem.MolToSmiles(mol))
        if natural_product is not None:
            return f"natural-product class {natural_product} (P-101)"
        if detect_lipid_backbone(mol) is not None:
            return "lipid backbone (P-100)"
        from ..rules.peptides import peptide_like
        if peptide_like(mol):
            return "peptide-like (P-103)"
        if _prostane_like(mol):
            return "prostane skeleton (P-101)"
        if _carotenoid_like(mol):
            return "carotenoid polyene (P-101)"
        failures = check_pin_spelling(mol, name)
    except Exception as exc:  # noqa: BLE001 - a guard that cannot read the molecule lets nothing through
        return f"the guard could not read the molecule ({type(exc).__name__})"
    if failures:
        return f"{failures[0].rule} {failures[0].detail}"
    return None


def _acyclic_carbon_chain_length(mol, start: int, ring: set) -> int:
    """The number of carbons on the longest path of acyclic carbons that starts at ``start`` and leaves ``ring``."""
    best = 0
    stack = [(start, frozenset([start]))]
    while stack:
        atom, seen = stack.pop()
        best = max(best, len(seen))
        for nb in mol.GetAtomWithIdx(atom).GetNeighbors():
            j = nb.GetIdx()
            if j not in seen and j not in ring and nb.GetSymbol() == "C" and not nb.IsInRing():
                stack.append((j, seen | {j}))
    return best


def _prostane_like(mol) -> bool:
    """True when a five-membered carbon ring holds, on two neighbouring ring atoms, an acyclic carbon chain of seven
    carbons and one of at least five: the skeleton of 'prostane', a fundamental parent of
    ('prostane (fundamental parent structure)', the Blue Book; the C1-C7 chain at C8, C13-C20 at C12). A guard
    only, so a wider net keeps a name below the PIN and never makes one."""
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 5 or any(mol.GetAtomWithIdx(i).GetSymbol() != "C" for i in ring):
            continue
        ring_set = set(ring)
        lengths = {}
        for i in ring:
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                if nb.GetIdx() not in ring_set and nb.GetSymbol() == "C" and not nb.IsInRing():
                    lengths[i] = max(lengths.get(i, 0), _acyclic_carbon_chain_length(mol, nb.GetIdx(), ring_set))
        for k in range(5):
            a, b = ring[k], ring[(k + 1) % 5]
            pair = sorted((lengths.get(a, 0), lengths.get(b, 0)))
            if (pair[1] == 7 and pair[0] >= 5) or (pair[0] == 7 and pair[1] >= 7):
                return True
    return False


def _carotenoid_like(mol) -> bool:
    """True when an acyclic run of conjugated carbon-carbon double bonds holds at least four of them and at least two
    methyl branches: the polyene of the carotenes and retinoids, fundamental parents of ('β,β-carotene
    (fundamental parent structure)', the Blue Book). A guard only, as for ``_prostane_like``."""
    double = [b for b in mol.GetBonds()
              if b.GetBondType() == Chem.BondType.DOUBLE and not b.IsInRing()
              and b.GetBeginAtom().GetSymbol() == "C" and b.GetEndAtom().GetSymbol() == "C"]
    if len(double) < 4:
        return False
    atoms = {a for b in double for a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())}
    # group the double bonds into conjugated runs: two double bonds are conjugated when a single bond joins them
    parent = {b.GetIdx(): b.GetIdx() for b in double}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    by_atom = {}
    for b in double:
        for a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx()):
            by_atom[a] = b.GetIdx()
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in by_atom and j in by_atom:
            parent[find(by_atom[i])] = find(by_atom[j])
    runs = {}
    for b in double:
        runs.setdefault(find(b.GetIdx()), []).append(b)
    for run in runs.values():
        if len(run) < 4:
            continue
        run_atoms = {a for b in run for a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())}
        # a methyl branch sits on a polyene atom with three heavy neighbours; a chain-end methyl is no branch
        methyls = sum(1 for a in run_atoms if mol.GetAtomWithIdx(a).GetDegree() >= 3
                      for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                      if nb.GetSymbol() == "C" and nb.GetDegree() == 1 and nb.GetIdx() not in atoms)
        if methyls >= 2:
            return True
    return False


def rerun_name_certified(mol, name: str) -> bool:
    """True iff the kept re-run name ``name`` of ``mol`` is certified as the PIN by evidence on every part (DT3 step C).

    A kept re-run name carries pieces a breadth producer built, so it is labelled ``pin_unverified`` unless the
    evidence decides each of its parts: no class of:func:`rerun_class_guard` "INTRODUCTION",
    the Blue Book), no spelling check that fails or abstains (``validation.pin_spelling.pin_spelling_report``),
    and a reader that models every parent-bearing unit of the name and no Chapter mark
    (``validation.spelling.coverage.NameCoverage.full``). The caller has already required the OPSIN read-back to the
    full key, the PIN path, no recorded non-PIN part and no multiplicative candidate. Any doubt gives False: the name
    keeps ``pin_unverified``. A check that returns no verdict (UNDECIDED) does not block: the certificate is the class
    guard, no failing or abstaining check, and full coverage, not a PASS from every check."""
    if mol is None or not name:
        return False
    # the blind panel of step C found one non-PIN among the certified names, an exocyclic double bond whose E/Z
    # descriptor sat inside the ylidene prefix instead of at the front with the parent locant
    # "Specification by stereodescriptors 'E' and 'Z'", the Blue Book-48281); by the plan's stop rule the class
    # of that row, a name with an ylidene prefix, is not certified
    if "ylidene" in name:
        return False
    try:
        if rerun_class_guard(mol, name) is not None:
            return False
        from ..validation.pin_spelling import ABSTAIN, FAIL, pin_spelling_report
        report = pin_spelling_report(mol, name)
        return (report.coverage is not None and report.coverage.full
                and report.count(FAIL) == 0 and report.count(ABSTAIN) == 0)
    except Exception:
        return False
