""".3 — the corrected three-metric coverage instrument.

Pure classifiers + aggregator. NO Java and NO Orthonym naming happen in
this module: the round-trip is performed by an injected callable
(``name_to_smiles: str -> Optional[str]``), so the unit tests exercise every
branch with a dict-backed fake OPSIN. The production runner
(``scripts/coverage_metrics.py``) supplies the real OPSIN subprocess.

Corrected metric contract (binds the v25 ship gate — see the master plan
"Metric definitions"):

* ``coverage`` = emitted / total. "Emitted" = Orthonym produced a real
  name (NOT ``errors.is_failure_name``).
* ``of_emitted_rt`` = round-trip-valid / emitted. A name is RT-valid iff
  OPSIN PARSES it AND the parsed structure matches the input under the
  chosen matcher.
* ``confidently_wrong`` = emitted AND OPSIN-parses AND structure-MISMATCH,
  as a fraction of TOTAL. **OPSIN-unparseable emissions are their OWN
  bucket (``unparseable``), never counted as wrong** — this correction is
  what flips the ship gate relative to the earlier conflated metric.
* ``unparseable`` = emitted AND OPSIN-does-not-parse, fraction of TOTAL.
* ``abs_rt_valid`` = coverage × of_emitted_rt = RT-valid / total. This is
  the ~99% target axis.

Every metric is reported under BOTH matchers (they are NOT interchangeable):

* ``parity`` — the most lenient lens, mirroring the forgiving RT axis a
  name-everything reports. Full standardization of both sides
  (fragment-parent → normalize → reionize → uncharge → canonical tautomer)
  then canonical-SMILES equality. This forgives (a) charge/protonation
  state, (b) tautomers broadly — RDKit's canonical tautomer merges keto-enol
  AND amide/imidol shifts, wider than InChI mobile-H — and (c) stereo,
  because ``CanonicalTautomer`` normalizes away stereo descriptors. It
  answers "does the name describe the right CONSTITUTION, ignoring charge,
  tautomer and stereo specificity?"
* ``strict`` — the Orthonym house matcher: InChIKey skeleton block (formula
  + connectivity + mobile-H, stereo- AND charge-insensitive) PLUS net formal
  charge. This is exactly the SELF-01 constitutional key. It is
  charge-SENSITIVE (catches a dropped/added charge the parity lens forgives)
  and tautomer-tolerant only to InChI's mobile-H scope (it does NOT merge
  keto-enol), while remaining stereo-insensitive.

**Stereo caveat (documented, not a bug).** BOTH matchers forgive stereo, so
neither flags a stereo INVERSION (name says S, input is R) — it counts as
RT-valid. This is by design and consistent with the SELF-01 RT gate being
stereo-insensitive; stereo-inversion residual wrongness is owned by the C.1
stratified residual-sampling audit, not by these automated matchers. The
upshot: of-emitted-RT is a (small) OVER-estimate to the extent stereo
inversions occur. Crediting stereo OMISSION as a match is intended (the tier
is allowed to emit constitution-only names).

``pin_pct`` (curated gold-gate pass rate) is a DIFFERENT population and is
carried through verbatim if supplied; it is never derived here and never
presented as commensurable with coverage.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Sequence

from rdkit import Chem
from rdkit.Chem import inchi

from ..errors import is_failure_name

__all__ = [
    "Matcher",
    "RowClassification",
    "RowResult",
    "classify_row",
    "aggregate",
    "parity_match",
    "strict_match",
]

# A name-to-SMILES round-tripper. Returns None when OPSIN cannot parse the
# name (or is unavailable — the caller decides how to treat unavailability;
# for metrics we treat None as "did not parse").
NameToSmiles = Callable[[str], Optional[str]]


class Matcher(str, Enum):
    PARITY = "parity"
    STRICT = "strict"


# ---------------------------------------------------------------------------
# Structure matchers
# ---------------------------------------------------------------------------

_STD_CACHE: Dict[str, Optional[str]] = {}


def _parity_key(smiles: str) -> Optional[str]:
    """Charge/tautomer/stereo-forgiving canonical key (the lenient lens).

    fragment-parent → normalize → reionize → uncharge → canonical tautomer,
    then canonical SMILES. ``CanonicalTautomer`` normalizes away stereo, so
    the key is stereo-insensitive (see the module stereo caveat). None on any
    failure.
    """
    if smiles in _STD_CACHE:
        return _STD_CACHE[smiles]
    key = None
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = rdMolStandardize.FragmentParent(mol)
            mol = rdMolStandardize.Normalize(mol)
            mol = rdMolStandardize.Reionize(mol)
            mol = rdMolStandardize.Uncharger().uncharge(mol)
            mol = rdMolStandardize.CanonicalTautomer(mol)
            if mol is not None:
                key = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        key = None
    _STD_CACHE[smiles] = key
    return key


def parity_match(input_smiles: str, opsin_smiles: str) -> bool:
    """True iff the two structures agree under the parity matcher
    (charge/tautomer-forgiving, stereo-comparing). False if either side
    cannot be standardized (fail-CLOSED for the metric — an
    unstandardizable emission is not credited as a match)."""
    a = _parity_key(input_smiles)
    b = _parity_key(opsin_smiles)
    if a is None or b is None:
        return False
    return a == b


def _skeleton(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        ik = inchi.MolToInchiKey(mol)
        return ik.split("-")[0] if ik else None
    except Exception:
        return None


def _net_charge(smiles: str) -> Optional[int]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return sum(a.GetFormalCharge() for a in mol.GetAtoms())
    except Exception:
        return None


def strict_match(input_smiles: str, opsin_smiles: str) -> bool:
    """True iff InChIKey skeleton block AND net formal charge agree (the
    SELF-01 constitutional key). False if either side cannot be hashed."""
    sa, sb = _skeleton(input_smiles), _skeleton(opsin_smiles)
    if sa is None or sb is None or sa != sb:
        return False
    ca, cb = _net_charge(input_smiles), _net_charge(opsin_smiles)
    if ca is None or cb is None:
        return False
    return ca == cb


# ---------------------------------------------------------------------------
# Per-row classification
# ---------------------------------------------------------------------------

class RowClassification(str, Enum):
    ABSTAINED = "abstained"        # Orthonym emitted no real name
    RT_VALID = "rt_valid"          # emitted, OPSIN parses, structure matches
    CONFIDENTLY_WRONG = "confidently_wrong"  # emitted, parses, mismatch
    UNPARSEABLE = "unparseable"    # emitted, OPSIN cannot parse it


@dataclass
class RowResult:
    smiles: str
    name: Optional[str]
    emitted: bool
    opsin_parsed: Optional[bool]      # None when not emitted
    parity: Optional[RowClassification]
    strict: Optional[RowClassification]


def classify_row(smiles: str, name: Optional[str],
                 name_to_smiles: NameToSmiles) -> RowResult:
    """Classify one (input, emitted-name) pair under both matchers.

    A single OPSIN call is made per emitted name; the parsed SMILES is then
    scored by each matcher (the parse/unparseable axis is matcher-invariant,
    only the match/mismatch split differs).
    """
    if is_failure_name(name):
        return RowResult(smiles, name, emitted=False, opsin_parsed=None,
                         parity=RowClassification.ABSTAINED,
                         strict=RowClassification.ABSTAINED)
    opsin_smiles = None
    try:
        opsin_smiles = name_to_smiles(name)
    except Exception:
        opsin_smiles = None
    if not opsin_smiles:
        return RowResult(smiles, name, emitted=True, opsin_parsed=False,
                         parity=RowClassification.UNPARSEABLE,
                         strict=RowClassification.UNPARSEABLE)

    def _cls(matched: bool) -> RowClassification:
        return (RowClassification.RT_VALID if matched
                else RowClassification.CONFIDENTLY_WRONG)

    return RowResult(
        smiles, name, emitted=True, opsin_parsed=True,
        parity=_cls(parity_match(smiles, opsin_smiles)),
        strict=_cls(strict_match(smiles, opsin_smiles)),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _matcher_block(rows: Sequence[RowResult], attr: str) -> Dict[str, float]:
    total = len(rows)
    covered = sum(1 for r in rows if r.emitted)
    cls = [getattr(r, attr) for r in rows]
    rt = sum(1 for c in cls if c is RowClassification.RT_VALID)
    wrong = sum(1 for c in cls if c is RowClassification.CONFIDENTLY_WRONG)
    unparse = sum(1 for c in cls if c is RowClassification.UNPARSEABLE)

    def _pct(num, den):
        return (num / den) if den else 0.0

    coverage = _pct(covered, total)
    of_emitted_rt = _pct(rt, covered)
    return {
        "total": total,
        "covered": covered,
        "rt_valid": rt,
        "confidently_wrong_count": wrong,
        "unparseable_count": unparse,
        "coverage": coverage,
        "of_emitted_rt": of_emitted_rt,
        # fractions of TOTAL (comparable with coverage)
        "confidently_wrong": _pct(wrong, total),
        "unparseable": _pct(unparse, total),
        "abs_rt_valid": coverage * of_emitted_rt,
    }


def aggregate(rows: Sequence[RowResult],
              pin_pct: Optional[float] = None) -> Dict[str, object]:
    """Aggregate row results into the corrected three-metric report under
    both matchers. ``pin_pct`` (curated-gold gate pass rate, a DIFFERENT
    population) is carried verbatim when supplied; never derived here."""
    return {
        "parity": _matcher_block(rows, "parity"),
        "strict": _matcher_block(rows, "strict"),
        "pin_pct": pin_pct,
    }
