"""v27 Phase 2 — linker & ring-suffix nomenclature (general ring engine).

Widens ``general_engine._RING_SUFFIX_STYLES`` and its emit logic so the
complete-tier ring engine expresses the high-enrichment linker/suffix groups
that previously forced abstention: sulfonamide (∞ blind spot), amidine, imine,
N-substituted amide/sulfonamide, sulfinyl/sulfonyl branch, ester (functional
class), urea/carbamate. Every emission is verified by SELF-01 round-trip
downstream; these tests assert the engine (a) emits a name and (b) that name
round-trips to the input via OPSIN — the actual complete-tier contract
(RT-valid breadth, NOT PIN-string exactness; PIN-quality fused-ring names are
Phase 5). The PIN default (``allow_aromatic_general=False``) must stay
byte-identical — pinned in ``test_pin_default_unchanged``.
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.assembly import general_engine as ge
from orthonym.validation.opsin_roundtrip import opsin_parse


_NAMER = Orthonym(general_fallback=True, allow_aromatic_general=True)


def _engine_name(smiles, allow=True):
    """Run perception+classification then the general ring engine (complete
    tier by default) and return the emitted name string or None."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    canonical = Chem.MolToSmiles(mol)
    feats = _NAMER._perceive(mol, smiles, canonical)
    _NAMER._classify(feats)
    res = ge.name_general(mol, feats, allow_aromatic_general=allow)
    return res.name if res else None


def _round_trips(name, ref_smiles):
    """True iff OPSIN parses ``name`` to a structure identical to ``ref``."""
    if not name:
        return False
    smi = opsin_parse(name)
    if not smi:
        return False
    try:
        return Chem.CanonSmiles(smi) == Chem.CanonSmiles(ref_smiles)
    except Exception:
        return False


def _assert_covers(smiles):
    """The engine emits a name that round-trips to the input (RT-valid)."""
    name = _engine_name(smiles)
    assert name is not None, f"engine abstained for {smiles}"
    assert _round_trips(name, smiles), f"{name!r} does not round-trip to {smiles}"
    return name


# ---------------------------------------------------------------------------
# Task 1 — sulfonamide ring suffix (the ∞ blind spot)
# ---------------------------------------------------------------------------

def test_fused_sulfonamide_covered():
    """naphthalene-2-sulfonamide: engine emits an RT-valid ring-sulfonamide
    (von-Baeyer form pending Phase-5 fusion names)."""
    name = _assert_covers("NS(=O)(=O)c1ccc2ccccc2c1")
    assert name.endswith("sulfonamide")


def test_monocycle_disulfonamide_covered():
    name = _assert_covers("NS(=O)(=O)c1ccc(S(N)(=O)=O)cc1")
    assert "disulfonamide" in name


# ---------------------------------------------------------------------------
# Task 5 — amidine (carboximidamide) + imine ring suffixes on fused/cage parents
# ---------------------------------------------------------------------------

def test_fused_amidine_covered():
    name = _assert_covers("NC(=N)c1ccc2ccccc2c1")
    assert name.endswith("carboximidamide")


def test_fused_imine_covered():
    _assert_covers("N=C1CCc2ccccc21")


def test_monocycle_imine_unchanged():
    """cyclohexan-1-imine already worked on monocycles — must stay byte-exact."""
    assert _engine_name("N=C1CCCCC1") == "cyclohexan-1-imine"


# ---------------------------------------------------------------------------
# PIN default byte-identical: the widened styles are gated on the tier flag
# ---------------------------------------------------------------------------

def test_pin_default_unchanged():
    """With ``allow_aromatic_general=False`` (PIN default) the ring engine still
    abstains on these fused-linker inputs exactly as before Phase 2."""
    for smi in ("NS(=O)(=O)c1ccc2ccccc2c1", "NC(=N)c1ccc2ccccc2c1",
                "N=C1CCc2ccccc21"):
        assert _engine_name(smi, allow=False) is None, smi
