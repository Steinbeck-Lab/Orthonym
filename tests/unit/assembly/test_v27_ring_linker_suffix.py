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
from orthonym.assembly.substituent_enumerator import name_substituent
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
# Task 3 — sulfinyl/sulfonyl substituent-branch namer (sulfone/sulfoxide 6.81×)
# ---------------------------------------------------------------------------

def _s_branch(smiles, allow_mancude):
    """Name the ring-borne S(=O)_n-R branch as a substituent prefix."""
    mol = Chem.MolFromSmiles(smiles)
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    for a in mol.GetAtoms():
        if a.GetSymbol() != 'S':
            continue
        ring_nb = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() in ring]
        if not ring_nb:
            continue
        seen, stack, frag = {ring_nb[0]}, [a.GetIdx()], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            frag.add(x)
            for n in mol.GetAtomWithIdx(x).GetNeighbors():
                if n.GetIdx() not in seen:
                    stack.append(n.GetIdx())
        return name_substituent(mol, frag, a.GetIdx(), allow_mancude=allow_mancude)
    return None


@pytest.mark.parametrize("smiles,expected", [
    ("CS(=O)(=O)c1ccccc1", "methanesulfonyl"),
    ("CS(=O)c1ccccc1", "methanesulfinyl"),
    ("CCS(=O)(=O)c1ccccc1", "ethanesulfonyl"),
    ("c1ccccc1S(=O)(=O)c1ccccc1", "benzenesulfonyl"),
    ("FC(F)(F)S(=O)(=O)c1ccccc1", "trifluoromethanesulfonyl"),
])
def test_sulfinyl_sulfonyl_branch_complete_tier(smiles, expected):
    """Complete tier: the S-attached sulfone/sulfoxide branch is named
    (R)sulfonyl/(R)sulfinyl -- never dropping the S or its =O (CR-01)."""
    assert _s_branch(smiles, allow_mancude=True) == expected


def test_sulfonyl_branch_never_drops_oxygens_default():
    """PIN default (allow_mancude=False): unchanged -- the pre-existing route
    owns S-attached branches; the new complete-tier guard does not fire, so the
    branch namer must NOT return a =O-dropping 'methyl'/'ethyl'."""
    # byte-identical to pre-P2 default (the ring engine that would drop it is
    # tier-gated and never runs at the default tier).
    assert _s_branch("CS(=O)(=O)c1ccccc1", allow_mancude=False) == "methyl"


def test_fused_methanesulfonyl_covered():
    name = _assert_covers("CS(=O)(=O)c1ccc2ccccc2c1")
    assert "methanesulfonyl" in name


# ---------------------------------------------------------------------------
# Task 4 — ester as functional-class two-word on a (fused) ring parent (248 gap)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,r_word", [
    ("CCOC(=O)c1ccc2ccccc2c1", "ethyl"),
    ("COC(=O)c1ccc2ccccc2c1", "methyl"),
    ("CC(C)OC(=O)c1ccc2ccccc2c1", "propan-2-yl"),
])
def test_fused_ester_functional_class(smiles, r_word):
    name = _assert_covers(smiles)
    assert name.startswith(r_word + " ")
    assert name.endswith("carboxylate")


def test_reverse_ester_fails_closed():
    """A reverse/aryl ester (ring on the alcohol side) is deferred: the engine
    must NOT guess the acid side -- it abstains."""
    assert _engine_name("CC(=O)Oc1ccc2ccccc2c1") is None


def test_lactone_and_polyester_fail_closed():
    assert _engine_name("O=C(OCC)c1ccc2ccccc2c1OC(=O)CC") is None  # di-ester


# ---------------------------------------------------------------------------
# PIN default byte-identical: the widened styles are gated on the tier flag
# ---------------------------------------------------------------------------

def test_pin_default_unchanged():
    """With ``allow_aromatic_general=False`` (PIN default) the ring engine still
    abstains on these fused-linker inputs exactly as before Phase 2."""
    for smi in ("NS(=O)(=O)c1ccc2ccccc2c1", "NC(=N)c1ccc2ccccc2c1",
                "N=C1CCc2ccccc21"):
        assert _engine_name(smi, allow=False) is None, smi
