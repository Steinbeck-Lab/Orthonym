""" spiro-hoist — cross-boundary substituent HOISTING made locant-correct.

 finding (a project rule, corenum-premise re-scope): cross-spiro-boundary
substituent hoisting ALREADY EXISTS (primed component substituents like
``3',6'-dihydroxy`` on the xanthene half emit correctly via
``composer._integrate_universal_prefixes`` consuming the ``combined_locants``
map). The corenum report's "missing hoisting" premise is REFUTED.

The real blocker for the component-spiro path (``_name_spiro_vonbaeyer_core``)
was a NUMBERING-CONSISTENCY bug: the spiro DESCRIPTOR locant comes from
``_canonical_spiro_locant`` (the lowest locant over the spiro atom's symmetry
orbit — e.g. 1 for the ``1,3-dihydro-2-benzofuran`` half of fluorescein) but the
SUBSTITUENT-hoisting locmap came straight from the fused-ring catalog with the
spiro atom at a DIFFERENT locant (3). The two disagreed, so the carbonyl was
cited ``1-oxo`` — colliding with ``spiro[...-1,...]`` — and the whole name was
OPSIN-unparseable. Fix: re-anchor the component locmap (via the fragment's own
graph automorphism) so ``locmap[spiro] == descriptor locant``; the carbonyl then
falls on 3 and the ring carboxy on 5, and fluorescein RT-verifies.

This is a strict improvement: the re-anchor fires ONLY when the two locants
disagree, and a currently-EMITTING substituted spiro-VB must already agree (else
it would be unparseable, like fluorescein was), so no emitted canary changes.
Every emission is still RT-gated (0-wrong).
"""
import pytest

from orthonym import errors
from orthonym.namer import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


def _be():
    # best-effort tier: exercises the full hoisting cascade
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


def _rt_full(smi: str, name) -> bool:
    """Full-InChIKey round-trip (constitution AND stereo)."""
    if not name or errors.is_failure_name(name):
        return False
    r = opsin_roundtrip_check(smi, name)
    return bool(r.get("passed") and r.get("inchi_match"))


# --- LANDED: fluorescein / xanthene spiro-benzofuranone dye family -----------
# Stereo-free spiro-of-fused whose ONLY blocker was the descriptor/locmap
# numbering inconsistency. RT-full after the re-anchor.
HOIST_TARGETS = [
    # fluorescein (carboxy form)
    "O=C(O)c1ccc2c(c1)C(=O)OC21c2ccc(O)cc2Oc2cc(O)ccc21",
    # 6-aminofluorescein
    "Nc1ccc2c(c1)C(=O)OC21c2ccc(O)cc2Oc2cc(O)ccc21",
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", HOIST_TARGETS)
def test_fluorescein_family_rt_full(smi):
    name = _be().name(smi)
    assert name and not errors.is_failure_name(name), f"abstained: {name!r}"
    assert "spiro[" in name, f"not a component-spiro name: {name!r}"
    assert _rt_full(smi, name), f"does not RT-full: {name!r}"


@pytest.mark.opsin_gate
def test_fluorescein_deterministic():
    """Atom-order independence: two SMILES orderings -> identical name."""
    from rdkit import Chem
    smi = HOIST_TARGETS[0]
    alt = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    n1 = _be().name(smi)
    n2 = _be().name(alt)
    assert n1 == n2, f"determinism: {n1!r} != {n2!r}"
    assert _rt_full(smi, n1)


# --- CANARIES: clean spiro-VB names must stay byte-identical -----------------
CANARIES = [
    ("c1ccc2c(c1)-c1ccccc1C21SC2CCC1CC2",
     "3-thiaspiro[bicyclo[2.2.2]octane-2,9'-fluorene]"),
    ("C1COCC2(C1)C1CCC2CC1", "spiro[bicyclo[2.2.1]heptane-7,3'-oxane]"),
    ("C1CCCCOC2(CC3CCC2C3)OCCCC1",
     "2',12'-dioxaspiro[bicyclo[2.2.1]heptane-2,1'-cyclododecane]"),
    ("O1CC2CNC1C23CCC3",
     "2-oxa-6-azaspiro[bicyclo[2.2.1]heptane-7,1'-cyclobutane]"),
]


@pytest.mark.parametrize("smi,expected", CANARIES)
def test_clean_spiro_vb_canary_byte_identical(smi, expected):
    got = Orthonym().name(smi)
    assert got == expected, f"CANARY REGRESSION {smi}: {got!r} != {expected!r}"


# --- NAMED 0-WRONG BLOCKER (RESOLVED by CP2): decorated spiro-of-fused ----
# The dominant real residual was decorated spiro-of-fused TERPENOIDS (spiro-epoxide
# on an acylated decalin), MULTI-BLOCKED (a project rule) by:
# (a) the systematic decalin component being numbered plain 1..10 (no 4a/8a fusion
# locants) so the descriptor/substituent locants were OPSIN-invalid, AND
# (b) the complex_ring path being stereo LOG-ONLY.
# CP2 (fused-atom numbering, '4a'/'8a' fusion locants +
# _enrich_complex_ring_with_subs mixed-locant plumbing) closed BOTH: this witness
# now names a determinate, full-InChIKey-RT-verified best-effort name (0-wrong).
@pytest.mark.opsin_gate
def test_decorated_terpenoid_spiro_epoxide_named_blocker():
    smi = "CC(=O)OC[C@@]12[C@@H](OC(C)=O)C[C@@H](C)[C@](C)([C@@H]3C[C@H]4CCO[C@H]4O3)[C@H]1CC[C@H](O)[C@]21CO1"
    name = _be().name(smi)
    assert _rt_full(smi, name), f"named blocker now lands: {name!r}"
