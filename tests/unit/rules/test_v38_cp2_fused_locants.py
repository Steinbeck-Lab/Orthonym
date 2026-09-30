""" CP2 -- fused-atom numbering in the spiro-fused composer.

Root-cause: ``spiro.py::_synthesize_fused_locants`` numbered a saturated
ortho-fused bicyclic component with a plain sequential integer ``1..N``, so the
two ring-fusion atoms of a decalin got the OPSIN-invalid locants ``9``/``10``.
Any decoration whose numbering direction forces a citation onto a fusion atom
then produced an OPSIN-unparseable name and the molecule abstained at every
tier.

The fix numbers the component per /: peripheral atoms get
integers, each fusion atom gets a letter locant ``"{preceding-peripheral}a"``
(decalin -> ``4a``/``8a``), and the numbering orientation is chosen by lowest
locants to the cited positions (spiro atom + substituents), deterministically
on RDKit canonical rank. The spiro-descriptor locant stays an integer peripheral
position (OPSIN rejects a lettered locant there); a spiro atom that can only land
on a fusion carbon is VOIDed (fail closed), never emitted with a faked integer.

Governing rules: (von Baeyer / fused parent hydride numbering),
 (lowest-locants orientation), (spiro-component citation +
low locants to the spiro atoms), the Blue Book (fusion-position letter
locants),:3755 (``naphthalene-4a,8a-diol (PIN)``). OPSIN-verified targets in
the CP2 build-grounding doc.

All emissions are round-trip gated (``general_fallback_unverified=False``), so an
emitted name is always the right molecule; the tests below additionally assert
byte-identity to the OPSIN-verified PIN targets and full-InChI round-trip.
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym, name_compound
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "O=C1NC(=O)C2(CCCCC2)C(=O)N1",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)



# Safe best-effort tier: RT-gated (unverified=False), so an emission is never a
# wrong molecule -- the exact tier the CP2 brief specifies.
_BE = Orthonym(
    general_fallback=True,
    general_fallback_unverified=False,
    allow_aromatic_general=True,
)


def _is_fail(name) -> bool:
    from orthonym.errors import is_failure_name
    return is_failure_name(name)


def _rt_ok(smiles: str, name: str) -> bool:
    return bool(opsin_roundtrip_check(smiles, name)["passed"])


# --- the two minimal reproducers, OPSIN-verified PIN targets ----------------

REPRO1_SMILES = "CC12CCC3(OC3)CC2CCCC1"
REPRO1_TARGET = "4a-methylspiro[decahydronaphthalene-2,2'-oxirane]"

REPRO2_SMILES = "CC12CCCCC2CCC2(OC2)C1"
REPRO2_TARGET = "8a-methylspiro[decahydronaphthalene-2,2'-oxirane]"


@pytest.mark.opsin_gate
def test_repro1_names_4a_methyl():
    """Methyl on the fusion carbon nearer position 4 -> ``4a``."""
    name = _BE.name(REPRO1_SMILES)
    assert name == REPRO1_TARGET, name
    assert _rt_ok(REPRO1_SMILES, name)


@pytest.mark.opsin_gate
def test_repro2_names_8a_methyl():
    """Methyl on the fusion carbon reached at ``8a`` when the spiro atom pins 2."""
    name = _BE.name(REPRO2_SMILES)
    assert name == REPRO2_TARGET, name
    assert _rt_ok(REPRO2_SMILES, name)


@pytest.mark.opsin_gate
@pytest.mark.parametrize(
    "smiles,target",
    [(REPRO1_SMILES, REPRO1_TARGET), (REPRO2_SMILES, REPRO2_TARGET)],
)
def test_reproducers_deterministic_across_atom_orders(smiles, target):
    """Numbering must be canonical-rank driven, not atom-input-order driven.

     hit atom-order bugs 3x; the fix must yield the identical name from
    several randomized SMILES writings of the same molecule.
    """
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    orders = {Chem.MolToSmiles(m)}
    for _ in range(8):
        orders.add(Chem.MolToSmiles(m, doRandom=True))
    seen = set()
    for smi in orders:
        seen.add(_BE.name(smi))
    assert seen == {target}, seen


# --- PIN / no-regression controls: byte-identical to HEAD -------------------
# Captured on HEAD 224cb242 before the fix (a temp dir cp2_baseline.py). The
# undecorated core is the load-bearing control: it exercises the SAME synthetic
# fused-locant path but carries no decoration, so it must stay byte-identical.

PIN_CONTROLS = {
    "C1C2(CO2)CCC3CCCCC13": "spiro[decahydronaphthalene-2,2'-oxirane]",
    "C1CCC2(CC1)CCCC2": "spiro[4.5]decane",
    "C1CCC2CCCCC2C1": "decahydronaphthalene",
    "C1CC2CCC1CC2": "bicyclo[2.2.2]octane",
    "c1ccc2ccccc2c1": "naphthalene",
    "O=C1CCCCC1": "cyclohexanone",
    "O=C1NC(=O)C2(CCCCC2)C(=O)N1": "1,3,5-trioxo-2,4-diazaspiro[5.5]undecane",
}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", sorted(PIN_CONTROLS.items()))
def test_pin_controls_byte_identical(smiles, expected):
    assert _BE.name(smiles) == expected
    # default tier must match too (the fix is scoped to the mixed-spiro path).
    assert _dt_name(smiles) == expected


@pytest.mark.opsin_gate
def test_trivial_controls():
    assert name_compound("CCO") == "ethanol"
    assert name_compound("c1ccccc1") == "benzene"


# --- complex census witnesses: RT-verify OR abstain, never wrong ------------

WITNESSES = {
    "W1": "CC(=O)OC[C@@]12[C@@H](OC(C)=O)C[C@@H](C)[C@](C)([C@@H]3C[C@H]4CCO[C@H]4O3)[C@H]1CC[C@H](O)[C@]21CO1",
    "W5": "CC(=O)OC[C@@]12[C@@H](OC(C)=O)C[C@@H](C)[C@](C)([C@@H]3C[C@H]4CCO[C@H]4O3)[C@H]1CCC[C@]21CO1",
    "W7": "CC(=O)OC[C@]12[C@H](OC(C)=O)C[C@H](C)[C@@](C)([C@H]3C[C@@H]4C=CO[C@@H]4O3)[C@@H]1CC[C@H](OC(C)=O)[C@]21CO1",
}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("label,smiles", sorted(WITNESSES.items()))
def test_witnesses_rt_verify_or_abstain(label, smiles):
    """A complex witness may still abstain for unrelated reasons; that is
    acceptable. What is NOT acceptable is a wrong-molecule emission."""
    name = _BE.name(smiles)
    assert _is_fail(name) or _rt_ok(smiles, name), (label, name)


# --- direct unit test of the numbering supplier -----------------------------

def test_synthesize_fused_locants_decalin_letters():
    """A bare decalin fragment: fusion atoms -> ``4a``/``8a`` (letter locants),
    peripheral -> ``1..8``. NEVER the old invalid ``9``/``10``."""
    from orthonym.rules.spiro import _synthesize_fused_locants

    frag = Chem.MolFromSmiles("C1CCC2CCCCC2C1")
    assert frag is not None
    locs = _synthesize_fused_locants(frag)
    assert locs, "expected a non-empty locant map for decalin"

    ring_info = frag.GetRingInfo()
    fusion_atoms = {
        a.GetIdx()
        for a in frag.GetAtoms()
        if ring_info.NumAtomRings(a.GetIdx()) == 2
    }
    assert len(fusion_atoms) == 2

    int_locants = {v for v in locs.values() if isinstance(v, int)}
    str_locants = {v for v in locs.values() if not isinstance(v, int)}
    assert int_locants == {1, 2, 3, 4, 5, 6, 7, 8}, int_locants
    assert str_locants == {"4a", "8a"}, str_locants
    # every invalid out-of-range integer is gone
    assert 9 not in int_locants and 10 not in int_locants
    # the two fusion atoms are exactly the two letter-locant atoms
    assert {a for a, v in locs.items() if not isinstance(v, int)} == fusion_atoms


# --- review FINDING 1: primed-fused-component decoration ---------------------
# When the decalin is the PRIMED spiro component (the side ring sorts
# alphabetically before ``decahydronaphthalene``) AND a fusion atom bears a
# substituent, that fusion locant is carried as a primed tuple (``('4a', "'")``).
# This exercises the composer's mixed int / '4a'-str / (int, primes)-tuple locant
# ordering (``_enrich_complex_ring_with_subs._orient_key``). Both witnesses must
# RT-verify (0-wrong), be atom-order deterministic, and cite DISTINCT fusion
# locants (the '4a' vs '8a' carbon), spelling-agnostic to prime placement.

PRIMED_FUSED_WITNESSES = {
    "CC12CCC3(CC3)CC2CCCC1": "4a",   # methyl on the fusion carbon nearer 4
    "CC12CCCCC2CCC2(CC2)C1": "8a",   # methyl on the fusion carbon reached at 8a
}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,fusion", sorted(PRIMED_FUSED_WITNESSES.items()))
def test_primed_fused_component_decoration_rt(smiles, fusion):
    name = _BE.name(smiles)
    assert not _is_fail(name), name
    assert _rt_ok(smiles, name), name
    # the primed fusion locant is cited (strip primes so this does not pin the
    # 4a' vs 4'a prime-placement spelling):
    assert fusion in name.replace("'", ""), (fusion, name)


@pytest.mark.opsin_gate
def test_primed_fused_witnesses_distinct_and_deterministic():
    names = {}
    for smiles in PRIMED_FUSED_WITNESSES:
        m = Chem.MolFromSmiles(smiles)
        orders = {Chem.MolToSmiles(m)}
        for _ in range(6):
            orders.add(Chem.MolToSmiles(m, doRandom=True))
        seen = {_BE.name(s) for s in orders}
        assert len(seen) == 1, (smiles, seen)   # atom-order deterministic
        names[smiles] = seen.pop()
    # the two distinct fusion positions must give two distinct names
    assert len(set(names.values())) == 2, names
