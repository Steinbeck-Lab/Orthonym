""" CP2b -- carry letter fusion locants through the CATALOG fused branch.

Root-cause (sibling of CP2): ``spiro.py::_name_fused_component`` has two
branches. CP2 fixed the SYSTEMATIC branch (``name_ortho_fused_bicyclic`` ->
``_synthesize_fused_locants``) to carry letter fusion locants (``4a``/``8a``) as
strings, so a substituent on a saturated-ortho-fused component's fusion atom is
cited ``8a-methyl…`` and the molecule names. The CATALOG branch
(``match_fused_heterocycle_core``) still COERCED a letter locant to a bare int
(``'9a' -> 9``), so a catalog-matched fused heterocycle (quinolizidine,
indolizidine, quinoline, …) used as a spiro component with a substituent on its
ring-fusion atom got a WRONG integer locant, OPSIN rejected it, and the molecule
abstained (suppressed the wrong-molecule candidate -> ``unknown organic
compound``).

The fix mirrors CP2's systematic-branch treatment: the catalog branch carries
the letter locant (str, e.g. ``'9a'``) for SUBSTITUENT citation, while the
spiro-descriptor locant stays an INTEGER peripheral position -- a spiro junction
that can only be numbered as a ring-fusion atom is VOIDed (fail-closed) by the
existing guard in ``name_mixed_spiro_fused`` (OPSIN rejects a lettered locant in
the spiro slot). 0-wrong is absolute; every emission is round-trip gated.

Governing rules: / (fusion-position letter locants),
 (spiro-component citation + low locants to the spiro atoms),
the Blue Book (fusion-position letter locants). OPSIN 2.9.0-verified.

VERIFIED witness: ``CC12CCCCN2CC2(CC1)OC2`` (quinolizidine spiro-oxirane, methyl
on the 9a ring-fusion carbon) abstained on HEAD; the fix names it to an
OPSIN-parseable name citing ``9a`` that round-trips to the input constitution.
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym, name_compound
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


# Safe best-effort tier: RT-gated (unverified=False), so an emission is never a
# wrong molecule -- the exact tier the CP2b brief specifies.
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


# --- the verified witness: catalog fused heterocycle spiro component, methyl on
# the ring-fusion carbon (9a) -----------------------------------------------
# oxirane sorts alphabetically before quinolizidine, so quinolizidine is the
# PRIMED (second-cited) component and its fusion locant is carried as a primed
# tuple ('9a', "'"); the undecorated core names spiro[oxirane-2,3'-quinolizidine].
WITNESS_SMILES = "CC12CCCCN2CC2(CC1)OC2"
WITNESS_TARGET = "9a'-methylspiro[oxirane-2,3'-quinolizidine]"


@pytest.mark.opsin_gate
def test_witness_names_9a_methyl_and_round_trips():
    """The methyl on the 9a ring-fusion carbon is cited with the LETTER locant
    (``9a``), not the coerced integer ``9`` (a peripheral CH2, a different
    molecule). The emitted name round-trips to the input constitution."""
    name = _BE.name(WITNESS_SMILES)
    assert not _is_fail(name), name
    # letter locant present (prime-placement agnostic):
    assert "9a" in name.replace("'", ""), name
    # 0-wrong: full-InChI round-trip to the input
    assert _rt_ok(WITNESS_SMILES, name), name
    # byte-identity to the OPSIN-verified emission (pinned AFTER RT-confirmation)
    assert name == WITNESS_TARGET, name


@pytest.mark.opsin_gate
def test_witness_deterministic_across_atom_orders():
    """Numbering must be canonical-rank driven, not atom-input-order driven."""
    m = Chem.MolFromSmiles(WITNESS_SMILES)
    assert m is not None
    orders = {Chem.MolToSmiles(m)}
    for _ in range(8):
        orders.add(Chem.MolToSmiles(m, doRandom=True))
    seen = {_BE.name(smi) for smi in orders}
    assert seen == {WITNESS_TARGET}, seen


# --- 0-wrong sweep: catalog fused heterocycle spiro components ---------------
# A wider batch of decorated / plain catalog-fused-heterocycle spiro molecules.
# Every one must either RT-verify (right molecule) or abstain -- NEVER a wrong
# molecule. This covers the VOID rule generically: a spiro junction that can only
# be a ring-fusion atom (lettered) must fail closed, never emit a fake integer.
ZERO_WRONG_SWEEP = [
    "CC12CCCCN2CC2(CC1)OC2",   # the witness (9a-methyl quinolizidine spiro-oxirane)
    "CC12CCCCN2CCC1CC1(CC1)",  # methyl on a fused-heterocycle fusion carbon (variant)
    "O1CC12CCCN1CCCCC1C2",     # spiro of a catalog fused heterocycle, plain
    "C1CN2CCCCC2C11CCC1",      # indolizidine/quinolizidine-shaped spiro
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ZERO_WRONG_SWEEP)
def test_zero_wrong_sweep_rt_or_abstain(smiles):
    name = _BE.name(smiles)
    assert _is_fail(name) or _rt_ok(smiles, name), name


# --- PIN / no-regression controls: byte-identical to HEAD -------------------
# Captured on HEAD 5b0562dd before the fix (scratchpad probe_cp2b_head.py). The
# undecorated catalog core spiro[oxirane-2,3'-quinolizidine] is the load-bearing
# control -- it exercises the SAME catalog branch but carries no decoration on a
# fusion atom, so it must stay byte-identical.
PIN_CONTROLS = {
    # systematic-branch sibling (CP2) -- must stay working:
    "CC12CCCCC2CCC2(N1)OC2": "8a-methylspiro[decahydroquinoline-2,2'-oxirane]",
    # catalog fused heterocycle, plain substituent (not on a fusion atom):
    "CC12CCCCN1CCCC2": "9a-methylquinolizidine",
    # undecorated catalog-branch spiro core (SAME code path, no fusion decoration):
    "O1CC11CCC2CCCCN2C1": "spiro[oxirane-2,3'-quinolizidine]",
}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", sorted(PIN_CONTROLS.items()))
def test_pin_controls_byte_identical(smiles, expected):
    assert _BE.name(smiles) == expected
    # default tier must match too (the fix is scoped to the catalog fused path).
    assert Orthonym().name(smiles) == expected


@pytest.mark.opsin_gate
def test_trivial_controls():
    assert name_compound("CCO") == "ethanol"
    assert name_compound("c1ccccc1") == "benzene"
