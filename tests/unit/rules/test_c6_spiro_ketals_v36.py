""" Milestone-C Wave D (phase C6) — component spiro / spiro-ketal.

Grounding trace: internal notes

The deliverable class this file locks is the ONE clean, RT-verified gap the
trace found: a monospiro system where one component is a von-Baeyer CAGE and the
OTHER is a SATURATED heteromonocycle too large for a Hantzsch-Widman stem
(ring size > 10). Such a component must be named by skeletal-replacement ('a')
nomenclature per

     (the Blue Book): "In the case of ring systems modified by skeletal
    replacement ('a') nomenclature, and are applied to name the
    ring system before skeletal replacement ('a') nomenclature is applied as
    described in."
    : "...at least one ring component requiring the use of skeletal
    replacement ('a') nomenclature are named as in; then, the skeletal
    replacement ('a') prefixes are introduced and cited before the 'spiro' term."

i.e. the large heteromonocycle is named as its all-carbon parent (`cyclododecane`)
inside the spiro bracket, and its ring heteroatoms are hoisted to a front
`2',12'-dioxa`-style 'a'-prefix. Before this fix the tree emitted the malformed,
OPSIN-unparseable `spiro[[1,3]dioxa-2,2'-bicyclo[2.2.1]heptane]` (parent stem
dropped) and abstained.

Every assertion is round-trip gated (name -> OPSIN -> InChIKey == input): a wrong
construction cannot pass, it can only abstain. Canaries lock the pre-existing spiro
/ spiro-VB / spiro-ketal outputs byte-for-byte so the shared dispatch is unmoved.
"""

import pytest
from rdkit import Chem
from rdkit import RDLogger

from orthonym.wallclock import wall_clock_limit

RDLogger.DisableLog("rdApp.*")


def _name(smiles: str) -> str:
    from orthonym import name_compound

    with wall_clock_limit(60):
        res = name_compound(smiles)
    return res.name if hasattr(res, "name") else str(res)


def _rt_ok(smiles: str, name: str) -> bool:
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

    return opsin_roundtrip_check(smiles, name)["passed"]


# /.4 skeletal-replacement of a large (>10) heteromonocyclic spiro
# component joined to a von-Baeyer cage. (smiles, exact expected PIN-shaped name)
# Both round-trip to the input skeleton (verified independently via opsin_parse).
FIX_TARGETS = [
    # W14 — the Blue Book PIN example: 12-membered 2,12-dioxa ring + bicyclo[2.2.1]heptane
    ("C1CCCCOC2(CC3CCC2C3)OCCCC1",
     "2',12'-dioxaspiro[bicyclo[2.2.1]heptane-2,1'-cyclododecane]"),
    # 11-membered oxa ring + bicyclo[2.2.1]heptane. The spiro atom of the
    # monocyclic component takes the LOW locant 1' example explanation:
    # "The spiro atom of the monocyclic hydrocarbon component is given preference
    # for low locant."), so the oxa follows at 2'.
    ("C1CCCCOC2(CCCC1)CC1CCC2C1",
     "2'-oxaspiro[bicyclo[2.2.1]heptane-2,1'-cycloundecane]"),
]

# Pre-existing outputs that MUST stay byte-identical (shared spiro dispatch,
# incl. the two spiro-VB canaries that traverse the exact modified path).
CANARIES = [
    ("C1CCC2(CC1)CCCCC2", "spiro[5.5]undecane"),
    ("C1CCC2(CC1)CCCC2", "spiro[4.5]decane"),
    ("C1CCC2(CC1)OCCO2", "1,4-dioxaspiro[4.5]decane"),                       # spiro-ketal
    ("c1ccc2c(c1)-c1ccccc1C21SC2CCC1CC2",
     "3-thiaspiro[bicyclo[2.2.2]octane-2,9'-fluorene]"),                     # spiro-VB path
    ("C1COCC2(C1)C1CCC2CC1", "spiro[bicyclo[2.2.1]heptane-7,3'-oxane]"),     # spiro-VB path
    ("C1CCC2(CC1)CCCCO2", "1-oxaspiro[5.5]undecane"),
    ("c1ccc2ccccc2c1", "naphthalene"),
]


@pytest.mark.parametrize("smiles,expected", FIX_TARGETS)
def test_c6_skeletal_replacement_monocycle_spiro(smiles, expected):
    got = _name(smiles)
    assert got == expected, f"{smiles}: got {got!r}, expected {expected!r}"
    assert _rt_ok(smiles, got), f"{smiles}: {got!r} did not round-trip"


@pytest.mark.parametrize("smiles,expected", CANARIES)
def test_c6_canaries_unchanged(smiles, expected):
    got = _name(smiles)
    assert got == expected, f"CANARY REGRESSION {smiles}: got {got!r}, expected {expected!r}"
    assert _rt_ok(smiles, got), f"{smiles}: canary {got!r} did not round-trip"
