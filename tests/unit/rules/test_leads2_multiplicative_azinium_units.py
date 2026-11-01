"""LEADS2 topic 5: multiplicative names whose identical units are N-substituted azinium rings.

``## **** Assemblies derived from parent cations`` (the Blue Book),:42113: "Cationic compounds with
cationic centers derived from the same parent hydride, but located in different parts of a structure, are named, if
possible, according to the principles of multiplicative nomenclature (see, using the multiplying prefixes
'bis-', 'tris-', etc. where necessary." The book prints '4,4'-(ethane-1,2-diyl)bis(1-methylpyridin-1-ium) (PIN)'
(:42130), '(1,4-phenylene)bis(phosphanium) (PIN)' and '2,2'-(1,3-phenylene)di(propan-2-ylium) (PIN)'.

The general multiplicative detector (``rules.multiplicative_general``) declined every molecule with a charge but the nitro
group (``_scope_ok``), so the PIN tier named none of these and the wider tiers gave the substitutive name of the floor
('1-methyl-4-[2-(1-methylpyridin-1-ium-4-yl)ethyl]pyridin-1-ium'). With the N-substituted azinium atom admitted the
detector builds the unit from the charged ring ('pyridin-1-ium', ``_ring_parent_name``), cites the group on the nitrogen
as a prefix of the unit and multiplies it: the name of the book. Scope: the pyridinium class only; the carbocation row
('di(propan-2-ylium)') and the phosphanium row stay with their namers (they need another unit kind, listed as leads).

Every expected name is read back by a fresh OPSIN call (``assert_pin_at_both_tiers``); the first row is the book's own.
"""
import pytest

from orthonym.rules import multiplicative_general
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

ROWS = [
    ("C[n+]1ccc(CCc2cc[n+](C)cc2)cc1", "4,4'-(ethane-1,2-diyl)bis(1-methylpyridin-1-ium)"),     #:42130 verbatim
    ("C[n+]1ccc(Cc2cc[n+](C)cc2)cc1", "4,4'-methylenebis(1-methylpyridin-1-ium)"),
    ("C[n+]1cccc(Cc2ccc[n+](C)c2)c1", "3,3'-methylenebis(1-methylpyridin-1-ium)"),
    ("C[n+]1ccc(Oc2cc[n+](C)cc2)cc1", "4,4'-oxybis(1-methylpyridin-1-ium)"),
    ("C[n+]1ccc(Sc2cc[n+](C)cc2)cc1", "4,4'-sulfanediylbis(1-methylpyridin-1-ium)"),
    ("CC[n+]1ccc(CCc2cc[n+](CC)cc2)cc1", "4,4'-(ethane-1,2-diyl)bis(1-ethylpyridin-1-ium)"),
    ("Cc1cc(Cc2cc(C)[n+](C)cc2)cc[n+]1C", "4,4'-methylenebis(1,2-dimethylpyridin-1-ium)"),
    ("C[n+]1ccc(CCCCc2cc[n+](C)cc2)cc1", "4,4'-(butane-1,4-diyl)bis(1-methylpyridin-1-ium)"),
]


@pytest.mark.parametrize("smiles,pin", ROWS)
def test_identical_azinium_units_are_multiplied(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles", [
    "C[n+]1ccc(Cc2cc[n+](CC)cc2)cc1",                        # the units differ (N-methyl, N-ethyl)
    "C[n+]1ccc(Cc2ccncc2)cc1",                               # one unit is a pyridine
    "C[n+]1ccc(-c2cc[n+](C)cc2)cc1",                         # joined by a bond: a ring assembly, not this
    "C[n+]1ccc(C(=O)OCCOC(=O)c2cc[n+](C)cc2)cc1",            # an ester linker
    "[O-][n+]1ccc(Cc2cc[n+]([O-])cc2)cc1",                   # N-oxides: the charge on the oxygen is not read
    # fix a performance pass (review C2): a cation ranks above acids, alcohols and amines, the Blue Book), so
    # these are bis(pyridin-1-ium) parents with a prefix on the nitrogen, which the units of the detector do not cite;
    # the detector read the alcohol as the parent ('2,2'-[methylenedi(pyridin-1-ium-4,1-diyl)]di(ethan-1-ol)')
    "OCC[n+]1ccc(Cc2cc[n+](CCO)cc2)cc1",
    "OC(=O)CC[n+]1ccc(Cc2cc[n+](CCC(O)=O)cc2)cc1",
    "NCC[n+]1ccc(Cc2cc[n+](CCN)cc2)cc1",
])
def test_a_molecule_outside_the_class_is_not_multiplied(smiles):
    assert name_default(smiles).get("producer") != "outer:multiplicative"
    assert name_breadth(smiles).get("producer") != "outer:multiplicative"


def test_without_the_azinium_atom_the_charge_declines(monkeypatch):
    """Mutation: with the azinium atom not read the detector declines the molecule again (PIN tier: no name)."""
    monkeypatch.setattr(multiplicative_general, "_is_azinium_n", lambda atom: False)
    row = name_default("C[n+]1ccc(CCc2cc[n+](C)cc2)cc1")
    assert row.get("producer") != "outer:multiplicative" and row.get("tier") != "pin_verified", row


@pytest.mark.parametrize("smiles", [
    "OCC[n+]1ccc(Cc2cc[n+](CCO)cc2)cc1",
    "OC(=O)CC[n+]1ccc(Cc2cc[n+](CCC(O)=O)cc2)cc1",
    "NCC[n+]1ccc(Cc2cc[n+](CCN)cc2)cc1",
])
def test_a_suffix_group_beside_the_cations_is_not_made_the_parent(smiles):
    """The PIN tier declines (no verified PIN is built) and the wider tier keeps a name that is no PIN."""
    assert name_default(smiles).get("tier") != "pin_verified"
    be = name_breadth(smiles)
    assert be.get("tier") != "pin_verified" and "di(ethan-1-ol)" not in be["name"], be
