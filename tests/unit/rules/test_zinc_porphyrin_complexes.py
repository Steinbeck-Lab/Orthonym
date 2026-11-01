"""Porphyrin complexes of zinc: the dianion of the ligand is named on the porphyrin parent
(``MACROCYCLE`` of ``data/metal_scope.py``, ``rules/porphyrin.name_porphyrin_dianion``,
``rules/metal_macrocycles``, read by ``rules/salts.name_salt`` and ``rules/adducts.name_adduct``).

The ionic and the bonded drawings of zinc tetraphenylporphyrin and of zinc porphine cite the ligand on the
porphyrin parent that ``rules/porphyrin.py`` names (``zinc 5,10,15,20-tetraphenylporphyrin-21,23-diide``), where
the general engine cites it as a von Baeyer cage (``zinc 2,7,12,17-tetraphenyl-21,22,23,24-tetraazapentacyclo[...]
-21,23-diide``); the free base beside ``[Zn+2]`` is an adduct of the porphyrin parent
(``5,10,15,20-tetraphenylporphyrin—zinc(2+) (1/1)``). The dianion is named on the parent hydride too:

* Anions derived from parent hydrides and their derivatives (the Blue Book-40902): "An anion
  derived formally by the removal of one or more hydrons from any position of a neutral parent hydride is
  preferably named by using the suffix '-ide'... Numerical prefixes 'di', 'tri', etc. are used to denote
  multiplicity; locants identify positions of the negative charges." The parent hydride is ``porphyrin``
  , whose name implies the N-H on N21 and N23 (.1), so the dianion is
  ``porphyrin-21,23-diide`` and its substituent prefixes are cited before it.
* The name has no PIN: "Preferred IUPAC names (PINs) are not identified for the compounds in this Chapter"
   INTRODUCTION,:50943), and "Coordination nomenclature is the primary nomenclature method used to name
  organometallic compounds containing elements of Groups 3 through 12",:39753). The label is
  ``systematic_verified`` and the default tier declines the compound.

Every name is read back by a fresh OPSIN call that does not go through the engine, to the full standard
InChIKey of the input.
"""
import csv
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.data import metal_scope
from orthonym.data.metal_scope import ELEMENT, MACROCYCLE, METAL_SCOPE, MetalScope
from orthonym.rules import adducts, metal_labels, metal_macrocycles
from orthonym.rules.metal_labels import is_macrocycle_complex_name_without_pin
from orthonym.rules.metal_macrocycles import (
    assembly_admits_macrocycle,
    macrocycle_anion_name,
    macrocycle_component_names,
)
from orthonym.rules.porphyrin import decline_reason, name_porphyrin_dianion
from tests.support.jars import jar_or_skip
from tests.support.metal_baselines import bonded_metal_symbols
from tests.support.rt_assert import assert_full_rt, name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

BONDED_METALS = bonded_metal_symbols()          # lanes B1 and B2

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "porphyrin"

TPP_DIANION = ("C1=Cc2nc1c(-c1ccccc1)c1ccc([n-]1)c(-c1ccccc1)c1nc(c(-c3ccccc3)c3ccc([n-]3)c2-c2ccccc2)C=C1")
PORPHINE_DIANION = "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[n-]5)C=C4)[n-]3"
TPP_FREE_BASE = ("C1=Cc2nc1c(-c1ccccc1)c1ccc([nH]1)c(-c1ccccc1)c1nc(c(-c3ccccc3)c3ccc([nH]3)c2-c2ccccc2)C=C1")
PORPHINE_FREE_BASE = "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3"
# the two Zn-N drawing and the charged four Zn-N drawing of zinc tetraphenylporphyrin (panel E1)
TPP_BONDED_2 = ("C1=CC2=C(c3ccccc3)c3ccc4[n]3[Zn][n]3c(ccc3=C(c3ccccc3)C3=NC(=C4c4ccccc4)C=C3)=C(c3ccccc3)C1=N2")
TPP_BONDED_4 = ("C1=CC2=[N+]3C1=C(c1ccccc1)c1ccc4[n]1[Zn-2]31[n]3c(ccc3=C(c3ccccc3)C3=[N+]1C(=C4c1ccccc1)C=C3)"
                "=C2c1ccccc1")
PORPHINE_BONDED_2 = "C1=CC2=NC1=Cc1ccc3[n]1[Zn][n]1c(ccc1=CC1=NC(=C3)C=C1)=C2"
PHTHALOCYANINE_ZN = ("C1=CC=C2C(=C1)C3=NC4=NC(=NC5=C6C=CC=CC6=C([N-]5)N=C7C8=CC=CC=C8C(=N7)N=C2[N-]3)"
                     "C9=CC=CC=C94.[Zn+2]")
PHTHALOCYANINE_ZN_NAME = (
    "zinc 2,11,20,29,37,38,39,40-octaazanonacyclo[28.6.1.1^3,10.1^12,19.1^21,28.0^4,9.0^13,18.0^22,27.0^31,36]"
    "tetraconta-1(36),2,4,6,8,10(40),11,13,15,17,19,21(38),22,24,26,28,30,32,34-nonadecaene-37,39-diide")

# one phenyl of tetraphenylporphyrin replaced by naphthalen-1-yl, or by 8-bromoanthracen-1-yl: a decorated fused-ring
# substituent, which the substituent namer names in the best-effort call only
NAPHTHYL_DIANION = ("C1=Cc2nc1c(-c1ccccc1)c1ccc([n-]1)c(-c1ccccc1)c1nc(c(-c3cccc4ccccc34)c3ccc([n-]3)c2-c2ccccc2)"
                    "C=C1")
BROMOANTHRACENYL_DIANION = ("Brc1cccc2cc3cccc(-c4c5nc(c(-c6ccccc6)c6ccc([n-]6)c(-c6ccccc6)c6nc(c(-c7ccccc7)c7ccc4[n-]7)"
                            "C=C6)C=C5)c3cc12")

ZN_TPP_NAME = "zinc 5,10,15,20-tetraphenylporphyrin-21,23-diide"
ZN_PORPHINE_NAME = "zinc porphyrin-21,23-diide"


def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


def _view(row):
    return (row["name"], row["tier"], row["is_pin"], row["limit_code"])


def _table(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _dianion(free_base_smiles):
    """The free base minus the hydrons of its two core N-H, drawn as ``[n-]``."""
    mol = Chem.MolFromSmiles(free_base_smiles)
    nh = [a.GetIdx() for a in mol.GetAtoms()
          if a.GetAtomicNum() == 7 and a.GetIsAromatic() and a.GetTotalNumHs() == 1 and not a.GetFormalCharge()]
    assert len(nh) == 2, free_base_smiles
    rw = Chem.RWMol(mol)
    for index in nh:
        atom = rw.GetAtomWithIdx(index)
        atom.SetFormalCharge(-1)
        atom.SetNumExplicitHs(0)
    Chem.SanitizeMol(rw)
    return Chem.MolToSmiles(rw)


def _assert_named_without_a_pin(smiles, name, tiers=("best-effort", "valid")):
    for tier in tiers:
        row = _row(smiles, tier)
        assert row["name"] == name, (tier, row)
        assert row["tier"] == "systematic_verified" and row["is_pin"] is False, (tier, row)
    assert_full_rt(name, smiles)
    declined = _row(smiles, "pin")
    assert declined["tier"] == "abstain" and declined["is_pin"] is False, declined


# ---- the producer: the free base's name with the diide parent, for the whole class of the fixture

FREE_BASES = _table("substituted_porphyrins.tsv")


@pytest.mark.parametrize("row", FREE_BASES, ids=lambda r: r["id"])
def test_the_dianion_is_named_as_its_free_base_on_the_diide_parent(row):
    """The hydrons that leave are those of the implied N-H (N21 and N23,.1), so the numbering is the free
    base's and the name differs by the ending alone. Read back to the dianion's full InChIKey."""
    jar_or_skip("opsin")
    dianion = _dianion(row["smiles"])
    name = name_porphyrin_dianion(Chem.MolFromSmiles(dianion))
    assert name == row["expected_name"] + "-21,23-diide", dianion
    assert name_is_rt_exact(name, dianion), name


def test_a_bare_porphine_dianion_has_a_name_though_the_bare_free_base_is_the_catalogue_entry():
    """The 24-atom free base is named by the exact lookup of ``data/natural_products.py``; the anion has no
    entry of its own, so the producer names it."""
    assert name_porphyrin_dianion(Chem.MolFromSmiles(PORPHINE_DIANION)) == "porphyrin-21,23-diide"
    assert name_is_rt_exact("porphyrin-21,23-diide", PORPHINE_DIANION)


@pytest.mark.parametrize("smiles", [
    PORPHINE_DIANION, TPP_DIANION,
    "CCc1c(CC)c2cc3[n-]c(cc4nc(cc5[n-]c(cc1n2)c(CC)c5CC)c(CC)c4CC)c(CC)c3CC",
])
def test_the_name_is_stable_over_calls(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert name_porphyrin_dianion(mol) == name_porphyrin_dianion(Chem.MolFromSmiles(smiles))


# a resonance form of the dianion with the negative charges on neighbouring nitrogens (RDKit's resonance supplier):
# the free base it comes from would carry N-H on N21 and N22, which no numbering of ``porphyrin`` implies, and
# ``21,22-diide`` reads as another structure (the plan, section 1.4); the ligand keeps the name it has
CIS_DIANION = "C1=Cc2cc3ccc(cc4C=Cc(cc5nc(cc1n2)cc5)[n-]4)[n-]3"
TPP_WITH_CARBOXYLATE = ("C1=Cc2nc1c(-c1ccc(C(=O)[O-])cc1)c1ccc([n-]1)c(-c1ccccc1)c1nc(c(-c3ccccc3)c3ccc([n-]3)c2-c2ccccc2)"
                        "C=C1")
TPP_WITH_ACID = ("C1=Cc2nc1c(-c1ccc(C(=O)O)cc1)c1ccc([n-]1)c(-c1ccccc1)c1nc(c(-c3ccccc3)c3ccc([n-]3)c2-c2ccccc2)C=C1")
TPP_WITH_STEREOCENTRE = ("C1=Cc2nc1c([C@H](C)CC)c1ccc([n-]1)c(-c1ccccc1)c1nc(c(-c3ccccc3)c3ccc([n-]3)c2-c2ccccc2)C=C1")

DECLINES = [
    ("the free base", TPP_FREE_BASE),
    ("the free base of porphine", PORPHINE_FREE_BASE),
    ("the cis resonance form of the dianion", CIS_DIANION),
    ("a monoanion", "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[n-]3"),
    ("a carboxylate beside the dianion (net charge -3)", TPP_WITH_CARBOXYLATE),
    ("a carboxylic acid (a principal group: another parent)", TPP_WITH_ACID),
    ("a stereo descriptor", TPP_WITH_STEREOCENTRE),
    ("the dication", "C1=Cc2[nH+]c1c(-c1ccccc1)c1ccc([nH]1)c(-c1ccccc1)c1[nH+]c(c(-c3ccccc3)c3ccc([nH]3)c2-c2ccccc2)C=C1"),
    ("the dianion beside its cation (two fragments)", TPP_DIANION + ".[Zn+2]"),
    ("a bonded drawing (no negative nitrogen)", TPP_BONDED_2),
    ("a phthalocyanine dianion (meso nitrogens: no porphyrin core)",
     "C1=CC=C2C(=C1)C3=NC4=NC(=NC5=C6C=CC=CC6=C([N-]5)N=C7C8=CC=CC=C8C(=N7)N=C2[N-]3)C9=CC=CC=C94"),
]


@pytest.mark.parametrize("what,smiles", DECLINES, ids=[d[0][:50] for d in DECLINES])
def test_the_producer_declines(what, smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, what
    assert name_porphyrin_dianion(mol) is None, what


def test_the_bare_parent_is_declined_by_the_free_base_guard_and_passed_for_the_anion():
    mol = Chem.MolFromSmiles(PORPHINE_FREE_BASE)
    assert decline_reason(mol) == "bare"
    assert decline_reason(mol, bare_ok=True) is None


# ---- the zinc complexes (panel E1, E2; the three drawings of one compound share one name)

@pytest.mark.parametrize("smiles", [TPP_DIANION + ".[Zn+2]", TPP_BONDED_2, TPP_BONDED_4],
                         ids=["ionic", "two Zn-N", "four Zn-N charged"])
def test_zinc_tetraphenylporphyrin_is_named_on_the_porphyrin_parent(smiles):
    """The three drawings of zinc tetraphenylporphyrin (panel E1) have one standard InChIKey and one name; the
    two bonded drawings are named through their ionic drawing (``DISCONNECT``)."""
    _assert_named_without_a_pin(smiles, ZN_TPP_NAME)


@pytest.mark.parametrize("smiles", [PORPHINE_DIANION + ".[Zn+2]", PORPHINE_BONDED_2], ids=["ionic", "two Zn-N"])
def test_zinc_porphine_is_named_on_the_porphyrin_parent(smiles):
    _assert_named_without_a_pin(smiles, ZN_PORPHINE_NAME)


def test_zinc_octaethylporphyrin_and_a_dissymmetric_porphyrin():
    oep = "CCc1c(CC)c2cc3[n-]c(cc4nc(cc5[n-]c(cc1n2)c(CC)c5CC)c(CC)c4CC)c(CC)c3CC.[Zn+2]"
    _assert_named_without_a_pin(oep, "zinc 2,3,7,8,12,13,17,18-octaethylporphyrin-21,23-diide")
    diphenyl = _dianion("c1ccc(cc1)-c1c2ccc(n2)cc2ccc([nH]2)c(-c2ccccc2)c2ccc(n2)cc2ccc1[nH]2") + ".[Zn+2]"
    row = _row(diphenyl, "best-effort")
    assert row["name"] == "zinc 5,15-diphenylporphyrin-21,23-diide", row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert_full_rt(row["name"], diphenyl)


def test_the_free_base_beside_zinc_keeps_the_name_it_has():
    """``5,10,15,20-tetraphenylporphyrin—zinc(2+) (1/1)`` is the adduct of the free base (the molecule u2 of the
    plan): the ligand is the free base, which the porphyrin parent names; no producer of the anion is reached."""
    _assert_named_without_a_pin(TPP_FREE_BASE + ".[Zn+2]", "5,10,15,20-tetraphenylporphyrin—zinc(2+) (1/1)",
                                tiers=("best-effort",))
    _assert_named_without_a_pin(PORPHINE_FREE_BASE + ".[Zn+2]", "porphyrin—zinc(2+) (1/1)", tiers=("best-effort",))


def test_the_diide_is_cited_in_the_adduct_of_a_complex_with_a_neutral_ligand():
    """A neutral co-ligand beside the complex: the salt route declines a neutral co-former, the adduct names the
    components, and the dianion is cited by its porphyrin name there too (``name_adduct``; the best-effort tier,
    as for every adduct of ions)."""
    _assert_named_without_a_pin(
        TPP_DIANION + ".[Zn+2].c1ccncc1",
        "5,10,15,20-tetraphenylporphyrin-21,23-diide—pyridine—zinc(2+) (1/1/1)", tiers=("best-effort",))


def test_the_diide_is_cited_in_the_salt_of_a_hydrate():
    """Water beside the complex is the hydrate of the salt (``monohydrate``): the salt route cites the anion by its
    porphyrin name and names the hydrate."""
    _assert_named_without_a_pin(PORPHINE_DIANION + ".[Zn+2].O", "zinc porphyrin-21,23-diide monohydrate")


def test_a_zinc_phthalocyanine_keeps_the_von_baeyer_ligand_name():
    """OPSIN has no phthalocyanine (hazard H7 of the plan) and the producer has no porphyrin core to name: the
    ionic drawing is named with the von Baeyer ligand of the general engine."""
    row = _row(PHTHALOCYANINE_ZN, "best-effort")
    assert row["name"] == PHTHALOCYANINE_ZN_NAME, row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False
    assert_full_rt(row["name"], PHTHALOCYANINE_ZN)


def test_the_two_names_of_the_dianion_read_back_and_the_cis_locants_do_not():
    """Both spellings of the 21,23-diide read back; ``21,22-diide`` reads another structure (the plan, section 1.4),
    so the locants of the diide are not free."""
    ionic = TPP_DIANION + ".[Zn+2]"
    assert name_is_rt_exact("zinc 5,10,15,20-tetraphenylporphyrin-21,23-diide", ionic)
    assert name_is_rt_exact("zinc 5,10,15,20-tetraphenyl-21H,23H-porphyrin-21,23-diide", ionic)
    assert not name_is_rt_exact("zinc 5,10,15,20-tetraphenylporphyrin-21,22-diide", ionic)


# ---- the label: no PIN, by the predicate of the structure,

def test_a_fused_ring_substituent_is_named_on_the_parent_and_labelled_without_a_pin():
    """The substituent namer names a naphthalenyl group of the ion at both tiers that name ions."""
    _assert_named_without_a_pin(NAPHTHYL_DIANION + ".[Zn+2]",
                                "zinc 5-(naphthalen-1-yl)-10,15,20-triphenylporphyrin-21,23-diide")


def test_a_decorated_fused_ring_substituent_has_the_porphyrin_name_at_both_tiers():
    """The substituent namer names a bromoanthracenyl group of the ion at the best-effort call and, since the fused
    '-yl' prefix builder runs inside it (NPL 2026-11-01; it cites the side-chain stereo), at the valid tier too: the
    ion has the porphyrin name at both. Before, the valid tier kept the name of the general engine, so this
    was a test of "the best-effort call only". Both labels are verified systematic names."""
    smiles = BROMOANTHRACENYL_DIANION + ".[Zn+2]"
    for tier in ("best-effort", "valid"):
        row = _row(smiles, tier)
        assert row["name"] == "zinc 5-(8-bromoanthracen-1-yl)-10,15,20-triphenylporphyrin-21,23-diide", (tier, row)
        assert row["tier"] == "systematic_verified" and row["is_pin"] is False, (tier, row)
        assert_full_rt(row["name"], smiles)


def test_the_predicate_is_no_longer_the_only_thing_that_lowers_the_label(monkeypatch):
    """Before NPL 2026-11-01 the salt name of the PIN path carried ``pin_unverified`` with the predicate switched
    off: only the predicate made the label ``systematic_verified``. The spelling check now marks
    'porphyrin' INTRODUCTION, the Blue Book "Preferred IUPAC names (PINs) are not identified for the
    compounds in this Chapter";, Table 10.1 (d)), so the label stays ``systematic_verified`` without the
    predicate: two independent reasons lower it. The predicate itself is pinned by ``test_the_label_predicate``."""
    smiles = TPP_DIANION + ".[Zn+2]"
    assert _row(smiles, "best-effort")["tier"] == "systematic_verified"
    monkeypatch.setattr(metal_labels, "is_macrocycle_complex_name_without_pin", lambda smiles, name: False)
    assert _row(smiles, "best-effort")["tier"] == "systematic_verified"


@pytest.mark.parametrize("smiles,expected", [
    (TPP_DIANION + ".[Zn+2]", True),
    (PORPHINE_DIANION + ".[Zn+2].O", True),
    (NAPHTHYL_DIANION + ".[Zn+2]", True),
    (BROMOANTHRACENYL_DIANION + ".[Zn+2]", True),
    (TPP_WITH_STEREOCENTRE + ".[Zn+2]", True),    # the producer declines a stereo descriptor: the structure decides
    (TPP_WITH_ACID + ".[Zn+2]", True),            # a suffix group: another parent, the ion is still a porphyrin ion
    (TPP_BONDED_2, False),                       # one fragment: the label of a rewritten drawing is the rewrite's
    (TPP_DIANION + ".[Cu+2]", True),             # copper is admitted to MACROCYCLE since lane B2
    (TPP_DIANION + ".[Tc+2]", False),            # technetium is not
    (TPP_DIANION + ".[Mg+2]", False),
    (TPP_DIANION, False),                        # no metal
    (TPP_FREE_BASE + ".[Zn+2]", False),          # the free base is no ion; the adduct predicate labels its name
    (PHTHALOCYANINE_ZN, False),                  # no porphyrin core
    ("CC(=O)[O-].CC(=O)[O-].[Zn+2]", False),
])
def test_the_label_predicate(smiles, expected):
    assert is_macrocycle_complex_name_without_pin(smiles, "a name") is expected
    assert is_macrocycle_complex_name_without_pin(smiles, None) is False
    assert is_macrocycle_complex_name_without_pin("", "a name") is False


def test_an_ion_the_producer_declines_keeps_the_general_engine_name_and_has_no_pin_label():
    """A dianion with a stereo descriptor has no porphyrin name (the producer declines it); the general engine names
    the ligand and the label is a verified systematic name at both tiers."""
    smiles = TPP_WITH_STEREOCENTRE + ".[Zn+2]"
    for tier in ("best-effort", "valid"):
        row = _row(smiles, tier)
        assert row["tier"] == "systematic_verified" and row["is_pin"] is False, (tier, row)
        assert "porphyrin" not in row["name"], row
        assert_full_rt(row["name"], smiles)


def test_the_label_predicate_does_not_read_the_producer(monkeypatch):
    """The label predicate holds whether or not the producer names the dianion with the bromoanthracenyl group.
    NPL 2026-11-01: the fused '-yl' prefix builder now runs inside the substituent namer, so the producer names
    this dianion outside the best-effort call too (it declined before); the producer is therefore made to decline
    here, and the predicate, which reads the structure, holds all the same."""
    from orthonym.rules.metal_macrocycles import macrocycle_ion_fragments
    smiles = BROMOANTHRACENYL_DIANION + ".[Zn+2]"
    mol = Chem.MolFromSmiles(smiles)
    monkeypatch.setattr("orthonym.rules.adducts.macrocycle_component_names", lambda m: {})
    monkeypatch.setattr("orthonym.rules.metal_macrocycles.macrocycle_component_names", lambda m: {})
    assert macrocycle_component_names(mol) != {}     # the producer names it now; the predicate does not call it
    assert len(macrocycle_ion_fragments(mol)) == 1
    assert is_macrocycle_complex_name_without_pin(smiles, "a name") is True


def test_the_ion_fragments_are_the_porphyrin_ions_of_an_admitted_assembly():
    from orthonym.rules.metal_macrocycles import macrocycle_ion_fragments
    zinc = Chem.MolFromSmiles(TPP_DIANION + ".[Zn+2].c1ccncc1")
    assert [Chem.MolToSmiles(f) for f in macrocycle_ion_fragments(zinc)] == [Chem.MolToSmiles(Chem.MolFromSmiles(TPP_DIANION))]
    for smiles in (TPP_DIANION + ".[Tc+2]", TPP_DIANION, TPP_FREE_BASE + ".[Zn+2]", "[Zn+2].[Cl-].[Cl-]",
                   TPP_DIANION + ".[Zn+2].[Tc+2]"):
        assert macrocycle_ion_fragments(Chem.MolFromSmiles(smiles)) == [], smiles


# ---- the scope: the guard reads the table for every metal of the assembly

@pytest.mark.parametrize("smiles,expected", [
    (TPP_DIANION + ".[Zn+2]", True),
    (TPP_DIANION + ".[Zn+2].c1ccncc1", True),
    (TPP_DIANION + ".[Zn+2].[Zn+2]", True),
    (TPP_DIANION, False),                        # no metal
    (TPP_DIANION + ".[Cu+2]", True),             # admitted since lane B2
    (TPP_DIANION + ".[Zn+2].[Cu+2]", True),
    (TPP_DIANION + ".[Tc+2]", False),
    (TPP_DIANION + ".[Zn+2].[Tc+2]", False),     # one metal outside: the whole assembly is outside
    (TPP_DIANION + ".[Na+].[Na+]", False),
    (TPP_DIANION + ".[Mg+2]", False),
    ("[Zn+2].[Cl-].[Cl-]", True),                # the guard reads the metals; the producer reads the fragments
])
def test_the_guard_reads_the_table_for_every_metal_of_the_assembly(smiles, expected):
    assert assembly_admits_macrocycle(Chem.MolFromSmiles(smiles)) is expected


def test_the_component_names_of_an_assembly():
    mol = Chem.MolFromSmiles(TPP_DIANION + ".[Zn+2].c1ccncc1")
    names = macrocycle_component_names(mol)
    assert list(names.values()) == ["5,10,15,20-tetraphenylporphyrin-21,23-diide"]
    assert macrocycle_component_names(Chem.MolFromSmiles(TPP_DIANION + ".[Tc+2]")) == {}
    assert macrocycle_component_names(Chem.MolFromSmiles("[Zn+2].[Cl-].[Cl-]")) == {}
    ion = Chem.MolFromSmiles(TPP_DIANION)
    assert macrocycle_anion_name(ion, Chem.MolFromSmiles("[Zn+2]")) == "5,10,15,20-tetraphenylporphyrin-21,23-diide"
    assert macrocycle_anion_name(ion, Chem.MolFromSmiles("[Cu+2]")) == "5,10,15,20-tetraphenylporphyrin-21,23-diide"
    assert macrocycle_anion_name(ion, Chem.MolFromSmiles("[Tc+2]")) is None
    assert macrocycle_anion_name(None, Chem.MolFromSmiles("[Zn+2]")) is None


def test_the_guard_follows_the_scope_table(monkeypatch):
    """Admitting a metal to MACROCYCLE is an edit of the table: technetium alone, then technetium with zinc."""
    cu = Chem.MolFromSmiles(TPP_DIANION + ".[Tc+2]")
    both = Chem.MolFromSmiles(TPP_DIANION + ".[Tc+2].[Zn+2]")
    assert assembly_admits_macrocycle(cu) is False and assembly_admits_macrocycle(both) is False
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Tc", MetalScope(frozenset({ELEMENT, MACROCYCLE}), frozenset({2})))
    assert assembly_admits_macrocycle(cu) is True and assembly_admits_macrocycle(both) is True
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Zn", MetalScope(frozenset({ELEMENT}), frozenset({2})))
    assert assembly_admits_macrocycle(Chem.MolFromSmiles(TPP_DIANION + ".[Zn+2]")) is False


def test_zinc_and_the_bonded_metals_alone_are_admitted_to_macrocycle():
    assert sorted(s for s, sc in METAL_SCOPE.items() if MACROCYCLE in sc.mechanisms) == sorted(
        ["Zn", *BONDED_METALS])


def test_the_adduct_reads_the_names_in_the_unverified_call_only():
    """The names of the macrocyclic anions are read in a call with ``general_fallback_unverified`` (the tiers
    that name ions): a plain call of ``name_adduct`` (the PIN tier) names no complex."""
    mol = Chem.MolFromSmiles(TPP_DIANION + ".[Zn+2].c1ccncc1")
    assert adducts.name_adduct(mol, Chem.MolToSmiles(mol)) is None
    name = adducts.name_adduct(mol, Chem.MolToSmiles(mol), general_fallback=True, general_fallback_unverified=True,
                               allow_aromatic_general=True)
    assert name == "5,10,15,20-tetraphenylporphyrin-21,23-diide—pyridine—zinc(2+) (1/1/1)"


def test_the_producer_is_metal_blind_and_the_scope_module_holds_the_metal():
    """The producer names the dianion of any prefix-only porphyrin; the metal is the scope module's."""
    assert metal_macrocycles.macrocycle_anion_name.__module__ == "orthonym.rules.metal_macrocycles"
    assert name_porphyrin_dianion(Chem.MolFromSmiles(TPP_DIANION)) == "5,10,15,20-tetraphenylporphyrin-21,23-diide"


# ---- the other metals and the unscoped ions are not moved (START literals, computed on the start tree)

_VB = ("2,7,12,17-tetraphenyl-21,22,23,24-tetraazapentacyclo[16.2.1.1^3,6.1^8,11.1^13,16]"
       "tetracosa-1(20),2,4,6(24),7,9,11,13(22),14,16,18-undecaene-21,23-diide")
_VB_PORPHINE = ("21,22,23,24-tetraazapentacyclo[16.2.1.1^3,6.1^8,11.1^13,16]"
                "tetracosa-1(20),2,4,6(24),7,9,11,13(22),14,16,18-undecaene-21,23-diide")
_ABSTAIN_PIN = {"Tc": "inorganic compound (not supported)", "Cu": "copper compound (not supported)", "Fe": "iron compound (not supported)",
                "Ni": "nickel compound (not supported)", "Mg": "magnesium compound (not supported)",
                "Na": "sodium compound (not supported)", "Ca": "calcium compound (not supported)"}

START_ROWS = {
    # technetium and plutonium: no mechanism (copper, iron and nickel, the witnesses of the start commit, are admitted
    # to MACROCYCLE since lane B2: ``ADMITTED_B2_PORPHYRINS``)
    TPP_DIANION + ".[Tc+2]": (
        (_ABSTAIN_PIN["Tc"], "abstain", False, "UNSUPPORTED_ELEMENT"),
        (_VB + "—technetium(2+) (1/1)", "systematic_verified", False, None)),
    PORPHINE_DIANION + ".[Tc+2]": (
        (_ABSTAIN_PIN["Tc"], "abstain", False, "UNSUPPORTED_ELEMENT"),
        (_VB_PORPHINE + "—technetium(2+) (1/1)", "systematic_verified", False, None)),
    PORPHINE_DIANION + ".[Pu+2]": (
        (_ABSTAIN_PIN["Tc"], "abstain", False, "UNSUPPORTED_ELEMENT"),
        (_VB_PORPHINE + "—plutonium(2+) (1/1)", "systematic_verified", False, None)),
    # the salts of Groups 1 and 2 are named from their ions; the table does not admit them to MACROCYCLE
    TPP_DIANION + ".[Mg+2]": (
        (_ABSTAIN_PIN["Mg"], "abstain", False, "UNSUPPORTED_ELEMENT"),
        ("magnesium " + _VB, "systematic_verified", False, None)),
    TPP_DIANION + ".[Ca+2]": (
        (_ABSTAIN_PIN["Ca"], "abstain", False, "UNSUPPORTED_ELEMENT"),
        ("calcium " + _VB, "systematic_verified", False, None)),
    TPP_DIANION + ".[Na+].[Na+]": (
        (_ABSTAIN_PIN["Na"], "abstain", False, "UNSUPPORTED_ELEMENT"),
        ("disodium " + _VB, "systematic_verified", False, None)),
    # zinc beside a metal that the table does not admit: the ligand keeps the name of the general engine
    TPP_DIANION + ".[Zn+2].[Tc+2]": (
        ("zinc compound (not supported)", "abstain", False, "UNSUPPORTED_ELEMENT"),
        (_VB + "—technetium(2+)—zinc(2+) (1/1/1)", "systematic_verified", False, None)),
    # the dianions alone: no metal in the assembly, the general engine's name
    TPP_DIANION: (
        ("unknown organic compound", "abstain", False, "UNNAMEABLE"),
        (_VB, "systematic_verified", False, None)),
    PORPHINE_DIANION: (
        ("unknown organic compound", "abstain", False, "UNNAMEABLE"),
        (_VB_PORPHINE, "systematic_verified", False, None)),
}

#: the rows of ``START_ROWS`` of the start commit that lane B2 admits to MACROCYCLE (``MORE_BONDED_METAL_VALENCES``):
#: the metal is cited with the porphyrin dianion as a salt, as the zinc complex is (the default tier keeps its label;
#: an assembly of admitted metals with zinc, a complete metal, is UNNAMEABLE there); each name is read back
ADMITTED_B2_PORPHYRINS = {
    TPP_DIANION + ".[Cu+2]": (
        (_ABSTAIN_PIN["Cu"], "abstain", False, "UNSUPPORTED_ELEMENT"),
        ("copper 5,10,15,20-tetraphenylporphyrin-21,23-diide", "systematic_verified", False, None)),
    PORPHINE_DIANION + ".[Cu+2]": (
        (_ABSTAIN_PIN["Cu"], "abstain", False, "UNSUPPORTED_ELEMENT"),
        ("copper porphyrin-21,23-diide", "systematic_verified", False, None)),
    PORPHINE_DIANION + ".[Fe+2]": (
        (_ABSTAIN_PIN["Fe"], "abstain", False, "UNSUPPORTED_ELEMENT"),
        ("iron porphyrin-21,23-diide", "systematic_verified", False, None)),
    TPP_DIANION + ".[Ni+2]": (
        (_ABSTAIN_PIN["Ni"], "abstain", False, "UNSUPPORTED_ELEMENT"),
        ("nickel 5,10,15,20-tetraphenylporphyrin-21,23-diide", "systematic_verified", False, None)),
    TPP_DIANION + ".[Zn+2].[Cu+2]": (
        (_ABSTAIN_PIN["Cu"], "abstain", False, "UNNAMEABLE"),
        ("5,10,15,20-tetraphenylporphyrin-21,23-diide—copper(2+)—zinc(2+) (1/1/1)", "systematic_verified", False, None)),
}


@pytest.mark.parametrize("smiles", sorted(ADMITTED_B2_PORPHYRINS), ids=lambda s: s[-28:])
def test_the_porphyrin_complexes_of_the_metals_of_lane_b2_are_named(smiles):
    default_view, best_effort_view = ADMITTED_B2_PORPHYRINS[smiles]
    assert _view(_row(smiles, "pin")) == default_view, smiles
    assert _view(_row(smiles, "best-effort")) == best_effort_view, smiles
    assert_full_rt(best_effort_view[0], smiles)


@pytest.mark.parametrize("smiles", sorted(START_ROWS), ids=lambda s: s[-28:])
def test_other_metals_and_unscoped_ions_are_not_moved(smiles):
    default_view, best_effort_view = START_ROWS[smiles]
    assert _view(_row(smiles, "pin")) == default_view, smiles
    assert _view(_row(smiles, "best-effort")) == best_effort_view, smiles


def test_the_other_metals_panel_holds_no_row_the_guard_could_reach():
    """The 73-drawing panel of the other metals is untouched by construction: no assembly of it is admitted to
    MACROCYCLE (every one holds a metal other than zinc, or none), but the iron and copper porphyrin rows of the metals
    that lane B2 admits: they are named as declared (``declared_moves_bonded_b2.tsv``)."""
    from tests.support.metal_baselines import baseline_path, declared_bonded_metal_moves, read_rows
    declared = declared_bonded_metal_moves()
    reached = []
    for row in read_rows(baseline_path("other")):
        mol = Chem.MolFromSmiles(row["smiles"])
        if any(a.GetSymbol() in BONDED_METALS for a in mol.GetAtoms()):
            if macrocycle_component_names(mol):
                reached.append(row["smiles"])
                assert (row["smiles"], "best-effort") in declared, row["smiles"]    # the default tier keeps its label
            continue
        assert macrocycle_component_names(mol) == {}, row["smiles"]
    assert sorted(set(reached)) == sorted([
        "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[n-]5)C=C4)[n-]3.[Fe+2]",
        "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[n-]5)C=C4)[n-]3.[Cu+2]"]), reached


def test_the_default_tier_is_not_moved_for_zinc():
    """The names are read at the tiers that name ions only: the default tier declines every zinc porphyrin
    drawing, with no PIN."""
    for smiles in (TPP_DIANION + ".[Zn+2]", TPP_BONDED_2, TPP_BONDED_4, PORPHINE_DIANION + ".[Zn+2]",
                   TPP_FREE_BASE + ".[Zn+2]"):
        row = _row(smiles, "pin")
        assert row["tier"] == "abstain" and row["is_pin"] is False, (smiles, row)
