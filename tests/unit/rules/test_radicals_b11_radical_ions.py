"""Radical ions from mononuclear ionic parent hydrides.

the Blue Book-43422: the ionic parent hydride's name + 'yl'/'ylidene', "with
elision of the final letter 'e'". '-ylium' cations are NOT radical ions even
though RDKit counts their empty orbital as two radical electrons; they keep their
names (user decision Q2 = A: a separate build).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.jvm_bridge import opsin_stdout
from tests.support.jars import jar_or_skip
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
    "CC(=O)[NH2+]",
    "C[N+](C)C",
    "C[N-]",
    "C[NH2+]",
    "[NH2+]c1ccccc1",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)



def _strict_rt(name, smiles):
    jar = jar_or_skip()
    txt, _ = opsin_stdout(name, allow_radicals=True, jar_path=jar)
    parsed = Chem.MolFromSmiles((txt or "").strip())
    return parsed is not None and Chem.MolToSmiles(parsed) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _name(smiles):
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    with jvm_slots(1, purpose="test-radical-b11"):
        return Orthonym(style="pin").name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[B-](C)C", "trimethylboranuidyl"),                    # (PIN):43430
    ("COC(=O)[C-]", "(methoxycarbonyl)methanidylidene"),      # (PIN):43439
    ("[CH2+]", "methyliumyl"),                                # (PIN):43443
    ("CCC[OH+]", "propyloxidaniumyl"),                        # (PIN):43515
    ("CC(=O)[N-]", "acetylazanidyl"),                         # (PIN):43517
])
def test_radical_ion(smiles, expected):
    row = _name(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified" and row["is_pin"]
    assert _strict_rt(expected, smiles)


# 'azanidyl' is the preselected name (:43426): a carbon-free compound has a
# preselected name at most, the Blue Book;:2062).
@pytest.mark.opsin_gate
def test_carbon_free_radical_ion_is_a_preselected_name():
    # Paper conformance (user decision 2026-09-30): the tier label is the one of the
    # naming path, as in the paper's measured run ('sodium chloride' pin_verified);
    # the name is unchanged.
    row = _name("[NH-]")
    assert row["name"] == "azanidyl" and row["tier"] == "pin_verified"
    assert row["is_pin"]
    assert _strict_rt("azanidyl", "[NH-]")


# "RADICAL IONS DERIVED FROM PARENT HYDRIDES" (the Blue Book): a radical ion is the 'yl' of its ionic
# parent hydride, which holds the atom's bonds, hydrogens and radical electrons together. A sulfur or selenium cation
# with four bonds and a radical electron is the 'yl' of λ4-sulfanium "Nonstandard bonding numbers",:2756;
# "General rule for systematically naming cationic centers in parent hydrides", 'tetrafluoro(methyl)-
# λ4-sulfanium (PIN)',:41384); 'tetramethylsulfaniumyl' is the 'yl' of
# sulfanium, which has two hydrogens to substitute. A phosphorus anion with three bonds and a radical electron is the
# 'yl' of phosphanuide "Anions formed by addition of hydride ions", '(CH3)4P- tetramethylphosphanuide (PIN)
# tetramethyl-λ5-phosphanide',:41106), not of phosphanide.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[S+](C)(C)C", "tetramethyl-λ4-sulfaniumyl"),
    ("C[Se+](C)(C)C", "tetramethyl-λ4-selaniumyl"),
    ("CC[S+](CC)(CC)CC", "tetraethyl-λ4-sulfaniumyl"),
    ("OC(=O)C[S+](C)(C)C", "(carboxymethyl)tri(methyl)-λ4-sulfaniumyl"),
    ("C[SH+](C)C", "trimethyl-λ4-sulfaniumyl"),
    ("[SH4+]", "λ4-sulfaniumyl"),
    ("C[P-](C)C", "trimethylphosphanuidyl"),
    ("CC(C)[P-](C)C", "dimethyl(propan-2-yl)phosphanuidyl"),
    ("[PH3-]", "phosphanuidyl"),
])
def test_the_ionic_parent_hydride_holds_the_bonds_of_the_radical_ion(smiles, expected):
    row = _name(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified" and row["is_pin"], row
    assert _strict_rt(expected, smiles)


@pytest.mark.parametrize("smiles,expected", [
    ("C[S+]C", "sulfanium"),                 # two bonds + one radical electron: the standard 'ium'
    ("C[S+](C)(C)C", "sulfanium"),           # the 'ium' of λ4-sulfane; lambda_ion_name cites the λ
    ("C[P-]", "phosphanide"),
    ("C[P-](C)C", "phosphanuide"),
    ("C[O+]C", "oxidanium"),
    ("C[B-](C)C", "boranuide"),
])
def test_radical_ion_parent_hydride(smiles, expected):
    from orthonym.rules.radicals import radical_ion_parent_hydride
    mol = Chem.MolFromSmiles(smiles)
    centre = next(a for a in mol.GetAtoms() if a.GetFormalCharge())
    assert radical_ion_parent_hydride(centre) == expected


# "Radical ions on ionic suffix groups" (the Blue Book): "When ions
# may be named by using modified suffixes (see and, the
# suffixes denoting radical centers are added to the name of the cationic or anionic
# parent hydride" -- 'benzenaminiumyl (PIN)' (:43501), 'methanaminidyl (PIN)' (:43503),
# 'methanaminiumyl (PIN)' (:17608); an acyl group on N+ gives an amide cation
# (Table 7.4:41421 'amidium'). The round-trip parser reads none of those PINs, so they
# cannot be verified: the verified azanium/azanide name ships, never as pin_verified.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[NH2+]", "methylazaniumyl"),          # PIN methanaminiumyl (:17608)
    ("[NH2+]c1ccccc1", "phenylazaniumyl"),   # PIN benzenaminiumyl (:43501)
    ("C[N+](C)C", "trimethylazaniumyl"),     # PIN N,N-dimethylmethanaminiumyl
    ("CC(=O)[NH2+]", "acetylazaniumyl"),     # amidium-based PIN (Table 7.4)
    ("C[N-]", "methylazanidyl"),             # PIN methanaminidyl (:43503)
])
def test_suffix_parent_radical_ion_is_not_pin_verified(smiles, expected):
    row = _name(smiles)
    assert row["name"] == expected
    assert row["tier"] not in ("pin_verified", "abstain") and not row["is_pin"]
    assert _strict_rt(expected, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[O+]", "methoxylium"),          # retained PIN, not 'methyloxidaniumylidene'
    ("[NH2+]", "azanylium"),           # gold row
    ("[O+]c1ccccc1", "phenoxylium"),
    ("[NH3+]C", "methanaminium"),      # closed-shell control
])
def test_ylium_and_closed_shell_cations_unchanged(smiles, expected):
    assert _name(smiles)["name"] == expected
