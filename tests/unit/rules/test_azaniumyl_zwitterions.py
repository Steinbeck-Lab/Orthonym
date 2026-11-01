"""The zwitterion of a fully ionised carboxylic acid with one primary ammonium group and other nitrogen
atoms (``rules.azaniumyl_zwitterions``).

 "Anionic and cationic centers on different parent structures" (the Blue Book): "may be named
by prefixing the name of the cationic center... to the name of the anionic parent structure";
INTRODUCTION (:42411): "anionic centers are preferred for lower locants and become the parent structure".
The group -NH3(+) is the prefix 'azaniumyl' ('(2S)-2-azaniumyl-3-(methylsulfanyl)propanoate',,:54569);
a carboxylate that is not the suffix of the parent is the prefix 'carboxylato' "Prefixes for anionic
centers derived from acid characteristic groups",:41199, '-CO-O(-) carboxylato (preferred prefix)',:41209).
 "Choice of an anionic parent structure" (:41261, (a):41265): the parent has the maximum number of anionic
centres, which for a fully ionised acid is the neutral acid's count of carboxy suffixes.

Every name is read back by an independent OPSIN call (the full standard InChIKey).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.perception.ions import get_ion_sites
from orthonym.rules import azaniumyl_zwitterions, charged_router
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

PIN_ROWS = [
    # the three zwitterions of the dev2000 population
    ("C[C@H](NC(=O)CC[C@H]([NH3+])C(=O)[O-])C(=O)[O-]",
     "(2S)-2-azaniumyl-5-{[(1S)-1-carboxylatoethyl]amino}-5-oxopentanoate"),
    ("[NH3+][C@@H](CCC(=O)N[C@H](CCC(=O)[O-])C(=O)[O-])C(=O)[O-]",
     "(2R)-2-[(4S)-4-azaniumyl-4-carboxylatobutanamido]pentanedioate"),
    ("CC/C=C\\C/C=C\\C[C@H](O)[C@@H](/C=C/C=C/C=C\\C/C=C\\CCC(=O)[O-])SC[C@H]([NH3+])C(=O)[O-]",
     "(4Z,7Z,9E,11E,13R,14S,16Z,19Z)-13-{[(2R)-2-azaniumyl-2-carboxylatoethyl]sulfanyl}-14-hydroxydocosa-"
     "4,7,9,11,16,19-hexaenoate"),
    # the class: peptides and acyl amino acids of the same kind
    ("[NH3+]CC(=O)NCC(=O)[O-]", "(2-azaniumylacetamido)acetate"),
    ("[NH3+]C(Cc1ccccc1)C(=O)NC(C)C(=O)[O-]", "2-(2-azaniumyl-3-phenylpropanamido)propanoate"),
    ("[NH3+]CC(=O)N[C@@H](CS)C(=O)[O-]", "(2R)-2-(2-azaniumylacetamido)-3-sulfanylpropanoate"),
    ("[NH3+]CC(=O)NC(CC(=O)[O-])C(=O)[O-]", "2-(2-azaniumylacetamido)butanedioate"),
    ("[NH3+]C(CCC(=O)NCC(=O)[O-])C(=O)[O-]", "2-azaniumyl-5-[(carboxylatomethyl)amino]-5-oxopentanoate"),
    ("[NH3+]CC(=O)N1CCCC1C(=O)[O-]", "1-(azaniumylacetyl)pyrrolidine-2-carboxylate"),
    # 'anilino' stands between 'amino' and 'azaniumyl',:3446): the name is cited and numbered for the ion's prefix,
    # not as the neutral molecule's with a word changed ('2-azaniumyl-3-(4-anilinophenyl)propanoate' was misordered)
    ("[NH3+]C(Cc1ccc(Nc2ccccc2)cc1)C(=O)[O-]", "3-(4-anilinophenyl)-2-azaniumylpropanoate"),
]

TIERS = ["pin", "best-effort"]


def _row(smiles, tier):
    engine = Orthonym(style="pin", **(_emit_tier_flags(tier) if tier != "pin" else {}))
    return engine.name_tiered(smiles)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("smiles, pin", PIN_ROWS)
def test_the_anion_is_the_parent_and_the_ammonium_a_prefix(smiles, pin, tier):
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("smiles, pin", PIN_ROWS)
def test_the_name_is_the_one_the_route_builds(smiles, pin):
    assert charged_router.route_charged(Chem.MolFromSmiles(smiles), "pin") == pin


def test_the_all_ammonium_builder_keeps_its_names():
    # only nitrogen atoms are ammonium groups: ``_name_primary_amine_azaniumyl_zwitterion`` names them, as before
    for smiles, name in [("[NH3+]C(CCC(=O)[O-])C(=O)[O-]", "2-azaniumylpentanedioate"),
                         ("CC(=O)CC([NH3+])C(=O)[O-]", "2-azaniumyl-4-oxopentanoate"),
                         ("[NH3+]C(CCSC)C(=O)[O-]", "2-azaniumyl-4-(methylsulfanyl)butanoate")]:
        row = _row(smiles, "pin")
        assert (row["name"], row["tier"]) == (name, "pin_verified"), row


#: what the builder leaves to other routes: an acid group still un-ionised (the parent is the one that carries the
#: anion,, two cations, a cation that is no primary ammonium, an anion that is no carboxylate
DECLINED = [
    "[NH3+]C(CCC(=O)O)C(=O)[O-]",
    "[NH3+]C(CCC(=O)[O-])C(=O)O",
    "NC(=[NH2+])NCCCC[C@H]([NH3+])C(=O)[O-]",
    "C[N+](C)(C)CC(=O)NCC(=O)[O-]",
    "[NH3+]CC(=O)NCS(=O)(=O)[O-]",
    "[NH3+]CC(=O)NCCC(=O)[O-].[Na+]",
]


@pytest.mark.parametrize("smiles", DECLINED)
def test_the_builder_declines_what_it_does_not_cover(smiles):
    mol = Chem.MolFromSmiles(smiles)
    sites = get_ion_sites(mol)
    assert azaniumyl_zwitterions.fully_ionised_azaniumyl_zwitterion(
        mol, sites["cations"], sites["anions"], "pin") == ""


@pytest.mark.parametrize("smiles", DECLINED[:5])
def test_a_declined_shape_is_never_named_wrong(smiles):
    for tier in TIERS:
        row = _row(smiles, tier)
        if row["name"] and row["tier"] != "abstain":
            assert name_is_rt_exact(row["name"], smiles), (tier, row)


#: phosphoserine zwitterion: the book is silent on whether the carboxylate or the phosphate monoester dianion is the anionic
#: parent. "Choice of an anionic parent structure" (the Blue Book), (a) "parent with the maximum number of anionic
#: centers, including anionic suffixes" (:41265) would pick the phosphate (two centres) if an ester were an anionic parent, and
#: (e) (:41289) goes by the general seniority of classes otherwise, which puts the carboxylate first; the examples
#: '2-(carboxylatomethyl)benzoate (PIN)' (:41293) and '3-oxidonaphthalene-2-carboxylate (PIN)' (:41295) do not decide it. The
#: maintainers' ruling of 2026-11-01: ship the carboxylate parent with 'phosphonatooxy' below the PIN and record the gap here.
def test_the_phosphoserine_zwitterion_is_named_below_the_pin_until_the_book_decides_the_parent():
    smiles = "[NH3+][C@H](COP(=O)([O-])[O-])C(=O)[O-]"
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert (row["name"], row["tier"]) == ("(2R)-2-azaniumyl-3-(phosphonatooxy)propanoate", "systematic_verified"), row
    assert name_is_rt_exact(row["name"], smiles), row
    assert Orthonym(style="pin").name_tiered(smiles)["tier"] == "abstain"


#: ruling R6 (RULINGS-ANION.md): a zwitterionic amino acid takes the name at every tier, the retained table name
#: staying only where no such name is built: the partial zwitterions of glutamic and aspartic acid (a free carboxy group
#: beside the carboxylate,:54554 gives no name for them) and carnitine, whose composed name breaks the
#: hyphen rule. "Zwitterionic amino acids are named in two ways" (the Blue Book, example
#::54561);:50943 gives the retained names no PIN.
_RETAINED_STAYS = {"L-glutamic acid", "L-aspartic acid", "L-carnitine", "carnitine"}


def _retained_zwitterion_rows():
    from orthonym.rules.salts import RETAINED_AMINO_ACID_ZWITTERIONS
    return sorted(RETAINED_AMINO_ACID_ZWITTERIONS.items())


@pytest.mark.parametrize("smiles, retained", _retained_zwitterion_rows())
def test_a_retained_zwitterion_takes_the_azaniumyl_name_unless_none_is_built(smiles, retained):
    for tier in TIERS:
        row = _row(smiles, tier)
        if retained not in _RETAINED_STAYS:
            assert row["name"] != retained and ("azaniumyl" in row["name"] or "-ium-" in row["name"]), (tier, row)
            assert row["tier"] == "pin_verified", (tier, row)
        elif tier == "pin":
            assert row["tier"] == "abstain", (tier, row)    # the retained name is no PIN:50943)
            continue
        else:
            assert (row["name"], row["tier"]) == (retained, "systematic_verified"), (tier, row)
        assert name_is_rt_exact(row["name"], smiles), (tier, row)
