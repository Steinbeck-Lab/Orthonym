"""Best-effort names a P or S oxoacid group as one substituent prefix (lane nq-forms-ps).

The group prefixes are those of "Compound and complex substituent groups"
(the Blue Book; examples:36333,:36335,:36337), (:36484;:36488,:36494),
 "Substituent groups derived from polyacids" (:36937) method (1), and
(:41211,:41213: 'sulfonato', 'phosphonato'). The 'a' chain each name had before ends on O
,:6465). Every name is read back by a fresh OPSIN call to the input's full
InChIKey; the default tier is unchanged."""
import pytest

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate


# (smiles, best-effort name) -- each read back by a fresh OPSIN call to the full InChIKey.
# The 'a' chain each had before is in the comment.
NAMED = [
    ("Nc1c(O)c(S(=O)(=O)O)c(O)c2c1C(=O)c1ccccc1C2=O",
     "1-amino-2,4-dihydroxy-9,10-dioxo-3-sulfo-9,10-dihydroanthracene"),   # 3-(1,1-dioxo-2-oxa-1λ6-thiaethyl)
    ("Nc1c(O)c(S(=O)(=O)[O-])c(O)c2c1C(=O)c1ccccc1C2=O",
     "1-amino-2,4-dihydroxy-9,10-dioxo-3-sulfonato-9,10-dihydroanthracene"),  # 3-(1-oxido-1-oxo-2-oxa-1-thiaeth-1-en-1-yl)
    ("CC(=O)NCCOS(=O)(=O)O", "N-[2-(sulfooxy)ethyl]acetamide"),           # N-(4,4-dioxo-3,5-dioxa-4λ6-thiapentyl)
    ("CCOP(=S)(OCC)SCC(=O)OC", "methyl [(diethoxyphosphorothioyl)sulfanyl]acetate"),
    ("CCSCCSP(=S)(OC)OC", "1-[(dimethoxyphosphorothioyl)sulfanyl]-2-(ethylsulfanyl)ethane"),
    ("CCOP(=S)(OCC)Oc1ccc(cc1)[N+](=O)[O-]", "1-[(diethoxyphosphorothioyl)oxy]-4-nitrobenzene"),
    ("CC(C)OP(C)(=O)F", "2-{[fluoro(methyl)phosphoryl]oxy}propane"),
    ("O=C(O)CCS(=O)(=O)NCC", "3-(ethylsulfamoyl)propanoic acid"),
    ("O=C(O)CCS(=O)(=O)N(C)CC", "3-[ethyl(methyl)sulfamoyl]propanoic acid"),
    # "Substituent groups derived from polyacids" (the Blue Book): method (2), the
    # diphosphoxane parent, is the PIN: '3-[(1,3,3-trihydroxy-1,3-dioxo-1λ5,3λ5-diphosphoxan-1-yl)
    # oxy]propanoic acid (PIN)' (:36949); method (1) is printed beside it
    ("CC(=O)NCCOP(=O)(O)OP(=O)(O)O",
     "1-acetamido-2-[(1,3,3-trihydroxy-1,3-dioxo-1λ5,3λ5-diphosphoxan-1-yl)oxy]ethane"),
    ("OCCOP(=O)(O)OP(=O)(O)OP(=O)(O)O",
     "2-[(1,3,5,5-tetrahydroxy-1,3,5-trioxo-1λ5,3λ5,5λ5-triphosphoxan-1-yl)oxy]ethan-1-ol"),
    ("CCCCC(=O)OCC(COP(=O)([O-])OC[C@H]([NH3+])C(=O)[O-])OC(=O)CCCC",
     "1-[3-({[(2S)-2-azaniumyl-3-oxido-3-oxopropoxy](oxido)phosphoryl}oxy)-2-(pentanoyloxy)propoxy]-1-oxopentane"),
]


@pytest.mark.parametrize("smiles,expected", NAMED)
def test_best_effort_names_the_group_as_a_prefix(smiles, expected):
    row = name_best_effort(smiles)
    assert row["name"] == expected, row
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles", [s for s, _ in NAMED])
def test_the_name_has_no_replacement_chain_for_the_group(smiles):
    name = name_best_effort(smiles)["name"]
    for term in ("phospha", "-thia", "dioxa", "trioxa"):
        assert term not in name.replace("sulfanyl", ""), name


# the name each had before this lane: still named, still read back (never a lost name)
KEPT = [
    "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCC)COP(=O)(O)OC1C(O)C(O)C(O)[C@@H](O)C1O",
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OCC(O)C(=O)[O-])[C@@H](O)[C@H]1O",
    "CCCCC(=O)O[C@H](COC(=O)CCCC)COP(=O)([O-])OC[C@H]([NH3+])C(=O)[O-]",
]


@pytest.mark.parametrize("smiles", KEPT)
def test_a_group_the_book_spelling_cannot_verify_keeps_a_verified_name(smiles):
    name = name_best_effort(smiles)["name"]
    assert name and "unknown" not in name
    assert_full_rt(name, smiles)


# the default tier is unchanged: the PIN path is untouched and the wider-tier prefix is
# not offered there
@pytest.mark.parametrize("smiles,tier,name", [
    ("OP(=O)(O)OCC(=O)O", "pin_verified", "(phosphonooxy)acetic acid"),       #:36333
    ("c1ccccc1S(=O)(=O)O", "pin_verified", "benzenesulfonic acid"),
    ("COP(=O)(O)O", "pin_verified", "methyl dihydrogen phosphate"),
])
def test_default_tier_is_unchanged(smiles, tier, name):
    row = default_tier_row(smiles)
    assert row["tier"] == tier and row["name"] == name, row


# compound ligands (review B2): every one was lost at the complete tier or fell back to the
# issue-7 forms ('1-azaethan-1-yl', 'cyclohexa-1,3,5-triene') before the ligand flag.
COMPOUND = [
    ("CNP(=O)(NC)OCC(=O)O", "2-{[bis(methylamino)phosphoryl]oxy}ethanoic acid"),
    ("OC(=O)CCOP(=O)(SCC)SCC", "3-{[bis(ethylsulfanyl)phosphoryl]oxy}propanoic acid"),
    ("OC(=O)CCOP(=S)(SC)SC", "3-{[bis(methylsulfanyl)phosphorothioyl]oxy}propanoic acid"),
    ("CN(C)P(=O)(N(C)C)OCC(=O)O", "2-{[bis(dimethylamino)phosphoryl]oxy}ethanoic acid"),
    ("OC(=O)CCOP(=O)(NCC)NCC", "3-{[bis(ethylamino)phosphoryl]oxy}propanoic acid"),
    ("OC(=O)c1ccc(cc1)P(=S)(Nc1ccccc1)Nc1ccccc1",
     "4-[bis(phenylamino)phosphorothioyl]benzene-1-carboxylic acid"),
    ("OC(=O)CCOP(=O)(OCc1ccccc1)OC", "3-{[(benzyloxy)(methoxy)phosphoryl]oxy}propanoic acid"),
]


def _name_complete(smiles):
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    return Orthonym(style="pin", **_emit_tier_flags("complete")).name_tiered(smiles)


@pytest.mark.parametrize("smiles,expected", COMPOUND)
def test_compound_ligands_are_enclosed_and_multiplied(smiles, expected):
    row = name_best_effort(smiles)
    assert row["name"] == expected and row["tier"] == "systematic_verified", row
    assert_full_rt(expected, smiles)
    comp = _name_complete(smiles)
    assert comp["name"] == expected, comp


# a prefix that is a valid spelling but not the preferred one labels the name below the PIN:
# the acid function left on the centre (adefovir anion), a phosphonoyl group (P-C ligand)
UNPREFERRED = [
    "Nc1ncnc2c1ncn2CCOCP(=O)(O)[O-]",
    "OC(=O)CCOP(=O)(O)OCCN",
    "OC(=O)CCP(=O)(O)c1ccccc1",
]


@pytest.mark.parametrize("smiles", UNPREFERRED)
def test_a_non_preferred_prefix_does_not_carry_a_pin_label(smiles):
    row = name_best_effort(smiles)
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert_full_rt(row["name"], smiles)


# a molecule that is only the group has no parent once the group is kept off the spine, and one
# centre the namer cannot name leaves the whole molecule to the earlier spellings; a P/S leaf
# that voids the floor costs only itself (the ring names of the molecule stay)
KEEP_EARLIER = [
    ("O=S(=O)(F)Cl", "2-chloro-2-fluoro-1,3-dioxa-2-thiapropa-1,2-diene"),
    ("COS(=O)(=O)OS(=O)(=S)OC", "3,3,5-trioxo-5-sulfanylidene-2,4,6-trioxa-3,5-dithiaheptane"),
    ("O=[N+]([O-])c1cccc(S(=O)(=O)NCCNS(=O)(=O)c2cccc([N+](=O)[O-])c2)c1",
     "1-nitro-3-[6-(3-nitrophenyl)-1,1,6-trioxo-7-oxa-1,6-dithia-2,5-diazahept-6-en-1-yl]benzene"),
]


@pytest.mark.parametrize("smiles,expected", KEEP_EARLIER)
def test_the_floor_keeps_the_earlier_spelling_where_a_group_cannot_be_a_prefix(smiles, expected):
    from rdkit import Chem
    from orthonym.assembly.universal_substituent import name_universal_substitutive
    from orthonym.validation.reconstruct import verify_or_none
    name = name_universal_substitutive(Chem.MolFromSmiles(smiles)).name
    assert name == expected
    assert verify_or_none(name, smiles) is not None
