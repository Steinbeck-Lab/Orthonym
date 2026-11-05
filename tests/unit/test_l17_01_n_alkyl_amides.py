"""An N-alkyl amide whose acyl carries a ring is named on its acyl chain (lane/l17 item 1).

 "N-Substitution", (the Blue Book): R-CO-NHR' is named "by citing the substituents R' and
R'' as prefixes preceded by the locant N" ('N-(propan-2-yl)acetamide (PIN)'): the parent chain of the amide is its acyl
chain and the N-substituent is a prefix, however many carbons it has. "SENIORITY ORDER FOR PARENT STRUCTURES"
(:18875): "The senior parent structure has the maximum number of substituents corresponding to the principal
characteristic group (suffix)": the acyl chain bears the amide, a ring that does not bear it is no senior parent.

The chain finder took the N-substituent carbon of the amide match as a host as well, so a three-carbon N-alkyl chain beat
the two-carbon acyl chain on length, held the group by no count, tied the ring at none, and the ring won as senior
: the ring cascade named the amide as a ring prefix, no PIN, and the amide namer was never reached.
"""
import pytest

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_TIERS = ["pin", "best-effort"]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


# an N-alkyl of three carbons or more on an acyl chain of two that carries a ring (or a hetero substituent)
_TARGETS = [
    ("CC(C)NC(=O)Cc1ccccc1", "2-phenyl-N-(propan-2-yl)acetamide"),
    ("CCCNC(=O)Cc1ccccc1", "2-phenyl-N-propylacetamide"),
    ("CC(C)NC(=O)CN1CCCC1", "N-(propan-2-yl)-2-(pyrrolidin-1-yl)acetamide"),
    ("CC(C)NC(=O)Cc1ccccc1C", "2-(2-methylphenyl)-N-(propan-2-yl)acetamide"),
    ("CCCN(C)C(=O)Cc1ccccc1", "N-methyl-2-phenyl-N-propylacetamide"),
    ("CC(C)N(C)C(=O)Cc1ccccc1", "N-methyl-2-phenyl-N-(propan-2-yl)acetamide"),
    ("CCCCNC(=O)Cc1ccccc1", "N-butyl-2-phenylacetamide"),
    ("CC(C)NC(=O)C(Cl)c1ccccc1", "2-chloro-2-phenyl-N-(propan-2-yl)acetamide"),
    ("CCCNC(=O)Cc1ccc(O)cc1", "2-(4-hydroxyphenyl)-N-propylacetamide"),
    ("CC(C)NC(=O)COc1ccccc1", "2-phenoxy-N-(propan-2-yl)acetamide"),
    ("CCCNC(=O)CSc1ccccc1", "2-(phenylsulfanyl)-N-propylacetamide"),
    ("CCCNC(=O)CNc1ccccc1", "2-anilino-N-propylacetamide"),
    ("CCCNC(=O)Cn1cccc1", "N-propyl-2-(1H-pyrrol-1-yl)acetamide"),
    ("CCCNC(=O)CC1CCOCC1", "2-(oxan-4-yl)-N-propylacetamide"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _TARGETS)
def test_the_acyl_chain_is_the_parent_of_an_n_alkyl_amide(smiles, name, tier):
    """ (the Blue Book): the N-substituents are prefixes with the locant N; the acyl chain is the
    parent, the ring on it a prefix ('2-phenyl-N-(propan-2-yl)acetamide')."""
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# the controls that were named before the chain choice changed
_CONTROLS = [
    ("CCNC(=O)Cc1ccccc1", "N-ethyl-2-phenylacetamide"),
    ("CNC(=O)CN1CCCC1", "N-methyl-2-(pyrrolidin-1-yl)acetamide"),
    ("C1CCCCC1NC(=O)Cc1ccccc1", "N-cyclohexyl-2-phenylacetamide"),
    ("CC(C)NC(=O)CC1CCCC1", "2-cyclopentyl-N-(propan-2-yl)acetamide"),
    # the acyl chain as long as the N-chain, and the ring on the carbonyl carbon (a ring amide is the parent,
    ("CC(C)NC(=O)CCc1ccccc1", "3-phenyl-N-(propan-2-yl)propanamide"),
    ("CC(C)NC(=O)c1ccccc1", "N-(propan-2-yl)benzamide"),
    ("CC(C)NC(=O)C1CCCCC1", "N-(propan-2-yl)cyclohexanecarboxamide"),
    # the acid outranks the amide: the amide is the prefix '2-oxo-2-(propylamino)ethyl' on the benzoic acid
    ("CCCNC(=O)Cc1ccc(C(=O)O)cc1", "4-[2-oxo-2-(propylamino)ethyl]benzoic acid"),
    # an ester outranks a lactone, a pseudoketone "Lactones", the Blue Book: "A lactone, as a
    # pseudoketone, ranks lower in the seniority of classes than an acid or an ester"; 'ethyl 3-(2-oxooxan-3-yl)propanoate
    # (PIN)',:32126): the lactone is the prefix on the acetic acid of the ester
    ("CCOC(=O)CCC1CCCOC1=O", "ethyl 3-(2-oxooxan-3-yl)propanoate"),
    ("CCOC(=O)CC1CCC(=O)O1", "ethyl (5-oxooxolan-2-yl)acetate"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _CONTROLS)
def test_the_controls_keep_their_names(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# the gold rows that name an amide on a chain through a polyfunctional N-substituent, and the formamide PIN whose
# one-carbon acyl is the single-carbon case of the parent choice (it keeps its route: 'formamide', not 'methanamide')
_PIN_ROWS = [
    # gold rows W2E-P1AM-17 and W2E-P1FS-17, the Blue Book verbatim)
    ("CN(CC(O)CO)C(=O)CN", "2-amino-N-(2,3-dihydroxypropyl)-N-methylacetamide"),
    ("O=CN/C=C\\c1ccc(O)cc1", "N-[(Z)-2-(4-hydroxyphenyl)ethenyl]formamide"),
    ("O=CNCCCc1ccccc1", "N-(3-phenylpropyl)formamide"),
]


@pytest.mark.parametrize("smiles,name", _PIN_ROWS)
def test_the_gold_rows_and_the_formamide_keep_their_pins(smiles, name):
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# the group -N(R')-CO-R of a three-carbon R' whose acyl carries a ring is the amido prefix of method (1): the prefix is the
# complete name of the amide with the final 'e' changed to 'o', the Blue Book; '2-(N-methylpropanamido)
# benzene-1-sulfonic acid (PIN)',:33040), and the amide it asks for is now named
_AMIDO = [
    ("OC(=O)c1ccc(N(C(C)C)C(=O)CN2CCCC2)cc1", "4-[N-(propan-2-yl)-2-(pyrrolidin-1-yl)acetamido]benzoic acid"),
    ("OC(=O)c1ccc(N(C(C)C)C(=O)Cc2ccccc2)cc1", "4-[2-phenyl-N-(propan-2-yl)acetamido]benzoic acid"),
    ("OC(=O)c1ccc(N(CCC)C(=O)Cc2ccccc2)cc1", "4-(2-phenyl-N-propylacetamido)benzoic acid"),
    ("COC(=O)c1ccc(N(C(C)C)C(=O)Cc2ccccc2)cc1", "methyl 4-[2-phenyl-N-(propan-2-yl)acetamido]benzoate"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _AMIDO)
def test_a_tertiary_acyl_nitrogen_of_a_three_carbon_substituent_is_the_amido_prefix(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert not row["name"].startswith("N-"), row
    assert_full_rt(row["name"], smiles)


# an acyl chain that carries a heteroatom (a ring nitrogen, an alkoxy, a halogen) and an N-chain that holds an ether, so the
# molecule is polyfunctional: the count-based amide namer sees a bare chain of two carbons and would drop the ring
# ('N-(3-methoxypropyl)acetamide'); the acyl is no simple chain when a heteroatom hangs on it, so no atom is dropped
_HETERO_ON_ACYL = [
    ("COCCCNC(=O)CN1CCCC1", "N-(3-methoxypropyl)-2-(pyrrolidin-1-yl)acetamide"),
    ("COCCNC(=O)CN1CCCC1", "N-(2-methoxyethyl)-2-(pyrrolidin-1-yl)acetamide"),
    ("n1nn(CC(=O)NCCCOC)nc1", "N-(3-methoxypropyl)-2-(2H-tetrazol-2-yl)acetamide"),
    ("Cc1ccc(cc1)c1nn(CC(=O)NCCCOCc2ccccc2)nn1",
     "N-[3-(benzyloxy)propyl]-2-[5-(4-methylphenyl)-2H-tetrazol-2-yl]acetamide"),
    ("COCCCNC(=O)CCl", "2-chloro-N-(3-methoxypropyl)acetamide"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _HETERO_ON_ACYL)
def test_a_heteroatom_substituent_of_the_acyl_chain_is_a_prefix_of_the_amide(smiles, name, tier):
    """ (the Blue Book): the substituents of the acyl parent and the N-substituents are prefixes of
    one amide name; no atom of the molecule is dropped."""
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# the acyl chain is the parent, so the comparison of the chain it is given (the N-substituent's chain of the same
# length bears more prefixes) never takes the N-substituent's chain: it is no host of the amide,
# (the Blue Book) cites it as prefixes with the locant N. The name stayed right at the wider tiers and was lost at
# the PIN tier ('senior_alternative').
_N_CHAIN_OF_THE_SAME_LENGTH = [
    ("ClCC(Cl)C(Cl)NC(=O)CCc1ccccc1", "3-phenyl-N-(1,2,3-trichloropropyl)propanamide"),
    ("OCC(O)C(O)NC(=O)CCc1ccccc1", "3-phenyl-N-(1,2,3-trihydroxypropyl)propanamide"),
    ("ClC(Cl)C(Cl)NC(=O)Cc1ccccc1", "2-phenyl-N-(1,2,2-trichloroethyl)acetamide"),
    ("FC(F)C(F)NC(=O)Cc1ccccc1", "2-phenyl-N-(1,2,2-trifluoroethyl)acetamide"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _N_CHAIN_OF_THE_SAME_LENGTH)
def test_an_n_chain_of_the_acyl_length_with_more_prefixes_is_no_parent(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("tier", _TIERS)
def test_the_amine_on_the_acyl_ring_is_no_oxo_ethyl_substituent_of_aniline(tier):
    """ (the Blue Book) and (:18189): the amide outranks the amine, so the acyl chain is the
    parent ('2-(2-aminophenyl)-N-propylacetamide'); the ring parent '2-[2-oxo-2-(propylamino)ethyl]aniline' was a false
    PIN of the unfixed chain choice."""
    smiles = "CCCNC(=O)Cc1ccccc1N"
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == ("2-(2-aminophenyl)-N-propylacetamide", "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def test_a_hydrogen_isotope_on_the_acyl_chain_is_no_hetero_substituent():
    """The count-based amide namer reads a hydrogen on the chain as no heteroatom (the isotope producer cites it)."""
    from rdkit import Chem

    from orthonym.rules.polyfunctional import _acyl_is_simple_saturated_chain as simple

    def verdict(smiles):
        mol = Chem.MolFromSmiles(smiles)
        carbonyl, _o, nitrogen = mol.GetSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[NX3]"))
        return simple(mol, carbonyl, nitrogen)

    assert verdict("OCCCNC(=O)CC[2H]") and verdict("OCCCNC(=O)CC[3H]")
    assert verdict("OCCCNC(=O)CCC")
    assert not verdict("OCCCNC(=O)CCCl")
    row = _row("OCCCNC(=O)CC[2H]", "pin")
    assert (row["name"], row["tier"]) == ("N-(3-hydroxypropyl)(3-2H1)propanamide", "pin_verified"), row


# the N-substituent carbons of the amide match are no atoms of the parent compound: acetamide keeps '2-' because its N-H
# are substitutable too, the Blue Book: "All locants are omitted for parent compounds when all
# substitutable hydrogen atoms have the same locant"; 'N-carbamoyl-2-phenylacetamide (PIN)',:33364), and an italic N
# locant says which atom carries the group,:2869). With them counted as the group's, the N held no hydrogen,
# the licence held for the chain carbon alone and the prefixes lost their locants ('[N-(2,3-dihydroxypropyl)]
# (N-methyl)(phenyl)acetamide', a false PIN).
_N_SUBSTITUTED_ACETAMIDE = [
    ("CN(CC(O)CO)C(=O)Cc1ccccc1", "N-(2,3-dihydroxypropyl)-N-methyl-2-phenylacetamide"),
    ("OCC(O)CN(CC(O)CO)C(=O)Cc1ccccc1", "N,N-bis(2,3-dihydroxypropyl)-2-phenylacetamide"),
    ("c1ccccc1CC(=O)N(CCO)CCO", "N,N-bis(2-hydroxyethyl)-2-phenylacetamide"),
    ("c1ccccc1CC(=O)N(C)CCO", "N-(2-hydroxyethyl)-N-methyl-2-phenylacetamide"),
    ("N#CCC(=O)N(C)C", "2-cyano-N,N-dimethylacetamide"),
    ("OCC(=O)N(C)C", "2-hydroxy-N,N-dimethylacetamide"),
    ("N#CCC(=O)NCCO", "2-cyano-N-(2-hydroxyethyl)acetamide"),
    # the primary amide holds its N-H: the licence never held for it
    ("NC(=O)CC#N", "2-cyanoacetamide"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _N_SUBSTITUTED_ACETAMIDE)
def test_an_n_substituted_acetamide_cites_every_locant(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
