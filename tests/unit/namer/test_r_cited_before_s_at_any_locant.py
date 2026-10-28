"""R is cited before S whatever locant each is attached at (names).

``### **** Sequence Rule **4a**`` (``the Blue Book``), its example (``:45465``)::

    (2R,3s,4S,6R)-2,6-dichloro-5-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]heptan-4-ol (PIN)

cites the (1R)-chloroethyl first, at the HIGHER locant 5; ``****`` (``:22603``, under ``### ****
CRITERIA RELATED ONLY TO CONFIGURATION``): "When names based on alphanumerical order and isotopic descriptors are the
same, further choice depends on the alphabetic order of the stereochemical descriptors 'R' and 'S'". The locant a
prefix is attached at is no tier of the order of citation: ``****`` (``:3517``, under ``### ****
ALPHANUMERICAL ORDER``) compares only the locants INSIDE the prefix names ("the fact that 'N' is lower than '4' is
irrelevant", ``:3527``).

Every row below named the S prefix first: the sites that write the order of citation tied the two names in
and let the locant they are attached at, or the order the atoms were written in, decide. The numbering of each
row is fixed by the parent (a suffix, a heteroatom, an N), so the R prefix stands at the higher locant. The rows
whose numbering is a choice (a symmetric chain, ring or ring system) pin only the order of citation.
"""
import re

import pytest

from orthonym import name_compound
from tests.support.jars import jar_or_skip
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

#: (the site that wrote the order, SMILES, the name). The numbering is fixed, R is cited first at the higher locant.
FIXED_NUMBERING = [
    ("polyamine N-substituents (composer._assemble_polyamine_name)",
     "CC[C@@H](C)NC(C)CCN[C@@H](C)CC",
     "N3-[(2R)-butan-2-yl]-N1-[(2S)-butan-2-yl]butane-1,3-diamine"),
    ("ring prefixes of a ketone (name_tree, handlers._handler_shared, composer ring prefixes)",
     "CC[C@@H](C)C1CCC(=O)C([C@@H](C)CC)C1",
     "4-[(2R)-butan-2-yl]-2-[(2S)-butan-2-yl]cyclohexan-1-one"),
    ("the same on a carboxylic acid ring",
     "CC[C@@H](C)c1ccc(C(=O)O)c([C@@H](C)CC)c1",
     "4-[(2R)-butan-2-yl]-2-[(2S)-butan-2-yl]benzoic acid"),
    ("benzene prefixes with an N (rules.benzene._build_prefix_string_with_locants)",
     "CC[C@@H](C)NS(=O)(=O)c1ccc([C@@H](C)CC)cc1",
     "N-[(2R)-butan-2-yl]-4-[(2S)-butan-2-yl]benzene-1-sulfonamide"),
    ("benzamide, the N before the 4",
     "CC[C@@H](C)NC(=O)c1ccc([C@@H](C)CC)cc1",
     "N-[(2R)-butan-2-yl]-4-[(2S)-butan-2-yl]benzamide"),
    ("amine prefixes, the 3 before the N (composer._assemble_amine_name)",
     "CC[C@@H](C)C1CCCC(N[C@@H](C)CC)C1",
     "3-[(2R)-butan-2-yl]-N-[(2S)-butan-2-yl]cyclohexan-1-amine"),
    ("fused heterocycle (rules.fused_rings)",
     "CC[C@@H](C)c1cccc2oc([C@@H](C)CC)cc12",
     "4-[(2R)-butan-2-yl]-2-[(2S)-butan-2-yl]-1-benzofuran"),
    ("fused heterocycle, R at the lower locant (it was cited second)",
     "CC[C@@H](C)c1c[nH]c2ccc([C@@H](C)CC)cc12",
     "3-[(2R)-butan-2-yl]-5-[(2S)-butan-2-yl]-1H-indole"),
    ("spiro parent (composer._enrich_complex_ring_with_subs)",
     "CC[C@@H](C)C1CC([C@@H](C)CC)C2(CCCCC2)C1",
     "3-[(2R)-butan-2-yl]-1-[(2S)-butan-2-yl]spiro[4.5]decane"),
    ("hydrogenated ring system", "CC[C@@H](C)C1CCc2ccccc2C1[C@@H](C)CC",
     "2-[(2R)-butan-2-yl]-1-[(2S)-butan-2-yl]-1,2,3,4-tetrahydronaphthalene"),
    ("ring substituent (rules.ring_substituents)",
     "C[C@H](Cl)c1cc([C@@H](C)Cl)ccc1CC(=O)O",
     "{4-[(1R)-1-chloroethyl]-2-[(1S)-1-chloroethyl]phenyl}acetic acid"),
    ("lactam (rules.lactams)", "C[C@H](Cl)C1CC([C@@H](C)Cl)NC1=O",
     "5-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]pyrrolidin-2-one"),
    ("chain with alkoxy prefixes (decomposition.weave)", "CC[C@@H](C)OC(CO)CO[C@@H](C)CC",
     "2-{[(2R)-butan-2-yl]oxy}-3-{[(2S)-butan-2-yl]oxy}propan-1-ol"),
    ("N-substituents of a guanidine", "CC[C@@H](C)NC(=N)N(C)[C@@H](C)CC",
     "N'-[(2R)-butan-2-yl]-N-[(2S)-butan-2-yl]-N-methylguanidine"),
    ("best-effort tier, general engine: a chain with amino prefixes",
     "CC(CCN[C@H](C)Cl)N[C@@H](C)Cl",
     "3-{[(1R)-1-chloroethyl]amino}-1-{[(1S)-1-chloroethyl]amino}butane"),
]


@pytest.mark.parametrize("site, smiles, expected", FIXED_NUMBERING, ids=[r[0][:60] for r in FIXED_NUMBERING])
def test_r_is_cited_first_at_the_higher_locant(site, smiles, expected):
    jar_or_skip("opsin")
    name = name_compound(smiles)
    assert name == expected
    assert name_is_rt_exact(name, smiles)


#: (SMILES, what the numbering is). The numbering is a choice here: only the order of citation is pinned.
FREE_NUMBERING = [
    ("C[C@H](Cl)SCCS[C@@H](C)Cl", "ethane, either end"),
    ("C[C@H](Cl)c1cccc2c([C@@H](C)Cl)cccc12", "naphthalene 1,5"),
    ("C[C@H](Cl)C1C2CCC(C2)C1[C@@H](C)Cl", "bicyclo[2.2.1]heptane 2,3"),
    ("CC[C@@H](C)OCC(O)CO[C@@H](C)CC", "propan-2-ol, either end"),
    ("C[C@H](Cl)NC1CCC(N[C@@H](C)Cl)CC1", "cyclohexane 1,4"),
    ("CC[C@@H](C)C(C(=O)OC)C(C(=O)OC)[C@@H](C)CC", "butanedioate, either end"),
]


@pytest.mark.parametrize("smiles, why", FREE_NUMBERING, ids=[r[1] for r in FREE_NUMBERING])
def test_r_is_cited_before_s_when_the_numbering_is_a_choice(smiles, why):
    jar_or_skip("opsin")
    name = name_compound(smiles)
    assert re.findall(r"\(\d+([RS])\)", name) == ["R", "S"], name
    assert name_is_rt_exact(name, smiles)


def test_the_book_example_that_the_engine_can_name():
    """``4-{3,4-bis[(1R)-1-chloroethyl]phenoxy}-1,2-bis[(1S)-1-chloroethyl]benzene (PIN)`` (:22597), parsed by OPSIN
    to the SMILES below: the name is the one the book prints."""
    jar_or_skip("opsin")
    smiles = "Cl[C@H](C)C=1C=C(OC2=CC(=C(C=C2)[C@H](C)Cl)[C@H](C)Cl)C=CC1[C@@H](C)Cl"
    assert name_compound(smiles) == ("4-{3,4-bis[(1R)-1-chloroethyl]phenoxy}-1,2-bis[(1S)-1-chloroethyl]"
                                     "benzene")


#: ``****`` (:3517): the locants INSIDE the prefix names decide, not those they are attached at.
P_14_5_4 = [
    ("CCC(C)CC1CCC(=O)C(CCC(C)C)C1", "4-(2-methylbutyl)-2-(3-methylbutyl)cyclohexan-1-one"),
    ("CCC(C)CC1CC(CCC(C)C)CCC1=O", "2-(2-methylbutyl)-4-(3-methylbutyl)cyclohexan-1-one"),
]


@pytest.mark.parametrize("smiles, expected", P_14_5_4)
def test_the_locant_a_prefix_is_attached_at_is_no_tier_of_the_order_of_citation(smiles, expected):
    """``4-(2-methylbutyl)-N-(3-methylbutyl)aniline (PIN)`` (:3523): "for ordering the substituents '2' is lower
    than '3'; the fact that 'N' is lower than '4' is irrelevant" (:3527); ``6-(1-chloroethyl)-5-(2-chloroethyl)-1H-indole
    (PIN)`` (:3529) cites the prefix at the higher locant first."""
    jar_or_skip("opsin")
    assert name_compound(smiles) == expected
    assert name_is_rt_exact(expected, smiles)


#: The locants INSIDE the prefix names,:3517) come before the configuration: ``****`` (:22603) is for
#: "names based on alphanumerical order... the same", and 1-chloropropyl and 2-chloropropyl are not. The numbering of
#: the chain is fixed by the suffix, so N1 and N3 stand where they do. Best-effort tier, the general engine: its keys
#: had the letters, then R/S, and a prefix met first was cited first otherwise.
LOCANTS_INSIDE_BEFORE_CONFIGURATION = [
    ("1-chloropropyl first, and it is the S one",
     "CC(CCN[C@@H](Cl)CC)NC[C@@H](C)Cl",
     "1-{[(1S)-1-chloropropyl]amino}-3-{[(2R)-2-chloropropyl]amino}butane"),
    ("1-chloropropyl first, at the higher locant, both R",
     "CC(CCNC[C@@H](C)Cl)N[C@H](Cl)CC",
     "3-{[(1R)-1-chloropropyl]amino}-1-{[(2R)-2-chloropropyl]amino}butane"),
    ("1-chloropropyl first, at the higher locant, S and R",
     "CC(CCNC[C@@H](C)Cl)N[C@@H](Cl)CC",
     "3-{[(1S)-1-chloropropyl]amino}-1-{[(2R)-2-chloropropyl]amino}butane"),
]


@pytest.mark.parametrize("label, smiles, expected", LOCANTS_INSIDE_BEFORE_CONFIGURATION,
                         ids=[r[0] for r in LOCANTS_INSIDE_BEFORE_CONFIGURATION])
def test_the_locants_inside_the_names_come_before_the_configuration(label, smiles, expected):
    jar_or_skip("opsin")
    name = name_compound(smiles)
    assert name == expected
    assert name_is_rt_exact(name, smiles)


#: C-prefixes of a polyamine carry their attachment locant in their text ('2-[(2S)-butan-2-yl]'); the numbering is
#: fixed by the amines and their N-methyls, so R stands at the higher locant and is cited first (:45465).
POLYAMINE_C_PREFIXES = [
    ("pentane-1,5-diamine, two C-prefixes", "CNCC(CC(CN(C)C)[C@H](C)Cl)[C@@H](C)Cl",
     "4-[(1R)-1-chloroethyl]-2-[(1S)-1-chloroethyl]-N1,N1,N5-trimethylpentane-1,5-diamine"),
    ("butane-1,4-diamine, a C-prefix and an N-prefix", "CC[C@@H](C)C(CN)CCN[C@@H](C)CC",
     "3-[(2R)-butan-2-yl]-N1-[(2S)-butan-2-yl]butane-1,4-diamine"),
]


@pytest.mark.parametrize("label, smiles, expected", POLYAMINE_C_PREFIXES, ids=[r[0] for r in POLYAMINE_C_PREFIXES])
def test_r_is_cited_first_among_the_c_prefixes_of_a_polyamine(label, smiles, expected):
    jar_or_skip("opsin")
    name = name_compound(smiles)
    assert name == expected
    assert name_is_rt_exact(name, smiles)


#: E and Z prefixes: the book gives no order of citation, so none was added: the order they were met in stands
E_Z = [
    ("C/C=C\\NS(=O)(=O)c1ccc(/C=C/C)cc1",
     "4-[(1E)-prop-1-en-1-yl]-N-[(1Z)-prop-1-en-1-yl]benzene-1-sulfonamide"),
    ("C/C=C\\C1CCC(=O)C(/C=C/C)C1",
     "2-[(1E)-prop-1-en-1-yl]-4-[(1Z)-prop-1-en-1-yl]cyclohexan-1-one"),
]


@pytest.mark.parametrize("smiles, was", E_Z)
def test_the_order_of_an_e_and_a_z_prefix_is_not_changed(smiles, was):
    jar_or_skip("opsin")
    name = name_compound(smiles)
    assert name_is_rt_exact(name, smiles)
    # these two names are the ones the engine gave before; the order of the E and the Z prefix is the same
    assert re.findall(r"\(\d+([EZ])\)", name) == re.findall(r"\(\d+([EZ])\)", was), name
