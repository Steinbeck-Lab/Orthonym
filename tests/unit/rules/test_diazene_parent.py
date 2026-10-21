"""HP lane, a performance pass: the diazene as the senior parent hydride of an azo compound,.

 (the Blue Book) puts the nitrogen above carbon and oxygen as the senior atom of a parent
"whether cyclic or acyclic... to choose between rings and chains", and class 21 (:18197)
holds "diazenes" with the hydrazines. "Azo compounds, R-N=N-R'" (:38746): "(1) substitutively
using the parent 'diazene', HN=NH; (2) by using the prefix 'azo'... Method (1) leads to preferred IUPAC
names" (:38755). "Symmetrical monoazo compounds" (:38759) are named "(1) by substituting the
parent diazene, HN=NH, by the appropriate substituent groups" (:38761), '(3-chlorophenyl)(4-chlorophenyl)
diazene (PIN)' (:38774), '(naphthalen-1-yl)(naphthalen-2-yl)diazene (PIN)' (:38776);
"Unsymmetrical monoazo compounds" (:38778) "(1) substitutively, by prefixing the names of the appropriate
substituent groups, in alphabetical order, before the parent hydride name diazene" (:38780),
'ethenyl(methyl)diazene (PIN)' (:38787), '(naphthalen-2-yl)(phenyl)diazene (PIN)' (:38789). "Monoazo
compounds... in which R is substituted by a principal characteristic group are named on the basis of the
parent hydride, RH, substituted by an organyl diazenyl group" (:38791): '4-(phenyldiazenyl)benzene-1-sulfonic
acid (PIN)' (:38798). The producer is
``rules.polyazane.name_polyazane``; the question "is the diazene the parent" is asked by
``validation.spelling.polyazane_parent.diazene_is_senior_parent`` and by the spelling check of the 'diazenyl' prefix
alike, so the producer names exactly the molecules for which that check calls the prefix name wrong.
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.polyazane_parent import diazene_is_senior_parent, hydrazine_is_senior_parent
from orthonym.rules.polyazane import name_polyazane
from orthonym.validation.pin_spelling import check_pin_spelling
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default

# (smiles, the PIN both tiers ship at pin_verified; OPSIN 2.9.0 reads it back to the full key)
PIN_ROWS = [
    #:38774 and:38776 verbatim
    ("Clc1cccc(N=Nc2ccc(Cl)cc2)c1", "(3-chlorophenyl)(4-chlorophenyl)diazene"),
    ("c1ccc2ccccc2c1N=Nc1ccc2ccccc2c1", "(naphthalen-1-yl)(naphthalen-2-yl)diazene"),
    # a halogen on the ring: the first cited prefix is the first in the alphanumerical order (c before p),
    # and each prefix is in marks when it is compound
    ("Clc1ccc(N=Nc2ccccc2)cc1", "(4-chlorophenyl)(phenyl)diazene"),
    ("CN=Nc1ccc(Cl)cc1", "(4-chlorophenyl)(methyl)diazene"),
    ("Cc1ccc(N=Nc2ccc(OC)cc2)cc1", "(4-methoxyphenyl)(4-methylphenyl)diazene"),
    ("FC(F)(F)c1ccc(N=Nc2ccccc2)cc1", "phenyl[4-(trifluoromethyl)phenyl]diazene"),
    # the same prefix twice: 'bis' for a compound prefix
    ("Clc1ccc(N=Nc2ccc(Cl)cc2)cc1", "bis(4-chlorophenyl)diazene"),
    # an ether in a chain or on a ring (class 41), a nitro group (a prefix only), a ring of oxygen
    ("CN=Nc1ccc(OC)cc1", "(4-methoxyphenyl)(methyl)diazene"),
    ("COCCN=NC", "(2-methoxyethyl)(methyl)diazene"),
    ("O=[N+]([O-])c1ccc(N=Nc2ccccc2)cc1", "(4-nitrophenyl)(phenyl)diazene"),
    ("c1ccccc1N=NC1CCOC1", "(oxolan-3-yl)(phenyl)diazene"),
    # a nitroso group on carbon,:25933), a peroxide (class 42,:18218): below the diazene
    ("O=Nc1ccc(N=Nc2ccccc2)cc1", "(4-nitrosophenyl)(phenyl)diazene"),
    ("CN=Nc1ccc(N=O)cc1", "methyl(4-nitrosophenyl)diazene"),
    ("CC(C)(C)OOc1ccc(N=Nc2ccccc2)cc1", "[4-(tert-butylperoxy)phenyl](phenyl)diazene"),
    # the mono-substituted diazene
    ("Clc1ccc(N=N)cc1", "(4-chlorophenyl)diazene"),
    # the geometry of the N=N bond is kept,:44643: no locant on a parent that cites none), with the
    # configuration of a stereogenic group of a prefix
    ("Clc1ccc(/N=N/c2ccccc2)cc1", "(E)-(4-chlorophenyl)(phenyl)diazene"),
    ("Clc1ccc(/N=N\\c2ccccc2)cc1", "(Z)-(4-chlorophenyl)(phenyl)diazene"),
    ("C[C@H](CC)/N=N/c1ccc(Cl)cc1", "(E)-[(2R)-butan-2-yl](4-chlorophenyl)diazene"),
    ("C/C=C/N=N/c1ccc(Cl)cc1", "(E)-(4-chlorophenyl)[(1E)-prop-1-en-1-yl]diazene"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_the_diazene_is_the_parent_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# the names that were right and stay: the book's own examples (:38768-:38789) and the C/N-only azo compounds
UNCHANGED = [
    ("c1ccccc1N=Nc1ccccc1", "diphenyldiazene"),
    ("CN=NC", "dimethyldiazene"),
    ("C=CN=NC", "ethenyl(methyl)diazene"),
    ("c1ccccc1N=Nc1ccc2ccccc2c1", "(naphthalen-2-yl)(phenyl)diazene"),
    ("c1ccc(/N=N\\c2ccccc2)cc1", "(Z)-diphenyldiazene"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", UNCHANGED)
def test_the_names_that_were_right_stay(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# The diazene is NOT the parent: a principal characteristic group is present (it is the suffix and the azo
# group a 'diazenyl' prefix,:38798), a nitrogen ring is (a heterocycle precedes the diazenes in class 21),
# or there are two azo groups (the bis(azo) case of,:38815, which the producer does not
# decide). The names that were right stay.
NOT_THE_PARENT = [
    ("OC(=O)c1ccc(N=Nc2ccccc2)cc1", "4-(phenyldiazenyl)benzoic acid"),
    ("c1ccccc1N=Nc1ccc(N)cc1", "4-(phenyldiazenyl)aniline"),
    ("c1ccccc1N=Nc1ccc(O)cc1", "4-(phenyldiazenyl)phenol"),
    ("c1ccccc1N=Nc1ccccn1", "2-(phenyldiazenyl)pyridine"),
]


@pytest.mark.parametrize("smiles", [s for s, _ in NOT_THE_PARENT] + ["c1ccccc1N=Nc1ccc(N=Nc2ccccc2)cc1"])
def test_the_producer_declines_where_the_diazene_is_not_the_parent(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert not diazene_is_senior_parent(mol)
    assert name_polyazane(mol) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", NOT_THE_PARENT)
def test_the_names_where_the_diazene_is_not_the_parent_stay(smiles, name):
    assert_pin_at_both_tiers(smiles, name)


# the structural question, on its own
@pytest.mark.parametrize("smiles,expected", [
    ("Clc1ccc(N=Nc2ccccc2)cc1", True),
    ("c1ccccc1N=Nc1ccccc1", True),
    ("CN=Nc1ccc(OC)cc1", True),
    ("O=[N+]([O-])c1ccc(N=Nc2ccccc2)cc1", True),
    ("C/N=N/C", True),
    ("Clc1ccc(N=N)cc1", True),                              # R-N=NH
    ("O=Nc1ccc(N=Nc2ccccc2)cc1", True),                     # nitroso on carbon: a prefix only (:25933)
    ("CC(C)(C)OOc1ccc(N=Nc2ccccc2)cc1", True),              # a peroxide: class 42 (:18218)
    ("CSSc1ccc(N=Nc2ccccc2)cc1", True),                     # a disulfide
    ("OOc1ccc(N=Nc2ccccc2)cc1", False),                     # a hydroperoxide: class 18
    ("OC(=O)c1ccc(N=Nc2ccccc2)cc1", False),                 # a carboxylic acid
    ("c1ccccc1N=Nc1ccc(O)cc1", False),                      # a hydroxy group
    ("c1ccccc1N=Nc1ccc(N)cc1", False),                      # an amine
    ("c1ccccc1N=Nc1ccccn1", False),                         # a nitrogen ring
    ("c1ccccc1N=Nc1ccc(N=Nc2ccccc2)cc1", False),            # two azo groups
    ("CN=NN(C)C", False),                                   # a triazene (a chain of three nitrogen atoms)
    ("CN=[N+]([O-])C", False),                              # an azoxy compound
    ("c1ccc2c(c1)ccnn2", False),                            # the N=N bond is in a ring
    ("NNc1ccc(Cl)cc1", False),                              # a hydrazine, not a diazene
    ("Clc1ccc(N=Nc2ccccc2)cc1.O", False),                   # a second component
])
def test_the_diazene_is_the_senior_parent(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert diazene_is_senior_parent(mol) is expected
    # one unit at a time: no molecule is the parent of both
    assert not (diazene_is_senior_parent(mol) and hydrazine_is_senior_parent(mol))


# the spelling check of: a 'diazenyl' prefix is no PIN where the diazene is the parent hydride
@pytest.mark.parametrize("smiles,name,fails", [
    ("Clc1ccc(N=Nc2ccccc2)cc1", "1-chloro-4-(phenyldiazenyl)benzene", True),         # the name the producer replaced
    ("CN=Nc1ccc(OC)cc1", "1-methoxy-4-(methyldiazenyl)benzene", True),
    ("Clc1ccc(N=Nc2ccccc2)cc1", "(4-chlorophenyl)(phenyl)diazene", False),
    ("OC(=O)c1ccc(N=Nc2ccccc2)cc1", "4-(phenyldiazenyl)benzoic acid", False),       #:38798 style: an acid is senior
    ("c1ccccc1N=Nc1ccccn1", "2-(phenyldiazenyl)pyridine", False),                    # a nitrogen ring
    ("c1ccccc1N=Nc1ccc(N=Nc2ccccc2)cc1", "1,4-bis(phenyldiazenyl)benzene", False),   # bis(azo): not decided here
    ("c1ccccc1N=Nc1ccc(O)cc1", "4-(phenyldiazenyl)phenol", False),
    ("O=Nc1ccc(N=Nc2ccccc2)cc1", "1-nitroso-4-(phenyldiazenyl)benzene", True),         # nitroso is a prefix only
    ("CC(C)(C)OOc1ccc(N=Nc2ccccc2)cc1", "1-(tert-butylperoxy)-4-(phenyldiazenyl)benzene", True),
    ("O=Nc1ccc(N=Nc2ccccc2)cc1", "(4-nitrosophenyl)(phenyl)diazene", False),
    ("Clc1ccc(N=Nc2ccccc2)cc1.O", "1-chloro-4-(phenyldiazenyl)benzene", False),      # a second component
])
def test_the_diazene_parent_check(smiles, name, fails):
    rules = [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)]
    assert ("P-44.1.2" in rules) is fails, rules


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["Clc1ccc(N=Nc2ccccc2)cc1", "CN=Nc1ccc(OC)cc1"])
def test_the_diazenyl_name_is_never_a_pin_label(smiles):
    """The default tier names the molecule on the diazene, and the best-effort tier ships the same name:
    no tier labels a 'diazenyl' name of it a PIN."""
    for res in (name_default(smiles), name_breadth(smiles)):
        assert "diazenyl" not in (res.get("name") or ""), res
