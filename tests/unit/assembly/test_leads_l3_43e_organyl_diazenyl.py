"""Leads program L3, item 43e: an aromatic azo compound with a principal characteristic group is named
on the parent hydride of the group, substituted by an organyl diazenyl prefix.

 'Unsymmetrical monoazo compounds are named in two ways' (the Blue Book):
"Monoazo compounds with the general structure R-N=N-R' in which R is substituted by a principal
characteristic group are named on the basis of the parent hydride, RH, substituted by an organyl
diazenyl group, R'-N=N-" (:38791); '4-(phenyldiazenyl)benzene-1-sulfonic acid (PIN)' (:38798),
'1-[(4-chloro-2-methylphenyl)diazenyl]naphthalen-2-amine (PIN)' (:38806). "The prefix 'diazenyl' is a
preselected prefix",:38732), '(methyldiazenyl)acetic acid (PIN)' (:38740). The aniline
parent: 'Nitrogenous compounds' (:17718, 'aniline (PIN)':17722).

``name_substituent_fragment`` had no step for a fragment rooted at an -N=N- unit: it was called once
for the -N=N-Ph branch at the pin tier and returned None, so every aromatic azo compound with a
principal suffix abstained there (10 of 10 pin rows of 9 evaluation molecules) and, with the validity
gate off, the amine handler wrote 'diazenyl-N,N-diethylanamine'. The fragment is now composed from its
parts by ``hetero_group_prefixes.group_shape`` / ``compose_group`` (the primitive the general engine
uses for the group), and the amine assembler declines a parent that has no name.

Every PIN below is read back to the molecule by a fresh OPSIN call and is pin_verified at the pin and
the best-effort tier. The multiplicative 'diazenediyl' case and bis(azo) assemblies are out of scope.
"""
import pytest
from rdkit import Chem

from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

ROWS = [
    ("CCN(CC)c1ccc(N=Nc2ccccc2)cc1", "N,N-diethyl-4-(phenyldiazenyl)aniline"),
    ("Nc1ccc(N=Nc2ccccc2)cc1", "4-(phenyldiazenyl)aniline"),
    ("Oc1ccc(N=Nc2ccccc2)cc1", "4-(phenyldiazenyl)phenol"),
    ("OC(=O)c1ccc(N=Nc2ccccc2)cc1", "4-(phenyldiazenyl)benzoic acid"),
    # the book's rows
    ("OS(=O)(=O)c1ccc(cc1)N=Nc1ccccc1", "4-(phenyldiazenyl)benzene-1-sulfonic acid"),                #:38798
    ("Cc1cc(Cl)ccc1N=Nc1c(N)ccc2ccccc12", "1-[(4-chloro-2-methylphenyl)diazenyl]naphthalen-2-amine"),  #:38806
    # a sulfonyl azo compound of dev2000 (once '1-{[4-(phenylamino)phenyl]diazenyl}-3-sulfobenzene'):
    # 'anilino' is the preferred prefix (:17800)
    ("O=S(=O)(O)c1cccc(N=Nc2ccc(Nc3ccccc3)cc2)c1", "3-[(4-anilinophenyl)diazenyl]benzene-1-sulfonic acid"),
    # the unsubstituted azo compound is unchanged
    ("c1ccccc1N=Nc1ccccc1", "diphenyldiazene"),
]


@pytest.mark.parametrize("smiles, pin", ROWS)
def test_the_azo_compound_is_named_on_the_parent_of_its_principal_group(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_the_books_hydroxynaphthyl_row_is_the_name_the_book_prints():
    """'4-[(2-hydroxynaphthalen-1-yl)diazenyl]benzene-1-sulfonic acid (PIN)' (the Blue Book):
    the strict path writes exactly this name and OPSIN reads it back to the molecule. Its label stays
    below pin_verified: the substituted naphthyl prefix comes from the recursive route the label
    policy records as not the PIN (the same for every substituted fused-ring prefix), so the pin tier
    still declines it; the wider tiers give the name. Before, no tier gave this name
    (the valid tier wrote '2-hydroxy-1-{[4-(2-hydroxy-1,3-dioxa-2-thiapropa-1,2-dienyl)phenyl]
    diazenyl}naphthalene')."""
    smiles = "OS(=O)(=O)c1ccc(cc1)N=Nc1c(O)ccc2ccccc12"
    name = "4-[(2-hydroxynaphthalen-1-yl)diazenyl]benzene-1-sulfonic acid"
    assert name_is_rt_exact(name, smiles), name
    from tests.support.default_tier import strict_path_row
    row = strict_path_row(smiles)
    assert row["name"] == name and row["verified"] == "opsin", row
    b = name_breadth(smiles)
    assert b["name"] == name, b


def test_the_n_n_geometry_is_cited_at_the_front_of_the_diazenyl_prefix():
    """(E)-azobenzene-4-carboxylic acid: the descriptor of the N=N bond belongs to the prefix
    , the Blue Book 'When they relate to substituent groups, they are cited at the
    front of the corresponding prefix'), locant 1 of diazenyl."""
    smiles = "OC(=O)c1ccc(cc1)/N=N/c1ccccc1"
    pin = "4-[(1E)-phenyldiazenyl]benzoic acid"
    assert name_is_rt_exact(pin, smiles), pin
    assert_pin_at_both_tiers(smiles, pin)
    z = "OC(=O)c1ccc(cc1)/N=N\\c1ccccc1"
    assert name_default(z)["name"] == "4-[(1Z)-phenyldiazenyl]benzoic acid"


def test_the_fragment_namer_composes_the_prefix_from_its_parts():
    from orthonym.assembly.substituent_naming import name_substituent_fragment
    mol = Chem.MolFromSmiles("Nc1ccc(N=Nc2ccccc2)cc1")
    # atoms 5-12 are the N=N-phenyl group, attached through N5 to ring atom 4
    assert name_substituent_fragment(mol, list(range(5, 13)), 5, [4]) == "phenyldiazenyl"
    mol = Chem.MolFromSmiles("CN=NC")
    assert name_substituent_fragment(mol, [1, 2, 3], 1, [0]) == "methyldiazenyl"


def test_a_part_that_cannot_be_named_fails_the_prefix_closed(monkeypatch):
    import orthonym.assembly.substituent_naming as sn
    mol = Chem.MolFromSmiles("CN=NC")                       # N=N-CH3 attached through atom 1 to atom 0
    real = sn.name_substituent_fragment
    monkeypatch.setattr(
        sn, "name_substituent_fragment",
        lambda m, atoms, attach, parent: None if attach == 3 else real(m, atoms, attach, parent))
    assert sn._diazenyl_prefix(mol, [1, 2, 3], 1, [0]) is None


@pytest.mark.parametrize("smiles, frag, root, parent", [
    ("CNC", [1, 2], 1, 0),                       # an amino group
    ("CN(C)C", [1, 2, 3], 1, 0),
    ("CN=C", [1, 2], 1, 0),                      # an imine
    ("CN=O", [1, 2], 1, 0),                      # nitroso
    ("c1ccccc1[N+]#N", [6, 7], 6, 5),            # a diazonium group
    ("CNN", [1, 2], 1, 0),                       # a hydrazinyl group
    # a hetero atom behind the diazenyl group is another class with its own producers: a triazene
    # (-N=N-NR2,, an azo ether (-N=N-OR) and a thio ether (-N=N-SR)
    ("CN=NN(C)C", [1, 2, 3, 4, 5], 1, 0),
    ("CN=NOC", [1, 2, 3, 4], 1, 0),
    ("CN=NSC", [1, 2, 3, 4], 1, 0),
    ("CN=NN1CCCC1", [1, 2, 3, 4, 5, 6, 7], 1, 0),
])
def test_a_fragment_that_is_not_a_diazenyl_group_is_not_read_as_one(smiles, frag, root, parent):
    from orthonym.assembly.substituent_naming import _diazenyl_prefix
    assert _diazenyl_prefix(Chem.MolFromSmiles(smiles), frag, root, [parent]) is None


@pytest.mark.parametrize("smiles, pin", [
    # a chain parent keeps its producers: the chain handlers cite the perceived azo group as the
    # bare prefix 'diazenyl' as well, so a prefix composed here would be cited twice
    ("NCN=Nc1ccc(C)cc1", "1-[(4-methylphenyl)diazenyl]methanamine"),
    # the principal group (the ester) is on R-prime: the parent is on that side
    ("CC(=O)OCCN=Nc1ccc(C)cc1", "2-[(4-methylphenyl)diazenyl]ethyl acetate"),
    ("C(=O)OCCN=Nc1ccc(C)cc1", "2-[(4-methylphenyl)diazenyl]ethyl formate"),
    # the principal group on R, a junior group on R-prime
    ("OC(=O)c1ccc(N=Nc2ccc(O)cc2)cc1", "4-[(4-hydroxyphenyl)diazenyl]benzoic acid"),
    ("COC(=O)c1ccc(N=Nc2ccccc2)cc1", "methyl 4-(phenyldiazenyl)benzoate"),
])
def test_the_rows_main_names_keep_their_name(smiles, pin):
    """The prefix is composed only for a ring parent that carries the principal group (R) with R-prime
    free of it. Before this guard the first version of the step took the parent of these rows: the
    chain methanamine became '1-diazenyl-1-[(4-methylphenyl)diazenyl]methanamine' (a different molecule),
    the esters became '1-{[2-(acetyloxy)ethyl]diazenyl}-4-methylbenzene' (the ring as parent, the
    ester as a prefix): the pin tier lost three rows that main names. The best-effort tier keeps the
    name it gave (the general engine's) and it reads back."""
    assert name_is_rt_exact(pin, smiles), pin
    d = name_default(smiles)
    assert (d["name"], d["tier"], d["is_pin"]) == (pin, "pin_verified", True), d
    b = name_breadth(smiles)
    assert b["name"] and name_is_rt_exact(b["name"], smiles), b


def test_both_sides_carrying_the_principal_group_is_the_multiplicative_case():
    """'If both R and R' are substituted by the same number of the principal characteristic group, a
    multiplicative name, using the prefix 'diazenediyl'... is preferred' (the Blue Book): out of
    scope here, so no substitutive name is offered at the pin tier (as on main)."""
    assert name_default("OC(=O)c1ccc(N=Nc2ccc(C(=O)O)cc2)cc1")["tier"] != "pin_verified"


def test_the_amine_assembler_declines_a_ring_it_gave_no_parent():
    """'diazenyl-N,N-diethylanamine': the bare suffix with the prefixes in front of it."""
    from rdkit import Chem as _Chem

    import orthonym.assembly.composer as composer
    from orthonym import Orthonym
    smiles = "CCN(CC)c1ccc(N=Nc2ccccc2)cc1"
    eng = Orthonym(style="pin")
    mol = _Chem.MolFromSmiles(smiles)
    feats = eng._perceive(mol, smiles, _Chem.MolToSmiles(mol))
    eng._classify(feats)
    assert composer._assemble_amine_name(feats, "pin") is None


@pytest.mark.xfail(strict=True, reason=(
    "an acyclic azo compound with a principal group is named by rules/polyfunctional.py "
    "('amino(diazenyl)acetic acid', a different molecule the validity gate rejects), a file outside "
    "lane L3; the PIN is '(methyldiazenyl)acetic acid' (BlueBookV2.md:38740). See the lane report."))
def test_the_chain_azo_acid_is_the_book_pin():
    assert_pin_at_both_tiers("OC(=O)CN=NC", "(methyldiazenyl)acetic acid")
