"""Atom-claim reconciliation of the prefix fragments (``handlers/prefix_claims.py``).

PB lane S, tasks S1 + S2. The functional-group loop of ``_generate_prefixes`` used to
spell an atom a branch prefix had already spelled ('...sulfanylpropanoic acid' for a
tripeptide, '5-carbamoylpentanoic acid' for glutamine); withheld those names, so
the molecules were declined. Rules cited:

* "Substituents of the types -NH-CO-R and -NH-SO2-R" (the Blue Book):
  "Method (1) generates preferred IUPAC names" ('4-formamidobenzoic acid (PIN)').
* "Amic acids" (the Blue Book): "The combination of the prefixes 'amino'
  and 'oxo' is used for describing the -CO-NH2 at the end of an acyclic chain in preferred
  IUPAC names" ('4-amino-4-oxobutanoic acid (PIN)', '3-carbamoylpropanoic acid' is not).
* (:38644) and (:34548): hydrazine branches are written
  '2-carbamoylhydrazin-1-yl' / '...hydrazido', never '[(carbamoylamino)amino]'.
* (:33308): 'urea' is the preferred IUPAC name; 'N-(propan-2-ylidene)urea (PIN)' (:33332), never
  '1-(propan-2-ylideneamino)formamide' (a urea is not named on 'formamide').
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.composer import NameFragment
from orthonym.assembly.handlers.prefix_claims import reconcile_prefix_claims

_BREADTH = dict(general_fallback=True, general_fallback_unverified=True,
                allow_aromatic_general=True)


def _frag(text, atoms, locants=()):
    return NameFragment(text=text, locants=tuple(locants), fragment_type="prefix",
                        atoms=None if atoms is None else frozenset(atoms))


def _features(smiles, chain=(), amides=()):
    return SimpleNamespace(mol=Chem.MolFromSmiles(smiles), principal_chain=list(chain),
                           oriented_ring=None, functional_groups={"primary_amide": list(amides)})


# --------------------------------------------------------------------------- S1 (unit)

@pytest.mark.unit
class TestDoubleSpelling:
    SMILES = "NC(CS)C(=O)NC(C)C(=O)O"  # Cys-Ala; atoms 0 N, 1 C, 2 C, 3 S, 4 C, 5 O, 6 N,...

    def test_strict_subset_is_dropped(self):
        branch = _frag("2-(2-amino-3-sulfanylpropanamido)", range(0, 7), (2,))
        thiol = _frag("sulfanyl", (2, 3))
        out = reconcile_prefix_claims(_features(self.SMILES), [branch, thiol])
        assert [f.text for f in out] == ["2-(2-amino-3-sulfanylpropanamido)"]

    def test_group_anchored_on_the_chain_is_kept(self):
        branch = _frag("2-(2-amino-3-sulfanylpropanamido)", range(0, 7), (2,))
        chain_oxo = _frag("oxo", (7, 8), (1,))  # contains a chain atom: never inside a branch
        out = reconcile_prefix_claims(_features(self.SMILES), [branch, chain_oxo])
        assert [f.text for f in out] == [branch.text, "oxo"]

    def test_equal_atoms_and_unreported_atoms_are_kept(self):
        a = _frag("sulfanyl", (2, 3))
        b = _frag("thio", (2, 3))
        c = _frag("hydroxy", None)
        out = reconcile_prefix_claims(_features(self.SMILES), [a, b, c])
        assert [f.text for f in out] == ["sulfanyl", "thio", "hydroxy"]

    def test_hydrazine_branch_leaves_the_list_unchanged(self):
        smiles = "NC(=O)NNCCC(=O)O"  # atoms 0 N, 1 C, 2 O, 3 N, 4 N,...
        branch = _frag("3-[(carbamoylamino)amino]", (0, 1, 2, 3, 4), (3,))
        dup = _frag("carbamoyl", (0, 1, 2))
        prefixes = [branch, dup]
        out = reconcile_prefix_claims(_features(smiles), prefixes)
        assert out is prefixes


    @pytest.mark.parametrize("smiles,branch_atoms,group_atoms,chain", [
        # -S-C#N: '(cyanosulfanyl)methyl' for the group whose preferred prefix is 'thiocyanato'
        ("OC(=O)CC(CSC#N)C", (5, 6, 7, 8), (6, 7, 8), (1, 3, 4, 9)),
        # -O-C#N: cyanato; -N=C=S: isothiocyanato; -N=C=O: isocyanato
        ("OC(=O)CC(COC#N)C", (5, 6, 7, 8), (6, 7, 8), (1, 3, 4, 9)),
        ("OC(=O)CC(CN=C=S)C", (5, 6, 7, 8), (6, 7, 8), (1, 3, 4, 9)),
        ("OC(=O)CC(CN=C=O)C", (5, 6, 7, 8), (6, 7, 8), (1, 3, 4, 9)),
    ])
    def test_pseudohalide_prefix_is_not_dropped(self, smiles, branch_atoms, group_atoms, chain):
        """ (the Blue Book): 'thiocyanato' for -S-CN, 'cyanato' for -O-CN; the branch
        namer's '(cyanosulfanyl)methyl' is not the preferred spelling, so the list stays as built."""
        branch = _frag("3-[(cyanosulfanyl)methyl]", branch_atoms, (3,))
        group = _frag("thiocyanato", group_atoms)
        prefixes = [branch, group]
        assert reconcile_prefix_claims(_features(smiles, chain=chain), prefixes) is prefixes

    def test_a_nitrile_on_carbon_is_not_a_pseudohalide(self):
        smiles = "OC(=O)CC(CC#N)C"  # 5 CH2, 6 C#N, 7 N (no chalcogen on the nitrile carbon)
        branch = _frag("3-(cyanomethyl)", (5, 6, 7), (3,))
        dup = _frag("cyano", (6, 7))
        out = reconcile_prefix_claims(_features(smiles, chain=(1, 3, 4, 8)), [branch, dup])
        assert [f.text for f in out] == [branch.text]

    def test_carbonic_acid_parent_leaves_the_list_unchanged(self):
        smiles = "CC(C)=NC(N)=O"  # 0 C, 1 C, 2 C, 3 N, 4 C(=O), 5 N, 6 O
        branch = _frag("(propan-2-ylideneamino)", (0, 1, 2, 3), (1,))
        dup = _frag("imino", (1, 3))
        prefixes = [branch, dup]
        out = reconcile_prefix_claims(_features(smiles, chain=(4,)), prefixes)
        assert out is prefixes

    def test_aminal_parent_without_a_double_bond_is_not_a_carbonic_acid_parent(self):
        smiles = "CC(C)=NC(N)N"  # 0 C, 1 C, 2 C, 3 N, 4 CH(N)(N), 5 N, 6 N: two N, no C=X
        branch = _frag("(propan-2-ylideneamino)", (0, 1, 2, 3), (1,))
        dup = _frag("imino", (1, 3))
        out = reconcile_prefix_claims(_features(smiles, chain=(4,)), [branch, dup])
        assert [f.text for f in out] == [branch.text]

    def test_formamide_parent_is_not_a_carbonic_acid_parent(self):
        smiles = "CC(C)=NC=O"  # 0 C, 1 C, 2 C, 3 N, 4 C(=O), 5 O
        branch = _frag("(propan-2-ylideneamino)", (0, 1, 2, 3), (1,))
        dup = _frag("imino", (1, 3))
        out = reconcile_prefix_claims(_features(smiles, chain=(4,)), [branch, dup])
        assert [f.text for f in out] == [branch.text]


# --------------------------------------------------------------------------- S2 (unit)

@pytest.mark.unit
class TestChainAmide:
    SMILES = "NC(=O)CCC(=O)O"  # 0 N, 1 C(amide), 2 O, 3 C, 4 C, 5 C(acid), 6 O, 7 O

    def test_chain_amide_becomes_amino_and_oxo(self):
        feats = _features(self.SMILES, chain=(5, 4, 3, 1), amides=[(1, 2, 0)])
        out = reconcile_prefix_claims(feats, [_frag("carbamoyl", (0, 1, 2), (4,))])
        assert sorted((f.text, f.locants) for f in out) == [("amino", (4,)), ("oxo", (4,))]

    def test_amide_off_the_chain_stays_carbamoyl(self):
        smiles = "NC(=O)C(C)CC(=O)O"  # 0 N, 1 C, 2 O, 3 C(H), 4 CH3, 5 C, 6 C(acid)
        feats = _features(smiles, chain=(6, 5, 3, 4), amides=[(1, 2, 0)])
        prefixes = [_frag("carbamoyl", (0, 1, 2), (3,))]
        assert reconcile_prefix_claims(feats, prefixes) == prefixes


# --------------------------------------------------------------------------- end to end

# (smiles, PIN, the wrong name the producer used to build)
S1_ROWS = [
    ("CC/C(O)=C/CCCC(=O)NCC(=O)O",
     "[(5Z)-6-hydroxyoct-5-enamido]acetic acid",
     "hydroxy[(5Z)-6-hydroxyoct-5-enamido]acetic acid"),
    ("C[C@H](NC(=O)[C@@H](N)CS)C(=O)N[C@@H](CO)C(=O)O",
     "(2S)-2-{(2S)-2-[(2R)-2-amino-3-sulfanylpropanamido]propanamido}-3-hydroxypropanoic acid",
     "(2S)-2-[(2R)-2-amino-3-sulfanylpropanamido]sulfanylpropanoic acid"),
    ("CSCC[C@H](NC(=O)[C@H](CS)NC(=O)[C@H](C)N)C(=O)O",
     "(2S)-2-{(2R)-2-[(2S)-2-aminopropanamido]-3-sulfanylpropanamido}-4-(methylsulfanyl)butanoic acid",
     "4-(methylsulfanyl)sulfanylbutanoic acid"),
    ("N=C(N)NCCC[C@H](N)C(=O)N[C@@H](CO)C(=O)O",
     "(2S)-2-[(2S)-2-amino-5-(carbamimidoylamino)pentanamido]-3-hydroxypropanoic acid",
     "(carbamimidoylamino)-3-hydroxypropanoic acid"),
]
S1_ROWS.append((
    "CCN(CC)CCNCCC(=O)NC(C)C",   # p62624852 (PubChem sample)
    "3-{[2-(diethylamino)ethyl]amino}-N-(propan-2-yl)propanamide",
    "amino-3-{[2-(diethylamino)ethyl]amino}"))
S2_ROWS = [
    ("C[C@@H](O)[C@H](N)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CCC(N)=O)C(=O)O",
     "(2S)-5-amino-2-{(2S)-5-amino-2-[(2S,3R)-2-amino-3-hydroxybutanamido]-5-oxopentanamido}-5-oxopentanoic acid",
     "(2S)-2-[(2S,3R)-2-amino-3-hydroxybutanamido]-5-carbamoylpentanoic acid"),
    ("C[C@H](N)C(=O)NCC(=O)N[C@@H](CC(N)=O)C(=O)O",
     "(2S)-4-amino-2-{2-[(2S)-2-aminopropanamido]acetamido}-4-oxobutanoic acid",
     "4-carbamoylbutanoic acid"),
]


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,pin,old", S1_ROWS + S2_ROWS)
def test_default_tier_names_the_pin_and_not_the_old_wrong_name(smiles, pin, old):
    row = Orthonym().name_tiered(smiles)
    assert row["tier"] == "pin_verified", row
    assert row["name"] == pin
    assert old not in row["name"]
    assert "carbamoyl" not in row["name"]  # no row above has an amide that belongs in a 'carbamoyl'


# Negative controls: names that are NOT a PIN and must never reach pin_verified.
# The first four are the brief's trace 4 (breadth names the PIN promotion re-run builds
# but the Blue Book does not choose); the last three are Blue Book hydrazine rows the
# reconciliation must not unlock (the branch namer spells their N-N link
# '[(carbamoylamino)amino]', the Blue Book '2-carbamoylhydrazin-1-yl' / '...hydrazido').
NEGATIVE_CONTROLS = [
    # ('CC(=S)c1ccc(C(=O)O)cc1' left this list: '4-(ethanethioyl)benzoic acid', the Blue Book) is
    # built by the strict path, test_pin_class_t12_acid_suffixes.py)
    "O=[N+]([O-])c1cnc2cc(Sc3ccc4c([N+](=O)[O-])c(Cl)cnc4c3)ccc2c1Cl",
    "N#CS(=O)(=O)C1CCCCC1C(=O)O",
    "CCC(=O)C1CCCCPC1=O",
    "NC(=O)NNCCC(=O)O",          # bb2005, PIN 3-(2-carbamoylhydrazin-1-yl)propanoic acid
    "N=CNNCCC(=O)O",             # bb2661, PIN 3-(methanimidohydrazido)propanoic acid
    "COC(=O)CNNC(C)=N",          # bb2796, PIN methyl (ethanimidohydrazido)acetate
    "CC(C)=NC(N)=O",             # bb3060, PIN N-(propan-2-ylidene)urea
    "OC(=O)CC(CSC#N)CC([N+](=O)[O-])C",  #: thiocyanato, not '(cyanosulfanyl)methyl'
]


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles", NEGATIVE_CONTROLS)
def test_negative_controls_stay_below_pin_verified(smiles):
    assert Orthonym().name_tiered(smiles)["tier"] != "pin_verified"


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
def test_hydrazine_row_keeps_its_best_effort_label():
    """bb769: the reconciliation used to relabel it pin_unverified with a non-PIN form."""
    row = Orthonym(**_BREADTH).name_tiered("NNC(=O)OC(=O)NNCCCC(=O)O")
    assert row["tier"] == "systematic_verified", row
