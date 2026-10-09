"""The compound prefix of a ring system joined by one heteroatom ('(R)amino', '(R)oxy',
'(R)sulfanyl', 'anilino', 'phenoxy') is decided from the atoms and from what the ring-yl
writer says it built, never from the text of the ring-yl name
(``rules.ring_substituents._bare_connective_ring_yl``).

- (the Blue Book): "The prefix name 'anilino' is retained as the preferred
  prefix for C6H5-NH- with full substitution allowed"; '4-chloroanilino' (:26153).
- / the prefix table (:17796): 'phenoxy (preferred prefix) (full substitution;
  see '; '2-(4-bromo-2-carboxyphenoxy)-5-fluorobenzoic acid (PIN)' (:6287).
- (:7255): a simple prefix is enclosed "to separate locants"; one without a
  locant is cited bare ('phenylsulfanyl').
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.rules import ring_substituents as rs

pytestmark = pytest.mark.opsin_gate

NAPHTHOIC = "OC(=O)c1ccc2cc(%s)ccc2c1"


def _fragment(connective_smiles):
    """(mol, frag_set, atoms of the ring-yl with its decorations, hetero atom, the ring atom
    the heteroatom is bonded to) of the substituent ``connective_smiles`` on the naphthoic
    acid; the heteroatom is atom 9 of the template."""
    mol = Chem.MolFromSmiles(NAPHTHOIC % connective_smiles)
    hetero, side = 9, 10
    assert mol.GetAtomWithIdx(hetero).GetSymbol() in 'NOS'
    seen, stack = {side}, [side]
    while stack:
        cur = stack.pop()
        for n in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = n.GetIdx()
            if j not in seen and j != hetero:
                seen.add(j)
                stack.append(j)
    return mol, seen | {hetero}, seen, hetero, side


def _conn(connective_smiles, ring_name, allow_mancude=False):
    mol, frag, ring_yl, hetero, side = _fragment(connective_smiles)
    return rs._bare_connective_ring_yl(mol, frag, ring_yl, hetero, ring_name, side,
                                       allow_mancude=allow_mancude)


# --- M5: the decorated phenoxy prefix is composed by the writer, not sliced from a name ---

def test_the_writer_closes_a_benzene_ring_with_the_stem_it_is_given():
    mol, frag, ring_yl, hetero, side = _fragment("Oc3ccc(Cl)cc3")
    facts = {}
    ring = tuple(sorted(ring_yl))
    assert rs._decorated_heteroaryl_substituent_name(
        mol, ring_yl, ring, side, carbocyclic_stem='phenoxy', facts_out=facts) \
        == '4-chlorophenoxy'
    assert facts == {'carbocyclic_stem': 'phenoxy'}
    # a ring that is not benzene keeps its own stem and leaves the facts empty
    mol, frag, ring_yl, hetero, side = _fragment("Oc3ccncc3")
    facts = {}
    assert rs._decorated_heteroaryl_substituent_name(
        mol, ring_yl, tuple(sorted(ring_yl)), side, carbocyclic_stem='phenoxy',
        facts_out=facts) in (None, 'pyridin-4-yl')
    assert facts == {}


@pytest.mark.parametrize("smiles,text,expected", [
    # the text of the ring-yl name is a payload: the atoms decide
    ("Oc3ccc(Cl)cc3", "X", "4-chlorophenoxy"),
    ("Oc3ccccc3", "something else", "phenoxy"),
    ("Oc3ccc(-c4ccncc4)cc3", "anything", "4-(pyridin-4-yl)phenoxy"),
    # M6: the same for the nitrogen
    ("Nc3ccc(Cl)cc3", "X", "4-chloroanilino"),
    ("Nc3ccccc3", "something else", "anilino"),
    ("Nc3cccc(-c4ccncc4)c3", "anything", "3-(pyridin-4-yl)anilino"),
    ("Nc3cccc(-n4ccnc4)c3", "anything", "3-(1H-imidazol-1-yl)anilino"),
])
def test_the_retained_prefix_is_read_from_the_atoms(smiles, text, expected):
    assert _conn(smiles, text) == expected


@pytest.mark.parametrize("smiles", ["Oc3ccncc3", "Nc3ccncc3"])
def test_a_ring_that_is_not_benzene_never_takes_the_retained_word(smiles):
    # the text says 'phenyl' where the atoms are pyridine: no 'phenoxy' / 'anilino'; the text
    # is enclosed as the pyridine's locant asks
    connective = 'oxy' if smiles[0] == 'O' else 'amino'
    assert _conn(smiles, "phenyl") == f"(phenyl){connective}"
    assert _conn(smiles, "pyridin-4-yl") == f"(pyridin-4-yl){connective}"


@pytest.mark.parametrize("smiles,text,expected", [
    # the ring-yl is not benzene: the text is cited as it is, enclosed by the atoms' fact
    ("Nc3ncccn3", "pyrimidin-2-yl", "(pyrimidin-2-yl)amino"),
    ("Sc3ccccc3", "phenyl", "phenylsulfanyl"),
    ("OC3CCCCC3", "cyclohexyl", "cyclohexyloxy"),
    ("Nc3nccc(n3)-c3cccnc3", "4-(pyridin-3-yl)pyrimidin-2-yl",
     "[4-(pyridin-3-yl)pyrimidin-2-yl]amino"),
])
def test_the_enclosing_mark_follows_the_locant_the_ring_yl_cites(smiles, text, expected):
    assert _conn(smiles, text) == expected


def test_no_text_classifier_decides_the_marks(monkeypatch):
    from orthonym.assembly import naming_utils
    calls = []

    def boom(*a, **k):
        calls.append(a)
        raise AssertionError("a text classifier was asked")

    monkeypatch.setattr(naming_utils, "is_complex_substituent", boom)
    monkeypatch.setattr(rs, "anilino_preferred_prefix", boom, raising=False)
    for smiles, text, expected in (
            ("Nc3ncccn3", "pyrimidin-2-yl", "(pyrimidin-2-yl)amino"),
            ("Sc3ccccc3", "phenyl", "phenylsulfanyl"),
            ("OC3CCCCC3", "cyclohexyl", "cyclohexyloxy"),
            ("Nc3nccc(n3)-c3cccnc3", "4-(pyridin-3-yl)pyrimidin-2-yl",
             "[4-(pyridin-3-yl)pyrimidin-2-yl]amino")):
        assert _conn(smiles, text) == expected
    assert not calls


@pytest.mark.parametrize("smiles,locant", [
    ("Nc3ncccn3", True), ("Nc3ccncc3", True), ("Sc3ccccc3", False), ("OC3CCCCC3", False),
    ("OC3CCCC3", False), ("Oc3ccc(Cl)cc3", True), ("OC3C=CCCC3", True),
    ("Oc3ccc4ccccc4c3", True), ("OC3CC4CC3C4", True),
])
def test_a_ring_yl_cites_a_locant_when_its_free_valence_has_a_position_to_name(smiles, locant):
    mol, frag, ring_yl, hetero, side = _fragment(smiles)
    assert rs._ring_yl_cites_locant(mol, ring_yl, side) is locant


# --- the names of the whole molecules keep their PIN -----------------------------------

@pytest.mark.parametrize("connective,expected", [
    ("Nc3ncccn3", "6-[(pyrimidin-2-yl)amino]naphthalene-2-carboxylic acid"),
    ("Sc3ccccc3", "6-(phenylsulfanyl)naphthalene-2-carboxylic acid"),
    ("OC3CCCCC3", "6-(cyclohexyloxy)naphthalene-2-carboxylic acid"),
    ("Nc3nccc(n3)-c3cccnc3",
     "6-{[4-(pyridin-3-yl)pyrimidin-2-yl]amino}naphthalene-2-carboxylic acid"),
    ("Oc3ccccc3", "6-phenoxynaphthalene-2-carboxylic acid"),
    ("Oc3ccc(Cl)cc3", "6-(4-chlorophenoxy)naphthalene-2-carboxylic acid"),
    ("Nc3ccccc3", "6-anilinonaphthalene-2-carboxylic acid"),
    ("Nc3ccc(Cl)cc3", "6-(4-chloroanilino)naphthalene-2-carboxylic acid"),
    ("Nc3cccc(-c4ccncc4)c3", "6-[3-(pyridin-4-yl)anilino]naphthalene-2-carboxylic acid"),
    ("Oc3ccc(-c4ccncc4)cc3", "6-[4-(pyridin-4-yl)phenoxy]naphthalene-2-carboxylic acid"),
    ("Nc3cccc(-n4ccnc4)c3", "6-[3-(1H-imidazol-1-yl)anilino]naphthalene-2-carboxylic acid"),
])
def test_the_whole_name_keeps_its_pin(connective, expected):
    row = Orthonym(style="pin").name_tiered(NAPHTHOIC % connective)
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


@pytest.mark.parametrize("smiles,name", [
    # the NH of a diarylamine is the amine suffix (the principal group), not an 'anilino'
    # prefix of a nitrobenzene parent; (the Blue Book) and the aniline
    # parent of '4-chloro-N-(4-methylphenyl)aniline'. 'isothiocyanato...nitrophenyl' is
    # also first in the alphanumerical order of (:22234)
    ("O=[N+]([O-])c1ccc(Nc2ccc(N=C=S)cc2)cc1", "4-isothiocyanato-N-(4-nitrophenyl)aniline"),
    ("Clc1ccc(Nc2ccc(C)cc2)cc1", "N-(4-chlorophenyl)-4-methylaniline"),
    ("COc1ccc(Nc2ccc([N+](=O)[O-])cc2)cc1", "4-methoxy-N-(4-nitrophenyl)aniline"),
])
def test_a_diarylamine_keeps_the_amine_as_its_suffix(smiles, name):
    for tier in ("pin", "best-effort"):
        namer = (Orthonym(style="pin") if tier == "pin"
                 else Orthonym(style="pin", **__import__(
                     "orthonym.cli", fromlist=["x"])._emit_tier_flags(tier)))
        row = namer.name_tiered(smiles)
        assert (row["name"], row["tier"]) == (name, "pin_verified"), (tier, row)
