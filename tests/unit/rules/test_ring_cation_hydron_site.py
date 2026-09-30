"""A ring cation drawn with its charge on a substituted, hydrogen-free atom carries
its added hydron on another ring atom; the '-ium' is cited there.

 "General rule for systematically naming cationic centers in parent
hydrides" (the Blue Book),:41368: "A cation derived formally by adding one or
more hydrons to any position of a neutral parent hydride... is named by replacing
the final letter 'e' of the parent hydride name, if any, by the suffix 'ium'";
example:41396 '1H-imidazol-3-ium (PIN)'. C[N+]1=CNc2ccccc21 is 1-methyl-1H-
benzimidazole with a hydron added at N-3 (its other resonance drawing is
CN1C=[NH+]c2ccccc21): '1-methyl-1H-benzimidazol-3-ium'. The name that used to ship,
'1-methyl-1H-benzimidazol-1-ium', adds the hydron at the methylated N-1 (OPSIN:
C[NH+]1C=NC2=C1C=CC=C2), a different cation with the same standard InChIKey.

Each name is read back by a FRESH OPSIN call (tests/support/rt_assert.
_independent_parse), and its fixed-H InChI (RDKit) must be the input's.
"""
import pytest

from tests.support.jars import jar_or_skip

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _fixed_h(smi):
    from rdkit import Chem
    from rdkit.Chem import inchi
    return inchi.MolToInchi(Chem.MolFromSmiles(smi), options="/FixedH /SNon")


def _key(smi):
    from rdkit import Chem
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smi))


def _exact(name, smiles):
    from tests.support.rt_assert import _independent_parse
    parsed = _independent_parse(name)
    return bool(parsed) and _key(parsed) == _key(smiles) and _fixed_h(parsed) == _fixed_h(smiles)


@pytest.mark.parametrize("smiles,name", [
    ("C[N+]1=CNc2ccccc21", "1-methyl-1H-benzimidazol-3-ium"),
    ("CN1C=[NH+]c2ccccc21", "1-methyl-1H-benzimidazol-3-ium"),   # the other drawing
    ("C1=CC=C2C(=C1)NC=[N+]2C3=COC=C3.[I-]", "1-(furan-3-yl)-1H-benzimidazol-3-ium iodide"),
    ("CC(C1=[N+](C2=CC=CC=C2N1)CC=CC3=CC=CC=C3)O",
     "2-(1-hydroxyethyl)-1-(3-phenylprop-2-en-1-yl)-1H-benzimidazol-3-ium"),
    ("CCC1=C([N+](=C(C=C(N1)C)C)c2ccccc2)CC",
     "2,3-diethyl-5,7-dimethyl-1-phenyl-1H-1,4-diazepin-4-ium"),
    # (the Blue Book): indicated hydrogen (b):3246 and the '-ium'
    # (c):3256 at 1 and 3 either way, (f):3301 puts the N-substituent at 1
    ("C[n+]1cc[nH]c1", "1-methyl-1H-imidazol-3-ium"),
    ("Cn1cc[nH+]c1", "1-methyl-1H-imidazol-3-ium"),
])
def test_ium_at_the_hydron_site(smiles, name):
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
    assert _exact(name, smiles)


@pytest.mark.parametrize("smiles", [
    "C[n+]1ccccc1",          # no hydron elsewhere: 1-methylpyridin-1-ium
    "Cc1cc[nH+]cc1",         # the in-place name already reads back exactly
])
def test_names_that_already_read_back_exactly_are_kept(smiles):
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["tier"] == "pin_verified" and _exact(r["name"], smiles), r


def test_hydron_site_is_found_on_the_other_ring_nitrogen():
    from rdkit import Chem
    from orthonym.rules.ions import _ring_cation_hydron_site
    mol = Chem.MolFromSmiles("C[N+]1=CNc2ccccc21")
    ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    res, site = _ring_cation_hydron_site(mol, ring, 1)
    assert mol.GetAtomWithIdx(site).GetSymbol() == "N" and mol.GetAtomWithIdx(site).GetTotalNumHs() == 1
    assert _ring_cation_hydron_site(Chem.MolFromSmiles("C[n+]1ccccc1"),
                                    list(range(1, 7)), 1) is None
