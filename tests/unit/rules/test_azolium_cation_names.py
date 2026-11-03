"""Azolium cations: the ring nitrogens all carry a substituent (1-ethyl-3-methylimidazolium),
or the ring carries a second NH beside the protonated nitrogen (the benzimidazolium of
c1ccc2[nH+]c[nH]c2c1), or the ring is an amidine of its own (a 2-imidazolinium).

 "General rule for systematically naming cationic centers in parent hydrides"
(the Blue Book),:41368: "A cation derived formally by adding one or more hydrons to
any position of a neutral parent hydride... is named by replacing the final letter 'e' of
the parent hydride name, if any, by the suffix 'ium'"; the example at:41396 is
'1H-imidazol-3-ium (PIN)'. A substituent replaces a hydrogen of that cation, so the
N,N'-disubstituted cation keeps the indicated hydrogen of the parent hydride at one nitrogen
and the added hydron at the other: '1,3-dimethyl-1H-imidazol-3-ium'.

Numbering, "NUMBERING" (:3219): (b) "indicated hydrogen for unsubstituted compounds"
(:3246), (c) "principal characteristic groups and free valences (suffixes)" (:3256), (f)
"detachable alphabetized prefixes, all considered together in a series of increasing
numerical order" (:3301), (g) "lowest locants for the substituent cited first as a prefix in
the name" (:3307, '1-methyl-4-nitronaphthalene (PIN) (not 4-methyl-1-nitronaphthalene)',:3318).
The charge is delocalised over both nitrogens, so the drawing does not change the name:
'1-ethyl-3-methyl-1H-imidazol-3-ium' from either drawing.

Every name is read back by a FRESH OPSIN call and must give the input's fixed-H InChI.
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
    parsed = _independent_parse(name, smiles)
    return bool(parsed) and _key(parsed) == _key(smiles) and _fixed_h(parsed) == _fixed_h(smiles)


@pytest.mark.parametrize("smiles,name", [
    # both ring nitrogens substituted
    ("Cn1cc[n+](C)c1", "1,3-dimethyl-1H-imidazol-3-ium"),
    ("CCn1cc[n+](C)c1", "1-ethyl-3-methyl-1H-imidazol-3-ium"),
    ("CC[n+]1ccn(C)c1", "1-ethyl-3-methyl-1H-imidazol-3-ium"),        # the other drawing
    ("CCCCn1cc[n+](C)c1", "1-butyl-3-methyl-1H-imidazol-3-ium"),
    ("CCCCCCCCn1cc[n+](C)c1", "1-methyl-3-octyl-1H-imidazol-3-ium"),
    ("CC[n+]1c(C)n(C)cc1", "1-ethyl-2,3-dimethyl-1H-imidazol-3-ium"),
    ("C[n+]1cccn1C", "1,2-dimethyl-1H-pyrazol-2-ium"),
    ("Cn1cc[n+](C)n1", "1,3-dimethyl-1H-1,2,3-triazol-3-ium"),
    ("Cn1cnc[n+]1C", "1,2-dimethyl-1H-1,2,4-triazol-2-ium"),
    ("C[n+]1cn(C)c2ccccc12", "1,3-dimethyl-1H-1,3-benzimidazol-3-ium"),
    ("CCn1c[n+](C)c2ccccc21", "1-ethyl-3-methyl-1H-1,3-benzimidazol-3-ium"),
    # the benzimidazolium whose two nitrogens both carry hydrogen: the indicated hydrogen and
    # the '-ium' name two nitrogens ('1H-1,3-benzimidazol-1-ium' is the NH2+ cation)
    ("c1ccc2[nH+]c[nH]c2c1", "1H-1,3-benzimidazol-3-ium"),
    ("c1ccc2c(c1)[nH]c[nH+]2", "1H-1,3-benzimidazol-3-ium"),
    ("Cc1[nH]c2ccccc2[nH+]1", "2-methyl-1H-1,3-benzimidazol-3-ium"),
    ("c1ccc2[nH+]n[nH]c2c1", "1H-1,2,3-benzotriazol-3-ium"),
    # the ring is an amidine of its own (naphazoline, tolazoline: no '-imidamide' group stands
    # on it, the parent is the heterocycle)
    ("CC1=[NH+]CCN1", "2-methyl-4,5-dihydro-1H-imidazol-3-ium"),
    ("c1ccccc1C1=[NH+]CCN1", "2-phenyl-4,5-dihydro-1H-imidazol-3-ium"),
    ("c1ccc2c(c1)CCCC2C1=[NH+]CCN1",
     "2-(1,2,3,4-tetrahydronaphthalen-1-yl)-4,5-dihydro-1H-imidazol-3-ium"),
    ("CC1=[NH+]CCN1C", "1,2-dimethyl-4,5-dihydro-1H-imidazol-3-ium"),
])
def test_azolium_name_is_the_retained_parent_with_the_added_hydron(smiles, name):
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
    assert _exact(name, smiles)


@pytest.mark.parametrize("a,b", [
    ("CC[n+]1ccn(C)c1", "CCn1cc[n+](C)c1"),
    ("C[n+]1ccn(CC)c1", "CCn1cc[n+](C)c1"),
    ("CN1C=C[N+](C)=C1", "Cn1cc[n+](C)c1"),
    ("Cn1c[n+](CC)c2ccccc21", "CC[n+]1cn(C)c2ccccc12"),
])
def test_the_drawing_does_not_change_the_name(a, b):
    jar_or_skip()
    from orthonym import Orthonym
    n = Orthonym(style="pin")
    assert n.name_tiered(a)["name"] == n.name_tiered(b)["name"]


@pytest.mark.parametrize("smiles,name", [
    ("CC1=NCCN1C", "1,2-dimethyl-4,5-dihydro-1H-imidazole"),
    ("CN1CCN=C1C", "1,2-dimethyl-4,5-dihydro-1H-imidazole"),         # the same molecule
    ("CC1=NCCN1CC", "1-ethyl-2-methyl-4,5-dihydro-1H-imidazole"),
    ("CC1=NCCN1c1ccccc1", "2-methyl-1-phenyl-4,5-dihydro-1H-imidazole"),
    ("C1CC=NN1C", "1-methyl-4,5-dihydro-1H-pyrazole"),
    ("CN1CCC=N1", "1-methyl-4,5-dihydro-1H-pyrazole"),
])
def test_partially_saturated_azole_has_one_name_for_every_drawing(smiles, name):
    """ (the Blue Book): 'imidazole (1H-isomer shown; the PIN is 1H-imidazole)',
    and the same ring with hydro prefixes, '2-hydrazinyl-4,5-dihydro-1H-imidazole (PIN)'
    (:19389). The mancude parent's indicated hydrogen is the molecule's own pyrrole-type
    nitrogen, the N-H or the N-substituent; it used to be the first nitrogen in atom order, which
    sent one drawing of 1,2-dimethyl-2-imidazoline to '1,2-dimethyl-4,5-dihydro-1H-1,3-diazole'."""
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
    assert _exact(name, smiles)


def test_the_azolium_centres_are_both_ring_nitrogens():
    from rdkit import Chem

    from orthonym.rules.azolium_cations import azolium_hydron_centres
    mol = Chem.MolFromSmiles("CCn1cc[n+](C)c1")
    ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    centre = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1)
    found = azolium_hydron_centres(mol, ring, centre)
    assert found[0] == centre and len(found) == 2
    assert all(mol.GetAtomWithIdx(i).GetSymbol() == "N" for i in found)
    # a pyridinium has one nitrogen
    pyr = Chem.MolFromSmiles("C[n+]1ccccc1")
    pring = [a.GetIdx() for a in pyr.GetAtoms() if a.IsInRing()]
    pc = next(a.GetIdx() for a in pyr.GetAtoms() if a.GetFormalCharge() == 1)
    assert azolium_hydron_centres(pyr, pring, pc) == [pc]


@pytest.mark.parametrize("smiles", [
    "C[n+]1ccccc1",          # 1-methylpyridin-1-ium: no second nitrogen
    "Cc1cc[nH+]cc1",         # the in-place name already reads back exactly
    "c1c[nH]c[nH+]1",        # 1H-imidazol-3-ium (PIN, the Blue Book)
    "Cn1cc[nH+]c1",          # 1-methyl-1H-imidazol-3-ium
])
def test_names_that_already_read_back_exactly_are_kept(smiles):
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["tier"] == "pin_verified" and _exact(r["name"], smiles), r
