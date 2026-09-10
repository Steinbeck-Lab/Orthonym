"""M3 : the canonical CIP path must label exocyclic ylidene double bonds.

The vendored `centres` engine (default CIP source) does not emit an E/Z label for
an exocyclic double bond to an aromatic-flagged ring atom (o-/p-quinoid / fulvenoid
systems), so `assign_stereochemistry` left those bonds' `_CIPCode` empty and
`collect_stereodescriptors` (which gates on `_CIPCode in {E,Z}`) dropped the stereo
-> the whole ylidene name failed the full-InChIKey offer gate -> abstain.

rdCIPLabeler labels these bonds correctly. The fix is a complementary rdCIP pass in
`assign_stereochemistry` that FILLS double-bond `_CIPCode` centres left empty, never
overwriting centres' labels. Finding: internal notes.
"""
from rdkit import Chem
from orthonym.perception.stereo import assign_stereochemistry


def _exocyclic_double_bond_cips(mol):
    """{ (begin,end): _CIPCode-or-None } for every exocyclic, stereogenic C=C."""
    out = {}
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE or b.IsInRing():
            continue
        if str(b.GetStereo()) == "STEREONONE":
            continue
        code = b.GetProp("_CIPCode") if b.HasProp("_CIPCode") else None
        out[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())] = code
    return out


def test_exocyclic_ylidene_bonds_get_ez_cipcode():
    # o-quinoid pyrrole: two adjacent exocyclic ylidene double bonds with defined
    # geometry (Z on the 2-chloroprop-2-en-1-ylidene, E on the ethan-1-ylidene).
    m = Chem.MolFromSmiles("C=C(Cl)/C=c1/cc[nH]/c1=C/C")
    assign_stereochemistry(m)
    cips = _exocyclic_double_bond_cips(m)
    # Both exocyclic ylidene C=C carry a valid E/Z CIP code (was None under centres).
    assert cips == {(3, 4): "Z", (8, 9): "E"}, cips


def test_exocyclic_methylene_quinoid_gets_ez():
    # p-quinodimethane-shaped: ring with two exocyclic ylidenes, one stereogenic.
    m = Chem.MolFromSmiles("C=c1cccc/c1=C/C=C(/N)CC")
    assign_stereochemistry(m)
    cips = _exocyclic_double_bond_cips(m)
    # every stereogenic exocyclic C=C must be labelled E/Z (none left None)
    labelled = [c for c in cips.values() if c in ("E", "Z")]
    assert None not in cips.values(), cips
    assert len(labelled) >= 1, cips


def test_centres_atom_labels_are_not_disturbed():
    # A defined tetrahedral centre + an exocyclic ylidene: the fill must add the
    # bond E/Z WITHOUT dropping or changing the atom's R/S (centres owns atoms).
    m = Chem.MolFromSmiles("C[C@H](O)/C=c1/cc[nH]/c1=C/C")
    assign_stereochemistry(m)
    atom_cips = {a.GetIdx(): a.GetProp("_CIPCode")
                 for a in m.GetAtoms() if a.HasProp("_CIPCode")}
    assert atom_cips, "the tetrahedral centre lost its CIP label"
    # and the exocyclic ylidene bond(s) are now labelled
    assert any(c in ("E", "Z") for c in _exocyclic_double_bond_cips(m).values())
