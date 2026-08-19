from rdkit import Chem
from orthonym.rules.purine import name_purine_substituent


def _frag(smi):
    """Whole molecule; the purine ring atoms are the fragment, attach = the ring
    N that carries the exocyclic bond leaving the ring system."""
    return Chem.MolFromSmiles(smi)


def test_adenin_9_yl_substituent():
    # 2-(6-amino-9H-purin-9-yl)acetic acid: OC(=O)C[N9]... ; the CH2 carbon is
    # the parent-side attach; the fragment is the purine ring.
    mol = Chem.MolFromSmiles("OC(=O)Cn1cnc2c(N)ncnc21")
    ring = mol.GetSubstructMatch(Chem.MolFromSmarts(
        "[#7]1~[#6]~[#7]~[#6]2~[#7]~[#6]~[#7]~[#6]2~[#6]1"))
    assert ring, "ring not found"
    # attach = the ring N bonded to the CH2 (idx of the CH2's ring neighbour)
    ch2 = [a.GetIdx() for a in mol.GetAtoms()
           if a.GetSymbol() == "C" and a.GetTotalNumHs() == 2][0]
    attach = next(n.GetIdx() for n in mol.GetAtomWithIdx(ch2).GetNeighbors()
                  if n.GetIdx() in set(ring))
    assert name_purine_substituent(mol, set(ring), attach) == "6-amino-9H-purin-9-yl"


def test_fusion_carbon_attach_declines():
    # A fusion carbon (C4 or C5 of the purine ring) has no exocyclic free
    # valence -- every neighbour is another ring atom -- so it can never be
    # a real -yl attachment point. name_purine_substituent must fail closed
    # (return None) rather than emit a syntactically well-formed but
    # semantically bogus name for it.
    mol = Chem.MolFromSmiles("OC(=O)Cn1cnc2c(N)ncnc21")
    ring = mol.GetSubstructMatch(Chem.MolFromSmarts(
        "[#7]1~[#6]~[#7]~[#6]2~[#7]~[#6]~[#7]~[#6]2~[#6]1"))
    ring_set = set(ring)
    # Fusion carbons are the ring atoms with 3 in-fragment ring neighbours
    # (the other 6 ring atoms each have only 2).
    fusion_carbons = [
        idx for idx in ring
        if sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
               if n.GetIdx() in ring_set) == 3
    ]
    assert len(fusion_carbons) == 2
    for attach in fusion_carbons:
        assert name_purine_substituent(mol, ring_set, attach) is None
