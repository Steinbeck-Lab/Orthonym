from rdkit import Chem
from orthonym.rules.purine import name_purine_substituent


def _frag(smi):
    """Whole molecule; the purine ring atoms are the fragment, attach = the ring
    N that carries the exocyclic bond leaving the ring system."""
    return Chem.MolFromSmiles(smi)


def test_adenin_9_yl_substituent():
    # 2-(6-amino-9H-purin-9-yl)acetic acid: OC(=O)C[N9]...; the CH2 carbon is
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


def test_purine_substituent_via_dispatch():
    from orthonym.rules.ring_substituents import name_ring_system_substituent
    mol = Chem.MolFromSmiles("OC(=O)Cn1cnc2c(N)ncnc21")
    ring = mol.GetSubstructMatch(Chem.MolFromSmarts(
        "[#7]1~[#6]~[#7]~[#6]2~[#7]~[#6]~[#7]~[#6]2~[#6]1"))
    ch2 = [a.GetIdx() for a in mol.GetAtoms()
           if a.GetSymbol() == "C" and a.GetTotalNumHs() == 2][0]
    assert name_ring_system_substituent(mol, set(ring), ch2) == "6-amino-9H-purin-9-yl"


def test_purine_substituent_end_to_end_pipeline():
    # Bug C, real caller: the dispatcher's frag_atoms is the WHOLE substituent
    # side (ring + its own exocyclic decorations, e.g. the C6-amino nitrogen),
    # NOT just the 9 bare ring atoms -- unlike the hand-built frag_set in
    # test_purine_substituent_via_dispatch above, which excludes the amino atom
    # and so cannot catch a frag_set/frag_ring_atoms mismatch. This test goes
    # through the full Orthonym.name pipeline (real dispatch call shape)
    # and round-trips the emitted name through OPSIN, so it is what actually
    # proves the purine tier fires on real input.
    from orthonym import Orthonym
    from orthonym.validation import opsin_roundtrip_check

    smiles = "OC(=O)Cn1cnc2c(N)ncnc21"
    name = Orthonym().name(smiles)

    assert name != "unknown organic compound"
    assert "purin" in name

    rt = opsin_roundtrip_check(smiles, name)
    assert rt["passed"], rt
