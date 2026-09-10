"""Generic (stereo-undefined and stereo-defined) glycosyloxy / nested-ring
substituent naming (tail #13 infrastructure).

Two capabilities, both fail-closed and additive:
  1. ``ring_substituents._ring_atom_simple_substituents`` names a ring atom
     bearing ``-O-(another ring)`` recursively -> ``[(inner-yl)oxy]`` (a
     di/tri-saccharide nests through the same path).
  2. ``natural_products._find_generic_glycosyloxy`` cites such a ring-oxy
     decoration on a steroid scaffold, RT-gated (0-wrong).

Each emitted name is asserted to round-trip to the full InChIKey via OPSIN.
"""
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.assembly.substituent_enumerator import name_substituent
from orthonym.namer import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse


def _full_rt(smiles: str, name: str) -> bool:
    osmi = opsin_parse(name)
    if not osmi:
        return False
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(osmi))


def _sub(smiles, attach_smiles_atom):
    """Name the whole molecule minus atom 0 as a substituent attached at
    ``attach_smiles_atom`` (helper for the nested-ring primitive)."""
    m = Chem.MolFromSmiles(smiles)
    frag = [a.GetIdx() for a in m.GetAtoms() if a.GetIdx() != 0]
    return name_substituent(m, sorted(frag), attach_smiles_atom)


def test_nested_disaccharide_names_as_substituent():
    # methyl-O-(furanose-O-pyranose): the furanose bears a nested pyranosyloxy.
    m = Chem.MolFromSmiles("COC1OCC(O)C1OC1OCC(O)C(O)C1O")
    ri = m.GetRingInfo()
    fur = [set(r) for r in ri.AtomRings() if len(r) == 5][0]
    # attach = furanose anomeric carbon (bonded to the methoxy O, atom 1)
    anom = [n.GetIdx() for n in m.GetAtomWithIdx(1).GetNeighbors()
            if n.GetIdx() in fur][0]
    frag = set()
    seen = {1}
    stack = [anom]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        frag.add(x)
        for n in m.GetAtomWithIdx(x).GetNeighbors():
            if n.GetIdx() != 1 and n.GetIdx() not in seen:
                stack.append(n.GetIdx())
    name = name_substituent(m, sorted(frag), anom)
    assert name and "oxan-2-yl)oxy" in name and name.endswith("oxolan-2-yl"), name


def test_fully_stereo_steroid_glycoside_round_trips():
    # cholesterol 3-O-(generic pyranoside): fully-stereo aglycone -> a clean,
    # full-InChIKey-round-tripping name via the generic-glycosyloxy path.
    smi = ("CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@H]"
           "(OC5OCC(O)C(O)C5O)CC[C@]4(C)[C@H]3CC[C@]12C")
    n = Orthonym(general_fallback=True, general_fallback_unverified=True,
                  allow_aromatic_general=True)
    name = n.name_tiered(smi).get("name")
    assert name and "unknown" not in name
    assert "oxan-2-yl)oxy" in name
    assert _full_rt(smi, name)


def test_simple_ring_naming_unchanged():
    # The nested-ring role is additive: a simple ring keeps its legacy form.
    n = Orthonym()
    assert n.name_tiered("OC1CCCCC1").get("name") == "cyclohexanol"
    assert n.name_tiered("c1ccc(OC2CCCCC2)cc1").get("name") == \
        "(cyclohexyloxy)benzene"
