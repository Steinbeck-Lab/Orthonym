""" (h) and (i) in the base-component selection of the two-ring fusion namer
(``fused_ring_selection.select_base_component``), read on each component's own
Hantzsch-Widman numbering, and never decided by the input atom order.

 (h) (the Blue Book) "A component with the lower locants for heteroatoms",
:12403 "pyrazino[2,3-d]pyridazine (PIN) (locants '1,2' of pyridazine preferred to locants '1,4'
of pyrazine)"; (i) (:12407) "A component with the lower locants for the heteroatoms when
considered in the order: F > Cl > Br > I > O > S > Se > Te > N > P > As > Sb > Bi > Si > Ge >
Sn > Pb > B > Al > Ga > In > Tl",:12416 "3H,5H-[1,3,2]oxathiazolo[4,5-d][1,2,3]oxathiazole
(PIN) (locants '1,2,3' are lower than '1,3,2')"; (:8284) the component's own
numbering: "The locant '1' is given to a heteroatom that occurs first in the seniority sequence
... The numbering is then chosen to give lowest locants to heteroatoms considered as a set"."""
import random

import pytest
from rdkit import Chem

from orthonym.rules.fused_ring_selection import select_base_component


def _base_bonds(smiles, seed):
    """The element pairs of the bonds of the base ring that select_base_component picks for
    ``smiles`` written in a shuffled atom order."""
    mol = Chem.MolFromSmiles(smiles)
    order = list(range(mol.GetNumAtoms()))
    if seed:
        random.Random(seed).shuffle(order)
    # a fresh parse of the reordered SMILES (rings perceived on that atom order)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False))
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    base, _ = select_base_component(mol, rings)
    return {tuple(sorted((b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol())))
            for b in mol.GetBonds() if b.GetBeginAtomIdx() in base and b.GetEndAtomIdx() in base}


@pytest.mark.parametrize("seed", range(8))
def test_h_pyridazine_is_the_base_over_pyrazine(seed):
    #:12403: pyridazine N1, N2 (an N-N bond) against pyrazine N1, N4
    assert ("N", "N") in _base_bonds("c1cnc2cnncc2n1", seed)


@pytest.mark.parametrize("seed", range(8))
def test_i_the_1_2_3_oxathiazole_is_the_base_over_the_1_3_2_one(seed):
    #:12416: O1, S2, N3 (an O-S bond) against O1, N2, S3 (O, then S, before N in the order
    # of (i))
    assert ("O", "S") in _base_bonds("[nH]1oc2os[nH]c=2s1", seed)


@pytest.mark.parametrize("smiles,locants", [
    ("c1ccnnc1", {"N": [1, 2]}), ("c1cnccn1", {"N": [1, 4]}), ("c1cocn1", {"O": [1], "N": [3]}),
    ("O1BOC=C1", {"O": [1, 3], "B": [2]}), ("O1SNC=C1", {"O": [1], "S": [2], "N": [3]}),
])
def test_the_component_numbering_is_the_hantzsch_widman_one(smiles, locants):
    from orthonym.rules.fused_ring_selection import _monocycle_numbering
    mol = Chem.MolFromSmiles(smiles)
    num = _monocycle_numbering(mol, list(range(mol.GetNumAtoms())))
    got = {}
    for a, loc in num.items():
        sym = mol.GetAtomWithIdx(a).GetSymbol()
        if sym != "C":
            got.setdefault(sym, []).append(loc)
    assert {k: sorted(v) for k, v in got.items()} == locants


def test_a_star_name_on_a_base_only_h_demotes_is_kept_below_the_pin():
    #:13994 'furo[3',4':5,6]pyrazino[2,3-c]pyridazine (PIN)' takes pyridazine by (h) and needs a
    # second-order attached component (slice S2c-2); the star-shaped name on pyrazine, which
    # ties with pyridazine on (a)-(g), still names the molecule (OPSIN 2.9.0 reads both to
    # an InChIKey) and is kept, labelled below the PIN
    from orthonym.metrics import provenance
    from orthonym.rules import fused_rings
    mol = Chem.MolFromSmiles("c1cc2nc3cocc3nc2nn1")
    provenance.clear_provenance()
    assert fused_rings._try_polycomponent_fusion_name(mol) == "furo[3,4-b]pyridazino[3,4-e]pyrazine"
    assert provenance.get_provenance().get("non_pin_fragments") == (
        "furo[3,4-b]pyridazino[3,4-e]pyrazin",)
