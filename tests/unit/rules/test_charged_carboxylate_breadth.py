""" charged breadth — the aromatic-carboxylate atom-drop 0-wrong fix + no
regression on simple substituted aromatic carboxylates.

Root cause: `_name_aromatic_carboxylate_with_substituents` named ring substituents
by ATOM TYPE alone (O -> 'hydroxy'), silently dropping an ether/ester arm and
producing a WRONG molecule (4-(glycosyloxy)benzoate -> '4-hydroxybenzoate'). The
fix fails closed on any non-simple substituent so the caller falls through to the
substituent-complete neutralize path.
"""
from rdkit import Chem
from orthonym.rules import ions as I
from orthonym.perception.ions import get_ion_sites


def _carbox_name(smi):
    m = Chem.MolFromSmiles(smi)
    site = get_ion_sites(m)['anions'][0]
    return I._name_carboxylate_systematic(m, site)


def test_glycosyloxy_benzoate_never_drops_to_hydroxy():
    # 4-(β-D-glucosyloxy)benzoate: the aromatic handler must NOT emit the
    # atom-dropped '4-hydroxybenzoate'. It either names it correctly (via the
    # fall-through neutralize path) or fails closed ('') -- never the wrong molecule.
    name = _carbox_name("[O-]C(=O)c1ccc(OC2OC(CO)C(O)C(O)C2O)cc1")
    assert name != "4-hydroxybenzoate", "atom-dropped wrong molecule leaked"


def test_aromatic_handler_fails_closed_on_ether_substituent():
    # the ring-substituent handler itself declines a complex (ether) substituent.
    m = Chem.MolFromSmiles("[O-]C(=O)c1ccc(OC2OC(CO)C(O)C(O)C2O)cc1")
    from orthonym.rules.ions import (
        _name_aromatic_carboxylate_with_substituents, _find_carboxyl_carbon,
        _detect_aromatic_carboxylate,
    )
    site = get_ion_sites(m)['anions'][0]
    cc = _find_carboxyl_carbon(m, site)
    base = _detect_aromatic_carboxylate(m, cc)
    assert _name_aromatic_carboxylate_with_substituents(m, cc, base) == ""


def test_simple_substituted_aromatics_unchanged():
    # simple hydroxy / chloro / bare benzoate must be byte-identical (no regression).
    assert _carbox_name("[O-]C(=O)c1ccccc1") == "benzoate"
    assert _carbox_name("[O-]C(=O)c1ccc(Cl)cc1") == "4-chlorobenzoate"
    assert _carbox_name("[O-]C(=O)c1ccc(O)cc1") == "4-hydroxybenzoate"
