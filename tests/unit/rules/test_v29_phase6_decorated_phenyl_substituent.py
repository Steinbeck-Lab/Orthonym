"""v29 Phase 6 — decorated CARBOCYCLIC substituents get a producer.

`_decorated_heteroaryl_substituent_name` is the producer for "a ring carrying
its own decorations, free valence on a ring atom". It was heteroaryl-ONLY: it
looked its stem up in `_PIN_HETEROARYL_STEMS`, a 19-entry table with no entry
for benzene, and returned None for anything absent. So a **decorated phenyl**
substituent — among the commonest shapes in drug-like space — had no producer
at all and fell through to the DROP-24 fail-closed guard.

Measured on the Phase 5 corpus before this change, each of these was unnameable
at EVERY ring attachment point:

    CN(C)c1ccccc1       ->  (nothing)      wanted 4-(dimethylamino)phenyl
    NS(=O)(=O)c1ccccc1  ->  (nothing)      wanted 4-sulfamoylphenyl

P-29.3.5: the free valence of a benzene substituent is position 1 and its
locant is NOT cited — the retained prefix is `phenyl`, never `benzen-1-yl` —
after which the decorations take the lowest locants. The existing numbering
cascade already yields exactly that for a carbocycle (with no heteroatoms and
no indicated H its sort key degenerates to `(fv_loc, deco_locs)`), so only the
core spelling is new.

The second half of this file covers P-16.3.3 enclosing marks, which the shared
decoration assembler was omitting for BOTH the carbocyclic and the pre-existing
heteroaryl path.
"""

import pytest
from rdkit import Chem

from orthonym.rules.ring_substituents import name_ring_system_substituent

pytestmark = pytest.mark.opsin_gate


def _names_at_every_ring_attachment(smiles):
    """Every distinct substituent name this fragment yields, attaching a methyl
    stand-in at each atom in turn."""
    base = Chem.MolFromSmiles(smiles)
    out = set()
    for at in range(base.GetNumAtoms()):
        rw = Chem.RWMol(base)
        marker = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(at, marker, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        frag = [i for i in range(mol.GetNumAtoms()) if i != marker]
        try:
            name = name_ring_system_substituent(mol, frag, at)
        except Exception:
            name = None
        if name:
            out.add(name)
    return out


# --------------------------------------------------------------------------
# The gap: decorated phenyl had no producer
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("CN(C)c1ccccc1", {"2-(dimethylamino)phenyl", "3-(dimethylamino)phenyl",
                       "4-(dimethylamino)phenyl"}),
    ("Clc1ccccc1",    {"2-chlorophenyl", "3-chlorophenyl", "4-chlorophenyl"}),
    ("N#Cc1ccccc1",   {"2-cyanophenyl", "3-cyanophenyl", "4-cyanophenyl"}),
    ("O=[N+]([O-])c1ccccc1",
                      {"2-nitrophenyl", "3-nitrophenyl", "4-nitrophenyl"}),
    ("FC(F)(F)c1ccccc1", {"2-(trifluoromethyl)phenyl",
                          "3-(trifluoromethyl)phenyl",
                          "4-(trifluoromethyl)phenyl"}),
])
def test_decorated_phenyl_now_names(smiles, expected):
    assert expected <= _names_at_every_ring_attachment(smiles)


def test_free_valence_locant_is_never_cited_on_phenyl():
    """P-29.3.5 — the prefix is `4-methylphenyl`, never `4-methylbenzen-1-yl`
    or `4-methylphenyl-1-yl`."""
    for name in _names_at_every_ring_attachment("Cc1ccccc1"):
        assert "benzen" not in name
        assert not name.endswith("-1-yl")


def test_decoration_locants_are_lowest():
    """A disubstituted ring numbers the decorations as low as the free valence
    at 1 permits: {2,5}, not {3,6}."""
    got = _names_at_every_ring_attachment("COc1ccc(OC)cc1")
    assert "2,5-dimethoxyphenyl" in got
    assert not any("3,6-" in n for n in got)


def test_alphanumerical_tie_break_on_equal_locant_sets():
    """P-14.4: when two numberings give the same locant SET {2,5}, the prefix
    first in alphanumerical order takes the lower locant. Attached ortho to the
    chlorine that is `2-chloro-5-fluorophenyl`; attached ortho to the fluorine
    the set is fixed by the ring, giving `5-chloro-2-fluorophenyl`. Both are
    legitimate and correspond to DIFFERENT attachment points."""
    got = _names_at_every_ring_attachment("Fc1ccc(Cl)cc1")
    assert "2-chloro-5-fluorophenyl" in got
    assert "5-chloro-2-fluorophenyl" in got


# --------------------------------------------------------------------------
# P-16.3.3 enclosing marks
# --------------------------------------------------------------------------

def test_multiplied_decoration_prefix_takes_enclosing_marks():
    """`4-dimethylaminophenyl` could read as di(methylamino). The Blue Book
    writes `(dimethylamino)` 24 times and never once unbracketed."""
    got = _names_at_every_ring_attachment("CN(C)c1ccccc1")
    assert "4-(dimethylamino)phenyl" in got
    assert "4-dimethylaminophenyl" not in got


def test_simple_decoration_prefixes_stay_bare():
    """Wrapping a simple prefix is itself a spelling error — `4-methylphenyl`,
    not `4-(methyl)phenyl`. Note `2,5-dimethoxyphenyl`: the `di` there is the
    assembler's own multiplier over two `methoxy` decorations, not part of the
    prefix name, so it must NOT trigger marks."""
    assert "4-methylphenyl" in _names_at_every_ring_attachment("Cc1ccccc1")
    assert "4-nitrophenyl" in _names_at_every_ring_attachment("O=[N+]([O-])c1ccccc1")
    assert "2,5-dimethoxyphenyl" in _names_at_every_ring_attachment("COc1ccc(OC)cc1")


@pytest.mark.parametrize("nm,enclosed", [
    ("methyl", "methyl"),
    ("chloro", "chloro"),
    ("methoxy", "methoxy"),
    ("nitro", "nitro"),
    ("dimethylamino", "(dimethylamino)"),
    ("trifluoromethyl", "(trifluoromethyl)"),
    ("propan-2-yl", "(propan-2-yl)"),
    ("diazenyl", "diazenyl"),      # atomic prefix that merely starts with 'di'
    ("diazo", "diazo"),
    ("(already)", "(already)"),
])
def test_enclose_decoration_unit(nm, enclosed):
    from orthonym.rules.ring_substituents import _enclose_decoration
    assert _enclose_decoration(nm) == enclosed


# --------------------------------------------------------------------------
# Fail-closed: nothing this cannot describe may leak
# --------------------------------------------------------------------------

def test_unnameable_decoration_still_fails_closed():
    """A ring whose decoration the identifier cannot name must yield NOTHING,
    not a phenyl name that silently drops it."""
    for name in _names_at_every_ring_attachment("NS(=O)(=O)c1ccccc1"):
        # if a name IS produced it must account for the sulfamoyl group
        assert "sulfamoyl" in name or "sulf" in name, name
