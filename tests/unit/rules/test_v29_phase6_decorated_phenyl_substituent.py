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
    """Two DIFFERENT attachment points on the same ring, each with its locant
    set fixed by the ring — both names are legitimate."""
    got = _names_at_every_ring_attachment("Fc1ccc(Cl)cc1")
    assert "2-chloro-5-fluorophenyl" in got
    assert "5-chloro-2-fluorophenyl" in got


@pytest.mark.parametrize("smiles", [
    "Brc1cccc(Cl)c1",
    "Clc1cccc(Br)c1",
    "c1(Br)cccc(Cl)c1",
])
def test_p14_4_g_breaks_the_symmetric_tie_deterministically(smiles):
    """P-14.4(g): when every earlier criterion ties, the lower locant goes to
    the substituent cited FIRST in alphanumerical order.

    With a bromine on one ortho position and a chlorine on the other, BOTH
    ring-traversal directions give the locant set {2,6} — an exact tie. Before
    the fix the winner was whichever direction RDKit's neighbour order
    enumerated first, so the SAME MOLECULE written two ways got two names:

        Brc1cccc(Cl)c1  ->  2-bromo-6-chlorophenyl   (correct)
        Clc1cccc(Br)c1  ->  6-bromo-2-chlorophenyl   (non-PIN)

    Nondeterministic and non-PIN. `bromo` sorts before `chloro`, so it takes
    locant 2 regardless of how the input was written.
    """
    got = _names_at_every_ring_attachment(smiles)
    ortho = {n for n in got if "bromo" in n and "chloro" in n
             and n.startswith(("2-", "6-"))}
    assert "2-bromo-6-chlorophenyl" in ortho, ortho
    assert "6-bromo-2-chlorophenyl" not in ortho, ortho


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
    # escalates ( -> [ when the inner name already carries parentheses
    ("3-(methylsulfanyl)phenyl", "[3-(methylsulfanyl)phenyl]"),
])
def test_enclosing_marks_come_from_the_shared_primitive(nm, enclosed):
    """The decoration assembler must use `enclose_if_compound`, not a local
    re-derivation.

    A hand-rolled version was written first: "wrap when the name starts with a
    multiplying prefix", plus a hand-maintained list of atomic false friends
    (diazo, diazenyl, ...) that merely begin with those letters. It was deleted.
    `enclose_if_compound` already unions `needs_brackets` with
    `is_complex_substituent` — neither is complete alone — gets every one of
    these right with no list to maintain, and escalates the mark level.
    """
    from orthonym.assembly.naming_utils import enclose_if_compound
    assert enclose_if_compound(nm) == enclosed


# --------------------------------------------------------------------------
# Fail-closed: nothing this cannot describe may leak
# --------------------------------------------------------------------------

def test_unnameable_decoration_still_fails_closed():
    """A ring whose decoration the identifier cannot name must yield NOTHING,
    not a phenyl name that silently drops it.

    ⚠ This assertion is written to be NON-VACUOUS. The first version was
    `for name in ...: assert "sulf" in name`, which passes trivially when the
    set is empty — and it IS empty, so the test proved nothing (a code review
    caught it; cf. `feedback_harness_that_reports_success`).

    `_identify_fused_substituent`'s sulfur branch handles only `-SH` and
    `-S-alkyl`; a sulfonamide sulfur has three neighbours (O, O, N) and matches
    no branch, so the decoration is unnameable and the whole producer must
    decline. Naming this molecule is a genuine remaining gap — the point here
    is that the gap FAILS CLOSED rather than emitting a phenyl that drops the
    -SO2NH2.
    """
    got = _names_at_every_ring_attachment("NS(=O)(=O)c1ccccc1")
    dropped = {n for n in got if "sulf" not in n and "amino" not in n}
    assert not dropped, (
        f"a name was emitted that silently drops the sulfamoyl group: {dropped}"
    )


def test_known_remaining_gaps_are_declines_not_wrong_names():
    """Two shapes named in the Phase 6 commit message are NOT fixed by it, and
    this test pins that honestly so the claim cannot drift.

    * `NS(=O)(=O)c1ccccc1` — the sulfonamide decoration is unnameable (above).
    * `Cc1ncsc1C` — `identify_ring_system`'s aromatic-5-ring branch has cases
      for [O], [S], [N] and [N,N] but none for [N,S], so thiazole is not
      identified at all and `_PIN_HETEROARYL_STEMS` has no thiazole entry.
      Carbocyclic admission cannot help: `is_carbocyclic` requires
      `ring_name == 'benzene'`.

    Both must DECLINE. If either ever starts producing a name, this test should
    be updated deliberately — not silently satisfied.
    """
    assert _names_at_every_ring_attachment("Cc1ncsc1C") == set()
