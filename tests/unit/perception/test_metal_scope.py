"""The scope table ``data/metal_scope.py`` and the function that reads it.

``assembly_has_out_of_scope_metal`` says whether the best-effort ion and adduct widening may fire
for a multi-fragment assembly. It reads the table, so admitting a metal is an edit of the table.
These tests pin three things: the function gives, for every element of the allow-list and for the
constructed assemblies below, the answers the engine gave before the table existed (the literals
were computed on the start commit of the zinc work) except for zinc, which the table admits to
``ION_SALTS`` and ``CARBANION_SALTS``, and Groups 1-2, which it admits to ``CARBANION_SALTS`` since 2026-10-30
(user decision, breadth lane); the table answers its lookups; and the function follows the
table (a metal added to it, here by a monkeypatch, changes the answer).

Blue Book: INTRODUCTION, "Organometallic compounds are compounds having at least one bond
between one metal atom and one carbon atom." An assembly with a carbanion or a hydride beside a
metal is organometallic for that metal.
"""
import pytest
from rdkit import Chem

from orthonym.data import metal_scope
from orthonym.data.metal_scope import (AMMINE, BONDED_NAMES, CARBANION_SALTS, CONNECT_ANION, DATIVE, DISCONNECT,
                                       ELEMENT, ION_SALTS, MACROCYCLE, METAL_SCOPE, ZERO_VALENT,
                                       ZERO_VALENT_METALS, MetalScope, has_mechanism, has_valence,
                                       is_named_metal)
from orthonym.perception.metals import METAL_ELEMENT_SYMBOLS, assembly_has_out_of_scope_metal

ACETATE = ".CC(=O)[O-]"
METHANIDE = ".[CH3-]"
HYDRIDE = ".[H-]"

GROUP_1 = ("Li", "Na", "K", "Rb", "Cs", "Fr")
GROUP_2 = ("Be", "Mg", "Ca", "Sr", "Ba", "Ra")
#: Groups 13-16 elements that the engine treats as metalloids names them substitutively)
METALLOIDS = ("B", "Al", "Ga", "In", "Tl", "Si", "Ge", "Sn", "Pb", "As", "Sb", "Bi", "Te", "Po")


def _out(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return assembly_has_out_of_scope_metal(mol)


def _cation(symbol, charge):
    return f"[{symbol}+{charge}]" if charge > 1 else f"[{symbol}+]"


@pytest.mark.parametrize("symbol", GROUP_1 + GROUP_2)
def test_scope_table_reproduces_the_old_set(symbol):
    """The 12 alkali and alkaline-earth elements, with their usual charge: a salt of an organic
    anion is in scope, and since 2026-10-30 (``CARBANION_SALTS``) so is a carbanion or a hydride beside the metal."""
    charge = 1 if symbol in GROUP_1 else 2
    cation = _cation(symbol, charge)
    assert _out(cation + ACETATE * charge) is False
    assert _out(cation + METHANIDE * charge) is False
    assert _out(cation + HYDRIDE * charge) is False


@pytest.mark.parametrize("smiles, expected", [
    # a true metal outside Groups 1-2 that the table does not admit: out of scope with any anion
    ("[Cu+2].CC(=O)[O-].CC(=O)[O-]", True),
    ("[Pt+2].CC(=O)[O-].CC(=O)[O-]", True),
    ("[Fe+3].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-]", True),
    ("[La+3].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-]", True),
    ("[Cu+2].[Cl-].[Cl-]", True),
    ("[Cu+2].[CH3-].[CH3-]", True),
    ("[Cu+2].[H-].[H-]", True),
    ("[Cu]", True),
    ("[Cl-].[Cl-].[Pt+2].N.N", True),
    ("[Mg+2].[Cu+2].[Cl-].[Cl-].[Cl-].[Cl-]", True),                       # one metal in scope, one not
    ("[Na+].[Cu+2].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-]", True),
    ("[Zn+2].[Cu+2].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-]", True),    # zinc is in scope, copper is not
    ("C1=Cc2nc1c(-c1ccccc1)c1ccc([n-]1)c(-c1ccccc1)c1nc(c(-c3ccccc3)c3ccc([n-]3)c2-c2ccccc2)C=C1.[Cu+2]", True),
    # zinc is admitted to ION_SALTS and CARBANION_SALTS: in scope with an organic anion, a carbanion or a hydride
    ("[Zn+2].CC(=O)[O-].CC(=O)[O-]", False),
    ("[Zn+2].[Cl-].[Cl-]", False),
    ("[Zn+2].[CH3-].[CH3-]", False),
    ("[Zn+2].[H-].[H-]", False),
    ("[Zn]", False),
    ("[Mg+2].[Zn+2].[Cl-].[Cl-].[Cl-].[Cl-]", False),
    ("[Na+].[Zn+2].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-]", False),
    ("C1=Cc2nc1c(-c1ccccc1)c1ccc([n-]1)c(-c1ccccc1)c1nc(c(-c3ccccc3)c3ccc([n-]3)c2-c2ccccc2)C=C1.[Zn+2]", False),
    ("[Ca+2].[CH3-].[CH3-]", False),                                       # Group 2 with a carbanion (since 2026-10-30)
    # a metalloid is named substitutively: never out of scope, with or without a carbanion
    ("[Sn+2].CC(=O)[O-].CC(=O)[O-]", False),
    ("[Sn+4].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-]", False),
    ("[Si+4].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-]", False),
    ("[Sn+2].[CH3-].[CH3-]", False),
    ("[Al+3].[H-].[H-].[H-]", False),
    # no metal at all
    ("[NH4+].CC(=O)[O-]", False),
    ("CC(=O)[O-].CC(=O)[O-]", False),
    ("[Na+].c1cc[n-]c1", False),                                           # a ring-nitrogen anion is not a carbanion
])
def test_the_function_gives_the_answers_of_the_start_commit(smiles, expected):
    assert _out(smiles) is expected


def test_the_two_positives_and_the_control_of_the_spy():
    """Copper sulfate and the ionic copper porphyrin are out of scope; sodium acetate and, since the
    table admits zinc, zinc diacetate and the ionic zinc porphyrin are not."""
    assert _out("[Cu+2].[O-]S(=O)(=O)[O-]") is True
    assert _out("C1=Cc2nc1c(-c1ccccc1)c1ccc([n-]1)c(-c1ccccc1)c1nc(c(-c3ccccc3)c3ccc([n-]3)c2-c2ccccc2)C=C1.[Cu+2]") is True
    assert _out("[Na+].CC(=O)[O-]") is False
    assert _out("[Zn+2].CC(=O)[O-].CC(=O)[O-]") is False


def test_every_element_of_the_allow_list_gives_the_old_answer():
    """Over all of ``METAL_ELEMENT_SYMBOLS``, with an acetate pair, a methanide pair and a hydride
    pair beside the dication: Groups 1-2 are out only beside a carbanion or a hydride, the
    metalloids never, zinc (admitted to both mechanisms) never, every other true metal always."""
    for symbol in sorted(METAL_ELEMENT_SYMBOLS):
        cation = f"[{symbol}+2]"
        if symbol in METALLOIDS or symbol == "Zn":
            expected = (False, False, False)
        elif symbol in GROUP_1 + GROUP_2:
            expected = (False, False, False)                               # CARBANION_SALTS since 2026-10-30
        else:
            expected = (True, True, True)
        got = tuple(_out(cation + tail * 2) for tail in (ACETATE, METHANIDE, HYDRIDE))
        assert got == expected, symbol


def test_zinc_has_its_mechanisms():
    """Zinc is admitted to ``DISCONNECT``, ``CONNECT_ANION``, ``BONDED_NAMES``, ``ELEMENT``, ``AMMINE``,
    ``MACROCYCLE``, ``DATIVE``, ``ION_SALTS`` and ``CARBANION_SALTS``, with the valence +2: a drawing with dative
    bonds is named through the same drawing without them, a drawing with Zn-X bonds is named
    through its ionic drawing, an organozinc cation drawn beside its anion is named through its
    bonded drawing, a bonded drawing with hydrogen, terminal ligands or a charge is named by its
    ligands, the atom, the ions and the isotopes are named by the element name, the ammonia of an
    ammine complex is cited as a component of the adduct, the porphyrin dianion of a zinc complex is cited
    on the porphyrin parent, and a salt or an adduct of zinc with organic ions, a carbanion or a hydride is
    composed from its ions."""
    assert is_named_metal("Zn") is True
    assert METAL_SCOPE["Zn"] == MetalScope(
        frozenset({DISCONNECT, CONNECT_ANION, BONDED_NAMES, ELEMENT, AMMINE, MACROCYCLE, DATIVE, ION_SALTS,
                   CARBANION_SALTS, ZERO_VALENT}),
        frozenset({2}),
        complete=True)
    assert has_mechanism("Zn", MACROCYCLE) is True
    assert all(has_mechanism(s, MACROCYCLE) is False for s in GROUP_1 + GROUP_2 + ("Cu", "Cd", "Hg", "Pt", "Fe", "Mg"))
    assert has_mechanism("Zn", DATIVE) is True
    assert all(has_mechanism(s, DATIVE) is False for s in GROUP_1 + GROUP_2 + ("Cu", "Cd", "Hg", "Pt", "Fe"))
    assert has_mechanism("Zn", AMMINE) is True
    assert all(has_mechanism(s, AMMINE) is False for s in GROUP_1 + GROUP_2 + ("Cu", "Cd", "Hg", "Pt", "Fe"))
    assert has_mechanism("Zn", ELEMENT) is True
    assert all(has_mechanism(s, ELEMENT) is False for s in GROUP_1 + GROUP_2 + ("Cu", "Cd", "Hg", "Pt", "Fe"))
    assert has_mechanism("Zn", BONDED_NAMES) is True
    assert all(has_mechanism(s, BONDED_NAMES) is False for s in GROUP_1 + GROUP_2 + ("Cu", "Cd", "Hg", "Pt"))
    assert has_mechanism("Zn", DISCONNECT) is True
    assert has_mechanism("Zn", CONNECT_ANION) is True
    assert all(has_mechanism(s, CONNECT_ANION) is False for s in GROUP_1 + GROUP_2 + ("Cu", "Cd", "Hg", "Pt"))
    assert has_mechanism("Zn", ION_SALTS) is True
    assert has_mechanism("Zn", CARBANION_SALTS) is True
    assert all(has_mechanism(s, DISCONNECT) is False for s in GROUP_1 + GROUP_2)
    assert has_mechanism("Cu", DISCONNECT) is False and has_mechanism("Cd", DISCONNECT) is False


def test_only_zinc_is_complete():
    """A metal is complete when the plan of its phase is done (``MetalScope.complete``): the failure label of
    an input of such metals that no producer names is UNNAMEABLE. Zinc is the one complete metal; the elements
    of Groups 1 and 2 are admitted to ``ION_SALTS`` alone and are not complete."""
    complete = sorted(s for s, scope in METAL_SCOPE.items() if scope.complete)
    assert complete == ["Zn"], complete
    assert all(METAL_SCOPE[s].complete is False for s in GROUP_1 + GROUP_2)
    assert MetalScope(frozenset({ION_SALTS}), frozenset({1})).complete is False      # the default


def test_mechanism_lookup():
    assert has_mechanism("Na", ION_SALTS) is True
    assert has_mechanism("Na", CARBANION_SALTS) is True
    assert has_mechanism("Xx", ION_SALTS) is False                         # an unknown symbol
    assert has_mechanism("Na", "no_such_mechanism") is False
    for symbol in GROUP_1 + GROUP_2:
        assert METAL_SCOPE[symbol].mechanisms == frozenset({ION_SALTS, CARBANION_SALTS, ZERO_VALENT}), symbol
        assert is_named_metal(symbol) is True, symbol
    assert is_named_metal("Xx") is False


def test_valence_field():
    assert has_valence("Na", 1) is True
    assert has_valence("Na", 2) is False
    assert has_valence("Ca", 2) is True
    assert has_valence("Ca", 1) is False
    assert has_valence("Xx", 1) is False                                   # an unknown symbol
    assert has_valence("Zn", 2) is True and has_valence("Zn", 1) is False and has_valence("Zn", 3) is False
    assert all(METAL_SCOPE[s].valences == frozenset({1}) for s in GROUP_1)
    assert all(METAL_SCOPE[s].valences == frozenset({2}) for s in GROUP_2)


def test_the_function_follows_the_table(monkeypatch):
    """A later phase admits a metal by editing the table: here copper is admitted to ``ION_SALTS``
    only, then to both mechanisms; the assemblies move out of the out-of-scope set accordingly."""
    copper_acetate = "[Cu+2].CC(=O)[O-].CC(=O)[O-]"
    copper_methanide = "[Cu+2].[CH3-].[CH3-]"
    assert _out(copper_acetate) is True                                    # copper has no entry
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Cu", MetalScope(frozenset({DISCONNECT}), frozenset({2})))
    assert _out(copper_acetate) is True                                    # DISCONNECT is no salt mechanism
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Cu", MetalScope(frozenset({ION_SALTS}), frozenset({2})))
    assert is_named_metal("Cu") is True and has_valence("Cu", 2) is True
    assert _out(copper_acetate) is False
    assert _out(copper_methanide) is True                                  # no carbanion mechanism yet
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Cu",
                        MetalScope(frozenset({ION_SALTS, CARBANION_SALTS}), frozenset({2})))
    assert _out(copper_methanide) is False
    assert _out("[Cu+2].[Pt+2].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-]") is True    # platinum is still out


def test_zero_valent_metals_are_admitted_to_that_alone():
    """The metals of the 2026-10-30 panel are admitted to ``ZERO_VALENT`` only: they cite a neutral atom beside
    ions as 'metal(0)' and nothing else; their failures keep the label UNSUPPORTED_ELEMENT (``is_named_metal``)."""
    for symbol in ZERO_VALENT_METALS:
        assert METAL_SCOPE[symbol] == MetalScope(frozenset({ZERO_VALENT}), frozenset()), symbol
        assert is_named_metal(symbol) is False, symbol
    assert _out("[Cu+2].CC(=O)[O-].CC(=O)[O-]") is True                     # no salt mechanism for copper
