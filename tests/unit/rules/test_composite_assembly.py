"""Composite locants for an unbranched ring assembly of three to six identical cyclic systems.

``## **** UNBRANCHED RING ASSEMBLIES OF THREE THROUGH SIX IDENTICAL CYCLIC SYSTEMS`` (the Blue Book),
``****`` (:15655): composite locants (the ring of the assembly, the position in the ring as a superscript) "is
recommended for preferred IUPAC names. The numbering method using serially primed locants used in earlier
recommendations (refs. 1 and 2) may be used in general nomenclature."; ``### ****`` (:24070),
``****`` (:24078): "The preferred numbering for ring assemblies composed of three or more identical cyclic
systems uses composite locants rather than primed locants (see ".

With an OPSIN that reads composite locants (``ORTHONYM_OPSIN=main``) the name is the PIN and is labelled so; with OPSIN
2.9.0, the release, every name is the primed name of general nomenclature, labelled below the PIN, as before.
"""
import itertools
import random

import pytest
from rdkit import Chem

from orthonym import opsin_capabilities as caps
from orthonym.rules import composite_assembly as ca
from tests.support.default_tier import default_tier_row
from tests.support.opsin_marks import needs_composite_opsin, release_capabilities
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

TERPHENYL = "c1ccc(cc1)-c1ccc(cc1)-c1ccccc1"
TERPHENYLOL = "Oc1cc(ccc1-c1ccccc1)-c1ccccc1"
TETROL = "Oc1cccc(O)c1-c1cc(O)c(-c2ccccc2)c(O)c1"
QUATERPYRIDINE = "c1ccnc(c1)-c1cccc(n1)-c1cccc(n1)-c1ccccn1"
TERCYCLOPROPANE = "C1CC1C1CC1C1CC1"
QUATERCYCLOBUTANE = "C1(CCC1)C2C(C3CC(C4CCC4)C3)CC2"
#: a para-chlorophenyl and a ring with its chlorine next to the bond to the third ring
DICHLORO = "Clc1ccc(cc1)-c1ccc(-c2ccccc2)c(Cl)c1"

#: (Blue Book line, PIN, structure): the PINs the book prints for assemblies of three or more identical systems
BOOK_PINS = [
    # (:20598) '1^1,2^1:2^4,3^1-terphenyl (PIN)' (primed: 1,1':4',1''-terphenyl)
    ("20598", "1^1,2^1:2^4,3^1-terphenyl", TERPHENYL),
    # under examples (:27220) '[1^1,2^1:2^4,3^1-terphenyl]-2^2-ol (PIN; see ]'
    ("27220", "[1^1,2^1:2^4,3^1-terphenyl]-2^2-ol", TERPHENYLOL),
    # (:26899) '[1^1,2^1:2^4,3^1-terphenyl]-1^2,1^6,2^3,2^5-tetrol (PIN)' (primed: 2,3',5',6-tetrol)
    ("26899", "[1^1,2^1:2^4,3^1-terphenyl]-1^2,1^6,2^3,2^5-tetrol", TETROL),
    # (:15667) '1^2,2^2:2^6,3^2:3^6,4^2-quaterpyridine (PIN)' (primed: 2,2':6',2'':6'',2'''-)
    ("15667", "1^2,2^2:2^6,3^2:3^6,4^2-quaterpyridine", QUATERPYRIDINE),
    # Note (:15657): "two alternative names are 1,1':2',1''-tercyclopropane and 1^1,2^1:2^2,3^1-tercyclopropane"
    ("15657", "1^1,2^1:2^2,3^1-tercyclopropane", TERCYCLOPROPANE),
    # (:15663) '1^1,2^1:2^2,3^1:3^3,4^1' is lower than '1^1,2^1:2^3,3^1:3^2,4^1' (quatercyclobutane)
    ("15663", "1^1,2^1:2^2,3^1:3^3,4^1-quatercyclobutane", QUATERCYCLOBUTANE),
]

#: derived by the same rules from the examples above, 'terphenyl, quaterphenyl, etc.',
DERIVED = [
    ("1^1,2^1:2^3,3^1-terphenyl", "c1ccc(cc1)-c1cccc(c1)-c1ccccc1"),
    ("1^1,2^1:2^2,3^1-terphenyl", "c1ccc(cc1)-c1ccccc1-c1ccccc1"),
    ("1^1,2^1:2^4,3^1:3^4,4^1:4^4,5^1:5^4,6^1-sexiphenyl",
     "c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccccc1"),
    ("1^2,2^2:2^6,3^2-terpyridine", "c1ccnc(c1)-c1cccc(n1)-c1ccccn1"),
    ("1^2,2^2:2^5,3^2-terthiophene", "c1csc(c1)-c1ccc(s1)-c1cccs1"),
    ("1^2,2^2:2^6,3^2-ternaphthalene", "c1ccc2cc(ccc2c1)-c1ccc2cc(ccc2c1)-c1ccc2ccccc2c1"),
    # a suffix group and prefixes together (c), (f)): the suffix takes the lowest locants
    ("3^4-chloro[1^1,2^1:2^4,3^1-terphenyl]-2^2-ol", "Oc1cc(ccc1-c1ccccc1)-c1ccc(Cl)cc1"),
    ("1^4,2^3-dichloro-1^1,2^1:2^4,3^1-terphenyl", DICHLORO),
    # (g): the prefix cited first in alphanumerical order ('bromo') takes the lower locant
    ("1^4-bromo-3^4-chloro-1^1,2^1:2^4,3^1-terphenyl", "Clc1ccc(cc1)-c1ccc(cc1)-c1ccc(Br)cc1"),
    ("1^4-nitro-1^1,2^1:2^4,3^1-terphenyl", "O=[N+]([O-])c1ccc(cc1)-c1ccc(cc1)-c1ccccc1"),
    # (:27665): 'butoxy (preferred prefix)'
    ("1^4-butoxy-1^1,2^1:2^4,3^1-terphenyl", "CCCCOc1ccc(cc1)-c1ccc(cc1)-c1ccccc1"),
    # seniority: the acid is the suffix, the alcohol its prefix 'hydroxy'
    ("3^4-hydroxy[1^1,2^1:2^4,3^1-terphenyl]-1^4-carboxylic acid", "OC(=O)c1ccc(cc1)-c1ccc(cc1)-c1ccc(O)cc1"),
    # a heteromonocycle is numbered by its own rule (S1, N3:, the junctions then take what they take
    ("1^2,2^5:2^2,3^5-ter-1,3-thiazole", "c1csc(n1)-c1cnc(s1)-c1cncs1"),
    ("1^2,2^2:2^5,3^2-terpyrimidine", "c1cnc(nc1)-c1ncc(cn1)-c1ncccn1"),
    ("1^1,2^4:2^1,3^4-terpiperidine", "C1CCN(CC1)C1CCN(CC1)C1CCNCC1"),
    ("1^2,2^6:2^2,3^6-terquinoline", "c1ccc2nc(ccc2c1)-c1ccc2nc(ccc2c1)-c1ccc2ncccc2c1"),
]


# ---------------------------------------------------------------- with an OPSIN that reads composite locants

@needs_composite_opsin
@pytest.mark.parametrize("line,name,smiles", BOOK_PINS, ids=[r[0] for r in BOOK_PINS])
def test_the_book_pins_are_given_as_pins(line, name, smiles):
    # (:24078): composite locants are the preferred numbering, so the name carries no label
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["spelling_failures"] == [], row
    assert name_best_effort(smiles)["name"] == name
    assert_full_rt(name, smiles)


@needs_composite_opsin
@pytest.mark.parametrize("name,smiles", DERIVED, ids=[r[0][:40] for r in DERIVED])
def test_the_names_derived_by_the_same_rules(name, smiles):
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["spelling_failures"] == [], row
    assert_full_rt(name, smiles)


@needs_composite_opsin
def test_the_composite_order_is_not_the_primed_order(monkeypatch):
    # (:15115), under '### **** Numbering of phane parent hydrides': "The seniority of a composite
    # locant is determined first on the basis of its primary locants... and, if these locants are identical, on the basis
    # of the complete composite locant itself": 1^4 is lower than 2^2. The primed locants compare by number first (2' is
    # lower than 4), so the two numberings of this molecule are chosen the other way round:
    # forward 1^4,2^3 (primed 3',4) reverse 2^2,3^4 (primed 2',4'')
    assert default_tier_row(DICHLORO)["name"] == "1^4,2^3-dichloro-1^1,2^1:2^4,3^1-terphenyl"
    release_capabilities(monkeypatch)
    assert name_best_effort(DICHLORO)["name"] == "2',4''-dichloro-1,1':4',1''-terphenyl"


@needs_composite_opsin
@pytest.mark.parametrize("smiles", [TETROL, DICHLORO, "Oc1cc(ccc1-c1ccccc1)-c1ccc(Cl)cc1", QUATERPYRIDINE,
                                    "c1ccc(cc1)-c1cccc(c1)-c1ccccc1"])
def test_the_name_does_not_depend_on_the_order_of_the_atoms(smiles):
    # a numbering is never chosen by the order of the atoms of the input
    names = set()
    rng = random.Random(7)
    mol = Chem.MolFromSmiles(smiles)
    for _ in range(8):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        shuffled = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
        names.add(name_best_effort(shuffled)["name"])
    assert len(names) == 1, names


@needs_composite_opsin
@pytest.mark.parametrize("smiles", [
    "c1ccc(cc1)-c1ccccc1",                                 # two rings:, primed (1,1'-biphenyl is the PIN)
    "c1cc[nH]c1-c1ccc[nH]1",                               # indicated hydrogen: not built, keeps its name
    "C[C@H]1C[C@@H]1C1CC1C1CC1",                           # stereo: the descriptors are not placed here
    "OCc1ccc(cc1)-c1ccc(cc1)-c1ccccc1",                    # a group outside the plain kinds
    "CCCCCCOc1ccc(cc1)-c1ccc(cc1)-c1ccccc1",               # (:27665): hexyloxy has no retained name
    "c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccccc1",   # seven rings:
])
def test_a_molecule_outside_the_scope_is_declined(smiles):
    from orthonym.perception.rings import get_ring_systems
    from orthonym.rules.ring_assemblies import detect_ring_assembly
    mol = Chem.MolFromSmiles(smiles)
    info = detect_ring_assembly(mol, get_ring_systems(mol))
    assert info is None or ca.composite_assembly_name(mol, info) is None


@needs_composite_opsin
def test_a_composite_name_that_does_not_read_back_is_voided(monkeypatch):
    # the name is kept only when OPSIN reads it back to the molecule: a wrong one falls back to the primed name
    monkeypatch.setattr(ca, "_composite_name", lambda mol, info: "1^1,2^1:2^3,3^1-terphenyl")
    assert name_best_effort(TERPHENYL)["name"] == "1,1':4',1''-terphenyl"
    monkeypatch.setattr(ca, "_composite_name", lambda mol, info: "1^1,2^1:2^4,3^1-tercyclohexane")
    assert name_best_effort(TERPHENYL)["name"] == "1,1':4',1''-terphenyl"


#: isotope-labelled assemblies: the decorator splices the descriptor into the name of the stripped skeleton and tries
#: integer, primed and letter locants only (``rules.isotopes``), so the skeleton keeps its primed name (``hydro_assembly_
#: withheld``); a composite name would take a bare '4' that OPSIN main reads as another position,:43718)
LABELLED = [
    ("[2H]c1ccc(cc1)-c1ccc(cc1)-c1ccccc1", "(4-2H)-1,1':4',1''-terphenyl"),
    ("[2H]c1cccc(c1)-c1ccc(cc1)-c1ccccc1", "(3-2H)-1,1':4',1''-terphenyl"),
    ("[2H]c1ccc(nc1)-c1cccc(n1)-c1ccccn1", "(5-2H)-2,2':6',2''-terpyridine"),
    ("[2H]c1cc(ccc1-c1ccccc1)-c1ccccc1", "(2'-2H)-1,1':4',1''-terphenyl"),
    ("[2H]c1ccc(cc1)-c1ccc(cc1)-c1ccc([2H])cc1", "(4,4''-2H2)-1,1':4',1''-terphenyl"),
    ("Clc1ccc(cc1)-c1ccc([2H])c(c1)-c1ccccc1", "4-chloro(4'-2H)-1,1':3',1''-terphenyl"),
    ("Oc1ccc(cc1)-c1ccc(cc1)-c1ccccc1[2H]", "(2''-2H)[1,1':4',1''-terphenyl]-4-ol"),
]


@pytest.mark.parametrize("smiles,name", LABELLED, ids=[r[1] for r in LABELLED])
def test_a_labelled_assembly_keeps_its_primed_name(smiles, name):
    # the same name and label with either OPSIN: no composite locant, no PIN,:24078), and no name lost
    row = name_best_effort(smiles)
    assert (row["name"], row["tier"], row["is_pin"]) == (name, "systematic_verified", False), row
    assert "^" not in row["name"]


@needs_composite_opsin
def test_the_composite_producer_offers_nothing_inside_the_isotope_decorator():
    from orthonym.perception.rings import get_ring_systems
    from orthonym.rules.ring_assemblies import detect_ring_assembly, hydro_assembly_withheld
    mol = Chem.MolFromSmiles(TERPHENYL)
    info = detect_ring_assembly(mol, get_ring_systems(mol))
    assert ca.composite_assembly_name(mol, info) == "1^1,2^1:2^4,3^1-terphenyl"
    with hydro_assembly_withheld():
        assert ca.composite_assembly_name(mol, info) is None


# ---------------------------------------------------------------- with OPSIN 2.9.0 (whichever OPSIN is in use)

RELEASE_NAMES = [
    (TERPHENYL, "1,1':4',1''-terphenyl"),
    (TERPHENYLOL, "[1,1':4',1''-terphenyl]-2'-ol"),
    ("OC(=O)c1ccc(cc1)-c1ccc(cc1)-c1ccccc1", "[1,1':4',1''-terphenyl]-4-carboxylic acid"),
    (QUATERPYRIDINE, "2,2':6',2'':6'',2'''-quaterpyridine"),
    (TERCYCLOPROPANE, "1,1':2',1''-tercyclopropane"),
    (DICHLORO, "2',4''-dichloro-1,1':4',1''-terphenyl"),
]


@pytest.mark.parametrize("smiles,primed", RELEASE_NAMES, ids=[r[1] for r in RELEASE_NAMES])
def test_with_opsin_2_9_0_the_primed_names_stay_below_the_pin(monkeypatch, smiles, primed):
    # (:24078): the primed locants of earlier recommendations "may be used in general nomenclature"; OPSIN 2.9.0
    # reads no composite locants, so nothing is offered and every name, tier and label is what it was
    release_capabilities(monkeypatch)
    assert not caps.composite_assembly_enabled()
    monkeypatch.setattr(ca, "_composite_name", lambda mol, info: pytest.fail("the composite producer must not run"))
    row = name_best_effort(smiles)
    assert row["name"] == primed and row["tier"] == "systematic_verified", row
    assert [f["rule"] for f in row["spelling_failures"]] == ["P-52.2.7.2"], row
    default = default_tier_row(smiles)
    assert default["tier"] == "abstain" and default["limit_code"] == "NO_VERIFIED_PIN", default


def test_the_capability_follows_the_probe_and_a_missing_entry_reads_as_unable(monkeypatch):
    assert caps._PROBES["composite_assembly"][0] == "1^1,2^1:2^4,3^1-terphenyl"
    monkeypatch.setattr(caps, "opsin_capabilities", lambda: {"phane": False, "composite_assembly": True})
    assert caps.composite_assembly_enabled() is True
    monkeypatch.setattr(caps, "opsin_capabilities", lambda: {"phane": True, "halogen_uide": True})
    assert caps.composite_assembly_enabled() is False


# ---------------------------------------------------------------- the pieces (no OPSIN)

@pytest.mark.parametrize("line,name,smiles", BOOK_PINS, ids=[r[0] for r in BOOK_PINS])
def test_the_spelling_checks_pass_the_book_pins(line, name, smiles):
    from orthonym.validation.pin_spelling import check_pin_spelling
    assert check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True) == [], name


def test_the_lexer_reads_the_bracket_of_a_composite_assembly_like_a_primed_one():
    # the brackets of a ring assembly descriptor are ignored in the nesting order of enclosing marks
    from orthonym.validation.spelling.lexer import enclosures, normalise
    for name in ("[1^1,2^1:2^4,3^1-terphenyl]-2^2-ol", "[1,1':4',1''-terphenyl]-2'-ol"):
        assert [e.kind for e in enclosures(normalise(name))] == ["assembly"], name


@pytest.mark.parametrize("smiles,count", [("c1ccccc1", 12), ("c1ccncc1", 2), ("C1CC1", 6), ("c1ccc2ccccc2c1", 4),
                                          ("c1ccsc1", 2), ("O1CCOCC1", 4)])
def test_the_symmetries_of_a_ring_system(smiles, count):
    mol = Chem.MolFromSmiles(smiles)
    assert len(ca._automorphisms(mol, list(range(mol.GetNumAtoms())))) == count


@pytest.mark.parametrize("smiles,count,heteroatoms,at_one", [
    ("c1ccccc1", 12, [], None),                    # a carbocycle: any atom is 1, either way round
    ("C1CC1", 6, [], None),
    ("c1ccncc1", 2, [1], "N"),                     # pyridine N1
    ("c1cncnc1", 2, [1, 3], "N"),                  # pyrimidine N1,N3
    ("c1cscn1", 1, [1, 3], "S"),                   # 1,3-thiazole: S1 (the senior element takes 1), N3, one direction
    ("c1cnoc1", 1, [1, 2], "O"),                   # 1,2-oxazole: O1, N2
    ("O1CCOCC1", 4, [1, 4], None),                 # 1,4-dioxane
])
def test_the_numberings_of_a_monocycle(smiles, count, heteroatoms, at_one):
    # (heteroatoms: lowest locants as a set, then O > S > Se > Te > N > P...): the numberings of the usual kind
    mol = Chem.MolFromSmiles(smiles)
    family = ca._monocycle_numberings(mol, list(mol.GetRingInfo().AtomRings()[0]))
    assert len(family) == count
    for numbering in family:
        assert sorted(numbering[a.GetIdx()] for a in mol.GetAtoms() if a.GetAtomicNum() != 6) == heteroatoms
        if at_one:
            assert [a.GetSymbol() for a in mol.GetAtoms() if numbering[a.GetIdx()] == 1] == [at_one]


def _chain_pieces(mol):
    from orthonym.perception.rings import get_ring_systems
    from orthonym.rules.ring_assemblies import detect_ring_assembly
    info = detect_ring_assembly(mol, get_ring_systems(mol))
    junction, count = ca._chain(info)
    families, groups = ca._families(mol, info), ca._groups(mol, info)
    return junction, count, families, groups, list(Chem.CanonicalRankAtoms(mol, breakTies=False))


def _every_numbering_best(mol):
    """The criteria of the best numbering of the whole chain found by trying every numbering of every ring and both
    orientations at once (no use of the per-ring choice that ``_oriented`` makes), as ``(junctions, written, suffix
    locants, prefix locants, prefixes in alphanumerical order)`` of composite locants."""
    from orthonym.assembly.naming_utils import alpha_sort_key
    junction, count, families, groups, _ = _chain_pieces(mol)
    best = None
    forward = list(range(count))
    for order in (forward, forward[::-1]):
        for combo in itertools.product(*[families[s] for s in order]):
            numbering = dict(zip(order, combo))
            junctions, written, suffix, prefixes = [], [], [], []
            try:
                for ring, s in enumerate(order, start=1):
                    ends = [junction[(s, order[ring - 1 + step])] for step in (-1, 1) if 0 <= ring - 1 + step < count]
                    cited = [(ring, numbering[s][j]) for j in ends]
                    written += cited
                    junctions += sorted(cited)
                    suffix += [(ring, numbering[s][g.atom]) for g in groups if g.system == s and g.name is None]
                    prefixes += [(alpha_sort_key(g.name), (ring, numbering[s][g.atom]))
                                 for g in groups if g.system == s and g.name is not None]
            except KeyError:
                continue
            key = (junctions, written, sorted(suffix), sorted(loc for _, loc in prefixes),
                   [loc for _, loc in sorted(prefixes)])
            if best is None or key < best:
                best = key
    return best


@pytest.mark.parametrize("smiles", [TERPHENYL, TETROL, DICHLORO, "Oc1cc(ccc1-c1ccccc1)-c1ccc(Cl)cc1",
                                    "Cc1cccc(c1)-c1cc(C)c(cc1Cl)-c1ccc(O)cc1", "Clc1ccc(cc1)-c1cccc(c1C)-c1ccccc1"])
def test_the_ring_by_ring_choice_is_the_choice_over_every_numbering(smiles):
    # each ring is numbered by the criteria of its own atoms alone (the rings share no atom), so the choice that
    # ``_oriented`` makes ring by ring is the lowest of the criteria over the product of every numbering of every ring
    mol = Chem.MolFromSmiles(smiles)
    junction, count, families, groups, ranks = _chain_pieces(mol)
    forward = list(range(count))
    keys = [r[0][:5] for order in (forward, forward[::-1])
            for r in [ca._oriented(order, junction, families, groups, ranks)]]
    assert min(keys) == _every_numbering_best(mol)
