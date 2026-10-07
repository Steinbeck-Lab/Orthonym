"""Heteromonocycles with a =X group on a ring atom take the hydrogen of their mancude parent
(``heterocycles.heteromonocycle_ring_hydrogen`` in the heterocycle assembly), at both tiers.

The rules: (the Blue Book) indicated hydrogen of the mancude monocycle;
 (:24768) it sits on the suffix position ('4H-pyran-4-one (PIN) pyran-4-one',
:28414); (:24794) the remaining one at the lowest position ('2H,4H-1,3-dioxin-4-
one'); (:24689) 'added indicated hydrogen' when the parent has none to give
('oxepin-3(2H)-one'; 'pyridin-2(1H)-one (PIN)':21316), "preferred over the use of
nondetachable hydro prefixes" ('2H-pyran-3(6H)-one', not '3,6-dihydro-2H-pyran-3-one');
 Note (:24691) hydrogen placed after the group is introduced is "not recommended"
('2H-oxepin-3-one'); (:24864) a prefix =X is placed after the hydrogen
('4-methylidene-4H-imidazole'). The retained stems of (imidazole, pyrazole,
pyrrole, pyran) replace the Hantzsch-Widman stems where the table holds the mancude ring.
Every expected name reads back to the input's full InChIKey with OPSIN 2.9.0."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots

MONOCYCLES = [
    ("O=c1occo1", "2H-1,3-dioxol-2-one"),
    ("O=c1ccss1", "3H-1,2-dithiol-3-one"),
    ("O=c1nncco1", "2H-1,3,4-oxadiazin-2-one"),
    ("O=C1C=COCO1", "2H,4H-1,3-dioxin-4-one"),
    ("O=C1C=CC=COC=N1", "4H-1,3-oxazocin-4-one"),
    ("O=C1C=CC=COC1", "oxepin-3(2H)-one"),
    ("O=C1C=COC=CN1", "1,4-oxazepin-5(4H)-one"),
    ("S=C1C=NN=C1", "4H-pyrazole-4-thione"),
    ("N=C1C=CSSS1", "4H-1,2,3-trithiin-4-imine"),
    ("C=C1C=NC=N1", "4-methylidene-4H-imidazole"),
    ("O=C1C=CO1", "2H-oxet-2-one"),
    ("O=c1cn1", "2H-azirin-2-one"),
    ("O=C1COCC=C1", "2H-pyran-3(6H)-one"),
    ("O=C1N=CC=N1", "2H-imidazol-2-one"),
    ("O=C1C=CN=N1", "3H-pyrazol-3-one"),
    ("O=[C]1C=C[CH]=[GeH]1", "2H-germol-2-one"),
    ("O=C1C=CC=P1", "2H-phosphol-2-one"),
    ("O=c1bccccc1", "2H-borepin-2-one"),
    ("C1CCC(=CC1)C[C@@H]2C(=O)C=C[C@H](O2)CO",
     "(2R,6S)-2-[(cyclohex-1-en-1-yl)methyl]-6-(hydroxymethyl)-2H-pyran-3(6H)-one"),
]

#: names that were right before and stay right
UNCHANGED = [
    ("C1=COCO1", "2H-1,3-dioxole"),
    ("CC1=COCO1", "4-methyl-2H-1,3-dioxole"),
    ("O=C1C=CC(=O)N1", "1H-pyrrole-2,5-dione"),                                  #:33843
    ("O=c1cccc[nH]1", "pyridin-2(1H)-one"),                                      #:21316
    ("O=C1CCOC=C1", "2,3-dihydro-4H-pyran-4-one"),
    ("O=C1C=CC=CO1", "2H-pyran-2-one"),
    ("O=C1CCOCC1", "oxan-4-one"),                                                #:24772
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="ring-hydrogen"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", MONOCYCLES + UNCHANGED)
def test_a_heteromonocycle_with_a_group_cites_its_hydrogen(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])


#: the hydrogen is right; the thione is still cited as a prefix, so the name is not the PIN
#:, the Blue Book: ketones and their chalcogen analogues are senior
#: to the classes these rings have). Kept as the producer writes it, labelled below the PIN
#: (``test_a_ring_group_cited_below_its_class_is_not_the_pin``).
THIONE_PREFIX = [
    ("S=c1occo1", "2-sulfanylidene-2H-1,3-dioxole"),
    ("S=C1C=CC=N1", "2-sulfanylidene-2H-pyrrole"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", THIONE_PREFIX)
def test_a_thione_prefix_ring_cites_its_hydrogen(smiles, name):
    assert _row(smiles, "best-effort").get("name") == name


#: a ring carbon's =X group cited as a PREFIX while the name cites no suffix, or only a suffix
#: of a class junior to the group's, is not the PIN. (the Blue Book) ranks
#: class 16, ketones, pseudoketones and heterones (:18189), with their chalcogen analogues
#::29502, '-thione' / 'sulfanylidene', '-selone' / 'selanylidene'), before class 17
#: hydroxy compounds (:18190) and class 19 amines (:18192). The PINs cite the group as the
#: suffix ('2H-1,3-dioxole-2-thione', '2H-pyrrole-2-thione', '2H-1,3-dioxole-2-selone',
#: '6-amino-5-fluoropyrimidine-2(1H)-thione', '6-aminopyrimidine-2(1H)-thione',
#: '5-hydroxy-2H-pyran-2-thione', '2-aminopyrimidine-4(1H)-thione'; OPSIN 2.9.0 FULL). The
#: producer's name is kept below the PIN: the default tier declines, best-effort keeps it.
GROUP_BELOW_ITS_CLASS = [
    ("S=c1occo1", "2-sulfanylidene-2H-1,3-dioxole"),
    ("S=C1C=CC=N1", "2-sulfanylidene-2H-pyrrole"),
    ("[Se]=c1occo1", "2-selanylidene-2H-1,3-dioxole"),
    ("Nc1[nH]c(=S)ncc1F", "5-fluoro-2-sulfanylidene-2,3-dihydropyrimidin-4-amine"),
    ("Nc1ccnc(=S)[nH]1", "2-sulfanylidene-2,3-dihydropyrimidin-4-amine"),
    ("Oc1ccc(=S)oc1", "2-sulfanylidene-2H-pyran-5-ol"),
    ("Nc1nc(=S)cc[nH]1", "4-sulfanylidene-1,4-dihydropyrimidin-2-amine"),
]

#: a group cited as a prefix beside a SENIOR suffix stays the PIN: nitriles 14 (:18187) and
#: aldehydes 15 (:18188) before class 16; amines 19 (:18192) before imines 20 (:18193); C=O
#: before C=S:29561, '2-sulfanylidene-1,3-thiazolidin-4-one (PIN)':29569).
GROUP_BESIDE_A_SENIOR_SUFFIX = [
    ("S=c1ccoc(C#N)c1", "4-sulfanylidene-4H-pyran-2-carbonitrile"),
    ("S=c1ccoc(C=O)c1", "4-sulfanylidene-4H-pyran-2-carbaldehyde"),
    ("Nc1ccc(=N)oc1", "2-imino-2H-pyran-5-amine"),
    ("O=c1[nH]c(=S)cc[nH]1", "4-sulfanylidene-3,4-dihydropyrimidin-2(1H)-one"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", GROUP_BELOW_ITS_CLASS)
def test_a_ring_group_cited_below_its_class_is_not_the_pin(smiles, name):
    assert _row(smiles, "pin")["tier"] == "abstain"
    best = _row(smiles, "best-effort")
    assert (best.get("name"), best["tier"]) == (name, "systematic_verified"), best


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", GROUP_BESIDE_A_SENIOR_SUFFIX)
def test_a_ring_group_beside_a_senior_suffix_stays_the_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])


def test_the_numberings_that_keep_the_heteroatom_locants():
    """One helper enumerates the numberings of a monocycle that keep its heteroatom
    locants (b) before (c)-(g)); the indicated-hydrogen writers choose among
    them. 1,3-dioxine: two (O1 and O3 swap), both with the oxygen atoms at 1 and 3;
    pyridine: two (the two directions from N1)."""
    from rdkit import Chem
    from orthonym.rules import heterocycles
    mol = Chem.MolFromSmiles("C1=COCOC1")          # atoms 0..5: C C O C O C
    orders = list(heterocycles._orders_keeping_heteroatom_map(mol, [2, 3, 4, 5, 0, 1]))
    assert orders == [[2, 3, 4, 5, 0, 1], [4, 3, 2, 1, 0, 5]]
    assert all(heterocycles._heteroatom_locant_map(mol, o) == ((1, "O"), (3, "O"))
               for o in orders)
    pyr = Chem.MolFromSmiles("c1ccncc1")           # N is atom 3
    assert len(list(heterocycles._orders_keeping_heteroatom_map(pyr, [3, 4, 5, 0, 1, 2]))) == 2
    assert list(heterocycles._orders_keeping_heteroatom_map(
        pyr, [3, 4, 5, 0, 1, 2], ((2, "N"),))) == [[4, 3, 2, 1, 0, 5], [2, 3, 4, 5, 0, 1]]


def test_the_retained_stem_search_lets_an_unexpected_error_through(monkeypatch):
    """``_mancude_retained_stem`` skips a placement that does not sanitize (no such
    tautomer) and an RDKit invariant error on it, and nothing else: an unexpected error
    is not read as 'no tautomer'."""
    from rdkit import Chem
    from orthonym.rules import heterocycles
    mol = Chem.MolFromSmiles("C1=CCOC=C1")
    ring = {a.GetIdx() for a in mol.GetAtoms()}
    assert heterocycles._mancude_retained_stem(mol, ring) == "pyran"

    def _boom(*args, **kwargs):
        raise ZeroDivisionError("planted")
    monkeypatch.setattr(heterocycles.Chem, "MolToSmiles", _boom)
    with pytest.raises(ZeroDivisionError):
        heterocycles._mancude_retained_stem(mol, ring)
