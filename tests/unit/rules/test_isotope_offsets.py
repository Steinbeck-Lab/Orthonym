"""Where the isotope placement search offers a descriptor in the skeleton name of a mixture (``rules.isotope_offsets``).

A descriptor is read onto the part of the name after it, within one component of a mixture name, and a component of
amount ``n`` is read as ``n`` identical molecules (OPSIN: '(2H2)ethan-1-ol—propane—methane (1/1/2)' is read as
C.CCC.C(C)(O)([2H])[2H].C). So a descriptor can stand only in a component of the structure of a labelled fragment of the
input, and one descriptor labels the copies of one species. The search used to offer every offset of the whole name with
every locant up to the heavy atoms of the whole mixture (PubChem, filtered run 2026-10-30: 71% of the 79,635 NATIVE_HANG
rows and 97% of the 6,079 TIMEOUT rows are isotope-labelled mixtures; one CD3 of five components: 42,098 OPSIN parses in
90 s, none read to the input). Counts of candidates, no clock.
"""
import pytest
from rdkit import Chem, RDLogger

from orthonym.rules import isotopes
from orthonym.rules.isotope_offsets import labelled_species, restrict_offsets

pytestmark = pytest.mark.unit

ETHANOL_METHANE = "ethan-1-ol—methane (1/1)"


def _mol(smiles):
    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smiles)


def test_labelled_species_counts_the_species_not_the_fragments():
    assert len(labelled_species(_mol("[2H]C.[2H]C.CCO"))) == 1          # two copies of one species
    assert len(labelled_species(_mol("[2H]C.[3H]F.CCO"))) == 2
    assert len(labelled_species(_mol("[2H]C.[3H]C"))) == 2              # the same skeleton, another nuclide
    assert labelled_species(_mol("CCO")) == []


@pytest.mark.parametrize("smiles", ["[2H]C.[3H]F.CCO", "[2H]C.[3H]C.CCO", "[2H]O.CC[2H]"])
def test_one_descriptor_does_not_label_two_species(smiles):
    skel = ETHANOL_METHANE
    assert restrict_offsets(list(range(1, len(skel))), skel, _mol(smiles)) == []


def test_the_descriptor_stands_in_the_component_of_the_labelled_fragment():
    skel = ETHANOL_METHANE
    offsets = list(range(1, len(skel)))
    start = skel.index("methane")
    kept = restrict_offsets(offsets, skel, _mol("[2H]C.CCO"))
    assert kept == [o for o in offsets if start <= o <= start + len("methane")]
    kept_ol = restrict_offsets(offsets, skel, _mol("[2H]OCC.C"))
    assert kept_ol == [o for o in offsets if o <= len("ethan-1-ol")]


def test_copies_of_one_labelled_species_keep_the_component_of_that_species():
    skel = "ethan-1-ol—methane (1/2)"
    offsets = list(range(1, len(skel)))
    start = skel.index("methane")
    assert restrict_offsets(offsets, skel, _mol("[2H]C.[2H]C.CCO")) == [
        o for o in offsets if start <= o <= start + len("methane")]


def test_a_name_that_is_no_mixture_keeps_every_offset():
    offsets = list(range(1, 9))
    assert restrict_offsets(offsets, "ethanol", _mol("[2H]OCC")) == offsets
    assert restrict_offsets(offsets, "sodium acetate", _mol("[2H]C([2H])([2H])C(=O)[O-].[Na+]")) == offsets


def test_a_component_that_opsin_does_not_read_is_kept():
    skel = "notaname—methane (1/1)"
    offsets = list(range(1, len(skel)))
    assert restrict_offsets(offsets, skel, _mol("[2H]C.CCO")) == offsets


def test_an_unlabelled_molecule_keeps_every_offset():
    offsets = list(range(1, len(ETHANOL_METHANE)))
    assert restrict_offsets(offsets, ETHANOL_METHANE, _mol("CCO.C")) == offsets


MIXTURE = "[2H]C.CCO.CCN(CC)CC.CC(=O)Nc1ccc(O)cc1.C=CC(=O)N.OC(=O)CC(O)(CC(O)=O)C(O)=O"


def _round_trips(monkeypatch):
    calls = []
    real = isotopes._isotope_round_trips
    monkeypatch.setattr(isotopes, "_isotope_round_trips", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    return calls


@pytest.mark.opsin_gate
def test_the_search_of_a_mixture_reads_only_the_component_of_the_label(monkeypatch):
    """A labelled methane among five other components: 204 candidates were read before, 16 now, the same name."""
    from orthonym import Orthonym
    calls = _round_trips(monkeypatch)
    row = Orthonym(style="pin", general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True).name_tiered(MIXTURE)
    assert row["name"].endswith("(2H1)methane (1/1/1/1/1/1)"), row
    assert len(calls) <= 60, len(calls)


@pytest.mark.opsin_gate
def test_two_labelled_species_are_not_swept_by_the_single_descriptor_search(monkeypatch):
    """No single descriptor reads back to a mixture labelled in two species: the 3,000 candidates of the single
    descriptor search (14,964 candidates in all, the other strategies included) are not read."""
    from orthonym import Orthonym
    calls = _round_trips(monkeypatch)
    namer = Orthonym(style="pin", general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)
    row = namer.name_tiered("[2H]C.[3H]F.CCO.CCN(CC)CC.CC(=O)Nc1ccc(O)cc1.C=CC(=O)N")
    assert row["tier"] == "abstain"
    assert len(calls) <= 4000, len(calls)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("text, expected", [
    ("methane", 1), ("ethan-1-ol", 2), ("naphthalene", 8), ("1H-indole", 7), ("1,1'-biphenyl", 6),
    ("dihydrogen", 0),                         # the atoms of the name have letter locants only
])
def test_the_highest_locant_opsin_gives_the_atoms_of_a_name(text, expected):
    from orthonym.rules.isotope_offsets import _highest_locant
    assert _highest_locant(text) == expected


@pytest.mark.opsin_gate
def test_a_name_whose_locants_are_not_numbers_has_no_limit():
    """L-arginine's atoms carry 'guanidino-C' and 'omega': no number says what is the highest, so every locant stays."""
    from orthonym.rules.isotope_offsets import _highest_locant
    assert _highest_locant("L-arginine") is None
    assert _highest_locant("notaname") is None


@pytest.mark.opsin_gate
def test_the_locants_asked_are_those_of_the_component_that_takes_the_descriptor():
    """A label on the methane of a mixture asks locants up to 1 and the margin of the gaps in OPSIN's numbering
    (``isotope_search.GAP_WINDOW``), not one for every heavy atom of the whole mixture."""
    from orthonym.rules.isotope_offsets import search_space
    from orthonym.rules.isotope_search import GAP_WINDOW
    offsets = list(range(1, len(ETHANOL_METHANE)))
    kept, n_pos = search_space(ETHANOL_METHANE, _mol("[2H]C.CCO"), offsets, 60)
    assert n_pos == 1 + GAP_WINDOW
    assert kept[0] == 0 and kept[1:] == restrict_offsets(offsets, ETHANOL_METHANE, _mol("[2H]C.CCO"))
    # a limit above the heavy atoms is no limit, and a name that is no mixture is read as a whole
    assert search_space(ETHANOL_METHANE, _mol("[2H]C.CCO"), offsets, 3)[1] == 3
    assert search_space("naphthalene", _mol("[2H]c1ccc2ccccc2c1"), [1, 2], 60) == ([0, 1, 2], 8 + GAP_WINDOW)
    assert search_space("L-arginine", _mol("[2H]OC(=O)C(N)CCCNC(N)=N"), [1, 2], 60)[1] == 60


def test_a_mixed_descriptor_of_many_labels_is_refused_before_its_locant_sets_are_built(monkeypatch):
    """``_decorate_mixed_locant`` built every combination of ``count`` positions before it applied its bound (256 sets):
    11 13C labels on a molecule of 78 heavy atoms are 4e12 tuples, and the row (PubChem, a protected nucleoside) did not
    return in 140 s. The bound is asked of the count first, and ends the search at the same place."""
    import itertools
    from orthonym.rules import isotopes

    def explode(*args, **kwargs):
        raise AssertionError("the combinations of the positions were built")

    monkeypatch.setattr(itertools, "combinations", explode)
    original = _mol("[13CH3][13CH2][13CH2][13CH2][13CH2][13CH2][13CH2][13CH2][13CH2][13CH2][13CH2]N")
    keys = [((13, "C"), 11, 11), ((15, "N"), 1, 1)]
    skeleton = "undecan-1-amine"
    assert isotopes._decorate_mixed_locant(skeleton, keys, original, 78, "CCCCCCCCCCCN", None) is None


def test_no_component_matching_the_labelled_species_keeps_every_offset():
    """A labelled species sits in some component: when OPSIN's reading of no component has its key (a component named
    through another drawing of the same compound, 'zinc diacetate' read as zinc(2+) and two acetates), the comparison
    failed and the search keeps every offset (review of lane C, 2026-10-31: it returned none, and the name was lost)."""
    skel = "zinc diacetate—ethane (1/1)"
    offsets = list(range(len(skel)))
    mol = _mol("[2H]C([2H])([2H])C(=O)O[Zn]OC(=O)C([2H])([2H])[2H].CC")
    assert restrict_offsets(offsets, skel, mol) == offsets
