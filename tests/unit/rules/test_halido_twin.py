"""Additive organometallic names with 'chlorido' ligands: read back through the 'chloro' twin.

The transition-metal additive names ('trichlorido(methyl)titanium') are names OPSIN 2.9.0 rejects (it has no
'chlorido' ligand word), and they used to ship with no read-back at all (the ``organometallic_additive``
carve-out). They are labelled below the PIN, and the default tier declines a name that is no PIN, so a name
that is read back only decides what a wider tier gives.

The rules:

* Compounds with at least one metal-carbon single bond (the Blue Book): "Compounds are named by
  citing the names of ligands, including hydrogen atom(s), in alphanumerical order, followed by the name of the
  metal"; '[Ti(CH3)Cl3] trichlorido(methanido)titanium, trichlorido(methyl)titanium' (:39789).
* Definitions of terms (:39761): "The names of anionic ligands, whether organic or inorganic, end in
  'o'. If the anion name ends in 'ate', 'ite', or 'ide', the final letter 'e' is replaced by 'o', giving 'ato',
  'ito', or 'ido'"; (:39767) "Because of a long established tradition, names for substituent groups as defined
  in Section can also be used as ligand names." OPSIN reads 'trichloro(methyl)titanium'.
* INTRODUCTION (:39735): "neither preferred IUPAC names or preselected names (see for
  organometallic compounds involving the transition elements (including the Group 3 elements) and Groups 1 and 2
  elements, except for 'ocene' compounds, are noted": the name is no PIN, so its label is below it and the
  default tier declines it (the wider tiers keep it).
"""
import pytest

from orthonym.validation.halido_twin import halido_twin
from orthonym.validation.twin_spellings import twin_of, twin_round_trip
from tests.support.default_tier import assert_default_tier_declines, default_tier_row, strict_path_row
from tests.support.rt_assert import _canon, _independent_parse, assert_full_rt, name_best_effort

# (smiles, the name, its wider-tier spelling = the twin)
ROWS = [
    ("C[Ti](Cl)(Cl)Cl", "trichlorido(methyl)titanium", "trichloro(methyl)titanium"),           #:39789
    ("C[Pt](C)(C)Br", "bromidotris(methyl)platinum", "bromotris(methyl)platinum"),
    ("Cl[Pt](C1=CC=CC=C1)(C1=CC=CC=C1)Cl", "dichloridobis(phenyl)platinum", "dichlorobis(phenyl)platinum"),
    ("C1(=CC=CC=C1)[Fe]I", "iodido(phenyl)iron", "iodo(phenyl)iron"),
    ("C(C)[Ni]Br", "bromido(ethyl)nickel", "bromo(ethyl)nickel"),
    ("Cl[Rh](C=C)C=C", "chloridobis(ethenyl)rhodium", "chlorobis(ethenyl)rhodium"),
    ("C(CCC)[Fe](CCCC)(CCCC)(CCCC)(Cl)(Cl)(Cl)Cl", "tetrakis(butyl)tetrachloridoiron",
     "tetrakis(butyl)tetrachloroiron"),
]


@pytest.mark.parametrize("smiles,name,twin", ROWS)
def test_the_twin_is_the_name_with_its_halide_ligands_spelled_chloro(smiles, name, twin):
    assert halido_twin(name) == twin
    assert twin_of(name) == (twin, ("halido_o",))


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name,twin", ROWS)
def test_opsin_reads_the_twin_to_the_input_and_rejects_the_name(smiles, name, twin):
    """A fresh OPSIN call: no parse of the 'chlorido' name, the sigma-bonded structure of the twin."""
    assert _independent_parse(name, smiles) is None
    parsed = _independent_parse(twin, smiles)
    assert parsed is not None and _canon(parsed) == _canon(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name,twin", ROWS)
def test_the_name_is_read_back_through_its_twin(smiles, name, twin):
    assert twin_round_trip(name, smiles) == "halido_o"
    row = strict_path_row(smiles)
    assert row["name"] == name, row
    assert row["gate_outcome"] == "twin_verified:halido_o" and row["verified"] == "opsin_twin", row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row     #: no PIN


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name,twin", ROWS)
def test_the_default_tier_declines_a_name_that_is_no_pin(smiles, name, twin):
    """It gave the name through a carve-out with no read-back; (:39735) notes no PIN for these metals."""
    assert_default_tier_declines(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name,twin", ROWS)
def test_best_effort_names_the_compound_in_the_blue_book_spelling(smiles, name, twin):
    """The 'chlorido' name had no read-back, so no wider tier gave it (best-effort abstained). The best-effort tier
    gives the Blue Book spelling -- (the Blue Book): an anion name ending in 'ide' ends in 'ido' as a
    ligand; prints '[Ti(CH3)Cl3]... trichlorido(methyl)titanium' (:39789) -- read back through its 'chloro'
    twin, which OPSIN reads to the input's full InChIKey (CO topic, user decision 2026-10-18)."""
    row = name_best_effort(smiles)
    assert row["name"] == name and row["tier"] == "systematic_verified", row
    assert row["gate_outcome"] == "twin_verified:halido_o" and row["verified"] == "opsin_twin", row
    assert_full_rt(twin, smiles)


@pytest.mark.opsin_gate
def test_a_name_that_cites_another_ligand_is_not_read_back():
    smiles = "C[Ti](Cl)(Cl)Cl"
    assert twin_round_trip("trichlorido(methyl)titanium", smiles) == "halido_o"
    assert twin_round_trip("dichlorido(methyl)titanium", smiles) is None
    assert twin_round_trip("trichlorido(ethyl)titanium", smiles) is None
    assert twin_round_trip("tribromido(methyl)titanium", smiles) is None


@pytest.mark.opsin_gate
def test_a_hapto_name_ships_nowhere_unverified():
    """A name with an eta descriptor has no spelling OPSIN reads (dan2097/opsin#350), no twin and no PIN
    INTRODUCTION, the Blue Book), so it does not ship (user decision 2026-11-01); the adduct name that reads
    back stands in its place."""
    row = default_tier_row("C=C.C=C.[Ni]")
    assert row["name"] != "bis(η²-ethene)nickel", row
    assert not str(row["gate_outcome"]).startswith("carveout:"), row
    assert twin_of("bis(η²-ethene)nickel") is None


def test_the_hapto_pattern_reads_every_descriptor():
    """The twin reads the eta, kappa and mu descriptors (``validation.halido_twin.HAPTO_DESCRIPTOR``)."""
    from orthonym.validation.halido_twin import HAPTO_DESCRIPTOR
    assert all(HAPTO_DESCRIPTOR.search(n) for n in ("bis(η2-ethene)nickel", "(κ2-x)iron", "μ-chlorido"))


@pytest.mark.parametrize("name", [
    "bis(η2-ethene)nickel", "dihydrido(naphthalen-2-yl)rhenium", "chloromethane", "trichloro(methyl)titanium",
    "chloridomethane", "(chlorido)", "ferrocene", None])
def test_no_twin_for_another_name(name):
    """A hapto name, a hydrido name (OPSIN reads the word), a name of a twin, a name without the shape of
    a transition-metal additive name."""
    assert halido_twin(name) == name


def test_the_twin_is_pure_and_not_rewritten_again():
    for _, name, twin in ROWS:
        assert halido_twin(name) == halido_twin(name)
        assert halido_twin(twin) == twin
