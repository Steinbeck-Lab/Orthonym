"""The ionic drawing of a metal halide offered beside a radical part that the rewrite does not touch.

Standard InChI ignores metal-ligand bonds, so ``Cl[Zr+2]Cl`` and ``[Cl-].[Cl-].[Zr+4]`` are one compound, and the
offer of ``assembly.metal_offer`` names the ionic drawing where the drawing as written has no name (mechanism DISCONNECT
of ``data/metal_scope.py``; the name is labelled below the PIN, "Structures for Lewis 'adducts'",
the Blue Book: "One of the fundamental principles of organic chemical nomenclature is to name the structure that is
drawn"). A radical centre used to block every rewrite, because a name holding a radical is judged by whole graph. A
radical in a part that holds no broken bond (the silylene bridge of a zirconocene drawn apart, a cyclopentadienyl drawn
as radicals) stays as drawn: the count of atoms and charges compares every atom's radical electrons, hydrogens and charge
in both drawings, and the name is judged against the ionic drawing by whole graph. A radical in the part the rewrite
breaks still blocks it.
"""
import pytest

from orthonym import opsin_capabilities as caps
from tests.support.rt_assert import name_best_effort


def _main_reads_radical_adducts() -> bool:
    try:
        return caps.radical_adduct_enabled()
    except Exception:
        return False


needs_main = pytest.mark.skipif(not _main_reads_radical_adducts(),
                                reason="the OPSIN in use reads no adduct with a radical component")
needs_290 = pytest.mark.skipif(_main_reads_radical_adducts(),
                               reason="the OPSIN in use reads adducts with a radical component")


@pytest.mark.opsin_gate
@needs_290
def test_a_metal_halide_beside_a_silylene_on_opsin_2_9_0():
    """OPSIN 2.9.0 reads no radical component of an adduct; the silylene is named as the λ2 silane."""
    r = name_best_effort("C[Si]C.Cl[Zn]Cl")
    assert r["name"] == "dimethyl-λ2-silane—chloride—zinc(2+) (1/2/1)" and r["tier"] == "systematic_verified", r
    assert not r["is_pin"], r


@pytest.mark.opsin_gate
@needs_main
@pytest.mark.parametrize("smiles,name", [
    ("C[Si]C.Cl[Zn]Cl", "chloride—zinc(2+)—dimethylsilylidene (2/1/1)"),
    ("C[SiH]C.Cl[Zn]Cl", "chloride—zinc(2+)—dimethylsilyl (2/1/1)"),
    ("[CH2][CH2].Cl[Zn]Cl", "chloride—zinc(2+)—ethane-1,2-diyl (2/1/1)"),
    ("[CH]1[CH][CH][CH][CH]1.[CH]1[CH][CH][CH][CH]1.Cl[Zr+2]Cl",
     "chloride—zirconium(4+)—cyclopentane-1,2,3,4,5-pentayl (2/1/2)"),
])
def test_a_metal_halide_beside_an_untouched_radical_part(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r


def test_an_ionic_drawing_a_carbon_radical_could_join_is_not_offered():
    """'[Cl-].[Cu+].[CH3]' would be judged against its joined drawing 'C[Cu]Cl', not against itself."""
    from rdkit import Chem
    from orthonym.assembly.metal_offer import metal_disconnected_spelling_and_order
    assert metal_disconnected_spelling_and_order(Chem.MolToSmiles(Chem.MolFromSmiles("Cl[Cu].[CH3]"))) is None
    assert metal_disconnected_spelling_and_order("C[Si]C.Cl[Zn]Cl") is not None


@pytest.mark.opsin_gate
def test_the_closed_shell_twins_keep_their_own_names():
    """The radical-blind key does not merge the twins: each name is judged by graph."""
    assert name_best_effort("C[SiH2]C.Cl[Zn]Cl")["name"] == "dimethylsilane—chloride—zinc(2+) (1/2/1)"
    assert name_best_effort("C=C.Cl[Zn]Cl")["name"] == "ethene—chloride—zinc(2+) (1/2/1)"


@pytest.mark.opsin_gate
def test_a_radical_in_the_part_the_rewrite_breaks_still_blocks_it():
    assert name_best_effort("Cl[Zn]O[CH2]")["tier"] == "abstain"
