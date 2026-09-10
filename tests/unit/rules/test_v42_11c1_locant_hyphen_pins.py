""" a phase Wave-C batch 1 (task 11C1): name-string pins for a batch of
narrow spelling / hyphen / P-14.3.4 locant-omission fixes. Each assertion pins a
Blue-Book PIN and cites its governing line. Grouped by the brief's item numbers.

The two ``general_fallback`` targets (#16, #23 full-molecule) abstain on the
default (PIN-only) config and are named through the bb_conformance config, which
is where the fix is measured (the CLAUDE.md invariant-16 config gap).
"""
import pytest

from orthonym import Orthonym, name_compound

# The bb_conformance measurement config (eval/bb_conformance/bb_measure.py:55).
_GF = dict(general_fallback=True, general_fallback_unverified=True,
           allow_aromatic_general=True)


@pytest.fixture(scope="module")
def gf_namer():
    return Orthonym(**_GF)


# --- Group A #12/#33: ring-cation prefix -> locanted-core hyphen (P-14.5.2) ----
# the Blue Book "a name-part is separated from a following locant by a hyphen". The
# prefix joins a LETTER-initial stem bare but a LOCANT-initial stem with '-'.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("C[N+]12CCC(CC1)C2.[Cl-]", "1-methyl-1-azabicyclo[2.2.1]heptan-1-ium chloride"),
    ("C[N+]12CCC(CC1)CC2", "1-methyl-1-azabicyclo[2.2.2]octan-1-ium"),
    ("C[n+]1ccsc1.[Cl-]", "3-methyl-1,3-thiazol-3-ium chloride"),
])
def test_ring_cation_prefix_locant_hyphen(smi, expected):
    assert name_compound(smi) == expected


# --- Group A #29: replacement 'a'-prefix -> spiro locant hyphen (P-14.5) -------
# '6,6'-dioxa' + '3,3'-spirobi[...]' -> '6,6'-dioxa-3,3'-spirobi[...]' (the Blue Book).
@pytest.mark.opsin_gate
def test_replacement_prefix_before_spiro_locant_hyphen():
    assert (name_compound("C1OC2CC1CC1(CC3COC(C3)C1)C2")
            == "6,6'-dioxa-3,3'-spirobi[bicyclo[3.2.1]octane]")


# --- Group A #32: isotope front-descriptor -> indicated-H locant hyphen --------
# the Blue Book "(15N)-1*H*-indole (PIN)": the parenthetical descriptor at the front
# of a locant-initial parent is hyphen-joined (P-14.5.2, the Blue Book).
@pytest.mark.opsin_gate
def test_isotope_front_descriptor_indicated_h_hyphen():
    assert name_compound("c1ccc2[15nH]ccc2c1") == "(15N)-1H-indole"


# --- Group C #16: isotope nested descriptor -> enclosing-mark escalation -------
# the Blue Book (P-16.5.4.1.3): a descriptor spliced INSIDE a substituent's marks makes
# it compound, so the outer '()' escalates to '[ ]'.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("N[14CH2]C1(O)CCCC1", "1-[amino(14C)methyl]cyclopentan-1-ol"),
    ("c1cc(C2[14CH2]CCS2)ccn1", "4-[(3-14C)thiolan-2-yl]pyridine"),
])
def test_isotope_nested_mark_escalation(gf_namer, smi, expected):
    assert gf_namer.name(smi) == expected


# --- Group B #18: imidohydrazide suffix locant omission (P-66.4.2.1) -----------
# The imidohydrazide characteristic C is always chain-terminal (like hydrazide /
# amidine), so the ring appended-carbon and chain-di suffix locants elide
# (P-14.3.4). The sibling MONO forms already elided and must stay elided.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("N=C(NN)C1CCCCC1", "cyclohexanecarboximidohydrazide"),
    ("N=C(NN)C(=N)NN", "ethanediimidohydrazide"),
    # regression pins: the MONO forms keep their elision
    ("CC(=N)NN", "ethanimidohydrazide"),
    ("N=CNN", "methanimidohydrazide"),
])
def test_imidohydrazide_suffix_locant_omission(smi, expected):
    assert name_compound(smi) == expected


def test_imidohydrazide_essential_locant_kept_when_substituted():
    # P-14.3.3: a second substituent restores the essential ring locant.
    assert (name_compound("N=C(NN)C1CCC(C)CC1")
            == "4-methylcyclohexane-1-carboximidohydrazide")


# --- Group B #23: hydrazinylidene substituent locant omission (P-14.3.4) -------
# the Blue Book "(dimethylcarbamoyl)hydrazinylidene (preferred prefix)"; the Blue Book makes
# 'hydrazinylidene' the systematic prefix. N1 (free valence) is valence-full, so
# the N1/N2 locants are unambiguous and omitted.
@pytest.mark.opsin_gate
def test_hydrazinylidene_substituent_locant_omission(gf_namer):
    assert (gf_namer.name("CC(C)=NN=C1CCC(C(=O)O)CC1")
            == "4-[(propan-2-ylidene)hydrazinylidene]cyclohexane-1-carboxylic acid")
