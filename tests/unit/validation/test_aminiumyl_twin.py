"""The twin of the '-aminiumyl' PIN prefix of a nitrogen with three methyl groups (``validation.twin_spellings``,
rule ``aminiumyl_azaniumyl``).

 "Cationic substituent groups" (the Blue Book), method (1):42296 "all prefix names are formed by adding the
suffixes 'yl', 'ylidene', etc. to the cation name",:42299 "Method (1) leads to preferred IUPAC names"; the cation name is
'N,N,N-trimethylmethanaminium (PIN)' (:41438); (:42473) '(N,N-dimethylmethanaminiumyl)acetate (PIN)
(trimethylammoniumyl)acetate'. OPSIN reads no '-aminiumyl' prefix and reads the azane-based spelling, so the PIN spelling is
verified through that twin (the maintainers' ruling of 2026-11-01).
"""
import pytest

from orthonym.metrics.provenance import wider_tier_ctx
from orthonym.validation.twin_spellings import twin_of, twin_round_trip

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

BETAINE = "C[N+](C)(C)CC(=O)[O-]"


def test_the_twin_of_the_pin_spelling_is_the_azaniumyl_spelling():
    assert twin_of("(N,N-dimethylmethanaminiumyl)acetate") == (
        "(trimethylazaniumyl)acetate", ("aminiumyl_azaniumyl",))
    assert twin_of("2-(N,N-dimethylmethanaminiumyl)ethyl ethyl phosphate")[0] == \
        "2-(trimethylazaniumyl)ethyl ethyl phosphate"


@pytest.mark.parametrize("name", ["(trimethylazaniumyl)acetate", "[ethyldi(methyl)azaniumyl]acetate",
                                  "(N,N-diethylmethanaminiumyl)acetate", "acetate"])
def test_no_other_spelling_has_a_twin(name):
    assert twin_of(name) is None


def test_the_twin_reads_back_to_the_full_key_of_the_betaine():
    assert twin_round_trip("(N,N-dimethylmethanaminiumyl)acetate", BETAINE) == "aminiumyl_azaniumyl"


def test_the_twin_of_another_structure_does_not_verify_the_name():
    # the betaine homologue is another molecule: the twin reads to another key
    assert twin_round_trip("(N,N-dimethylmethanaminiumyl)acetate", "C[N+](C)(C)CCC(=O)[O-]") is None


def test_a_wider_tier_spells_the_azaniumyl_form_and_the_default_tier_the_pin():
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    assert Orthonym(style="pin").name(BETAINE) == "(N,N-dimethylmethanaminiumyl)acetate"
    assert Orthonym(style="pin", **_emit_tier_flags("best-effort")).name(BETAINE) == "(trimethylazaniumyl)acetate"
    assert not wider_tier_ctx.get()
