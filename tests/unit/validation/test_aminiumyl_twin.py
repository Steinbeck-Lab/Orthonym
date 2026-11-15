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
from tests.support.opsin_marks import needs_aminiumyl_opsin, release_aminiumyl_only

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


@release_aminiumyl_only
def test_a_wider_tier_spells_the_azaniumyl_form_and_the_default_tier_the_pin():
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    assert Orthonym(style="pin").name(BETAINE) == "(N,N-dimethylmethanaminiumyl)acetate"
    assert Orthonym(style="pin", **_emit_tier_flags("best-effort")).name(BETAINE) == "(trimethylazaniumyl)acetate"
    assert not wider_tier_ctx.get()


@needs_aminiumyl_opsin
@pytest.mark.parametrize("smiles, name, tier", [
    (BETAINE, "(N,N-dimethylmethanaminiumyl)acetate", "pin_verified"),
    ("C[N+](C)(C)CCCC(=O)[O-]", "4-(N,N-dimethylmethanaminiumyl)butanoate", "pin_verified"),
    ("C[N+](C)(C)CCOP(=O)([O-])OCC(O)CO", "2,3-dihydroxypropyl 2-(N,N-dimethylmethanaminiumyl)ethyl phosphate",
     "pin_verified"),
    ("C[N+](C)(C)[C-](C)C", "2-(N,N-dimethylmethanaminiumyl)propan-2-ide", "pin_verified"),        #:42517
    ("C[N+](C)(C)CCOS(=O)(=O)[O-]", "2-(N,N-dimethylmethanaminiumyl)ethyl sulfate", "pin_verified"),
    ("C[N+](C)(C)c1ccccc1C(=O)[O-]", "2-(N,N-dimethylmethanaminiumyl)benzoate", "pin_verified"),
])
def test_an_opsin_that_reads_aminiumyl_gives_every_tier_the_pin(smiles, name, tier):
    # dan2097/opsin#355: OPSIN's main branch reads the '-aminiumyl' prefix, so the wider tiers spell the PIN too
    from orthonym.cli import _tiered_namer
    for t in ("pin", "best-effort"):
        row = _tiered_namer("pin", t).name_tiered(smiles)
        assert (row["name"], row["tier"]) == (name, tier)
    assert not wider_tier_ctx.get()
