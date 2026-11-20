"""A binary name ('carbon dioxide') in a charged adduct with a hydrogen word is not read part by part.

OPSIN reads the words of a binary name as ions and balances them with the rest of a charged mixture: in an OPSIN that
reads 'dihydrogen', 'carbon dioxide—iodide—dihydrogen (1/1/1)' is a charged carbon (35 PubChem rows read back as
another structure with OPSIN main). The hydrogen-word reader (validation.hydrogen_word_read_back) now refuses such a
reading, so the adduct namer takes the spelling OPSIN keeps neutral ('dioxomethane', as its own route already writes
'dioxomethane—iodide (1/1)'). "Organic adducts" (the Blue Book): names of the components joined by em
dashes.
"""
import pytest

from orthonym.validation.hydrogen_word_read_back import read_back_with_hydrogen_words
from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def test_binary_name_in_a_charged_hydrogen_adduct_is_refused():
    assert read_back_with_hydrogen_words("carbon dioxide—iodide—dihydrogen (1/1/1)") is None


def test_binary_name_in_a_neutral_hydrogen_adduct_still_reads():
    assert read_back_with_hydrogen_words("carbon dioxide—dihydrogen (1/1)") is not None


def test_the_charged_mixture_takes_the_neutral_spelling():
    row = tier_row("[HH].C(=O)=O.[I-]", "best-effort")
    assert row["name"] == "dioxomethane—iodide—dihydrogen (1/1/1)", row


def test_an_acyl_halide_is_not_a_binary_name():
    row = tier_row("[HH].CC(Cl)=O.[I-]", "best-effort")
    assert row["name"] == "acetyl chloride—iodide—dihydrogen (1/1/1)", row
