"""v26 BP-2 (L3): malformed substituent-string bugs.

RC-1 (accuracy, whole-class): parent_to_prefix must FAIL CLOSED (return None) on a
functional-parent residue it cannot express as a '+yl' prefix, instead of
fabricating 'isothiocyanic acidyl' / 'ethyl formatyl' / 'prop-2-enalyl'. BB
P-65.1.1 (acid-as-substituent is 'carboxy', never '«acid»yl') + P-66 note (p).

RC-2a (coverage): the terminal pseudohalides -N=C=O / -N=C=S / -N#C / -S-C#N are
always cited as prefixes (isocyanato / isothiocyanato / isocyano / thiocyanato) in
PINs — BB P-66.5.1.2 (BlueBookV2.md:1710). name_substituent_fragment now detects
them structurally and returns the authoritative prefix.
"""
import pytest

from orthonym.namer import Orthonym
from orthonym.assembly.substituent_naming import parent_to_prefix


@pytest.fixture(scope="module")
def namer_pin():
    return Orthonym(style="pin")


@pytest.mark.parametrize("smiles,expected", [
    # RC-2a pseudohalide-on-ring coverage wins (were garbage -> now correct PIN)
    ("O=C(O)c1ccccc1N=C=S", "2-isothiocyanatobenzoic acid"),
    ("O=C(O)c1ccccc1N=C=O", "2-isocyanatobenzoic acid"),
    # RC-1 bonus: declining the bad ester-substituent promotes the ester to
    # principal group -> correct carboxylate PIN.
    ("CCOC(=O)C1(c2ccccc2)CCN(C)CC1", "ethyl 1-methyl-4-phenylpiperidine-4-carboxylate"),
])
def test_bp2_coverage_wins(namer_pin, smiles, expected):
    assert namer_pin.name(smiles) == expected


@pytest.mark.parametrize("smiles", [
    "C=C(C=O)C1CCC(C)C1C=O",         # was 'prop-2-enalyl...' garbage
    "CCCC(C)(COC(N)=O)COC(=O)NC(C)C",  # was '...carbamatyl carbamate' garbage
])
def test_bp2_no_fabricated_garbage(namer_pin, smiles):
    """RC-1 contract: the fabricated OPSIN-unparseable '+yl' string never appears.

    (The molecule then either fails closed to 'unknown' or the SELF-01 OPSIN gate
    suppresses a wrong candidate; that final outcome is OPSIN-availability
    dependent and verified via diagnose.py, so it is not asserted here — the
    robust, environment-independent invariant is the absence of the garbage.)
    """
    result = namer_pin.name(smiles)
    for garbage in ("acidyl", "formatyl", "enalyl", "carbamatyl", " acidyl"):
        assert garbage not in result, (smiles, result)


def test_rc1_parent_to_prefix_fail_closed():
    """parent_to_prefix declines functional-parent residues (returns None)."""
    assert parent_to_prefix("isothiocyanic acid", chain_length=1) is None
    assert parent_to_prefix("ethyl formate", chain_length=1) is None
    assert parent_to_prefix("prop-2-enal", chain_length=3) is None
    # ...but never touches legitimate hydrocarbon / ring substituent parents:
    assert parent_to_prefix("propane", chain_length=3) == "propyl"
    assert parent_to_prefix("pyridine", chain_length=0) == "pyridinyl"
    assert parent_to_prefix("prop-1-ene", chain_length=3) == "prop-1-enyl"
