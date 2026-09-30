"""Task 8B RC3, the Blue Book): substituted URONIUM / THIOURONIUM cations.

A cation formed by adding a hydron to (iso)urea -- a carbon bonded to two N and one
O (uronium) or one S (thiouronium) -- is named on the retained parent cation
``uronium`` / ``thiouronium`` with N/N'/O/S substituent locants (numerical locants
are dropped in the PIN; the letter locants follow urea/isourea). The neutralize path
cannot reach this class (the neutral isourea names as a carbamimidate ester), so
``emit_uronium`` is a bespoke skeleton producer, RT-gated in ``route_charged``.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.namer import Orthonym, _validity_gate_name_to_smiles
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "NC(=[NH2+])O",
    "NC(=[NH2+])S",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset({
    "NC(=[NH2+])O",
    "NC(=[NH2+])S",
})


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)



def _pin_name(smiles):
    return _dt_name(smiles)


def _ik(s):
    m = Chem.MolFromSmiles(s)
    return inchi.MolToInchiKey(m) if m else None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    # the Blue Book (PIN): N,N'-dimethyl-O-phenyluronium (O-phenyl isourea, both N methyl).
    ("CNC(=[NH+]C)Oc1ccccc1", "N,N'-dimethyl-O-phenyluronium"),
    # the Blue Book (PIN): N,S-dimethyl-N'-phenylthiouronium (methyl at one N + S,
    # phenyl at the other N -> alphabetically-earliest 'methyl' takes the unprimed N).
    ("CNC(=[NH+]c1ccccc1)SC", "N,S-dimethyl-N'-phenylthiouronium"),
])
def test_substituted_uronium_rt(smiles, expected):
    """RT is checked by InChIKey, not canonical SMILES: names the
    delocalised uronium cation on a parent that represents BOTH tautomeric
    structures, so the name's localised OPSIN parse can carry a charge-shifted
    resonance form of the input (identical InChIKey, differing canonical SMILES).
    InChIKey is the headline correctness oracle and the same test the emit gate
    (`_uronium_rt_ok`) applies."""
    name = _pin_name(smiles)
    assert name == expected, f"{smiles} -> {name!r}, expected {expected!r}"
    opsin = _validity_gate_name_to_smiles(name)
    assert opsin is not None, f"OPSIN could not parse {name!r}"
    assert _ik(opsin) == _ik(smiles), (
        f"NOT rt (InChIKey): {name!r} -> {_ik(opsin)} != {_ik(smiles)}")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    # The unsubstituted retained parents must keep their existing routing -- the
    # substituted producer requires >= 1 substituent and declines these.
    ("NC(N)=[OH+]", "uronium"),
    ("NC(=[NH2+])O", "isouronium"),
    ("NC(=[NH2+])S", "isothiouronium"),
    # Guanidinium (three N) never matches the uronium skeleton (two N + one O/S).
    ("NC(N)=[NH2+]", "guanidinium"),
    ("CNC(N)=[NH2+]", "N-methylguanidinium"),
])
def test_uronium_family_unchanged(smiles, expected):
    name = _pin_name(smiles)
    assert name == expected, f"{smiles} -> {name!r}, expected {expected!r}"
