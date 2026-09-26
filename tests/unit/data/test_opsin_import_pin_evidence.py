"""An OPSIN-import trivial name is a PIN candidate only with Blue Book evidence (fix
a performance pass, wp7; whole-branch verification panel NIT on 'lepidine', TODO 'Open from T12
fix a performance pass (wp6)' item 2).

The promotion gate cannot tell a retained PIN from a trivial name, and the import's
is_pin flag is a uniform default; 'lepidine' (4-methylquinoline, 0 Blue Book hits)
shipped at pin_verified. Measured 2026-09-26: 488 promoted OPSIN-import names had no
PIN evidence. They left the whole-molecule PIN lookup: the systematic pipeline names
the molecule (316 of them, e.g. '4-methylquinoline'), and only when it derives nothing
does the trivial name ship, labelled non-PIN (breadth kept). A producer that builds a
larger name around such a component ('xanthosine 5'-(...)') still may, and the shipped
name is labelled non-PIN.
"""
import pytest

pytestmark = pytest.mark.unit


def _tiered(smiles):
    from orthonym import Orthonym
    return Orthonym(style="pin").name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", [
    ("Cc1ccnc2ccccc12", "4-methylquinoline"),                         # was 'lepidine'
    ("CN1CCCC1c1cccnc1", "3-(1-methylpyrrolidin-2-yl)pyridine"),      # was 'nicotine'
    ("Cc1ncc(CO)c(CO)c1O", "4,5-bis(hydroxymethyl)-2-methylpyridin-3-ol"),  # 'pyridoxine'
    ("CC(C)CCCC(C)CCCC(C)CCCC(C)C", "2,6,10,14-tetramethylpentadecane"),     # 'pristane'
])
def test_systematic_pin_replaces_the_trivial_name(smiles, pin):
    from tests.support.rt_assert import name_is_rt_exact
    r = _tiered(smiles)
    assert r["name"] == pin and r["tier"] == "pin_verified", r
    assert name_is_rt_exact(pin, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,trivial", [
    ("CC(C)(c1ccc(O)cc1)c1ccc(O)cc1", "bisphenol a"),
])
def test_trivial_name_is_the_last_resort_labelled_non_pin(smiles, trivial):
    r = _tiered(smiles)
    assert r["name"] == trivial, r
    assert r["tier"] != "pin_verified" and not r["is_pin"], r


@pytest.mark.opsin_gate
def test_component_trivial_name_is_labelled_non_pin():
    smiles = ("O=c1[nH]c(=O)c2ncn([C@@H]3O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]3O)"
              "c2[nH]1")
    r = _tiered(smiles)
    assert r["name"] == "xanthosine 5'-(trihydrogen diphosphate)", r
    assert r["tier"] != "pin_verified" and not r["is_pin"], r


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("C[O-]", "methoxide"),          #:41017
    ("CC[O-]", "ethoxide"),
    ("c1ccc2nc3ccccc3nc2c1", "phenazine"),   # printed '(PIN)'
])
def test_blue_book_evidence_keeps_the_retained_pin(smiles, name):
    r = _tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r


def test_the_lookup_split_is_disjoint():
    from orthonym.data import (
        ALL_RETAINED_NAMES,
        OPSIN_UNVERIFIED_RETAINED_NAMES,
        _OPSIN_IMPORT_PIN_EVIDENCE,
    )
    from orthonym.data.retained_names import RETAINED_NAMES as hand_curated
    for smi, name in OPSIN_UNVERIFIED_RETAINED_NAMES.items():
        assert name.lower() not in _OPSIN_IMPORT_PIN_EVIDENCE
        if smi not in hand_curated:
            assert smi not in ALL_RETAINED_NAMES, (smi, name)
