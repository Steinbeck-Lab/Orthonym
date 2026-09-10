""" — β-carotene "not a verified PIN" honest-label fix (Both/Step-1, Fix B).

Root cause (verified, internal notes):
name_tiered stamped `pin_verified` on a name the STRICT PIN path does NOT itself
produce -- β-carotene names only because a breadth flag enabled a best-effort
producer (the allow_mancude ring-substituent path renders the polyene; the strict
path abstains). The name round-trips, but its PIN PREFERENCE is uncertified.

Fix: name_tiered demotes such a name to `pin_unverified` (is_pin=False) using a
strict TWIN engine (identical construction minus the breadth flags). The
discriminator is the whole strict pipeline, because the enabling gate is read at
many sites through the substituent recursion -- no single producer frame captures
it (the review's universal-namer / allow_mancude sites were off-path or
non-discriminating for this molecule).

All tests run with the OPSIN validity gate DISABLED (no JVM), so they are
deterministic and safe to run alongside a large OPSIN benchmark; the twin inherits
that gate setting.
"""
import pytest

from orthonym import Orthonym

CAROTENE = ("CC1CCC/C(C)=C1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)"
            "/C=C/C2=C(C)/CCCC2(C)C")


def _all_flags():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True, _disable_opsin_validity_gate=True)


@pytest.mark.unit
def test_carotene_all_flags_is_not_pin_verified():
    r = _all_flags().name_tiered(CAROTENE)
    # the NAME is correct and unchanged (the polyene is rendered in full)
    assert r.get("name") and "octadeca" in r["name"] and "unknown" not in r["name"]
    # but its PIN status is uncertified (best-effort component + unverified
    # ring-parent choice) -> never pin_verified.
    assert r.get("tier") == "pin_unverified", r.get("tier")
    assert r.get("is_pin") is False


@pytest.mark.unit
def test_real_pins_still_pin_verified_under_all_flags():
    # genuine strict-path PINs must be untouched: the twin produces the identical
    # name, so no demotion. Covers plain, ring, fused-ring and ring-substituted PINs.
    be = _all_flags()
    for smi, expect in [
        ("OC(=O)c1ccccc1", "benzoic acid"),
        ("Cc1ccccc1", "toluene"),
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("c1ccc(-c2ccc3ccccc3c2)cc1", "2-phenylnaphthalene"),
        ("OC(=O)c1c2ccccc2cc2ccccc12", "anthracene-9-carboxylic acid"),
        ("C1CCCCC1c1ccccc1", "cyclohexylbenzene"),
    ]:
        r = be.name_tiered(smi)
        assert r.get("name") == expect, (smi, r.get("name"))
        assert r.get("tier") == "pin_verified", (smi, r.get("tier"))
        assert r.get("is_pin") is True


@pytest.mark.unit
def test_default_config_byte_identical_no_twin():
    # the demotion path NEVER runs on the default/PIN engine (no breadth flag) ->
    # byte-identical, and no twin is constructed.
    eng = Orthonym(_disable_opsin_validity_gate=True)
    r = eng.name_tiered("CCO")
    assert r.get("name") == "ethanol"
    assert r.get("tier") == "pin_verified"
    assert r.get("is_pin") is True
    assert getattr(eng, "_pin_twin", None) is None
