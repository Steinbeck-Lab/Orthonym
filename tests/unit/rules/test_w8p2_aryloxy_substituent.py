""" W8 P2 Task 2.1 — heteroaryl / fused-aryl aryloxy substituent namer.

Before this fix, an ether O attached to a BARE heteroaryl or fused-aryl ring made
`get_alkoxy_prefix` return None (it only handled bare benzene -> phenoxy and
*decorated* aryloxy), and the polyfunctional caller then SILENTLY DROPPED the whole
substituent: `OC(=O)COc1ccccn1` -> gate-off `ethanoic acid` (a different molecule;
the OPSIN self-consistency gate hid it only WITH Java). Now the bare aromatic ring
SYSTEM is named as `<ring>-yloxy` with an atom-drop veto. /.

Every SMILES is OPSIN-authoritative and every expected name RT-verified 2026-07-17.
Tests run gate-off (RAW) to exercise the source path, not the OPSIN gate.
"""
import pytest
from orthonym.namer import Orthonym

pytestmark = pytest.mark.unit

RAW = Orthonym(_disable_opsin_validity_gate=True)  # gate-off: proves the raw namer


# No `2-` locant: acetic acid has ONE substitutable position, so the substituent
# locant is omitted in the PIN; cf. `(1H-indol-1-yl)acetic acid (PIN)`
# the Blue Book). The locant-bearing heteroaryl/fused '-yl' keeps its OWN enclosing
# marks, the Blue Book '(pyridin-2-yl)oxy (preferred prefix)'; the Blue Book
# '(naphthalen-2-yl)oxy'), so the citation escalates to `[(...)oxy]acetic acid`
# (F-spell-oxy 2026-08-08: was the non-PIN unenclosed `(pyridin-2-yloxy)…`).
@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)COc1ccccn1",       "[(pyridin-2-yl)oxy]acetic acid"),    # bare heteroaryl-oxy
    ("OC(=O)COc1cccc2ccccc12", "[(naphthalen-1-yl)oxy]acetic acid"), # bare fused-aryl-oxy
    ("OC(=O)COc1ccc2ccccc2c1", "[(naphthalen-2-yl)oxy]acetic acid"),
])
def test_bare_aryloxy_named_not_dropped(smiles, expected):
    assert RAW.name(smiles) == expected


def test_bare_phenoxy_unregressed():
    """Bare benzene ether still contracts to the retained phenoxy (simple
    substituent, no enclosing marks)."""
    assert RAW.name("OC(=O)COc1ccccc1") == "phenoxyacetic acid"


def test_decorated_aryloxy_unregressed():
    """A decorated aryloxy keeps its ring substituents (the earlier `dec` branch);
    the new bare-ring branch must not interfere."""
    assert RAW.name("OC(=O)COc1ccc(Cl)cc1") == "(4-chlorophenoxy)acetic acid"


# CARBOCYCLIC 3+-ring PAH substituents (anthracene / phenanthrene / fluorene) are
# mis-numbered by get_ring_substituent_name (wrong locant = a different molecule) —
# a broad PRE-EXISTING shared-namer bug this aryloxy path newly reaches. The SHIPPED
# (gated) namer stays SAFE for them: the full-coverage wrong-locant name is caught by
# and abstains to 'unknown' (verified out-of-band: this suite runs gate-OFF via
# conftest, so a gated assertion is not expressible here). Do NOT add a source-side veto
# in get_alkoxy_prefix that returns None for these: that reroutes them to the ester-family
# handler whose DROP escapes the gate (ships 'ethanoic acid' even gated) — strictly worse
# than the gated-safe wrong-locant. The get_ring_substituent_name PAH numbering bug and
# the ester-family gated-escape drop are FLAGGED for a dedicated shared-namer / P10
# structure-conservation fix, not band-aided in this Task-2.1 aryloxy namer.
