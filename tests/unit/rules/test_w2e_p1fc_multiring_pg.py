"""W2E-P1FC Task 10 — P-59.2.1.5 (BB 25191): PG in more than one ring system
selects the senior ring.
"If the principal group occurs in more than one cyclic system, the cyclic
system chosen as parent hydride ... in accordance with the criteria for
choosing a senior ring or ring system" -> 6-(4-carboxyphenyl)-9H-fluorene-
2-carboxylic acid (PIN).

STATUS (2026-07-09): PARTIAL + xfail. Parent selection is CORRECT (fluorene is
chosen). The gap is naming the pendant 4-carboxyphenyl ring AS a substituent:
the substituent machinery flattens a benzene-ring-bearing-COOH to a
'6-carboxyhexyl' chain (the phenyl->hexyl corruption). This pass SHIPS the
fail-closed root guard in parent_to_prefix (a ring acid name like 'benzoic acid'
no longer converts to a carboxy-alkyl chain), so the wrong '6-carboxyhexyl'
candidate is never built and the molecule fails closed (never a wrong name).
The full (4-carboxyphenyl)-as-substituent rendering needs a substituted-aromatic
ring-substituent namer (carboxy suffix-substituent) with determinism risk —
W2E-P1FC follow-up (see ).
"""
import pytest

from orthonym.namer import name_compound
from orthonym.assembly.substituent_naming import parent_to_prefix

# OPSIN-canonical evidence SMILES for '6-(4-carboxyphenyl)-9H-fluorene-2-
# carboxylic acid' (guaranteed round-trip).
FLUORENE_DIACID = "C(=O)(O)C1=CC=C(C=C1)C=1C=C2C=3C=CC(=CC3CC2=CC1)C(=O)O"


@pytest.mark.unit
class TestP59215MultiRingPG:
    def test_ring_acid_not_flattened_to_chain(self):
        # Root fail-closed guard: a RING acid parent name must never be
        # converted to a carboxy-alkyl chain prefix (the phenyl->hexyl leak).
        assert parent_to_prefix("benzoic acid", 7) == ""
        # A genuine systematic CHAIN acid still converts.
        assert parent_to_prefix("butanoic acid", 4) == "3-carboxypropyl"

    def test_no_carboxyhexyl_flattening_leak(self):
        # The phenyl->hexyl flattened name must NEVER be emitted (the historical
        # structure-loss corruption). In production the whole molecule fails
        # closed via SELF-01; regardless of the SELF-01/OPSIN environment, the
        # WRONG '6-carboxyhexyl' spelling must never appear in the output.
        # (Under a no-OPSIN test env SELF-01 fails OPEN and a substituent-dropped
        # 'fluorene-2-carboxylic acid' may surface — still never the hexyl leak.)
        assert "carboxyhexyl" not in name_compound(FLUORENE_DIACID)

    @pytest.mark.xfail(
        reason="P-59.2.1.5 pendant (4-carboxyphenyl)-as-substituent rendering "
        "needs a substituted-aromatic ring-substituent namer (carboxy "
        "suffix-substituent) — W2E-P1FC follow-up",
        strict=True,
    )
    def test_fluorene_diacid_target(self):
        assert name_compound(FLUORENE_DIACID) == \
            "6-(4-carboxyphenyl)-9H-fluorene-2-carboxylic acid"
