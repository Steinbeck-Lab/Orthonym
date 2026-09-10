"""W2E-P1FC Task 10 — P-59.2.1.5 (BB 25191): PG in more than one ring system
selects the senior ring.
"If the principal group occurs in more than one cyclic system, the cyclic
system chosen as parent hydride... in accordance with the criteria for
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
W2E-P1FC follow-up (see internal notes).
"""
import pytest

from orthonym.namer import name_compound
from orthonym.assembly.substituent_naming import (
    ATTACH_LOCANT_UNKNOWN, parent_to_prefix)

# OPSIN-canonical evidence SMILES for '6-(4-carboxyphenyl)-9H-fluorene-2-
# carboxylic acid' (guaranteed round-trip).
FLUORENE_DIACID = "C(=O)(O)C1=CC=C(C=C1)C=1C=C2C=3C=CC(=CC3CC2=CC1)C(=O)O"


@pytest.mark.unit
class TestP59215MultiRingPG:
    def test_ring_acid_not_flattened_to_chain(self):
        # Root fail-closed guard: a RING acid parent name must never be
        # converted to a carboxy-alkyl chain prefix (the phenyl->hexyl leak).
        assert parent_to_prefix("benzoic acid", 7, attach_locant=ATTACH_LOCANT_UNKNOWN) == ""
        # A genuine systematic CHAIN acid is still RECOGNISED as a chain (it
        # does not hit the ring-acid '' guard). Task A: it no longer emits
        # '3-carboxypropyl', because that locant came from a carbon COUNT
        # rather than a proof of the chain; the whole-molecule name is
        # unaffected (OC(=O)c1ccc(CCC(=O)O)cc1 ->
        # '4-(2-carboxyethyl)benzoic acid'). The one-position form, which
        # needs no locant at all (P-14.3.4.6), still converts.
        assert parent_to_prefix("butanoic acid", 4, attach_locant=ATTACH_LOCANT_UNKNOWN) is None
        assert parent_to_prefix("acetic acid", 2, attach_locant=ATTACH_LOCANT_UNKNOWN) == "acetyl"

    def test_no_carboxyhexyl_flattening_leak(self):
        # The phenyl->hexyl flattened name must NEVER be emitted (the historical
        # structure-loss corruption). In production the whole molecule fails
        # closed via SELF-01; regardless of the SELF-01/OPSIN environment, the
        # WRONG '6-carboxyhexyl' spelling must never appear in the output.
        # (Under a no-OPSIN test env SELF-01 fails OPEN and a substituent-dropped
        # 'fluorene-2-carboxylic acid' may surface — still never the hexyl leak.)
        assert "carboxyhexyl" not in name_compound(FLUORENE_DIACID)

    def test_fluorene_diacid_target(self):
        # P-59.2.1.5 pendant (4-carboxyphenyl)-as-substituent rendering. HEALED by
        # the W2E-D7 carboxy-prefix root fix: _ring_atom_simple_substituents now
        # recognizes -COOH, so decorated_ring_substituent_name names the pendant
        # benzene bearing a carboxylic acid as '4-carboxyphenyl' (P-65.1.7.2.1).
        # OPSIN-RT verified.
        assert name_compound(FLUORENE_DIACID) == \
            "6-(4-carboxyphenyl)-9H-fluorene-2-carboxylic acid"
