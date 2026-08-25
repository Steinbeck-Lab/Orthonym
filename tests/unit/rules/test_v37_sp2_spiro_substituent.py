"""v37 SP2.1' — spiro-VB substituent supplier: acyloxy ester must be named
(threaded as a P-65.6.3.2.3 detachable prefix), never dropped or mis-spelled.

RE-SCOPED from the refuted deep-VB-numbering task (SP2.0/rescope: 0/38 of the
spiro-VB abstainer bucket is blocked by ring numbering). The real blocker for the
two confirmed witnesses is the spiro-VB substituent supplier: a bare acyloxy
substituent (``-O-C(=O)-R`` attached via its ester O) whose acyl group is NOT a
retained acid is mis-spelled by ``name_substituent`` as an OPSIN-grammar-invalid
oxa-replacement chain (``...-2-oxo-1-oxabutyl``), so the whole complex-ring name
is suppressed and the molecule abstains — a breadth loss (0-wrong already held).

STEP-1 (invariant 8) VERIFIED on HEAD in a fresh process: the drop is at the
``_enrich_complex_ring_with_subs`` consumer of ``name_spiro_vonbaeyer``
(composer.py), NOT in the spiro-VB numbering core. ``name_substituent`` returned
``(3S)-4-chloro-3-hydroxy-3-methyl-2-oxo-1-oxabutyl`` for W1's ester (grammar-
invalid) and ``4,4-dimethyl-2-oxo-1,5-dioxapentyl`` / ``substituent`` for A's.

Fix: recognise the acyloxy shape structurally and name it via the shared acid
engine (``rules.lipids._acyloxy_for_site`` -> full ``name_compound`` on the
isolated acid) — the SAME machinery the ordinary substitutive path uses
(``substituent_naming.py`` Pass 1c / enumerator Tier 1.95). Byte-identical for a
retained acyl (``acetyloxy`` stays bare); it only REPLACES the grammar-invalid
oxa-chain for a systematic acyl. SELF-01 round-trip is the 0-wrong gate.
"""
import pytest

from orthonym import errors
from orthonym.namer import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

# Witness A (census, 3 esters: 2 acetyloxy + 1 systematic 3-hydroxy-3-methyl-
# butanoyloxy) and W1 (SP2.0's confirmed tricyclo+ witness, 1 systematic
# 3-chloro-2-hydroxy-2-methylpropanoyloxy ester). Both are spiro-VB cores.
A = "CC(=O)OCC12CC(OC(=O)CC(C)(C)O)C(C)=CC1OC1C(O)C(OC(C)=O)C2(C)C12CO2"
W1 = "C=C1C(=O)O[C@H]2[C@H]1[C@@H](OC(=O)[C@](C)(O)CCl)CC(=C)[C@@H]1C[C@H](O)[C@@]3(CO3)[C@H]21"
WITNESSES = [A, W1]


def _rt_full(smi: str, name) -> bool:
    """Full-InChIKey round-trip: name -> OPSIN -> InChI == input's InChI."""
    if not name or errors.is_failure_name(name):
        return False
    r = opsin_roundtrip_check(smi, name)
    return bool(r.get("passed") and r.get("inchi_match"))


def _best_effort() -> Orthonym:
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


# --- 0-wrong ABSOLUTE: never a dropped-atom / wrong-molecule name -------------
class TestNeverWrong:
    @pytest.mark.opsin_gate
    @pytest.mark.parametrize("smi", WITNESSES)
    def test_spiro_vb_acyloxy_not_dropped(self, smi):
        """Best-effort either abstains or names the RIGHT molecule (RT-exact).
        Holds in BOTH the pre-fix (abstain) and post-fix (threaded) states."""
        name = _best_effort().name(smi)
        assert name is None or errors.is_failure_name(name) or _rt_full(smi, name), (
            f"0-wrong violation: {smi} -> {name!r} does not round-trip")


# --- THE DELIVERABLE: the acyloxy ester is threaded, not dropped -------------
class TestAcyloxyThreaded:
    @pytest.mark.opsin_gate
    @pytest.mark.parametrize("smi", WITNESSES)
    def test_spiro_vb_acyloxy_names_and_round_trips(self, smi):
        name = _best_effort().name(smi)
        assert name and not errors.is_failure_name(name), (
            f"expected a threaded acyloxy name for {smi}, got {name!r}")
        assert _rt_full(smi, name), f"threaded name does not RT: {name!r}"

    @pytest.mark.opsin_gate
    def test_witness_names_carry_the_acyl_prefix(self):
        """The systematic acyl is spelled as an acyloxy prefix, never an
        oxa-replacement chain (the pre-fix mis-name)."""
        n_a = _best_effort().name(A)
        assert n_a and "3-hydroxy-3-methylbutanoyloxy" in n_a, n_a
        assert "oxabut" not in n_a and "oxapent" not in n_a, n_a
        n_w1 = _best_effort().name(W1)
        assert n_w1 and "3-chloro-2-hydroxy-2-methylpropanoyloxy" in n_w1, n_w1
        assert "oxabut" not in n_w1, n_w1
