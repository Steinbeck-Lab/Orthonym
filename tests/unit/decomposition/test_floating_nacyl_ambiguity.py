"""v30 (task 24) — the decomposition engine must NOT float a bare `N-acyl-` prefix onto an amine
parent that has >=2 acylatable nitrogens, because the bare `N-` does not say WHICH nitrogen bears
the acyl -> the name is AMBIGUOUS (denotes >=2 molecules) and, with the OPSIN jar absent, ships
unverified (latent wrong-molecule on the DEFAULT PIN path).

acylspy proof: `OC(=O)c1cc(CN(C)C(C)=O)c(CNC)o1` (ORIG) and its 4/5-mirror produce the IDENTICAL
name `N-acetyl-4,5-bis(methylamino)methylfuran-2-carboxylic acid`; OPSIN resolves it to ORIG, so
ORIG shipped (SELF-01 ok) and the mirror abstained. The fix refuses to float when the amine
fragment has >=2 acylatable N (N with >=1 H), so ORIG fails closed to an honest abstention rather
than shipping an ambiguous T1 name. Single-acylatable-N amides are unaffected.

The gate is disabled suite-wide; this behaviour is about which candidate the engine BUILDS (the
float vs abstain decision at the producer), independent of the OPSIN validity gate, so no
opsin_gate marker is needed.
"""
from orthonym import Orthonym
from orthonym.errors import is_failure_name


def _name(smi):
    return Orthonym(style="pin").name(smi)


def test_ambiguous_diamine_acyl_does_not_float():
    # 2 acylatable N in the amine part -> the bare N-acetyl- float is ambiguous -> must NOT ship it.
    nm = _name("OC(=O)c1cc(CN(C)C(C)=O)c(CNC)o1")
    assert is_failure_name(nm) or "N-acetyl" not in nm, nm


def test_single_acylatable_n_amide_unchanged():
    # exactly one acylatable N -> unambiguous -> keeps naming.
    assert _name("CC(=O)NCCC(=O)O") == "3-acetamidopropanoic acid"


def test_simple_amide_unchanged():
    # N-methylacetamide-style: one N, unambiguous.
    nm = _name("CC(=O)NC")
    assert not is_failure_name(nm), nm


def test_heteroaromatic_ring_n_acyl_float_is_ambiguous():
    """v30 #29-fable-BLOCKER: a heteroaromatic amine parent (thiazole) with a ring N
    PLUS an exocyclic amino makes a bare N-<acyl> float ambiguous — OPSIN dearomatizes
    the ring N to host the acyl. The H-only acylatable count missed the 0-H aromatic
    ring N; #29's route (which now names the 2-(methylamino) leftover) then unmasked a
    wrong-constitution float that shipped at gate-off. Must fail closed (abstain), never
    ship `N-<acyl>-2-(methylamino)-1,3-thiazole-5-carboxylic acid` (acyl on the ring N)."""
    for smi in ("OC(=O)c1cnc(N(C)C=O)s1", "OC(=O)c1cnc(N(C)C(C)=O)s1"):
        nm = _name(smi)
        assert is_failure_name(nm) or "2-(methylamino)" not in nm, nm


def test_amine_acyl_ambiguous_counts_aromatic_ring_n():
    from orthonym.decomposition.engine import _amine_acyl_ambiguous
    # thiazole ring N (0 H, aromatic) + exocyclic methylamino N (1 H) -> ambiguous
    assert _amine_acyl_ambiguous("OC(=O)c1cnc(NC)s1") is True
    # plain glycine: single amino N, no aromatic ring N -> unambiguous
    assert _amine_acyl_ambiguous("OC(=O)CN") is False
