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
