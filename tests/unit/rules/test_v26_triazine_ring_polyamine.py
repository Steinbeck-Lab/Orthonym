""" companion — ring-parent poly-amine with per-nitrogen N-locants.

A ring parent bearing >=2 exocyclic -NHR amines (as the principal group) is a
'ring-x,y-diamine' whose N-substituents must carry each amine nitrogen's RING
locant as an italic-N superscript: N2-methyl-N4-methyl-1,3,5-triazine-2,4-diamine.

name_substituted_heterocycle merged every amine's N-substituents under one
'amine' key and emitted a single bare 'N-...' prefix -> the triazine polyamines
collapsed to 'N-methyl-...triamine' (wrong / OPSIN-unparseable -> unknown). Fixed
by tracking N-substituents per ring locant (amine_n_by_locant) and emitting
N{locant}- prefixes when the ring carries >1 amine. Mono-amine (bare 'N-'/'N,N-')
and carboxamide paths are unchanged.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.parametrize("smi,pin", [
    # identical N-substituents are one multiplied prefix with every N locant
    # (b), the Blue Book; Polyamines,:26369
    # 'N3-ethyl-N1,N'3-dimethylhexane-1,3,3,6-tetramine (PIN)')
    ("CNc1nc(NC)nc(NC)n1",
     "N2,N4,N6-trimethyl-1,3,5-triazine-2,4,6-triamine"),
    ("CNc1nc(NC)ncn1", "N2,N4-dimethyl-1,3,5-triazine-2,4-diamine"),
    # one alphanumerical series, the Blue Book; 'tert-butyl' under
    # 'b', '4-butyl-4-tert-butylcyclohexan-1-ol (PIN)':3465)
    ("CNc1nc(NC2CC2)ncn1", "N4-cyclopropyl-N2-methyl-1,3,5-triazine-2,4-diamine"),
    ("CNc1nc(NC(C)(C)C)ncn1", "N4-tert-butyl-N2-methyl-1,3,5-triazine-2,4-diamine"),
])
def test_ring_polyamine_n_locants(smi, pin):
    assert name_compound(smi) == pin


@pytest.mark.parametrize("smi,pin", [
    # mono-amine: bare N- / N,N- (must stay unchanged)
    ("CNc1ccccn1", "N-methylpyridin-2-amine"),
    ("CN(C)c1ccncc1", "N,N-dimethylpyridin-4-amine"),
    ("Nc1ncncn1", "1,3,5-triazin-2-amine"),
    # bare (unsubstituted) ring diamine unchanged
    ("Nc1nc(N)ncn1", "1,3,5-triazine-2,4-diamine"),
])
def test_mono_and_bare_amine_regression(smi, pin):
    assert name_compound(smi) == pin
