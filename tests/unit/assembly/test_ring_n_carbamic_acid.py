""" composition lever: a carboxylic (carbamic) acid on a RING nitrogen names as
the ring '-carboxylic acid' suffix at the N locant (piperazine-1-carboxylic acid,
morpholine-4-carboxylic acid). Best-effort tier; RT-exact.

Root cause (measured 2026-08-10): the classifier labels ring-N-COOH as
principal_group='carbamic_acid'; get_suffix maps that to 'carbamic acid', which is
not a ring-suffix style, so name_general_ring refused ('unsupported ring suffix for
pg=carbamic_acid') and the molecule abstained. The N stays in the cage and the
C(=O)OH is the appended '-carboxylic acid' suffix at the N's locant.
"""
import pytest
from tests.support.rt_assert import assert_rt_exact


# These require the PRODUCTION OPSIN validity gate ON (the `opsin_gate` fixture):
# without it, an atom-dropping handler ships bare 'piperazine' (drops the COOH); in
# production suppresses that and the recovery lane's general engine emits the
# correct ring-N carboxylic acid. conftest disables the gate suite-wide, so opt in.
@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles", [
    "O=C(O)N1CCNCC1",      # piperazine-4-carboxylic acid (symmetric N; RT-exact)
    "O=C(O)N1CCOCC1",      # morpholine-4-carboxylic acid
    "CN1CCN(C(=O)O)CC1",   # 4-methylpiperazine-1-carboxylic acid
    "O=C(O)N1CCCCC1",      # piperidine-1-carboxylic acid
])
def test_ring_n_carboxylic_acid(opsin_gate, smiles):
    assert_rt_exact(smiles)
