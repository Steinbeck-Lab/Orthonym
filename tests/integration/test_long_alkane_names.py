"""Unbranched alkanes from C80 up are named, verified, at the PIN tier.

TRIAGE 'Suite fix -- j3-long-alkanes' (g1 C1): the candidate pool's
RATIO_REJECT_FLOOR (len(name) / heavy atoms / 1.5 < 0.10) threw away the only
candidate for every unbranched alkane from C80 ('octacontane', ratio 0.092) and the
engine abstained. Numerical-term names are short by design:
(the Blue Book) names the unbranched alkanes by the numerical terms of Table
1.4 (:2794) + 'ane'. general_acyclic now passes its measured complete atom
partition, which replaces the character-count floor; the OPSIN gate still verifies.

Beyond ~1000 heavy atoms RDKit gives no standard InChIKey, so the gate's
comparison was 'inconclusive' and the name shipped unverified under the
pin_verified label; equal canonical SMILES now verify it (namer.
_self_consistency_verdict).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.data.chain_names import get_chain_name

pytestmark = pytest.mark.opsin_gate


def _independent_rt_exact(name: str, smiles: str) -> bool:
    """OPSIN parse of ``name`` outside the engine, compared on the full InChIKey,
    or on canonical SMILES where RDKit gives no InChIKey (> ~1000 heavy atoms)."""
    from tests.support.jars import jar_or_skip
    from tests.support.rt_assert import _independent_parse
    jar_or_skip()
    parsed = _independent_parse(name)
    if not parsed:
        return False
    a, b = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(parsed)
    ka, kb = inchi.MolToInchiKey(a), inchi.MolToInchiKey(b)
    if ka and kb:
        return ka == kb
    return Chem.MolToSmiles(a) == Chem.MolToSmiles(b)


@pytest.mark.parametrize("n", [79, 80, 92, 100, 101, 120, 145, 341, 500, 1100])
def test_unbranched_alkane_is_named_and_verified_at_the_pin_tier(n):
    smiles = "C" * n
    res = Orthonym(style="pin").name_tiered(smiles)
    assert res["name"] == get_chain_name(n), res
    assert res["tier"] == "pin_verified", res
    assert res["gate_outcome"] == "self_consistency_verified", res
    assert _independent_rt_exact(res["name"], smiles)


def test_canonical_smiles_verify_beyond_the_inchi_limit():
    from orthonym.namer import _self_consistency_verdict
    big = "C" * 1100
    assert not inchi.MolToInchiKey(Chem.MolFromSmiles(big))  # the premise
    assert _self_consistency_verdict(big, big) == "ok"
    # A different molecule is not verified; the pre-existing fail-open policy for an
    # uncomputable key is unchanged ("inconclusive").
    assert _self_consistency_verdict(big, "C" * 1101) == "inconclusive"
