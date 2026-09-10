""" RISK 5 Class 2 — the general-engine substituent path must name a substituted amine
(-N(R)R') as a composed (dialkylamino) prefix, not the malformed `amino-N,N-diethyl...` token.

Root cause: `_name_polyfunctional_acyclic_substituent` (assembly/substituent_naming.py) declined a
substituted amine, so the fragment fell to `parent_to_prefix` string-surgery which prepended
`amino` onto a stem still carrying `N,N-diethyl` italic locants -> `amino-N,N-diethylethyl`
(OPSIN-unparseable). The PIN path already names these correctly via `_assemble_amino_prefix_core`;
this test pins the general-engine (best-effort) path to the same composer.

Targets verified by OPSIN round-trip:
  CNCC1=CC=CO1 -> 2-[(methylamino)methyl]furan
  OC(=O)c1ccc(CCN(CC)CC)cc1 ->...[2-(diethylamino)ethyl]... (benzenecarboxylic-acid form ok)
"""
from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags


def _best_effort_name(smi):
    eng = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    r = eng.name_tiered(smi)
    return r["name"] if isinstance(r, dict) else r


def test_no_malformed_amino_locant_token():
    # the malformed `amino-N,N-diethyl...` / `amino-N-methyl...` token must never appear.
    for smi in ("OC(=O)c1ccc(CCN(CC)CC)cc1", "CNCC1=CC=CO1", "CCN(CC)CCN1C(=O)CN=C(c2ccccc2F)c2cc(Cl)ccc21"):
        nm = _best_effort_name(smi)
        if nm:
            assert "amino-N," not in nm and "amino-N-" not in nm, (smi, nm)


def test_diethylaminoethyl_prefix_is_composed():
    nm = _best_effort_name("OC(=O)c1ccc(CCN(CC)CC)cc1")
    assert nm is not None
    assert "(diethylamino)ethyl" in nm, nm


def test_methylaminomethyl_furan_no_longer_abstains():
    nm = _best_effort_name("CNCC1=CC=CO1")
    assert nm is not None, "should emit a name, not abstain"
    assert "(methylamino)methyl" in nm, nm
