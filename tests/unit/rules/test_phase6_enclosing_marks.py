""" a phase (B): enclosing-mark escalation (-> [ -> { at 3 render sites.
All are LIVE, RT-OK-today spelling defects — the fix must keep RT and
never wrap a simple prefix or mangle a fusion descriptor."""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi
from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.fixture(scope="module")
def be():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


def _rt(smi, name):
    if not name or "unknown" in name:
        return False
    o = opsin_parse(name)
    return bool(o) and inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(o))


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # The two biphenyl rows are the ring assembly now: (the Blue Book:
    # 15560), '(4'-cyano[1,1'-biphenyl]-4-yl)oxy... (PIN)' (:7455); the square
    # brackets of '[1,1'-biphenyl]' are part of the parent name and do not count
    # in the nesting order,:7446), so one '(' and one '[' remain. The
    # old '{[4-(4-chlorophenyl)phenyl]methyl}' was the phenyl-on-phenyl spelling
    # (TRIAGE g5 C14, suite fix j5). OPSIN 2.9.0 full-InChIKey EXACT.
    ("CC(=O)NCc1ccc(-c2ccc(Cl)cc2)cc1",
     "N-[(4'-chloro[1,1'-biphenyl]-4-yl)methyl]acetamide"),
    ("CC(=O)N(Cc1ccc(-c2ccc(Cl)cc2)cc1)C",
     "N-[(4'-chloro[1,1'-biphenyl]-4-yl)methyl]-N-methylacetamide"),
    # Keeps the three-level (-> [ -> { escalation covered (OPSIN exact).
    ("CC(=O)NCc1ccc(Oc2ccc(Cl)cc2)cc1", "N-{[4-(4-chlorophenoxy)phenyl]methyl}acetamide"),
])
def test_n_substituent_brace_escalation(namer, smi, expected):
    n = namer.name(smi)
    assert "[[" not in (n or "") and "]]" not in (n or "") and "((" not in (n or ""), n
    assert n == expected, n
    assert _rt(smi, n), n


@pytest.mark.opsin_gate
def test_simple_n_substituent_stays_bare(namer):
    # a bare simple N-substituent must NOT be wrapped
    assert namer.name("CC(=O)NC") == "N-methylacetamide"
    assert namer.name("CC(=O)NCC") == "N-ethylacetamide"


@pytest.mark.opsin_gate
def test_polyfunctional_acyloxy_no_double_paren(be):
    # a diacylglycerol whose acyloxy arms carry an inner (9Z) stereo mark:
    # old raw f"({acyloxy})" gave `((9Z)-...enoyloxy)`; the fix escalates to `[...]`.
    smi = "CCCCC/C=C\\CCCCCCCC(=O)OC[C@H](CO)OC(=O)CCCCCCC/C=C\\CCCCCCCC"
    n = be.name(smi)
    assert n and "unknown" not in n, n
    assert "((" not in n and "[[" not in n, n
    assert _rt(smi, n), n
