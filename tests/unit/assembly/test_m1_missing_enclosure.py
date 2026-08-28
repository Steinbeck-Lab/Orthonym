import pytest
from orthonym import Orthonym

@pytest.fixture(scope="module")
def eng():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)

# W1 — bis() around a complex ring substituent (composer.py count>1 branch)
def test_w1_bis_3_chlorophenyl_enclosed(eng):
    smi = "C[C@@H]1C(=O)N[C@H](c2cccc(Cl)c2)C2(C(=O)Nc3cc(Cl)ccc32)[C@@H]1c1cccc(Cl)c1"
    name = eng.name(smi)
    assert name is not None and "unknown" not in name.lower()
    # the multiplied complex substituent must be enclosed
    assert "bis(3-chlorophenyl)" in name
    assert "bis3-chlorophenyl" not in name

# W3 — [] escalation around a substituent that already contains () (count==1 branch)
def test_w3_methanesulfonyl_quinazolinyl_enclosed(eng):
    smi = "COc1ccc2c(c1)C1(CC1)CN(c1ncnc3ccc(S(C)(=O)=O)cc13)C2"
    name = eng.name(smi)
    assert name is not None and "unknown" not in name.lower()
    assert "[6-(methanesulfonyl)quinazolin-4-yl]" in name

# NEGATIVE — a simple ring substituent must stay bare (no over-enclosure)
def test_simple_substituent_stays_bare(eng):
    # 1,3-dimethyl on a ring: 'methyl' is simple, must NOT gain parens
    name = eng.name("Cc1cccc(C)c1")
    assert name is not None
    assert "(methyl)" not in name
    assert "methyl" in name

# W2 — complex N-substituent must be bracket-wrapped (fragment_assembly _assemble_amide)
def test_w2_n_substituent_wrapped(eng):
    smi = "O=C1C[C@@H](C(=O)N[C@H]2CCS(=O)(=O)C2)C2(CCCCC2)O1"
    name = eng.name(smi)
    assert name is not None and "unknown" not in name.lower()
    assert "[(3S)-1,1-dioxothiolan-3-yl]" in name

@pytest.mark.roundtrip
def test_w2_roundtrips_to_input(eng):
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    smi = "O=C1C[C@@H](C(=O)N[C@H]2CCS(=O)(=O)C2)C2(CCCCC2)O1"
    name = eng.name(smi)
    res = opsin_roundtrip_check(smi, name)
    assert res["passed"], res


# ---------------------------------------------------------------------------
# Task 3 (M1-PLAN) — audit of the remaining bare-concat substituent sites.
# ---------------------------------------------------------------------------

# W4 — _assemble_ring_with_ester_prefixes, count==1 NON-acyloxy branch.
# At HEAD (pre-fix, `27db95e0`/`0bab4e9c`) this molecule shipped the bare,
# BB-nonconformant '4-1-chloroethyl-1,2-bis(acetyloxy)benzene' -- OPSIN's
# tolerant parser accepted it (structure correct) but the locant-substituent
# boundary is ambiguous per P-16.3.3, and it does not survive P-14.5
# alphanumerical ordering either (the bare compound token sorted BEFORE
# 'acetyloxy' instead of after it).
def test_w4_ring_ester_count1_complex_enclosed(eng):
    smi = "CC(=O)Oc1cc(C(C)Cl)ccc1OC(C)=O"
    name = eng.name(smi)
    assert name is not None and "unknown" not in name.lower()
    assert "(1-chloroethyl)" in name
    assert "4-1-chloroethyl" not in name
    # acyloxy branch (LEAVE, composer.py :4104 byte-identity note) unchanged
    assert "bis(acetyloxy)" in name


@pytest.mark.roundtrip
def test_w4_roundtrips_to_input(eng):
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    smi = "CC(=O)Oc1cc(C(C)Cl)ccc1OC(C)=O"
    name = eng.name(smi)
    res = opsin_roundtrip_check(smi, name)
    assert res["passed"], res


# W5 — _assemble_ring_with_ester_prefixes, count>1 NON-acyloxy branch
# ('bis' + multiplied complex substituent). At HEAD this internal candidate
# was 'bis1-chloroethyl...', OPSIN-UNPARSEABLE, so the OPSIN validity gate
# suppressed it and the engine fell through to a degraded alternate name
# (still correct-structure, but not the direct PIN-conformant construction).
def test_w5_ring_ester_bis_complex_enclosed(eng):
    smi = "CC(=O)Oc1cc(C(C)Cl)cc(C(C)Cl)c1OC(C)=O"
    name = eng.name(smi)
    assert name is not None and "unknown" not in name.lower()
    assert "bis(1-chloroethyl)" in name
    assert "bis1-chloroethyl" not in name


@pytest.mark.roundtrip
def test_w5_roundtrips_to_input(eng):
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    smi = "CC(=O)Oc1cc(C(C)Cl)cc(C(C)Cl)c1OC(C)=O"
    name = eng.name(smi)
    res = opsin_roundtrip_check(smi, name)
    assert res["passed"], res


# W6 — substituent_naming._compose_n_substituent_prefix, the count==1 complex
# N-substituent on an onium cation (P-16.3.3). At HEAD this shipped the bare
# '(ethylmethylpropan-2-ylazaniumyl)acetate' -- OPSIN-valid (structure
# correct) but the compound 'propan-2-yl' substituent is not enclosed, so it
# reads ambiguously against the two simple N-substituents beside it.
def test_w6_onium_single_complex_n_substituent_enclosed(eng):
    smi = "CC[N+](C)(C(C)C)CC(=O)[O-]"
    name = eng.name(smi)
    assert name is not None and "unknown" not in name.lower()
    assert "(propan-2-yl)" in name
    assert "methylpropan-2-yl" not in name


@pytest.mark.roundtrip
def test_w6_roundtrips_to_input(eng):
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    smi = "CC[N+](C)(C(C)C)CC(=O)[O-]"
    name = eng.name(smi)
    res = opsin_roundtrip_check(smi, name)
    assert res["passed"], res
