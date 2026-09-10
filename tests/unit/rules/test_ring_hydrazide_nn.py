"""
Tests for ring-parent N/N'-substituted hydrazides (Blue Book P-66.3.1.1 /
P-66.3.1.2.1 / P-66.3.3), closing the Wave2 Tier-3 documented micro-gap
"ring N-substituted hydrazide".

Three independent root causes were fixed:

(1) The composer hydrazide gate branch excluded ring parents
    (`not features.is_cyclic`), so an N/N'-substituted ring hydrazide fell to
    the ring paths, which emit only the BARE suffix -> the N-substituent was
    unclaimed -> SELF-01 -> 'unknown'. New `_assemble_ring_hydrazide_name`
    builds the base by deleting the N-substituents (RWMol) and re-entering the
    namer (the substitutive-oxime re-entry pattern), then prepends the
    N/N' prefix. Fail-closed: coverage guard rejects ring-substituted +
    N-substituted combos (needs P-14.5.2 combined alphanumeric locants).

(2) P-66.3.1.2.1: 'benzohydrazide' is one of the five RETAINED hydrazide PINs
    and "can be substituted in the same way as corresponding amides" — the
    benzene assembly emitted the systematic 'benzenecarbohydrazide' stem.

(3) The decomposition engine labelled hydrazide C(=O)-N bonds as cleavable
    amides; functional-class cleavage cannot express WHICH nitrogen carries
    the acyl ('N-benzoylphenylhydrazine' names the wrong constitution) and
    P-66.3.1.1 rejects acyl-hydrazine names outright
    ['not (cyclohexanecarbonyl)hydrazine'].

Also fixed while here (shipped-T3d formatter defect): identical substituents
on N and N' now share one multiplier across the nitrogens
("N,N'-dimethyl...", the BB 'N1,N'4-dimethylnaphthalene-1,4-dicarbohydrazide'
style, P-16.3.3) instead of being cited twice ("N-methyl-N'-methyl...").

All expected PINs below are OPSIN-round-trip verified (scripts/diagnose.py).
"""

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# (1) Ring-parent N/N'-substituted hydrazides -- previously 'unknown'.
# N = the nitrogen bonded to the acyl C; N' = the terminal nitrogen
# (P-66.3.3).
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # benzene parent -> retained benzo stem (P-66.3.1.2.1)
        ("CNNC(=O)c1ccccc1", "N'-methylbenzohydrazide"),
        ("CN(N)C(=O)c1ccccc1", "N-methylbenzohydrazide"),
        ("CCNNC(=O)c1ccccc1", "N'-ethylbenzohydrazide"),
        ("CN(C)NC(=O)c1ccccc1", "N',N'-dimethylbenzohydrazide"),
        ("CNN(C)C(=O)c1ccccc1", "N,N'-dimethylbenzohydrazide"),
        ("CN(C)N(C)C(=O)c1ccccc1", "N,N',N'-trimethylbenzohydrazide"),
        ("O=C(NNc1ccccc1)c1ccccc1", "N'-phenylbenzohydrazide"),
        # saturated carbocycle -> -carbohydrazide (P-66.3.1.1)
        ("CNNC(=O)C1CCCCC1", "N'-methylcyclohexanecarbohydrazide"),
        ("CN(N)C(=O)C1CCCCC1", "N-methylcyclohexanecarbohydrazide"),
        ("CNN(C)C(=O)C1CCCCC1", "N,N'-dimethylcyclohexanecarbohydrazide"),
        # heterocycle parent
        ("CNNC(=O)c1ccncc1", "N'-methylpyridine-4-carbohydrazide"),
        # thiohydrazide sibling (P-66.3.4)
        ("CNNC(=S)c1ccccc1", "N'-methylbenzenecarbothiohydrazide"),
    ],
)
def test_ring_hydrazide_nn_substituted(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# (2) Retained benzo stem for the benzene carbohydrazide (P-66.3.1.2.1),
# substituted the same way as benzamide.
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("NNC(=O)c1ccccc1", "benzohydrazide"),
        ("Cc1ccc(cc1)C(=O)NN", "4-methylbenzohydrazide"),
    ],
)
def test_benzohydrazide_retained_stem(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# (T3d formatter) merged multiplier ACROSS the two nitrogens for identical
# substituents on the acyclic path too (P-16.3.3).
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CNN(C)C(=O)CC", "N,N'-dimethylpropanehydrazide"),
        ("CNN(CC)C(=O)CC", "N-ethyl-N'-methylpropanehydrazide"),
    ],
)
def test_chain_hydrazide_merged_nn_multiplier(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# Protection: neighbours that already worked must be unchanged.
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # bare ring hydrazides (retained / ring suffix paths)
        ("NNC(=O)C1CCCCC1", "cyclohexanecarbohydrazide"),
        ("NNC(=O)c1ccncc1", "pyridine-4-carbohydrazide"),
        ("NNC(=S)c1ccccc1", "benzenecarbothiohydrazide"),
        # multi-instance keeps the systematic path (P-66.3.1.2.2)
        ("NNC(=O)c1ccc(C(=O)NN)cc1", "benzene-1,4-dicarbohydrazide"),
        # sulfonohydrazide branch untouched (P-65.3.1)
        ("NNS(=O)(=O)c1ccccc1", "benzenesulfonohydrazide"),
        ("Cc1ccc(cc1)S(=O)(=O)NN", "4-methylbenzenesulfonohydrazide"),
        # acyclic N/N' path (T3d) unchanged
        ("CNNC(C)=O", "N'-methylethanehydrazide"),
        ("CN(N)C(C)=O", "N-methylethanehydrazide"),
        ("CNNC(=O)O", "N'-methylhydrazinecarboxylic acid"),
        ("CNNC(=S)C", "N'-methylethanethiohydrazide"),
        # dihydrazide via the decomposition quality gate (C1)
        ("NNC(=O)CCC(=O)NN", "butanedihydrazide"),
        # amide neighbours (decomposition amide cleavage NOT affected)
        ("CNC(=O)c1ccccc1", "N-methylbenzamide"),
        ("O=C(Nc1ccccc1)c1ccccc1", "N-phenylbenzamide"),
        ("O=C(NCC(=O)O)c1ccccc1", "2-benzamidoethanoic acid"),
        # semicarbazide / urea family (urea C(=O)-NH2 side still cleavable)
        # P-68.3.1.2.4 (BB 38623): hydrazinecarboxamide is the PIN;
        # semicarbazide is general nomenclature only (plan P1AM Task 2).
        ("O=C(NN)N", "hydrazinecarboxamide"),
        ("NC(=O)N", "urea"),
        # phenylhydrazine parent itself (polyazane)
        ("NNc1ccccc1", "phenylhydrazine"),
    ],
)
def test_ring_hydrazide_protection(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# Fail-closed: combinations the prefix-prepend model cannot express must stay
# 'unknown' (never a wrong-ordered or atom-dropping name). The suite's autouse
# fixture disables the OPSIN validity gate (raw output leaks); these cases are
# ABOUT the gate suppressing in production -- re-enable it (the
# test_tier3b_sulfoxide_substitutive.py pattern).
# ---------------------------------------------------------------------------

@pytest.fixture
def _validity_gate_on(monkeypatch):
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    yield


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles",
    [
        # ring-substituted + N-substituted (needs P-14.5.2 combined
        # alphanumeric locant ordering: 'N',4-dimethyl...' etc.)
        "CNNC(=O)c1ccc(C)cc1",
        # N-acyl ring hydrazide (N'-benzoylbenzohydrazide, P-66.3.3.2 acyl
        # N-subs are out of the pure-alkyl collector's scope)
        "O=C(NNC(=O)c1ccccc1)c1ccccc1",
        # N-attached ring hydrazide, N-substituted (piperidine-1-carbohydrazide
        # derivative -- acyl C attaches to a ring NITROGEN, deferred)
        "CNNC(=O)N1CCCCC1",
    ],
)
def test_ring_hydrazide_fail_closed(smiles, _validity_gate_on):
    assert "unknown" in name_compound(smiles)
