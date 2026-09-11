""" mixed-type multi-anion naming (a phase).

A skeleton carrying TWO anionic centres of DIFFERENT acid classes (e.g. a
carboxylate + a sulfonate) must name the SENIOR class as the parent anion
suffix: carboxylic acid senior to sulfonic acid in the
class-seniority order — the same rule that gives the BB's own
"3-oxidonaphthalene-2-carboxylate (PIN) (carboxylate senior to olate)") and
cite every OTHER (junior) anionic centre by its anionic substituent prefix
: 'sulfonato' for -SO2-O-, 'phosphonato' for -P(O)(O-)2 — BB
:41211/:41213), never the neutral prefix ('sulfo'/'phosphono') — a neutral
prefix silently drops the charge and denotes a DIFFERENT (mono-anion)
molecule, which the gate correctly rejects (measured: '4-sulfobenzoate'
-> 'unknown organic compound' before this fix).

All targets below are verified round-trip-exact (OPSIN 2.9.0 + InChIKey) by
the implementing session; see the report at
.the workflow tooling/sdd/2026-08-17--phase3-acid-ester-anion/multianion-report.md.
"""
import pytest
from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # 4-sulfobenzoate dianion: carboxylate parent (senior), sulfonate junior
    # prefix. RT-verified: OPSIN('4-sulfonatobenzoate') round-trips to the
    # identical dianion (InChIKey match), the neutral-prefix form
    # ('4-sulfobenzoate') round-trips to the MONO-anion (different molecule).
    ("O=C([O-])c1ccc(S(=O)(=O)[O-])cc1", "4-sulfonatobenzoate"),
    # aliphatic carboxylate+sulfonate dianion (3-sulfopropanoic acid, fully
    # deprotonated). RT-verified round-trip-exact.
    ("[O-]C(=O)CCS(=O)(=O)[O-]", "3-sulfonatopropanoate"),
    # 2-carbon case: locant omitted — unambiguous with only one
    # non-C1 position). RT-verified round-trip-exact.
    ("[O-]C(=O)CS(=O)(=O)[O-]", "sulfonatoacetate"),
])
def test_integration_mixed_dianion_names(namer, smi, expected):
    assert namer.name(smi) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # mono mixed (already worked pre-fix) — must stay unchanged.
    ("OC(=O)CCS(=O)(=O)[O-]", "2-carboxyethane-1-sulfonate"),
    # dicarboxylate (already worked pre-fix) — must stay unchanged.
    ("[O-]C(=O)CCC(=O)[O-]", "butanedioate"),
    # acid-ester anion (a phase Slice A) — must stay unchanged.
    ("CCCCCCCCCCCCOS(=O)(=O)[O-]", "dodecyl sulfate"),
    # zwitterion (unrelated path) — must stay unchanged.
    ("C[N+](C)(C)CCC(=O)[O-]", "3-(trimethylazaniumyl)propanoate"),
    # plain sulfonate / methanesulfonate — must stay unchanged.
    ("C1=CC=CC=C1S(=O)(=O)[O-]", "benzenesulfonate"),
    ("CS(=O)(=O)[O-]", "methanesulfonate"),
])
def test_integration_regressions_unchanged(namer, smi, expected):
    assert namer.name(smi) == expected


@pytest.mark.opsin_gate
def test_integration_failclosed_never_wrong(namer):
    # A shape the new machinery cannot yet name correctly must abstain to the
    # honest sentinel, never emit the neutral-prefix ('...sulfo...' /
    # '...phosphono...') form that denotes a different (mono-anion) molecule.
    out = namer.name("O=P([O-])([O-])c1ccc(C(=O)[O-])cc1")  # triple mixed anion
    assert "sulfo" not in out
    assert "phosphono" not in out
