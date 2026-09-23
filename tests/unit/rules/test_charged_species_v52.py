"""v52 a phase — charged species (cations / anions / zwitterions) name on the
proper parent-hydride + onium/ide/uide suffix /, not the T4
`-a`/`-onia`-replacement spelling.

SMILES + expected PINs are pulled VERBATIM from the Blue-Book conformance oracle
(`benchmarks/bb_conformance/bb_measure_rows.jsonl`); each `expected` is the
BB `(PIN)` string. The bb harness names with
`Orthonym(general_fallback=True, general_fallback_unverified=True,
allow_aromatic_general=True).name_tiered(smiles)` — the plain CLI abstains for
these because it lacks `general_fallback_unverified`, so EVERY assertion here
uses those flags + `name_tiered`, or it will not reproduce.

RED baseline (measured 2026-09-22, fresh process): each ROW currently emits the
`-a`-replacement spelling (`1-oxaprop-1-en-1-ium` for `CC=[OH+]`) or abstains;
each CONTROL already matches and must never regress.
"""
import pytest

from orthonym import Orthonym

FLAGS = dict(general_fallback=True, general_fallback_unverified=True,
             allow_aromatic_general=True)


@pytest.fixture(scope="module")
def namer():
    return Orthonym(**FLAGS)


# (smiles, expected_PIN, bb_def_id) — pulled from bb_measure_rows.jsonl
ROWS = [
    # --- A: substituted oxidanium / sulfanium (=O(+)/=S(+) parent) ---
    ("CC=[OH+]", "ethylideneoxidanium", "73.1.2.1"),
    ("CC(=O)[OH2+]", "acetyloxidanium", "73.1.2.1"),
    ("CC[O+]=C(C)C", "ethyl(propan-2-ylidene)oxidanium", "73.1.2.1"),
    ("C[O+](C)C(=O)C1CCCCC1", "(cyclohexanecarbonyl)di(methyl)oxidanium", "73.1.2.1"),
    ("C[S+](C)C(=O)c1ccccc1", "benzoyldi(methyl)sulfanium", "73.1.2.1"),
    ("O=C(O[OH2+])c1ccccc1", "2-benzoyldioxidan-1-ium", "73.1.2.1"),
    # --- B:.x catenated / substituted parent-hydride cations ---
    ("CN(C)[N+](C)(C)C", "pentamethylhydrazinium", "73.1.1.2"),
    ("CS[S+](C)SC", "1,2,3-trimethyltrisulfan-2-ium", "73.1.1.2"),
    ("C[P+](C)(C)Cl", "chlorotri(methyl)phosphanium", "73.1.1.1"),
    ("C[P+](C)(C)P(Cl)Cl", "2,2-dichloro-1,1,1-trimethyldiphosphan-1-ium", "73.1.1.2"),
    ("CC#[O+]", "ethylidyneoxidanium", "73.1.1.1"),
    # abstainers to recover:
    ("C[F+]Cl", "chloro(methyl)fluoranium", "73.1.1.1"),
    ("C[Cl+]C(C)=O", "acetyl(methyl)chloranium", "73.1.2.1"),
    # --- C: ring-cation lowest locant (currently 4-methyl-, want 1-) ---
    ("C[N+]12CCN(CC1)C2", "1-methyl-1,4-diazabicyclo[2.2.1]heptan-1-ium", "73.4"),
    # --- D:.x mononuclear / multi -ide / -uide (anions) ---
    ("[NH-]c1ccccc1", "benzenaminide", "72.2.2.2.3"),
    ("C#[Si-]", "methylidynesilanide", "72.2.2.1"),
    ("N#C[C-](C#N)C#N", "tricyanomethanide", "72.2.2.1"),
    # --- E (Task 6): bis(ylium) — a single carbon bearing a DOUBLE
    # positive charge is the geminal (one-atom) degenerate of the poly-carbenium:
    # two -ylium free valences on ONE skeletal carbon (the Blue Book 'bis(ylium)', not
    # 'diylium'). get_ion_sites reports it as ONE +2 cation, so the >=2-centre
    # poly-ylium path never saw it -> HEAD dropped a charge to 'propylium'.
    ("C[C+2]C", "propane-2,2-bis(ylium)", "73.2.2.1.1"),
]

# Positive controls that ALREADY match — must never regress.
CONTROLS = [
    ("[NH4+]", "azanium"),
    ("[SH3+]", "sulfanium"),
    ("[PH4+]", "phosphanium"),
    ("C[S+](C)C", "trimethylsulfanium"),
    ("C[SH+]C", "dimethylsulfanium"),
    ("C[n+]1ccccc1", "1-methylpyridin-1-ium"),
    ("c1ccc([I+]c2ccccc2)cc1", "diphenyliodanium"),
    ("CN(C)C(N)=[NH2+]", "N,N-dimethylguanidinium"),
    ("[C-]#[C-]", "ethynediide"),
    ("CCCC=[N-]", "butaniminide"),
    ("[NH-]CC[NH-]", "ethane-1,2-bis(aminide)"),
    ("C[P-]C", "dimethylphosphanide"),
    # Task 6: the >=2-centre poly-ylium / poly-aminium bis-forms the single-atom
    # bis(ylium) generalization must NOT regress /, the Blue Book).
    ("[CH2+][CH2+]", "ethane-1,2-bis(ylium)"),
    ("[CH2+]C[CH2+]", "propane-1,3-bis(ylium)"),
    ("[NH3+]CC[NH3+]", "ethane-1,2-bis(aminium)"),
]


@pytest.mark.parametrize("smiles,expected,bb", ROWS)
def test_charged_pin(namer, smiles, expected, bb):
    # BB rule <bb> — section headings in internal notes
    assert namer.name_tiered(smiles)["name"] == expected


@pytest.mark.parametrize("smiles,expected", CONTROLS)
def test_charged_controls(namer, smiles, expected):
    assert namer.name_tiered(smiles)["name"] == expected


# Task 7: single-cation shapes that fall through every cation branch to the common
# ionize TAIL must not raise. route_charged referenced `_single` (assigned only in
# the anion-only branch) from the shared tail, so a pure cation with a falsy
# `ionized` hit an UnboundLocalError — caught upstream as an abstain, so 0-wrong
# held, but a caught crash is a latent robustness defect. These stay ABSTAINING
# (blocked residuals: silylium / N-onium-ylidene producers, see -ions.md);
# the assertion is only that route_charged TERMINATES without an exception and never
# ships a name (name_tiered would MASK a regression by catching the exception, so
# this asserts on route_charged directly).
_FALLTHROUGH_CATIONS = [
    "C[Si+]([Si](C)(C)C)[Si](C)(C)C",     # heptamethyltrisilan-2-ylium (blocked)
    "c1ccc([Si+](c2ccccc2)c2ccccc2)cc1",  # triphenylsilylium (blocked)
    "CC(O)=[N+](C)C",                     # (1-hydroxyethylidene)di(methyl)azanium (blocked)
]


@pytest.mark.parametrize("smiles", _FALLTHROUGH_CATIONS)
def test_fallthrough_cation_no_crash(smiles):
    from rdkit import Chem

    from orthonym.rules.charged_router import route_charged
    # Must not raise (regression: route_charged `_single` UnboundLocalError).
    assert route_charged(Chem.MolFromSmiles(smiles), "pin") == ""
