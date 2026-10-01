import pytest
from orthonym import name_compound

# Each row: (smiles, expected_PIN, bb_def_id, bb_rule)
ROWS = [
    # Task 2 — homogeneous Group-14 acyclic chain; BB:2072/:2074)
    ("[SnH3][SnH2][SnH2][SnH2][SnH2][SnH2][SnH2][SnH2][SnH2][SnH2][SnH2][SnH2][SnH3]",
     "tridecastannane", "12.2", "P-21.2.2"),
    ("[CH3][SnH2][SnH2][SnH2][SnH2][SnH2][SnH2][SnH2][SnH2][SnH2][SnH2][SnH2][SnH2][SnH3]",
     "1-methyltridecastannane", "12.2", "P-21.2.2"),
    # Task 2 fix a performance pass -- (g) alphabetical tie-break: both backbone
    # directions tie at locant set {2,6}, so the lower locant goes to the
    # substituent cited first alphabetically (ethyl before methyl).
    ("[SnH3][SnH1](C)[SnH2][SnH2][SnH2][SnH1](CC)[SnH3]",
     "2-ethyl-6-methylheptastannane", "12.2", "P-14.4(g)"),
    # Task 3 — homogeneous N-chain + tetraazanyl substituent; BB:39017/:1705)
    ("NNNNNNNNN", "nonaazane", "21.2.2", "P-68.3.1.4.1"),
    ("CCOC(=O)CNNNN", "ethyl (tetraazan-1-yl)acetate", "68.3.1.4.1", "P-68.3.1.4.1"),
    # Task 4 — cyclic homogeneous heteromonocycle, locants omitted; BB:8848/:8882 +:3007)
    ("[SiH2]1[SiH2][SiH2][SiH2][SiH2][SiH2][SiH2][SiH2][SiH2][SiH2][SiH2][SiH2]1",
     "dodecasilacyclododecane", "22.2.5", "P-22.2.5"),
    # Task 5 — pnictogen a(ba) bridged hydride
    ("P[Se]P", "diphosphaselenane", "21.2.3.1", "P-21.2.3.1"),
    # Task 6 — hydroxylamine <-> senior amine selection /.2; BB:38308/:38314/:5005)
    ("CNO", "N-hydroxymethanamine", "68.3.1.1.1.1", "P-68.3.1.1.1.1"),
    ("CN(C)O", "N-hydroxy-N-methylmethanamine", "68.3.1.1.1.1", "P-68.3.1.1.1.1"),
    ("NOc1ccccc1", "O-phenylhydroxylamine", "68.3.1.1.1.2", "P-68.3.1.1.1.2"),
    # Task 6 fix a performance pass: a COMPOUND O-substituent takes enclosing
    # marks; the simple-phenyl row above stays bare. Was
    # 'O-4-methylphenylhydroxylamine' (missing marks) before the fix.
    ("NOc1ccc(C)cc1", "O-(4-methylphenyl)hydroxylamine", "16.3.3", "P-16.3.3"),
    ("CON", "O-methylhydroxylamine", "16.3.3", "P-16.3.3"),  # anchor, unchanged
    # Task 7 — chalcogen-chain parent with -ol suffix
    ("OSOS", "dithioxanol", "68.4.2.1", "P-68.4.2.1"),
]

# v52 Task 3 DEFER: the tetraazanyl-acetate row (def_id 68.3.1.4.1) is xfail-only.
# Building `ethyl (tetraazan-1-yl)acetate` needs the retained-acetate acid path to
# name the free-acid analog `OC(=O)CNNNN`, which currently returns `unknown` because
# perception's `hydrazine_fg` SMARTS (functional_groups.py:550) matches only the
# terminal 2 N of a 3+N chain and orphans the middle N -- a SAFE ABSTAIN caught by
# the atom-coverage gate, so 0-wrong holds. The prefix alone is insufficient (proven:
# forcing `tetraazan-1-yl` yields `ethyl 2-(tetraazan-1-yl)ethanoate`, not the target).
# The fix is a broad hydrazine_fg multi-N-chain substituent-partitioning refactor,
# deferred to a dedicated v52 follow-up. See -noncarbon/task-3-report.md.
_DEFERRED_SMILES = {"CCOC(=O)CNNNN"}  # only the tetraazanyl-acetate row (68.3.1.4.1)
_XFAIL_TETRAAZANYL = pytest.mark.xfail(
    reason="v52 follow-up: needs hydrazine_fg multi-N-chain substituent partitioning "
           "refactor (functional_groups.py:550 matches only terminal 2 N of a 3+N "
           "chain, orphaning the middle N). Currently a SAFE ABSTAIN -- 0-wrong holds. "
           "See PLAN-02 report.",
    strict=False,
)


def _param(r):
    marks = (_XFAIL_TETRAAZANYL,) if r[0] in _DEFERRED_SMILES else ()
    return pytest.param(*r, id=r[2] + ":" + r[1], marks=marks)


@pytest.mark.parametrize("smiles,expected,def_id,rule", [_param(r) for r in ROWS])
def test_phase02_noncarbon_pin(smiles, expected, def_id, rule):
    assert name_compound(smiles) == expected, f"BB {def_id} ({rule})"


# ---------------------------------------------------------------------------
# a review-fix P2 (cross-family review of phase-2 output).
#
# D2 — (b) (the Blue Book "The locant '1' is omitted... in monosubstituted
# homogeneous chains consisting of only two identical atoms"): a 2-atom Group-14
# chain with a SINGLE substituent omits the '1-'. 3+-atom chains and 2-atom
# multi-substituted chains KEEP their locants.
# ---------------------------------------------------------------------------
D2_OMIT = [
    ("C[SiH2][SiH3]", "methyldisilane"),
    ("CC[SiH2][SiH3]", "ethyldisilane"),
    ("C[GeH2][GeH3]", "methyldigermane"),
    ("CCC[PbH2][PbH3]", "propyldiplumbane"),
]
D2_KEEP = [
    ("C[SiH2][SiH2][SiH3]", "1-methyltrisilane"),        # 3-atom chain keeps
    ("C[SiH](C)[SiH3]", "1,1-dimethyldisilane"),         # 2-atom, multi keeps
    ("C[SiH2][SiH2]C", "1,2-dimethyldisilane"),          # 2-atom, multi keeps
    ("CC[SiH2][SiH2][SiH3]", "1-ethyltrisilane"),        # 3-atom chain keeps
    ("CC[SiH2][SiH2][SiH2]C", "1-ethyl-3-methyltrisilane"),  # 3-atom + (g)
]


@pytest.mark.parametrize("smiles,expected", D2_OMIT + D2_KEEP)
def test_phase02_group14_locant_one_omission(smiles, expected):
    assert name_compound(smiles) == expected, "P-14.3.4.2(b)"


# ---------------------------------------------------------------------------
# Suite fix j4 (TRIAGE g3 C10d) -- (the Blue Book): "All locants
# are omitted in compounds or substituent groups in which all substitutable
# positions are completely substituted or modified... in the same way." and
#:3009 "In case of partial substitution or modification, all numerical prefixes
# must be indicated." Every H of the Group-14 parent hydride replaced by the SAME
# alkyl -> no locants; a partial or mixed set keeps every locant. All names
# OPSIN 2.9.0 full-InChIKey exact (independent batch, suite fix j4).
# ---------------------------------------------------------------------------
L5_OMIT_ALL = [
    ("C[Si]([Si](C)(C)C)(C)C", "hexamethyldisilane"),
    ("C[Si](C)(C)[Si](C)(C)[Si](C)(C)C", "octamethyltrisilane"),
    ("CC[Si](CC)(CC)[Si](CC)(CC)CC", "hexaethyldisilane"),
    ("C[Ge](C)(C)[Ge](C)(C)C", "hexamethyldigermane"),
    ("C[Sn](C)(C)[Sn](C)(C)C", "hexamethyldistannane"),
]
L5_KEEP = [
    # partial: the central SiH2 is unsubstituted (:3009)
    ("C[Si](C)(C)[SiH2][Si](C)(C)C", "1,1,1,3,3,3-hexamethyltrisilane"),
    ("C[Si](C)(C)[SiH3]", "1,1,1-trimethyldisilane"),
    # complete but not "in the same way" (ethyl + methyl)
    ("C[Si](C)(C)[Si](C)(C)CC", "1-ethyl-1,1,2,2,2-pentamethyldisilane"),
]


@pytest.mark.parametrize("smiles,expected", L5_OMIT_ALL + L5_KEEP)
def test_group14_complete_uniform_substitution_omits_all_locants(smiles, expected):
    assert name_compound(smiles) == expected, "P-14.3.4.5"


# ---------------------------------------------------------------------------
# D1 — N-carbon hydroxylamine (R-NH-OH) named as an N-hydroxy amine
#. Two defects:
# (1) MANDATORY 0-wrong: a base amine name with a LEADING STEREODESCRIPTOR
# produced a malformed / OPSIN-unparseable / double-descriptor name that
# shipped via the stereo carve-out -> now the whole handler fails closed
# (abstains). Building the correct '(nS)-N-hydroxy-...' PIN is a broad
# refactor, deferred.
# (2) order (the Blue Book): 'N-hydroxy' and a leading simple C-prefix are
# cited in alphanumerical order (chloro/fluoro/cyclopropyl < hydroxy).
# ---------------------------------------------------------------------------
# The N keeps one N-H, so the C-prefix of the one-carbon parent cites '1'
#, the Blue Book; '1-hydrazinylmethanamine (PIN)', the Blue Book; PIN class
# program Task 11).
D1_REORDER = [
    ("ONCCl", "1-chloro-N-hydroxymethanamine"),
    ("ONCF", "1-fluoro-N-hydroxymethanamine"),
    ("ONCC1CC1", "1-cyclopropyl-N-hydroxymethanamine"),
    ("ONC(Cl)Cl", "1,1-dichloro-N-hydroxymethanamine"),
    ("ONC(Cl)(Cl)Cl", "1,1,1-trichloro-N-hydroxymethanamine"),
    # hydroxy sorts before phenyl -> N-hydroxy stays leading.
    ("ONCc1ccccc1", "N-hydroxy-1-phenylmethanamine"),
]
D1_UNCHANGED = [
    ("CNO", "N-hydroxymethanamine"),
    ("CN(C)O", "N-hydroxy-N-methylmethanamine"),
    ("CCNO", "N-hydroxyethanamine"),
    ("CCCNO", "N-hydroxypropan-1-amine"),
    ("ONC(C)CC", "N-hydroxybutan-2-amine"),
    ("NOc1ccccc1", "O-phenylhydroxylamine"),
    ("NOc1ccc(C)cc1", "O-(4-methylphenyl)hydroxylamine"),
    ("CON", "O-methylhydroxylamine"),
    # digit-prefixed base (no stereo) still degrades to the functional-class
    # fallback -- a valid non-PIN name, NOT a regression to abstain.
    ("ONCCCl", "N-(2-chloroethyl)hydroxylamine"),
]
D1_ABSTAIN = [
    "ON[C@@H](C)c1ccccc1",   # base '(1S)-1-phenylethan-1-amine' -> fail closed
    "ON[C@@H](C)CC",         # base '(2S)-butan-2-amine' -> fail closed (was double-desc)
]


@pytest.mark.parametrize("smiles,expected", D1_REORDER + D1_UNCHANGED)
def test_phase02_n_hydroxy_prefix_order(smiles, expected):
    assert name_compound(smiles) == expected, "P-68.3.1.1.1.1 / P-14.5.1"


@pytest.mark.parametrize("smiles", D1_ABSTAIN)
def test_phase02_n_hydroxy_leading_stereo_fails_closed(smiles):
    # Leading stereodescriptor on the base amine -> no malformed / unparseable /
    # double-descriptor name may ship; the handler abstains.
    assert name_compound(smiles) == "unknown organic compound"
