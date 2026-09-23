import pytest
from orthonym import name_compound

# (smiles, expected_PIN, def_id) — expected copied verbatim from the v52 evidence TSVs
# (a temp dir bb_rules/.tsv, bb_rules/.tsv), column 2 (gold BB name).
CARBOTHIOAMIDE = [
    ("NC(=S)c1ccccn1", "pyridine-2-carbothioamide", "66.1.4.1.1"),
    ("NC(=S)c1ccncc1", "pyridine-4-carbothioamide", "66.1.4.1.1"),
]
METHANETHIOAMIDE = [
    ("NC=S", "methanethioamide", "66.1.4.1.1"),
]
IMIDIC_HYDRAZONIC = [
    ("N=C(O)C1CCCCC1",  "cyclohexanecarboximidic acid",   "65.1.3.1.1"),
    ("NN=C(O)C1CCCCC1", "cyclohexanecarbohydrazonic acid", "65.1.3.2.1"),
]
CYCLIC_IMIDE_DIONE = [
    ("O=C1NC(=O)O1", "1,3-oxazetidine-2,4-dione", "66.2.1"),
]
HETEROATOM_ADDED_CARBON = [
    ("O=CP",  "phosphanecarbaldehyde", "66.6.1.1.3"),
    ("N#C[SiH3]", "silanecarbonitrile",  "66.5.1.1.3"),
    # Task F -- added-carbon -carboxylic acid on the Group-14 parent hydride
    #: the carboxy carbon is an ADDED carbon on the senior silane/
    # germane/... parent, NOT a silyl prefix on a C1 methanoic acid). Generalised
    # over the whole Group-14 family + organyl hub substituents; all
    # OPSIN-round-trip to the input InChIKey.
    ("[SiH3]C(=O)O", "silanecarboxylic acid", "65.1.2.2"),
    ("[GeH3]C(=O)O", "germanecarboxylic acid", "65.1.2.2"),
    ("C[Si](C)(C)C(=O)O", "trimethylsilanecarboxylic acid", "65.1.2.2"),
]

# Task 7 -- ester trap: a fully-esterified SYMMETRIC di/poly-ester built from
# two identical DIBASIC acid units bridged by a central symmetric divalent diol
# and capped by identical monovalent alcohols has NO surviving senior suffix
# (every acid is esterified => principal_group == "ester"), so the functional-
# class multiplicative ester name is the PIN, NOT the substitutive bis(acyloxy)
# form. "Polyester names formed by using functional class
# multiplicative nomenclature" (the Blue Book): "Symmetrical esters are
# named by including the organyl constituent in the multiplied anion component
# name." Its verbatim example is the first row below.
ESTER_TRAP = [
    # gold row -- def 65.6.3.3.4.1, expected copied verbatim from the BB example
    ("COC(=O)CCC(=O)OCCOC(=O)CCC(=O)OC",
     "dimethyl ethane-1,2-diyl dibutanedioate", "65.6.3.3.4.1"),
    # symmetric generalisations of the same class (RT-verified), to prove the
    # builder is not a single-row special case: different terminal alcohol,
    # different central diol, and a different dibasic acid.
    ("CCOC(=O)CCC(=O)OCCOC(=O)CCC(=O)OCC",
     "diethyl ethane-1,2-diyl dibutanedioate", "65.6.3.3.4.1-gen"),
    ("COC(=O)CCC(=O)OCCCOC(=O)CCC(=O)OC",
     "dimethyl propane-1,3-diyl dibutanedioate", "65.6.3.3.4.1-gen"),
    ("COC(=O)CCCC(=O)OCCOC(=O)CCCC(=O)OC",
     "dimethyl ethane-1,2-diyl dipentanedioate", "65.6.3.3.4.1-gen"),
]

# Negative control -- a surviving free -COOH keeps the acid as the principal
# characteristic group (principal_group == "carboxylic_acid", NOT "ester"), so
# the molecule NEVER reaches the ester multi-ester dispatcher: the substitutive
# acyloxy-prefix name stays the PIN. (the Blue Book) "Substitutive
# nomenclature is senior to functional class nomenclature... for esters" --
# substitutive wins WHEN a senior suffix survives. This row MUST stay unchanged.
ESTER_TRAP_SURVIVING_SUFFIX_CONTROL = [
    ("CCCCOC(=O)CCC(C)(OC(C)=O)C(=O)O",
     "2-(acetyloxy)-5-butoxy-2-methyl-5-oxopentanoic acid", "65.6.3.3.5"),
]


@pytest.mark.parametrize("smi,expected,def_id",
    CARBOTHIOAMIDE + METHANETHIOAMIDE + IMIDIC_HYDRAZONIC
    + CYCLIC_IMIDE_DIONE + HETEROATOM_ADDED_CARBON)
def test_v52_acid_deriv_pins(smi, expected, def_id):
    assert name_compound(smi) == expected, f"def_id {def_id}: {smi}"


@pytest.mark.roundtrip
@pytest.mark.parametrize("smi,expected,def_id", ESTER_TRAP)
def test_v52_ester_trap_functional_class_multiplicative(smi, expected, def_id):
    """Fully-esterified symmetric diacid diester -> functional-class
    multiplicative PIN. RT-gated in the builder, so this needs
    a JVM."""
    assert name_compound(smi) == expected, f"def_id {def_id}: {smi}"


@pytest.mark.parametrize("smi,expected,def_id",
    ESTER_TRAP_SURVIVING_SUFFIX_CONTROL)
def test_v52_ester_trap_surviving_suffix_unchanged(smi, expected, def_id):
    """Negative control: a surviving free -COOH keeps the substitutive
    acyloxy-prefix name as the PIN. The ester-trap change must
    NOT touch it."""
    assert name_compound(smi) == expected, f"def_id {def_id}: {smi}"


@pytest.mark.roundtrip
def test_v52_carbothioamide_n_substituted_guard():
    """N-methyl thioamide on pyridine. Superseded by D4 (a review-fix P3): the
     detection pattern now accepts N-substituted nitrogen, so this
    keeps the -carbothioamide suffix with an N-substituent prefix rather than
    degrading to a (methylamino)sulfanylidenemethyl prefix. See
    D4_N_SUBSTITUTED_CARBOTHIOAMIDE."""
    from orthonym.validation import opsin_roundtrip_check

    smi = "CNC(=S)c1ccccn1"
    name = name_compound(smi)
    assert name == "N-methylpyridine-2-carbothioamide"  # D4: suffix kept
    rt = opsin_roundtrip_check(smi, name)
    assert rt["passed"], f"round-trip failed for {name!r}: {rt}"


@pytest.mark.roundtrip
def test_v52_carbothioamide_demoted_to_carbamothioyl_prefix():
    """Fix a performance pass regression: a ring with a carbothioamide co-present with a
    more senior ring suffix (carboxamide) must demote carbothioamide to its
    -carbamothioyl PREFIX form, not fall back to the bare suffix WORD as a
    prefix. Before the fix, _SUFFIX_TO_PREFIX had no 'carbothioamide' entry,
    so.get(suf, suf) emitted the malformed
    '2-carbothioamidepyridine-4-carboxamide', which OPSIN rejects -> abstain."""
    from orthonym.validation import opsin_roundtrip_check

    smi = "NC(=S)c1cc(C(N)=O)ccn1"
    name = name_compound(smi)
    assert name == "2-carbamothioylpyridine-4-carboxamide"
    rt = opsin_roundtrip_check(smi, name)
    assert rt["passed"], f"round-trip failed for {name!r}: {rt}"


# --- D3 (a review-fix P3): mononuclear (methane) pseudoketone locant omission ---
# (a) (the Blue Book) "The locant '1' is omitted: (a) in
# substituted mononuclear parent hydrides" (cf. CH3Cl -> chloromethane,:2897).
# A one-carbon acyl-on-ring pseudoketone (H-CO-NR2 / H-CS-NR2) has methane as
# its parent hydride, so BOTH the ketone locant and the ring-substituent locant
# drop. Multi-carbon stems (ethane-1-thione,...) keep their '-1-'.
D3_MONONUCLEAR_PSEUDOKETONE = [
    ("S=CN1CCCC1", "(pyrrolidin-1-yl)methanethione"),
    ("S=CN1CCOCC1", "(morpholin-4-yl)methanethione"),
    ("O=CN1CCCC1", "(pyrrolidin-1-yl)methanone"),
]
D3_MULTICARBON_STEM_UNCHANGED = [
    ("CC(=S)N1CCCC1", "1-(pyrrolidin-1-yl)ethane-1-thione"),
]


@pytest.mark.roundtrip
@pytest.mark.parametrize("smi,expected", D3_MONONUCLEAR_PSEUDOKETONE)
def test_v52_d3_mononuclear_pseudoketone_no_locant(smi, expected):
    """C1 (methane) pseudoketone omits the '1' locants (a))."""
    from orthonym.validation import opsin_roundtrip_check

    name = name_compound(smi)
    assert name == expected, smi
    rt = opsin_roundtrip_check(smi, name)
    assert rt["passed"], f"round-trip failed for {name!r}: {rt}"


@pytest.mark.parametrize("smi,expected", D3_MULTICARBON_STEM_UNCHANGED)
def test_v52_d3_multicarbon_stem_keeps_locant(smi, expected):
    """A >=2-carbon stem keeps its '-1-' locant (guard for D3)."""
    assert name_compound(smi) == expected, smi


# --- D4 (a review-fix P3): N-substituted ring carbothioamide keeps the suffix ---
# (the Blue Book): N-substituted ring thioamides are
# named like carboxamides -> N-alkylpyridine-2-carbothioamide, NOT degraded to
# a (methylamino)sulfanylidenemethyl prefix.
D4_N_SUBSTITUTED_CARBOTHIOAMIDE = [
    ("CNC(=S)c1ccccn1", "N-methylpyridine-2-carbothioamide"),
    ("CN(C)C(=S)c1ccccn1", "N,N-dimethylpyridine-2-carbothioamide"),
    ("NC(=S)c1ccccn1", "pyridine-2-carbothioamide"),
]


@pytest.mark.roundtrip
@pytest.mark.parametrize("smi,expected", D4_N_SUBSTITUTED_CARBOTHIOAMIDE)
def test_v52_d4_n_substituted_ring_carbothioamide(smi, expected):
    """N-substituted ring carbothioamide keeps the -carbothioamide suffix."""
    from orthonym.validation import opsin_roundtrip_check

    name = name_compound(smi)
    assert name == expected, smi
    rt = opsin_roundtrip_check(smi, name)
    assert rt["passed"], f"round-trip failed for {name!r}: {rt}"


# --- D2 (a review-fix P3): symmetric multiplicative diester multiplier spelling ---
# /: the multiplied terminal-organyl prefix uses 'di'+name for
# a simple terminal (dimethyl), 'di-' + hyphen for an italic-prefixed terminal
# (di-tert-butyl), and the enclosed 'di(...)' / 'bis(...)' form for a locanted or
# compound terminal (di(propan-2-yl), bis(2-methylpropyl)). Never 'ditert-butyl'
# or 'dipropan-2-yl'. BB verbatim: 'di(propan-2-yl) disulfite' (:36921),
# 'bis(2-methylpropyl)...' (:41118).
D2_MULTIPLIER_SPELLING = [
    ("CC(C)(C)OC(=O)CCC(=O)OCCOC(=O)CCC(=O)OC(C)(C)C",
     "di-tert-butyl ethane-1,2-diyl dibutanedioate"),
    ("CC(C)OC(=O)CCC(=O)OCCOC(=O)CCC(=O)OC(C)C",
     "di(propan-2-yl) ethane-1,2-diyl dibutanedioate"),
    ("CC(C)COC(=O)CCC(=O)OCCOC(=O)CCC(=O)OCC(C)C",
     "bis(2-methylpropyl) ethane-1,2-diyl dibutanedioate"),
]
D2_SIMPLE_MULTIPLIER_UNCHANGED = [
    ("COC(=O)CCC(=O)OCCOC(=O)CCC(=O)OC",
     "dimethyl ethane-1,2-diyl dibutanedioate"),
    ("CCOC(=O)CCC(=O)OCCOC(=O)CCC(=O)OCC",
     "diethyl ethane-1,2-diyl dibutanedioate"),
]


@pytest.mark.roundtrip
@pytest.mark.parametrize("smi,expected", D2_MULTIPLIER_SPELLING)
def test_v52_d2_diester_multiplier_spelling(smi, expected):
    """Italic/locanted/compound terminal organyls get the correct multiplier."""
    from orthonym.validation import opsin_roundtrip_check

    name = name_compound(smi)
    assert name == expected, smi
    rt = opsin_roundtrip_check(smi, name)
    assert rt["passed"], f"round-trip failed for {name!r}: {rt}"


@pytest.mark.roundtrip
@pytest.mark.parametrize("smi,expected", D2_SIMPLE_MULTIPLIER_UNCHANGED)
def test_v52_d2_simple_multiplier_unchanged(smi, expected):
    """A simple terminal (methyl/ethyl) keeps the plain 'di'+name multiplier."""
    from orthonym.validation import opsin_roundtrip_check

    name = name_compound(smi)
    assert name == expected, smi
    rt = opsin_roundtrip_check(smi, name)
    assert rt["passed"], f"round-trip failed for {name!r}: {rt}"


# --- D1 (a review-fix P3): hetero ring with two ring C=O uses -dione suffix ---
# /: two ring C=O carbons bonded to skeletal heteroatoms
# take the -dione SUFFIX, not the -dioxo PREFIX on the bare ring.
D1_HETERO_RING_DIONE = [
    ("O=C1NCC(=O)O1", "1,3-oxazolidine-2,5-dione"),
    ("O=C1COCC(=O)O1", "1,4-dioxane-2,6-dione"),
    ("O=C1CN(C)CC(=O)O1", "4-methylmorpholine-2,6-dione"),
    ("O=C1CSCC(=O)O1", "1,4-oxathiane-2,6-dione"),
]
# Adjacent-ring guards: none of these may flip to a wrong form.
D1_ADJACENT_RING_UNCHANGED = [
    ("O=C1OCCO1", "1,3-dioxolan-2-one"),   # one ring C=O -> stays -one
    ("O=C1OCCCO1", "1,3-dioxan-2-one"),    # one ring C=O -> stays -one
    ("O=C1CCCC(=O)N1", "piperidine-2,6-dione"),  # already -dione
    ("O=C1CCCCC1=O", "cyclohexane-1,2-dione"),   # carbocyclic diketone
]


@pytest.mark.roundtrip
@pytest.mark.parametrize("smi,expected", D1_HETERO_RING_DIONE)
def test_v52_d1_hetero_ring_dione_suffix(smi, expected):
    """Hetero ring with two ring carbonyls -> -dione suffix, not -dioxo prefix."""
    from orthonym.validation import opsin_roundtrip_check

    name = name_compound(smi)
    assert name == expected, smi
    rt = opsin_roundtrip_check(smi, name)
    assert rt["passed"], f"round-trip failed for {name!r}: {rt}"


@pytest.mark.parametrize("smi,expected", D1_ADJACENT_RING_UNCHANGED)
def test_v52_d1_adjacent_rings_unchanged(smi, expected):
    """Selection-blast guard: single-C=O and carbocyclic rings stay put."""
    assert name_compound(smi) == expected, smi
