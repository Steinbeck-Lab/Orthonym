"""Wave 1 root-fix tests — R9: heteroatom-variety order P-44.2.1.8."""
from orthonym.namer import name_compound


def test_r9_morpholine_senior_to_pyrimidine():
    # O (in morpholine) ranks above N (extra in pyrimidine) -> morpholine is the parent.
    out = name_compound("C1COCCN1Cc1cncnc1", style="pin")
    assert out.endswith("morpholine")          # e.g. 4-(pyrimidin-5-ylmethyl)morpholine
    assert "pyrimidin" not in out.split("morpholine")[0][-12:]  # pyrimidine is the substituent


# ---------------------------------------------------------------------------
# R8a — acyclic/aromatic hydrazide suffix + terminal-1 elision (P-66.3.1.1)
# ---------------------------------------------------------------------------

def test_r8a_hydrazide_pins():
    """P-66.3.1.1: hydrazide suffix = chain stem + hydrazide (no locant-1).
    - pentanehydrazide: C5 chain; suffix 'hydrazide', terminal -> locant-1 elided,
      stem+ane kept (h is consonant -> no vowel elision).
    - acetohydrazide: retained acyl-stem for C2 (aceto-).
    - formohydrazide: retained acyl-stem for C1 (formo-).
    - benzohydrazide: benzene ring with C(=O)NN suffix -> retained PIN 'benzohydrazide'.
    """
    assert name_compound("CCCCC(=O)NN", style="pin") == "pentanehydrazide"
    assert name_compound("CC(=O)NN", style="pin") == "acetohydrazide"
    assert name_compound("O=CNN", style="pin") == "formohydrazide"
    assert name_compound("O=C(NN)c1ccccc1", style="pin") == "benzohydrazide"


# ---------------------------------------------------------------------------
# R8b — amidine prefix carbamimidoyl (P-66.4.1.3.1)
# ---------------------------------------------------------------------------

def test_r8b_amidine_prefix_carbamimidoyl():
    """P-66.4.1.3.1: amidine as non-principal substituent uses prefix 'carbamimidoyl'.
    When COOH is the principal group and C(=NH)NH2 is a ring substituent,
    the amidine must be named as 'carbamimidoyl', not 'carbamoyl' (amide prefix).
    """
    assert name_compound("N=C(N)c1ccc(C(=O)O)cc1", style="pin") == "4-carbamimidoylbenzoic acid"
    # REGRESSION GUARD: saturated-ring amidine suffix must remain correct
    assert name_compound("N=C(N)C1CCCCC1", style="pin") == "cyclohexanecarboximidamide"


# ---------------------------------------------------------------------------
# R8c — hydroxamic acid -> N-hydroxy...amide PIN (P-66.1.1.3.2 / P-65.1.3.4)
# ---------------------------------------------------------------------------

def test_r8c_hydroxamic_acid_pin():
    """P-66.1.1.3.2 / P-65.1.3.4: -C(=O)-NH-OH is an amide with an N-hydroxy
    substituent; the PIN is 'N-hydroxy<stem>amide', not the retained
    'hydroxamic acid' string.
    """
    assert name_compound("CC(=O)NO", style="pin") == "N-hydroxyacetamide"
    assert name_compound("CCC(=O)NO", style="pin") == "N-hydroxypropanamide"
    assert name_compound("O=C(NO)C1CCCCC1", style="pin") == "N-hydroxycyclohexanecarboxamide"


def test_r8c_ring_benzene_hydroxamic():
    """C3: ring-attached hydroxamic acid on benzene -> N-hydroxybenzamide.
    OPSIN-verified: O=C(NO)c1ccccc1 round-trips correctly.
    """
    assert name_compound("O=C(NO)c1ccccc1", style="pin") == "N-hydroxybenzamide"
    # Substituted benzene hydroxamic: 2-hydroxy on ring
    assert name_compound("O=C(NO)c1ccccc1O", style="pin") == "N-hydroxy-2-hydroxybenzamide"


def test_r8c_ring_pyridine_hydroxamic():
    """C3: ring-attached hydroxamic acid on pyridine rings -> N-hydroxypyridine-X-carboxamide.
    OPSIN-verified: all four isomers.
    """
    assert name_compound("O=C(NO)c1ccncc1", style="pin") == "N-hydroxypyridine-4-carboxamide"
    assert name_compound("O=C(NO)c1cccnc1", style="pin") == "N-hydroxypyridine-3-carboxamide"
    assert name_compound("O=C(NO)c1ccccn1", style="pin") == "N-hydroxypyridine-2-carboxamide"


def test_r8c_ring_pyrimidine_hydroxamic():
    """C3: ring-attached hydroxamic acid on pyrimidine -> N-hydroxypyrimidine-5-carboxamide.
    OPSIN-verified.
    """
    assert name_compound("O=C(NO)c1cncnc1", style="pin") == "N-hydroxypyrimidine-5-carboxamide"


def test_r8c_ring_guard_plain_benzamide():
    """Guard: plain benzamide and N-methylbenzamide must be UNCHANGED after C3 fix."""
    assert name_compound("O=C(N)c1ccccc1", style="pin") == "benzamide"
    assert name_compound("O=C(NC)c1ccccc1", style="pin") == "N-methylbenzamide"


def test_r5_medium_rings_use_hantzsch_widman():
    """P-22.2.2.1: saturated heterocyclic rings of size 7-10 use Hantzsch-Widman
    (oxepane/oxocane/oxonane/oxecane), not '1-oxacyclo...ane' skeletal replacement.

    C6b (completes 1.7) extends this to SATURATED MULTI-heteroatom medium rings:
    they too take the HW PIN (1,4-dioxepane), not the replacement form.

    P-22.2.3 (``BlueBookV2.md:8482``) then settles the unsaturated half, which
    P-22.2.2.1 above does not speak to: "Mancude and saturated heteromonocyclic
    compounds with up to and including ten ring members are named by the
    extended Hantzsch-Widman system (see P-22.2.2). For monocyclic rings with
    eleven and more ring members, skeletal replacement ('a') nomenclature (see
    P-15.4) is used".  The boundary is ring size alone.  ONLY rings > 10 keep
    skeletal replacement.
    """
    assert name_compound("O1CCCCCC1", style="pin") == "oxepane"    # 7
    assert name_compound("O1CCCCCCC1", style="pin") == "oxocane"   # 8
    assert name_compound("O1CCCCCCCC1", style="pin") == "oxonane"  # 9
    assert name_compound("O1CCCCCCCCC1", style="pin") == "oxecane" # 10
    assert name_compound("S1CCCCCC1", style="pin") == "thiepane"   # S analogue
    # REGRESSION GUARD: 11+ rings stay skeletal-replacement
    assert "oxacycloundecane" in name_compound("O1CCCCCCCCCC1", style="pin")
    # C6b: saturated 2-heteroatom 7-ring now takes the HW PIN (was replacement).
    assert name_compound("O1CCOCCC1", style="pin") == "1,4-dioxepane"
    # P-22.2.3 + P-31.2.3.1: an UNSATURATED 7-ring is HW too, with the degree of
    # hydrogenation carried by hydro prefixes on the mancude parent (oxepine).
    # Hydro prefixes take the lowest locants, giving '2,3,4,5' not '4,5,6,7'.
    assert name_compound("O1CCCCC=C1", style="pin") == "2,3,4,5-tetrahydrooxepine"


# ---------------------------------------------------------------------------
# R3 — amines go substitutive, not aza-replacement (P-62.2.2)
# ---------------------------------------------------------------------------

def test_r3_amine_not_aza_replacement():
    """P-62.2.2: trivalent N bonded only to C uses substitutive naming (N-propylpropan-1-amine),
    NOT skeletal replacement (4-azaheptane). Polyazane (N–N chain) and ether replacement
    must not be affected."""
    assert name_compound("CCCNCCC", style="pin") == "N-propylpropan-1-amine"   # was 4-azaheptane
    # REGRESSION GUARDS — these must stay correct:
    assert name_compound("OCCOCCOCC", style="pin") == "3,6-dioxaoctan-1-ol"     # ether replacement OK
    assert name_compound("NNN", style="pin") == "triazane"                       # polyazane OK (N–N bonded)


# ---------------------------------------------------------------------------
# R4 — simple O-ether chains go substitutive, not skeletal oxa-replacement
#       (Blue Book P-12.1 / P-63.2.4)
# ---------------------------------------------------------------------------

def test_r4_simple_ethers_substitutive():
    """P-12.1 / P-63.2.4: single/dual embedded-O ethers with no principal
    characteristic group suffix are named substitutively (alkoxy prefix),
    NOT by skeletal 'oxa' replacement.
    - 1-ethoxypropane: single embedded O in 6-atom chain (was 3-oxahexane).
    - 1,2-dimethoxyethane: 2 homogeneous O, no -ol suffix (was 2,5-dioxahexane).
    REGRESSION GUARD: 3,6-dioxaoctan-1-ol carries a principal characteristic
    group (-ol) that anchors the replacement parent -> must stay skeletal.
    """
    assert name_compound("CCOCCC", style="pin") == "1-ethoxypropane"      # was 3-oxahexane
    assert name_compound("COCCOC", style="pin") == "1,2-dimethoxyethane"  # was 2,5-dioxahexane
    # REGRESSION GUARD: genuine replacement chain (terminal -ol) must NOT change:
    assert name_compound("OCCOCCOCC", style="pin") == "3,6-dioxaoctan-1-ol"


# ---------------------------------------------------------------------------
# C5 — branched alkoxy substituents (P-63.2.3.2 / P-14.5.2)
# ---------------------------------------------------------------------------

def test_c5_branched_alkoxy_substitutive():
    """P-63.2.3.2 + P-14.5.2: branched alkoxy groups (isopropyl, sec-butyl, etc.)
    must use (alkan-n-yl)oxy form with enclosing marks, not the retained
    n-alkyl names (propoxy/butoxy).  The decomposition path lacks per-atom
    locant info and must be vetoed for pure-alkyl ethers where both sides
    have >= 3 heavy atoms — these must be routed to the GENERAL substitutive path.

    OPSIN-verified expected names:
      CC(C)OCC         -> 2-ethoxypropane          (isopropyl side < 3C still GENERAL)
      CCC(C)OC         -> 2-methoxybutane           (sec-butyl side < 3C still GENERAL)
      C(C)(C)(C)OC     -> 2-methoxy-2-methylpropane (t-butyl side < 3C still GENERAL)
      CC(C)OC(C)C      -> 2-(propan-2-yloxy)propane  (both sides branched 3C — NEW FIX)
      CC(C)OCCC        -> 1-(propan-2-yloxy)propane  (isopropyl vs propyl  — NEW FIX)
      CCCCOC(C)C       -> 1-(propan-2-yloxy)butane   (butyl vs isopropyl   — NEW FIX)

    Linear controls (must stay correct — these pass through GENERAL path already):
      CCCOCCC          -> 1-propoxypropane
      CCC(C)OCCC       -> 2-propoxybutane
      CCCCOCCCC        -> 1-butoxybutane
    """
    # ALREADY PASSING (isopropyl/sec-butyl/t-butyl with methoxy/ethoxy — one side < 3C):
    assert name_compound("CC(C)OCC", style="pin") == "2-ethoxypropane"
    assert name_compound("CCC(C)OC", style="pin") == "2-methoxybutane"
    assert name_compound("C(C)(C)(C)OC", style="pin") == "2-methoxy-2-methylpropane"

    # C5 NEW FIXES — both sides >= 3 heavy atoms, one/both branched:
    assert name_compound("CC(C)OC(C)C", style="pin") == "2-(propan-2-yloxy)propane"
    assert name_compound("CC(C)OCCC", style="pin") == "1-(propan-2-yloxy)propane"
    assert name_compound("CCCCOC(C)C", style="pin") == "1-(propan-2-yloxy)butane"

    # LINEAR CONTROLS — must still pass:
    assert name_compound("CCCOCCC", style="pin") == "1-propoxypropane"
    assert name_compound("CCC(C)OCCC", style="pin") == "2-propoxybutane"
    assert name_compound("CCCCOCCCC", style="pin") == "1-butoxybutane"
