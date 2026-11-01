"""Rows and the producer a trace observed for the name that ships.

Every pair is the first seam (deepest builder) that returned the shipped string, read from a trace
in a fresh process with the OPSIN validity gate on (the shipped configuration).

Rows owned by another lane (carbon-free species of the other metals, lowercase r/s centres) are not
used: a lane that merges first could move such a row to another producer. Two groups keep metal
rows: table:coordination (the list is of metal complexes) and the zinc rows of the metal lane
(``ZINC``): the lane's new producers (the metal atom class, the other drawing of a metal drawing, the
dative split) report through the seams that call them, and a pair of each pins the id.
"""
# 13C and 18O labels on an ester: the engine builds a name, OPSIN reads it as ambiguous, so the default
# tier declines it (NO_VERIFIED_PIN; tests/unit/opsin_ambiguity/test_amb7_ambiguity_gate.py)
LABELLED_ESTER = "CC1=CC=C(C=C1)[13CH](C)OC(=[18O])C2=C(C(=C(C(=C2F)F)F)F)F"
HEROIN = "CC(=O)O[C@H]1C=C[C@H]2[C@H]3CC4=C5[C@@]2([C@H]1OC5=C(C=C4)OC(C)=O)CCN3C"

# cob(I)alamin and cob(II)alamin: the exact-structure list of metal-complex names
# (tests/unit/rules/test_d1_coordination_v36.py has the table's fixtures)
COB_I = ("CC1=C2[N]3[C@@H]4[C@H](CC(N)=O)[C@@]2(C)CCC(=O)NC[C@@H](C)OP(=O)(O)O[C@H]2[C@@H](O)"
         "[C@H](O[C@@H]2CO)n2c[n+](c5cc(C)c(C)cc52)[Co-4]325[N+]3=C1[C@@H](CCC(N)=O)C(C)(C)C3=CC1="
         "[N+]2C(=C(C)C2=[N+]5[C@]4(C)[C@@](C)(CC(N)=O)[C@@H]2CCC(N)=O)[C@@](C)(CC(N)=O)[C@@H]1CCC(N)=O")
COB_II = ("CC1=C2[N]3[C@@H]4[C@H](CC(N)=O)[C@@]2(C)CCC(=O)NC[C@@H](C)OP(=O)([O-])O[C@H]2[C@@H](O)"
          "[C@H](O[C@@H]2CO)n2c[n+](c5cc(C)c(C)cc52)[Co-3]325[N+]3=C1[C@@H](CCC(N)=O)C(C)(C)C3=CC1="
          "[N+]2C(=C(C)C2=[N+]5[C@]4(C)[C@@](C)(CC(N)=O)[C@@H]2CCC(N)=O)[C@@](C)(CC(N)=O)[C@@H]1CCC(N)=O")
# a PubChem row whose name fails its round trip at the stereo layer only: the best-effort tier ships
# the floor's name with the prefixes in another order (tests/unit/rules/test_prefix_order_fallback.py).
# Witness replaced 2026-10-23: the nucleoside phosphoramidite that stood here (PubChem 500k row 493267) is named
# in the order now, since the read-back reads a lone-pair stereocentre the OpenSMILES way
# (``perception.lone_pair_reading``); PubChem CID 173762620 (a bis-phosphine whose two P(III) centres are in
# rings, a row of benchmarks/unnamed_panel) still reaches the floor, with the book spellings and without.
PREFIX_ORDER_ROW = "CC(C)(C)[P@@]1COC2=CC=CC(=C21)C3=C4C(=CC=C3)OC[P@@]4C"

# Rows whose shipped name is the string of a producer with the input's stereo composed onto it
# (Orthonym._stereo_emit_decision): the general engine and omit the descriptor of an exocyclic double
# bond between two rings, and the emit decision of the best-effort tier adds it back. The first two rows
# ship from the inline lane of _name_impl, the next two from the late recovery; the last row is composed
# by the validity gate, not by the emit decision, and keeps the label of the handler's string. The
# first two are small molecules made for the purpose (no corpus row ships from the inline lane); the
# others are ChEBI molecules. None is in a holdout split (checked by full InChIKey and by connectivity).
STEREO_COMPOSED = [
    ("CC1=CC(=O)OC/1=C2\\CCCO2", "best-effort", "engine:general"),
    ("O=C1OCC(=O)/C1=C2\\CCCO2", "best-effort", "engine:general"),
    ("O=C1Nc2ccccc2/C1=C1/Nc2ccccc2/C1=N\\O", "best-effort", "engine:t4"),
    ("O=C1N=C(O)/C(=C2\\C=C(O)C(=O)N=C2O)C=C1O", "best-effort", "engine:general"),
    ("C1O[C@H]1[C@H]1CO1", "best-effort", "inner:ring_assembly"),
]

# Rows whose shipped name is the cis/trans name of a ring pair (rules/relative_stereo.py,,
# the Blue Book "Relative configuration: the stereodescriptors 'cis' and 'trans'"). OPSIN reads no
# lowercase r/s, so at best-effort the name with r/s that a producer built for the whole molecule
#,:48103) is converted, and the row names that producer: the string the conversion
# started from was returned by the seam named here. The last two rows hold the pair in a prefix and
# two pairs in one name, where the conversion also re-derives the enclosing marks.
RELATIVE_STEREO = [
    ("O[C@H]1CC[C@@H](O)CC1", "best-effort", "inner:general_acyclic"),
    ("O[C@H]1CC[C@H](O)CC1", "best-effort", "inner:general_acyclic"),
    ("N[C@H]1CC[C@@H](C(=O)O)CC1", "best-effort", "inner:ester_family"),
    ("OC(=O)C[C@H]1CC[C@H](C)CC1", "best-effort", "inner:general_acyclic"),
    ("O[C@H]1CC[C@H](CC[C@H]2CC[C@H](O)CC2)CC1", "best-effort", "inner:general_acyclic"),
]

# The zinc rows of the metal lane (ZN). The producers it added sit under the seams of the outer classes:
# the class METAL_ATOM (the atom, an ion), the organometallic and salt classes (the drawing as written, and
# the other drawing that ``assembly/metal_offer.py`` offers: the ionic drawing of a bonded one, the bonded
# drawing of an organometallic cation beside its anion, the drawing of a dative one without its dative
# bonds), the isotope step (a labelled atom) and the multi-component recovery (an ammine). Each id was
# read from a trace of the panel ``tests/fixtures/zinc/panel_pairs.tsv`` with the gate on; a name that the
# offer finds is the name of a string that the producer of that class built, so the id is the class's.
ZINC = [
    ("[Cl-].[Cl-].[Zn+2]", "pin", "outer:salt"),                    # the strict path names the ions
    ("Cl[Zn]Cl", "best-effort", "outer:salt"),                       # the ionic drawing of a bonded one
    ("CC[Zn]CC", "best-effort", "outer:organometallic"),
    ("[Cl-].[Zn+]c1ccccc1", "best-effort", "outer:organometallic"),  # the bonded drawing of a cation beside its anion
    ("[CH3-].[CH3-].[Zn+2]", "best-effort", "outer:salt"),           # a carbanion salt
    ("[Zn]", "best-effort", "outer:metal_atom"),
    ("[Zn+2]", "best-effort", "outer:metal_atom"),
    ("[65Zn]", "best-effort", "isotope"),
    ("C1=CC2=NC1=Cc1ccc3[n]1[Zn][n]1c(ccc1=CC1=NC(=C3)C=C1)=C2", "best-effort", "outer:salt"),  # dative porphyrin
    ("[Zn+2].N.N.N.N", "best-effort", "multifragment"),               # the ammine
    ("CC[Zn]CC.CN(C)CCN(C)C", "best-effort", "outer:multi_component_neutral"),
]

PAIRS = [
    # the first dispatch of the outer classes
    ("CCO", "pin", "outer:retained_name"),
    ("c1ccccc1", "pin", "outer:retained_name"),
    ("O=C([O-])[C@@H]1CCC=N1", "pin", "outer:anion_small"),
    ("C[C@H](N)C(=O)O", "best-effort", "outer:amino_acid"),   #: the PIN tier declines L-alanine
    # PINC lane: glyceraldehyde has a systematic PIN, '(2R)-2,3-dihydroxypropanal', the Blue Book), so
    # 'O=C[C@H](O)CO' ships from inner:ester_family now; a hexopyranose still ships from the lookup, at the default tier
    #,:50943, names no PIN in the carbohydrate chapter, so the PIN label of the lookup names may go)
    ("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "best-effort", "outer:carbohydrate_lookup"),
    ("C[C@@]12C(CC[C@H]1[C@@H]1CCC3=CC(CC[C@]3(C)[C@H]1CC2)=O)=O", "pin", "outer:natural_product"),
    ("NCCOCCN", "pin", "outer:multiplicative"),
    ("S=C(S)S", "pin", "outer:inorganic_acid"),
    ("O=P(O)(Cl)c1ccccc1", "pin", "outer:inorganic_acid"),
    ("[CH3+]", "pin", "outer:cation_retained"),
    ("C[NH3+].[Cl-]", "pin", "outer:salt"),
    ("CC(=O)[O-].[NH4+]", "pin", "outer:salt"),
    # PB lane S: a tripeptide is named substitutively now method (1); gives no PIN
    # for peptide names), by the general acyclic producer
    ("NCC(=O)N[C@@H](CC(N)=O)C(=O)NCC(=O)O", "best-effort", "inner:general_acyclic"),
    # the cascade that goes on when the first class's handler returns None (NPL 2026-11-01: a name is no PIN,
    # INTRODUCTION, the Blue Book, so the PIN tier declines it and best-effort ships it)
    ("C[C@]12CC[C@H](O[C@@H]3O[C@H](C(=O)O)[C@@H](O)[C@H](O)[C@H]3O)C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12",
     "best-effort", "outer:natural_product"),
    ("C/C(=C\\[C@H]1OC(=O)[C@@H](C)[C@@H]1C)[C@H]1CCC2=C3C=CC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@@]21C",
     "best-effort", "outer:natural_product"),
    # the inner handlers
    ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "pin", "inner:tier_a_ring"),
    ("C1CC2CCC1C2", "pin", "inner:tier_a_ring"),
    ("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "pin", "inner:general_acyclic"),
    ("CC(=O)Nc1ccc(O)cc1", "pin", "inner:ester_family"),
    ("O=C1CCCO1", "pin", "inner:lactone"),
    ("c1ccc(-c2ccccc2)cc1", "pin", "inner:ring_assembly"),
    ("CC(=O)OOC(C)=O", "pin", "inner:anhydride"),
    # the composer, where no inner handler built the name
    ("N=C(N)NC=O", "pin", "composer"),
    ("CC[S+]", "pin", "composer"),
    # the decomposition engine
    ("CC(=O)N(C(=O)c1ccccc1)c1ccc2ccccc2c1", "pin", "decomposition"),
    ("CCOC(=O)C#CCC1CCCCC1", "best-effort", "decomposition"),
    # the general engine: the late recovery, the inline lane, and the engine inside T4
    ("C=C[C@@]1(O)CC(C#N)=CC1=O", "best-effort", "engine:general"),
    ("O=C(O)c1ccccc1SCCCN1C(=O)c2cccc3cccc(c23)C1=O", "best-effort", "engine:general"),
    ("FOc1ccccc1", "best-effort", "engine:general"),
    ("CC(=O)OCl", "best-effort", "engine:general"),
    ("NCC(CN)CN", "best-effort", "engine:general"),
    # the isotope decorator, which names a labelled molecule before the outer classes see it
    ("Cl[12CH](Cl)Cl", "pin", "isotope"),
    ("[2H]C([2H])([2H])O", "pin", "isotope"),
    # the retained fused name of a purine-containing molecule that is not itself a fused ring (ADP-ribose,
    # an acyl coenzyme A), which the late recovery and the inline lane try before the von Baeyer fallback
    ("Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]2O[C@@H](O)[C@H](O)[C@@H]2O)[C@@H](O)[C@H]1O",
     "best-effort", "retained_fused"),
    ("CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)"
     "NCCSC(=O)C=Cc1ccccc1", "best-effort", "retained_fused"),
    # the late recovery of the general engine names the components of a multi-component input and joins them
    ("C1CCC(C1)NN.[Cl-]", "best-effort", "multifragment"),
    ("CCOC(=O)C.C/C=C/C=C/C(=O)[O-]", "best-effort", "multifragment"),
    # the exact-match list of natural-product parents, the last resort before a decline (germacrane is
    # named by outer:natural_product at the default tier)
    ("CC(C)[C@@H]1CC[C@H](C)CCC[C@H](C)CC1", "best-effort", "table:np_parent"),
    # no organic positive: the list of metal-complex names, taken by name at its exit
    (COB_I, "pin", "table:coordination"),
    (COB_II, "pin", "table:coordination"),
    # T4: the handoff of the late recovery, the rescue, and the floor offer
    ("O=[N+]([O-])c1ccc(/C=C/c2ccc([N+](=O)[O-])cc2S(=O)(=O)O)c(S(=O)(=O)O)c1", "best-effort", "engine:t4"),
    ("COC(=O)c1ccc(OC(CO)C(O)c2ccc(O)c(OC)c2)c(OC)c1", "best-effort", "engine:t4"),
    ("C[C@@H]1CCC2C(C)(C)[C@H](O)CC[C@]2(C)[C@@]12Cc1c(O)cc3c(c1O2)CNC3=O", "best-effort", "engine:t4"),
    # the stereo of the input composed onto the string of a producer
    *STEREO_COMPOSED,
    # the zinc rows of the metal lane
    *ZINC,
]
