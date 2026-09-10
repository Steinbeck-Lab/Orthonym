""" a phase — P-67 oxoacids + shared functional-replacement (FRN) engine.

Covers:
  * the shared FRN name-builder (``rules.functional_replacement``)
  * organyl-stem phosphonic acids (``rules.phosphorus.name_phosphonic_acid`` +
    the ``phosphonic_acid`` inner handler) — P-67.1.1.2
  * acid-halide / amide functional-class derivatives + carbonic-family FRN acids
    tabled in ``rules.inorganic_acids`` — P-67.1.2 / P-65.2
  * Se/Te suffix acids (P-65.3) end-to-end
All end-to-end expectations are OPSIN-round-trip-confirmed PINs.
"""
import pytest
from rdkit import Chem

from orthonym.rules.functional_replacement import (
    build_acyl_halide_name,
    build_frn_acid_name,
    build_p_frn_acid_stem_word,
    build_p_frn_amide_name,
    build_p_frn_halide_name,
    build_polyacid_name,
)
from orthonym.rules.inorganic_acids import name_inorganic_acid


# --- the shared FRN engine (pure string builder) -----------------------------

@pytest.mark.parametrize("base,infix,count,expected", [
    ("carbon", "peroxo", 1, "carbonoperoxoic acid"),
    ("carbon", "thio", 2, "carbonodithioic acid"),
    ("carbon", "thio", 3, "carbonotrithioic acid"),
    ("carbon", "imido", 1, "carbonimidic acid"),
    ("carbam", "imido", 1, "carbamimidic acid"),
    ("carbam", "thio", 1, "carbamothioic acid"),
    ("phosphor", "thio", 1, "phosphorothioic acid"),
    ("sulfur", "thio", 1, "sulfurothioic acid"),
])
def test_build_frn_acid_name(base, infix, count, expected):
    assert build_frn_acid_name(base, infix, count) == expected


def test_build_frn_acid_name_unknown_infix_fail_closed():
    assert build_frn_acid_name("carbon", "bogus", 1) is None


def test_build_polyacid_name():
    assert build_polyacid_name("carbonic acid", 2) == "dicarbonic acid"
    assert build_polyacid_name("carbonic acid", 3) == "tricarbonic acid"
    assert build_polyacid_name("carbonic acid", 1) is None  # single -> not multiplicative


def test_build_acyl_halide_name():
    assert build_acyl_halide_name("phosphoryl", "chloride", 3) == "phosphoryl trichloride"
    assert build_acyl_halide_name("sulfuryl", "chloride", 2) == "sulfuryl dichloride"
    assert build_acyl_halide_name("phosphorothioyl", "chloride", 3) == "phosphorothioyl trichloride"


# --- a phase: SHARED halide/amide builders (a phase consumes these) -------

def test_build_p_frn_acid_stem_word():
    # the acid name minus " acid" — reused by the halide/amide builders (P-67.1.2.5/.6)
    assert build_p_frn_acid_stem_word("N,N-dimethyl", "phosphor", {"amido": 1}) \
        == "N,N-dimethylphosphoramidic"
    assert build_p_frn_acid_stem_word("", "phosphor", {"chlorido": 1}) \
        == "phosphorochloridic"


def test_build_p_frn_halide_name():
    # P-67.1.2.5.1: identical halides multiplied, different ones cited in seniority order
    assert build_p_frn_halide_name("phenylphosphonous", {"chloride": 2}) \
        == "phenylphosphonous dichloride"
    assert build_p_frn_halide_name("phenylphosphonous", {"bromide": 1, "chloride": 1}) \
        == "phenylphosphonous bromide chloride"          # BB L35718
    assert build_p_frn_halide_name("N,N-dimethylphosphoramidic", {"chloride": 2}) \
        == "N,N-dimethylphosphoramidic dichloride"
    assert build_p_frn_halide_name("x", {"bogus": 1}) is None   # unknown class -> None


def test_build_p_frn_amide_name():
    # P-67.1.2.6.1: 'acid' -> 'amide'/'hydrazide', multiplied
    assert build_p_frn_amide_name("N,N,P,P-tetramethylphosphinic") \
        == "N,N,P,P-tetramethylphosphinic amide"
    assert build_p_frn_amide_name("P-phenylphosphonic", "amide", 2) \
        == "P-phenylphosphonic diamide"
    assert build_p_frn_amide_name("x", "bogus") is None         # unknown class -> None


# --- a phase: SHARED P-67.1.4.1.1 acyl-prefix table (a phase consumes it) --

def test_acyl_prefix_table_p67_1_4_1_1():
    from orthonym.rules.functional_replacement import acyl_prefix_for
    # =O fundamental acyl groups (BB L36044-36050)
    assert acyl_prefix_for("P", "oxo", 0) == "phosphoryl"
    assert acyl_prefix_for("P", "oxo", 1) == "phosphonoyl"
    assert acyl_prefix_for("P", "oxo", 2) == "phosphinoyl"
    assert acyl_prefix_for("As", "oxo", 0) == "arsoryl"
    assert acyl_prefix_for("Sb", "oxo", 0) == "stiboryl"
    # FRN-modified acyl groups (BB L36132/L36150/L36162)
    assert acyl_prefix_for("P", "thio", 0) == "phosphorothioyl"
    assert acyl_prefix_for("P", "thio", 2) == "phosphinothioyl"
    assert acyl_prefix_for("P", "imido", 2) == "phosphinimidoyl"
    assert acyl_prefix_for("P", "nitrido", 0) == "phosphoronitridoyl"
    assert acyl_prefix_for("As", "imido", 0) == "arsorimidoyl"
    assert acyl_prefix_for("P", "oxo", 9) is None               # untabled -> None


# --- inorganic_acids table (acid halides / amides / carbonic-FRN) -------------

@pytest.mark.parametrize("smiles,expected", [
    # free acids (regression guard — pre-existing rows)
    ("OP(=O)(O)O", "phosphoric acid"),
    ("OC(=O)O", "carbonic acid"),
    # acid halides
    ("ClP(=O)(Cl)Cl", "phosphoryl trichloride"),
    ("ClS(=O)(=O)Cl", "sulfuryl dichloride"),
    ("ClP(=S)(Cl)Cl", "phosphorothioyl trichloride"),
    # amides
    ("NP(=O)(N)N", "phosphoric triamide"),
    ("NS(=O)(=O)N", "sulfuric diamide"),
    ("NS(=O)(=O)O", "sulfamic acid"),
    # carbonic-family FRN
    ("OOC(=O)O", "carbonoperoxoic acid"),
    ("SC(=O)S", "carbonodithioic acid"),
    ("SC(=S)S", "carbonotrithioic acid"),
    ("N=C(O)O", "carbonimidic acid"),
    ("N=C(N)O", "carbamimidic acid"),
    ("OC(=O)OC(=O)O", "dicarbonic acid"),
])
def test_name_inorganic_acid_table(smiles, expected):
    assert name_inorganic_acid(Chem.MolFromSmiles(smiles)) == expected


def test_name_inorganic_acid_fail_closed():
    # an organyl phosphonic acid is NOT a free inorganic acid -> None (cascades)
    assert name_inorganic_acid(Chem.MolFromSmiles("CCP(=O)(O)O")) is None
    # a phosphate ester likewise
    assert name_inorganic_acid(Chem.MolFromSmiles("COP(=O)(O)O")) is None


# --- a phase A.0: mononuclear preselected free acids (P-67.1.1.1) ---------
# P-67.1.1.1 "Names of mononuclear noncarbon oxoacids": "Preselected names (see
# P-12.2) of the mononuclear noncarbon oxoacids used for deriving preferred IUPAC
# names... are noted in the following list." Every expected name below is
# verbatim "(preselected name)" in that P-67.1.1.1 list — the CLOSED preselected
# table. Each SMILES is the Blue Book free-acid formula (as-written); the table
# lookup recomputes the RDKit canonical key, so any spelling resolves. Every name
# is OPSIN-round-trip-confirmed (full InChIKey == input structure).
@pytest.mark.parametrize("smiles,expected", [
    # arsenic (As)
    ("O=[AsH2]O", "arsinic acid"),          # H2As(O)(OH)
    ("O[AsH2]", "arsinous acid"),           # H2As(OH)
    ("O=[AsH](O)O", "arsonic acid"),        # HAs(O)(OH)2
    ("O[AsH]O", "arsonous acid"),           # HAs(OH)2
    ("O=[As](O)(O)O", "arsoric acid"),      # As(O)(OH)3 (preferred to 'arsenic acid')
    ("O[As](O)O", "arsorous acid"),         # As(OH)3
    # nitrogen (N) oxoacids
    ("[O-][NH2+]O", "azinic acid"),         # H2N(O)(OH)
    ("[O-][NH+](O)O", "azonic acid"),       # HN(O)(OH)2
    ("ONO", "azonous acid"),                # HN(OH)2
    ("ON(O)O", "azorous acid"),             # N(OH)3
    ("[O-][N+](O)(O)O", "nitroric acid"),   # N(O)(OH)3
    ("O=NO", "nitrous acid"),               # HO-NO
    # fluorine (F)
    ("OF", "hypofluorous acid"),            # F(OH)
    # phosphorus (P)
    ("O=[PH2]O", "phosphinic acid"),        # H2P(O)(OH)
    ("OP", "phosphinous acid"),             # H2P(OH)
    ("O=[PH](O)O", "phosphonic acid"),      # HP(O)(OH)2
    ("OPO", "phosphonous acid"),            # HP(OH)2
    ("OP(O)O", "phosphorous acid"),         # P(OH)3
    # selenium (Se) / tellurium (Te)
    ("O=[Se](=O)(O)O", "selenic acid"),     # Se(O)2(OH)2
    ("O=[Se](O)O", "selenous acid"),        # Se(O)(OH)2
    ("O=[Te](=O)(O)O", "telluric acid"),    # Te(O)2(OH)2
    # antimony (Sb)
    ("[O]=[SbH2][OH]", "stibinic acid"),    # H2Sb(O)(OH)
    ("[OH][SbH2]", "stibinous acid"),       # H2Sb(OH)
    ("[O]=[SbH]([OH])[OH]", "stibonic acid"),  # HSb(O)(OH)2
    ("[OH][SbH][OH]", "stibonous acid"),    # HSb(OH)2
    ("[O]=[Sb]([OH])([OH])[OH]", "stiboric acid"),  # Sb(O)(OH)3 (preferred to 'antimonic acid')
    ("[OH][Sb]([OH])[OH]", "stiborous acid"),  # Sb(OH)3
])
def test_mononuclear_preselected_free_acids_p67_1_1_1(smiles, expected):
    # P-67.1.1.1 CLOSED preselected free-acid table (a phase A.0).
    assert name_inorganic_acid(Chem.MolFromSmiles(smiles)) == expected


def test_mononuclear_preselected_free_acids_no_false_positives():
    # Exact full-molecule canonical-SMILES keys => a substituted / near-miss
    # structure must NOT match a tabled parent name (P-67.1.1.1; the substituted
    # organyl acids are P-67.1.1.2 producer work, not this table).
    # CH3-P(O)(OH)2 is methylphosphonic acid, NOT 'phosphonic acid';
    # (CH3)2As(O)(OH) is dimethylarsinic acid, NOT 'arsinic acid'.
    assert name_inorganic_acid(Chem.MolFromSmiles("CP(=O)(O)O")) is None
    assert name_inorganic_acid(Chem.MolFromSmiles("C[As](C)(=O)O")) is None
    # hydroxylamine (H2N-OH) is a functional parent (P-68.3.1.1.1), never an
    # oxoacid table row.
    assert name_inorganic_acid(Chem.MolFromSmiles("NO")) is None


# --- end-to-end production names ---------------------------------------------

@pytest.fixture(scope="module")
def namer():
    from orthonym.namer import Orthonym
    return Orthonym()


@pytest.mark.parametrize("smiles,expected", [
    # 9a organyl-stem phosphonic
    ("CP(=O)(O)O", "methylphosphonic acid"),
    ("CCP(=O)(O)O", "ethylphosphonic acid"),
    ("CCCP(=O)(O)O", "propylphosphonic acid"),
    ("OP(=O)(O)c1ccccc1", "phenylphosphonic acid"),
    # phosphinic must NOT regress
    ("CCP(=O)(O)CC", "diethylphosphinic acid"),
    # 9b
    ("ClP(=O)(Cl)Cl", "phosphoryl trichloride"),
    ("NP(=O)(N)N", "phosphoric triamide"),
    ("NS(=O)(=O)N", "sulfuric diamide"),
    # 9c
    ("OOC(=O)O", "carbonoperoxoic acid"),
    ("SC(=S)S", "carbonotrithioic acid"),
    ("OC(=O)OC(=O)O", "dicarbonic acid"),
    # 9d Se/Te suffix
    ("CC[Se](=O)(=O)O", "ethaneselenonic acid"),
    ("CC[Te](=O)O", "ethanetellurinic acid"),
    ("OC(=O)CC[Se](=O)(=O)O", "3-selenonopropanoic acid"),
    # sulfonic must NOT regress
    ("CCS(=O)(=O)O", "ethanesulfonic acid"),
])
def test_production_names(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --- a phase: trivalent -ous organo-oxoacid producers (P-67.1.1.2) --------
# BB P-67.1.1.2 "Mononuclear oxoacids of phosphorus, arsenic, antimony": the
# central-atom H of the preselected -ous acids (phosphonous HP(OH)2, phosphinous
# H2P(OH), and As/Sb analogues) is substituted by an organyl group, giving the
# substituent-prefix PIN. BB L35465 verbatim (C6H5)2As(OH) -> "diphenylarsinous
# acid"; BB L35467 verbatim C6H5-Sb(OH)2 -> "phenylstibonous acid". Before this
# phase these round-tripped as the wrong substitutive form
# ((hydroxy(phenyl)arsanyl)benzene) or abstained (phosphorus).
@pytest.mark.parametrize("smiles,expected", [
    ("OP(O)c1ccccc1", "phenylphosphonous acid"),        # C6H5-P(OH)2 (P-67.1.1.2)
    ("OP(O)C", "methylphosphonous acid"),               # CH3-P(OH)2
    ("OP(c1ccccc1)c1ccccc1", "diphenylphosphinous acid"),  # (C6H5)2P-OH
    ("O[As](c1ccccc1)c1ccccc1", "diphenylarsinous acid"),  # (C6H5)2As(OH) BB L35465
    ("O[As](O)c1ccccc1", "phenylarsonous acid"),        # C6H5-As(OH)2
    ("O[Sb](O)c1ccccc1", "phenylstibonous acid"),       # C6H5-Sb(OH)2 BB L35467
    ("O[Sb](c1ccccc1)c1ccccc1", "diphenylstibinous acid"),  # (C6H5)2Sb-OH
])
def test_ous_organo_oxoacid_producers_p67_1_1_2(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_ous_free_parents_not_stolen_by_organo_producers(namer):
    """The bare -ous parents (no C on the central atom) stay tabled free acids,
    never mis-claimed by the organo -ous SMARTS (P-67.1.1.1 carbon guard)."""
    # HP(OH)2 / P(OH)3 have no C-P bond -> the organo -onous SMARTS must not fire.
    assert namer.name("OPO") == "phosphonous acid"       # HP(OH)2 (free parent, tabled)
    assert namer.name("OP(O)O") == "phosphorous acid"    # P(OH)3 (free parent, tabled)


# --- a phase: class-infix FRN generalized to As/Sb + P-67.1.2.4.2 guard ---
# P-67.1.2.4.2 "Name construction guidelines for functional replacement
# nomenclature": the -onic/-inic stems can only be used when the central atom
# bonds to H, C, or a parent-hydride atom, so a halogen/N is a CLASS infix, never
# a skeletal substituent. Hence ClP(O)(OH)2 -> phosphorochloridic acid, never
# chlorophosphonic acid (BB L35702 verbatim). Generalized to As/Sb by the
# _FRN_CENTRAL_STEMS table; build_p_frn_acid_name is unchanged.
@pytest.mark.parametrize("smiles,expected", [
    ("ClP(=O)(O)O", "phosphorochloridic acid"),        # BB L35702 GUARD (P)
    ("Cl[As](=O)(O)O", "arsorochloridic acid"),        # BB L35702 GUARD (As)
    ("Cl[Sb](=O)(O)O", "stiborochloridic acid"),       # BB L35702 GUARD (Sb)
    ("C[As](=O)(Cl)O", "methylarsonochloridic acid"),  # 1 organyl -> -ono stem
    ("C[As](=O)(OC#N)O", "methylarsonocyanatidic acid"),  # P-67.1.2.4.1.3
])
def test_asb_class_infix_frn_and_guard_p67_1_2_4_2(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --- a phase: acid HALIDE / AMIDE producers (P-67.1.2.5 / P-67.1.2.6) ------
# P-67.1.2.5.1 "Acid halides and pseudohalides": the class word is added to the
# acid name (except the phosphoryl/sulfuryl acyl-word cases). BB L35718 verbatim
# C6H5-PBrCl -> "phenylphosphonous bromide chloride"; BB L35712 (C6H5)2P-Cl ->
# "diphenylphosphinous chloride". P-67.1.2.6.1 "Amides and hydrazides": 'acid' ->
# 'amide'. The oh_count==0 shape; every candidate stays OPSIN-round-trip gated.
@pytest.mark.parametrize("smiles,expected", [
    ("ClP(Cl)c1ccccc1", "phenylphosphonous dichloride"),        # C6H5-PCl2 BB L35716
    ("ClP(Br)c1ccccc1", "phenylphosphonous bromide chloride"),  # C6H5-PBrCl BB L35718
    ("ClP(c1ccccc1)c1ccccc1", "diphenylphosphinous chloride"),  # (C6H5)2P-Cl BB L35712
    ("CN(C)P(=O)(Cl)Cl", "N,N-dimethylphosphoramidic dichloride"),  # amido-FRN halide
    ("ClP(=O)(Cl)c1ccccc1", "phenylphosphonic dichloride"),     # C6H5-P(O)Cl2 BB L35726
    ("NP(=O)(N)c1ccccc1", "phenylphosphonic diamide"),          # P-67.1.2.6.1 amide
    ("NP(N)c1ccccc1", "phenylphosphonous diamide"),             # trivalent -ous amide
])
def test_acid_halide_amide_producers_p67_1_2_5_6(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_bare_trivalent_hydride_amide_is_substitutive_not_frn(namer):
    """ B1 regression — the FRN amide producer must DECLINE the *bare*
    mononuclear trivalent hydride amide so the substitutive amine name (the PIN)
    is built. H2P-NH2 is ``phosphanamine`` (P-41: the amine characteristic group
    is expressed as a suffix on the preselected parent hydride phosphane), NOT the
    functional-class ``phosphinous amide``. This mirrors P-67.1.2.6.2's boron/
    silicon treatment (H2B-NH2 -> boranamine, *not* borinic amide): when the acid
    parent is a preselected mononuclear hydride with no oxo (oxo == 0) and no
    organyl substituent, the amide is named substitutively. Commit 9aedffd6f's
    shared -ous amide builder over-reached onto this shape and regressed the PIN.
    The organyl/oxo FRN amide cases (test above: phenylphosphonous/phosphonic
    diamide) keep the functional-class name."""
    assert namer.name("[PH2]N") == "phosphanamine"


@pytest.mark.parametrize("smiles,expected", [
    ("[PH2]N", "phosphanamine"),
    ("[AsH2]N", "arsanamine"),
    ("[SbH2]N", "stibanamine"),
    ("[BiH2]N", "bismuthanamine"),
])
def test_pnictogen_hydride_amine_is_substitutive_pin(namer, smiles, expected):
    """ B1a — the bare H2E-NH2 amine of every preselected Group-15 parent
    hydride (E = P/As/Sb/Bi) is named substitutively on the pnictogen skeleton
    with the senior N as the '-amine' suffix (P-41 seniority; arsane/stibane/
    bismuthane are preselected parent hydrides, P-21.1.1). B1 fixed
    P but left As/Sb/Bi abstaining ('arsenic compound (not supported)'); each now
    reaches its PIN and round-trips through OPSIN 2.9.0 to the input structure."""
    assert namer.name(smiles) == expected


def test_phosphoryl_acyl_word_exception_not_stolen(namer):
    """POCl3 is the P-67.1.2.5.1 acyl-word exception ('phosphoryl trichloride'),
    NOT 'phosphoric trichloride' — the halide perceiver must fail closed on the
    bare skeletal-0 no-infix shape so the existing acyl path keeps it."""
    assert namer.name("ClP(=O)(Cl)Cl") == "phosphoryl trichloride"


# --- a phase: acyl-prefix BRIDGE consumes the shared table (P-67.1.4.1.1.5) -
# The multiplicative -E(=O)(OH)- bridge linking two benzoic acids now looks up the
# acyl base ('phosphoryl'/'arsoryl') from ACYL_PREFIX_TABLE instead of a hardcoded
# string, and the compound-prefix enclosing generalises to every acyl sibling.
@pytest.mark.parametrize("smiles,expected", [
    # byte-identical: -P(=O)(OH)- bridge (BB P-67.1.4.1.1.5)
    ("OC(=O)c1ccc(cc1)P(=O)(O)c1ccc(cc1)C(=O)O",
     "4,4'-(hydroxyphosphoryl)dibenzoic acid"),
    # generalised: -As(=O)(OH)- bridge -> (hydroxyarsoryl), now correctly enclosed
    ("OC(=O)c1ccc(cc1)[As](=O)(O)c1ccc(cc1)C(=O)O",
     "4,4'-(hydroxyarsoryl)dibenzoic acid"),
])
def test_acyl_prefix_bridge_consumes_table(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --- a phase: chalcogen-infix FRN, locant-free di+ subset (P-67.1.2.4.1.1) -
# P-67.1.2.4.1.1 lists As(O)(OH)(SH)2 -> "arsorodithioic acid" (BB L35636) with NO
# italic tautomer locant (the position is undetermined under mobile H). The build
# fires ONLY on the locant-free di+ same-chalcogen case; every emission passes the
# InChI round-trip gate (tautomers collapse). Mono / mixed / =S cases fail closed
# (they need a phase Group A's P-65.1.5.1 tautomer-locant derivation).
@pytest.mark.parametrize("smiles,expected", [
    ("O=[As](O)(S)S", "arsorodithioic acid"),           # As(O)(OH)(SH)2 BB L35636
    ("O=P(O)(S)S", "phosphorodithioic acid"),           # P(O)(OH)(SH)2
    ("O=P(S)(S)S", "phosphorotrithioic acid"),          # P(O)(SH)3
    ("O=P(S)(S)c1ccccc1", "phenylphosphonodithioic acid"),  # C6H5-P(O)(SH)2
])
def test_chalcogen_infix_frn_locantfree_p67_1_2_4_1_1(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --- a phase: FRN-modified phosphorus ESTER (P-67.1.2.4) -------------------
# An ester of a functional-replacement-modified P acid: -ate for the pentavalent
# (P=O) acid, -ite for the trivalent one. BB P-67.1.2.5 region: the target
# CH3-O-P(Cl)-N(CH3)2 is "methyl N,N-dimethylphosphoramidochloridite". A separate,
# non-overlapping sibling of name_phosphate_ester (which stays byte-identical).
@pytest.mark.parametrize("smiles,expected", [
    ("CN(C)P(Cl)OC", "methyl N,N-dimethylphosphoramidochloridite"),   # trivalent, -ite
    ("COP(=O)(Cl)N(C)C", "methyl N,N-dimethylphosphoramidochloridate"),  # pentavalent, -ate
    ("NP(Cl)(=O)OC", "methyl phosphoramidochloridate"),              # bare amido + chlorido
    ("CCOP(OCC)N(C)C", "diethyl N,N-dimethylphosphoramidite"),       # 2 owners + amido
])
def test_p_frn_ester_producers_p67_1_2_4(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_plain_phosphate_phosphite_esters_not_regressed(namer):
    """Plain esters have no class infix -> name_p_frn_ester defers, the existing
    name_phosphate_ester path keeps them byte-identical."""
    assert namer.name("COP(OC)OC") == "trimethyl phosphite"
    assert namer.name("COP(=O)(OC)OC") == "trimethyl phosphate"


def test_chalcogen_infix_frn_locant_cases_fail_closed():
    """MONO / mixed-chalcogen cases need a tautomer locant (a phase P-65.1.5) —
    the locant-free chalcogen FRN must NOT emit a wrong-spelling name for them."""
    from orthonym.rules.inorganic_acids import name_p_oxoacid_chalcogen_frn
    assert name_p_oxoacid_chalcogen_frn(Chem.MolFromSmiles("O=[As](O)(O)S")) is None  # mono
    assert name_p_oxoacid_chalcogen_frn(Chem.MolFromSmiles("O=P(O)(S)[SeH]")) is None  # mixed
    assert name_p_oxoacid_chalcogen_frn(Chem.MolFromSmiles("O=[As](O)(O)O")) is None   # no chalcogen


# --- a phase B.1: P-67.2.1 di-/tri-nuclear PRESELECTED free acids ----------
# P-67.2.1 "Preselected names": the CLOSED traditional-name table for the di- and
# poly-nuclear noncarbon oxoacids. Each is verbatim "(preselected name)" in that
# list, so the PIN is a table lookup. Keys are the BB PIN's own OPSIN-parse
# canonical SMILES (the bb_conformance oracle key), so every name round-trips
# through OPSIN to its exact key by construction. Before B.1 each of these
# abstained ("inorganic/arsenic/antimony compound (not supported)").
@pytest.mark.parametrize("smiles,expected", [
    ("OB(O)OB(O)O", "diboric acid"),                          # (HO)2B-O-B(OH)2
    ("O[Si](O)(O)O[Si](O)(O)O", "disilicic acid"),            # (HO)3Si-O-Si(OH)3
    ("O=[PH](O)O[PH](=O)O", "diphosphonic acid"),             # (HO)HP(O)-O-HP(O)(OH)
    ("OPOPO", "diphosphonous acid"),                          # HO-PH-O-PH-OH
    ("OP(O)OP(O)O", "diphosphorous acid"),                    # (HO)2P-O-P(OH)2
    ("O=[As](O)(O)O[As](=O)(O)O", "diarsoric acid"),          # (HO)2As(O)-O-As(O)(OH)2
    ("O[As](O)O[As](O)O", "diarsorous acid"),                 # (HO)2As-O-As(OH)2
    ("[O]=[Sb]([OH])([OH])[O][Sb](=[O])([OH])[OH]", "distiboric acid"),  # (HO)2Sb(O)-O-Sb(O)(OH)2
    ("[OH][Sb]([OH])[O][Sb]([OH])[OH]", "distiborous acid"),  # (HO)2Sb-O-Sb(OH)2
    ("O=S(=O)(O)S(=O)(=O)O", "dithionic acid"),               # HO-SO2-SO2-OH
    ("O=S(O)S(=O)O", "dithionous acid"),                      # HO-SO-SO-OH
    ("O=[PH](O)OP(=O)(O)O[PH](=O)O", "triphosphonic acid"),   # (HO)HP(O)-O-HP(O)-O-HP(O)(OH)
    ("O=P(O)(O)OP(=O)(O)OP(=O)(O)O", "triphosphoric acid"),   # (HO)2P(O)-O-P(O)(OH)-O-P(O)(OH)2
    ("O=S(=O)(O)OS(=O)(=O)OS(=O)(=O)O", "trisulfuric acid"),  # HO-SO2-O-SO2-O-SO2-OH
])
def test_p67_2_1_dinuclear_preselected_free_acids(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --- a phase B.2: di-nuclear DERIVATIVE producer element-set extension ------
# P-67.2.3/.2.4: functional-class derivatives of the P-67.2.1 di-acids. B.2 adds
# the S-S direct 'dithionic' family and the pentavalent As/Sb '-oric' families to
# _POLYACID_PARENT / _POLYACID_OXO, so their derivatives build through the same
# fail-closed, OPSIN-RT-gated producer as diphosphoric tetrachloride. Every
# emission below OPSIN-round-trips to its input structure.
@pytest.mark.parametrize("smiles,expected", [
    ("O=[As](Cl)(Cl)O[As](=O)(Cl)Cl", "diarsoric tetrachloride"),   # P-67.2.3
    ("[O]=[Sb](Cl)(Cl)[O][Sb](=[O])(Cl)Cl", "distiboric tetrachloride"),
    ("O=S(=O)(Cl)S(=O)(=O)Cl", "dithionic dichloride"),
    ("N[As](=O)(N)O[As](=O)(N)N", "diarsoric tetraamide"),          # P-67.2.4
    ("O=[As](O)(O)S[As](=O)(O)O", "2-thiodiarsoric acid"),          # P-67.2.2.2 bridge-thio
])
def test_p67_2_dinuclear_derivative_element_set_b2(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_p67_2_new_element_free_acids_stay_tabled_not_producer():
    """The new-element FREE di-acids must be matched by the _INORGANIC_OXOACIDS
    table (B.1), never routed through the derivative producer (B.2) — the table
    lookup is first in name_inorganic_acid, and the producer runs last."""
    from orthonym.rules.inorganic_acids import name_polyacid_derivative
    for smi in ("O=[As](O)(O)O[As](=O)(O)O",                         # diarsoric acid
                "[O]=[Sb]([OH])([OH])[O][Sb](=[O])([OH])[OH]",       # distiboric acid
                "O=S(=O)(O)S(=O)(=O)O"):                             # dithionic acid
        # the producer alone would build a bare parent name off these, but it is
        # never reached because the table matches first; assert the table wins.
        assert name_inorganic_acid(Chem.MolFromSmiles(smi)) in (
            "diarsoric acid", "distiboric acid", "dithionic acid")


# --- a phase Group B: carbonic / poly-carbonic FRN acids -------------------
# P-65.2.1.2/.1.3 (mononuclear) + P-65.2.3.1.2-.4 (di/tri/tetra-carbonic). The
# mononuclear multi-infix builder is pure string; the poly perceiver is a carbon-
# backbone analogue of name_polyacid_derivative. Every end-to-end expectation is a
# verbatim BB (PIN) confirmed to OPSIN-round-trip to its input structure (2.9.0).

@pytest.mark.parametrize("stem,infixes,expected", [
    ("carbam", {"peroxo": 1}, "carbamoperoxoic acid"),          # BB 30784
    ("carbon", {"peroxo": 2}, "carbonodiperoxoic acid"),        # BB 30788
    ("carbon", {"hydrazono": 1}, "carbonohydrazonic acid"),     # BB 30804
    ("carbon", {"imido": 1, "thio": 1}, "carbonimidothioic acid"),        # BB 30806
    ("carbam", {"imido": 1, "seleno": 1}, "carbamimidoselenoic acid"),    # BB 30828
    ("carbon", {"hydrazono": 1, "seleno": 2}, "carbonohydrazonodiselenoic acid"),  # BB 30810
    ("carbon", {"imido": 1}, "carbonimidic acid"),              # existing single-infix parity
    ("carbon", {"thio": 2}, "carbonodithioic acid"),
])
def test_build_carbonic_mono_frn(stem, infixes, expected):
    from orthonym.rules.functional_replacement import build_carbonic_mono_frn
    assert build_carbonic_mono_frn(stem, infixes) == expected


def test_build_carbonic_mono_frn_fail_closed():
    from orthonym.rules.functional_replacement import build_carbonic_mono_frn
    assert build_carbonic_mono_frn("carbon", {"bogus": 1}) is None
    assert build_carbonic_mono_frn("phosphor", {"thio": 1}) is None   # wrong stem
    assert build_carbonic_mono_frn("carbon", {}) is None


@pytest.mark.parametrize("smiles,expected", [
    # mononuclear (P-65.2.1.2/.1.3)
    ("NC(=O)OO", "carbamoperoxoic acid"),
    ("O=C(OO)OO", "carbonodiperoxoic acid"),
    ("NN=C(O)O", "carbonohydrazonic acid"),
    ("N=C(O)S", "carbonimidothioic acid"),
    ("NN=C([SeH])[SeH]", "carbonohydrazonodiselenoic acid"),
    ("N=C(N)[SeH]", "carbamimidoselenoic acid"),   # P-66.1.6.1.3.2: not isoselenourea
    # di/tri/tetra-carbonic (P-65.2.3.1.2)
    ("O=C(O)NC(=O)O", "2-imidodicarbonic acid"),
    ("N=C(O)OC(=O)O", "1-imidodicarbonic acid"),
    ("N=C(O)OC(=N)O", "1,3-diimidodicarbonic acid"),
    ("N=C(O)NC(=N)NC(=N)O", "1,2,3,4,5-pentaimidotricarbonic acid"),
    ("N=C(O)NC(=N)NC(=N)NC(=N)O", "1,2,3,4,5,6,7-heptaimidotetracarbonic acid"),
    ("O=C(O)OOC(=O)O", "2-peroxydicarbonic acid"),
    ("O=C(O)OC(=O)OO", "1-peroxydicarbonic acid"),
    ("O=C(OO)OC(=O)OO", "1,3-diperoxydicarbonic acid"),
    ("S=C(S)OC(=S)S", "1,1,3,3-tetrathiodicarbonic acid"),
    ("S=C(S)SC(=S)S", "pentathiodicarbonic acid"),
    ("OC(=S)OC(O)=S", "1,3-dithiodicarbonic acid"),   # P-65.2.3.1.2.2 (acyl=S tautomer)
    # terminal halide / amido / hydrazido (P-65.2.3.1.3/.4)
    ("O=C(O)OC(=O)Cl", "chlorodicarbonic acid"),
    ("NC(=O)SC(=O)O", "1-amido-2-thiodicarbonic acid"),
    ("NNC(=O)NC(=O)NC(=O)O", "1-hydrazido-2,4-diimidotricarbonic acid"),
])
def test_carbonic_frn_family_end_to_end(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_carbonic_frn_determined_S_acid_fails_closed():
    """P-65.2.3.1.2.3: the DETERMINED '=O + -SH' carbon (S-acid) would round-trip
    to the OTHER tautomer under the nonspecific 'dithio' spelling, so the perceiver
    must NOT emit a poly-carbonic FRN name for it (degrade to systematic, never a
    wrong constitution)."""
    from orthonym.rules.inorganic_acids import name_carbonic_frn_family
    assert name_carbonic_frn_family(Chem.MolFromSmiles("O=C(S)OC(O)=S")) is None


def test_carbonic_frn_family_no_false_positives():
    """Esters, anhydrides, ureas and directly C-C-bonded diacids must fail closed."""
    from orthonym.rules.inorganic_acids import name_carbonic_frn_family
    for smi in ("CCOC(=O)OCC",        # diethyl carbonate (ester)
                "COC(=O)OC(=O)OC",    # dimethyl dicarbonate (ester)
                "CC(=O)OC(C)=O",      # acetic anhydride
                "NC(=O)N",            # urea (diamide, not an acid)
                "NC(=N)N",            # guanidine
                "OC(=O)C(=O)O",       # oxalic acid (direct C-C, no bridge)
                "CCOC(=O)N"):         # ethyl carbamate (ester)
        assert name_carbonic_frn_family(Chem.MolFromSmiles(smi)) is None


def test_carbonic_frn_requires_retained_acid_group():
    """A poly-carbonic backbone whose -OH groups are ALL replaced by halide / amido
    is an acid HALIDE / AMIDE functional class (P-65.6 seniority), owned by another
    path -- the FRN-acid perceiver must fail closed, not name it '...dicarbonic
    acid'. Regression guard for the -P2B measure losses."""
    from orthonym.rules.inorganic_acids import name_carbonic_frn_family
    for smi in ("O=C(Cl)OC(=O)Cl",   # dicarbonyl dichloride (no -OH)
                "O=C(Cl)OC(=O)Br",   # mixed dihalide
                "O=C(N)OC(N)=O",     # dicarbonic diamide
                "N=C(N)NC(N)=O"):    # N-carbamimidoylurea (all-N terminals)
        assert name_carbonic_frn_family(Chem.MolFromSmiles(smi)) is None
