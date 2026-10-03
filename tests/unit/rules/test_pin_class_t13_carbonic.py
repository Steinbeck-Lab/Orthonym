"""PIN class program, Task 13: the carbonic acid family (CARBONIC, DICYANIDE, UREA_PARENT).

(A) Acyl halides and pseudohalides of cyanic acid. (the Blue Book):
    "Method (1) generates preferred IUPAC names" -- 'carbononitridic chloride (PIN)',
    'carbononitridic azide (PIN)' (:31513,:31515). ``rules/inorganic_acids.py:
    _name_cyanic_acyl_halide``.
(B) Nitriles of carbonic acid. (:34844): "Nitriles corresponding to carbonic
    acid and di- and polycarbonic acids are named by functional class nomenclature" --
    'carbonyl dicyanide (PIN)' (:34848), 'carbonimidoyl dicyanide (PIN) (not
    2-iminopropanedinitrile)' (:34854), 'carbonohydrazonoyl dicyanide (PIN)' (:34856),
    'carbamoyl cyanide (PIN)' (:34858); 'NC-CO-Cl carbonocyanidoyl chloride
    (PIN)' (:31482). ``_name_carbonic_nitrile``.
(C) Amides of di- and polycarbonic acids. (:33539) "formed by adding the
    functional class name 'amide' to that of the corresponding acid, preceded by the
    numerical prefix 'di'... Numerical and letter locants are used": 'dicarbonic diamide
    (PIN)', 'N1-(propan-2-yl)dicarbonic diamide (PIN)', 'N1-methyl-2-thiotricarbonic
    diamide (PIN)' (:33543-:33549); condensed ureas (:33505)
    '2,4-diimidotricarbonic diamide (PIN)', 'N1-methyl-2-imidodicarbonic diamide (PIN)',
    'N1-methyl-2-imido-1-thiodicarbonic diamide (PIN)' (:33509-:33519).
    ``_name_polycarbonic_diamide``.
(D) The acid of urea. (:33366): 'carbamoylcarbamic acid (PIN)' (:33374);
    the poly-carbonic replacement builder declines the dicarbonic amido/imido acid.
(E) Urea ranked as an amide of carbonic acid. (:33320): "Derivatives of
    urea formed by substitution on the nitrogen atom(s) are named as substitution
    products in accordance with the seniority order of urea that is ranked as an amide of
    carbonic acid" -- "N-[1-cyano-3-(methylsulfanyl)propyl]-N'-methylurea (PIN)"
    (:33336). Amides are senior to nitriles and the classes after them, so the urea
    handler takes a urea whose other groups are all junior to it.
"""
import pytest

from tests.support.pin_tiers import (
    assert_declined_at_default,
    assert_not_pin_labelled,
    assert_pin_at_both_tiers,
)

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    # (A)
    ("N#CN=[N+]=[N-]", "carbononitridic azide"),                         #:31515
    ("BrC#N", "carbononitridic bromide"),
    ("FC#N", "carbononitridic fluoride"),
    ("IC#N", "carbononitridic iodide"),
    ("N#CN=C=O", "carbononitridic isocyanate"),
    ("N#CN=C=S", "carbononitridic isothiocyanate"),
    # (B)
    ("N#CC(=N)C#N", "carbonimidoyl dicyanide"),                          #:34854
    ("N#CC(C#N)=NN", "carbonohydrazonoyl dicyanide"),                    #:34856
    ("N#CC(=S)C#N", "carbonothioyl dicyanide"),
    ("N#CC(=[Se])C#N", "carbonoselenoyl dicyanide"),
    ("NC(=O)C#N", "carbamoyl cyanide"),                                  #:34858
    ("NC(=S)C#N", "carbamothioyl cyanide"),
    ("ClC(=O)C#N", "carbonocyanidoyl chloride"),                         #:31482
    ("BrC(=O)C#N", "carbonocyanidoyl bromide"),
    # (C)
    ("NC(=O)OC(N)=O", "dicarbonic diamide"),                             #:33543
    ("CC(C)NC(=O)OC(N)=O", "N1-(propan-2-yl)dicarbonic diamide"),        #:33547
    ("CNC(=O)SC(=O)OC(N)=O", "N1-methyl-2-thiotricarbonic diamide"),     #:33549
    ("NC(=O)NC(=O)NC(N)=O", "2,4-diimidotricarbonic diamide"),           #:33509
    ("CNC(=O)NC(N)=O", "N1-methyl-2-imidodicarbonic diamide"),           #:33514
    ("CNC(=S)NC(N)=O", "N1-methyl-2-imido-1-thiodicarbonic diamide"),    #:33519
    ("NC(=O)NC(N)=O", "2-imidodicarbonic diamide"),
    ("NC(=O)OC(=O)OC(N)=O", "tricarbonic diamide"),
    ("NC(=O)SC(N)=O", "2-thiodicarbonic diamide"),
    ("NC(=O)OOC(N)=O", "2-peroxydicarbonic diamide"),
    ("NC(=S)OC(N)=S", "1,3-dithiodicarbonic diamide"),
    ("CNC(=O)OC(=O)NC", "N1,N3-dimethyldicarbonic diamide"),
    ("CN(C)C(=O)OC(N)=O", "N1,N1-dimethyldicarbonic diamide"),
    ("CCN(CC)C(=O)OC(=O)N(CC)CC", "N1,N1,N3,N3-tetraethyldicarbonic diamide"),
    ("c1ccccc1NC(=O)OC(N)=O", "N1-phenyldicarbonic diamide"),
    # (D)
    ("NC(=O)NC(=O)O", "carbamoylcarbamic acid"),                         #:33374
    # (E)
    ("CNC(=O)Nc1ccc(C#N)cc1", "N-(4-cyanophenyl)-N'-methylurea"),
    ("NC(=O)NCC#N", "(cyanomethyl)urea"),
    ("CNC(=O)NCC#N", "N-(cyanomethyl)-N'-methylurea"),
    ("NC(=O)Nc1ccccc1C#N", "(2-cyanophenyl)urea"),
    ("NC(=O)NCC(C)=O", "(2-oxopropyl)urea"),
    ("NC(=O)NCC=O", "(2-oxoethyl)urea"),
    ("NC(=O)NCCO", "(2-hydroxyethyl)urea"),           # O-H not substitutable,:3007
    ("NC(=O)Nc1ccc(O)cc1", "(4-hydroxyphenyl)urea"),
    ("O=C(NCCO)NCCO", "N,N'-bis(2-hydroxyethyl)urea"),
    ("NC(=O)NCCN", "N-(2-aminoethyl)urea"),           # N-H: 'N-carbamimidoylurea',:34292
    ("Cc1ncc(CNC(=O)N(CCCl)N=O)c(N)n1",
     "N'-[(4-amino-2-methylpyrimidin-5-yl)methyl]-N-(2-chloroethyl)-N-nitrosourea"),
    ("O=C(Nc1ccc([N+](=O)[O-])cc1O)Nc1ccccc1Br",
     "N-(2-bromophenyl)-N'-(2-hydroxy-4-nitrophenyl)urea"),
    # (:3447): 'tert-butyl' is filed under 'b' ('4-butyl-4-tert-butyl-
    # cyclohexan-1-ol (PIN)',:3463)
    ("CC(C)(C)NC(=O)NC", "N-tert-butyl-N'-methylurea"),
]
NOT_PIN_ROWS = [
    ("N#CN=[N+]=[N-]", "azidomethanenitrile"),
    ("BrC#N", "bromomethanenitrile"),
    ("N#CC(=N)C#N", "iminopropanedinitrile"),                 #:34854 'not'
    ("N#CC(C#N)=NN", "hydrazinylidenepropanedinitrile"),       #:34856 'not'
    ("NC(=O)OC(N)=O", "bis(1-aminomethanoic) anhydride"),
    ("NC(=O)NC(=O)O", "1-amido-2-imidodicarbonic acid"),
    ("NC(=O)NCCO", "2-(carbamoylamino)ethan-1-ol"),
    ("NC(=O)NCC#N", "(carbamoylamino)ethanenitrile"),
    ("CC(C)(C)NC(=O)NC", "N-methyl-N'-tert-butylurea"),
]
# the boundary: classes senior to urea keep their own parents
CONTROL_ROWS = [
    ("N#CC(=O)C#N", "carbonyl dicyanide"),                    #:34848
    ("ClC#N", "carbononitridic chloride"),                    #:31513
    ("N#CC#N", "oxalonitrile"),
    ("NC(=S)SC(N)=S", "1,2,3-trithiodicarbonic diamide"),     #:33553
    ("OC(=O)NC(=O)O", "2-imidodicarbonic acid"),              #:31051
    ("OC(=O)OC(=O)O", "dicarbonic acid"),
    ("NC(=O)NCC(=O)O", "(carbamoylamino)acetic acid"),        # acid senior
    ("CC(=O)NC(N)=O", "N-carbamoylacetamide"),                # carboxamide senior
    ("CNC(=O)NC", "N,N'-dimethylurea"),                       #:33326
    ("NC(=O)N", "urea"),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS + CONTROL_ROWS)
def test_carbonic_family_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", NOT_PIN_ROWS)
def test_carbonic_family_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


def test_urea_with_a_nitrile_branch_best_effort():
    #:33336 -- the best-effort tier names the PIN; the PIN tier has no substituent
    # namer for the cyano + methylsulfanyl chain yet (see the xfail below)
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    assert namer.name("CNC(=O)NC(C#N)CCSC") == \
        "N-[1-cyano-3-(methylsulfanyl)propyl]-N'-methylurea"


# Left as strict xfails, with the reason:
# * HO-C(=S)-O-C(=S)-OH: the Blue Book PIN of the known structure is '1,3-dithio-
# dicarbonic O1,O3-acid (PIN)',:31085); OPSIN 2.9.0 does not parse
# the superscript letter locants, so it cannot be certified. (The plan's
# '[(thiocarboxy)oxy]methanethioic O-acid' is the PIN for a structure whose sulfur
# location is unknown on one carbon,,:31093.)
# * The urea + nitrile row at the PIN tier: the PIN-tier substituent namer does not
# name -CH(C#N)CH2CH2SCH3 ('1-cyano-3-(methylsulfanyl)propyl'), so the urea handler
# declines; the nitrile-parent name the polyfunctional handler builds is then
# labelled below the PIN (test_junior_parent_name_is_not_the_pin), so the default
# tier declines and the best-effort tier ships the urea name, pin_unverified.
XFAIL_ROWS = [
    ("OC(=S)OC(O)=S", "1,3-dithiodicarbonic O1,O3-acid"),
    ("CNC(=O)NC(C#N)CCSC", "N-[1-cyano-3-(methylsulfanyl)propyl]-N'-methylurea"),
]


@pytest.mark.xfail(strict=True, reason="see the comment above")
@pytest.mark.parametrize("smiles,pin", XFAIL_ROWS)
def test_carbonic_family_open_rows(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_carbonochloridic_amide():
    # H2N-CO-Cl: prints 'carbonochloridic amide (PIN) (not
    # 1-chloroformamide)' (:32707), while names the acyl halides of carbamic
    # acid on 'carbamoyl' ('carbamoyl isocyanate (PIN)',:31488) and acid halides rank
    # above amides. The user ruled (2026-10-02, D1) that the printed PIN is
    # followed for H2N-CO-X; tests/unit/rules/test_user_rulings_d1_d3.py has the class.
    assert_pin_at_both_tiers("NC(=O)Cl", "carbonochloridic amide")



# (F) Review a performance pass, F-01 and F-03. When urea outranks the perceived principal group
#,:18158-:18200: 11 amides, urea among them as an amide of carbonic
# acid, above 12 hydrazides, 13 imides, 14 nitriles) and another handler names the
# molecule -- the urea handler declined, or the principal is a hydrazide, which the
# urea builder cannot spell as a prefix -- that name is correct but not the PIN: it is
# labelled below the PIN and the default tier declines.
JUNIOR_PARENT_ROWS = [
    ("CNC(=O)NCCC#N", "3-[(methylcarbamoyl)amino]propanenitrile"),
    ("CNC(=O)NC(C)C#N", "2-[(methylcarbamoyl)amino]propanenitrile"),
    ("CNC(=O)NC(C#N)CCSC", "2-[(methylcarbamoyl)amino]-4-(methylsulfanyl)butanenitrile"),
    ("NC(=O)NCC(=O)NN", "2-(carbamoylamino)ethanehydrazide"),
    ("NC(=O)Nc1ccc(C(=O)NN)cc1", "4-(carbamoylamino)benzohydrazide"),
]


@pytest.mark.parametrize("smiles,non_pin", JUNIOR_PARENT_ROWS)
def test_junior_parent_name_is_not_the_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)
    # nothing else is certified either: the urea builder has no correct prefix for a
    # hydrazide (it would write 'aminoamino'), and the urea + nitrile rows need a
    # PIN-tier substituent namer that is not there
    assert_declined_at_default(smiles)


@pytest.mark.parametrize("smiles,name", [
    ("CNC(=O)NCCC#N", "N-(2-cyanoethyl)-N'-methylurea"),
    ("CNC(=O)NC(C)C#N", "N-(1-cyanoethyl)-N'-methylurea"),
])
def test_junior_parent_urea_name_at_best_effort(smiles, name):
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    assert Orthonym(style="pin", **_emit_tier_flags("best-effort")).name(smiles) == name


# not a plain urea: a semicarbazide is named on hydrazinecarboxamide
#:38623), an N-nitrosourea keeps the urea parent (:25965)
@pytest.mark.parametrize("smiles,pin", [
    ("NC(=O)NNc1ccccc1", "2-phenylhydrazine-1-carboxamide"),
    ("CN(N=O)C(N)=O", "N-methyl-N-nitrosourea"),
    # an N-acyl urea is a carboxamide ('N-carbamoyl-2-phenylacetamide (PIN)',:33364)
    ("NC(=O)NC(=O)c1ccc(C#N)cc1", "N-carbamoyl-4-cyanobenzamide"),
])
def test_urea_boundary_controls(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# (G) Review a performance pass, F-07: the hetero-rooted prefix of the best-effort located FG
# namer names only a single-bonded O/S/N with the hydrogens its bonds leave; a nitrile
# N (a triple bond, no H) or an =NH is not 'amino'.
@pytest.mark.parametrize("smiles,atoms,expected", [
    ("CC#N", [2], None),
    ("CC=N", [2], None),
    ("CCN", [2], "amino"),
    ("CCO", [2], "hydroxy"),
    ("CCNC", [2, 3], "methylamino"),
])
def test_hetero_root_prefix(smiles, atoms, expected):
    from rdkit import Chem
    from orthonym.assembly.substituent_naming import _located_fg_hetero_root
    assert _located_fg_hetero_root(Chem.MolFromSmiles(smiles), atoms, 2) == expected


# (H) Review a performance pass, F-09: urea with all four N-H replaced by one substituent cites no
# N locants. (:3007) "All locants are omitted in compounds... in which all
# substitutable positions are completely substituted or modified... in the same way"
# -- 'tetrafluorourea (PIN)' (:3025). A mixed set keeps them.
@pytest.mark.parametrize("smiles,pin", [
    ("CN(C)C(=O)N(C)C", "tetramethylurea"),
    ("CCN(CC)C(=O)N(CC)CC", "tetraethylurea"),
    ("ClCCN(CCCl)C(=O)N(CCCl)CCCl", "tetrakis(2-chloroethyl)urea"),
    ("FN(F)C(=O)N(F)F", "tetrafluorourea"),                       #:3025
    ("CN(C)C(=O)N(C)CC", "N-ethyl-N,N',N'-trimethylurea"),
])
def test_completely_substituted_urea(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_completely_substituted_urea_not_pin():
    assert_not_pin_labelled("CN(C)C(=O)N(C)C", "N,N,N',N'-tetramethylurea")


# Review a performance pass, F-06, then the user ruling D1 (2026-10-02): 'carbonochloridic amide
# (PIN)' (:32707) is followed for H2N-CO-X, so the 'carbamoyl' halide names of
# (:31474) are not the PIN of the unsubstituted amide; the pseudohalides keep their (PIN).
@pytest.mark.parametrize("smiles,name", [
    ("NC(=O)Cl", "carbamoyl chloride"),
    ("NC(=O)F", "carbamoyl fluoride"),
])
def test_carbamoyl_halides_are_not_pin(smiles, name):
    from tests.support.pin_tiers import name_breadth, name_default
    for res in (name_default(smiles), name_breadth(smiles)):
        assert res.get("name") != name, res
    assert_not_pin_labelled(smiles, name)


@pytest.mark.parametrize("smiles,pin", [
    ("NC(=O)N=C=O", "carbamoyl isocyanate"),     #:31488
    ("NC(=O)C#N", "carbamoyl cyanide"),          #:34858
    ("CN(C)C(=O)Cl", "dimethylcarbamoyl chloride"),   # no BB row for the structure
])
def test_carbamoyl_pseudohalides_keep_the_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)



# (e) (:7188): 'bis' before a name beginning with a multiplicative 'di' --
# 'bis(diazenyl) (preselected prefix...) (not didiazenyl)' (:7194); '3,4-bis(diazenyl)-
# benzoic acid' is the Blue Book row of. The diazenyl prefix reached the
# benzoic acid namer once the bare 'amino' of the hetero root needed its hydrogens.
@pytest.mark.parametrize("smiles,pin", [
    ("OC(=O)c1ccc(N=N)c(N=N)c1", "3,4-bis(diazenyl)benzoic acid"),
    ("SSc1ccc(SS)cc1", "1,4-bis(disulfanyl)benzene"),
])
def test_bis_before_a_di_initial_prefix(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_bis_before_a_di_initial_prefix_not_pin():
    assert_not_pin_labelled("OC(=O)c1ccc(N=N)c(N=N)c1", "3,4-didiazenylbenzoic acid")
