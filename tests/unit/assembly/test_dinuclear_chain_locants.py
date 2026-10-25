"""The two-carbon chain with one multiple bond: no 'eth-1-ene' / 'eth-1-yne'.

Defect: ``C=CS(=O)(=O)O`` was named ``eth-1-enesulfonic acid``, ``C=CO`` ``eth-1-en-1-ol``,
``C#CS(=O)(=O)O`` ``eth-1-ynesulfonic acid``, all labelled ``pin_verified``. The string
``eth-1-en`` occurs nowhere in the Blue Book. The spelling came from one primitive,
``composition_primitives._build_unsaturation_infix`` (the first function to return ``-1-en``
for every PIN-only-tier producer: traced in fresh processes on 14 molecules), and from three
hand-rolled copies of the same decision at the best-effort tier (``general_engine._stem_block``,
``terminal_fragment._chain_stem_with_unsaturation``, ``rules.ions`` cumulative suffixes).

Rule table (``the Blue Book Blue Book``; every row VERIFIED against the quoted line, a row
marked ASSUMED is derived, not printed).

 "The locant '1' is omitted:" (:2891)

* (d) "in unsubstituted dinuclear and trinuclear alkenes and alkynes..." (:2921):
  ``CH2=CH2 ethene (PIN)`` (:2925). The one multiple bond of a two-carbon chain lies between C1
  and C2 whatever else the chain carries, and the book writes it without a locant on a
  substituted chain too: ``2-chloroethen-1-yl (preferred prefix)`` (:3003, under,
  ``(1Z)-1,2-dibromo-1-chloro-2-iodoethene (PIN)`` (:46763), ``2-bromoethen-1-ylidene`` (:45805),
  ``1,1'-(ethene-1,2-diyl)dibenzene (PIN)`` (:16615), ``ethenone (PIN)`` (:28314). So the
  ene/yne locant of a two-carbon chain is never cited. (The licence's own text says
  "unsubstituted"; the substituted rows are the printed examples above.)
* (b) "in monosubstituted homogeneous chains consisting of only two identical atoms" (:2903):
  ``CH3-CH2-OH ethanol (PIN)`` (:2907), ``hydrazinecarboxylic acid (PIN)`` (:18893),
  ``diazenecarbohydrazide (PIN)`` (:38900). One suffix on ethene/ethyne (ASSUMED, no printed
  ethene example): ``ethenol``, ``ethenamine``, ``ethenesulfonic acid``, ``ethynesulfonic acid``.

 "Citation of locants" (:2869): a second decoration makes the suffix locant essential
('2-chloroethan-1-ol is the PIN'), so ``2-chloroethen-1-ol`` and ``2-chloroethene-1-sulfonic
acid`` keep the suffix and the prefix locants and drop only the ene/yne one (the printed
``2-chloroethen-1-yl`` pattern). Two suffixes: ``ethene-1,2-diol`` / ``ethene-1,1-diol`` keep
theirs (an isomer exists,:2953), as ``ethane-1,2-diol`` does. An isotopic label that
forces a locant:44180) restores the suffix locant: ``(2-13C)ethen-1-ol``.

 /: the lone centre of a two-carbon chain takes no locant (``ethenide``:40171,
``ethenylidene``:24547, ``ethynediide (PIN)``:40918); a substituent restores it
('2-bromoethen-1-ylidene':45805).

Not in this class and left alone: a replacement ('a') chain of two skeletal atoms
('1,2-diazaeth-1-ene', ``engine:t4``), whose stem is not the bare ``eth`` (the primitive
never claims it).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.assembly.composition_primitives import (
    _build_hydrocarbon_name,
    _build_unsaturation_infix,
    bond_locant_omitted_on_dinuclear_chain,
    dinuclear_chain_length,
)
from orthonym.tiers import emit_tier_flags

pytestmark = pytest.mark.unit


def _pin():
    """The PIN-only tier, named explicitly (the default of ``name_compound`` is the best-effort tier)."""
    return Orthonym(style="pin", **emit_tier_flags("pin"))


def _best_effort():
    return Orthonym(style="pin", **emit_tier_flags("best-effort"))


def _key(smiles):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


# --------------------------------------------------------------------------- #
# the shared primitive (no JVM)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stem,double,triple,expected", [
    ("eth", [1], [], True),                      # ethene, (d):2921
    ("eth", [], [1], True),                      # ethyne
    ("eth", [], [], False),                      # ethane: no multiple bond
    ("prop", [1], [], False),                    # a three-carbon chain keeps its own rules
    ("but", [2], [], False),                     # but-2-ene
    ("2-oxaeth", [1], [], False),                # a replacement chain is another parent
    ("1,2-diazaeth", [1], [], False),
    ("cycloprop", [1], [], False),
])
def test_primitive_claims_only_the_bare_two_carbon_chain(stem, double, triple, expected):
    assert bond_locant_omitted_on_dinuclear_chain(stem, double, triple) is expected


def test_dinuclear_chain_length_is_the_one_place_that_names_the_stem():
    assert dinuclear_chain_length("eth") == 2
    assert dinuclear_chain_length("meth") == 0
    assert dinuclear_chain_length("2-oxaeth") == 0


@pytest.mark.parametrize("double,triple,stem,infix", [
    ([1], [], "eth", "en"),
    ([], [1], "eth", "yn"),
    ([1], [], "prop", "-1-en"),                  # no claim about a longer chain
    ([2], [], "but", "-2-en"),
    ([1], [], "", "-1-en"),                      # stem unknown: locants are cited
    ([1, 3], [], "hex", "a-1,3-dien"),
])
def test_unsaturation_infix_with_a_stem(double, triple, stem, infix):
    assert _build_unsaturation_infix(double, triple, stem) == infix


def test_hydrocarbon_writer_and_infix_writer_agree_on_ethene_and_ethyne():
    """ (d) (:2921): 'ethene (PIN)' (:2925); the suffix writer must spell the same stem."""
    assert _build_hydrocarbon_name("eth", [1], []) == "ethene"
    assert _build_hydrocarbon_name("eth", [], [1]) == "ethyne"
    assert "eth" + _build_unsaturation_infix([1], [], "eth") + "e" == "ethene"
    assert "eth" + _build_unsaturation_infix([], [1], "eth") + "e" == "ethyne"


# --------------------------------------------------------------------------- #
# PIN-only tier: the names
# --------------------------------------------------------------------------- #

# (smiles, name, governing rule)
ONE_SUFFIX = [
    ("C=CO", "ethenol", "P-14.3.4.2 (b) :2903 + (d) :2921"),
    ("C=CS", "ethenethiol", "P-14.3.4.2 (b) :2903"),
    ("C=CN", "ethenamine", "P-14.3.4.2 (b) :2903"),
    ("C#CN", "ethynamine", "P-14.3.4.2 (b) :2903"),
    ("C=CNC", "N-methylethenamine", "N-substituents do not decorate the chain, :2903"),
    ("C=CN(C)C", "N,N-dimethylethenamine", "ditto"),
    ("C=C[N+](C)(C)C", "N,N,N-trimethylethenaminium", "ditto"),
    ("C=C[NH3+]", "ethenaminium", "ditto"),
    ("C=CS(=O)O", "ethenesulfinic acid", "P-14.3.4.2 (b) :2903"),
    ("C=CS(=O)(=O)O", "ethenesulfonic acid", "P-14.3.4.2 (b) :2903"),
    ("C#CS(=O)(=O)O", "ethynesulfonic acid", "P-14.3.4.2 (b) :2903"),
    ("C=CS(=O)(=O)N", "ethenesulfonamide", "P-14.3.4.2 (b) :2903"),
    ("C=CS(=O)(=O)NC", "N-methylethenesulfonamide", "P-14.3.4.2 (b) :2903"),
    ("C=CS(=O)(=O)Cl", "ethenesulfonyl chloride", "P-14.3.4.2 (b) :2903"),
    ("C=CS(=O)(=O)OC", "methyl ethenesulfonate", "P-14.3.4.2 (b) :2903"),
    ("C=CS(=O)(=O)[O-]", "ethenesulfonate", "P-14.3.4.2 (b) :2903"),
    ("[Na+].C=CS(=O)(=O)[O-]", "sodium ethenesulfonate", "P-14.3.4.2 (b) :2903"),
    ("C=CS(=O)(=O)C#N", "ethenesulfonyl cyanide", "P-14.3.4.2 (b) :2903"),
    ("C=C[Se](=O)(=O)O", "etheneselenonic acid", "P-14.3.4.2 (b) :2903"),
    ("C=C[N+]#N", "ethenediazonium", "P-14.3.4.2 (b) :2903"),
]

SUFFIX_AND_PREFIX = [
    #:2869: a second decoration makes the suffix locant essential, the ene locant is
    # still never cited ('2-chloroethen-1-yl':3003)
    ("ClC=CO", "2-chloroethen-1-ol"),
    ("C=C(Cl)O", "1-chloroethen-1-ol"),
    ("ClC(Cl)=CO", "2,2-dichloroethen-1-ol"),
    ("ClC=CN", "2-chloroethen-1-amine"),
    ("ClC=CS", "2-chloroethene-1-thiol"),
    ("ClC=CS(=O)(=O)O", "2-chloroethene-1-sulfonic acid"),
    ("c1ccccc1C=CS(=O)(=O)O", "2-phenylethene-1-sulfonic acid"),
    ("C=C(c1ccccc1)S(=O)(=O)N", "1-phenylethene-1-sulfonamide"),
    ("ClC=CNC", "2-chloro-N-methylethen-1-amine"),
    ("OC=CN", "2-aminoethen-1-ol"),
    ("ClC#CS(=O)(=O)O", "2-chloroethyne-1-sulfonic acid"),   # locants kept: deny by default,
]

TWO_SUFFIXES = [
    # an isomer exists (1,1 against 1,2), so the locants stay, as 'ethane-1,2-diol' does
    ("OC=CO", "ethene-1,2-diol"),
    ("C=C(O)O", "ethene-1,1-diol"),
    ("NC=CN", "ethene-1,2-diamine"),
    ("C=C(N)N", "ethene-1,1-diamine"),
    ("SC=CS", "ethene-1,2-dithiol"),
    ("C=C(S(=O)(=O)O)S(=O)(=O)O", "ethene-1,1-disulfonic acid"),
    ("OC(O)=CO", "ethene-1,1,2-triol"),
    ("C=C(S(=O)(=O)O)O", "1-hydroxyethene-1-sulfonic acid"),
]

STEREO_AND_ISOTOPE = [
    ("Cl/C=C\\O", "(1Z)-2-chloroethen-1-ol"),
    ("Cl/C=C/N", "(1E)-2-chloroethen-1-amine"),
    ("O=S(=O)(O)/C=C/c1ccccc1", "(1E)-2-phenylethene-1-sulfonic acid"),
    ("O/C=C/O", "(1E)-ethene-1,2-diol"),
    # (:44180): a locant forced by the isotopic label restores every locant
    ("[13CH2]=CO", "(2-13C)ethen-1-ol"),
    ("C=C[15NH2]", "(15N)ethenamine"),
]

COMPLETELY_OR_ISOMERLESS = [
    # (:3007) "all locants are omitted... completely substituted" and (:2953)
    ("ClC#CCl", "dichloroethyne"),
    ("ClC(Cl)=C(Cl)Cl", "tetrachloroethene"),
    ("ClC(Cl)=CCl", "trichloroethene"),
    ("OC(=O)c1ccc(C=C=O)cc1", "4-(oxoethenyl)benzoic acid"),           # 'oxoethenyl':2983
    ("OC(=O)c1ccc(C#CCl)cc1", "4-(chloroethynyl)benzoic acid"),
    ("OC(=O)c1ccc(C(Cl)=C(Cl)Cl)cc1", "4-(trichloroethenyl)benzoic acid"),
    # the locants stay where an isomer exists
    ("ClC=CCl", "1,2-dichloroethene"),
    ("ClC(Cl)=C", "1,1-dichloroethene"),
    ("OC(=O)c1ccc(C=CCl)cc1", "4-(2-chloroethen-1-yl)benzoic acid"),    #:3003
    ("OC(=O)c1ccc(C(Cl)=C)cc1", "4-(1-chloroethen-1-yl)benzoic acid"),
    ("OC(=O)c1ccc(C=C(Cl)Cl)cc1", "4-(2,2-dichloroethen-1-yl)benzoic acid"),
]

UNCHANGED = [
    # hydrocarbon and prefix forms were already right; they must stay
    ("C=C", "ethene"),
    ("ClC=C", "chloroethene"),
    ("ClC=CCl", "1,2-dichloroethene"),
    ("ClC(Cl)=C", "1,1-dichloroethene"),
    ("Br/C=C\\Br", "(1Z)-1,2-dibromoethene"),
    ("C=Cc1ccccc1", "ethenylbenzene"),
    ("C=CP(=O)(O)O", "ethenylphosphonic acid"),
    ("C=CC#N", "prop-2-enenitrile"),
    ("C=CC(=O)O", "prop-2-enoic acid"),
    ("C=C=O", "ethenone"),
    ("[C-]#[C-]", "ethynediide"),
    ("c1ccccc1C=Cc1ccccc1", "1,1'-(ethene-1,2-diyl)dibenzene"),
    # the locant is REQUIRED on a longer chain and is kept
    ("CC=CO", "prop-1-en-1-ol"),
    ("C=C(C)O", "prop-1-en-2-ol"),
    ("C=CCO", "prop-2-en-1-ol"),
    ("CC=CS(=O)(=O)O", "prop-1-ene-1-sulfonic acid"),
    ("CC=CN", "prop-1-en-1-amine"),
    ("OCC=CCO", "but-2-ene-1,4-diol"),
    # the saturated sibling cites its suffix locant when a decoration forces it
    ("ClCCO", "2-chloroethan-1-ol"),
    ("OCCO", "ethane-1,2-diol"),
    ("CCS(=O)(=O)O", "ethanesulfonic acid"),
]

PIN_TIER = ONE_SUFFIX + [(s, n) for s, n in SUFFIX_AND_PREFIX] + TWO_SUFFIXES \
    + STEREO_AND_ISOTOPE + COMPLETELY_OR_ISOMERLESS + UNCHANGED


def _rows(table):
    return [(r[0], r[1]) for r in table]


@pytest.mark.parametrize("smiles,name", _rows(PIN_TIER))
def test_pin_tier_name(smiles, name):
    assert _pin().name(smiles) == name


@pytest.mark.parametrize("smiles,name", _rows(ONE_SUFFIX + SUFFIX_AND_PREFIX + TWO_SUFFIXES
                                               + STEREO_AND_ISOTOPE))
def test_pin_tier_names_label_and_never_spell_eth_1_ene(smiles, name):
    res = _pin().name_tiered(smiles)
    assert res["name"] == name
    assert "eth-1-en" not in res["name"] and "eth-1-yn" not in res["name"], res["name"]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", _rows(ONE_SUFFIX + SUFFIX_AND_PREFIX + TWO_SUFFIXES))
def test_opsin_2_9_0_reads_each_name_back_to_the_full_inchikey(smiles, name):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    back = opsin_parse(name)
    assert back is not None, f"OPSIN cannot read {name!r}"
    assert _key(back) == _key(smiles), (name, back)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("name,smiles", [
    ("ethenol", "C=CO"),
    ("ethenamine", "C=CN"),
    ("ethenesulfonic acid", "C=CS(=O)(=O)O"),
    ("ethynesulfonic acid", "C#CS(=O)(=O)O"),
    ("2-chloroethene-1-sulfonic acid", "ClC=CS(=O)(=O)O"),
    ("ethene-1,2-diol", "OC=CO"),
])
def test_the_spellings_the_book_licenses_parse(name, smiles):
    """The omitted-locant spellings are not just shorter: OPSIN 2.9.0 reads each to the
    structure (the read-back that the engine's own gate would apply)."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    back = opsin_parse(name)
    assert back is not None and _key(back) == _key(smiles), (name, back)


def test_eth_1_ene_never_reaches_the_pin_tier():
    """A name that cites the ene locant of a two-carbon chain is not a PIN spelling: 0 'eth-1-en'
    in the PIN-only tier names of the whole panel (the signature has 100% precision here,
    because no other parent writes 'eth-1-en' without a replacement prefix)."""
    names = [_pin().name(s) for s, _ in _rows(PIN_TIER)]
    assert not [n for n in names if "eth-1-en" in n or "eth-1-yn" in n]


# --------------------------------------------------------------------------- #
# best-effort tier: the three hand-rolled copies of the decision
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("smiles,name", [
    # general_engine._stem_block
    ("C#CO", "hydroxyethyne"),
    ("C=C=S", "sulfanylideneethene"),
    # terminal_fragment._chain_stem_with_unsaturation
    ("c1ccccc1C#CO", "(hydroxyethynyl)benzene"),
    # (:2953) on the engine's own prefixes and ionic suffix: only C2 of an ethyne can carry
    # the decoration, so no locant; the carbanion '2-chloroethen-1-ide' keeps them (the centre could
    # sit on either carbon of an ethene)
    ("OC#CO", "dihydroxyethyne"),
    ("[C-]#CCl", "chloroethynide"),
    ("[C-]#C[Si](C)(C)C", "(trimethylsilyl)ethynide"),
    ("[CH-]=CCl", "2-chloroethen-1-ide"),
    ("O=C=Cc1ccccc1", "(oxoethenyl)benzene"),
    ("OC(=O)c1ccc(C#C[Si](C)(C)C)cc1", "4-[(trimethylsilyl)ethynyl]benzoic acid"),
])
def test_best_effort_stem_writers_omit_the_locant(smiles, name):
    assert _best_effort().name(smiles) == name


@pytest.mark.parametrize("smiles,name", [
    ("C=[CH-]", "ethenide"),                       #:40171 'sodium ethenide'
    ("C=[C]", "ethenylidene"),                     #:24547 'ethenylidene'
    ("[Na+].C#[C-]", "sodium ethynide"),
    ("[CH]=[CH]", "ethene-1,2-diyl"),              #:16615
    ("C=[CH]", "ethenyl"),
])
def test_cumulative_suffix_names_omit_the_locant(smiles, name):
    assert _pin().name(smiles) == name


# --------------------------------------------------------------------------- #
# the floor and terminal-fragment prefix writers (no gate involved)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("smiles,atoms,attach,book,mechanical", [
    # (d):2921 with '2-chloroethen-1-yl (preferred prefix)':3003 and 'ethenyl':24547:
    # '1-chloroethen-1-yl' is an isomer, so the locants of the ethenyl group stay
    ("C=Cc1ccccc1", [0, 1], 1, "ethenyl", "ethen-1-yl"),
    ("ClC=Cc1ccccc1", [0, 1, 2], 2, "2-chloroethen-1-yl", "2-chloroethen-1-yl"),
    ("C(Cl)(Cl)=Cc1ccccc1", [0, 1, 2, 3], 3, "2,2-dichloroethen-1-yl", None),
    #:2953: only C2 of an ethynyl or oxoethenyl group can carry the decoration, no isomer,
    # so no locant at all ('oxoethenyl (preferred prefix)':2983, 'chlorodiazenyl':2987); ASSUMED
    # for the ethynyl rows (the OCR'd book prints no ethynyl prefix)
    ("C#Cc1ccccc1", [0, 1], 1, "ethynyl", "ethyn-1-yl"),
    ("ClC#Cc1ccccc1", [0, 1, 2], 2, "chloroethynyl", None),
    ("O=C=Cc1ccccc1", [0, 1, 2], 2, "oxoethenyl", None),
    ("[Si](C)(C)(C)C#Cc1ccccc1", [0, 1, 2, 3, 4, 5], 5, "(trimethylsilyl)ethynyl", None),
    ("OC#Cc1ccccc1", [0, 1, 2], 2, "hydroxyethynyl", None),
])
def test_floor_and_terminal_fragment_prefix_writers_omit_the_locant(
        smiles, atoms, attach, book, mechanical):
    from orthonym.assembly.book_prefixes import mechanical_forms
    from orthonym.assembly.universal_substituent import name_universal_substituent_prefix
    from orthonym.rules.terminal_fragment import terminal_fragment_name
    mol = Chem.MolFromSmiles(smiles)
    assert name_universal_substituent_prefix(mol, atoms, attach) == book
    if mechanical is not None:
        with mechanical_forms():
            assert name_universal_substituent_prefix(mol, atoms, attach) == mechanical
            assert terminal_fragment_name(mol, set(atoms), attach).name == mechanical


@pytest.mark.parametrize("smiles,chain,expected", [
    #:2953, "no isomer can be generated by moving suffixes and/or prefixes"
    ("C#Cc1ccccc1", [0, 1], True),                    # ethynyl
    ("ClC#Cc1ccccc1", [1, 2], True),                  # chloroethynyl: C1 has no hydrogen to take it
    ("O=C=Cc1ccccc1", [1, 2], True),                  # oxoethenyl:2983
    ("C=Cc1ccccc1", [0, 1], True),                    # ethenyl
    ("ClC=Cc1ccccc1", [1, 2], False),                 # 1-chloroethen-1-yl exists,:3003
    ("ClC(Cl)=Cc1ccccc1", [1, 3], False),
    ("ClC(Cl)=C(Cl)Cl", [1, 3], True),                #: completely substituted
    ("ClC(Cl)=CCl", [1, 3], True),                    # 1,2,2- is 1,1,2-
    ("ClC=CCl", [1, 2], False),                       # 1,1-dichloroethene is an isomer
    ("ClC(Cl)=C", [1, 3], False),
    ("ClC#CCl", [1, 2], True),                        #
    ("OC#CO", [1, 2], True),
    ("C#CO", [0, 1], True),                           # (b), one decoration
    ("Cl/C=C/Cl", [1, 2], False),                     # the geometry cites locants
    ("C=C(Cl)c1ccccc1", [0, 1], False),
    ("CCC", [0, 1], False),                           # no multiple bond
    ("C1=CC1", [0, 1], False),                        # a ring
])
def test_two_carbon_chain_locant_licence(smiles, chain, expected):
    from orthonym.assembly.locant_omission import two_carbon_chain_locants_omitted
    assert two_carbon_chain_locants_omitted(Chem.MolFromSmiles(smiles), chain) is expected


def test_the_licence_ignores_stereo_and_fragments_outside_its_unit_of_structure():
    """ (:2869) reads a unit of structure: a stereocentre or a counter-ion elsewhere in the
    molecule does not change the spelling of the two-carbon group."""
    from orthonym.assembly.locant_omission import two_carbon_chain_locants_omitted
    mol = Chem.MolFromSmiles("O=C=C[C@H](C)Cl.[Na+].[Cl-]")
    assert two_carbon_chain_locants_omitted(mol, [1, 2]) is True


def test_a_replacement_chain_is_not_a_dinuclear_carbon_chain():
    """'2-oxaeth-1-en-1-yl' is skeletal replacement, whose own rules:6465) apply; the
    shared primitive never claims it, so the locant stays and the 'a' chain keeps its spelling."""
    from orthonym.assembly.universal_substituent import _dinuclear_carbon_parent
    assert _dinuclear_carbon_parent(2, {"double_bonds": [1], "triple_bonds": []}) == "ethene"
    assert _dinuclear_carbon_parent(2, {"double_bonds": [], "triple_bonds": [1]}) == "ethyne"
    assert _dinuclear_carbon_parent(3, {"double_bonds": [1], "triple_bonds": []}) is None
    assert _dinuclear_carbon_parent(2, {"double_bonds": [], "triple_bonds": []}) is None
