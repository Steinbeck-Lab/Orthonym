"""PIN spelling and labels -- review fixes, a performance pass (TRIAGE.md 'Review fixes (a performance pass)'
under 'PIN spelling -- four classes from the PubChem 1M job').

Every expected name below was read back by an independent OPSIN 2.9.0 call to the
input's full InChIKey (``tests/support/rt_assert``); the old spelling is the mutation
each name test rejects (``scripts/mutation_check.py``).

  A2-1 a multiplied prefix that is substituted takes 'bis', also when its first
        detachable prefix is an italicized one after a locant ('4-tert-butylphenyl'):
         (c) (the Blue Book, "any component which is substituted
        automatically requires use of the multiplicative forms 'bis', 'tris', etc."),
         (a) (:7104, "compound or complex (i.e. substituted) prefixes");
        a multiplied italic prefix inside an aryl prefix keeps its hyphen,
         (d) (:6958, 'di-*tert*-butyl':6964).
  A2-4 every producer joins a multiplier through the shared primitive
        (``naming_utils.multiplied_component``): (the Blue Book,
        "Parentheses... are used to enclose multiplied components that are:")
        (a) "simple substituent prefixes having locants" ('di(propan-2-yl)
        disulfite (PIN)':36921), (c) those "beginning with a multiplicative
        prefix" and (d) those "beginning with 'dec'" (:7104, 'di(dodecyl)',
        'di(decyl)'; 'di(dodecyl)silane (PIN)':38222); (a) 'bis' for a
        substituted prefix (:7104). The Blue Book gives no row for 'undecyl' or
        'icosyl' (c)), so those stay bare, as before.
  A1-1 a label on one member of an identical-prefix group: the members are cited
        apart, the Blue Book, "When two substituent groups
        are isotopically modified in different ways so that they cannot be combined
        together using multiplicative terms... they are cited separately"); the
        isotope decorator names the skeleton again with the groups apart
        (``composition_primitives.identical_prefixes_cited_apart``). Best-effort
        rows, labelled below the PIN.
  A1-2 a charge-delocalized group is one species whatever atom a drawing puts the
        charge on, the Blue Book): the isotope round trip's
        Kekule check accepts two drawings that differ by an alternating path whose
        ends exchange the charge, and still refuses a bond-shift isomer.
  A2-2 a fused mancude parent cites no ene locants, so of the two bond-shift
        drawings of a labelled or substituted pentalene, heptalene or s-indacene one
        is numbered away from the lowest locants ('(3-2H)pentalene', '5-methyl-
        heptalene'); its PIN carries a Delta descriptor "Localized
        double bonds", the Blue Book,:14597, '1,6-dimethyl-Delta1(10a)-
        heptalene (PIN)':14601) that OPSIN 2.9.0 cannot read, so the name is kept
        and labelled below the PIN. The lowest-locant drawing keeps its PIN label
        ('3-(as-indacen-3-yl)-5-(s-indacen-1-yl)pyridine (PIN)',. An
        italic designator of the parent is set off by a hyphen, (d)
        (:6960, '*as*-indacene':6966): '3-methyl-s-indacene'.
  A1-5 the decorated name carries the provenance of the skeleton it is built on,
        so one name gets one label at both tiers.
  A2-3 an N-substituted ring carboxamide / amine: the suffix-nitrogen prefixes
        are cited in the one alphanumerical series of the ring prefixes,
        the Blue Book;,:3477); the suffix of an N-substituted amide
        takes the lowest locant before the prefixes (c),:3256;
        '2-chloropyridine-3-carboxamide (PIN)':32895); the mark around an aryl
        prefix escalates past its own marks,:7444,
        '{[4-(hydroxymethyl)phenyl]methoxy}').
  A2-5 isotope descriptors: after an italic element locant ('N-(2H3)methyl',
        ,:43718); a shared locant must name the labelled position
        ('(2,2,2-2H3)ethyl', (1),:15813); a D on a parent nitrogen with
        other hydrogen-bearing positions takes its locant,:44202;
        '(N-2H1)acetamide (PIN)':43828).
  A2-6 pin tests of the class-2 grouping of the chain-amide, chain-amine and
        ring-parent producers (``composer._amide_with_identical_prefixes_merged``,
        ``heterocycles._identical_prefix_names_to_merge``).
    (lane-3 re-review) the ester prefixes of every mononuclear noncarbon
        oxoacid, derived from the acyl prefixes ('nitrooxy', 'sulfooxy',...), are
        labelled below the PIN under a class junior to esters,
        the Blue Book-35918;,:36401).
    (lane-3 re-review) binary names of salts of element ions ('aluminium
        azanetriide', 'trimagnesium bis(phosphanetriide)').
"""
import pytest

from tests.support.rt_assert import assert_full_rt, name_best_effort
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "CC1=C2C=C3C=CC=C3C=C2C=C1",
    "CC1=C2C=CC=CC=C2C=CC=C1",
    "[2H]C1=C2C=C3C=CC=C3C=C2C=C1",
    "[2H]C1=C2C=CC=C2C=C1",
    "[2H]C1=C2C=CC=CC=C2C=CC=C1",
    "[2H]C1=CC=CC2=CC=CC=CC2=C1",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)


pytestmark = pytest.mark.opsin_gate


def _pin_verified_name(smiles):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    return name, res.get("tier")


# --------------------------------------------------------------------------
# A2-1 'bis(4-tert-butylphenyl)', never 'di(4-tert-butylphenyl)'
# --------------------------------------------------------------------------

SUBSTITUTED_ITALIC_LED_ROWS = [
    # symmetric diesters (rules/esters.py, the organyl multiplier)
    ("CC(C)(C)c1ccc(OC(=O)C(=O)Oc2ccc(C(C)(C)C)cc2)cc1",
     "bis(4-tert-butylphenyl) oxalate"),
    ("CC(C)(C)c1ccc(OC(=O)c2ccc(C(=O)Oc3ccc(C(C)(C)C)cc3)cc2)cc1",
     "bis(4-tert-butylphenyl) benzene-1,4-dicarboxylate"),
    ("CC(C)(C)c1ccc(OC(=O)CCC(=O)Oc2ccc(C(C)(C)C)cc2)cc1",
     "bis(4-tert-butylphenyl) butanedioate"),
    ("CC(C)(C)c1ccccc1OC(=O)C(=O)Oc1ccccc1C(C)(C)C",
     "bis(2-tert-butylphenyl) oxalate"),
    ("CC(C)(C)c1cc(OC(=O)C(=O)Oc2cc(C(C)(C)C)cc(C(C)(C)C)c2)cc(C(C)(C)C)c1",
     "bis(3,5-di-tert-butylphenyl) oxalate"),
    ("CC(C)(C)c1cc(OC(=O)c2ccc(C(=O)Oc3cc(C(C)(C)C)cc(C(C)(C)C)c3)cc2)cc(C(C)(C)C)c1",
     "bis(3,5-di-tert-butylphenyl) benzene-1,4-dicarboxylate"),
    # amide N-substituents (rules/amides.py)
    ("CC(=O)N(c1ccc(C(C)(C)C)cc1)c1ccc(C(C)(C)C)cc1",
     "N,N-bis(4-tert-butylphenyl)acetamide"),
    # the aryl prefix itself (rules/ring_substituents.py): 'di-tert-butyl'
    ("CC(C)(C)c1cc(C(C)(C)C)cc(OC(C)=O)c1", "3,5-di-tert-butylphenyl acetate"),
]


@pytest.mark.parametrize("smiles,expected", SUBSTITUTED_ITALIC_LED_ROWS)
def test_substituted_italic_led_prefix_takes_bis(smiles, expected):
    name, tier = _pin_verified_name(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


@pytest.mark.parametrize("prefix,substituted", [
    ("4-tert-butylphenyl", True),
    ("2,4-di-tert-butylphenyl", True),
    ("3,5-di-tert-butylphenyl", True),
    ("2-tert-butyl-4-methylphenyl", True),
    ("4-sec-butylphenyl", True),
    ("N-tert-butylcarbamoyl", True),
    # simple prefixes stay simple (a),:7033)
    ("tert-butyl", False),
    ("sec-butyl", False),
    ("tert-pentyl", False),
    ("propan-2-yl", False),
    ("naphthalen-2-yl", False),
])
def test_is_substituted_substituent_reads_past_an_italic_prefix(prefix, substituted):
    from orthonym.assembly.naming_utils import is_substituted_substituent
    assert is_substituted_substituent(prefix) is substituted


def test_bis_term_is_enclosed():
    """ (the Blue Book): a term multiplied by 'bis' is enclosed,
    whatever the caller's own marks ('bis(sulfanyl)', never 'bissulfanyl')."""
    from orthonym.assembly.naming_utils import multiplied_component
    assert multiplied_component(2, "sulfanyl", "sulfanyl") == "bis(sulfanyl)"
    assert multiplied_component(2, "4-tert-butylphenyl", "(4-tert-butylphenyl)") == \
        "bis(4-tert-butylphenyl)"
    assert multiplied_component(2, "tert-butyl", "tert-butyl") == "di-tert-butyl"


# --------------------------------------------------------------------------
# A2-4 one multiplier primitive for every producer
# --------------------------------------------------------------------------

MULTIPLIER_ROWS = [
    # (c)/(d): esters, amides, amines, phosphates
    ("CCCCCCCCCCCCOC(=O)C(=O)OCCCCCCCCCCCC", "di(dodecyl) oxalate"),
    ("CCCCCCCCCCOC(=O)C(=O)OCCCCCCCCCC", "di(decyl) oxalate"),
    ("CCCCCCCCCCCCOC(=O)CCC(=O)OCCCCCCCCCCCC", "di(dodecyl) butanedioate"),
    ("CCCCCCCCCCCCN(CCCCCCCCCCCC)C(C)=O", "N,N-di(dodecyl)acetamide"),
    ("CCCCCCCCCCNC(=O)CC(=O)NCCCCCCCCCC", "N1,N3-di(decyl)propanediamide"),
    ("CCCCCCCCCCN(CCCCCCCCCC)CCCCCCCCCC", "N,N-di(decyl)decan-1-amine"),
    ("CCCCCCCCCCCCN(CCCCCCCCCCCC)c1ccccc1", "N,N-di(dodecyl)aniline"),
    ("CCCCCCCCCCCCOP(=O)(O)OCCCCCCCCCCCC", "di(dodecyl) hydrogen phosphate"),
    ("CCCCCCCCCCCCOP(=O)(OCCCCCCCCCCCC)OCCCCCCCCCCCC", "tri(dodecyl) phosphate"),
    ("CCCCCCCCCCCCc1ccc2[nH]c3ccc(CCCCCCCCCCCC)cc3c2c1", "3,6-di(dodecyl)-9H-carbazole"),
    # (a): fused ring C-prefixes (rules/fused_rings.py)
    ("Clc1ccc(-c2ccc3[nH]c4cc(-c5ccc(Cl)cc5)ccc4c3c2)cc1",
     "2,6-bis(4-chlorophenyl)-9H-carbazole"),
    ("CC(C)(C)c1ccc(-c2ccc3[nH]c4ccc(-c5ccc(C(C)(C)C)cc5)cc4c3c2)cc1",
     "3,6-bis(4-tert-butylphenyl)-9H-carbazole"),
    ("CC(C)(C)Sc1ccc2[nH]c3ccc(SC(C)(C)C)cc3c2c1", "3,6-bis(tert-butylsulfanyl)-9H-carbazole"),
    ("BrCc1ccc2[nH]c3ccc(CBr)cc3c2c1", "3,6-bis(bromomethyl)-9H-carbazole"),
    ("Clc1ccc(-c2[nH]c3ccccc3c2-c2ccc(Cl)cc2)cc1", "2,3-bis(4-chlorophenyl)-1H-indole"),
    ("Clc1ccc(-c2cc(-c3ccc(Cl)cc3)c3ccccc3n2)cc1", "2,4-bis(4-chlorophenyl)quinoline"),
    # (a): inorganic-acid esters (rules/phosphorus.py)
    ("CC(C)OP(=O)(O)OC(C)C", "di(propan-2-yl) hydrogen phosphate"),
    ("CC(C)OP(=O)(OC(C)C)OC(C)C", "tri(propan-2-yl) phosphate"),
    ("CC(C)OP(OC(C)C)OC(C)C", "tri(propan-2-yl) phosphite"),
    ("CC(C)OS(=O)(=O)OC(C)C", "di(propan-2-yl) sulfate"),
    ("CCC(C)OS(=O)(=O)OC(C)CC", "di(butan-2-yl) sulfate"),
    # (a): diamine N-substituents (assembly/composer.py)
    ("CC(C)N(C(C)C)CCCCCN", "N1,N1-di(propan-2-yl)pentane-1,5-diamine"),
    ("CC(C)N(C(C)C)CCN", "N1,N1-di(propan-2-yl)ethane-1,2-diamine"),
    ("CC(C)NCCCCCN", "N1-(propan-2-yl)pentane-1,5-diamine"),
    # unchanged: no marks below C10 or where the Blue Book is silent
    ("CCCCCCOC(=O)C(=O)OCCCCCC", "dihexyl oxalate"),
    ("CCCCCCCCCCCOC(=O)C(=O)OCCCCCCCCCCC", "diundecyl oxalate"),
    ("CCCCCCCCCCc1ccc(CCCCCCCCCC)cc1", "1,4-di(decyl)benzene"),
    ("CC(C)(C)OP(=O)(O)OC(C)(C)C", "di-tert-butyl hydrogen phosphate"),
]


@pytest.mark.parametrize("smiles,expected", MULTIPLIER_ROWS)
def test_multiplier_comes_from_the_shared_primitive(smiles, expected):
    name, tier = _pin_verified_name(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


def test_p1634_marks_in_the_primitive():
    from orthonym.assembly.naming_utils import multiplied_component
    assert multiplied_component(2, "dodecyl", "dodecyl") == "di(dodecyl)"
    assert multiplied_component(3, "decyl", "decyl") == "tri(decyl)"
    assert multiplied_component(2, "undecyl", "undecyl") == "diundecyl"
    assert multiplied_component(1, "dodecyl", "dodecyl") == "dodecyl"


# --------------------------------------------------------------------------
# A1-1 labelled member of an identical-prefix group: cited apart
# --------------------------------------------------------------------------

LABEL_ON_ONE_MEMBER_ROWS = [
    ("[2H]C1=C(C(=C(C(=C1[2H])[2H])C2=CC3=C(C=C2)N(C4=C3C=C(C=C4)N(C5=CC=CC=C5)"
     "C6=CC=CC=C6)C7=CC=C(C=C7)C(C)(C)C8=CC=CC=C8)[2H])[2H]",
     # a performance pass: the labelled copy is cited first, the Blue Book,
     # "The isotopically modified substituent is preferred alphabetically to the
     # unmodified substituent") and a complete phenyl cites no locants
     #,:44196)
     "6-(2H5)phenyl-N,N-diphenyl-9-[4-(2-phenylpropan-2-yl)phenyl]-"
     "9H-carbazol-3-amine", "pin_unverified"),
    ("[2H]C1=C(C(=C(C(=C1[2H])[2H])NC(=O)C2=C(N(C(=C2C3=CC=CC=C3)C4=CC=C(C=C4)F)"
     "CCC5CC(OC(O5)(C)C)CC(=O)OC(C)(C)C)C(C)C)[2H])[2H]",
     # the carboxamide suffix the heterocycle name cites takes the lowest locant, 3
     # (c), the Blue Book, "principal characteristic groups and free valences
     # (suffixes)"); it was 4
     "1-{2-[6-(2-tert-butoxy-2-oxoethyl)-2,2-dimethyl-1,3-dioxan-4-yl]ethyl}-5-"
     "(4-fluorophenyl)-N-(2H5)phenyl-4-phenyl-2-(propan-2-yl)-1H-"
     "pyrrole-3-carboxamide", "pin_unverified"),
    # roadmap N5b/N5d (name-quality lane L2): the carbazole takes its fusion name and
    # the phenyl group its prefix ('9-(2H5)phenyl-N,N-diphenyl-9-(9-phenyl-9H-carbazol-
    # 2-yl)-9H-fluoren-3-amine'), so the name keeps no part the label guard records as
    # not the PIN: pin_unverified, is_pin False (a label, not a derivation of the PIN).
    # It was '...-9-[8-(cyclohexa-1,3,5-trien-1-yl)-8-azatricyclo[7.4.0.0^2,7]trideca-
    # 1(13),2,4,6,9,11-hexaen-5-yl]-...', systematic_verified.
    ("[2H]C1=C(C(=C(C(=C1[2H])[2H])C2(C3=C(C=C(C=C3)N(C4=CC=CC=C4)C5=CC=CC=C5)"
     "C6=CC=CC=C62)C7=CC8=C(C=C7)C9=CC=CC=C9N8C1=CC=CC=C1)[2H])[2H]",
     "9-(2H5)phenyl-N,N-diphenyl-9-(9-phenyl-9H-carbazol-2-yl)-9H-fluoren-3-amine",
     "pin_unverified"),
    ("[2H]C1=C(C(=C(C(=C1[2H])[2H])C2(C3=C(C=C(C=C3)N(C4=CC=CC=C4)C5=CC=CC6=CC=CC=C65)"
     "C7=CC=CC=C72)C8=CC9=C(C=C8)SC1=CC=CC=C19)[2H])[2H]", None, "pin_unverified"),
]


@pytest.mark.parametrize("smiles,expected,tier", LABEL_ON_ONE_MEMBER_ROWS)
def test_label_on_one_member_of_a_group_is_named(smiles, expected, tier):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    if expected is not None:
        assert name == expected, name
    # spelling a performance pass cites the labelled member apart without a non-PIN record; where
    # the strict path (the strict twin) does not build the name it is "a name in PIN
    # form that only a breadth producer built" (the paper, Methods, "Tiers"):
    # pin_unverified; a name that keeps a non-PIN record is systematic_verified
    assert res.get("tier") == tier, res.get("tier")


def test_prefixes_apart_context_keeps_sources_apart():
    """Inside the context the producers cite each source's group on its own; outside
    it the class-2 grouping holds (the skeleton of the carbazole row above)."""
    from orthonym.assembly.composition_primitives import identical_prefixes_cited_apart
    from tests.support.rt_assert import _NAMER
    skel = ("c1ccc(cc1)-c1ccc2c(c1)c1cc(ccc1n2-c1ccc(cc1)C(C)(C)c1ccccc1)"
            "N(c1ccccc1)c1ccccc1")
    grouped = _NAMER.name(skel)
    with identical_prefixes_cited_apart():
        apart = _NAMER.name(skel)
    assert "N,N,6-triphenyl" in grouped, grouped
    assert "N,N-diphenyl-6-phenyl" in apart, apart


# --------------------------------------------------------------------------
# A1-2 one delocalized species, whatever atom carries the charge
# --------------------------------------------------------------------------

DELOCALIZED_ROWS = [
    ("[18O-][N+](=O)c1ccccc1", "(18O)nitrobenzene"),
    ("[18O]=[N+]([O-])c1ccccc1", "(18O)nitrobenzene"),
    ("C[N+](=O)[18O-]", "(18O)nitromethane"),
    ("[18O-][N+](=O)c1ccc(C(=O)O)cc1", "4-(18O)nitrobenzoic acid"),
    ("[18O-][N+](=O)c1ccncc1", "4-(18O)nitropyridine"),
    ("[18O-][N+](=O)c1ccc(cc1)[N+](=O)[O-]", "1-(18O)nitro-4-nitrobenzene"),
    ("[Na+].CS(=O)(=O)[18O-]", "sodium (18O)methanesulfonate"),
    ("[2H]C1=CC=C[CH+]C=C1", "(1-2H)cyclohepta-2,4,6-trien-1-ylium"),
    ("[2H][C+]1C=CC=CC=C1", "(1-2H)cyclohepta-2,4,6-trien-1-ylium"),
    ("[2H]C1=CC=C[CH-]1.[Na+]", "sodium (1-2H)cyclopenta-2,4-dien-1-ide"),
    ("[2H][C-]1C=CC=C1.[Na+]", "sodium (1-2H)cyclopenta-2,4-dien-1-ide"),
    ("NC(N)=[15NH2+].[Cl-]", "(15N)guanidinium chloride"),
    ("[15NH2]C(N)=[NH2+].[Cl-]", "(15N)guanidinium chloride"),
    ("CC(=O)[18O-].[Na+]", "sodium (18O)acetate"),
    ("CC(=[18O])[O-].[Na+]", "sodium (18O)acetate"),
]


@pytest.mark.parametrize("smiles,expected", DELOCALIZED_ROWS)
def test_delocalized_charge_is_one_species(smiles, expected):
    name, tier = _pin_verified_name(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


def test_sodium_nitrate_18o_is_named():
    name, tier = _pin_verified_name("[Na+].[O-][N+](=O)[18O-]")
    assert name == "sodium (18O)nitrate", name
    # carbon-free: no PIN; the label is the one of its naming path, as in
    # the paper's measured run (user decision 2026-09-30)
    assert tier == "pin_verified", tier


@pytest.mark.parametrize("a,b,same", [
    # one compound: a charge moved along an alternating path
    ("[O-][N+](=[18O])c1ccccc1", "[18O-][N+](=O)c1ccccc1", True),
    ("CC(=O)[18O-]", "CC(=[18O])[O-]", True),
    ("NC(N)=[15NH2+]", "[15NH2]C(N)=[NH2+]", True),
    ("CC(=[15NH2+])N", "CC([15NH2])=[NH2+]", True),
    ("[2H]C1=CC=C[CH+]C=C1", "[2H]C1=CC=CC=C[CH+]1", True),
    ("[2H]C1=CC=C[CH-]1", "[2H]C1=C[CH-]C=C1", True),
    ("[2H]C=CC=C[CH2+]", "[2H]C=C[CH+]C=C", True),
    # two compounds: bond-shift isomers (4n circuits), a moved hydron
    ("[2H]C1=CC=CC=CC=C1[2H]", "[2H]C1=C([2H])C=CC=CC=C1", False),
    ("CC1=CC=CC=CC=C1C", "CC1=C(C)C=CC=CC=C1", False),
    ("[2H]C1=CC=CC2=CC=CC=CC2=C1", "[2H]C1=CC2=CC=CC=CC2=CC=C1", False),
    ("[2H]OC(=O)CN", "OC(=O)CN[2H]", False),
    ("CC(=O)O[2H]", "CC(=O)[O-].[2H+]", False),
])
def test_kekule_check_models_charge_delocalization(a, b, same):
    from rdkit import Chem
    from orthonym.rules.isotopes import _kekule_forms_of_one_compound
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    assert _kekule_forms_of_one_compound(ma, mb) is same
    assert _kekule_forms_of_one_compound(mb, ma) is same


def test_kekule_check_is_bounded_without_a_charge_preserving_map():
    """A1-3: the atom search runs on the charge-free graphs, so a complete mapping
    exists whenever the constitutions agree and the mapping cap bounds the work (an
    exhaustive search took 24 s at eight tert-butyl groups)."""
    import time
    from rdkit import Chem
    from orthonym.rules.isotopes import _kekule_forms_of_one_compound
    tb = "".join("C(C(C)(C)C)" for _ in range(8))
    a = Chem.MolFromSmiles(f"CC(C)(C)C{tb}c1ccc(cc1)[N+](=O)[18O-]")
    b = Chem.MolFromSmiles(f"CC(C)(C)C{tb}c1ccc(cc1)[N+](=[18O])[O-]")
    t = time.time()
    assert _kekule_forms_of_one_compound(b, a)
    assert time.time() - t < 2.0


# --------------------------------------------------------------------------
# A2-2 (spelling part) an italic designator of the parent keeps its hyphen,
# (d) (the Blue Book, '*as*-indacene':6966)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("CC1=CC=C2C=C3C=CC=C3C=C12", "1-methyl-s-indacene"),
    ("CC1=C2C=CC3=CC=CC3=C2C=C1", "3-methyl-as-indacene"),
])
def test_italic_parent_designator_takes_a_hyphen(smiles, expected):
    name, tier = _pin_verified_name(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


def test_italic_designator_predicate():
    from orthonym.assembly.naming_utils import begins_with_italic_designator
    assert begins_with_italic_designator("s-indacene")
    assert begins_with_italic_designator("as-indacen-1-yl")
    assert not begins_with_italic_designator("indacene")
    assert not begins_with_italic_designator("sulfanyl")
    assert not begins_with_italic_designator("1H-indene")


# --------------------------------------------------------------------------
# A2-2 bond-shift drawings of fused mancude parents; the italic hyphen
# --------------------------------------------------------------------------

BOND_SHIFT_DRAWING_ROWS = [
    ("[2H]C1=C2C=CC=C2C=C1", "(3-2H)pentalene", "systematic_verified"),
    ("[2H]C1=CC=C2C=CC=C12", "(1-2H)pentalene", "pin_verified"),
    ("[2H]C1=CC=CC2=CC=CC=CC2=C1", "(4-2H)heptalene", "systematic_verified"),
    ("[2H]C1=CC2=CC=CC=CC2=CC=C1", "(2-2H)heptalene", "pin_verified"),
    ("[2H]C1=C2C=CC=CC=C2C=CC=C1", "(5-2H)heptalene", "systematic_verified"),
    ("[2H]C1=CC=CC=C2C=CC=CC=C12", "(1-2H)heptalene", "pin_verified"),
    ("CC1=C2C=CC=CC=C2C=CC=C1", "5-methylheptalene", "systematic_verified"),
    ("CC1=CC=CC=C2C=CC=CC=C12", "1-methylheptalene", "pin_verified"),
    ("CC1=C2C=C3C=CC=C3C=C2C=C1", "3-methyl-s-indacene", "systematic_verified"),
    ("CC1=CC=C2C=C3C=CC=C3C=C12", "1-methyl-s-indacene", "pin_verified"),
    ("[2H]C1=C2C=C3C=CC=C3C=C2C=C1", "(3-2H)s-indacene", "systematic_verified"),
    ("[2H]C1=CC=C2C=C3C=CC=C3C=C12", "(1-2H)s-indacene", "pin_verified"),
    ("CC1=C2C=CC3=CC=CC3=C2C=C1", "3-methyl-as-indacene", "pin_verified"),
    # controls: ene locants fix the bonds of a monocycle; a 4n+2 system is one
    # compound in both drawings; the Blue Book's own (PIN) example
    ("CC1=CC=CC=CC=C1[2H]", "1-methyl(8-2H)cycloocta-1,3,5,7-tetraene", "pin_verified"),
    ("CC1=CC=C2C=CC=CC=C12", "1-methylazulene", "pin_verified"),
    ("CC1=C2C=CC=CC=C2C=C1", "1-methylazulene", "pin_verified"),
    ("C1=Cc2cc3c(cc2=C1)C(c1cncc(C2=c4ccc5c(c4C=C2)C=CC=5)c1)=CC=3",
     "3-(as-indacen-3-yl)-5-(s-indacen-1-yl)pyridine", "pin_verified"),
]


@pytest.mark.parametrize("smiles,expected,tier", BOND_SHIFT_DRAWING_ROWS)
def test_bond_shift_drawing_of_a_fused_mancude_parent(smiles, expected, tier):
    from orthonym import Orthonym
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert name == expected, name
    assert res.get("tier") == tier, res.get("tier")
    # A1-5: the PIN tier gives the same name the same label
    pin = _dt_row(smiles)
    assert (pin.get("name"), pin.get("tier")) == (expected, tier), pin


# --------------------------------------------------------------------------
# A2-3 N-substituted ring carboxamides and amines
# --------------------------------------------------------------------------

RING_AMIDE_ROWS = [
    ("CN(C1CCCCC1)C(=O)C2=C[C@@H](C[C@@H](O2)OCC3=CC=C(C=C3)CO)C4CCCCC4",
     "(2R,4S)-N,4-dicyclohexyl-2-{[4-(hydroxymethyl)phenyl]methoxy}-N-methyl-"
     "3,4-dihydro-2H-pyran-6-carboxamide"),
    ("CN(C1CCCCC1)C(=O)C1=CC(C2CCCCC2)CCO1",
     "N,4-dicyclohexyl-N-methyl-3,4-dihydro-2H-pyran-6-carboxamide"),
    ("CN(C1CCCCC1)C(=O)c1ccc(C2CCCCC2)o1", "N,5-dicyclohexyl-N-methylfuran-2-carboxamide"),
    ("CN(C1CCCCC1)C(=O)c1ccc(C2CCCCC2)nc1",
     "N,6-dicyclohexyl-N-methylpyridine-3-carboxamide"),
    ("CN(C)C(=O)c1ccc(C)nc1", "N,N,6-trimethylpyridine-3-carboxamide"),
    ("CNC(=O)c1ccc(C2CCCCC2)nc1", "6-cyclohexyl-N-methylpyridine-3-carboxamide"),
    ("CN(C1CCCCC1)C(=O)c1ccc(Cl)nc1", "6-chloro-N-cyclohexyl-N-methylpyridine-3-carboxamide"),
    ("CC(C)(C)NC(=O)c1ccc(Cl)nc1", "N-tert-butyl-6-chloropyridine-3-carboxamide"),
    ("CNc1ccc(Cl)cn1", "5-chloro-N-methylpyridin-2-amine"),
    ("CC(C)(C)Nc1nc(NC2CC2)ncn1", "N4-tert-butyl-N2-cyclopropyl-1,3,5-triazine-2,4-diamine"),
    ("NC(=O)c1ccc(OCc2ccc(CO)cc2)o1", "5-{[4-(hydroxymethyl)phenyl]methoxy}furan-2-carboxamide"),
    ("OC(=O)c1ccc(OCc2ccc(CO)cc2)nc1",
     "6-{[4-(hydroxymethyl)phenyl]methoxy}pyridine-3-carboxylic acid"),
    # unchanged controls
    ("NC(=O)c1ccc(C2CCCCC2)nc1", "6-cyclohexylpyridine-3-carboxamide"),
    ("CN(C)C(=O)c1cc(C)n(C)n1", "N,N,1,5-tetramethyl-1H-pyrazole-3-carboxamide"),
    ("CC1=CC=C(O1)C(=O)NO", "N-hydroxy-5-methylfuran-2-carboxamide"),
    ("COc1ccc(COc2ccc(C(N)=O)o2)cc1", "5-[(4-methoxyphenyl)methoxy]furan-2-carboxamide"),
]


@pytest.mark.parametrize("smiles,expected", RING_AMIDE_ROWS)
def test_n_substituted_ring_carboxamide_spelling(smiles, expected):
    name, tier = _pin_verified_name(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


# --------------------------------------------------------------------------
# A2-5 isotope descriptors: after an italic locant, a locant that names the
# labelled position, the locant of a D on a parent nitrogen
# --------------------------------------------------------------------------

ISOTOPE_DESCRIPTOR_ROWS = [
    # (:43718): the descriptor follows the italic element locant
    ("[2H]C([2H])([2H])NC(=O)c1ccccc1", "N-(2H3)methylbenzamide", "pin_verified"),
    ("[2H]C([2H])([2H])NC(C)=O", "N-(2H3)methylacetamide", "pin_verified"),
    ("[2H]C([2H])([2H])Nc1ccccc1", "N-(2H3)methylaniline", "pin_verified"),
    ("[2H]C([2H])([2H])SC(C)=O", "S-(2H3)methyl ethanethioate", "pin_verified"),
    ("[2H]C([2H])([2H])NCCc1ccccc1", "N-(2H3)methyl-2-phenylethan-1-amine", "pin_verified"),
    # (1) (:15813): C1 of ethyl bears the free valence, two H only
    ("[2H]C([2H])([2H])Cc1ccccc1", "(2,2,2-2H3)ethylbenzene", "pin_verified"),
    ("[2H]C([2H])([2H])Cc1ccc(C(N)=O)cc1", "4-(2,2,2-2H3)ethylbenzamide", "pin_verified"),
    ("[2H]C([2H])([2H])Cc1ccc(C(=O)O)cc1", "4-(2,2,2-2H3)ethylbenzoic acid", "pin_verified"),
    ("[2H]C([2H])([2H])Cc1ccncc1", "4-(2,2,2-2H3)ethylpyridine", "pin_verified"),
    ("[2H]C([2H])([2H])CNC(C)=O", "N-(2,2,2-2H3)ethylacetamide", "pin_verified"),
    # (:44202): a D on a parent nitrogen with other H positions
    ("[2H]Nc1ccccc1", "(N-2H1)aniline", "pin_verified"),
    ("[2H]N1CCCCC1", "(1-2H)piperidine", "pin_verified"),
    ("[2H]N1CCCC1", "(1-2H)pyrrolidine", "pin_verified"),
    ("[2H]N1CCOCC1", "(4-2H)morpholine", "pin_verified"),
    ("[2H]N1CCNCC1", "(1-2H)piperazine", "pin_verified"),
    ("[2H]n1ccnc1", "(1-2H)-1H-imidazole", "pin_verified"),
    ("[2H]n1cccc1", "(1-2H)-1H-pyrrole", "pin_verified"),
    ("[2H]N1CCC1", "(1-2H)azetidine", "pin_verified"),
    ("[2H]N1CCNC1", "(1-2H)imidazolidine", "pin_verified"),
    ("[2H]N1CCC(C)CC1", "4-methyl(1-2H)piperidine", "pin_verified"),
    ("[2H]NC", "(N-2H1)methanamine", "pin_verified"),
    ("[2H]NCc1ccccc1", "1-phenyl(N-2H1)methanamine", "pin_verified"),  # (Task 11)
    # the Blue Book cites 'N' once ('(N-2H2)aniline (PIN)',:43830), which OPSIN
    # 2.9.0 cannot read: the readable spelling ships below the PIN, a correct
    # systematic name that is not the PIN (user decision 2026-09-30)
    ("[2H]N([2H])c1ccccc1", "(N,N-2H2)aniline", "systematic_verified"),
    # unchanged: a prefix's one position, the suffix slot of -ol, a monocycle
    ("[2H]N([2H])c1ccc(C(=O)O)cc1", "4-(2H2)aminobenzoic acid", "pin_verified"),
    ("[2H]NC(C)=O", "(N-2H1)acetamide", "pin_verified"),
    # One H per ring position: "When polysubstitution at a single position is
    # possible, the number of atoms substituted is always specified as a right
    # subscript", the Blue Book) does not apply, so the preferred
    # spelling is '(2H)benzene', which OPSIN 2.9.0 reads as 2H-benzene; the
    # readable '(2H1)' fallback ships below the PIN (a performance pass,).
    # spelling a performance pass records the forced count subscript as not the PIN form; a
    # verified name the code records as not the PIN is systematic_verified (the
    # paper, Methods, "Tiers"; user decision 2026-09-30)
    ("[2H]c1ccccc1", "(2H1)benzene", "systematic_verified"),
    ("[2H]C([2H])([2H])C(C)C=O", "2-methyl(3,3,3-2H3)propanal", "pin_verified"),
]


@pytest.mark.parametrize("smiles,expected,tier", ISOTOPE_DESCRIPTOR_ROWS)
def test_isotope_descriptor_placement(smiles, expected, tier):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert name == expected, name
    assert res.get("tier") == tier, res.get("tier")


# --------------------------------------------------------------------------
# A2-6 the class-2 grouping of chain amides, chain amines and ring parents
#: 'N,2-diphenylacetamide', not '2-phenyl-N-phenylacetamide'.
# (the Blue Book) (b) (:7067); '*N*,4-dimethyl-*N*-(3-methyl-
# phenyl)benzamide (PIN)' (:32879), '*N*,*N*,2-trimethyl-...propanamide (PIN)'
# (:21624); locants in the order (:3195).
# --------------------------------------------------------------------------

IDENTICAL_N_AND_C_PREFIX_ROWS = [
    ("O=C(Cc1ccccc1)Nc1ccccc1", "N,2-diphenylacetamide"),
    ("CNC(=O)C(C)C", "N,2-dimethylpropanamide"),
    ("CN(C)C(=O)C(C)C", "N,N,2-trimethylpropanamide"),
    ("CNC(=O)C1CCC(C)CC1", "N,4-dimethylcyclohexane-1-carboxamide"),
    ("CNC1CCC(C)CC1", "N,4-dimethylcyclohexan-1-amine"),
    ("CNc1ccnc(C)c1", "N,2-dimethylpyridin-4-amine"),
    ("CNC(=O)c1cccn1C", "N,1-dimethyl-1H-pyrrole-2-carboxamide"),
    ("C1CC1Nc1ccnc(NC2CC2)n1", "N2,N4-dicyclopropylpyrimidine-2,4-diamine"),
]


@pytest.mark.parametrize("smiles,expected", IDENTICAL_N_AND_C_PREFIX_ROWS)
def test_identical_n_and_c_prefixes_are_one_group(smiles, expected):
    name, tier = _pin_verified_name(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


def test_amide_with_identical_prefixes_merged_direct():
    from orthonym.assembly.composer import _amide_with_identical_prefixes_merged as merge
    # an acyl prefix and an N-substituent of one name: one group, N first
    assert merge([("2-phenyl", "phenyl", [2])], [{"name": "phenyl"}],
                 "acetamide") == "N,2-diphenylacetamide"
    # the acyl prefix recognised from its rendered text alone
    assert merge([("2-methyl", None, [2])], [{"name": "methyl"}],
                 "propanamide") == "N,2-dimethylpropanamide"
    # no shared name: a performance pass cites the N-substituent in the one alphanumerical
    # series all the same, the Blue Book;,:3477)
    assert merge([("2-methyl", "methyl", [2])], [{"name": "ethyl"}],
                 "propanamide") == "N-ethyl-2-methylpropanamide"
    # nothing to merge or reorder: the caller keeps its own spelling
    assert merge([("2-methyl", "methyl", [2])], [], "propanamide") is None
    # the context (identical_prefixes_cited_apart) keeps them apart,
    # the italic N first,:3195)
    from orthonym.assembly.composition_primitives import identical_prefixes_cited_apart
    with identical_prefixes_cited_apart():
        assert merge([("2-phenyl", "phenyl", [2])], [{"name": "phenyl"}],
                     "acetamide") == "N-phenyl-2-phenylacetamide"


def test_identical_prefix_names_to_merge_direct():
    from orthonym.rules.heterocycles import _identical_prefix_names_to_merge as names
    assert names([("methyl", "N")], {"methyl": [2]}, {}) == {"methyl"}
    assert names([("methyl", "N")], {}, {"methyl": [1]}) == {"methyl"}
    assert names([("cyclopropyl", "N2"), ("cyclopropyl", "N4")], {}, {}) == {"cyclopropyl"}
    assert names([("ethyl", "N")], {"methyl": [2]}, {}) == set()
    from orthonym.assembly.composition_primitives import identical_prefixes_cited_apart
    with identical_prefixes_cited_apart():
        assert names([("methyl", "N")], {"methyl": [2]}, {}) == set()


# --------------------------------------------------------------------------
# (lane-3 re-review) the ester prefixes of the noncarbon oxoacids,
# derived from the acyl prefixes: 'nitrooxy' under a class junior to esters is
# not the PIN, the Blue Book-35918;,:36401;
# '2-hydroxyethyl nitrate' is the PIN of the first row)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected,tier", [
    # quick-wins Q4e: the nitrate ester outranks the junior group,:18182) and names
    # the molecule; was '2-(nitrooxy)ethan-1-ol' (systematic_verified)
    ("OCCO[N+](=O)[O-]", "2-hydroxyethyl nitrate", "pin_verified"),
    # the ester is the only characteristic group: its functional-class name is
    # built,:35918, "Alkyl groups, aryl groups, etc. are cited as
    # separate words... followed by the name of the appropriate anion")
    ("CCO[N+](=O)[O-]", "ethyl nitrate", "pin_verified"),
    # quick-wins Q4e (as the first row): the amide's prefix is '3-amino-3-oxopropyl', as
    # '5-(2-amino-2-oxoethyl)furan-2-carboxylic acid (PIN)' (:32940); the old names were
    # '3-(nitrooxy)propanamide', '4-(nitrooxy)phenol', '4-(nitrooxy)butan-2-one' and
    # '2-(nitrooxy)ethan-1-amine' (systematic_verified)
    ("NC(=O)CCO[N+](=O)[O-]", "3-amino-3-oxopropyl nitrate", "pin_verified"),
    ("Oc1ccc(O[N+](=O)[O-])cc1", "4-hydroxyphenyl nitrate", "pin_verified"),
    ("CC(=O)CCO[N+](=O)[O-]", "3-oxobutyl nitrate", "pin_verified"),
    ("NCCO[N+](=O)[O-]", "2-aminoethyl nitrate", "pin_verified"),
    ("OCC(O[N+](=O)[O-])CO[N+](=O)[O-]", "2,3-bis(nitrooxy)propan-1-ol", "systematic_verified"),
    ("NC(=O)CCOS(=O)(=O)O", "3-(sulfooxy)propanamide", "systematic_verified"),
    # a senior head keeps the prefix PIN ('2-(tert-butylimino)-3-methyl-3-
    # (nitrooxy)butanoic acid (PIN)',:25963; '3-(sulfooxy)propanoic acid (PIN)')
    ("OC(=O)CCO[N+](=O)[O-]", "3-(nitrooxy)propanoic acid", "pin_verified"),
    ("CCOC(=O)CCO[N+](=O)[O-]", "ethyl 3-(nitrooxy)propanoate", "pin_verified"),
    ("OC(=O)CCOS(=O)(=O)O", "3-(sulfooxy)propanoic acid", "pin_verified"),
])
def test_oxoacid_ester_prefix_label(smiles, expected, tier):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert name == expected, name
    assert res.get("tier") == tier, res.get("tier")


def test_oxoacid_ester_prefixes_come_from_the_prefix_table():
    from orthonym.rules.pin_vocabulary import _oxoacid_ester_prefix_re
    rx = _oxoacid_ester_prefix_re()
    for word in ("nitrooxy", "nitrosooxy", "sulfooxy", "phosphonooxy", "arsonooxy",
                 "boronooxy", "sulfinooxy", "nitrosulfanyl"):
        assert rx.search(f"2-({word})ethan-1-ol"), word
    for word in ("nitro", "hydroxy", "carboxy", "methoxy", "acetyloxy"):
        assert not rx.search(f"2-({word})ethan-1-ol"), word


# --------------------------------------------------------------------------
# (lane-3 re-review) binary names of salts of element ions: the cation
# name followed by the anion name, the Blue Book), the anion of
# a mononuclear parent hydride by its '-ide' name,:40900-40902)
# where OPSIN 2.9.0 reads no structure from the table word; 'bis' for a
# multiplied hydride anion (d),:7037). No PIN for these inorganic
# compounds,:4667;,:2056-2058).
# --------------------------------------------------------------------------

# Labels: a carbon-free salt takes the label of its naming path, as in the paper's
# measured run (user decision 2026-09-30, 'sodium chloride' pin_verified); an
# aluminium compound keeps the no-PIN-status label (the Blue Book,:2062) and the
# metal carbide the / one.
@pytest.mark.parametrize("smiles,expected,tier", [
    ("[Al+3].[N-3]", "aluminium azanetriide", "systematic_verified"),
    ("[Mg+2].[Mg+2].[Mg+2].[P-3].[P-3]", "trimagnesium bis(phosphanetriide)",
     "pin_verified"),
    ("[Ca+2].[Ca+2].[Ca+2].[P-3].[P-3]", "tricalcium bis(phosphanetriide)",
     "pin_verified"),
    ("[Fe+3].[N-3]", "iron(III) azanetriide", "pin_verified"),
    ("[Mg+2].[Mg+2].[C-4]", "dimagnesium methanetetraide", "systematic_verified"),
    # table names OPSIN reads stay
    ("[Li+].[Li+].[Li+].[N-3]", "lithium nitride", "pin_verified"),
    ("[Na+].[Na+].[S-2]", "disodium sulfide", "pin_verified"),
    ("[Na+].[Cl-]", "sodium chloride", "pin_verified"),
])
def test_binary_salt_of_element_ions(smiles, expected, tier):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert name == expected, name
    assert res.get("tier") == tier, res.get("tier")


def test_multiplied_hydride_anion_takes_bis():
    from orthonym.rules.salts import _apply_stoichiometric_prefix
    assert _apply_stoichiometric_prefix("phosphanetriide", 2) == "bis(phosphanetriide)"
    assert _apply_stoichiometric_prefix("sulfanediide", 3) == "tris(sulfanediide)"
    assert _apply_stoichiometric_prefix("chloride", 2) == "dichloride"
