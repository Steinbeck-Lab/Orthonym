""" -- acetonitrile as the SUBSTITUTABLE retained parent of the two-carbon nitrile.

Blue Book section heading `` Nitriles derived from retained names of carboxylic
acids``, paragraph ```` (the Blue Book): "The following names are preferred
IUPAC names, with unlimited substitution, except for formonitrile whose substitution rules are
the same as formic acid (see and, obviously, for oxalonitrile for which no
substitution is possible." The row ``CH3-CN acetonitrile (PIN) ethanenitrile`` (:34754) makes
``ethanenitrile`` the non-preferred alternative, exactly as ``ethanoic acid`` is for acetic acid
,:29725).

Like acetic acid, CH3-CN has one substitutable position (the alpha carbon: the nitrile carbon
carries no hydrogen), so (:3031, "All locants are omitted for parent compounds when
all substitutable hydrogen atoms have the same locant") and (:7304, "Simple
substituents of a parent where only one possible position can be substituted do not require
locants. Enclosing marks are used with second and subsequent simple substituents") give
``cyclopropyl(fluoro)acetonitrile (PIN)`` (:7314), and the stereodescriptor of the alpha
carbon is cited bare when no locant is present,:44643, "preceded by a numerical or
letter locant... when such locants are present").

Verbatim Blue Book PINs asserted below: ``(3-bromophenyl)acetonitrile`` "Substituted
nitriles",:34827), ``cyclopropyl(fluoro)acetonitrile``,:7314),
``hydrazinylacetonitrile``,:38545), ``aminoacetonitrile``,:34787),
``trichloroacetonitrile`` "as in trichloroacetonitrile",:44182).

Before the fix every one of these was spelled ``...ethanenitrile`` and labelled ``pin_verified``
(a non-PIN spelling carrying the PIN label). Every emitted name OPSIN-round-trips to the input.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.composition_primitives import RETAINED_C2_PARENT_BY_SUFFIX
from orthonym.cli import _emit_tier_flags
from orthonym.rules.pin_vocabulary import non_pin_vocabulary
from orthonym.validation.opsin_roundtrip import opsin_parse

# the default-tier labels (pin_verified / is_pin) exist only with the OPSIN validity gate on
pytestmark = pytest.mark.opsin_gate

# (smiles, PIN). Blue Book verbatim rows are marked; the rest follow the acetic-acid arm of the
# same rule Type 2a: substitution by prefixes junior to the suffix).
TARGETS = [
    ("N#CC(c1ccccc1)c1ccccc1", "diphenylacetonitrile"),
    ("N#CCc1cccc(Br)c1", "(3-bromophenyl)acetonitrile"),          #:34827 (PIN)
    ("N#CC(F)C1CC1", "cyclopropyl(fluoro)acetonitrile"),          #:7314 (PIN)
    ("N#CCNN", "hydrazinylacetonitrile"),                         #:38545 (PIN)
    ("N#CC(Cl)(Cl)Cl", "trichloroacetonitrile"),                  #:44182
    ("N#CCN", "aminoacetonitrile"),                               #:34787
    ("N#CC(O)(c1ccccc1)c1ccccc1", "hydroxydi(phenyl)acetonitrile"),
    ("N#CC(c1ccc(Cl)cc1)c1ccccc1", "(4-chlorophenyl)(phenyl)acetonitrile"),
    ("N#CC(Br)(Cl)c1ccccc1", "bromo(chloro)(phenyl)acetonitrile"),
    ("N#C[C@H](O)c1ccccc1", "(R)-hydroxy(phenyl)acetonitrile"),   # bare alpha descriptor,
    ("N#CC(=NO)c1ccccc1", "(hydroxyimino)(phenyl)acetonitrile"),
    ("N#CC=C1CCCCC1", "cyclohexylideneacetonitrile"),
    ("CCN(C)CC#N", "[ethyl(methyl)amino]acetonitrile"),
    ("N#CCOC", "methoxyacetonitrile"),
    ("N#CCc1ccc(O)cc1", "(4-hydroxyphenyl)acetonitrile"),
    ("N#CCc1c[nH]c2ccccc12", "(1H-indol-3-yl)acetonitrile"),
    ("N#CC(F)(F)F", "trifluoroacetonitrile"),
    ("N#CCSC", "(methylsulfanyl)acetonitrile"),
    ("N#CCn1ccnc1", "(1H-imidazol-1-yl)acetonitrile"),
]

# Rows the retained arm must NOT touch: the unsubstituted parent, a longer chain, the retained
# formonitrile/oxalonitrile/benzonitrile families, ring carbonitriles, acyl cyanides
#:34797, "Compounds of the type R-CO-CN can be named as acyl cyanides"),
# a nitrile junior to another principal group 'cyano'), and the multiplied
# parents ('2,2'-azanediyldiacetonitrile',.
CONTROLS = [
    ("CC#N", "acetonitrile"),
    ("[2H]C([2H])([2H])C#N", "(2H3)acetonitrile"),                 #:44182 (PIN)
    ("CCC#N", "propanenitrile"),
    ("N#CCC#N", "propanedinitrile"),                              # no 'cyanoacetonitrile'
    ("N#CC#N", "oxalonitrile"),
    ("N#CC(N)C", "2-aminopropanenitrile"),
    ("N#CC1CC1", "cyclopropanecarbonitrile"),
    ("N#CC(=O)c1ccccc1", "benzoyl cyanide"),
    ("N#CC=O", "formyl cyanide"),                                 #:34799 (PIN)
    ("N#CCC(=O)O", "cyanoacetic acid"),
    ("N#CCNCC#N", "2,2'-azanediyldiacetonitrile"),
    ("N#CCc1ccc(CC#N)cc1", "2,2'-(1,4-phenylene)diacetonitrile"),
]


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


@pytest.fixture(scope="module")
def default_engine():
    return Orthonym()


@pytest.fixture(scope="module")
def best_effort_engine():
    return Orthonym(**_emit_tier_flags("best-effort"))


def test_retained_parent_table_is_the_acid_and_the_nitrile():
    """One table drives the acid and nitrile arms of both producers."""
    assert RETAINED_C2_PARENT_BY_SUFFIX == {"oic acid": "acetic acid", "nitrile": "acetonitrile"}


@pytest.mark.parametrize("smiles,expected", TARGETS)
def test_substituted_acetonitrile_is_the_pin_at_the_default_tier(default_engine, smiles, expected):
    row = default_engine.name_tiered(smiles)
    assert row["name"] == expected, f"{smiles}: got {row['name']!r}"
    # a name is pin_verified only if it is the PIN
    assert row["tier"] == "pin_verified" and row["is_pin"] is True, row
    rt = opsin_parse(row["name"])
    assert rt and _inchikey(rt) == _inchikey(smiles), f"round trip failed for {row['name']!r}"


@pytest.mark.parametrize("smiles,expected", TARGETS)
def test_substituted_acetonitrile_is_the_same_name_at_best_effort(best_effort_engine, smiles, expected):
    """The wider tier must not keep the non-preferred spelling the default tier gave up."""
    got = best_effort_engine.name_tiered(smiles)["name"]
    assert got == expected, f"{smiles}: got {got!r}"


@pytest.mark.parametrize("smiles,expected", CONTROLS)
def test_retained_arm_does_not_over_fire(default_engine, smiles, expected):
    assert default_engine.name_tiered(smiles)["name"] == expected


def test_spelling_vocabulary_marks_ethanenitrile_non_pin():
    """ (:34754): ``ethanenitrile`` is the alternative of the PIN acetonitrile, as
    ``ethanoic acid`` is of acetic acid. A name that still carries it is never labelled a PIN."""
    for bad in ("2-hydroxyethanenitrile", "phenylethanenitrile", "2,2'-azanediyldiethanenitrile",
                "(2E)-2-(hydroxyimino)-2-phenylethanenitrile"):
        assert non_pin_vocabulary(bad) == "ethanenitrile", bad
    for good in ("acetonitrile", "(3-bromophenyl)acetonitrile", "propanenitrile",
                 "bromomethanenitrile", "azidomethanenitrile", "butanenitrile", "oxalonitrile"):
        assert non_pin_vocabulary(good) is None, good


def test_a_chain_stereo_double_bond_is_never_labelled_pin(default_engine, best_effort_engine):
    """The retained arm declines an E/Z bond on the two carbons (no book row cites its locant
    on a retained name), as it does for acetic acid; the located systematic name that remains
    is a right-molecule name, never labelled PIN: the default tier declines it and the
    best-effort tier ships it below the PIN."""
    smi = "N#C/C(=N/O)c1ccccc1"
    d = default_engine.name_tiered(smi)
    assert d["tier"] != "pin_verified" and not d["is_pin"], d
    b = best_effort_engine.name_tiered(smi)
    assert b["name"] == "(2E)-2-(hydroxyimino)-2-phenylethanenitrile", b
    assert b["tier"] not in ("pin_verified", "pin_unverified") and not b["is_pin"], b
    rt = opsin_parse(b["name"])
    assert rt and _inchikey(rt) == _inchikey(smi), b


GLUCOSIDES = [
    # glycosyloxy prefix; the prefixes are cited in alphanumerical order,
    #:3446: Greek letters and the stereodescriptor 'D' do not count, so 'glucopyranosyloxy'
    # precedes 'hydroxyphenyl'), as for the acid analogue
    # '(R)-(β-D-glucopyranosyloxy)(4-hydroxyphenyl)acetic acid'.
    ("N#C[C@H](O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O)c1ccc(O)cc1",
     "(R)-(β-D-glucopyranosyloxy)(4-hydroxyphenyl)acetonitrile"),
    ("N#C[C@@H](O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O)c1ccc(O)cc1",
     "(S)-(β-D-glucopyranosyloxy)(4-hydroxyphenyl)acetonitrile"),
    ("N#C[C@H](O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O)c1ccccc1",
     "(R)-(β-D-glucopyranosyloxy)(phenyl)acetonitrile"),
]


@pytest.mark.parametrize("smiles,expected", GLUCOSIDES)
def test_glycoside_of_a_cyanohydrin_cites_its_prefixes_in_order(default_engine, smiles, expected):
    """The glycosyloxy placeholder splice (``sugar_names.name_glycosyloxy_aglycone``) names the
    aglycone methyl ether and swaps its 'methoxy' token. Once the ether is spelled
    '(4-hydroxyphenyl)(methoxy)acetonitrile' that token is a prefix of its own, and the splice
    gave '(4-hydroxyphenyl)[(β-D-glucopyranosyloxy)]acetonitrile' (double marks, out of order)
    at pin_verified. It declines that shape; the structural route names the glycoside.:50943 gives no PIN in Chapter (relabel 2026-10-31, internal notes):
    the glycosyloxy prefix is a name, so the name is read at the best-effort tier, below the PIN."""
    from orthonym.cli import _emit_tier_flags
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    rt = opsin_parse(row["name"])
    assert rt and _inchikey(rt) == _inchikey(smiles), row


def test_glycosyloxy_splice_declines_a_bare_enclosed_token_only():
    from orthonym.data.sugar_names import _escalate_enclosing_marks
    glyc = "(β-D-glucopyranosyloxy)"
    # a group that holds nothing but the token is a prefix of the parent: decline
    assert _escalate_enclosing_marks("(4-hydroxyphenyl)(methoxy)acetonitrile", "methoxy", glyc) is None
    # a methoxy inside a compound prefix is still spliced and its marks escalated
    assert (_escalate_enclosing_marks("1-(4-methoxyphenyl)ethan-1-one", "methoxy", glyc)
            == "1-[4-(β-D-glucopyranosyloxy)phenyl]ethan-1-one")
