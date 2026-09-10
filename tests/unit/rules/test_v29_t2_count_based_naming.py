""" — a carbon COUNT may never stand in for a STRUCTURE PROOF.

``get_alkyl_name(n)`` can spell exactly one shape: an unbranched, SATURATED,
acyclic, all-carbon chain attached at one of its two termini. ``(name, count)``
is not injective, so any site that hands it a count it has not proved to
describe that shape renames the molecule::

    -CH2CH2CH3 -> 'propyl' (honest)
    -CH2CH=CH2 -> 'propyl' (allyl spelled as the alkane)
    -CH2C#CH -> 'propyl' (propargyl spelled as the alkane)
    -CH2CH2SO3- -> 'ethyl' (the sulfonate dropped on the floor)

Three demonstrated sites did exactly that, each in a different way:

* **S3** ``substituent_enumerator._name_sulfanyl_branch`` — its heteroatom guard
  tested a list that by construction only ever held carbons, so it could never
  fire. Worse, the walk stopped at the first heteroatom, so it never VISITED the
  atoms it dropped: on CHEBI:131797 it returned ``'propylsulfanyl'`` while
  discarding a 7-heavy-atom cysteinyl-glycine arm.
* **S4** ``substituent_prefix_forms._name_alkyl_branch_from_atom`` — its
  structure proof checked rings, branching and the attachment terminus but never
  BOND ORDER, so three distinct ureas emitted one identical name.
* **S2** ``composer._name_r_group`` — nitrogen was excluded from its heteroatom
  guard by design, and the O/S/Se/Te branch carried an explicit "fall through to
  simple alkyl" escape; both funnelled into the same bare count.

All three now route through the ONE shared primitive that already asked the
right question, ``substituent_naming.fragment_is_linear_terminal_alkyl``
("is ``get_alkyl_name(carbon_count)`` an HONEST name for this fragment?"),
rather than each growing its own guard list.

Every assertion below FAILS on the pre-fix tree. The producer-level tests are
the load-bearing ones: they pin the defect at the site rather than at a
whole-molecule name that some other layer could coincidentally repair.
"""

import pytest
from rdkit import Chem

from orthonym.assembly.composer import _name_r_group
from orthonym.assembly.substituent_enumerator import _name_sulfanyl_branch
from orthonym.assembly.substituent_prefix_forms import (
    _name_alkyl_branch_from_atom,
    get_n_alkyl_carbamoyl_prefix,
    get_n_n_dialkyl_carbamoyl_prefix,
)
from orthonym.assembly.substituent_naming import (
    fragment_is_linear_terminal_alkyl,
)
from orthonym.namer import Orthonym


@pytest.fixture
def ungated_namer(monkeypatch):
    """A namer with the SELF-01 OPSIN validity gate explicitly DISABLED.

    Most of these fabrications were GATE-SUPPRESSED, so asserting on the default
    path would pass vacuously against the pre-fix tree too. The gate is the
    margin here, not the producer, and these tests are about the producer.
    """
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", True, raising=False)
    return Orthonym()


# ---------------------------------------------------------------------------
# The primitive itself — it must really prove all four properties
# ---------------------------------------------------------------------------

def test_the_shared_primitive_checks_bond_order_not_just_shape():
    """The property S4's private guard lacked, stated on the primitive.

    A propenyl fragment is an unbranched acyclic all-carbon chain attached at a
    terminus — every check S4 performed — and is still not 'propyl'.
    """
    saturated = Chem.MolFromSmiles("NCCC")
    assert fragment_is_linear_terminal_alkyl(saturated, [1, 2, 3], 1) is True

    alkene = Chem.MolFromSmiles("NCC=C")
    assert fragment_is_linear_terminal_alkyl(alkene, [1, 2, 3], 1) is False

    alkyne = Chem.MolFromSmiles("NCC#C")
    assert fragment_is_linear_terminal_alkyl(alkyne, [1, 2, 3], 1) is False


# ---------------------------------------------------------------------------
# S4 — three molecules must not share one name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("NCCC", "propyl"),            # honest: must stay byte-identical
    ("NCC=C", "prop-2-en-1-yl"),   # was 'propyl'
    ("NCC#C", "prop-2-yn-1-yl"),   # was 'propyl'
])
def test_s4_alkyl_branch_respects_bond_order(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert _name_alkyl_branch_from_atom(mol, 1, exclude={0}) == expected


def test_s4_three_distinct_ureas_do_not_collapse_to_one_name(ungated_namer):
    """THE injectivity property, at whole-molecule level.

    Pre-fix these three emitted the identical string
    '3-[(propylcarbamoyl)amino]propanoic acid'. No function of
    (fragment, carbon-count) can be right here.
    """
    names = {
        smi: ungated_namer.name(smi)
        for smi in ("OC(=O)CCNC(=O)NCCC",
                    "OC(=O)CCNC(=O)NCC=C",
                    "OC(=O)CCNC(=O)NCC#C")
    }
    assert len(set(names.values())) == 3, names
    assert names["OC(=O)CCNC(=O)NCCC"] == (
        "3-[(propylcarbamoyl)amino]propanoic acid")
    # the unsaturated two must not be spelled as the saturated alkane
    for smi in ("OC(=O)CCNC(=O)NCC=C", "OC(=O)CCNC(=O)NCC#C"):
        assert "propylcarbamoyl" not in names[smi], names[smi]


def test_s4_butenyl_is_not_butyl(ungated_namer):
    out = ungated_namer.name("OC(=O)CCNC(=O)NCC=CC")
    assert "butylcarbamoyl" not in out, out
    assert "but-2-en-1-yl" in out, out


# ---------------------------------------------------------------------------
# S4 follow-on — the enclosing marks P-16.3.4 requires once locants can arrive
# ---------------------------------------------------------------------------

def test_multiplied_alkenyl_prefix_takes_di_and_parentheses():
    """P-16.3.4 'Parentheses (round brackets)... are used to enclose multiplied
    components that are:... (b) simple substituent prefixes modified by 'ene'
    and 'yne' endings and that have locants' (the Blue Book,:7104; its own
    example is 'di(prop-1-en-2-yl)' (preferred prefix), and the Blue Book carries the
    PIN '1,1-dimethyl-3,4-di(prop-1-en-2-yl)germolane').

    'di', NOT 'bis': every clause of P-16.3.5 (:7104) is gated on the component
    being SUBSTITUTED — '(a) compound or complex (i.e. substituted) prefixes' —
    and an unsubstituted alkenyl prefix is neither.

    Before the bond-order fix only bare saturated stems reached this
    interpolation, so the missing marks were latent.
    """
    mol = Chem.MolFromSmiles("OC(=O)CCCC(=O)N(CC=C)CC=C")
    # (carbonyl_C, carbonyl_O, amide_N, alkyl_C1, alkyl_C2)
    match = mol.GetSubstructMatch(
        Chem.MolFromSmarts("[CX3](=O)[NX3]([#6])[#6]"))
    assert match, "the tertiary-amide SMARTS must match"
    out = get_n_n_dialkyl_carbamoyl_prefix(mol, match)
    # P-66.1.1.4.1.1: carbamoyl N-locant omitted; 'di' + enclosed alkenyl stem.
    assert out == "di(prop-2-en-1-yl)carbamoyl", out
    assert "bis(" not in out


def test_saturated_dialkyl_carbamoyl_is_byte_identical():
    """The no-op half: a simple stem needs no marks and must not acquire any."""
    mol = Chem.MolFromSmiles("OC(=O)CCCC(=O)N(CCC)CCC")
    match = mol.GetSubstructMatch(
        Chem.MolFromSmarts("[CX3](=O)[NX3]([#6])[#6]"))
    assert get_n_n_dialkyl_carbamoyl_prefix(mol, match) == (
        "dipropylcarbamoyl")  # P-66.1.1.4.1.1: N-locant omitted


def test_mono_n_alkyl_carbamoyl_saturated_is_byte_identical():
    # a single unambiguous secondary-amide match (the urea in the S4 reproducer
    # matches the SMARTS twice, and the first match is the side this generator
    # correctly declines)
    mol = Chem.MolFromSmiles("OC(=O)CCCC(=O)NCCC")
    matches = mol.GetSubstructMatches(
        Chem.MolFromSmarts("[CX3](=O)[NX3H1][#6]"))
    assert len(matches) == 1, matches
    out = get_n_alkyl_carbamoyl_prefix(mol, matches[0])
    assert out == "propylcarbamoyl", out  # P-66.1.1.4.1.1: N-locant omitted


def test_mono_n_alkyl_carbamoyl_alkenyl_gains_its_marks():
    """The other half of P-16.3.4: once an 'ene'-ending prefix with locants can
    reach the interpolation, it must be enclosed."""
    mol = Chem.MolFromSmiles("OC(=O)CCCC(=O)NCC=C")
    matches = mol.GetSubstructMatches(
        Chem.MolFromSmarts("[CX3](=O)[NX3H1][#6]"))
    assert len(matches) == 1, matches
    out = get_n_alkyl_carbamoyl_prefix(mol, matches[0])
    # N-locant omitted (P-66.1.1.4.1.1); the alkenyl stem keeps its marks.
    assert out == "(prop-2-en-1-yl)carbamoyl", out


# ---------------------------------------------------------------------------
# S3 — the vacuous guard
# ---------------------------------------------------------------------------

def test_s3_names_a_genuine_s_alkyl_branch():
    """The honest path must survive the fix (positive control)."""
    mol = Chem.MolFromSmiles("CSCCC")
    assert _name_sulfanyl_branch(mol, [1, 2, 3, 4], 1, {0}) == "propylsulfanyl"


@pytest.mark.parametrize("smiles,frag,label", [
    ("CSCC=C", [1, 2, 3, 4], "allyl is not propyl"),
    ("CSCC#C", [1, 2, 3, 4], "propargyl is not propyl"),
    ("CSCCO", [1, 2, 3, 4], "the hydroxyl must not be dropped"),
    ("CSCC(C)C", [1, 2, 3, 4, 5], "isobutyl is not butyl"),
])
def test_s3_refuses_what_a_count_cannot_describe(smiles, frag, label):
    """Pre-fix every one of these returned a bare '<alkane>sulfanyl'.

    The heteroatom case is the sharpest: the old guard tested a list that held
    carbons by construction, so it could never fire, and the walk stopped at the
    oxygen and therefore never saw what it discarded.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert _name_sulfanyl_branch(mol, frag, 1, {0}) is None, label


def test_s3_does_not_claim_propylsulfanyl_for_a_peptide_arm(ungated_namer):
    """CHEBI:131797 — the S-side is a whole cysteinyl-glycine arm and the name
    asserted -S-CH2CH2CH3, silently discarding 7 heavy atoms (C,C,N,N,O,O,O)."""
    out = ungated_namer.name(
        "C=CC(C)(SC[C@H](NC(=O)CC[C@H]([NH3+])C(=O)[O-])"
        "C(=O)NCC(=O)[O-])C(=O)[O-]")
    assert "propylsulfanyl" not in out, out


# ---------------------------------------------------------------------------
# S2 — the fall-through that kept counting
# ---------------------------------------------------------------------------

def test_s2_does_not_count_a_sulfonate_bearing_r_group_as_ethyl():
    """N-acyltaurine's -CH2CH2SO3- R-group was counted as 2 carbons -> 'ethyl',
    and the sulfonate was re-attached to the fatty chain: a DIFFERENT
    constitution. The n>=3 isomer-ambiguity bar does not excuse this — a
    carbon-only count drops a heteroatom at EVERY n."""
    mol = Chem.MolFromSmiles("CC(=O)NCCS(=O)(=O)[O-]")
    out = _name_r_group(mol, 4, exclude_atoms={0, 1, 2, 3})
    assert out != "ethyl", out


def test_s2_still_names_a_plain_linear_alkyl_r_group():
    """Positive control: the honest count path is untouched."""
    mol = Chem.MolFromSmiles("CC(=O)NCC")
    assert _name_r_group(mol, 4, exclude_atoms={0, 1, 2, 3}) == "ethyl"
    mol2 = Chem.MolFromSmiles("CC(=O)NCCC")
    assert _name_r_group(mol2, 4, exclude_atoms={0, 1, 2, 3}) == "propyl"


def test_s2_taurine_amide_names_correctly_end_to_end(ungated_namer):
    """The fix does not merely remove a wrong name — it UNBLOCKS the right one.

    Pre-fix this emitted '1-(ethylamino)-1-oxooctadecanesulfonate' with the gate
    off and 'unknown organic compound' with the gate on.

    UPDATE (2026-08-18, a phase lead a): the general engine's principal-
    chain selection (perception/chains.py::find_principal_chain) previously
    picked the longer all-carbon octadecanoyl chain over the sulfonic-
    acid-bearing chain even after this S2 fix, so the emitted name here was
    briefly frozen as 'N-octadecanoyl-2-aminoethane-1-sulfonate' -- an
    N-acyl-substituent form built on top of that wrong-chain selection.
    a phase lead a fixed principal-chain selection for heteroatom-only-suffix
    acids (sulfonic/sulfinic/phosphonic/phosphinic now register their
    S/P-bearing carbon so criterion 1, P-44.1, picks the correct chain), so
    this molecule now names via the same acylamido form as its siblings in
    test_acyl_taurine.py (2-acetamidoethane-1-sulfonate,
    2-formamidoethane-1-sulfonate, 2-propanamidoethane-1-sulfonate).
    RT-verified: InChIKey LMIJIHJZVURGQK-UHFFFAOYSA-M on both sides.
    """
    out = ungated_namer.name("CCCCCCCCCCCCCCCCCC(=O)NCCS(=O)(=O)[O-]")
    assert out == "2-octadecanamidoethane-1-sulfonate", out


def test_s2_does_not_fabricate_a_c43_chain(ungated_namer):
    """A phosphate-bearing arm was counted and spelled 'N-tritetracontyl' — a
    straight C43 chain asserted for a molecule that contains none."""
    out = ungated_namer.name(
        "CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)OCC(COP(=O)(O)OCCNC)"
        "OC(=O)CCCCCCCCCCCCCCCCC")
    assert "tritetracontyl" not in out, out


# ---------------------------------------------------------------------------
# Regression controls — the honest paths must stay byte-identical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)CCNC(=O)NCCC", "3-[(propylcarbamoyl)amino]propanoic acid"),
    ("OC(=O)CCNC(=O)NC", "3-[(methylcarbamoyl)amino]propanoic acid"),
    ("OC(=O)CCCC(=O)N(CCC)CCC",
     "5-(dipropylamino)-5-(dipropylcarbamoyl)pentanoic acid"),  # P-66.1.1.4.1.1: N omitted
    ("OC(=O)CCSCCC", "3-(propylsulfanyl)propanoic acid"),
    ("OC(=O)CCSCC(C)C", "3-[(2-methylpropyl)sulfanyl]propanoic acid"),
    # P-63.2.5 (the Blue Book): the PIN for a chalcogen analogue of an ether
    # is substitutive (method 1), not the functional-class "R R' sulfide" (method 2).
    ("CCSCC", "(ethylsulfanyl)ethane"),          # was "diethyl sulfide" (the Blue Book)
    ("CSc1ccccc1", "(methylsulfanyl)benzene"),   # was "methyl phenyl sulfide" (the Blue Book "(not thioanisole)")
    ("CSCCN", "2-(methylsulfanyl)ethan-1-amine"),
    ("CCCNC(=O)NCCC", "N,N'-dipropylurea"),
    ("CC(=O)NCC", "N-ethylacetamide"),
    ("CCNC(=O)OC", "methyl N-ethylcarbamate"),
    ("COC(=O)NCCC", "methyl N-propylcarbamate"),
    ("CCCB(O)O", "propylboronic acid"),
    ("CCN=C=O", "isocyanatoethane"),
    ("CCCCCCCCCCCCCCCCCC(=O)NCC", "N-ethyloctadecanamide"),
    ("CCO", "ethanol"),
])
def test_honest_names_are_unchanged(ungated_namer, smiles, expected):
    assert ungated_namer.name(smiles) == expected
