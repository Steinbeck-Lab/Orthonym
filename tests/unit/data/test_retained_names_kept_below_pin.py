"""Retained table names that are not PINs and stay in the table: labelled below PIN, declined at the default tier.

User decision 2026-10-20 (PS lane 1 a performance pass): the PIN label of glycine, pentazolidine and the
nucleoside and nucleotide table names is lowered ("Keep the names but label them below PIN
(systematic_verified). The default tier would then decline them."). The wider class (sugars,
steroids, N-acetylneuraminic acid) is not part of the decision. The relabel of 2026-10-31
(internal notes) extends it to the to names (sugars, all amino
acids, cyclitols, nucleosides); the list names (steroids) keep their labels.

Blue Book basis, each anchor opened (the Blue Book Blue Book):

* section "PREFERRED IUPAC NAMES" (:1978),:2052: "Preferred IUPAC names (PINs) for the
  natural products in Chapter are not identified."; section " INTRODUCTION" (:50939),
  :50943: "Preferred IUPAC names (PINs) are not identified for the compounds in this Chapter."
* glycine: Table 10.4 "Retained names of 'common' alpha-amino acids" (section,
  :54182) prints 'glycine / aminoacetic acid' at:54202, no (PIN) tag.
* nucleosides: section "RETAINED NAMES OF NUCLEOSIDES" (:54943), "The following names are
  retained." (adenosine... uridine,:54947-:54959), no (PIN) tag; (:54963) for the
  2'-deoxy names; nucleotides: section "RETAINED NAMES" (:55003), "The following are
  traditional names for esters of nucleosides with phosphoric acid."
* pentazolidine: ':16964 pentazolidine (preselected name; see;...)' (section,
  :16926), GLOSSARY:1870 "Preselected name. A name chosen for a noncarbon (inorganic) structure to be
  used as the basis for derivation of a preferred IUPAC name", PREFACE:1639 "the decision to accept
  them as 'preferred IUPAC names' will be made later".

Mechanism (the one the three adenylic acids have had since): a ``pin: false`` row with
``hc_override`` in ``data/iupac_2013_pin_list.json`` keeps the name in the hand-curated table, and
the table-hit site (``routing.dispatch_table._handle_retained_name``) records a kept name that the
list denies as a non-PIN fragment, so the name is labelled systematic_verified and the default tier
declines it. The glycine ZWITTERION is named from its own lookup (``rules.salts.name_zwitterion``,
``RETAINED_AMINO_ACID_ZWITTERIONS``) that never passes the table-hit site; it records its name
through ``data.amino_acids.record_if_non_pin_amino_acid_name``, which reads the same deny set, so
the name 'glycine' carries one label whichever drawing is named. Only 'glycine' is a denied name
of the amino-acid tables (alanine, serine, beta-alanine... keep pin_verified). A deny WITHOUT hc_override would have withdrawn the name: measured, glycine then ships
'2-aminoethanoic acid' and the nucleosides ship long substitutive names labelled pin_verified.

Changed asserted values: none in this file (new rows). Independent check: every best-effort name
reads back to the full InChIKey of its structure with OPSIN 2.9.0 (``name_is_rt_exact`` is a separate
OPSIN call). The tests fail on the tree before the change (an A/B check).
"""
import pytest
from rdkit import Chem

from orthonym.data import ALL_RETAINED_NAMES
from orthonym.data.pin_policy import PIN_DENY, PIN_DENY_HC
from tests.support.pin_tiers import name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

_ROWS = [
    ("NCC(=O)O", "glycine"),
    ("[nH]1[nH][nH][nH][nH]1", "pentazolidine"),
    ("Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O", "adenosine"),
    ("Nc1nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1", "guanosine"),
    ("Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1", "cytidine"),
    ("Cc1cn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]c1=O", "thymidine"),
    ("O=c1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1", "uridine"),
    ("O=c1[nH]cnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O", "inosine"),
    ("Nc1ncnc2c1ncn2[C@H]1C[C@H](O)[C@@H](CO)O1", "2'-deoxyadenosine"),
    ("Nc1nc2c(ncn2[C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1", "2'-deoxyguanosine"),
    ("Nc1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)n1", "2'-deoxycytidine"),
    ("O=c1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1", "2'-deoxyuridine"),
    ("Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O", "5'-adenylic acid"),
    ("Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1OP(=O)(O)O", "2'-adenylic acid"),
    ("Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](OP(=O)(O)O)[C@H]1O", "3'-adenylic acid"),
    ("O=c1[nH]cnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O", "5'-inosinic acid"),
]


@pytest.mark.parametrize("smiles, name", _ROWS)
def test_the_name_is_kept_in_the_table_and_denied_as_a_pin(smiles, name):
    """Kept (hand-curated side, hc_override) and denied (PIN_DENY, not PIN_DENY_HC)."""
    assert ALL_RETAINED_NAMES.get(Chem.CanonSmiles(smiles)) == name
    assert name.lower() in PIN_DENY
    assert name.lower() not in PIN_DENY_HC


@pytest.mark.parametrize("smiles, name", _ROWS)
def test_the_default_tier_declines_the_name(smiles, name):
    row = name_default(smiles)
    assert (row.get("tier"), row.get("is_pin"), row.get("limit_code")) == (
        "abstain", False, "NO_VERIFIED_PIN"), row


@pytest.mark.parametrize("smiles, name", _ROWS)
def test_best_effort_ships_the_name_labelled_systematic_verified(smiles, name):
    row = name_breadth(smiles)
    assert (row.get("name"), row.get("tier"), row.get("is_pin")) == (
        name, "systematic_verified", False), row
    assert name_is_rt_exact(name, smiles), name


@pytest.mark.parametrize("smiles, name", [
    # table names that stay PINs: the label moves only for the rows of the list above
    ("C1CNC1", "azetidine"),
    ("C#C", "acetylene"),
    ("COc1ccccc1", "anisole"),
    ("CN(C)C=O", "dimethylformamide"),
])
def test_other_table_names_keep_their_pin_label(smiles, name):
    for row in (name_default(smiles), name_breadth(smiles)):
        assert (row.get("name"), row.get("tier"), row.get("is_pin")) == (name, "pin_verified", True), row


# the other drawing of glycine, named from the zwitterion lookup of rules/salts.py
_ZWITTERION_ROWS = [("[NH3+]CC(=O)[O-]", "glycine")]


@pytest.mark.parametrize("smiles, name", _ZWITTERION_ROWS)
def test_the_glycine_zwitterion_is_declined_at_default_and_labelled_below_pin_at_best_effort(smiles, name):
    row = name_default(smiles)
    assert (row.get("tier"), row.get("is_pin"), row.get("limit_code")) == (
        "abstain", False, "NO_VERIFIED_PIN"), row
    row = name_breadth(smiles)
    assert (row.get("name"), row.get("tier"), row.get("is_pin")) == (
        name, "systematic_verified", False), row
    assert name_is_rt_exact(name, smiles), name


@pytest.mark.parametrize("smiles, name", [
    # relabel (2026-10-31, the user's delegated ruling, internal notes): the
    # other amino-acid names are no PIN either:50943, the Table 10.4 rows print no '(PIN)'); the
    # spelling check validation/spelling/checks_natural_products.py lowers them. Changed asserted value:
    # 'pin_verified' -> declined at the default tier and systematic_verified at best-effort; independent
    # check: name_is_rt_exact (OPSIN reads the name back to the input); mutation: with the check removed
    # from pin_spelling.BUILTIN_CHECK_MODULES the default tier ships the name pin_verified again.
    ("C[C@H](N)C(=O)O", "L-alanine"),
    ("C[C@H]([NH3+])C(=O)[O-]", "L-alanine"),
    ("N[C@@H](CO)C(=O)O", "L-serine"),
    ("[NH3+][C@@H](CO)C(=O)[O-]", "L-serine"),
    ("[NH3+]CCC(=O)[O-]", "beta-alanine"),
])
def test_other_amino_acid_names_are_labelled_below_pin(smiles, name):
    row = name_default(smiles)
    assert (row.get("tier"), row.get("is_pin"), row.get("limit_code")) == (
        "abstain", False, "NO_VERIFIED_PIN"), row
    row = name_breadth(smiles)
    assert (row.get("name"), row.get("tier"), row.get("is_pin")) == (
        name, "systematic_verified", False), row
    assert name_is_rt_exact(name, smiles), name
