"""The label audit of the zinc names, and the failure label of a metal the engine names (ZN11).

A PIN label belongs to a name only where the Blue Book defines a PIN (the user's rule of 2026-10-12);
two classes keep the label ``pin_verified`` by name: carbon-free salts (the convention of 2026-09-30,
kept for every metal alike by the user's answer 3 of 2026-10-13) and the salts of acids, alcohols,
thiols and the other anions that the pattern of and names (the book prints no zinc
salt: the status of the zinc salt is ASSUMED). ``CLASS_BASIS`` carries, for each class of the audit
(``tests/support/zinc_label_classes.py``), the Blue Book basis (rule, heading, line, sentence) and a
status: ``VERIFIED`` (the book says what the class is), ``ASSUMED`` (the book is silent for zinc and
the label is the pattern's, listed with its row count) or ``EXCEPTION`` (a decision of the user).

The default tier names only PIN-labelled classes; every other class is ``systematic_verified`` or lower at
every tier. A zinc input that no producer names is ``UNNAMEABLE``, not ``UNSUPPORTED_ELEMENT``
(``errors.classify_failure_limit``): the failure is the engine's, the element is named
(``data.metal_scope.has_complete_scope``).

Every row is named in this process, once per tier.
"""
import csv
import functools
import logging
import re
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym, OrthonymLimitError, classify_limit
from orthonym.cli import _emit_tier_flags
from orthonym.data import metal_scope
from orthonym.data.metal_scope import has_complete_scope, is_complete_metal, is_named_metal
from orthonym.errors import classify_failure_limit, is_failure_name, unsupported_element_branch
from orthonym.perception.metals import _TRUE_METAL_SYMBOLS_FOR_VETO
from tests.support.zinc_label_classes import NON_PIN_CLASSES, PIN_CLASSES, UNPLACED, zinc_label_class

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "zinc"

#: the Blue Book basis of each class of the audit: the label policy of section 3 of
#: ``docs/the workflow tooling/plans/2026-10-12-zinc-and-organometallics.md`` (the Blue Book Blue Book, line numbers).
CLASS_BASIS = {
    # ---- classes whose names the default tier may emit (pin_verified)
    "SALT_OF_ACID": dict(
        status="ASSUMED", pin_label=True,
        rule="P-65.6.2.1", heading="P-65.6.2 Salts", line=31563,
        sentence="Neutral salts of acids are named by citing the name of the cation(s) followed by the name of the "
                 "anion (see P-72.2.2.2) as a separate word. Different cations are cited in alphabetical order.",
        note="examples: calcium diacetate (PIN) :31571, germanium tetraacetate (PIN) :31581, disodium carbonate (PIN) "
             ":31579, potassium sodium butanedioate (PIN) :31575; also P-77.3.1 (heading P-77.3 SALTS DERIVED FROM "
             "ORGANIC ACIDS, :43609), sodium acetate (PIN) :43611. A salt of two cations ('strontium zinc tetraacetate') "
             "is the pattern of :31575. No zinc salt is printed: the status of the zinc salt is ASSUMED."),
    "SALT_OF_ALCOHOL_OR_THIOL": dict(
        status="ASSUMED", pin_label=True,
        rule="P-63.8.1", heading="P-63.8 SALTS OF HYDROXY COMPOUNDS, HYDROPEROXY COMPOUNDS AND THEIR CHALCOGEN ANALOGUES",
        line=28178,
        sentence="Neutral salts of hydroxy compounds and their chalcogen analogues and peroxy compounds are named by "
                 "citing the cation(s) followed by the name of the anion as a separate word.",
        note="examples: lead tetraphenoxide (PIN) :28196, disodium benzene-1,2-bis(thiolate) (PIN) :28194; P-77.2.1 "
             "(heading P-77.2 SALTS DERIVED FROM ALCOHOLS (INCLUDING PHENOLS), PEROXOLS, AND THEIR CHALCOGEN ANALOGUES, "
             ":43590), sodium 4-methylbenzene-1-thiolate (PIN) :43599. No zinc salt is printed: ASSUMED."),
    "SALT_OF_AMIDE_ANION": dict(
        status="ASSUMED", pin_label=True,
        rule="P-72.2.2.2.4", heading="P-72.2.2.2.4 Anionic centers on other characteristic groups", line=41071,
        sentence="Anionic centers generated formally by the removal of hydrons from atoms of characteristic groups other "
                 "than those considered in P-72.2.2.2.1, P-72.2.2.2.2, and P-72.2.2.2.3 are named on the basis of the "
                 "corresponding anionic parent hydrides.",
        note="example: acetylazanide (PIN) :41079. The anion is a PIN; the book prints no salt of it with a metal "
             "(the pattern of P-65.6.2.1): ASSUMED."),
    "CYANIDE_SALT": dict(
        status="ASSUMED", pin_label=True,
        rule="P-65.5.1 (Table 6.3), P-66.5.1.2.1", heading="P-65.5.1 Acyl halides from suffix acids", line=31421,
        sentence="'cyanide' is the name of the anion -CN of the pseudohalides of Table 6.3 (:31421); "
                 "'formonitrile (PIN) ... hydrogen cyanide' (:34752, under P-66.5.1.2.1).",
        note="the book prints no cyanide of a metal and no PIN for the compound of a metal with this anion: the label "
             "is the pattern of a salt of an acid (hydrocyanic acid, P-66.5 'derived from hydrocyanic acid', :34687). "
             "ASSUMED, the weakest of the PIN classes."),
    "ORGANIC_CATION_SALT": dict(
        status="ASSUMED", pin_label=True,
        rule="P-62.6.1", heading="P-62.6.1 Cation and anion names", line=26672,
        sentence="methanaminium chloride (PIN): an organic cation named with the suffix 'ium', the anion added as a "
                 "separate word.",
        note="the pattern of the cation is the book's (:26662-26672). The anion 'zinc tetrachloride' of a zincate beside "
             "the cation has no printed name; P-12.1 PREFERRED IUPAC NAMES (:2058) leaves the rules for a compound "
             "'that cannot be named on the basis of the principles of organic nomenclature as described in this book' "
             "'to a further publication'. The label is the path's; the user's answer 3 demotes no class but the "
             "carbanion salts. ASSUMED, listed for a decision."),
    "CARBON_FREE_SALT": dict(
        status="EXCEPTION", pin_label=True,
        rule="P-12.1", heading="P-12.1 PREFERRED IUPAC NAMES", line=2058,
        sentence="Rules for the selection of preferred IUPAC names (PINs) for compounds ... that do not contain carbon, "
                 "or that cannot be named on the basis of the principles of organic nomenclature as described in this "
                 "book will be discussed in a further publication.",
        note="the Blue Book gives no PIN to a compound without carbon. The project keeps the path label of a carbon-free "
             "salt (convention of 2026-09-30; the user's answer 3 of 2026-10-13, for every metal alike: 'sodium "
             "chloride', 'copper(II) sulfate', 'zinc dichloride', 'zinc dihydride')."),
    # ---- classes whose names are never labelled a PIN
    "ORGANOZINC": dict(
        status="VERIFIED", pin_label=False,
        rule="P-69.0", heading="P-69.0 INTRODUCTION", line=39735,
        sentence="neither preferred IUPAC names or preselected names (see P-12.2) for organometallic compounds involving "
                 "the transition elements (including the Group 3 elements) and Groups 1 and 2 elements, except for "
                 "'ocene' compounds, are noted.",
        note="Group 12 is read with the transition elements by the heading P-69.2 ORGANOMETALLIC COMPOUNDS INVOLVING THE "
             "ELEMENTS IN GROUPS 3 THROUGH 12 (:39751)."),
    "CARBANION_SALT": dict(
        status="ASSUMED", pin_label=False,
        rule="P-69.3", heading="P-69.3 Groups 1 and 2 (compositional names)", line=40173,
        sentence="(sodium ethenide is also acceptable; it is a compositional name analogous to sodium chloride)",
        note="P-69.0 (:39735) notes no PIN for organometallic compounds of these metals; the compositional name of a "
             "carbanion salt of zinc is an analogy (no zinc example): the user's answer 3 puts it below the PIN."),
    "BONDED_DRAWING": dict(
        status="VERIFIED", pin_label=False,
        rule="P-68.1.6.1", heading="P-68.1.6.1 Structures for Lewis 'adducts'", line=37540,
        sentence="One of the fundamental principles of organic chemical nomenclature is to name the structure that is "
                 "drawn, and not to ascertain what an actual electronic configuration might be.",
        note="a name reached through the ionic drawing of a bonded drawing, and a name of the ligands of the metal "
             "(P-69.2.1, :39753: 'Coordination nomenclature is the primary nomenclature method used to name "
             "organometallic compounds containing elements of Groups 3 through 12'), is no PIN of the drawing as drawn."),
    "DATIVE_DRAWING": dict(
        status="VERIFIED", pin_label=False,
        rule="P-68.1.6.1", heading="P-68.1.6.1 Structures for Lewis 'adducts'", line=37540,
        sentence="One of the fundamental principles of organic chemical nomenclature is to name the structure that is "
                 "drawn, and not to ascertain what an actual electronic configuration might be.",
        note="the structure as drawn has dative bonds; a name of the split drawing names another drawing."),
    "ADDUCT": dict(
        status="VERIFIED", pin_label=False,
        rule="P-14.8.2", heading="P-14.8.2 Mixed organic - inorganic adducts", line=4667,
        sentence="preferred IUPAC names cannot be assigned to mixed adducts because preferred IUPAC names have not yet "
                 "been determined for inorganic components.",
        note="a neutral molecule or a neutral metal atom beside the metal."),
    "MACROCYCLE_COMPLEX": dict(
        status="VERIFIED", pin_label=False,
        rule="P-100", heading="P-100 INTRODUCTION (the chapter of the porphyrin parent)", line=50943,
        sentence="Preferred IUPAC names (PINs) are not identified for the compounds in this Chapter.",
        note="P-69.2.1 (:39753) and P-14.8.2 (:4667) apply as well."),
    "ATOM_OR_ION": dict(
        status="VERIFIED", pin_label=False,
        rule="P-12.1", heading="P-12.1 PREFERRED IUPAC NAMES", line=2056,
        sentence="the label 'PIN' is added to the names of compounds whose parent hydride contains at least one of the "
                 "following elements ... and that also contain at least one carbon atom in their structure",
        note="a lone atom, ion or isotope of zinc has no carbon and is no salt."),
}

ASSUMED_OR_EXCEPTION = ("ASSUMED", "EXCEPTION")


def _rows_of(path, column="smiles"):
    with open(path, newline="", encoding="utf-8") as f:
        return [r[column] for r in csv.DictReader(f, delimiter="\t")]


def _zinc_smiles():
    """The zinc drawings of the 87-pair panel and of the battery (constructed molecules, no corpus row)."""
    seen = []
    for name in ("panel_pairs.tsv", "battery_43.tsv"):
        for smiles in _rows_of(FIXTURES / name):
            if "Zn" in smiles and smiles not in seen:
                seen.append(smiles)
    return seen


#: constructed drawings beside the panel and the battery: the review leads of ZN2-ZN6 (a salt of two cations, a
#: carbanion salt beside a Group 1 cation), the dative drawings of ZN10 and the classes that the audit found
EXTRA_ROWS = [
    "[Sr+2].[Zn+2].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-]",       # strontium zinc tetraacetate:,:31575
    "[Na+].[Zn+2].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-]",                    # sodium zinc triacetate
    "[Li+].[Li+].[Zn+2].[CH3-].[CH3-].[CH3-].[CH3-]",                   # dilithium zinc tetrakis(methanide): a carbanion salt
    "Cl[Zn](Cl)(<-N)<-N",                                               # dative drawings: ZN10
    "CC(=O)[O-]->[Zn+2]<-[O-]C(C)=O",
    "C[N+](C)(C)C.[Cl-].[Cl-].[Cl-].[Zn+2]",                            # organic cation + zincate: ZN8 lead 23
    "c1cc[nH+]cc1.c1cc[nH+]cc1.[Cl-].[Cl-].[Cl-].[Cl-].[Zn+2]",
    "CS(=O)(=O)[N-]c1ccccc1.CS(=O)(=O)[N-]c1ccccc1.[Zn+2]",             # a salt of an N-H acid: the azanide anion
]

ZINC_SMILES = _zinc_smiles() + [s for s in EXTRA_ROWS if s not in _zinc_smiles()]


@functools.lru_cache(maxsize=None)
def _namer(tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier))


@functools.lru_cache(maxsize=None)
def _row(smiles, tier):
    return _namer(tier).name_tiered(smiles)


def _holds_a_metal_the_table_does_not_admit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return any(a.GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO and not is_named_metal(a.GetSymbol())
               for a in mol.GetAtoms())


# --------------------------------------------------------------------------- the table of bases

def test_every_class_has_a_basis_and_a_status():
    """No class of the audit lacks a rule, a heading, a line, a sentence or a status; the classes that
    may carry a PIN label are ASSUMED or EXCEPTION (the book prints no zinc PIN), the others say that
    they carry none."""
    assert set(CLASS_BASIS) == set(PIN_CLASSES) | set(NON_PIN_CLASSES)
    for cls, basis in CLASS_BASIS.items():
        for field in ("status", "pin_label", "rule", "heading", "line", "sentence", "note"):
            assert basis.get(field) not in (None, ""), (cls, field)
        assert basis["status"] in ("VERIFIED", "ASSUMED", "EXCEPTION"), cls
        assert isinstance(basis["line"], int) and basis["line"] > 0, cls
        assert basis["pin_label"] is (cls in PIN_CLASSES), cls
        if basis["pin_label"]:
            assert basis["status"] in ASSUMED_OR_EXCEPTION, cls


CLASSIFIER_CASES = [
    # smiles, class: the classifier reads the structure alone (no naming)
    ("CC(=O)[O-].CC(=O)[O-].[Zn+2]", "SALT_OF_ACID"),
    ("[O-]C(=O)c1ccccc1.[O-]C(=O)c1ccccc1.[Zn+2]", "SALT_OF_ACID"),
    ("[Sr+2].[Zn+2].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-]", "SALT_OF_ACID"),
    ("[Na+].[Na+].[Zn+2].[O-]C(=O)CN(CCN(CC([O-])=O)CC([O-])=O)CC([O-])=O", "SALT_OF_ACID"),
    ("CN(C)C(=S)[S-].CN(C)C(=S)[S-].[Zn+2]", "SALT_OF_ACID"),
    ("[O-]C(=O)[O-].[Zn+2]", "SALT_OF_ACID"),
    ("CC[O-].CC[O-].[Zn+2]", "SALT_OF_ALCOHOL_OR_THIOL"),
    ("[S-]c1ccccc1.[S-]c1ccccc1.[Zn+2]", "SALT_OF_ALCOHOL_OR_THIOL"),
    ("[Zn+2].[O-]c1cccc2cccnc12.[O-]c1cccc2cccnc12", "SALT_OF_ALCOHOL_OR_THIOL"),
    ("[Zn+2].C[Si](C)(C)[N-][Si](C)(C)C.C[Si](C)(C)[N-][Si](C)(C)C", UNPLACED),
    ("[C-]#N.[C-]#N.[Zn+2]", "CYANIDE_SALT"),
    ("[C-]#[N+]c1cc[n-]c1.[C-]#[N+]c1cc[n-]c1.[Zn+2]", UNPLACED),            # an isocyano group is no cyanide anion
    ("C[N+](C)(C)C.[Cl-].[Cl-].[Cl-].[Zn+2]", "ORGANIC_CATION_SALT"),
    ("[Cl-].[Cl-].[Zn+2]", "CARBON_FREE_SALT"),
    ("[Na+].[Na+].[Cl-].[Cl-].[Cl-].[Cl-].[Zn+2]", "CARBON_FREE_SALT"),
    ("[H-].[H-].[Zn+2]", "CARBON_FREE_SALT"),
    ("O.O.O.O.O.O.O.[O-]S(=O)(=O)[O-].[Zn+2]", "CARBON_FREE_SALT"),
    ("CC[Zn]CC", "ORGANOZINC"),
    ("[CH2-]C.[CH2-]C.[Zn+2]", "CARBANION_SALT"),
    ("[Li+].[Li+].[Zn+2].[CH3-].[CH3-].[CH3-].[CH3-]", "CARBANION_SALT"),
    ("Cl[Zn]Cl", "BONDED_DRAWING"),
    ("O=[Zn]", "BONDED_DRAWING"),
    ("[ZnH2]", "BONDED_DRAWING"),
    ("Cl[Zn](Cl)(<-N)<-N", "DATIVE_DRAWING"),
    ("c1ccc(nc1)-c1ccccn1.[Cl-].[Cl-].[Zn+2]", "ADDUCT"),
    ("[Zn+2].N.N.N.N", "ADDUCT"),
    ("[Zn].CC(=O)O.CC(=O)O", "ADDUCT"),
    ("[Zn+2]", "ATOM_OR_ION"),
    ("[65Zn]", "ATOM_OR_ION"),
    ("c1ccc(cc1)-c1c2ccc(n2)c(-c2ccccc2)c2ccc([n-]2)c(-c2ccccc2)c2ccc(n2)c(-c2ccccc2)c2ccc1[n-]2.[Zn+2]",
     "MACROCYCLE_COMPLEX"),
    ("CCO", "NO_ZINC"),
]


@pytest.mark.parametrize("smiles,cls", CLASSIFIER_CASES)
def test_the_classifier_places_a_drawing_by_its_structure(smiles, cls):
    assert zinc_label_class(smiles) == cls, smiles


def test_the_rows_hold_every_class_the_policy_names():
    """The audit reads the 87-pair panel, the battery and the constructed rows above: between them every class
    of the table occurs."""
    found = {zinc_label_class(s) for s in ZINC_SMILES}
    assert set(PIN_CLASSES) | set(NON_PIN_CLASSES) <= found, (set(PIN_CLASSES) | set(NON_PIN_CLASSES)) - found


def test_the_assumed_classes_are_listed_with_their_row_counts(capsys):
    """Printed (run with ``-s`` or ``-rP``): the classes of the table whose status is ASSUMED, with the number
    of drawings of the rows that fall in each (a count of structures, no naming)."""
    counts = {}
    for smiles in ZINC_SMILES:
        counts[zinc_label_class(smiles)] = counts.get(zinc_label_class(smiles), 0) + 1
    with capsys.disabled():
        for cls, basis in CLASS_BASIS.items():
            if basis["status"] == "ASSUMED":
                print(f"\nASSUMED class {cls}: {counts.get(cls, 0)} drawings of the rows "
                      f"({basis['rule']}, {basis['heading']}, :{basis['line']})")
    assert sum(counts.values()) == len(ZINC_SMILES)


# --------------------------------------------------------------------------- the labels of the rows

@pytest.mark.parametrize("smiles", ZINC_SMILES)
def test_default_tier_emits_only_pin_labelled_zinc_names(smiles):
    """The default tier names a zinc drawing only as a verified PIN of a class that carries a PIN label
    (ASSUMED or EXCEPTION in ``CLASS_BASIS``); every other drawing is declined, with the label NO_VERIFIED_PIN
    (a name was built and is no PIN) or UNNAMEABLE (none was built), never UNSUPPORTED_ELEMENT."""
    row = _row(smiles, "pin")
    cls = zinc_label_class(smiles)
    if row["tier"] != "abstain":
        assert row["tier"] == "pin_verified" and row["is_pin"] is True, (smiles, row["name"], row["tier"])
        assert cls in PIN_CLASSES, (smiles, cls, row["name"])
        assert CLASS_BASIS[cls]["status"] in ASSUMED_OR_EXCEPTION, (smiles, cls)
    else:
        assert row["is_pin"] is False
        if _holds_a_metal_the_table_does_not_admit(smiles):
            assert row["limit_code"] in ("UNSUPPORTED_ELEMENT", "NO_VERIFIED_PIN"), (smiles, row["limit_code"])
        else:
            assert row["limit_code"] in ("NO_VERIFIED_PIN", "UNNAMEABLE"), (smiles, row["limit_code"], row["name"])


NON_PIN_ROWS = [s for s in ZINC_SMILES if zinc_label_class(s) not in PIN_CLASSES]


@pytest.mark.parametrize("smiles", NON_PIN_ROWS)
def test_non_pin_classes_are_never_pin_verified(smiles):
    """Organozinc, carbanion salts, bonded and dative drawings, adducts, macrocycle complexes and atoms or ions
    are ``systematic_verified`` or lower at every tier, with ``is_pin`` False; a drawing the classifier cannot
    place is not labelled a PIN either (the audit places it first)."""
    cls = zinc_label_class(smiles)
    assert cls in NON_PIN_CLASSES or cls == UNPLACED, (smiles, cls)
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["tier"] not in ("pin_verified", "pin_unverified"), (smiles, tier, row["name"], row["tier"])
        assert row["is_pin"] is False, (smiles, tier, row["name"])


UNREADABLE = ("κ", "kappa", "chlorido", "bromido", "iodido", "fluorido", "acetato", "aqua", "ammine")


@pytest.mark.parametrize("smiles", ZINC_SMILES)
def test_no_unreadable_vocabulary(smiles):
    """OPSIN 2.9.0 reads hydrido, oxido, sulfido, azido, carbonyl and the plain substituent prefixes, and not
    chlorido, acetato, kappa, aqua or ammine (hazard H11): no emitted zinc name uses a word it cannot read."""
    for tier in ("pin", "best-effort"):
        name = _row(smiles, tier)["name"] or ""
        if is_failure_name(name):
            continue
        assert not any(word in name for word in UNREADABLE), (smiles, tier, name)
        assert not re.search(r"[a-z]ato\b|ato\)", name), (smiles, tier, name)


# --------------------------------------------------------------------------- the failure label

def _code(smiles):
    return classify_failure_limit(Chem.MolFromSmiles(smiles)).code


@pytest.mark.parametrize("smiles, message", [
    ("[Zn-]", "zinc compound (not supported)"),                  # a negative ion of the metal: no producer
    ("[Zn]=[Zn]", "zinc compound (not supported)"),
    ("CC[Zn]CC", "zinc compound (not supported)"),               # the classifier is asked about an input that failed
    ("[Zn+2]", "zinc compound (not supported)"),
    ("[Zn-].[Na+]", "sodium compound (not supported)"),          # beside a metal of Group 1 or 2: admitted to ION_SALTS
    ("[Zn-].[Be+2]", "beryllium compound (not supported)"),
    ("[Zn-].[Mg+2]", "magnesium compound (not supported)"),
    ("[Li+].[Zn+2].[H-].[H-].[H-]", "lithium compound (not supported)"),
    ("[Zn+2].[Cu+2]", "copper compound (not supported)"),        # beside a metal that the table admits since 2026-10-30
    ("CC[Zn]CC.CC[Pb](CC)(CC)CC", "zinc compound (not supported)"),   # lead: admitted to PARENT_HYDRIDE_IONS and more
])
def test_a_failure_for_a_metal_the_engine_names_is_unnameable_not_unsupported(smiles, message):
    """Every metal of the input is admitted to the scope table and one of them is complete (zinc): the engine
    attempted the compound in every drawing and found no name; the element is supported. The text is the
    descriptive fallback of the metal of lowest atomic number, as it is for every other label: the default tier
    reads the text 'unknown organic compound' as the decline of a name that was built (NO_VERIFIED_PIN)."""
    error = classify_failure_limit(Chem.MolFromSmiles(smiles))
    assert (error.code, error.message) == ("UNNAMEABLE", message), (smiles, error.code, error.message)
    assert is_failure_name(error.message)


@pytest.mark.parametrize("smiles, message", [
    ("Cl[Tc]Cl", "inorganic compound (not supported)"),          # a metal that the table does not admit
    ("CC[Tc]CC", "inorganic compound (not supported)"),
    ("Cl[Cu]Cl", "copper compound (not supported)"),             # a metal that the table admits and does not complete
    ("Cl[Fe]Cl", "iron compound (not supported)"),
    ("[Pt+2].[Cl-].[Cl-]", "platinum compound (not supported)"),
    ("CC[Hg]CC", "mercury compound (not supported)"),
    ("CC[Cd]CC", "cadmium compound (not supported)"),
    ("CC[Sn](CC)CC", "tin compound (not supported)"),            # tin and lead are admitted and not complete
    ("[Na+].[Cl-]", "sodium compound (not supported)"),          # Groups 1 and 2 alone: admitted, not complete
    ("[Mg+2]", "magnesium compound (not supported)"),
    ("[Zn+2].[Tc+2]", "zinc compound (not supported)"),          # zinc beside a metal that is not admitted
    ("CC[Zn]CC.CC[Tc]CC", "zinc compound (not supported)"),
    ("[Zn+2].[Ge]", "zinc compound (not supported)"),            # a metalloid is not a metal of the table
    ("[He]", "inorganic compound (not supported)"),
])
def test_every_other_failure_keeps_its_unsupported_element_label(smiles, message):
    """The label of the elements that the table does not complete is unchanged; so is its text."""
    error = classify_failure_limit(Chem.MolFromSmiles(smiles))
    assert (error.code, error.message) == ("UNSUPPORTED_ELEMENT", message), smiles


def test_the_sentinel_catalog_still_recognises_the_metal_texts():
    """The texts of the other labels stay failure names, and the old text of a zinc input still is one
    (a stored row or a nested refusal may carry it)."""
    for text in ("zinc compound (not supported)", "copper compound (not supported)",
                 "sodium compound (not supported)", "unknown organic compound", "inorganic compound (not supported)"):
        assert is_failure_name(text), text
    assert not is_failure_name("zinc dichloride")


def test_the_refusal_of_a_branch_follows_the_same_rule():
    """``unsupported_element_branch`` (a substituent branch that holds a metal and cannot be named) labels a
    branch of a complete metal UNNAMEABLE, as the whole input is."""
    error = unsupported_element_branch("Zn")
    assert (error.code, error.message) == ("UNNAMEABLE", "zinc compound (not supported)")
    for symbol, message in (("Cu", "copper compound (not supported)"), ("Na", "sodium compound (not supported)"),
                            ("Hg", "mercury compound (not supported)")):
        error = unsupported_element_branch(symbol)
        assert (error.code, error.message) == ("UNSUPPORTED_ELEMENT", message)


def test_the_scope_function_reads_the_table(monkeypatch):
    """``has_complete_scope`` follows the table: a metal is completed by an edit of its entry, and a Group 1
    or Group 2 element alone is not complete."""
    assert has_complete_scope(["Zn"]) and has_complete_scope(["Zn", "Na", "Mg"])
    assert not has_complete_scope([]) and not has_complete_scope(["Na"]) and not has_complete_scope(["Na", "Ca"])
    assert has_complete_scope(["Zn", "Cu"])                        # copper is admitted since 2026-10-30, zinc completes
    assert not has_complete_scope(["Zn", "Tc"]) and not has_complete_scope(["Zn", "Xx"])
    assert not has_complete_scope(["Cu"]) and not has_complete_scope(["Sn"])        # admitted, not complete
    assert is_complete_metal("Zn") and not is_complete_metal("Na") and not is_complete_metal("Cu")
    assert _code("Cl[Bi]Cl") == "UNSUPPORTED_ELEMENT"              # bismuth has a name (errors._METAL_NAMES), no entry
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Bi", metal_scope.MetalScope(
        frozenset({metal_scope.ION_SALTS}), frozenset({3}), complete=True))
    assert has_complete_scope(["Bi"]) and has_complete_scope(["Zn", "Bi"])
    assert _code("Cl[Bi]Cl") == "UNNAMEABLE"                      # a later phase completes a metal by its entry


@pytest.mark.parametrize("smiles, message", [("[Zn-]", "zinc compound (not supported)"),
                                             ("[Zn]=[Zn]", "zinc compound (not supported)"),
                                             ("[Zn-].[Na+]", "sodium compound (not supported)")])
def test_the_named_rows_of_a_zinc_failure_carry_the_label_at_every_tier(smiles, message):
    """name_tiered, the call that asks for the label (``on_failure="label"``), ``raise_on_limit`` and
    ``classify_limit`` agree: UNNAMEABLE at the default and at the best-effort tier; the call returns the
    descriptive fallback of the metal."""
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["tier"] == "abstain" and row["limit_code"] == "UNNAMEABLE", (smiles, tier, row)
        assert row["name"] in (None, message), (smiles, tier, row["name"])
    plain = _namer("pin").name(smiles, on_failure="label")
    assert plain == message, plain
    with pytest.raises(OrthonymLimitError) as raised:
        _namer("pin").name(smiles, raise_on_limit=True)
    assert raised.value.code == "UNNAMEABLE"
    assert classify_limit(smiles).code == "UNNAMEABLE"


@pytest.mark.parametrize("smiles, code, label", [
    ("[Zn-]", "UNNAMEABLE", "zinc compound (not supported)"),
    ("[Zn]=[Zn]", "UNNAMEABLE", "zinc compound (not supported)"),
    ("[Zn-].[Na+]", "UNNAMEABLE", "sodium compound (not supported)"),
    ("Cl[Zn]Cl", "NO_VERIFIED_PIN", "unknown organic compound"),     # a name was built and is no PIN
    ("Cl[Cu]Cl", "UNSUPPORTED_ELEMENT", "copper compound (not supported)"),
])
def test_the_unnamed_zinc_call_logs_the_zinc_code(caplog, smiles, code, label):
    """The caller's own call with no name (GitHub issue 45) returns '' and logs one warning on the logger
    ``orthonym``; the code of the warning is the zinc-specific one (UNNAMEABLE for a zinc input that no
    producer names, NO_VERIFIED_PIN for a name the default tier declines), never UNSUPPORTED_ELEMENT for zinc."""
    with caplog.at_level(logging.WARNING, logger="orthonym"):
        assert _namer("pin").name(smiles) == ""
    records = [r for r in caplog.records if getattr(r, "orthonym_code", None)]
    assert [(r.orthonym_smiles, r.orthonym_code, r.orthonym_label) for r in records] == [(smiles, code, label)]


def test_an_organozinc_halide_the_default_tier_does_not_build_is_unnameable_there():
    """The assembler names a functional organozinc halide at the best-effort tier only: the default tier
    builds no name for it, so the label is UNNAMEABLE (not UNSUPPORTED_ELEMENT, not NO_VERIFIED_PIN: no name
    exists at that tier to decline)."""
    smiles = "CCOC(=O)C[Zn]Br"
    row = _row(smiles, "pin")
    assert row["tier"] == "abstain" and row["limit_code"] == "UNNAMEABLE", row
    assert _row(smiles, "best-effort")["tier"] == "systematic_verified"


@pytest.mark.parametrize("smiles, message", [
    ("Cl[Tc]Cl", "inorganic compound (not supported)"),
    ("CC1=CC(=O)[N-]S(=O)(=O)O1.[K+]", "potassium compound (not supported)"),
    ("[NH3][Bi]([NH3])(Cl)Cl", "bismuth compound (not supported)"),
])
def test_other_metals_keep_their_failure_rows(smiles, message):
    """Technetium (a metal that no mechanism admits), a Group 1 salt and a bismuth complex that no producer names at
    the default tier: the label and the text of the start commit (iron and copper are admitted since 2026-10-30, their
    rows moved: ``declared_moves_bonded_b2.tsv``)."""
    row = _row(smiles, "pin")
    assert row["tier"] == "abstain" and row["limit_code"] == "UNSUPPORTED_ELEMENT", row
    assert _namer("pin").name(smiles, on_failure="label") == message
