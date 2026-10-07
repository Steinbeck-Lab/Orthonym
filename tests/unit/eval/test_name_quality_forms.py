"""``eval/name_quality/forms.py``: the reject-on-sight form detector (roadmap N5).

Three properties make the detector usable as an end check:

1. It never fires on a Blue Book PIN: 0 hits on every PIN / preferred-prefix string of
   ``benchmarks/bb_conformance`` (3,806 distinct strings at the base). A pattern that
   matches a book PIN is wrong.
2. Every tag fires on a known positive (an engine spelling of the class), so a zero count
   in a run means "no such name", not "the pattern is broken".
3. The boundary rows of `internal notes` (forms that look rejected but are PINs or
   book general names) never fire.

No JVM, no engine: the detector reads names only.
"""
import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from name_quality import forms as F  # noqa: E402

BB = PROJECT_ROOT / "benchmarks" / "bb_conformance"


def _bb_strings():
    names = {r["bb_name"] for r in csv.DictReader(open(BB / "all_pins_distinct.csv"))}
    with open(BB / "bb_measure_rows.jsonl") as fh:
        names |= {json.loads(line)["expected"] for line in fh if line.strip()}
    return sorted(n for n in names if n)


def test_no_form_on_any_blue_book_pin_string():
    strings = _bb_strings()
    assert len(strings) > 3000
    hits = {n: F.detect(n) for n in strings}
    hits = {n: h for n, h in hits.items() if h}
    assert hits == {}


POSITIVES = [
    ("BENZ", "4-(cyclohexa-1,3,5-trien-1-yl)phenol"),
    ("ACH", "2-(cyclopropan-1-yl)-2-azaethyl"),
    ("ACH", "1-[3-(...)]-2-oxaeth-1-en-1-yl"),
    ("ACH", "1-oxamethyl"),
    ("ACH_TERM", "N-(11-amino-3,7-disulfanylidene-2,8,12-trioxa-4,5,6-trithia-10-azadodec-11-en-1-yl)urea"),
    ("MONO", "1-azacyclobutan-1-yl"),
    ("MONO", "3-(4-methyl-1,3-diazacyclopenta-2,4-dien-1-yl)"),
    ("OXOM_M", "2-{[2-methyl-5-({[3-(trifluoromethyl)phenyl]amino}-oxomethyl)phenyl]amino}pyrimidine"),
    ("OXOM_M", "1,2-bis[(2-butoxyethoxy)-oxomethyl]benzene"),
    ("OXOM_1", "4-(1-oxopropyl)phenol"),
    ("VBF", "5-oxo-3,4-diazabicyclo[4.4.0]deca-1(10),2,6,8-tetraene"),
    ("VBF", "bicyclo[4.4.0]decane"),
    ("VBB", "tricyclo[5.2.1.0^2,6]decane"),
    ("RLOC", "5-(1,1,1-trifluoromethyl)phenyl"),
    ("RLOC", "N-[1-(carbamoylamino)methyl]urea"),
    ("RLOC", "1,1,2,2,2-pentafluoroethyl"),
    ("RLOC", "1,1,1,2,3,3,3-heptafluoropropan-2-yl"),
    ("ANYL_ME", "4-(1,1,1-trifluoromethan-1-yl)phenyl"),
    ("ANYL_ME", "1-(phenyl)methan-1-yl"),
    ("ANYL", "2-(cyclopropan-1-yl)ethyl"),
    # (2) (the Blue Book): method (2) is for free valences at positions
    # other than '1'; reported (SOFT), the writers spell '2-chloroethyl' (method (1))
    ("ANYL_SUB", "2-chloroethan-1-yl"),
    ("ANYL_SUB", "N-[2-(pyridin-2-yl)ethan-1-yl]acetamide"),
    ("ANYL_SUB", "2,2-dimethylpropan-1-yl"),
]


@pytest.mark.parametrize("tag,name", POSITIVES)
def test_each_tag_fires_on_a_positive(tag, name):
    assert tag in F.detect(name), F.detect(name)


#: forms that look rejected but are PINs or book general names (RULES.md boundaries)
NEGATIVES = [
    "2,2,2-trifluoroethyl",                                    # partial substitution,:3009
    "1-chloro-2-(pentafluoroethyl)benzene",                    #:3023
    "trifluoromethyl",
    "1,1,1,3,3,3-hexafluoropropan-2-yl",                      # C2-H left: partial
    "bicyclo[4.2.0]octa-1,3,5,7-tetraene",                    # PIN,:23725
    "bicyclo[4.1.0]hepta-1,3,5-triene",                       # PIN,:16653
    "tricyclo[4.4.0.0^2,7]dec-3-ene",                          # rings share 3 atoms
    "bicyclo[2.2.1]heptane",                                   # bridged, no fusable pair
    "3,6,9,12-tetraoxatetradecanedioic acid",                  # 4 heterounits, C ends
    "2-oxa-4-thia-1,5-disilapentane",                          # Si ends,:23436
    "2-amino-2-oxoethyl",                                      # PIN,:30373,:32924
    "oxomethylidene",                                          # preferred prefix
    "(oxomethyl)benzene",                                      # bare oxomethyl,:30444
    "cyclohexyl(oxo)methyl",                                   #:30628
    "oxo(phenyl)methyl",                                       #:30446
    "1,2,5,6-tetrasilacyclooct-3-en-7-yne",                    # triple bond,:21091
    "1-oxacyclododecane",                                      # 12 members: 'a' name
    "1-(13C)methyl(2-13C)benzene",                             # isotope descriptors
    "2-(hydroxymethyl)benzene-1,4-diol",                       # PIN,:6802
    "dichloromethyl",
    "decahydronaphthalene",
    "propanoyl",
    "azetidin-1-yl",
    "propan-1-yl",                                             # printed beside propyl,:15939
    "7-propyl-2,4,6,8-tetrasilanonan-1-yl",                    # 'a' chain, Blue Book PIN string
    "(bicyclo[2.2.2]octan-1-yl)",                              # von Baeyer prefix
    "ethan-1-yl-2-ylidene",                                    # two free valences
    "bicyclo[4.2.1]nonan-1-yl acetate",                        # von Baeyer prefix,:16049
    "cyclohexa-1,3-dien-1-yl",                                 #:16904
    "N,1-bis(4-chlorophenyl)methanimine",                      # PIN,:26524
    "1,1,1-trimethylsilanamine",                               # PIN,:26221
]


@pytest.mark.parametrize("name", NEGATIVES)
def test_boundary_forms_do_not_fire(name):
    assert F.detect(name) == {}


def test_count_rows_counts_named_rows_once_per_tag():
    rows = [
        {"smiles": "C", "name": "1-azacyclobutan-1-yl-1-azacyclobutane", "tier": "best_effort"},
        {"smiles": "C", "name": "benzene", "tier": "pin_verified"},
        {"smiles": "C", "name": None, "tier": "abstain"},
    ]
    st = F.count_rows(rows)
    assert st["named"] == 2
    assert st["tags"]["MONO"] == 1
    assert st["any_target"] == 1
