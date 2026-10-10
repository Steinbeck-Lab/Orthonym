"""The spelling checks on every name the Blue Book prints as a PIN or a preferred prefix.

``data/bb_book_pin_names.json``: [line, name, SMILES] for each name printed before '(PIN' or
'(preferred prefix' on a line of the Blue Book (3,404 distinct pairs), the name being the longest
run of the words before the mark that OPSIN 2.9.0 reads as one molecule (radicals allowed), and the
SMILES OPSIN's reading of it. This reaches the rows the conformance report has no structure for:
radicals and ions, isotopically modified compounds, preferred prefixes. The structure is
only the input of the checks here; what is tested is the spelling the book prints.

A check flags exactly the rows of ``EXPLAINED``, each with its reason; a new flag fails the test
until it is explained (or the check corrected)."""
import json
from pathlib import Path

from rdkit import Chem, RDLogger

from orthonym.data.natural_products import NAME_EXACT_NP_PARENTS
from orthonym.validation.pin_spelling import check_pin_spelling
from tests.unit.validation._book_explained import LISTING_ROWS, OWN_RULE_ROWS

RDLogger.DisableLog("rdApp.*")
DATA = Path(__file__).parent / "data/bb_book_pin_names.json"

_LISTING = "a table that lists the ring component; P-14.7.1 (:3721) requires the indicated hydrogen in a PIN"

#: (line, name, rule) -> why the check and the book disagree: the rows both book corpora share
#: (``_book_explained``), and the rows of this corpus only
EXPLAINED = {
    **LISTING_ROWS,
    **OWN_RULE_ROWS,
    # Table 2.9 (:11700-:11714) and listings print the ring components without
    # indicated hydrogen; the Table 2.9 listing also prints 'indole (PIN)'
    (11708, "indole", "P-14.7.1"): _LISTING,
    (11710, "isoindole", "P-14.7.1"): _LISTING,
    (11714, "phosphinolizine", "P-14.7.1"): _LISTING,
    (11714, "quinolizine", "P-14.7.1"): _LISTING,
    (11509, "xanthene", "P-14.7.1"): _LISTING,
    # the book's row against (:3448): 'oxo' sorts before 'phenyl'
    (38894, "2-(2-phenyl-2-oxo-2λ5-diazenyl)naphthalen-1-yl", "P-14.5"): "book row against :3448",
    # two names of one radical prefix extracted as one row: the name is the non-preferred and the
    # preferred prefix side by side, which OPSIN reads as two radicals joined by a bond (a
    # ring assembly of two copies); the book line prints each name once,:15542)
    (17313, "3,4-dihydro-1-aza[12]annulen-6-yl 1-azacyclododeca-1,5,7,9,11-pentaen-6-yl", "P-28.1"):
        "two names extracted into one row",
    (17315, "12,13-dihydro-1H-1-aza[13]annulen-4-yl 1-azacyclotrideca-2,4,6,8,10-pentaen-4-yl", "P-28.1"):
        "two names extracted into one row",
    (17319, "dihydro-2H-pyran-3(4H)-ylidene oxan-3-ylidene", "P-28.1"): "two names extracted into one row",
    (24882, "1,4-dihydronaphthalene-1,4-diylidene naphthalene-1,4-diylidene", "P-28.1"):
        "two names extracted into one row",
}


def test_every_disagreement_with_a_printed_pin_is_explained():
    rows = json.loads(DATA.read_text())
    assert len(rows) == 3404, len(rows)
    flagged, read = {}, 0
    for line, name, smiles in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or name in NAME_EXACT_NP_PARENTS:
            continue
        read += 1
        for f in check_pin_spelling(mol, name, strict=True):
            flagged[(line, name, f.rule)] = f.detail
    assert read > 3300, read
    unexplained = {k: v for k, v in flagged.items() if k not in EXPLAINED}
    assert not unexplained, unexplained
    missing = set(EXPLAINED) - set(flagged)
    assert not missing, missing


def test_the_corpus_reaches_radicals_ions_and_preferred_prefixes():
    names = {name for _line, name, _smiles in json.loads(DATA.read_text())}
    for name in ("methoxyl", "bis(chloromethyl)aminoxyl", "phenyldisulfanylium", "methanide",
                 "4,4-dimethylpiperazin-4-ium-1-ylium", "1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl"):
        assert name in names, name
