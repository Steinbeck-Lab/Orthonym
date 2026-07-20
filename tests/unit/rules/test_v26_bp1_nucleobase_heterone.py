"""v26 BP-1: mancude ring-heterone seniority + non-PIN nucleobase gating.

Root causes (see *.md):
 - FIX 2: a ring-carbon exocyclic =O on a mancude ring (a heterone) was invisible
   to functional-group perception (the ketone SMARTS needs two C neighbours), so an
   amine wrongly won the principal-group slot. BB P-64.7.1 (line 29585): ketones/
   heterones are senior to amines. Registering the heterone makes the =O the suffix
   (-one) and the amine an amino- prefix.
 - FIX 1: uracil/thymine/cytosine/fluorouracil are absent from the Blue Book
   (not PINs). Under style=pin they must re-derive the systematic pyrimidinedione/
   pyrimidinone PIN; the retained name stays available for the --trivial path.
"""
import pytest

from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer_pin():
    return Orthonym(style="pin")


@pytest.mark.parametrize("smiles,expected", [
    # non-PIN nucleobases -> systematic PIN under style=pin (FIX 1)
    ("O=C1C=CNC(=O)N1", "pyrimidine-2,4(1H,3H)-dione"),          # uracil
    ("Cc1c[nH]c(=O)[nH]c1=O", "5-methylpyrimidine-2,4(1H,3H)-dione"),  # thymine
    ("O=C1NC(=O)NC=C1F", "5-fluoropyrimidine-2,4(1H,3H)-dione"),  # fluorouracil
    ("Nc1cc[nH]c(=O)n1", "4-aminopyrimidin-2(1H)-one"),          # cytosine (FIX 1+2)
    # general amino-oxo-heteroarene class: heterone (-one) senior to amine (FIX 2)
    ("Nc1ccc(=O)[nH]c1", "5-aminopyridin-2(1H)-one"),
    ("Nc1cc(=O)[nH]c2ccccc12", "4-aminoquinolin-2(1H)-one"),
])
def test_heterone_and_nucleobase_pins(namer_pin, smiles, expected):
    assert namer_pin.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # FIX 2 must NOT regress saturated rings or carbocyclic ketones (scoped to
    # residual unsaturation) or the mancude oxo/lactone parents.
    ("NC1CCC(=O)CC1", "4-aminocyclohexan-1-one"),   # carbocyclic ketone > amine (already correct)
    ("O=C1CCCCN1", "piperidin-2-one"),               # saturated lactam
    ("O=C1CCCO1", "oxolan-2-one"),                   # saturated lactone
    ("O=C1CCCCC1", "cyclohexan-1-one"),              # saturated ketone
    ("O=c1ccc2ccccc2o1", "2H-1-benzopyran-2-one"),   # coumarin (heterone already handled)
    ("O=c1ccoc2ccccc12", "4H-1-benzopyran-4-one"),   # chromone
    ("O=c1ccc2ccccc2[nH]1", "quinolin-2(1H)-one"),   # quinolinone
])
def test_no_regression(namer_pin, smiles, expected):
    assert namer_pin.name(smiles) == expected


def test_nucleobase_retained_name_gated_out_of_pin_dict():
    """uracil moves from the PIN dict to the general-only dict (served by --trivial)."""
    from rdkit import Chem
    from orthonym.data import ALL_RETAINED_NAMES, GENERAL_RETAINED_NAMES
    u = Chem.CanonSmiles("O=C1C=CNC(=O)N1")
    assert u not in ALL_RETAINED_NAMES
    assert GENERAL_RETAINED_NAMES.get(u) == "uracil"


def test_nucleobase_denies_present():
    """The four monocyclic nucleobases are denied (BP-1 FIX 1)."""
    from orthonym.data import _PIN_DENY
    for n in ("uracil", "thymine", "cytosine", "fluorouracil"):
        assert n in _PIN_DENY, n
    # The purine-base bicyclics are DEFERRED (still emit their retained names).
    for n in ("adenine", "guanine", "xanthine", "hypoxanthine"):
        assert n not in _PIN_DENY, n
