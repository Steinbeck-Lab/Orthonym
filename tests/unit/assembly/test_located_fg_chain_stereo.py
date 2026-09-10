""": chain stereodescriptors on a deep located-FG (`_located_fg_assemble`)
substituent must be emitted from the chain's own free-valence numbering.

Regression for the acyl-CoA pantetheine 3-hydroxy stereocentre: when the
pantetheine chain is reached as a deep `-O-<chain>` (butoxy) substituent (its
parent is a more-senior ring), its on-chain stereocentre used to be silently
dropped, so a full-stereo acyl-CoA abstained on a stereo mismatch (0-wrong held
- it never shipped wrong stereo, it just could not name it). `_located_fg_assemble`
now cites its own `chain_pos` stereodescriptors via `_located_stereo_block`.

These assert best-effort-tier behaviour, so the namer is constructed explicitly
with the general-fallback flags (the PIN/default path is unaffected - a separate
byte-identity check covers that).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.validation import opsin_roundtrip_check

pytestmark = pytest.mark.opsin_gate

_BE = dict(general_fallback=True, general_fallback_unverified=True,
           allow_aromatic_general=True)

# (R)-3-hydroxymyristoyl-CoA, 66 heavy atoms, 6 defined stereocentres.
HYDROXYACYL_COA = (
    "CCCCCCCCCCCCCC[C@@H](O)C(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)"
    "OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O"
)
# acetyl-CoA (no stereo) - must keep round-tripping (no regression).
ACETYL_COA = (
    "CC(=O)SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OCC1OC"
    "(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O"
)


def _be_name(smiles):
    return Orthonym(**_BE).name(smiles)


def test_full_stereo_acyl_coa_round_trips_full_inchikey():
    name = _be_name(HYDROXYACYL_COA)
    assert name and name != "unknown organic compound", name
    rt = opsin_roundtrip_check(HYDROXYACYL_COA, name)
    assert rt["passed"], rt  # FULL InChIKey (stereo layer included)


def test_pantetheine_3_hydroxy_descriptor_is_emitted():
    # the previously-dropped on-chain stereocentre now carries its locant+CIP
    name = _be_name(HYDROXYACYL_COA)
    assert "(3R)" in name, name


def test_acetyl_coa_no_stereo_still_round_trips():
    name = _be_name(ACETYL_COA)
    assert name and name != "unknown organic compound", name
    assert opsin_roundtrip_check(ACETYL_COA, name)["passed"], name


def test_pin_default_path_stereo_unchanged():
    # PIN/default path must be byte-identical for ordinary stereo molecules -
    # the located-FG stereo block only runs on the best-effort FG path.
    pin = Orthonym()
    assert pin.name("CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O") == \
        "(1R,2S,5R)-5-methyl-2-(propan-2-yl)cyclohexan-1-ol"
    assert pin.name("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O") == \
        "(2R)-2-[4-(2-methylpropyl)phenyl]propanoic acid"
