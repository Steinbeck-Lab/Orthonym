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
# The same skeleton with the thioester C=O reduced to CH2 (a sulfide): no ester, so
# the pantetheine chain is still reached as the deep -O-<chain> (butoxy) substituent.
HYDROXYALKYL_SULFIDE_COA = (
    "CCCCCCCCCCCCCC[C@@H](O)CSCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)"
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
    # The thioester is named by its chalcogen-ester handler since breadth job 3
    # (review finding 8): each word carries its own stereodescriptors, "at the front of
    # the corresponding prefix" "NAMING OF STEREOISOMERS", the Blue Book),
    # so the pantetheine centre is C-2 of its acyl prefix, '[(2R)-4-{...}-2-hydroxy-3,3-
    # dimethyl-1-oxobutyl]', and the acid's own is '(2R)-2-hydroxyhexadecanethioate'.
    # It was '(3R)-3-hydroxy-...-4-oxobutoxy' under the purine amine. Both read back to
    # the input's full InChIKey (independent OPSIN 2.9.0 call); best-effort label
    # pin_unverified.
    # Leads program L3 (N8f): the pantoyl acyl chain has three one-carbon arms at its
    # quaternary C-3, so the chain through a methyl and the chain through the CH2-O are
    # equally long; 'THE PRINCIPAL SUBSTITUENT CHAIN' (the Blue Book)
    # criterion (k), (:22740) "The principal substituent chain has the greatest
    # number of substituents of any kind", takes the chain through the CH2-O: five
    # substituents (1-oxo, 2-hydroxy, 3,3-dimethyl, 4-oxy) against four (1-oxo, 2-hydroxy,
    # 3-methyl, 3-{...methyl}), as in '6,7-dichloro-5-(2-chloropropyl)octan-2-yl (preferred
    # prefix) [not 7-chloro-5-(1,2-dichloropropyl)octan-2-yl]' (:22746). The old string kept
    # the first of the tied arms in atom order. Read back to the input by a fresh OPSIN call
    # in test_leads_l3_n8f_chain_selector.py.
    name = _be_name(HYDROXYACYL_COA)
    assert name == (
        "S-(2-{[3-({(2R)-4-[(3-{[(2R,3S,4R,5R)-5-(6-amino-9H-purin-9-yl)-4-hydroxy-"
        "3-(phosphonooxy)oxolan-2-yl]methoxy}-1,3-dihydroxy-1,3-dioxo-1λ5,3λ5-"
        "diphosphoxan-1-yl)oxy]-2-hydroxy-3,3-dimethyl-1-oxobutyl}amino)-1-"
        "oxopropyl]amino}ethyl) (2R)-2-hydroxyhexadecanethioate"), name


def test_pantetheine_3_hydroxy_descriptor_is_emitted_on_the_butoxy_chain():
    # the on-chain stereocentre of the deep -O-<chain> (butoxy) substituent carries its
    # locant + CIP descriptor ('(3R)-3-hydroxy-...-4-oxobutoxy')
    name = _be_name(HYDROXYALKYL_SULFIDE_COA)
    assert name == (
        "9-[(2R,3R,4S,5R)-5-{[(1,3-dihydroxy-3-[(3R)-3-hydroxy-4-({3-[(2-{[(2R)-2-"
        "hydroxyhexadecyl]sulfanyl}ethyl)amino]-3-oxopropyl}amino)-2,2-dimethyl-4-"
        "oxobutoxy]-1,3-dioxo-1λ5,3λ5-diphosphoxan-1-yl)oxy]methyl}-3-hydroxy-"
        "4-(phosphonooxy)oxolan-2-yl]-9H-purin-6-amine"), name
    assert opsin_roundtrip_check(HYDROXYALKYL_SULFIDE_COA, name)["passed"], name


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
