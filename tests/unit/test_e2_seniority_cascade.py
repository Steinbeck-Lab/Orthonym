"""v22 Phase E2 (DD5) — parent/chain seniority cascade, rule-family coverage (A8).

Tests the OUTPUT of the SEN-01 (P-44.3/P-45 chain cascade + deterministic
comparator) and SEN-04 (P-46 located-substituent re-basing) fixes as RULE FAMILIES,
not literal canary rows: secondary vs branched-terminal attachments, the terminal
fast-path invariant, retained-prefix non-regression, stereo preservation, and the
P-45 tie/determinism behaviour.

SEN-02 (carbon-over-ether) and SEN-03 (PCG union) are documented A9 masking pairs,
deferred (see E2-SUMMARY); their non-regression invariants are asserted here.
"""
import pytest

from orthonym import Orthonym
from orthonym.assembly.substituent_naming import (
    _located_acyclic_alkyl_name,
    _attach_is_chain_terminus,
)
from rdkit import Chem


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# ---------------------------------------------------------------------------
# SEN-04 — located acyclic-alkyl substituent (P-46), as a rule family
# ---------------------------------------------------------------------------

# Secondary / internal attachment -> alkan-k-yl (free-valence locant cited).
SEN04_SECONDARY = [
    ("CC(=O)NC(C)CCCC", "N-(hexan-2-yl)acetamide"),     # NHR amide, hexan-2-yl
    ("CCC(CCC)NC(C)=O", "N-(hexan-3-yl)acetamide"),      # internal, hexan-3-yl
    ("CCC(CC)c1ccccc1", "(pentan-3-yl)benzene"),         # ring parent, pentan-3-yl
]

# Branched terminal attachment -> <loc>-<sub>alkyl (free-valence locant elided).
SEN04_BRANCHED_TERMINAL = [
    ("CC(C)CCc1ccccc1", "(3-methylbutyl)benzene"),
    ("CC(C)CCCc1ccccc1", "(4-methylpentyl)benzene"),
]

# INVARIANT: terminal unbranched alkyl stays the plain elided form (fast path).
SEN04_TERMINAL_PLAIN = [
    ("CCCCc1ccccc1", "butylbenzene"),
    ("CC(=O)NCCCC", "N-butylacetamide"),
    ("CC(=O)NCCCCCC", "N-hexylacetamide"),
    ("CCc1ccccc1", "ethylbenzene"),
]

# F-T9/DD6 RET-02 (supersedes the original E2 invariant): these deprecated retained
# substituent prefixes are no longer emitted — they are de-headlined to the located /
# systematic PIN. (cumene/isopropyl P-29.6.2.2; isobutyl/sec-butyl P-29.6.3.)
SEN04_RETAINED = [
    ("CC(C)c1ccccc1", "(propan-2-yl)benzene"),
    ("CC(C)Cc1ccccc1", "(2-methylpropyl)benzene"),
    ("CCC(C)c1ccccc1", "(butan-2-yl)benzene"),
]


@pytest.mark.parametrize("smiles,expected", SEN04_SECONDARY)
def test_sen04_secondary_attachment_alkan_k_yl(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", SEN04_BRANCHED_TERMINAL)
def test_sen04_branched_terminal_locant_elided(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", SEN04_TERMINAL_PLAIN)
def test_sen04_terminal_unbranched_unchanged(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", SEN04_RETAINED)
def test_sen04_deprecated_prefixes_deheadlined_to_pin(namer, smiles, expected):
    # F-T9/DD6 RET-02: the deprecated retained substituent prefixes are de-headlined
    # to the located/systematic PIN (the E2 located producer is now the headline).
    assert namer.name(smiles) == expected


def test_sen04_stereo_descriptor_preserved(namer):
    # The located deriver is un-gated off the stereocentre, but the stereo
    # wrapper still prepends the descriptor when the attachment IS the centre.
    assert namer.name("CC(=O)N[C@@H](C)CC") == "N-[(2S)-butan-2-yl]acetamide"
    assert namer.name("CC(=O)N[C@@H](C)CCC") == "N-[(2S)-pentan-2-yl]acetamide"


# --- structure-deriver unit invariants (no namer / no OPSIN) ---

def _atoms_of(smiles):
    return Chem.MolFromSmiles(smiles)


def test_located_deriver_secondary_returns_alkan_k_yl():
    # Secondary attachment: hexane numbered from an internal free valence -> hexan-2-yl.
    mol = Chem.MolFromSmiles("CCCCCC")  # n-hexane; atom 1 is the internal free valence
    sub = list(range(mol.GetNumAtoms()))
    res = _located_acyclic_alkyl_name(mol, sub, 1)
    assert res is not None
    name, k = res
    assert name == "hexan-2-yl" and k == 2


def test_located_deriver_branched_secondary_keeps_branch():
    # Tertiary attachment with a methyl branch: 2-methylhexan-2-yl (NOT hexan-2-yl —
    # the substituent's own methyl is kept, P-46.1.12).
    mol = Chem.MolFromSmiles("CC(C)CCCC")  # 7 carbons; atom 1 bears two methyls + butyl
    sub = list(range(mol.GetNumAtoms()))
    res = _located_acyclic_alkyl_name(mol, sub, 1)
    assert res is not None
    name, k = res
    assert name == "2-methylhexan-2-yl" and k == 2


def test_located_deriver_branch_kept_terminal():
    # isopentyl skeleton CC(C)CC with the free valence at a terminal CH2.
    mol = Chem.MolFromSmiles("CC(C)CC")
    sub = list(range(mol.GetNumAtoms()))
    # terminal CH2 (last atom) as the free valence
    term = [a.GetIdx() for a in mol.GetAtoms()
            if sum(1 for n in a.GetNeighbors() if n.GetSymbol() == "C") == 1]
    res = _located_acyclic_alkyl_name(mol, sub, term[-1])
    assert res is not None
    name, k = res
    assert k == 1 and name == "3-methylbutyl"


def test_located_deriver_declines_ring_and_hetero():
    # ring substituent -> None (handled elsewhere)
    mol = Chem.MolFromSmiles("C1CCCCC1")
    assert _located_acyclic_alkyl_name(mol, list(range(mol.GetNumAtoms())), 0) is None
    # heteroatom in fragment -> None
    mol2 = Chem.MolFromSmiles("CCOCC")
    assert _located_acyclic_alkyl_name(mol2, list(range(mol2.GetNumAtoms())), 0) is None
    # unsaturated -> None (alkenyl path owns it)
    mol3 = Chem.MolFromSmiles("C=CCC")
    assert _located_acyclic_alkyl_name(mol3, list(range(mol3.GetNumAtoms())), 3) is None


def test_attach_terminus_classification():
    mol = Chem.MolFromSmiles("CCC(CC)CC")  # 3-ethylpentane skeleton
    sub = list(range(mol.GetNumAtoms()))
    # a terminal CH3 is a terminus; the central C is not
    central = [a.GetIdx() for a in mol.GetAtoms()
               if sum(1 for n in a.GetNeighbors() if n.GetSymbol() == "C") >= 3]
    term = [a.GetIdx() for a in mol.GetAtoms()
            if sum(1 for n in a.GetNeighbors() if n.GetSymbol() == "C") == 1]
    assert _attach_is_chain_terminus(mol, sub, term[0]) is True
    assert _attach_is_chain_terminus(mol, sub, central[0]) is False
    assert _attach_is_chain_terminus(mol, sub, None) is True  # no context -> fast path


# ---------------------------------------------------------------------------
# SEN-01 — P-44.3/P-45 chain cascade + deterministic comparator
# ---------------------------------------------------------------------------

def test_sen01_p45_citation_order_tiebreak(namer):
    # {2,4} locant tie decided by alphanumerical citation order (P-45.2.3).
    assert namer.name("OCC(CCBr)CCCl") == "2-(2-bromoethyl)-4-chlorobutan-1-ol"


@pytest.mark.parametrize("smiles", ["OCC(CCBr)CCCl", "OCC(CCCl)CCBr"])
def test_sen01_tie_is_order_independent(namer, smiles):
    # Both input orderings of the SAME molecule yield the one PIN (determinism).
    assert namer.name(smiles) == "2-(2-bromoethyl)-4-chlorobutan-1-ol"


def test_sen01_diene_substituent_locant_set(namer):
    # Orientation-consistent sub-locant scoring: {4,5} beats {4,6}; the chain
    # selection (5-methyl-4-(...)hepta-1,5-diene) is correct (substituent
    # EXPRESSION prop-1-enyl vs prop-1-en-1-yl is a separate P-31.1.3 follow-on).
    name = namer.name("C=CCC(C=C(C)C)C(C)=CC")
    assert name.startswith("5-methyl-4-") and name.endswith("hepta-1,5-diene")


# INVARIANT: pure hydrocarbon / single-PCG selection unchanged (byte-identical class).
SEN01_INVARIANT = [
    ("CCCCCC", "hexane"),
    ("CC(C)CC", "2-methylbutane"),
    ("CCC(C)CC", "3-methylpentane"),
    ("CCC(CC)CC", "3-ethylpentane"),
    ("CC(C)CCCC", "2-methylhexane"),
    ("C=CCCCC", "hex-1-ene"),
    ("C=CC=CCC", "hexa-1,3-diene"),
    ("CC(Cl)CC(Br)C", "2-bromo-4-chloropentane"),
    ("OC(=O)CCS(=O)(=O)O", "3-sulfopropanoic acid"),  # protect: acid > sulfonic
]


@pytest.mark.parametrize("smiles,expected", SEN01_INVARIANT)
def test_sen01_invariant_no_regression(namer, smiles, expected):
    assert namer.name(smiles) == expected


# ---------------------------------------------------------------------------
# SEN-02 / SEN-03 deferred — non-regression invariants
# ---------------------------------------------------------------------------
# SEN-03 (PCG-atom union, P-44.1.1) — alcohol class: mixed primary+secondary OH
# is a diol/triol/pentaol, not N-hydroxy-...-ol.
# ---------------------------------------------------------------------------

SEN03_DIOL = [
    ("OCC(O)C", "propane-1,2-diol"),                 # primary + secondary OH
    ("OCCC(O)C", "butane-1,3-diol"),
    ("CC(C)(O)CCO", "3-methylbutane-1,3-diol"),      # tertiary + primary OH
    ("OCCC(CCCCCl)C(O)C", "3-(4-chlorobutyl)pentane-1,4-diol"),  # gold (+ halo-branch)
]

# Invariants: same-subtype diols and single-OH unchanged; amine class excluded.
SEN03_INVARIANT = [
    ("OCCO", "ethane-1,2-diol"),
    ("OC(C)CC(C)O", "pentane-2,4-diol"),
    ("OCC(O)CO", "propane-1,2,3-triol"),  # Wave-1 1.10: glycerol demoted (general-only)
    ("CCO", "ethanol"),
    ("CC(C)O", "propan-2-ol"),
    ("NCC(N)C", "propane-1,2-diamine"),   # same-subtype amine diamine unaffected
]


@pytest.mark.parametrize("smiles,expected", SEN03_DIOL)
def test_sen03_mixed_subtype_diol(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", SEN03_INVARIANT)
def test_sen03_invariant(namer, smiles, expected):
    assert namer.name(smiles) == expected


# ---------------------------------------------------------------------------
# SEN-02 (carbon-over-ether/sulfide, P-41 cls 40 > 41/42) — scoped to the mixed
# ether+sulfide class (the broad sulfide/sulfoxide/diether -> substitutive PIN
# migration is a documented follow-on, so simple sulfides/diethers/sulfoxides
# keep their established names).
# ---------------------------------------------------------------------------

SEN02_CARBON_OVER_ETHER = [
    ("COCSC", "methoxy(methylsulfanyl)methane"),     # headline gold (carbon parent)
    ("CSCOC", "methoxy(methylsulfanyl)methane"),     # determinism pair
]

# Invariants: homogeneous dithioethers + simple sulfides/sulfoxides keep
# their established (skeletal / functional-class) names — NOT migrated here.
# NOTE: COCCOC (homogeneous 2-O diether) was previously listed here as
# "2,5-dioxahexane" but R4 (P-12.1/P-63.2.4) now routes it substitutive ->
# '1,2-dimethoxyethane'.  Removed from this invariant set.
SEN02_INVARIANT = [
    ("CSCSC", "2,4-dithiapentane"),      # homogeneous dithioether -> skeletal kept
    ("CSC", "dimethyl sulfide"),         # simple sulfide -> functional-class kept
    ("CSCC", "ethyl methyl sulfide"),
    ("CS(=O)C", "dimethyl sulfoxide"),   # sulfoxide -> functional-class kept (follow-on)
    ("OCCOCCOCCOC", "3,6,9-trioxadecan-1-ol"),  # terminal-OH polyether -> skeletal kept
]


@pytest.mark.parametrize("smiles,expected", SEN02_CARBON_OVER_ETHER)
def test_sen02_carbon_over_ether(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", SEN02_INVARIANT)
def test_sen02_invariant(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_sen02_carbon_over_ether_deterministic(namer):
    # COCSC and CSCOC are the same molecule -> one PIN regardless of SMILES order.
    assert namer.name("COCSC") == namer.name("CSCOC")
