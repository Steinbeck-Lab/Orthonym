"""Wave2 Tier 6 — fail-closed conversions + splice fixes for exotic classes.

6a Spiro ring-unsaturation splice / + tri+ polyspiro
    fail-closed /.2.3) + propene/propyne locant elision
    (d)).
6b Substituent-mode hydro emitter for heteromonocyclic mancude substituents
     + the saturated-stem (oxanyl) wrong-constitution guard.
6c Acyl pseudohalides functional-class PINs), mixed
    divalent-chalcogen bridge, phane production.

Every positive expectation below was OPSIN-round-trip verified at build time
(phane names, which OPSIN cannot parse, by molecular-formula conservation).
 (item 2, 2026-09-04): tri+ polyspiro, the S-attached chalcogen bridge and
 phane production are now BUILT (capability gains) — see
internal notes
"""
import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.spiro import name_spiro_system
from orthonym.rules.ring_substituents import (
    get_ring_substituent_name,
    _unsaturated_heteromonocyclic_substituent,
)


@pytest.fixture
def _validity_gate_on(monkeypatch):
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    yield


# ---------------------------------------------------------------------------
# 6a — spiro unsaturation splice
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("C1CCCC12C=CCCC2", "spiro[4.5]dec-6-ene"),      # BB verbatim
    ("C1=CCCC11CCCCC1", "spiro[4.5]dec-1-ene"),      # OPSIN-corrected ledger row
    ("C1CCC2(CC1)C=CCCC2", "spiro[5.5]undec-1-ene"),
    ("N1CC=CC12CCCCC2", "1-azaspiro[4.5]dec-3-ene"),  # BB verbatim
])
def test_spiro_unsaturation_splice(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("C1CCC2(CC1)CCCC2", "spiro[4.5]decane"),
    ("O1CCCC12CCCCC2", "1-oxaspiro[4.5]decane"),
    ("C1CC2(C1)CCC3(CC2)CCC3", "dispiro[3.2.3^7.2^4]dodecane"),
])
def test_saturated_spiro_protected(smiles, expected):
    """The splice pass must not disturb saturated spiro/dispiro naming."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
def test_polyspiro_unsaturated_fails_closed():
    """A dispiro with a ring double bond declines (splice is monospiro-only) —
    never a silently-saturated '-ane'."""
    mol = Chem.MolFromSmiles("C1CC2(C1)CCC3(CC2)CC=C3")
    assert name_spiro_system(mol) is None


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    # branched trispiro (was a leaked 'cyclononane', dropping 3 rings)
    ("C1CC12CCC1(CC1)CCC1(CC1)CC2", "trispiro[2.2.2^6.2.2^11.2^3]pentadecane"),
    # linear 4-ring trispiro
    ("C1CCC2(CC1)CCC3(CC2)CCC4(CC3)CCCC4", "trispiro[4.2.2.5^11.2^8.2^5]icosane"),
])
def test_tri_plus_polyspiro_now_named(smiles, expected):
    """ (item 2): tri+ polyspiro with superscript revisit locants now builds
    (capability gain); the spiro module returns the trispiro name and it OPSIN
    round-trips to the input structure (verified 2026-09-04, ITEM2-VERIFICATION.md).
    Was fail-closed pending the superscript-locant engine."""
    mol = Chem.MolFromSmiles(smiles)
    assert name_spiro_system(mol)[0] == expected
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# 6a — propene/propyne bond-locant elision (d))
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("C=CC", "propene"),
    ("CC#C", "propyne"),
    ("C=C", "ethene"),
    # substituent restores the locant
    ("C=CCCl", "3-chloroprop-1-ene"),
    # length-4+ always cites
    ("C=CCC", "but-1-ene"),
    ("CC=CC", "but-2-ene"),
    # diene unaffected (two locants, no elision path)
    ("C=C=C", "propa-1,2-diene"),
])
def test_trinuclear_bond_locant_elision(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# 6b — heteromonocyclic mancude-substituent hydro emitter
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)CCC1CCC=CO1",
     "3-(3,4-dihydro-2H-pyran-2-yl)propanoic acid"),
    ("OC(=O)CCC1=CCCCO1",
     "3-(3,4-dihydro-2H-pyran-6-yl)propanoic acid"),
    ("OC(=O)CCC1CC=CO1",
     "3-(2,3-dihydrofuran-2-yl)propanoic acid"),
    ("OC(=O)CCC1CCC=CS1",
     "3-(3,4-dihydro-2H-thiopyran-2-yl)propanoic acid"),
    ("OC(=O)CCC1CCC=CN1",
     "3-(1,2,3,4-tetrahydropyridin-2-yl)propanoic acid"),
    ("OC(=O)CCC1CCCC=N1",
     "3-(2,3,4,5-tetrahydropyridin-2-yl)propanoic acid"),
])
def test_hetero_hydro_substituent(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    # saturated het ring keeps its plain stem
    ("OC(=O)CCC1CCCCO1", "3-(oxan-2-yl)propanoic acid"),
    # aromatic het rings keep their mancude stems
    ("OC(=O)CCc1ccco1", "3-(furan-2-yl)propanoic acid"),
    ("OC(=O)CCc1ccncc1", "3-(pyridin-4-yl)propanoic acid"),
    # carbocyclic ene-substituent path untouched
    ("OC(=O)CCC1CCC=CC1", "3-(cyclohex-3-en-1-yl)propanoic acid"),
])
def test_hydro_emitter_protections(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
def test_unsaturated_het_ring_never_saturated_stem():
    """The oxanyl wrong-constitution guard: an unsaturated heteromonocycle
    whose emitter declines must return None (fail closed), never the
    saturated dictionary stem."""
    mol = Chem.MolFromSmiles("OC(=O)CCC1OC=CO1")  # 2 heteroatoms -> declines
    ring = next(
        r for r in mol.GetRingInfo().AtomRings()
    )
    attach = next(
        i for i in ring
        if any(n.GetIdx() not in ring for n in
               mol.GetAtomWithIdx(i).GetNeighbors())
    )
    assert _unsaturated_heteromonocyclic_substituent(mol, ring, attach) is None
    assert get_ring_substituent_name(mol, ring, attach) is None


@pytest.mark.unit
def test_het_attach_on_heteroatom_declines():
    """N-attached would need added indicated hydrogen -> decline."""
    mol = Chem.MolFromSmiles("OC(=O)CCN1CCC=CC1")
    ring = next(r for r in mol.GetRingInfo().AtomRings())
    n_idx = next(i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "N")
    assert _unsaturated_heteromonocyclic_substituent(mol, ring, n_idx) is None


# ---------------------------------------------------------------------------
# 6c — acyl pseudohalides
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("CCCC(=O)N=[N+]=[N-]", "butanoyl azide"),       # BB verbatim
    ("CC(=O)N=[N+]=[N-]", "acetyl azide"),
    ("O=C(N=[N+]=[N-])c1ccccc1", "benzoyl azide"),
    ("CC(=O)N=C=O", "acetyl isocyanate"),            # BB verbatim
    ("CCC(=O)C#N", "propanoyl cyanide"),             # BB verbatim
    ("CC(=O)C#N", "acetyl cyanide"),
    ("O=C(C#N)c1ccccc1", "benzoyl cyanide"),
    ("CC(C)C(=O)N=[N+]=[N-]", "2-methylpropanoyl azide"),
    # -3, the Blue Book '3-chloropropanoyl cyanide (PIN)'): a
    # substituent on the acyl chain must number from the carbonyl C = 1
    #. The nitrile carbon inflated the locant before this fix
    # ('4-chloro...', which OPSIN rejected -> abstain).
    ("ClCCC(=O)C#N", "3-chloropropanoyl cyanide"),
    # -3 /, the Blue Book): a diacyl
    # DIpseudohalide -> 'oxalyl dicyanide' (was dropped to 'acetyl cyanide').
    ("N#CC(=O)C(=O)C#N", "oxalyl dicyanide"),
])
def test_acyl_pseudohalides(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    # -3, the Blue Book 'CH3-SO2-CN methanesulfonyl cyanide
    # (PIN)'): the cyanide of a sulfonic acid, functional-class naming. Was the
    # RT-correct-but-non-PIN '(methanesulfonyl)methanenitrile'.
    ("CS(=O)(=O)C#N", "methanesulfonyl cyanide"),
    ("CCS(=O)(=O)C#N", "ethanesulfonyl cyanide"),
    # plain sulfonyl halide + sulfone + nitrile keep their names (regressions).
    ("CS(=O)(=O)Cl", "methanesulfonyl chloride"),
    ("CCC#N", "propanenitrile"),
])
def test_sulfonyl_cyanide(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    # plain halides through the same word table — untouched
    ("CCCC(=O)Cl", "butanoyl chloride"),
    ("CC(=O)Cl", "acetyl chloride"),
    ("ClCCC(=O)Cl", "3-chloropropanoyl chloride"),
    ("O=C(Cl)CCCC(=O)Cl", "pentanedioyl dichloride"),
    # bare azide/isocyanate/nitrile (no acyl) keep their substitutive names
    ("CCN=[N+]=[N-]", "azidoethane"),
    ("CCN=C=O", "isocyanatoethane"),
    ("CCC#N", "propanenitrile"),
    # -3: sulfonyl cyanide is now the functional-class PIN
    # 'methanesulfonyl cyanide', the Blue Book), not the prior
    # RT-correct-but-non-PIN '(methanesulfonyl)methanenitrile'. Asserted in
    # test_sulfonyl_cyanide above; the acyl-pseudohalide ([CX3] anchor) path
    # still does NOT claim it (a separate sulfonyl_cyanide FG does).
])
def test_pseudohalide_protections(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
def test_carbonyl_dicyanide_retained_pin(_validity_gate_on):
    """ (Wave-2 completion C): functional-class IS the PIN for
    carbonic-acid nitriles -- retained exact-SMILES row, OPSIN-RT verified."""
    assert name_compound("O=C(C#N)C#N") == "carbonyl dicyanide"


# ---------------------------------------------------------------------------
# 6c — mixed divalent-chalcogen bridge
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    # /: the compound '(methylsulfanyl)oxy' prefix already
    # carries parentheses, so its citation mark escalates to brackets even when
    # the single-substituent locant '1' is omitted -- BB 27914
    # '[(methylsulfanyl)oxy]ethane (PIN)'. The bare '(methylsulfanyl)oxyethane'
    # was a dropped-outer-bracket defect on the unlocanted citation path.
    ("CCOSC", "[(methylsulfanyl)oxy]ethane"),
    # (item 2): COSC now names via the (methoxysulfanyl) parent (both the
    # old and new spellings round-trip to the same structure; RT-verified
    # 2026-09-04, ITEM2-VERIFICATION.md).
    ("COSC", "(methoxysulfanyl)methane"),
])
def test_mixed_chalcogen_bridge(smiles, expected):
    """R-O-S-R' concatenated prefix — was a mangled
    '(hydroxymethane-SO-thioperoxyl)ethane' wrong-constitution fragment."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    # like-pair chalcogen bridges untouched
    ("COOC", "(methylperoxy)methane"),
    ("CCOOC", "(methylperoxy)ethane"),
    ("CSSC", "(methyldisulfanyl)methane"),
    ("CCSSC", "(methyldisulfanyl)ethane"),
    # terminal-H thioperoxol suffixes untouched
    ("CSO", "methane-SO-thioperoxol"),
    ("COS", "methane-OS-thioperoxol"),
])
def test_chalcogen_bridge_protections(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
def test_s_attached_mixed_bridge_now_named(_validity_gate_on):
    """ (item 2): -S-O-R now names via the (methoxysulfanyl) contraction
    (capability gain, OPSIN-RT verified 2026-09-04, ITEM2-VERIFICATION.md);
    was fail-closed pending the alkoxy contraction."""
    assert name_compound("CCSOC") == "(methoxysulfanyl)ethane"


# ---------------------------------------------------------------------------
# 6c — phane production refusal
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("C1Cc2ccc(cc2)CCc2ccc1cc2", "1,4(1,4)-dibenzenacyclohexaphane"),
])
def test_phane_production_now_named(smiles, expected):
    """ (item 2): phane skeletal PINs now produced via the hard-gated,
    formula-conservation-vetoed rules.phane composer (capability gain). No phane
    name is OPSIN-parseable, so correctness is verified by molecular-formula
    conservation (C16H16 preserved), not round-trip — see
    ITEM2-VERIFICATION.md.

    Suite fix j4 (TRIAGE g3 C10a / g6 C20): the ortho row
    ('1,5(1,2)-dibenzenacyclooctaphane') moved to the tests below -- it is not a
    cyclophane for a PIN."""
    assert name_compound(smiles) == expected


# Suite fix j4 (TRIAGE g3 C10a / g6 C20): two benzene rings ortho-fused to a
# 10-membered alicyclic ring. (1) (the Blue Book): a cyclophane
# for a PIN needs a mancude ring "attached to adjacent atoms or chains at
# nonadjacent ring positions"; (:23835-23841): "Mancude systems
# attached to adjacent atoms of an alicyclic ring are either fused systems or
# bridged fused systems [...] A cyclophane name is not allowed."
_ORTHO_PHANE_LIKE = [
    "C1CCc2ccccc2CCCc2ccccc21",
    "C1CCCc2ccccc2CCCCc2ccccc2C1",
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles", _ORTHO_PHANE_LIKE)
def test_ortho_attached_benzenes_get_no_phane_name(smiles):
    from orthonym.rules.phane import build_phane_pin
    assert build_phane_pin(Chem.MolFromSmiles(smiles)) is None
    assert "phane" not in name_compound(smiles)


@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason=(
    "needs the hydro fusion name for mancude rings ortho-fused to a large "
    "alicyclic ring (P-52.2.5.2.1, BlueBookV2.md:23835-23841; class example "
    ":23839 'dodecahydrobenzo[14]annulene (PIN)'); PIN spelling ASSUMED "
    "(OPSIN full-InChIKey exact); TODO .planning/preexisting-triage/TRIAGE.md "
    "'Suite fix -- j4-pin-labels-a'"))
def test_ortho_attached_benzenes_fused_pin():
    assert name_compound(_ORTHO_PHANE_LIKE[0]) == (
        "5,6,7,12,13,14-hexahydrodibenzo[a,f][10]annulene")


@pytest.mark.unit
def test_phane_composer_machinery_intact():
    """The rules.phane composer classifies + composes the skeletal PIN
    (upgraded from the old '[2.2]paracyclophane' bracket form)."""
    from orthonym.rules.phane import is_cyclophane, name_cyclophane
    mol = Chem.MolFromSmiles("C1Cc2ccc(cc2)CCc2ccc1cc2")
    assert is_cyclophane(mol)
    assert name_cyclophane(mol) == "1,4(1,4)-dibenzenacyclohexaphane"
