""" W8 P1 cross-cutting-root fixes — unit tests.

R12: ring-fusion diol on a mancude fused carbocycle. The bare
hydro-parent emitter `name_hydrogenated_fused_carbocycle` counts sp3 ring atoms
as "hydro" positions and emits `<locants>-dihydro<parent>` with NO substituent
slot, so for `naphthalene-4a,8a-diol` it dropped both OH and emitted
`4a,8a-dihydronaphthalene` — a DIFFERENT molecule. Gated, caught it; but
gate-off (no Java) it shipped the atom-dropped name. Task 1.3 = source-level
atom-conservation veto (fail closed); Task 1.4 = actually name the diol.
"""
import pytest
from rdkit import Chem

from orthonym.rules.partial_saturation import name_hydrogenated_fused_carbocycle
import orthonym.namer as _namer

pytestmark = pytest.mark.unit

# OPSIN-authoritative naphthalene-4a,8a-diol (BB 3755/24744/26866 verbatim PIN).
DIOL = "OC12C=CC=CC1(O)C=CC=C2"


def test_r12_fusion_diol_emitter_fails_closed():
    """The bare hydro-parent emitter must DECLINE the fusion-diol rather than
    drop both OH and emit `4a,8a-dihydronaphthalene`."""
    mol = Chem.MolFromSmiles(DIOL)
    assert name_hydrogenated_fused_carbocycle(mol) is None


@pytest.mark.parametrize("smiles,expected", [
    # Unsubstituted partially-saturated naphthalene-types: must be UNAFFECTED by
    # the atom-conservation veto (no exocyclic heavy atoms).
    ("C1CCC2=CCCCC2C1", "1,2,3,4,4a,5,6,7-octahydronaphthalene"),
    ("C1CC2=CCCC=C2CC1", "1,2,3,4,6,7-hexahydronaphthalene"),
])
def test_r12_unsubstituted_hydrofused_unregressed(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert name_hydrogenated_fused_carbocycle(mol) == expected


def test_r7_benzene_carbothioamide():
    """R7, BB 18762): a ring-attached -C(=S)NH2 on benzene is named
    with the -carbothioamide suffix. PIN uses the no-locant `benzenecarbo...` form
    (BB 6666 benzenecarbodithioic acid, 30000 benzenecarboximidic acid), matching
    the existing benzenecarboximidamide / benzenecarbothiohydrazide siblings.
    Was abstaining (fell to substituent naming -> mangled -> unknown)."""
    from orthonym.namer import name_compound
    assert name_compound("NC(=S)c1ccccc1", style="pin") == "benzenecarbothioamide"


@pytest.mark.parametrize("smiles,label", [
    ("CNC(=S)c1ccccc1", "N-methyl (suffix role)"),
    ("CN(C)C(=S)c1ccccc1", "N,N-dimethyl (suffix role)"),
    ("S=C(Nc1ccccc1)c1ccccc1", "N-phenyl (suffix role)"),
    ("CNC(=S)c1ccc(C(=O)O)cc1", "N-methyl + senior COOH (prefix role)"),
])
def test_r7_n_substituted_thioamide_no_atom_drop(smiles, label):
    """An N-SUBSTITUTED benzene thioamide cannot be cited by the bare
    -carbothioamide suffix / carbamothioyl prefix (no N-substituent support), so
    the carbothioamide detector must NOT fire for it — else it silently drops the
    N-substituent and emits a name for a DIFFERENT molecule (e.g. the methyl-less
    'benzenecarbothioamide' / '4-carbamothioylbenzoic acid'). Test env runs
    gate-off, so this exercises the raw no-Java path directly. Fail closed."""
    from orthonym.namer import name_compound
    out = name_compound(smiles, style="pin")
    assert out not in ("benzenecarbothioamide", "4-carbamothioylbenzoic acid"), \
        f"{label}: N-substituent dropped -> {out!r}"


def test_r6_benzophenone_demoted_to_diphenylmethanone():
    """R6, BB 28326/28378): benzophenone is retained for GENERAL
    nomenclature only; the PIN is the systematic diphenylmethanone. Mirror of the
    already-demoted acetophenone -> 1-phenylethan-1-one."""
    from orthonym.namer import name_compound
    assert name_compound("O=C(c1ccccc1)c1ccccc1", style="pin") == "diphenylmethanone"


def test_r6_substituted_diaryl_ketone_unregressed():
    """The systematic diaryl-ketone builder must stay correct for non-retained
    cases (regression guard for the benzophenone demotion)."""
    from orthonym.namer import name_compound
    assert name_compound("O=C(c1ccccc1)c1ccc(C)cc1", style="pin") == "(4-methylphenyl)phenylmethanone"


def test_r12_fusion_diol_no_java_no_atom_drop():
    """Gate-off (no-Java) the namer must NOT emit an atom-dropping
    `...dihydronaphthalene` for the diol — fail closed (or name it correctly)."""
    saved = _namer._DISABLE_VALIDITY_GATE
    _namer._DISABLE_VALIDITY_GATE = True
    try:
        out = _namer.name_compound(DIOL, style="pin")
    finally:
        _namer._DISABLE_VALIDITY_GATE = saved
    # Either fail closed, or emit the correct diol PIN — never the atom-dropped name.
    assert "dihydronaphthalene" not in (out or "") or out == "naphthalene-4a,8a-diol"
