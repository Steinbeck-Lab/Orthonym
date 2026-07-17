"""v24 W8 P1 cross-cutting-root fixes — unit tests.

R12 (P-58.2.2.3): ring-fusion diol on a mancude fused carbocycle. The bare
hydro-parent emitter `name_hydrogenated_fused_carbocycle` counts sp3 ring atoms
as "hydro" positions and emits `<locants>-dihydro<parent>` with NO substituent
slot, so for `naphthalene-4a,8a-diol` it dropped both OH and emitted
`4a,8a-dihydronaphthalene` — a DIFFERENT molecule. Gated, SELF-01 caught it; but
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
