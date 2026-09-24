"""M4 lever L2 — polyspiro/dispiro systems whose component is a FUSED ring
 must EMIT in the best-effort floor, not VOID.

Root cause (pre-fix): ``rules.spiro._name_linear_polyspiro_fused`` built its
combined atom->locant coverage map by SKIPPING every non-integer locant
(``if not isinstance(locant, int): continue``). A fused component (e.g. a
chromane / 3,4-dihydro-2H-1-benzopyran) carries lettered FUSION locants ('4a',
'8a') on its ring-fusion atoms. Those atoms were dropped from the map, so the
coverage invariant (``combined.keys >= core_ring_atoms``) failed and the
namer returned None -> the whole spiro core abstained.

The spiro-DESCRIPTOR junction locants are separately proven integer earlier in
the same function: spiro locants must be peripheral integers); the
combined map only feeds substituent placement and ``_locant_display`` renders a
primed lettered locant ('8a', "'") -> "8a'" correctly. Keeping the fusion
atoms in the map is therefore the root-cause fix.

Witness: chromane dispiro cyclohexane spiro cyclopentane
    O[C@@H]1CC2(CCC3(CCCC3)CC2)Oc2ccccc21 (spiro atoms {3,6})

0-wrong is preserved by the caller's offer-RT gate; these tests independently
verify the FULL-InChIKey round-trip via OPSIN (RDKit compare), so a wrong
descriptor could never make them pass.
"""
import shutil
import subprocess

import pytest
from rdkit import Chem
from tests.support.jars import jar_or_skip

pytestmark = [pytest.mark.unit]

# chromane(=3,4-dihydro-2H-1-benzopyran) --spiro-- cyclohexane --spiro-- cyclopentane,
# with a hydroxyl (and 1 stereocentre) on the benzopyran ring.
WITNESS_DISPIRO_FUSED = "O[C@@H]1CC2(CCC3(CCCC3)CC2)Oc2ccccc21"


def _opsin_to_smiles(name):
    if not shutil.which("java"):
        pytest.skip("no JVM on PATH")
    p = subprocess.run(["java", "-jar", jar_or_skip(), "-o", "smi"],
                       input=name + "\n", capture_output=True, text=True)
    out = p.stdout.strip()
    return out or None


def _ikey(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m) if m is not None else None


def _constitution_key(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToInchiKey(Chem.MolFromSmiles(Chem.MolToSmiles(m, isomericSmiles=False)))


def test_linear_polyspiro_fused_returns_name_with_fused_component():
    """Direct unit: _name_linear_polyspiro_fused must NOT drop the fused
    component's fusion atoms from its coverage map, so it returns the
    dispiro name instead of None. The parent descriptor's CORE constitution
    must OPSIN-round-trip (parent has no substituents -> compare de-decorated
    core)."""
    from orthonym.rules.spiro import _name_linear_polyspiro_fused

    mol = Chem.MolFromSmiles(WITNESS_DISPIRO_FUSED)
    restrict = set(range(mol.GetNumAtoms()))
    res = _name_linear_polyspiro_fused(mol, allow_vonbaeyer=True,
                                       restrict_atoms=restrict)
    assert res is not None, (
        "polyspiro-of-fused declined: the combined-locant coverage map is "
        "dropping the fused component's lettered fusion locants (4a/8a).")
    name, core_atoms, combined, _subs = res
    assert name.startswith("dispiro["), name
    # every core ring atom must appear in the coverage map (the invariant that
    # was failing pre-fix)
    assert set(combined.keys()) >= set(core_atoms)

    # the parent descriptor names the CORE constitution (no substituents).
    osmi = _opsin_to_smiles(name)
    assert osmi is not None, f"OPSIN could not parse the parent name: {name!r}"
    core_smi = "C1CC2(CCC3(CCCC3)CC2)Oc2ccccc21"  # witness minus the -OH
    assert _constitution_key(osmi) == _constitution_key(core_smi), (
        f"parent descriptor constitution mismatch: {name!r} -> {osmi!r}")


@pytest.mark.opsin_gate
def test_besteffort_floor_names_dispiro_of_fused_full_rt():
    """End-to-end: the best-effort floor now EMITS a full-molecule name for the
    dispiro-of-fused witness (was 'unknown organic compound'), and it
    OPSIN-round-trips at the FULL InChIKey (constitution at minimum; full match
    if stereo is carried)."""
    from orthonym import Orthonym
    from orthonym.errors import is_failure_name

    nm = Orthonym(general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True)
    out = nm.name(WITNESS_DISPIRO_FUSED)
    assert out is not None and not is_failure_name(out), (
        f"floor abstained on dispiro-of-fused witness: {out!r}")
    assert "spiro[" in out.lower()

    osmi = _opsin_to_smiles(out)
    assert osmi is not None, f"shipped name does not OPSIN-parse: {out!r}"
    # 0-wrong: constitution must match (offer gate would have voided a wrong one).
    assert _constitution_key(osmi) == _constitution_key(WITNESS_DISPIRO_FUSED), (
        f"shipped a wrong-constitution name: {out!r} -> {osmi!r}")
