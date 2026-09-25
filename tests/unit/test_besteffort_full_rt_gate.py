"""Best-effort/complete tier is FULL-round-trip-gated: it ships a name only when
OPSIN re-perceives it to the input's FULL InChIKey (constitution AND stereo AND
charge), else abstains. A stereo-incomplete, OPSIN-unparseable, or wrong-valence
name is NOT a lesser name -- it is a WRONG name and must abstain (user-directed
2026-08-30; a review precision-leak + the 1500-mol head-to-head).

The PIN/default tier is UNCHANGED (its correct-by-construction OPSIN-unparseable
carve-outs -- inositol / np-stereoparent / thioperoxol /... -- still ship), so
the 1656 PIN gold gate is byte-identical. These tests run BEST-EFFORT only.
"""
import ast
import subprocess
import sys
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

sys.path.insert(0, "/home/kohulan/OpenSTOUT/Orthonym/src")
from orthonym.jvm_flags import java_cmd  # noqa: E402
from tests.support.jars import jar_or_skip

PROJ = "/home/kohulan/OpenSTOUT/Orthonym"


def _be(smiles):
    # Name in a FRESH subprocess: the validity gate reads a module-global OPSIN
    # oracle + jar-presence state that a shared pytest process (with other tests'
    # OPSIN calls / env) can leave in a fail-open configuration, so the
    # deterministic production behaviour (one molecule per process, as the eval
    # harness and the head-to-head shards run) is only reproduced in a clean
    # interpreter -- the same subprocess pattern test_floor_ring_stereo_completion uses.
    code = (
        "import logging; logging.disable(logging.CRITICAL)\n"
        "from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')\n"
        "from orthonym.namer import Orthonym\n"
        "eng = Orthonym(general_fallback=True, general_fallback_unverified=False, allow_aromatic_general=True)\n"
        f"print(repr(eng.name({smiles!r})))\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PROJ)
    out = p.stdout.strip().splitlines()
    return ast.literal_eval(out[-1]) if out else None


def _abstains(name):
    return (not name) or name.startswith("unknown") or "not supported" in name


def _full_rt(name, smiles):
    if _abstains(name):
        return False
    p = subprocess.run(java_cmd() + ["-jar", jar_or_skip(), "-o", "smi"],
                       input=name + "\n", capture_output=True, text=True)
    out = p.stdout.strip().splitlines()
    out = out[0] if out else ""
    mo = Chem.MolFromSmiles(out) if out else None
    if mo is None:
        return False
    return inchi.MolToInchiKey(mo) == inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


# OPSIN cannot parse these ring stereo descriptors (multi-centre saturated-ring
# absolute stereo) -> no full-exact name exists -> best-effort must ABSTAIN,
# never ship the OPSIN-unparseable full name (was the precision leak).
@pytest.mark.parametrize("smiles", [
    "O=C(/C=C/c1ccc(O)c(O)c1)O[C@H]1[C@H](O)C[C@](O)(C(=O)O)C[C@H]1O",  # chlorogenic acid
])
def test_opsin_unparseable_ring_stereo_abstains(smiles):
    assert _abstains(_be(smiles)), f"shipped an unverifiable name: {_be(smiles)!r}"


# 2026-09-25 (pre-existing-failures plan, Task 4 continuation; controller
# ruling): a name OPSIN rejects ONLY for its PSEUDOASYMMETRIC (lowercase r/s)
# descriptor is no longer unverifiable. OPSIN 2.9.0 parses no such descriptor,
# so '(1R,3s,5S)-tropan-3-ol' can never round-trip, but its stereo-stripped form
# round-trips to the input's full InChIKey (with the pseudoasymmetric centre left
# unspecified) and the centres labeller gives C-3 the same single code 's'
# (namer._pseudoasymmetric_name_verified). Best-effort ships it only as the last
# resort, when no OPSIN-round-trippable name exists. Chlorogenic acid's rejected
# descriptors are true stereocentres, so it still abstains (above).
def test_pseudoasymmetric_ring_stereo_ships_verified():
    from orthonym.namer import _pseudoasymmetric_name_verified
    smiles = "CN1[C@@H]2CC[C@H]1C[C@H](O)C2"  # tropan-3-ol
    name = _be(smiles)
    assert name == "(1R,3s,5S)-tropan-3-ol"
    assert _pseudoasymmetric_name_verified(name, smiles)
    assert not _pseudoasymmetric_name_verified("(1R,3r,5S)-tropan-3-ol", smiles)


def test_out_of_scope_organotin_abstains():
    # organo-tin: PIN abstains ("tin compound (not supported)"); best-effort
    # must too (Sn is in REPLACEMENT_TERMS, so the core would otherwise build a
    # `stanna` name whose OPSIN re-perception mis-valences Sn and never full-RTs).
    smi = "CC(C)OP(=O)(C(C[Sn]Cl)P(=O)(OC(C)C)OC(C)C)OC(C)C"
    assert _abstains(_be(smi)), f"named an out-of-scope organometallic: {_be(smi)!r}"


def test_in_scope_organosilicon_still_names():
    # an IN-scope covalent Si heterocycle must NOT be caught by the metal guard.
    smi = "C[Si]1(C)CCCCC1"  # 1,1-dimethylsilinane
    assert _full_rt(_be(smi), smi), f"in-scope organosilicon lost: {_be(smi)!r}"


def test_nameable_fused_cage_still_full_rt():
    # grayanotoxane skeleton: a systematic tetracyclo name DOES full-round-trip,
    # so it must still ship (the fix abstains only the unverifiable ones).
    smi = "C[C@H]1C[C@@]23CC[C@H]4[C@@H](CCC4(C)C)[C@H]2CC[C@]3(C)[C@@H]1O"
    assert _full_rt(_be(smi), smi), f"nameable cage lost: {_be(smi)!r}"


@pytest.mark.parametrize("smiles", ["CCO", "CC(=O)Oc1ccccc1C(=O)O",
                                    "Cn1cnc2c1c(=O)n(C)c(=O)n2C"])
def test_controls_still_full_rt(smiles):
    assert _full_rt(_be(smiles), smiles)
