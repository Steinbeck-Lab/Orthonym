"""P7 (Wave-8) natural-product semisystematic-parent recognizer tests.

7a.1 — permissive-matcher ring-topology safety (accuracy-first guard).

The flexible scaffold matcher (``perception/natural_products.py``) builds
bond-generic, stereo-free queries so unsaturated / substituted derivatives
still match their saturated parent. A concern was raised that this could let a
homologated skeleton (e.g. a D-homosteroid, ring D expanded to six members,
ring-size multiset {6,6,6,6}) be mis-recognised as its {6,6,6,5} retained
parent (androstane/gonane) — which would ship a WRONG name even with the OPSIN
gate on, because these retained parents are OPSIN-unparseable and therefore
name-exact-whitelisted (the RT gate cannot catch them).

Reproduce-first result: the concern does NOT reproduce. ``makeBondsGeneric``
relaxes only bond ORDER, never ring SIZE — RDKit substructure matching still
requires every query ring-closure bond to be present in the target, so a
{6,6,6,6} skeleton cannot satisfy a {6,6,6,5} query. Genuine homolog/nor
skeletons already fail closed (detect -> None). These tests lock that invariant
in so a future looser matcher (or a new scaffold) cannot silently reopen it.

(The originally-proposed "D-homo leak" SMILES ``CC12CCCCC1CCC1C2CCC2(C)CCCC12``
was a hand-drawn structure that is in fact *constitutionally androstane* — same
skeleton InChIKey QZLYKIGBANMMBK, C19H32, rings {5,6,6,6}. Recognising it as
androstane is correct, not a leak.)
"""

import pytest
from rdkit import Chem

from orthonym.perception.natural_products import detect_natural_product
from orthonym.namer import Orthonym

# Gate-off raw namer: exposes the emitter the OPSIN validity gate would mask.
RAW = Orthonym(_disable_opsin_validity_gate=True)

# Ground-truth exact scaffold SMILES from data/natural_products.py — these MUST
# keep matching (the matcher's decoration path must stay intact).
GONANE = "C1CC[C@H]2C(C1)CC[C@H]1[C@@H]3CCC[C@H]3CC[C@@H]12"          # {6,6,6,5}
ANDROSTANE = "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2"      # {5,6,6,6}
ESTRANE = "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@@H]3[C@H]1CC2"
CHOLESTANE = "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C"
# Cholesterol: an unsaturated + hydroxylated cholestane derivative — the
# flexible matcher must still recognise it (decoration path unaffected).
CHOLESTEROL = "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"

# Genuine homologated / nor skeletons — ring topology differs from any retained
# steroid parent, so they MUST fail closed (never claim a {6,6,6,5} name).
D_HOMOGONANE = "C1CC[C@H]2C(C1)CC[C@H]1[C@@H]3CCCC[C@H]3CC[C@@H]12"    # {6,6,6,6}
C_HOMOGONANE = "C1CC[C@H]2C(C1)CCC[C@H]1[C@@H]3CCC[C@H]3CC[C@@H]12"   # {6,6,7,5}
A_NORGONANE = "C1C[C@H]2C(C1)CC[C@H]1[C@@H]3CCC[C@H]3CC[C@@H]12"      # {5,5,6,6}
D_HOMOANDROSTANE = "C[C@@]12CCCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2"  # {6,6,6,6}


def _scaffold_name(smi):
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"invalid test SMILES: {smi}"
    info = detect_natural_product(mol)
    return None if info is None else info["scaffold_name"]


@pytest.mark.parametrize("smi,expected", [
    (GONANE, "gonane"),
    (ANDROSTANE, "androstane"),
    (ESTRANE, "estrane"),
    (CHOLESTANE, "cholestane"),
    (CHOLESTEROL, "cholestane"),
])
def test_legitimate_steroid_scaffolds_matched(smi, expected):
    # The matcher must recognise exact + decorated steroids (decoration path).
    assert _scaffold_name(smi) == expected


@pytest.mark.parametrize("label,smi", [
    ("D-homogonane {6,6,6,6}", D_HOMOGONANE),
    ("C-homogonane {6,6,7,5}", C_HOMOGONANE),
    ("A-norgonane {5,5,6,6}", A_NORGONANE),
    ("D-homoandrostane {6,6,6,6}", D_HOMOANDROSTANE),
])
def test_homolog_and_nor_skeletons_fail_closed(label, smi):
    # Ring-size mismatch must not be recognised as a retained steroid parent.
    assert _scaffold_name(smi) is None, f"{label} wrongly matched"


def test_homolog_skeleton_namer_emits_no_steroid_retained_name():
    # Gate-off (no-Java) path must not ship a retained steroid name for a
    # genuine D-homosteroid; a systematic von Baeyer name (or unknown) is fine.
    out = RAW.name(D_HOMOANDROSTANE)
    for retained in ("androstane", "gonane", "estrane", "cholestane"):
        assert retained not in out, f"leaked '{retained}' in: {out!r}"
