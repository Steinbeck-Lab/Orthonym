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


# 7a.2 — complex polycyclic diterpene/triterpene stereoparents (P-101.2.7 Table
# 10.1c). OPSIN-unparseable -> name-exact carve-out. Structures two-source
# verified (PubChem + NCI CACTUS full InChIKey agreement). These are direct
# structural analogs of the shipped abietane/lanostane-class name-exact PINs
# (unwieldy systematic von-Baeyer name -> the semisystematic Table-10.1 name is
# the PIN per P-101.2 BB:50985 "more complicated structure -> semisystematic").
@pytest.mark.parametrize("smi,expected", [
    # podocarpane — tricyclic diterpane (abietane class); was 'unknown'.
    ("CC1(C)CCC[C@]2(C)[C@H]3CCCC[C@@H]3CC[C@@H]12", "podocarpane"),
    # protostane — pentacyclic triterpane (lanostane/dammarane class).
    ("CC(C)CCC[C@@H](C)[C@H]1CC[C@@]2(C)[C@H]1CC[C@H]1[C@@]3(C)CCCC(C)(C)[C@@H]3CC[C@@]12C",
     "protostane"),
    # grayanotoxane — tetracyclic diterpane (kaurane/atisane class).
    ("C[C@@H]1[C@@H]2CCC(C)(C)[C@H]2CC[C@@]23C[C@@H](CC[C@@H]12)[C@@H](C)C3", "grayanotoxane"),
    # rosane — tricyclic diterpane (abietane class); was mis-named (substituent drop).
    ("CC[C@]1(C)CC[C@]2(C)[C@H](CC[C@@H]3[C@H]2CCCC3(C)C)C1", "rosane"),
])
def test_name_exact_diterpene_triterpene_parents(smi, expected):
    can = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    assert RAW.name(can) == expected   # gate-off raw namer (name-exact path)


# 7a.2b — complex polycyclic alkaloid stereoparents (P-101.2.7 Table 10.1a).
# Clean CHN-only saturated parent hydrides, OPSIN-unparseable -> name-exact, two-
# source verified (PubChem + NCI CACTUS full InChIKey). Direct analogs of the
# shipped yohimban/aspidospermidine/vincane class (semisystematic PIN, P-101.2).
# Several currently ship WRONG atom-dropped names (cevane->methylpiperidine,
# corynan->diethylpiperidine, emetan->tetrahydroisoquinoline) -> this also closes
# those structural leaks.
@pytest.mark.parametrize("smi,expected", [
    ("C[C@H]1CC[C@H]2[C@H](C)[C@H]3CC[C@@H]4[C@@H](C[C@H]5[C@H]4CC[C@@H]4CCCC[C@@]45C)[C@@H]3CN2C1", "cevane"),
    ("C[C@H]1[C@H]2CC[C@H]3[C@@H]4CCC5CCCC[C@]5(C)[C@H]4CC[C@]23CN1C", "conanine"),
    ("CC[C@H]1C[C@H]2c3[nH]c4ccccc4c3CCN2C[C@@H]1CC", "corynan"),
    ("CC[C@@H]1CN2CC[C@@]3(CNc4ccccc43)[C@@H]2C[C@@H]1CC", "corynoxan"),
    ("CC[C@H]1CN2CCc3ccccc3[C@@H]2C[C@@H]1C[C@H]1NCCc2ccccc21", "emetan"),
    ("c1ccc2c(c1)CCN1CC[C@@H]3CCCC[C@@]231", "erythrinan"),
    ("c1ccc2c(c1)CN1CC[C@@H]3CCC[C@@H]2[C@@H]31", "galanthan"),
    ("c1ccc2c(c1)CC[C@@]13CCCC[C@@]21CCN3", "hasubanan"),
    ("c1ccc2c(c1)N[C@]13CC[C@]45CCCN6CC(C[C@@H]1C4)[C@]23[C@@H]65", "kopsan"),
    ("C1C[C@@H]2CC[C@@H]3CCC[C@@]24[C@@H]3CCCN4C1", "lycopodane"),
    ("C[C@H]1CC[C@@H]2[C@@H](C)[C@H]3[C@H](C[C@H]4[C@@H]5CCC6CCCC[C@]6(C)[C@H]5CC[C@]34C)N2C1", "solanidane"),
    ("CC[C@H]1CN2CCc3ccccc3[C@@H]2C[C@@H]1C[C@H]1NCCc2c1[nH]c1ccccc21", "tubulosan"),
])
def test_name_exact_alkaloid_parents(smi, expected):
    can = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    assert RAW.name(can) == expected


# 7a.2c — bicyclic+ sesqui/di/sesterterpene parents (P-101.2.7 Table 10.1c).
# Complex polycyclic (2-4 rings), analogs of the shipped cadinane/guaiane/eudesmane
# (bicyclic sesquiterpane) class -> semisystematic PIN. Two-source verified,
# OPSIN-unparseable -> name-exact. himachalane/ophiobolane were mis-named
# (substituent/ring drop); picrasane/trichothecane carry a skeletal ether O.
@pytest.mark.parametrize("smi,expected", [
    ("C[C@H]1CC[C@H]2C(C)(C)CCC[C@]2(C)[C@H]1C", "drimane"),
    ("CC(C)[C@@H]1CC[C@H](C)[C@@H]2CCC[C@@]2(C)C1", "ambrosane"),
    ("CC(C)[C@@H]1CC[C@H]2CCC[C@H](C)[C@@]2(C)C1", "eremophilane"),
    ("C[C@@H]1CCC[C@@H]2CC[C@@H]3[C@@H](C3(C)C)[C@]21C", "aristolane"),
    ("CC1CC[C@H]2C(C)CCCC(C)(C)[C@H]2C1", "himachalane"),
    ("CC(C)CCC[C@H](C)[C@H]1CC[C@]2(C)C[C@H]3[C@H](CC[C@@H]3C)[C@@H](C)CC[C@@H]12", "ophiobolane"),
    ("C[C@@H]1CCC[C@]2(C)[C@H]3CC[C@H](C)[C@@H]4CCO[C@H](C[C@@H]12)[C@]34C", "picrasane"),
    ("CC1CC[C@@]2(C)[C@@H](C1)O[C@@H]1CC[C@@]2(C)[C@@H]1C", "trichothecane"),
])
def test_name_exact_sesqui_di_terpene_parents(smi, expected):
    can = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    assert RAW.name(can) == expected
