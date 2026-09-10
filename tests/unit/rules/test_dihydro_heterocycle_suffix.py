""" breadth — partially-saturated (dihydro/tetrahydro) monocyclic heterocycle
bearing a principal-group SUFFIX.

Root cause (trace-verified + BB-derived): the parent stem string (saturation prefix +
indicated-H) was numbered PCG-BLIND, so its dihydro locants disagreed with the
suffix locant the substituted path appends — `3,6-dihydro-2H-1,4-thiazine` +
`-3-carboxylic acid` instead of the correct `5,6-dihydro-2H-1,4-thiazine-3-...`.
 caught the mismatch and abstained (0-wrong held), so these molecules named
nothing.

Fix: add a principal-characteristic-group locant term to the two heterocyclic
hydro-name cascade keys, AFTER indicated hydrogen (b) and BEFORE hydro prefixes (e)
per (the Blue Book): (a) fixed -> (b) indicated-H -> (c) PCG/suffix ->
(e) hydro. Mirrors the carbocyclic sibling (partial_saturation.py:546). Additive:
`principal_group_atoms=None` (bare rings / substituents) leaves the key unchanged.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.unit


def _pin():
    return Orthonym(style="pin")


@pytest.mark.parametrize("smi,expected", [
    # _mancude_hydro_name path (parents needing indicated hydrogen)
    ("O=C(O)C1=NCCSC1", "5,6-dihydro-2H-1,4-thiazine-3-carboxylic acid"),
    ("O=C(O)C1=NCCOC1", "5,6-dihydro-2H-1,4-oxazine-3-carboxylic acid"),
    # _aromatizable_hydro_name path (pyridine-type)
    ("O=C(O)C1=CCCCN1", "1,4,5,6-tetrahydropyridine-2-carboxylic acid"),
    ("O=C(O)C1=NCCCC1", "3,4,5,6-tetrahydropyridine-2-carboxylic acid"),
])
def test_dihydro_heterocycle_carboxylic_acid(smi, expected):
    assert _pin().name(smi) == expected


def test_dihydro_heterocycle_suffix_rt_exact():
    """Every emitted name round-trips (0-wrong): the fix converts abstentions to
    correct names, never to a wrong molecule."""
    import sys
    sys.path.insert(0, "scripts")
    from diagnose import diagnose
    smis = ["O=C(O)C1=NCCSC1", "O=C(O)C1=NCCOC1",
            "O=C(O)C1=CCCCN1", "O=C(O)C1=NCCCC1"]
    rows = diagnose(smis, style="pin", use_opsin=True)
    assert all(r.get("verdict") == "OK" for r in rows), [
        (r["smiles"], r.get("verdict"), r.get("name")) for r in rows]


# ---- RISK 2: indicated-H OUTRANKS the suffix (b), verbatim
# `2H-pyran-6-carboxylic acid (PIN)` at the Blue Book) ----
# The mancude parent's indicated hydrogen takes the LOW locant (2H) and FORCES the
# suffix to the higher locant (6), NOT `2H-pyran-2-...`. Before the stem and suffix
# numbering were unified onto ONE authority (`_mancude_hydro_numbering`), the stem
# correctly said 2H while `orient_heterocycle_with_substituents` independently
# numbered PCG-first (locant 2) -> a self-contradictory name -> abstained.
# The suffix is read from a single [hetero, indicated_hydrogen, principal,...]
# keyed map; Orthonym now does the same.

@pytest.mark.parametrize("smi,expected", [
    ("O=C(O)C1=CCCCO1", "3,4-dihydro-2H-pyran-6-carboxylic acid"),
])
def test_indicated_h_outranks_suffix(smi, expected):
    assert _pin().name(smi) == expected


def test_indicated_h_outranks_suffix_rt_exact():
    """0-wrong: the IH-outranks-suffix class emits a correct name, never a wrong
    molecule (the abstention it replaces was catching the collision)."""
    import sys
    sys.path.insert(0, "scripts")
    from diagnose import diagnose
    rows = diagnose(["O=C(O)C1=CCCCO1"], style="pin", use_opsin=True)
    assert all(r.get("verdict") == "OK" for r in rows), [
        (r["smiles"], r.get("verdict"), r.get("name")) for r in rows]


# ---- RISK 2 sibling: the FULLY-MANCUDE indicated-H parent + suffix ----
# The verbatim BB (PIN) example `2H-pyran-6-carboxylic acid` (:3252) is the mancude
# parent itself (d == max_match), which the hydro namer declines. Its suffix was
# numbered PCG-first too -> collided with the stem's `2H`. `_mancude_parent_suffix_
# numbering` unifies it. Aromatic rings (pyridine/furan/thiophene) bail to None and
# are numbered by the heteroatom cascade exactly as before.

@pytest.mark.parametrize("smi,expected", [
    ("O=C(O)C1=CC=CCO1", "2H-pyran-6-carboxylic acid"),        # verbatim BB:3252
    ("O=C(O)C1=CC=CCS1", "2H-thiopyran-6-carboxylic acid"),
    ("O=C(O)C1=CN=CCO1", "2H-1,4-oxazine-6-carboxylic acid"),
])
def test_mancude_parent_indicated_h_suffix(smi, expected):
    assert _pin().name(smi) == expected


@pytest.mark.parametrize("smi,expected", [
    # already-OK mancude parents: must stay byte-identical (numbering agrees)
    ("O=C(O)C1=COC=CC1", "4H-pyran-3-carboxylic acid"),
    ("O=C(O)C1=CSC=CC1", "4H-thiopyran-3-carboxylic acid"),
    ("O=C(O)C1=NC=CCO1", "6H-1,3-oxazine-2-carboxylic acid"),
    ("O=C(O)C1=NC=CCS1", "6H-1,3-thiazine-2-carboxylic acid"),
    # aromatic controls: bail to None -> heteroatom cascade unchanged
    ("O=C(O)c1ccccn1", "pyridine-2-carboxylic acid"),
    ("O=C(O)c1ccco1", "furan-2-carboxylic acid"),
    ("O=C(O)c1ccsc1", "thiophene-3-carboxylic acid"),
])
def test_mancude_parent_no_regression(smi, expected):
    assert _pin().name(smi) == expected


@pytest.mark.parametrize("smi,expected", [
    # bare partially-saturated rings must stay byte-identical (PCG term is empty)
    ("C1=NCCSC1", "3,6-dihydro-2H-1,4-thiazine"),
    ("C1=NCCOC1", "3,6-dihydro-2H-1,4-oxazine"),
    ("C1C=CC=CN1", "1,2-dihydropyridine"),
    # saturated rings + their suffixes are on a different path, unaffected
    ("C1CSCCN1", "thiomorpholine"),
    ("O=C(O)C1CSCCN1", "thiomorpholine-3-carboxylic acid"),
    ("c1ccncc1", "pyridine"),
])
def test_no_regression_bare_and_saturated_rings(smi, expected):
    assert _pin().name(smi) == expected


# ---- a review BLOCKER 1/2: decorated symmetric rings need (f) + canon tiers ----
# When (het, indicated-H, suffix, hydro) TIE (a symmetric ring bearing a plain
# substituent), the unified numbering must still minimise the substituent locant
# (f) `:3300`) and break the residual automorphism by an input-invariant
# canonical rank. Without those tiers the map fell to ring-atom iteration order:
# `5-methyl-1,4-dihydropyridine` (non-PIN) and a representation-dependent name.

@pytest.mark.parametrize("smi,expected", [
    ("CC1=CNC=CC1", "3-methyl-1,4-dihydropyridine"),   # nifedipine/Hantzsch scaffold
    ("CC1COC=CO1", "2-methyl-2,3-dihydro-1,4-dioxine"),
])
def test_decorated_tie_ring_uses_lowest_substituent_locant(smi, expected):
    assert _pin().name(smi) == expected


def test_decorated_tie_ring_representation_stable():
    """Same molecule (InChIKey VZJRBSCLGISQQW), three SMILES spellings -> ONE name.
    The canonical tie-break must be input-order-invariant (a review BLOCKER 2)."""
    p = _pin()
    names = {p.name(s) for s in ("CC1COC=CO1", "CC1OC=COC1", "O1C=COCC1C")}
    assert names == {"2-methyl-2,3-dihydro-1,4-dioxine"}, names


# ---- a review-dihydro BLOCKER: -ol/-amine ring-atom suffix over-inclusion (fixed) ----

@pytest.mark.parametrize("smi,expected", [
    # For -ol/-amine the pg tuple carries the RING bearing-carbon; the neighbor
    # clause must NOT leak onto its ring neighbours (which flipped same-element-
    # adjacent rings pyridazine/1,2-dithiine, regressing them to abstain and, gate
    # off, to a wrong enol/enamine isomer). The exocyclic-only neighbor clause keeps
    # these correct.
    ("NC1CC=CNN1", "1,2,3,4-tetrahydropyridazin-3-amine"),
    ("OC1CC=CNN1", "1,2,3,4-tetrahydropyridazin-3-ol"),
    ("OC1CC=CSS1", "3,4-dihydro-1,2-dithiin-3-ol"),
    ("NC1CC=CSS1", "3,4-dihydro-1,2-dithiin-3-amine"),
])
def test_ol_amine_ring_suffix_not_over_included(smi, expected):
    assert _pin().name(smi) == expected
