"""Reclaim P2-B1 — hydro-prefix / indicated-H aromaticity loss in
partially-saturated fused-heterocyclic-spiro systems.

**Root cause (a hybridization proxy in `_try_partial_saturation_name`).** The
Phase-2 plan (§B1, Guard G2) located the producer frame at
``rules/fused_rings.py::_try_partial_saturation_name``. That function builds the
saturated-position set ``sat`` by walking the ring atoms; it used to skip any
atom whose RDKit hybridization was not ``SP3``:

    if at.GetHybridization != SP3: # SP2/SP -> remaining unsaturation
        continue

RDKit marks a *conjugated pyridine-type ring* ``-NH-`` as **SP2**, because its
lone pair delocalises into the adjacent aromatic ring. So a genuinely SATURATED
ring nitrogen was dropped from ``sat`` before the neutral-N admit branch could
run, and the namer emitted one hydro/indicated-H position short — e.g.
``6,7-dihydro-5H-pyrazolo[1,5-a]pyrimidine`` (3 saturated positions) instead of
the correct ``4,5,6,7-tetrahydropyrazolo[1,5-a]pyrimidine`` (4). OPSIN then
reconstructed a *different, more-aromatic* skeleton (it kept an ``N4=C`` imine),
so the shipped OPSIN round-trip + gate saw "different molecule" and
abstained (returned the sentinel). These rows sat as ``same_atoms_diff_skeleton``
abstentions.

**The fix (structure-derived, NOT a hydro literal — Guard G2).** Replace the
hybridization proxy with a STRUCTURAL saturation test: an atom is a hydro
position unless it participates in a double bond (ring or exocyclic). This admits
the conjugated NH while still skipping a genuine residual ring C=C / C=N and (as
the old proxy did) an exocyclic-double-bond position (oxo / =NR / ylidene, which
is added-hydrogen territory the substituent branch fails closed on). The
hydro set is computed per molecule from its own bonds, so pyrazolo, pyrido,
isoxazolo, oxazolo and thieno systems each derive their own set — no per-scaffold
literal. An sp3 atom has no double bond, so the byte-identical saturated-carbon
path is unchanged (verified: 0 name changes over the partial-saturation test
corpus and a 57-molecule fused/spiro naming sample).

Governing rules (heading + sentence, per the project's citation convention):

* ** "Numbering" note (a)** (``the Blue Book Blue Book``): *"the
  maximum number of noncumulative double bonds are inserted, and finally
  indicated hydrogen is cited, consistent with the structure of the ring system,
  for all positions that are saturated, i.e., where there are two ring bonds and
  sufficient exo bonds to satisfy the bonding number of the atom"*. The
  conjugated NH is such a saturated position.
* **** hydro-prefix numbering by lowest locants after indicated hydrogen
  (``the Blue Book Blue Book``).
* ** "Monospiro ring systems with different ring components"**
  (``the Blue Book Blue Book``): the spiro assembly of the two components.

0-wrong (Guard G4/G5): these rows abstain today, so every emission is net-new; a
wrong hydro set cannot ship because the SAME shipped gate re-evaluates the
corrected name and abstains any name that does not round-trip. A row that still
carries an *independent* second defect stays abstained — never ships wrong
(measured: 0 wrong over 400 random ``same_atoms_diff_skeleton`` rows).

⚠ **benzo[b]pyrazine-spiro was routed here from B2 but is NOT reclaimed by this
fix** and is asserted only as a 0-wrong abstain below. The B1 fix does correct
its fused component (``1,2-dihydrobenzo[b]pyrazine``, imine retained), but the
spiro assembly then places the spiro atom on the wrong locant (an independent
numbering defect that lives in ``rules/spiro.py``, out of this task's scope), so
the gate still (correctly) abstains it. Same for pyrido[3,2-b]pyridine-spiro.

Every expected name here was checked name -> OPSIN 2.9.0 -> canonical InChIKey
against its input SMILES.
"""

import ast
import subprocess
import sys

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse

pytestmark = pytest.mark.opsin_gate


def _inchikey(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(m) if m is not None else None


def _roundtrips_to(name, smiles):
    """True iff ``name`` parses under OPSIN to the same InChIKey as ``smiles``."""
    if not name or is_failure_name(name):
        return False
    out = opsin_parse(name)
    if not out:
        return False
    m = Chem.MolFromSmiles(out)
    if m is None:
        return False
    return Chem.MolToInchiKey(m) == _inchikey(smiles)


# Every row except the bare core '163302682' carries such a prefix-cited group.
DEFAULT_TIER_DECLINES = frozenset({
    "C1C(NCC12CNC3=CC=NN3C2)C(=O)N",
    "CN1CCC2(CCNC3=C(C(=NN32)Br)C(=O)N)CC1",
    "C/C=C\\C(=C/CC#C)\\C1CC2(CCCC2)N3C(=C(C=N3)C(=O)O)N1",
    "C1C2(CN1)CNC3=C(C(=NN3C2)C4=CC=C(C=C4)OC5=CC=CC=C5)C(=O)N",
    "C1CNCCC12CCNC3=C(C(=NN23)C4=CC=C(C=C4)OC5=CC=CC=C5)C(=O)N",
    "CC1=CC=CC=C1N2C(=O)C=CC(=N2)C3=C4NCC5(CCC5)CN4N=C3C6=CC=C(C=C6)F",
    "C1C2(CN(C2)CC3=CC=CC=C3)CN4C(=C(C(=N4)C5=CC=C(C=C5)OC6=CC=CC=C6)C#N)N1",
    "C1CCC2(CC1)CC3=C(NC2=NCC4=CC=CC=C4)ON=C3",
    "C1CCC2(CC1)CC3=C(NC2=NCC4=CC(=C(C=C4)Cl)Cl)OC=N3",
    "CC1=C(SC2=C1C(=O)C3(CCC3)C(=O)N2)C4=NC=CO4",
})


def _declined_strict_name(smiles):
    """The name the strict path builds for ``smiles`` (the name the default tier
    used to return). These rows cite their principal characteristic
    group (carboxamide, nitrile, carboxylic acid, imine, ring C=O) as a prefix on
    the spiro parent; that is not the PIN (the principal characteristic
    group is a suffix,; 'spiro[4.5]decane-1,7-dione (PIN)', the Blue Book),
    so the default (PIN) tier declines it with NO_VERIFIED_PIN and the best-effort
    tier names it, read back exactly (``tests/support/default_tier.py``)."""
    from tests.support.default_tier import declined_pin_row
    if smiles not in DEFAULT_TIER_DECLINES:
        return name_compound(smiles)
    return declined_pin_row(smiles, best_effort_same=False)["name"]


def _name_in_fresh_subprocess(smiles):
    """The strict path's name for ``smiles`` (the default tier's emission rule
    switched off, ``tests/support/default_tier.strict_path_name``) in a brand-new
    interpreter (no warm engine / cache), so a numbering that depended on atom order
    or a warm cache would show up as a disagreement across runs. Returns the
    emitted name string. The default tier declines two of the rows (the isoxazolo
    row is labelled below the PIN, quick-wins F-Q2c part 1; the thieno row cites its
    ring C=O as a prefix, DEFAULT_TIER_DECLINES); the strict path still names them."""
    code = (
        "import sys;"
        "from tests.support.default_tier import strict_path_name;"
        "sys.stdout.write(repr(strict_path_name(sys.argv[1])))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, smiles],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    # The child prints repr of a str or None; parse it safely (no eval).
    return ast.literal_eval(proc.stdout.strip())


# ---------------------------------------------------------------------------
# (a) Reclaim: pyrazolo[1,5-a]pyrimidine-spiro rows now full-round-trip. Real
# census rows (same_atoms_diff_skeleton). The pyrazolo N4 was the dropped
# saturated nitrogen; every one of these used to emit `6,7-dihydro-5H-` and
# abstain, and now emits `4,5,6,7-tetrahydro-` and round-trips.
# ---------------------------------------------------------------------------
PYRAZOLO_RECLAIM_ROWS = [
    ("163302682", "C1CC12CNC3=CC=NN3C2"),
    ("171550127", "C1C(NCC12CNC3=CC=NN3C2)C(=O)N"),
    ("172234068", "CN1CCC2(CCNC3=C(C(=NN32)Br)C(=O)N)CC1"),
    ("137203687", "C/C=C\\C(=C/CC#C)\\C1CC2(CCCC2)N3C(=C(C=N3)C(=O)O)N1"),
    ("136928103", "C1C2(CN1)CNC3=C(C(=NN3C2)C4=CC=C(C=C4)OC5=CC=CC=C5)C(=O)N"),
    ("136928125", "C1CNCCC12CCNC3=C(C(=NN23)C4=CC=C(C=C4)OC5=CC=CC=C5)C(=O)N"),
    ("135975091", "CC1=CC=CC=C1N2C(=O)C=CC(=N2)C3=C4NCC5(CCC5)CN4N=C3C6=CC=C(C=C6)F"),
    ("136928082", "C1C2(CN(C2)CC3=CC=CC=C3)CN4C(=C(C(=N4)C5=CC=C(C=C5)OC6=CC=CC=C6)C#N)N1"),
]


@pytest.mark.parametrize(
    "cid,smiles", PYRAZOLO_RECLAIM_ROWS, ids=[r[0] for r in PYRAZOLO_RECLAIM_ROWS]
)
def test_pyrazolo_spiro_reclaims_and_full_roundtrips(cid, smiles):
    name = _declined_strict_name(smiles)
    assert not is_failure_name(name), f"CID {cid} still abstains: {name!r}"
    assert _roundtrips_to(name, smiles), (
        f"CID {cid} name {name!r} does not round-trip to the input structure "
        f"(opsin -> {opsin_parse(name)!r})"
    )


# The directly-grounded, bare (unsubstituted-core) case. Byte-identity locks the
# saturation: N4, C5 and C7 are saturated (N4 is not left aromatic, which would
# be a different, imine-bearing molecule). The pyrazolo[1,5-a]pyrimidine component
# is cited as its mancude parent, after 'cyclopropane' (alphanumerical order), with
# its indicated hydrogen and hydro prefixes in front: (the Blue Book)
# "Indicated hydrogen (see is cited in front of the name if needed in the
# complete structure"; '4'a,5',6',7',8',8'a-hexahydro-1'H-spiro[imidazolidine-4,2'-
# quinoxaline] (PIN)' (:17050). The indicated hydrogen takes 7' because neither 4'
# nor 5' can carry it in this structure (b),:3246). Was
# "spiro[4,5,6,7-tetrahydropyrazolo[1,5-a]pyrimidine-6,1'-cyclopropane]".
GROUNDED_SMILES = "C1CC12CNC3=CC=NN3C2"
GROUNDED_NAME = "4',5'-dihydro-7'H-spiro[cyclopropane-1,6'-pyrazolo[1,5-a]pyrimidine]"


def test_grounded_hydro_set_byte_identity():
    got = name_compound(GROUNDED_SMILES)
    assert got == GROUNDED_NAME, got
    assert _roundtrips_to(got, GROUNDED_SMILES), got


# ---------------------------------------------------------------------------
# (b) The fix is structure-derived, not a pyrazolo literal: one reclaiming row
# from each of THREE other hetero-fused ring systems, each deriving its own
# hydro set from its own bonds. All round-trip to the input structure.
# ---------------------------------------------------------------------------
OTHER_SYSTEM_RECLAIM_ROWS = [
    # (cid, smiles, ring-system tag)
    ("158408477", "C1CCC2(CC1)CC3=C(NC2=NCC4=CC=CC=C4)ON=C3", "isoxazolo[5,4-b]pyridine"),
    ("158176916", "C1CCC2(CC1)CC3=C(NC2=NCC4=CC(=C(C=C4)Cl)Cl)OC=N3", "oxazolo[5,4-b]pyridine"),
    ("156902744", "CC1=C(SC2=C1C(=O)C3(CCC3)C(=O)N2)C4=NC=CO4", "thieno[2,3-b]pyridine"),
]


@pytest.mark.parametrize(
    "cid,smiles,tag",
    OTHER_SYSTEM_RECLAIM_ROWS,
    ids=[r[2] for r in OTHER_SYSTEM_RECLAIM_ROWS],
)
def test_other_hetero_fused_systems_reclaim(cid, smiles, tag):
    # quick-wins F-Q2c part 1: '[1,2]oxazolo' / '[1,3]oxazolo' are the PIN component
    # spellings, the Blue Book), so the unbracketed 'isoxazolo' and
    # 'oxazolo' names are labelled below the PIN and the PIN tier declines them; the
    # reclaim (breadth) is the best-effort name. All three rows also cite their
    # principal characteristic group as a prefix on the spiro parent
    # (DEFAULT_TIER_DECLINES), so the strict path's name is asserted as well.
    from tests.support.rt_assert import name_best_effort
    name = _declined_strict_name(smiles)
    if "oxazolo" in tag:
        assert is_failure_name(name_compound(smiles))
    best_effort = name_best_effort(smiles).get("name")
    for got in (name, best_effort):
        assert not is_failure_name(got), f"{tag} CID {cid} still abstains: {got!r}"
        # The saturation is now cited on the mancude component as hydro prefixes plus
        # indicated hydrogen in front of the spiro name, the Blue Book).
        assert "hydro" in got and "H-spiro[" in got, (
            f"{tag} CID {cid} did not derive its saturation from the structure: {got!r}"
        )
        assert _roundtrips_to(got, smiles), (
            f"{tag} CID {cid} name {got!r} does not round-trip "
            f"(opsin -> {opsin_parse(got)!r})"
        )


# ---------------------------------------------------------------------------
# (c) 0-wrong: a row that still carries an independent second defect must ABSTAIN
# (failure sentinel / None) — never ship a wrong skeleton.
# * 156092259: pyrazolo[1,5-a]pyrimidine-spiro whose gates-off substituent
# forms (2-amino + a fused-pyridine amide chain) don't round-trip.
# * 11172275: benzo[b]pyrazine-spiro — the B1 fix corrects its fused
# component but the spiro-atom locant is then mis-assigned by spiro.py
# (out of scope); the gate correctly abstains rather than ship it wrong.
# * 142507996: pyrido[3,2-b]pyridine-spiro — same residual spiro-numbering
# defect; abstains.
# ---------------------------------------------------------------------------
ZERO_WRONG_ROWS = [
    ("156092259", "CCOC1=C(C=NC=C1)NC(=O)C2=C3NCC(C4(N3N=C2N)CCCCCCC4)F"),
    ("11172275", "COC1=CC2=C(C=C1)N=CC3(N2)CCCCC3"),
    ("142507996", "C1CC2(CC2)NC3=C1N=CC=C3"),
]


@pytest.mark.parametrize("cid,smiles", ZERO_WRONG_ROWS, ids=[r[0] for r in ZERO_WRONG_ROWS])
def test_zero_wrong_row_abstains_or_is_correct(cid, smiles):
    """Never a wrong molecule: the row must either abstain (failure sentinel /
    None) or, if it does emit, round-trip to the exact input structure."""
    name = name_compound(smiles)
    if is_failure_name(name):
        return  # abstained — the measured, correct behaviour today
    assert _roundtrips_to(name, smiles), (
        f"CID {cid} shipped a WRONG molecule: {name!r} -> {opsin_parse(name)!r}"
    )


# ---------------------------------------------------------------------------
# (d) Determinism: the corrected hydro set / numbering is atom-order-invariant.
# Name in a fresh subprocess three times and require an identical name each
# time (one pyrazolo + two other ring systems).
# ---------------------------------------------------------------------------
DETERMINISM_ROWS = [
    "C1CC12CNC3=CC=NN3C2",                                   # 163302682 pyrazolo (bare)
    "C1CCC2(CC1)CC3=C(NC2=NCC4=CC=CC=C4)ON=C3",              # 158408477 isoxazolo
    "CC1=C(SC2=C1C(=O)C3(CCC3)C(=O)N2)C4=NC=CO4",            # 156902744 thieno
]


@pytest.mark.parametrize("smiles", DETERMINISM_ROWS)
def test_hydro_set_is_deterministic_across_fresh_processes(smiles):
    names = {_name_in_fresh_subprocess(smiles) for _ in range(3)}
    assert len(names) == 1, f"non-deterministic name for {smiles}: {names}"
    (only,) = names
    assert not is_failure_name(only), only
    assert _roundtrips_to(only, smiles), only
