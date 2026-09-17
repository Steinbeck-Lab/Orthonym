"""Reclaim P2-B2 — fused-component numbering for isoindole-spiro systems.

**Root cause (a DATA bug, not a chooser bug).** The abstention-reclaim Phase-2
plan (§B2) located the producer frame at ``rules/spiro.py::_orient_catalog_numbering``
and hypothesised the *chooser* was picking a topologically-impossible numbering
from among valid alternatives. Spying the actual flow refuted that: on an
asymmetric benzo-substituted isoindoline BOTH numberings the chooser enumerates
are invalid, because the raw catalog map it starts from is itself inconsistent.
The defect is upstream, in the catalog entry for ``2,3-dihydro-1H-isoindole``
(``src/orthonym/data/fused_heterocycles.py``, SMILES ``c1ccc2c(c1)CNC2``): its
``iupac_locants`` placed the two benzylic carbons **1 and 3 SWAPPED** relative to
their ring-fusion neighbours — loc 1 sat on the carbon bonded to the ``3a``
fusion atom and loc 3 on the carbon bonded to ``7a``, the mirror image of the
fixed isoindole numbering (loc 1 bonds 7a, loc 3 bonds 3a).

On a symmetric (unsubstituted) isoindoline the swap is invisible. On a
benzo-substituted or asymmetric-spiro isoindoline it puts the ring substituent
(and the oxo/spiro pair) on a **mirror-wrong position** — a different
constitution the shipped OPSIN round-trip + gate then correctly abstains
(returns the sentinel), which is why these 161 rows sat as `same_atoms_diff_skeleton`
abstentions. Correcting the catalog map to ``{... 6: 3, 7: 2, 8: 1}`` converts
abstain -> correct emission at zero precision cost.

Governing rules (heading + sentence, per the project's citation convention):

* ** "Numbering" item (a)** (``the Blue Book Blue Book``): *"when the
  numbering of a system is fixed, for example in purine, anthracene, and
  phenanthrene, this numbering must be used, both in PINs and in general
  nomenclature"*. Isoindole is a retained fused system with fixed numbering:
  C1 bonds C7a, C3 bonds C3a.
* ** item (3)** (``the Blue Book Blue Book``): *"if there is
  still a choice, low locants are selected considering all locants... as a
  set"* — the tie-break between the two mirror numberings once the fixed
  adjacency is respected.
* ** "Monospiro ring systems with different ring components"**
  (``the Blue Book Blue Book``): the spiro assembly of the two components.

0-wrong (Guard G4/G5): these rows abstain today, so every emission is net-new; a
wrong numbering cannot ship because the SAME shipped gate re-evaluates the
corrected name and rejects every wrong benzo position (``skel_eq=False``) at both
tiers. A row that still carries an *independent* second defect (e.g. an
adamantan-2-yl substituent the substituent namer cannot build) stays abstained —
never ships wrong.

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


def _name_in_fresh_subprocess(smiles):
    """Name ``smiles`` in a brand-new interpreter (no warm engine / cache), so a
    numbering that depended on atom order or a warm cache would show up as a
    disagreement across runs. Returns the emitted name string."""
    code = (
        "import sys;"
        "from orthonym import name_compound;"
        "sys.stdout.write(repr(name_compound(sys.argv[1])))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, smiles],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    # The child prints repr of a str or None; parse it safely (no eval).
    return ast.literal_eval(proc.stdout.strip())


# ---------------------------------------------------------------------------
# (a) Reclaim: benzo-substituted isoindole-spiro rows now full-round-trip.
# Real census rows (same_atoms_diff_skeleton), each carrying a substituent
# on a benzo-ring position (4/5/6) whose locant depends on the fixed
# isoindole numbering the catalog swap used to corrupt.
# ---------------------------------------------------------------------------
RECLAIM_ROWS = [
    # (cid, smiles) — expected name asserted only for the grounded case below.
    ("175867998", "C1CN(C(=O)[C@]12C3=C(C(=CC=C3)Cl)C(=O)N2)CC4=CC=C(C=C4)F"),
    ("121270277", "C1CC12C3=C(C=CC(=C3)Cl)C(=O)N2C4=CC(=CN=C4)I"),
    ("138613923", "C1CNCCC12C3=C(C=C(C=C3F)N4CCOCC4)C(=O)N2"),
    ("169137925", "CN1C=C(C=N1)CN2CCC3(CC2)C4=C(C=CC(=C4)Cl)C(=O)N3"),
    ("177071379", "C[C@H]1CC2(C[C@H](N1)C3=CN(N=N3)C)C4=C(C=C(C=C4)Cl)C(=O)N2"),
    ("135977806", "C1CC2(C1)C3=C(C(=CC(=C3)OC4=CC=CC=C4)O)C(=N)N2O"),
]


@pytest.mark.parametrize("cid,smiles", RECLAIM_ROWS, ids=[r[0] for r in RECLAIM_ROWS])
def test_isoindole_spiro_reclaims_and_full_roundtrips(cid, smiles):
    name = name_compound(smiles)
    assert not is_failure_name(name), f"CID {cid} still abstains: {name!r}"
    assert _roundtrips_to(name, smiles), (
        f"CID {cid} name {name!r} does not round-trip to the input structure "
        f"(opsin -> {opsin_parse(name)!r})"
    )


# The directly-grounded numbering case: CID 143635820's core with its
# adamantan-2-yl N-substituent replaced by methyl (which removes the independent
# second defect). Byte-identity locks the corrected fixed numbering: the Cl lands
# at 4 and the gem-dimethyl at 1 (Option A: spiro at 3), NOT the pre-fix
# mirror-wrong "4-chloro-3,3-dimethyl...-1,4'-piperidine" (a different molecule).
GROUNDED_SMILES = "CC1(C2=C(C(=CC=C2)Cl)C3(N1C)CCNCC3)C"
GROUNDED_NAME = "4-chloro-1,1,2-trimethylspiro[2,3-dihydro-1H-isoindole-3,4'-piperidine]"


def test_grounded_numbering_byte_identity():
    got = name_compound(GROUNDED_SMILES)
    assert got == GROUNDED_NAME, got
    assert _roundtrips_to(got, GROUNDED_SMILES), got


# ---------------------------------------------------------------------------
# (b) Determinism: the corrected numbering is atom-order-invariant. Name in a
# fresh subprocess three times and require an identical name each time.
# ---------------------------------------------------------------------------
DETERMINISM_ROWS = [
    "C1CNCCC12C3=C(C=C(C=C3F)N4CCOCC4)C(=O)N2",             # 138613923
    "CN1C=C(C=N1)CN2CCC3(CC2)C4=C(C=CC(=C4)Cl)C(=O)N3",      # 169137925
    GROUNDED_SMILES,
]


@pytest.mark.parametrize("smiles", DETERMINISM_ROWS)
def test_numbering_is_deterministic_across_fresh_processes(smiles):
    names = {_name_in_fresh_subprocess(smiles) for _ in range(3)}
    assert len(names) == 1, f"non-deterministic name for {smiles}: {names}"
    (only,) = names
    assert not is_failure_name(only), only
    assert _roundtrips_to(only, smiles), only


# ---------------------------------------------------------------------------
# (c) 0-wrong: rows that still carry an independent second defect (or a wholly
# different mechanism) must ABSTAIN — never ship a wrong isomer/constitution.
# * CID 143635820: core numbering is now correct, but the adamantan-2-yl
# N-substituent cannot be built -> Guard G5 abstain.
# * CID 11172275: a benzo[b]pyrazine-spiro. These 23 rows never touch the
# B2 catalog path (match_fused_heterocycle_core returns None); they fail
# on a systematic indicated-H/imine mechanism (B1 territory) — the B2 fix
# does not and must not silently mis-name them.
# ---------------------------------------------------------------------------
ZERO_WRONG_ROWS = [
    ("143635820", "CC1(C2=C(C(=CC=C2)Cl)C3(N1C4C5CC6CC(C5)CC4C6)CCNCC3)C"),
    ("11172275", "COC1=CC2=C(C=C1)N=CC3(N2)CCCCC3"),
    ("129071676",
     "COC1=CC=C(C=C1)CN2C(=O)C3=C(C24CCCCC4)C=C(C=C3C(C(=O)C5=CC=CC=C5)(F)F)NC(=O)O"),
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
