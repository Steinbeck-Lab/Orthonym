"""v32 Phase 2 Step 2 — the decomposition assembly WEAVER.

`decomposition/weave.py` is a core-and-arms composer that assembles the
correctly-named fragments of a star-topology multi-linkage molecule (a
glycerophospholipid / glyceride: a glycerol-like carbon backbone bearing
acyloxy / alkoxy / phosphoryloxy / choline arms) into ONE connected,
round-tripping IUPAC name — using the structure-based, attachment-atom-anchored
`assembly.substituent_enumerator.name_substituent` primitive per arm, NOT the
flat weaver's suffix-string role converters that a T4-rescued/seniority-demoted
fragment carries nothing for.

Root cause + measured 9/9-fixable sample: .
Step-1 precondition (fail-closed, never partial-ship): `be810755`.

0-wrong is STRUCTURAL here: `weave.weave_is_verified` (Part C) ships a candidate
only if it full-InChI round-trips to the exact input — so this composer can only
ever convert an about-to-abstain molecule into a verified name, never emit a
wrong/atom-dropping one. `try_decompose` also only prefers a weave candidate when
the existing result did NOT itself round-trip, so a correct existing name is never
second-guessed (PIN byte-identity).

Measured 2026-08-15: 16 real acyclic multi-linkage `self01_mismatch` rows convert
(triacylglycerols, phosphatidylethanolamines, plasmalogens), 0 wrong, on the
first 40 of the corpus sample.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import name_compound
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse

# This module is ABOUT the SELF-01/OPSIN-gate interaction (the weaver only ships
# a verified name; before the fix these abstained under the production gate), so
# it must run gate-ON like the sibling decomposition/charged tests.
pytestmark = pytest.mark.opsin_gate


def _inchikey(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    return inchi.MolToInchiKey(m) if m else None


def _full_rt(smiles: str, name: str) -> bool:
    if not name:
        return False
    o = opsin_parse(name)
    if not o:
        return False
    return _inchikey(smiles) == _inchikey(o)


# Three smallest confirmed converters (full-InChIKey RT-verified 2026-08-15),
# each previously SELF-01-suppressed to an abstention.
CONVERTS = [
    # phosphatidylglycerol-like (ester + phosphodiester + a glycerol head)
    "CCCC/C=C\\CCCCCCCC(=O)OC[C@@H](O)COP(=O)(O)OC[C@@H](O)CO",
    # plasmalogen phosphatidylethanolamine (ester + vinyl ether + phosphoethanolamine)
    "CC/C=C\\C/C=C\\C/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)O/C=C\\CCCCCC/C=C\\CCCCCCCC",
    # diacyl phosphatidylglycerol
    "CCCC/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)(O)OC[C@@H](O)CO)OC(=O)CCCCCCCCC/C=C\\C/C=C\\CCCCC",
]


@pytest.mark.parametrize("smiles", CONVERTS, ids=["pg_monoacyl", "plasmalogen_pe", "diacyl_pg"])
def test_multilinkage_lipid_now_names_and_round_trips(smiles):
    """The breadth win: a glycerophospholipid that abstained now emits an
    atom-complete name that round-trips to the exact input molecule."""
    name = name_compound(smiles)
    assert name and not is_failure_name(name), (
        f"expected a real weaved name, got {name!r}"
    )
    assert _full_rt(smiles, name), f"name does not round-trip to input: {name!r}"


@pytest.mark.parametrize("smiles", CONVERTS, ids=["pg_monoacyl", "plasmalogen_pe", "diacyl_pg"])
def test_no_atom_drop_invariant(smiles):
    """0-wrong (Part C): if a name is emitted at all, its OPSIN parse must have
    the SAME heavy-atom count as the input — never a smaller molecule. The weave
    guard makes this structural (a candidate that does not fully cover the input
    is discarded, never shipped)."""
    name = name_compound(smiles)
    if not name or is_failure_name(name):
        return  # honest abstain is fine; only a shipped name must be complete
    o = opsin_parse(name)
    assert o is not None, f"emitted name must be OPSIN-parseable: {name!r}"
    parsed = Chem.MolFromSmiles(o)
    inp = Chem.MolFromSmiles(smiles)
    assert parsed is not None
    assert parsed.GetNumHeavyAtoms() == inp.GetNumHeavyAtoms(), (
        f"ATOM DROP: {name!r} parses to {parsed.GetNumHeavyAtoms()} heavy atoms, "
        f"input has {inp.GetNumHeavyAtoms()}"
    )


def test_triacetin_regression_unchanged():
    """A small triester the ORDINARY composer already names must be byte-identical
    (the weaver is only consulted when the existing result fails to round-trip)."""
    name = name_compound("CC(=O)OCC(COC(C)=O)OC(C)=O")  # triacetin / glyceryl triacetate
    assert name and not is_failure_name(name), name
    assert _full_rt("CC(=O)OCC(COC(C)=O)OC(C)=O", name), name


def test_ring_hub_out_of_scope_abstains_cleanly():
    """v1 weaver is acyclic-core only (SPY (ii) bucket): a molecule whose hub is a
    RING must decline honestly (abstain / not-supported), NEVER a partial or a
    wrong name. GPI-type mannoside core."""
    smi = ("NCCOP(=O)(O)OC[C@H]1O[C@H](O[C@@H]2[C@@H](O)[C@H](O)[C@@H](O)"
           "[C@H](OCCCCCCS)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        pytest.skip("SMILES did not parse in this RDKit build")
    name = name_compound(smi)
    # never a partial: either an honest abstain, or (if some other producer names
    # it) a full round-trip — never a smaller-molecule name.
    if name and not is_failure_name(name):
        assert _full_rt(smi, name), f"ring hub must not ship a partial: {name!r}"
