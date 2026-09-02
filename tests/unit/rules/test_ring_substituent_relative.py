"""v41 M2.3 — descriptor-override parameter on ``decorated_ring_substituent_name``.

Task 2 adds an OPTIONAL ``relative_override`` parameter that, when set to
``'cis'`` / ``'trans'``, makes the ring-substituent renderer emit that
OPSIN-parseable RELATIVE prefix (e.g. ``trans-4-methylcyclohexyl``) in place of
the ring's OPSIN-ungrammatical PSEUDOASYMMETRIC ``(1r,4r)-`` CIP block. Task 3
will call it from the T4 best-effort floor with both senses and let round-trip
pick.

The BLOCKER-2 guard here is load-bearing: the default (no-override) path is
PIN-PATH-LIVE (``rules/amides.py`` / ``rules/benzene.py`` / ``rules/heterocycles.py``
call it) and pins the gold row ``[(1r,4r)-4-methylcyclohexyl]benzene`` (BB
48117 family, ``gold_pins.json`` def_id ``W4-S2``). If the override ever leaks
into the default path, that gold row moves — this test fails BEFORE it reaches
the gate.
"""
import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.perception.stereo import assign_stereochemistry
from orthonym.rules.ring_substituents import decorated_ring_substituent_name

# The gold witness: C[C@@H]1CC[C@H](CC1)c1ccccc1 ->
# '[(1r,4r)-4-methylcyclohexyl]benzene' (gold_pins.json def_id W4-S2).
GOLD_SMILES = "C[C@@H]1CC[C@H](CC1)c1ccccc1"
GOLD_FULL_NAME = "[(1r,4r)-4-methylcyclohexyl]benzene"
GOLD_SUBSTITUENT = "(1r,4r)-4-methylcyclohexyl"


def _cyclohexyl_call_args(smiles: str):
    """Locate the (mol, ring, attachment, expected_atoms) for the saturated
    all-carbon ring substituent in ``smiles`` — the exact tuple a production
    caller passes to ``decorated_ring_substituent_name``. Discovered
    structurally (no hard-coded indices) so the fixture cannot silently drift.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assign_stereochemistry(mol)  # canonical CIP path -> ring atoms get _CIPCode
    ri = mol.GetRingInfo()
    cyclo = None
    for ring in ri.AtomRings():
        if len(ring) == 6 and all(
            mol.GetAtomWithIdx(i).GetSymbol() == "C"
            and not mol.GetAtomWithIdx(i).GetIsAromatic()
            for i in ring
        ):
            cyclo = ring
            break
    assert cyclo is not None, "no saturated carbocyclic ring found"
    # attachment = ring carbon bonded to the (aromatic) parent fragment
    attachment = None
    for i in cyclo:
        for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
            if nbr.GetIdx() not in cyclo and nbr.GetIsAromatic():
                attachment = i
    assert attachment is not None
    # expected_atoms = ring + its own non-parent heavy decoration (the methyl)
    expected = set(cyclo)
    for i in cyclo:
        for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
            if (
                nbr.GetIdx() not in cyclo
                and not nbr.GetIsAromatic()
                and nbr.GetAtomicNum() > 1
            ):
                expected.add(nbr.GetIdx())
    return mol, cyclo, attachment, expected


# --------------------------------------------------------------------------- #
# Step 1 — BLOCKER-2 PIN-safety guard: the DEFAULT path is byte-identical.     #
# MUST pass before AND after the parameter is added.                          #
# --------------------------------------------------------------------------- #
def test_default_pipeline_pin_row_byte_identical():
    """The full name_compound pipeline still emits the pinned gold PIN."""
    assert name_compound(GOLD_SMILES) == GOLD_FULL_NAME


def test_default_decorated_ring_substituent_byte_identical():
    """The renderer with NO override is byte-identical for a cyclohexyl."""
    mol, ring, att, exp = _cyclohexyl_call_args(GOLD_SMILES)
    assert (
        decorated_ring_substituent_name(mol, ring, att, expected_atoms=exp)
        == GOLD_SUBSTITUENT
    )


def test_default_via_explicit_none_override():
    """Passing relative_override=None explicitly is the default path."""
    mol, ring, att, exp = _cyclohexyl_call_args(GOLD_SMILES)
    assert (
        decorated_ring_substituent_name(
            mol, ring, att, expected_atoms=exp, relative_override=None
        )
        == GOLD_SUBSTITUENT
    )


# --------------------------------------------------------------------------- #
# Step 2 — override path: emit the relative cis/trans WORD, suppress (1r,4r).  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sense", ["cis", "trans"])
def test_relative_override_emits_word_and_suppresses_pseudoasym(sense):
    mol, ring, att, exp = _cyclohexyl_call_args(GOLD_SMILES)
    out = decorated_ring_substituent_name(
        mol, ring, att, expected_atoms=exp, relative_override=sense
    )
    assert out == f"{sense}-4-methylcyclohexyl"
    # the ring's OWN pseudoasymmetric CIP block must be gone
    assert "1r,4r" not in out
    assert "(" not in out  # no ring CIP parenthesis at all
