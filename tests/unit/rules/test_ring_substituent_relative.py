"""PIN byte-identity guard for ``decorated_ring_substituent_name``.

The default path of this renderer is PIN-PATH-LIVE (``rules/amides.py`` /
``rules/benzene.py`` / ``rules/heterocycles.py`` call it) and pins the gold row
``[(1r,4r)-4-methylcyclohexyl]benzene`` (BB 48117 family, ``gold_pins.json``
def_id ``W4-S2``): the ring's own pseudoasymmetric ``(1r,4r)-`` CIP block, keyed
to the substituent's numbering, must be emitted byte-identically. If any change
moves that block, the tests here fail BEFORE the change reaches the gate.

(This file formerly exercised a M2.3 ``relative_override`` parameter; the
cis/trans reclaim was implemented at ``assembly/universal_substituent.py``'s
``_stereo_prefix`` instead, so the parameter was removed as dead code. The
byte-identity guards below are kept — they are still load-bearing.)
"""
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
# BLOCKER-2 PIN-safety guard: the default renderer path is byte-identical. #
# --------------------------------------------------------------------------- #
def test_default_pipeline_pin_row_byte_identical():
    """The full name_compound pipeline still emits the pinned gold PIN."""
    assert name_compound(GOLD_SMILES) == GOLD_FULL_NAME


def test_default_decorated_ring_substituent_byte_identical():
    """The renderer is byte-identical for a pseudoasymmetric cyclohexyl."""
    mol, ring, att, exp = _cyclohexyl_call_args(GOLD_SMILES)
    assert (
        decorated_ring_substituent_name(mol, ring, att, expected_atoms=exp)
        == GOLD_SUBSTITUENT
    )
