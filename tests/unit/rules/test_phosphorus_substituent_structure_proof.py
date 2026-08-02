"""``_characterize_substituent`` must PROVE the alkyl shape, never COUNT it.

The audited defect shape: *a name component derived from a COUNT of atoms,
guarded by a list of shapes that must not be counted*. ``_characterize_substituent``
walked ONLY the carbon skeleton and returned ``get_alkyl_name(count)``, so every
fragment that is not an unbranched saturated acyclic chain attached at a terminus
was renamed as the straight chain of the same carbon count:

    CP(C)C(C)C        isopropyl  -> 'dimethyl(propyl)phosphane'          WRONG MOLECULE
    CP(C)CC(C)C       isobutyl   -> 'butyldi(methyl)phosphane'           WRONG MOLECULE
    CP(C)Cc1ccccc1    benzyl     -> 'heptyl(methyl)phosphanylmethane'    WRONG MOLECULE
    CP(C)C1CCCCC1     cyclohexyl -> 'cyclohexane'                        WRONG MOLECULE
    CP(C)CC=C         allyl      -> 'dimethyl(propyl)phosphane'          WRONG MOLECULE
    CP(C)CCCO         3-hydroxypropyl -> the -OH is not counted and is DROPPED

Each of those was produced and then suppressed by the SELF-01 OPSIN gate, so the
molecule abstained rather than shipping. The gate is the margin, not the producer:
``_final_opsin_validity_gate`` has ten ``return name`` carve-outs that never reach
a self-consistency decision, so a producer that emits a wrong molecule is a latent
ship. The fix is the shape that has worked at the five prior sites -- prove the
structure, and fail closed when the proof fails.

``substituent_purity._narrow_walk_name`` already carries exactly this proof and
already wraps this function; the raw call sites simply bypassed it.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym.rules.phosphorus import _characterize_substituent


def _char(smiles: str):
    """Characterise the substituent on the first P, as the rule module does."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    p = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "P")
    # the substituent under test is the LAST P-neighbour written in the SMILES
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(p).GetNeighbors()]
    return mol, p, nbrs


# --------------------------------------------------------------------------
# The proof must REFUSE every shape a carbon count cannot express.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "smiles,label,forbidden",
    [
        ("CP(C)C(C)C", "isopropyl (branched)", "propyl"),
        ("CP(C)CC(C)C", "isobutyl (branched)", "butyl"),
        ("CP(C)C(C)(C)C", "tert-butyl (branched)", "butyl"),
        ("CP(C)C1CCCCC1", "cyclohexyl (ring)", "hexyl"),
        ("CP(C)Cc1ccccc1", "benzyl (ring)", "heptyl"),
        ("CP(C)CC=C", "allyl (unsaturated)", "propyl"),
        ("CP(C)CC#C", "propargyl (unsaturated)", "propyl"),
        ("CP(C)CCCO", "3-hydroxypropyl (heteroatom dropped)", "propyl"),
        ("CP(C)CCOC", "2-methoxyethyl (heteroatom dropped)", "ethyl"),
    ],
)
def test_refuses_every_shape_a_count_cannot_express(smiles, label, forbidden):
    mol, p, nbrs = _char(smiles)
    results = [_characterize_substituent(mol, n, {p}) for n in nbrs]
    named = [r for r in results if r is not None]
    # The straight-chain word must never be produced for these fragments.
    assert all(r[1] != forbidden for r in named), (
        f"{label}: count-derived {forbidden!r} produced for {smiles} -> {named}"
    )


# --------------------------------------------------------------------------
# the contributor guide #9 -- the honest cases must KEEP working. A refusal that also
# refuses the shapes the count DOES determine is an over-correction.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CP(C)C", ("alkyl", "methyl")),
        ("CCP(C)C", ("alkyl", "ethyl")),
        ("CCCP(C)C", ("alkyl", "propyl")),
        ("CCCCP(C)C", ("alkyl", "butyl")),
        ("CCCCCP(C)C", ("alkyl", "pentyl")),
    ],
)
def test_unbranched_terminal_alkyl_still_named(smiles, expected):
    mol, p, nbrs = _char(smiles)
    results = [_characterize_substituent(mol, n, {p}) for n in nbrs]
    assert expected in results, f"{smiles}: expected {expected} in {results}"


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CP(C)c1ccccc1", ("aryl", "phenyl")),
        ("CP(C)c1ccc2ccccc2c1", ("aryl", "naphthyl")),
    ],
)
def test_aryl_path_unaffected(smiles, expected):
    """The aromatic branch is a lookup on ring size, not a chain count."""
    mol, p, nbrs = _char(smiles)
    results = [_characterize_substituent(mol, n, {p}) for n in nbrs]
    assert expected in results, f"{smiles}: expected {expected} in {results}"


# --------------------------------------------------------------------------
# End-to-end: the wrong MOLECULE must not be emitted, and the correct ones must.
# --------------------------------------------------------------------------
def test_emission_does_not_ship_a_count_derived_wrong_molecule():
    """The count-derived straight-chain names must no longer be produced.

    These are the names the carbon COUNT produced at this site. Under the test
    configuration the OPSIN gate is disabled, so this asserts on the PRODUCER
    rather than on what the gate lets through -- which is the point: in
    production every one of these was caught by SELF-01 as "different molecule",
    so the gate was the only thing standing between them and a shipped name.

    NOT asserted here: ``CP(C)C1CCCCC1`` -> ``'cyclohexane'`` (the phosphine is
    dropped entirely). A pinned A/B against 8dd2576f shows that name is emitted
    IDENTICALLY before and after this change, so it is a separate pre-existing
    defect on the ring path, not a count and not an unmasking. It is recorded in
    the audit report rather than silenced here.
    """
    from orthonym import name_compound

    for smiles, forbidden in [
        ("CP(C)C(C)C", "dimethyl(propyl)phosphane"),
        ("CP(C)CC(C)C", "butyldi(methyl)phosphane"),
        ("CP(C)Cc1ccccc1", "heptyl(methyl)phosphanylmethane"),
        ("CP(C)CC=C", "dimethyl(propyl)phosphane"),
        ("CP(C)CCOC", "ethyldi(methyl)phosphane"),
    ]:
        name = name_compound(smiles)
        assert name != forbidden, f"{smiles} shipped the wrong molecule {name!r}"


def test_emission_keeps_the_correct_names():
    from orthonym import name_compound

    assert name_compound("CP(C)C") == "trimethylphosphane"
    assert name_compound("CCCP(C)C") == "dimethyl(propyl)phosphane"
