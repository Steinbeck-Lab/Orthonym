"""v29 Task S2 — the von Baeyer legality proof, wired into the paths that SPELL.

Task S built a correct Java-free legality audit
(``vonbaeyer_universal.audit_von_baeyer_descriptor``) and left it consulted by
exactly one opt-in path. Meanwhile ``VonBaeyerAnalyzer`` could return — and the
PIN path could spell — a descriptor whose brackets do not account for every
skeletal atom, because:

* the atom-count invariant was ``logger.error``-ed inside ``_analyze_impl`` and
  the descriptor returned anyway, and
* ``analyze``'s fallback branch ran no validity check at all, while its own
  docstring claimed the fallback "is never worse than the un-renumbered path".

``analyze`` now adjudicates every return path with ``_legality_verified`` and
publishes the verdict as ``PolycyclicDescriptor.legality``; the three
name-producers in this module refuse anything but ``True``.

Measured before wiring (3 corpora, 7,415 unique SMILES, 2,131 cages reaching
``analyze`` through the live path's own ring-atom computation): 63 cages carried
a descriptor that does not rebuild their cage, 48 of them non-aromatic, and
**0** cages had the string audit pass where the arithmetic check failed — so
composing the two adds no false rejection. The same held over 8,201 enumerated
cages.

No OPSIN here. The whole point of this floor is that both downstream gates are
documented FAIL-OPEN without Java, so a Java-free refusal is the only thing
standing between a bridge-dropping descriptor and an emitted name.
"""
import pytest
from rdkit import Chem

from orthonym.errors import OrthonymLimitError
from orthonym.rules.polycyclic import (
    VonBaeyerAnalyzer,
    generate_polycyclic_name,
    name_polycyclic_complete,
    name_polycyclic_with_heteroatoms,
)

# A cage whose emitted descriptor does NOT rebuild the cage bond set, so the
# legality audit must reject it and every producer must refuse. From the
# systematic N<=12 enumeration
# (.planning/audit-v29/TaskS-vonbaeyer-legality.md). The specific descriptor
# string, and whether it fails on the atom-count arithmetic or on the bond-set
# edge-audit, are implementation details the v41 P-23.2.4 main-bridge selection
# legitimately changed (it now emits ``tetracyclo[4.2.2.1^1,9.1^3,5]``, whose
# bracket sum matches the atom count but whose bonds still do not rebuild the
# cage) -- so this pins only the DURABLE invariant: legality=False + refusal.
#
# (The 10-atom cage ``C1CC23CC(C2)C12CC3C2`` used to sit here too; the same
# main-bridge fix turns it into a LEGAL, OPSIN-round-tripping name -- a breadth
# gain, pinned by test_main_bridge_fix_rescues_a_previously_refused_cage below.)
ILLEGAL_CAGES = [
    ("C1C2CC1C1CCC3(C2)CC1C3", 12),
]

# Cages whose descriptor does rebuild them. These must keep naming exactly as
# before — the failure mode a legality gate introduces is false rejection.
LEGAL_CAGES = [
    ("C1C2CC3CC1CC(C2)C3", "tricyclo[3.3.1.1^3,7]"),   # adamantane
    ("C1CC2CC3CCC2CC13", "tricyclo[4.4.0.0^3,8]"),     # twistane
    ("C1C2CC3C1C3C2", "tricyclo[2.2.1.0^2,6]"),        # nortricyclene
    ("C1CC2CCC1CC2", "bicyclo[2.2.2]"),
]


def _cage(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    return mol, ring


# --------------------------------------------------------------------------
# `analyze` adjudicates -- it no longer returns an unlabelled descriptor
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", LEGAL_CAGES)
def test_analyze_marks_a_rebuilding_descriptor_legal(smiles, expected):
    mol, ring = _cage(smiles)
    desc = VonBaeyerAnalyzer().analyze(mol, ring)
    assert desc.descriptor_string == expected
    assert desc.legality is True


@pytest.mark.parametrize("smiles,n_cage", ILLEGAL_CAGES)
def test_analyze_marks_an_unrebuildable_descriptor_illegal(smiles, n_cage):
    mol, ring = _cage(smiles)
    desc = VonBaeyerAnalyzer().analyze(mol, ring)
    assert len(ring) == n_cage
    assert desc is not None
    # The descriptor+numbering does not rebuild the cage bond set, so the
    # Java-free legality audit refuses to certify it -- whether it trips on the
    # P-23.2.6.1.4 atom-count arithmetic (BlueBookV2.md:9651) or on the
    # bond-set edge-audit. Either way nothing correct can be spelled from it.
    assert desc.legality is False


def test_main_bridge_fix_rescues_a_previously_refused_cage():
    """v41 M4 subpart #1 side effect (verified 0-wrong improvement): the P-23.2.4
    largest-main-bridge selection turns a cage HEAD refused -- its shortest-bridge
    descriptor ``tetracyclo[3.1.1.2^1,4]`` dropped an atom (9 bracketed, 10 in the
    cage) -- into one that names with a LEGAL descriptor. The name OPSIN-round-
    trips to the input InChIKey (verified separately), so this is a breadth gain
    under the legality gate, not a hole."""
    mol, ring = _cage("C1CC23CC(C2)C12CC3C2")
    desc = VonBaeyerAnalyzer().analyze(mol, ring)
    assert desc.descriptor_string == "tetracyclo[2.2.2.1^1,8.1^2,4]"
    assert desc.legality is True
    base = generate_polycyclic_name(mol)
    assert base is not None
    assert base.startswith("tetracyclo[2.2.2.1^1,8.1^2,4]")


# --------------------------------------------------------------------------
# The three producers refuse an unverified descriptor
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,_n", ILLEGAL_CAGES)
def test_name_polycyclic_complete_fails_closed(smiles, _n):
    """Raising, not returning None: returning None cascades to a fragment namer
    that would name a single sub-ring (same reason as the G0 aromaticity
    refusal directly above the gate)."""
    mol = Chem.MolFromSmiles(smiles)
    with pytest.raises(OrthonymLimitError):
        name_polycyclic_complete(mol)


@pytest.mark.parametrize("smiles,_n", ILLEGAL_CAGES)
def test_generate_polycyclic_name_refuses(smiles, _n):
    assert generate_polycyclic_name(Chem.MolFromSmiles(smiles)) is None


@pytest.mark.parametrize("smiles,_n", ILLEGAL_CAGES)
def test_name_polycyclic_with_heteroatoms_refuses(smiles, _n):
    """All-carbon input routes through `generate_polycyclic_name`, so this also
    covers the no-heteroatom delegation at the top of that function."""
    assert name_polycyclic_with_heteroatoms(Chem.MolFromSmiles(smiles)) is None


# --------------------------------------------------------------------------
# ... and keep naming everything that IS legal
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,descriptor", LEGAL_CAGES)
def test_legal_cages_still_name(smiles, descriptor):
    mol = Chem.MolFromSmiles(smiles)
    base = generate_polycyclic_name(mol)
    assert base is not None
    assert base.startswith(descriptor)
    # `name_polycyclic_complete` consults the retained-name tables before it
    # analyses (adamantane returns `adamantane`), so assert only that it still
    # produces a name -- the descriptor assertion above is the systematic one.
    result = name_polycyclic_complete(mol)
    assert result is not None
    assert result[0]


def test_legality_gate_is_not_a_blanket_refusal():
    """A substituted cage -- the branch of `analyze` that does NOT renumber --
    still names. Guards against the gate being wired so tightly that only the
    canonical/unsubstituted branch survives."""
    mol = Chem.MolFromSmiles("CC12CC3CC(C1)CC(C3)C2")   # 1-methyladamantane
    result = name_polycyclic_complete(mol)
    assert result is not None
    assert result[0] == "1-methyltricyclo[3.3.1.1^3,7]decane"


# --------------------------------------------------------------------------
# The helper itself: it must be able to FAIL, and to pass
# --------------------------------------------------------------------------
def test_legality_verified_rejects_a_tampered_numbering():
    """Swapping two locants breaks the bond-set equality without touching the
    descriptor string -- proves the check reads the numbering, not just the
    bracket arithmetic (which is unchanged by a swap)."""
    mol, ring = _cage("C1C2CC3CC1CC(C2)C3")
    desc = VonBaeyerAnalyzer().analyze(mol, ring)
    assert VonBaeyerAnalyzer._legality_verified(mol, ring, desc) is True
    a, b = sorted(ring)[:2]
    desc.numbering[a], desc.numbering[b] = desc.numbering[b], desc.numbering[a]
    assert VonBaeyerAnalyzer._descriptor_is_valid(desc, ring) is True
    assert VonBaeyerAnalyzer._legality_verified(mol, ring, desc) is False


def test_legality_verified_rejects_a_tampered_descriptor_string():
    """The ring-count word is part of the name (P-23.2.6.1.1, :9645), so a cage
    spelled `tetracyclo` when its circuit rank is 3 denotes a different system
    even though every bracket number is right."""
    mol, ring = _cage("C1C2CC3CC1CC(C2)C3")
    desc = VonBaeyerAnalyzer().analyze(mol, ring)
    desc.descriptor_string = "tetracyclo[3.3.1.1^3,7]"
    assert VonBaeyerAnalyzer._legality_verified(mol, ring, desc) is False


def test_legality_verified_handles_absent_input():
    mol, ring = _cage("C1C2CC3CC1CC(C2)C3")
    assert VonBaeyerAnalyzer._legality_verified(mol, ring, None) is False
