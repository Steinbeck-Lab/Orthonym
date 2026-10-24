"""A ring of a polycyclic system is never the benzene parent (large-polycycle plan, Task 3).

 "MONOCYCLIC HYDROCARBONS", (the Blue Book,:8073): "Benzene is the retained
name for C6H6". A six-ring that shares an atom with another ring belongs to a polycyclic system.
Corannulene (C20H10) reached the benzene assembler: ``rules.benzene._preferred_benzene_parent_ring``
picked one of its five rim six-rings, collected no substituent (the substituent walk stops at the
ring system) and returned the bare string 'benzene' -- a candidate for a different molecule that
only and the default tier's emission rule kept out of the output.
"""
import pytest
from rdkit import Chem

import orthonym.assembly.composer as composer
from orthonym.rules.benzene import is_benzene_ring
from tests.support.large_polycycles import be, pin

CORANNULENE = "c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51"


def _benzene_candidates(monkeypatch, tier_call, smiles):
    """Every string composer._assemble_benzene_name returns while ``smiles`` is named."""
    seen = []
    orig = composer._assemble_benzene_name

    def spy(features, style):
        out = orig(features, style)
        seen.append(out)
        return out

    monkeypatch.setattr(composer, "_assemble_benzene_name", spy)
    tier_call(smiles)
    return seen


@pytest.mark.unit
@pytest.mark.parametrize("tier_call", [pin, be], ids=["pin", "best-effort"])
def test_a_ring_inside_a_ring_system_is_never_offered_as_benzene(monkeypatch, tier_call):
    assert "benzene" not in _benzene_candidates(monkeypatch, tier_call, CORANNULENE)


@pytest.mark.unit
@pytest.mark.parametrize("smiles", ["Brc1ccccc1", "Clc1ccc(Br)cc1"])
def test_the_spy_sees_the_benzene_assembler_for_an_isolated_ring(monkeypatch, smiles):
    # a positive control for the trace above: the assembler does run for these, and offers a name
    assert _benzene_candidates(monkeypatch, pin, smiles)


@pytest.mark.unit
@pytest.mark.parametrize("smiles,ring_is_benzene", [
    ("c1ccccc1", True),
    ("Cc1ccccc1", True),
    ("c1ccc(cc1)-c1ccccc1", True),                        # a ring assembly: the biaryl bond is no ring bond
    ("c1ccc2ccccc2c1", False),                            # naphthalene: both rings share the fusion bond
    ("c1ccc2c(c1)CCC2", False),                           # indane: the benzene ring is fused
    (CORANNULENE, False),                                 # peri-fused: rim rings share atoms with the five-ring
    ("C1CCCCC1", False),
    ("c1ccncc1", False),
])
def test_is_benzene_ring(smiles, ring_is_benzene):
    mol = Chem.MolFromSmiles(smiles)
    rings = [r for r in mol.GetRingInfo().AtomRings() if len(r) == 6]
    assert any(is_benzene_ring(mol, r) for r in rings) is ring_is_benzene


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [("Brc1ccccc1", "bromobenzene"),       # BB:2919 "(PIN)"
                                             ("Cc1ccccc1", "toluene"),            # BB:8097 "(PIN)"
                                             ("c1ccc(cc1)-c1ccccc1", "1,1'-biphenyl")])  # BB:2000; assembly rings stay isolated
def test_benzene_parent_controls(smiles, expected):
    assert pin(smiles)["name"] == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [("c1ccc2ccccc2c1", "naphthalene"),
                                             ("c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61", "coronene")])
def test_fused_hydrocarbon_controls(smiles, expected):
    assert pin(smiles)["name"] == expected


# A chain that bears the principal characteristic group stays the parent when the ring system it hangs on has a
# fused benzene ring. The partition filter of ``namer._classify`` drops the principal-group instance of a
# substituent (a second -ol on the ring side chain) by testing its atoms against the enumerated substituents; it
# read ``features.ring_substituents`` of the principal ring even when the chain is the parent, and that dict of a
# fused ring (no longer called benzene) lists the chain itself, so every instance was "inside a substituent", the
# filter would have emptied the suffix set and left the set alone, and the chain candidate was refused
# ('PG instance not on parent chain'): the ring became the parent and the -ol a prefix.
SUFFIX_ROWS = {
    "ol": ("Cc1cc2c(c(C)c1CCO)C[C@@](C)(CO)C2",
           "2-[(S)-2-(hydroxymethyl)-2,4,6-trimethyl-2,3-dihydro-1H-inden-5-yl]ethan-1-ol"),
    "ol_flat": ("Cc1cc2c(c(C)c1CCO)CC(C)(CO)C2",
                "2-[2-(hydroxymethyl)-2,4,6-trimethyl-2,3-dihydro-1H-inden-5-yl]ethan-1-ol"),
    "acid": ("Cc1cc2c(c(C)c1CC(=O)O)C[C@@](C)(CO)C2",
             "[(S)-2-(hydroxymethyl)-2,4,6-trimethyl-2,3-dihydro-1H-inden-5-yl]acetic acid"),
}


@pytest.mark.unit
@pytest.mark.opsin_gate                 # with the validity gate off (the suite's default) the first chain candidate drops the hydroxymethyl
@pytest.mark.parametrize("label", sorted(SUFFIX_ROWS))
def test_the_chain_with_the_principal_group_stays_the_parent_of_a_fused_benzo_ring(label):
    """ (the Blue Book, section `` SENIORITY ORDER FOR PARENT STRUCTURES``): "The senior parent
    structure has the maximum number of substituents corresponding to the principal characteristic group (suffix)".
    The chain holds one -OH (or the carboxy group) and the ring system none, so the chain is the parent and the
    name ends in the suffix; a name that cites it as ``2-hydroxyethyl`` has no parent that carries a suffix."""
    from tests.support.rt_assert import assert_full_rt
    smiles, expected = SUFFIX_ROWS[label]
    assert be(smiles)["name"] == expected
    assert_full_rt(expected, smiles)
