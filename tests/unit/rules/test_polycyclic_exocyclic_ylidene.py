"""Exocyclic C= on a von Baeyer parent must be NAMED, never dropped.

Phase 1b, PIN-path half.

``get_polycyclic_substituents`` used to blanket-``continue`` on EVERY exocyclic
double bond, with the comment "=O, =S for suffixes". That premise holds for the
chalcogens -- which are in any case already removed one step earlier via
``exclude_atoms`` from ``_detect_ring_functional_groups`` -- and is false for
carbon: there is no suffix path for an exocyclic ``=CH2``, so the atom simply
vanished from the name. A name that silently omits an atom is a wrong
STRUCTURE; only the downstream OPSIN round-trip caught it, which means with no
JVM present the wrong name shipped.

Two properties are pinned:

  * an exocyclic ``=CH2`` is emitted as a P-29.2 ``methylidene`` prefix at the
    ring locant the numbering machinery already assigned;
  * when the exocyclic carbon fragment is outside the ylidene class the whole
    ring handler FAILS CLOSED (``OrthonymLimitError``) instead of dropping
    the atom -- the drop must not survive in any form.

The ``=O``/``=S`` behaviour is pinned as UNCHANGED: this fix narrows the skip
to non-carbon, it does not remove it, because promoting an unrecognised ring
ketone to an ``oxo`` prefix would ship a non-PIN name where the code correctly
abstains today.
"""
import pytest
from rdkit import Chem

from orthonym.errors import OrthonymLimitError
from orthonym.rules.polycyclic import (
    VonBaeyerAnalyzer,
    _get_largest_connected_ring_component,
    get_polycyclic_substituents,
    name_polycyclic_complete,
)


def analyze_polycyclic(mol):
    """The von Baeyer descriptor for ``mol``'s largest ring component, exactly
    as ``name_polycyclic_complete`` derives it."""
    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)
    if not ring_atoms:
        return None
    ring_atoms = _get_largest_connected_ring_component(mol, ring_atoms)
    return VonBaeyerAnalyzer().analyze(mol, ring_atoms)

# The user-supplied molecule: tetracyclic von Baeyer cage, CHO, COOH, two ring
# methyls and ONE exocyclic =CH2 at ring locant 13.
ACCEPTANCE_SMILES = (
    "C(=O)[C@@H]1[C@@]23[C@H]([C@@]4(CCC[C@]([C@@H]14)(C(=O)O)C)C)"
    "CC[C@@H](C(C2)=C)C3"
)
ACCEPTANCE_TARGET = (
    "(1R,2S,3S,4R,8S,9S,12R)-2-formyl-4,8-dimethyl-13-methylidene"
    "tetracyclo[10.2.1.0^1,9.0^3,8]pentadecane-4-carboxylic acid"
)


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def _substituents(smiles, exclude=None):
    """Run the substituent detector over the molecule's von Baeyer system."""
    mol = _mol(smiles)
    desc = analyze_polycyclic(mol)
    assert desc is not None, f"no von Baeyer analysis for {smiles!r}"
    ring_atoms = set(desc.numbering)
    return mol, get_polycyclic_substituents(
        mol, ring_atoms, desc.numbering, exclude_atoms=exclude or set())


class TestExocyclicCarbonIsNotDropped:

    def test_methylidene_is_emitted_for_a_simple_cage(self):
        """bicyclo[2.2.1]heptane bearing an exocyclic =CH2."""
        mol, subs = _substituents("C=C1CC2CCC1C2")
        names = sorted(s["name"] for s in subs)
        assert names == ["methylidene"], (
            f"expected a single 'methylidene' prefix, got {names!r}"
        )

    def test_every_heavy_atom_is_accounted_for(self):
        """The atom-conservation property the drop violated: the union of the
        ring atoms and every substituent's atom_indices is the whole molecule."""
        smiles = "C=C1CC2CCC1C2"
        mol, subs = _substituents(smiles)
        desc = analyze_polycyclic(mol)
        claimed = set(desc.numbering)
        for s in subs:
            claimed |= set(s["atom_indices"])
        heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
        assert claimed == heavy, f"unclaimed atoms: {sorted(heavy - claimed)}"

    def test_acceptance_molecule_names_to_the_verified_target(self):
        mol = _mol(ACCEPTANCE_SMILES)
        result = name_polycyclic_complete(mol)
        assert result is not None
        assert result[0] == ACCEPTANCE_TARGET


class TestFailsClosedInsteadOfDropping:

    def test_unnameable_exocyclic_carbon_raises(self):
        """A branched exocyclic ylidene is outside the built class. The handler
        must refuse the whole ring system rather than lose the atom."""
        # 2-methylpropylidene on norbornane -- branched, not in the class.
        mol = _mol("CC(C)C=C1CC2CCC1C2")
        desc = analyze_polycyclic(mol)
        assert desc is not None
        with pytest.raises(OrthonymLimitError):
            get_polycyclic_substituents(mol, set(desc.numbering), desc.numbering)


class TestChalcogenSkipUnchanged:
    """=O / =S keep the existing skip: narrowing, not removing."""

    @pytest.mark.parametrize("smiles", ["O=C1CC2CCC1C2", "S=C1CC2CCC1C2"])
    def test_exocyclic_chalcogen_still_skipped(self, smiles):
        mol, subs = _substituents(smiles)
        assert subs == [], (
            f"exocyclic chalcogen must remain the FG/suffix path's business, "
            f"got {subs!r}"
        )
