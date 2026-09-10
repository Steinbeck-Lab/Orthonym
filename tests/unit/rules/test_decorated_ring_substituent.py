""".2 — decorated-ring substituent namer (the S2 co-delivery emitter).

A ring demoted to a substituent must carry its OWN substituent prefixes:
``2-(2-nitrothiophen-3-yl)pyrimidine``, ``5-(2-oxocyclohexyl)pyridine-3-
carboxylic acid`` — not the bare ``2-thienyl`` / ``5-cyclohexyl`` forms that
drop the decoration (the rt75_0582 RT regression and the P2/P3 producer gaps).

Numbering per (the Blue Book): (a) ring heteroatoms (fixed/lowest)
-> (b) indicated hydrogen -> (c) FREE VALENCE (attachment; suffix rank — see
the '6-carboxynaphthalen-2-yl' example at:3262 where the free valence locant
beats the carboxy prefix) -> (f) detachable-prefix set -> (g) first-cited
alphabetical. The primitive is GUARDED: returns None (caller keeps its legacy
form, zero regression) for fused rings, NH-azoles, unsupported substituent
classes, or rings with no supported decoration at all.

Every end-to-end expectation was OPSIN-verified (name -> structure -> InChI
match) before being baked in.
"""
import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.ring_substituents import decorated_ring_substituent_name


def _ring_and_attachment(smiles, ring_pred, other_ring_pred):
    """Return (mol, ring_atoms, attachment_idx): the ring satisfying ring_pred,
    and its atom bonded to the ring satisfying other_ring_pred."""
    mol = Chem.MolFromSmiles(smiles)
    rings = mol.GetRingInfo().AtomRings()
    ring = next(r for r in rings if ring_pred(mol, r))
    other = next(set(r) for r in rings if other_ring_pred(mol, r))
    for a in ring:
        for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
            if nbr.GetIdx() in other:
                return mol, ring, a
    raise AssertionError("no inter-ring attachment found")


def _is_thiophene(mol, ring):
    return len(ring) == 5 and any(
        mol.GetAtomWithIdx(a).GetSymbol() == 'S' for a in ring)


def _is_pyrimidine(mol, ring):
    return len(ring) == 6 and sum(
        1 for a in ring if mol.GetAtomWithIdx(a).GetSymbol() == 'N') == 2


def _is_carbocycle(mol, ring):
    return all(mol.GetAtomWithIdx(a).GetSymbol() == 'C' for a in ring)


def _is_pyridine(mol, ring):
    return len(ring) == 6 and sum(
        1 for a in ring if mol.GetAtomWithIdx(a).GetSymbol() == 'N') == 1


class TestPrimitive:
    def test_nitrothiophene(self):
        # The rt75_0582 blocker: S=1, attachment beats nitro for low locant
        # (c) before (f)) -> attachment 3, nitro 2.
        mol, ring, att = _ring_and_attachment(
            "O=[N+]([O-])c1sccc1-c1ncccn1", _is_thiophene, _is_pyrimidine)
        assert decorated_ring_substituent_name(mol, ring, att) == \
            "2-nitrothiophen-3-yl"

    def test_oxocyclohexyl(self):
        # All-carbon ring: attachment fixed at 1 (elided), oxo gets 2.
        mol, ring, att = _ring_and_attachment(
            "OC(=O)c1ccc(cc1)C1CCCCC1=O",
            lambda m, r: _is_carbocycle(m, r) and not m.GetAtomWithIdx(r[0]).GetIsAromatic(),
            lambda m, r: m.GetAtomWithIdx(r[0]).GetIsAromatic())
        assert decorated_ring_substituent_name(mol, ring, att) == \
            "2-oxocyclohexyl"

    def test_clean_ring_returns_none(self):
        # No supported decoration -> None (caller keeps legacy form; zero churn)
        mol, ring, att = _ring_and_attachment(
            "C1CCCCC1c1ccccc1",
            lambda m, r: not m.GetAtomWithIdx(r[0]).GetIsAromatic(),
            lambda m, r: m.GetAtomWithIdx(r[0]).GetIsAromatic())
        assert decorated_ring_substituent_name(mol, ring, att) is None

    def test_unsupported_substituent_returns_none(self):
        # Acetyloxy (ester) on the ring is NOT in the v1 table -> None
        mol, ring, att = _ring_and_attachment(
            "CC(=O)OC1CCCCC1c1ccccc1",
            lambda m, r: not m.GetAtomWithIdx(r[0]).GetIsAromatic(),
            lambda m, r: m.GetAtomWithIdx(r[0]).GetIsAromatic())
        assert decorated_ring_substituent_name(mol, ring, att) is None

    def test_fused_ring_returns_none(self):
        # Decorated naphthalene ring: fused -> guard out
        mol = Chem.MolFromSmiles("Oc1ccc2ccccc2c1-c1ccncc1")
        rings = mol.GetRingInfo().AtomRings()
        pyridine = next(set(r) for r in rings if _is_pyridine(mol, r))
        naph_ring = next(r for r in rings
                         if _is_carbocycle(mol, r) and not set(r) & pyridine)
        att = next(a for a in naph_ring
                   if any(n.GetIdx() in pyridine
                          for n in mol.GetAtomWithIdx(a).GetNeighbors()))
        assert decorated_ring_substituent_name(mol, naph_ring, att) is None

    def test_nh_azole_returns_none(self):
        # Decorated pyrrole-type ring (indicated-H interplay) -> guard out
        mol = Chem.MolFromSmiles("Cc1cc[nH]c1-c1ccncc1")
        rings = mol.GetRingInfo().AtomRings()
        pyrrole = next(r for r in rings if len(r) == 5)
        pyridine = next(set(r) for r in rings if len(r) == 6)
        att = next(a for a in pyrrole
                   if any(n.GetIdx() in pyridine
                          for n in mol.GetAtomWithIdx(a).GetNeighbors()))
        assert decorated_ring_substituent_name(mol, pyrrole, att) is None

    def test_chloro_methyl_phenyl_prefix_set(self):
        # Two prefixes on phenyl: (f) lowest prefix SET decides the
        # direction (methyl ortho -> {2,3} beats {5,6}); citation order stays
        # alphabetical: chloro-3, methyl-2.
        mol, ring, att = _ring_and_attachment(
            "Cc1c(Cl)cccc1-c1ccncc1",
            lambda m, r: _is_carbocycle(m, r),
            _is_pyridine)
        assert decorated_ring_substituent_name(mol, ring, att) == \
            "3-chloro-2-methylphenyl"

    def test_alpha_tiebreak_ortho_ortho(self):
        # Cl and CH3 on the two ortho positions: both directions give {2,6};
        # (g) gives the first-cited alphabetical prefix (chloro) the
        # lower locant -> 2-chloro-6-methylphenyl.
        mol, ring, att = _ring_and_attachment(
            "Cc1cccc(Cl)c1-c1ccncc1",
            lambda m, r: _is_carbocycle(m, r),
            _is_pyridine)
        assert decorated_ring_substituent_name(mol, ring, att) == \
            "2-chloro-6-methylphenyl"


class TestEndToEndP2P3:
    """The producer paths that drop ring FGs even WITHOUT S2 (probe-verified).
    All expectations OPSIN-verified against the input structures."""

    @pytest.mark.parametrize("smiles,expected", [
        # P2: heterocycle parent, oxo dropped on HEAD
        ("OC(=O)c1cncc(c1)C1CCCCC1=O",
         "5-(2-oxocyclohexyl)pyridine-3-carboxylic acid"),
        # P3: benzene parent, wrong '1-oxo' locant on HEAD
        ("OC(=O)c1ccc(cc1)C1CCCCC1=O",
         "4-(2-oxocyclohexyl)benzoic acid"),
        # P3 five-ring variant
        ("OC(=O)c1ccc(cc1)C1CCCC1=O",
         "4-(2-oxocyclopentyl)benzoic acid"),
        # P3 cyano variant (HEAD: 'unknown organic compound')
        ("OC(=O)c1ccc(cc1)C1CCCCC1C#N",
         "4-(2-cyanocyclohexyl)benzoic acid"),
    ])
    def test_targets(self, smiles, expected):
        assert name_compound(smiles) == expected
