"""
Monocyclic lactone naming using IUPAC heterocyclic replacement nomenclature.

Lactones are cyclic esters. Monocyclic lactones are named as heterocyclic
ketones: the ring oxygen gives the heterocyclic parent name, and the
carbonyl is expressed as a -one suffix at position 2.

Naming algorithm:
1. Detect lactone: ring contains -C(=O)-O- where both C and ester O are
   in the same ring, and the exocyclic O is a double-bonded carbonyl.
2. Determine ring size (3-10 membered supported).
3. Get the saturated heterocyclic parent name for the ring with one O:
   - 3-membered: oxirane
   - 4-membered: oxetane
   - 5-membered: oxolane
   - 6-membered: oxane
   - 7-membered: oxepane
4. Apply vowel elision (remove terminal 'e' before '-one').
5. Return "{parent_stem}-2-one" (carbonyl always at position 2, adjacent
   to ring O at position 1).

Reference: IUPAC 2013 Blue Book, P-25.5.2 (Lactones)

Examples:
    O=C1CCO1    (beta-propiolactone)    -> oxetan-2-one
    O=C1CCCO1   (gamma-butyrolactone)   -> oxolan-2-one
    O=C1CCCCO1  (delta-valerolactone)   -> oxan-2-one
    O=C1CCCCCO1 (epsilon-caprolactone)  -> oxepan-2-one
"""

from typing import Dict, Optional

from rdkit import Chem

from ..rules.heterocycles import build_hw_name


# ---------------------------------------------------------------------------
# Lactone detection
# ---------------------------------------------------------------------------

def is_monocyclic_lactone(mol) -> Optional[Dict]:
    """
    Detect whether a molecule is (or contains) a monocyclic lactone.

    A monocyclic lactone has:
    - A ring containing an ester motif: -C(=O)-O- where both the carbonyl
      carbon and the ester oxygen are in the same ring.
    - The carbonyl oxygen is exocyclic (double-bonded to the carbonyl C).
    - The ring is not fused with another ring (monocyclic only).

    Args:
        mol: RDKit Mol object (or None).

    Returns:
        Dict with detection info if lactone found:
            ring_atoms: tuple of atom indices in the lactone ring
            carbonyl_idx: atom index of the carbonyl carbon (C=O)
            ester_O_idx: atom index of the ring (ester) oxygen
            carbonyl_O_idx: atom index of the exocyclic carbonyl oxygen
            ring_size: number of atoms in the ring
        None if not a monocyclic lactone.
    """
    if mol is None:
        return None

    # SMARTS: carbonyl carbon with double-bonded O and single-bonded O
    # [CX3](=O)[OX2] matches the ester/acid core
    # match[0] = carbonyl carbon
    # match[1] = carbonyl oxygen (=O, exocyclic)
    # match[2] = ester oxygen (-O-, must be in ring)
    pattern = Chem.MolFromSmarts("[CX3](=O)[OX2]")
    matches = mol.GetSubstructMatches(pattern)

    if not matches:
        return None

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    if not atom_rings:
        return None

    for match in matches:
        carbonyl_c = match[0]
        carbonyl_o = match[1]
        ester_o = match[2]

        # Both carbonyl C and ester O must be in the SAME ring
        for ring in atom_rings:
            ring_set = set(ring)
            if carbonyl_c in ring_set and ester_o in ring_set:
                # Exocyclic carbonyl O must NOT be in the ring
                if carbonyl_o in ring_set:
                    continue

                # Check monocyclic: no ring atom should appear in another ring
                is_monocyclic = True
                for other_ring in atom_rings:
                    if set(other_ring) == ring_set:
                        continue
                    if ring_set & set(other_ring):
                        is_monocyclic = False
                        break

                if not is_monocyclic:
                    continue

                return {
                    "ring_atoms": ring,
                    "carbonyl_idx": carbonyl_c,
                    "ester_O_idx": ester_o,
                    "carbonyl_O_idx": carbonyl_o,
                    "ring_size": len(ring),
                }

    return None


# ---------------------------------------------------------------------------
# Lactone ring naming
# ---------------------------------------------------------------------------

# Supported ring sizes for monocyclic lactone naming
_SUPPORTED_RING_SIZES = frozenset(range(3, 11))


def name_lactone_ring(ring_size: int) -> Optional[str]:
    """
    Get the IUPAC name for a monocyclic lactone of a given ring size.

    Builds the heterocyclic parent name for a saturated ring with one O,
    then applies vowel elision and appends '-2-one'.

    Args:
        ring_size: Number of atoms in the lactone ring (3-10 supported).

    Returns:
        IUPAC name string (e.g., 'oxolan-2-one'), or None if ring size
        is not supported.

    Examples:
        >>> name_lactone_ring(4)
        'oxetan-2-one'
        >>> name_lactone_ring(5)
        'oxolan-2-one'
        >>> name_lactone_ring(6)
        'oxan-2-one'
        >>> name_lactone_ring(7)
        'oxepan-2-one'
    """
    if ring_size not in _SUPPORTED_RING_SIZES:
        return None

    # Get the saturated heterocyclic parent name for a ring with one O
    # build_hw_name expects heteroatoms as (locant, element) tuples
    parent_name = build_hw_name(
        heteroatoms=[(1, "O")],
        ring_size=ring_size,
        is_saturated=True,
        is_aromatic=False,
    )

    if not parent_name:
        return None

    # Apply vowel elision: remove terminal 'e' before '-one'
    # oxetane -> oxetan, oxolane -> oxolan, oxane -> oxan, oxepane -> oxepan
    if parent_name.endswith("e"):
        stem = parent_name[:-1]
    else:
        stem = parent_name

    return f"{stem}-2-one"


# ---------------------------------------------------------------------------
# Full lactone naming
# ---------------------------------------------------------------------------

def name_monocyclic_lactone(mol) -> Optional[str]:
    """
    Generate the IUPAC name for a monocyclic lactone.

    Detects whether the molecule is a monocyclic lactone and returns
    its name using heterocyclic replacement nomenclature with -one suffix.

    Args:
        mol: RDKit Mol object (or None).

    Returns:
        IUPAC name string (e.g., 'oxolan-2-one'), or None if the
        molecule is not a monocyclic lactone.

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('O=C1CCCO1')
        >>> name_monocyclic_lactone(mol)
        'oxolan-2-one'
    """
    info = is_monocyclic_lactone(mol)
    if info is None:
        return None

    return name_lactone_ring(info["ring_size"])
