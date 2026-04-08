"""
Fragment capping for decomposition engine.

After bond cleavage, replaces dummy atoms with H or OH to produce
valid molecule fragments for recursive naming.

IUPAC P-44.1: Principal characteristic group determines which side of a
cleavage retains the parent suffix. Acid-side fragments are capped with OH
to reconstruct the parent acid; alkyl/amine-side fragments are capped with H.
"""

from typing import Dict, List, Set

from rdkit import Chem
from rdkit.Chem import RWMol, rdCIPLabeler


def cleave_and_cap(mol, bond_infos: List[Dict], acid_side_oh: bool = True) -> List[Dict]:
    """Cleave molecule at specified bonds and return H/OH-capped fragments.

    Uses RDKit FragmentOnBonds to cleave, then replaces dummy atoms:
    - Acid-side fragments: cap with OH (produces carboxylic acid) if acid_side_oh=True
      per IUPAC P-44.1 principal characteristic group preservation
    - Alkyl/amine-side fragments: cap with H

    Fragment side labeling uses dummyLabels to track which dummy came from
    which side of the bond: label 1 = acid side, label 2 = alkyl/amine side.

    Args:
        mol: RDKit Mol object
        bond_infos: List of bond info dicts from find_cleavable_bonds()
        acid_side_oh: If True, cap acid-side fragments with OH

    Returns:
        List of dicts with keys: smiles, side, original_atoms
    """
    if not bond_infos:
        return []

    # Collect bond indices and track acid-side atom indices
    bond_indices = []
    # Map: bond_idx -> (acid_atom_idx, other_atom_idx, bond_type)
    bond_side_map: Dict[int, Dict] = {}

    for info in bond_infos:
        bidx = info["bond_idx"]
        bond_indices.append(bidx)

        acid_atom = info.get("acid_atom")
        if info["type"] in ("ester", "carbamate"):
            other_atom = info.get("alkyl_atom")
        elif info["type"] == "amide":
            other_atom = info.get("amine_atom")
        else:
            other_atom = info.get("alkyl_atom")

        bond_side_map[bidx] = {
            "acid_atom": acid_atom,
            "other_atom": other_atom,
            "type": info["type"],
        }

    # Collect all acid-side atom indices across all bonds
    acid_side_atoms: Set[int] = set()
    for binfo in bond_side_map.values():
        if binfo["acid_atom"] is not None:
            acid_side_atoms.add(binfo["acid_atom"])

    # Use distinct dummy labels: (1, 2) for each bond
    # Label 1 = on the first atom side, Label 2 = on the second atom side
    # We need to figure out which end of the RDKit bond is the acid side
    dummy_labels = []
    for bidx in bond_indices:
        rdkit_bond = mol.GetBondWithIdx(bidx)
        begin_atom = rdkit_bond.GetBeginAtomIdx()
        end_atom = rdkit_bond.GetEndAtomIdx()

        binfo = bond_side_map[bidx]
        acid_atom_idx = binfo["acid_atom"]

        if begin_atom == acid_atom_idx:
            # begin is acid side -> label begin with 1, end with 2
            dummy_labels.append((1, 2))
        else:
            # end is acid side -> label begin with 2, end with 1
            dummy_labels.append((2, 1))

    # Fragment the molecule
    frag_mol = Chem.FragmentOnBonds(
        mol,
        bond_indices,
        addDummies=True,
        dummyLabels=dummy_labels,
    )

    # Get individual fragment molecules and their atom index mappings
    frag_atom_lists: List[List[int]] = []
    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False,
                              fragsMolAtomMapping=frag_atom_lists)

    results: List[Dict] = []

    for frag, frag_atoms in zip(frags, frag_atom_lists):
        rwm = RWMol(frag)

        # Determine which side this fragment is on by checking dummy labels
        has_acid_dummy = False  # has a dummy with isotope label 1 (acid-side cap point)
        has_alkyl_dummy = False  # has a dummy with isotope label 2 (alkyl-side cap point)

        for atom in rwm.GetAtoms():
            if atom.GetAtomicNum() == 0:
                iso = atom.GetIsotope()
                if iso == 1:
                    has_acid_dummy = True
                elif iso == 2:
                    has_alkyl_dummy = True

        # Determine side label
        # A fragment on the acid side gets alkyl-side dummies (label 2)
        # A fragment on the alkyl side gets acid-side dummies (label 1)
        if has_alkyl_dummy and not has_acid_dummy:
            side = "acid"
        elif has_acid_dummy and not has_alkyl_dummy:
            side = "alkyl"
        else:
            # Both labels present (middle fragment in multi-bond cleavage)
            # or neither (shouldn't happen normally)
            side = "middle"

        # For amide and sulfonamide bonds, label the non-acid side as "amine"
        # Check if any of the bond_infos for this fragment is amide/sulfonamide type
        for binfo in bond_side_map.values():
            if binfo["type"] == "amide":
                other = binfo["other_atom"]
                # Check if this fragment contains the amine atom
                # (using original atom indices from frag_atom_lists)
                if other is not None and other in set(frag_atoms):
                    if side == "alkyl":
                        side = "amine"
                    break
            if binfo["type"] == "sulfonamide":
                other = binfo["other_atom"]
                if other is not None and other in set(frag_atoms):
                    if side == "alkyl":
                        side = "amine"
                    break

        # Cap dummy atoms
        for atom in rwm.GetAtoms():
            if atom.GetAtomicNum() == 0:
                iso = atom.GetIsotope()

                # Alkyl-side dummy (label 2) on acid or middle fragment -> cap with OH or H
                # Middle fragments (between two cleavage points) get OH on label-2
                # to reconstruct the acid-side functional group (DECO-21).
                if iso == 2 and acid_side_oh and side in ("acid", "middle"):
                    atom.SetAtomicNum(8)  # Oxygen
                    atom.SetIsotope(0)
                    atom.SetNoImplicit(False)
                    atom.SetNumExplicitHs(1)
                else:
                    # Default: H-cap
                    atom.SetAtomicNum(1)  # Hydrogen
                    atom.SetIsotope(0)

        # Sanitize and clean up
        try:
            Chem.SanitizeMol(rwm)
        except Exception:
            # If sanitization fails, try without kekulization
            Chem.SanitizeMol(rwm, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE)

        clean = Chem.RemoveHs(rwm)

        # Re-assign CIP labels on the fragment after capping.
        # CIP depends on the substituent tree, which changes when a bond
        # is cleaved and capped. Re-assignment ensures fragment CIP labels
        # reflect the fragment's actual context, not the pre-cleavage parent.
        # This is defensive: the recursive naming path also re-assigns CIP,
        # but doing it here ensures SMILES encodes correct stereo even if
        # fragments are used outside the recursive naming pipeline.
        rdCIPLabeler.AssignCIPLabels(clean)

        smiles = Chem.MolToSmiles(clean)

        results.append({
            "smiles": smiles,
            "side": side,
            "original_atoms": set(frag_atoms),
        })

    return results
