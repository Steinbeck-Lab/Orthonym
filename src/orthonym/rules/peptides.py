"""
Peptide naming using the acylamino convention.

IUPAC P-66.6.6: Linear peptides are named using amino acid nomenclature:
1. Detecting peptide bonds (-C(=O)-NH-)
2. Walking the chain to extract residue SMILES
3. Identifying each residue by trivial name (P-66.6.6.1)
4. Adding L-/D- stereo prefixes based on CIP labels (P-66.6.6.3)
5. Assembling: acyl forms for N-terminal residues, full name for C-terminal

IUPAC P-66.6.6.2: peptide naming convention uses N->C direction,
with acyl (glycyl, alanyl...) forms for all residues except C-terminal.

Examples:
    Gly-Gly        -> glycylglycine
    L-Ala-Gly      -> L-alanylglycine
    Gly-L-Ala-L-Leu -> glycyl-L-alanyl-L-leucine
"""

from typing import Optional, List, Dict
from rdkit import Chem
from ..perception.stereo import assign_stereochemistry
from ..data.amino_acids import (
    STANDARD_AMINO_ACIDS,
    AMINO_ACID_ACYL_NAMES,
    get_amino_acid_name,
    get_amino_acid_acyl_name,
)


# SMARTS patterns
# Peptide bond: carbonyl_C - amide_N
# [CX3](=O)[NX3;H1][CX4] matches C(=O)-NH-CH pattern
_PEPTIDE_BOND_SMARTS = "[CX3](=O)[NX3;H1][CX4]"

# Terminal primary amine (free NH2, not part of amide C(=O)N)
_TERMINAL_NH2_SMARTS = "[NX3;H2;!$([NX3][CX3]=O)]"

# Terminal carboxylic acid
_TERMINAL_COOH_SMARTS = "[CX3](=O)[OX2H1]"

# Alpha amino acid core: NH2-CH(R)-C(=O)
_ALPHA_AA_CORE_SMARTS = "[NX3;H2,H1][CX4][CX3](=O)"


def name_peptide(mol) -> Optional[str]:
    """
    Name a peptide molecule using the acylamino convention.

    Returns None if the molecule is not a valid peptide or residues
    cannot be identified as standard amino acids.

    Args:
        mol: RDKit Mol object

    Returns:
        Peptide name (e.g., "glycyl-L-alanine") or None
    """
    if mol is None:
        return None

    # Step 1: Verify this is a true peptide
    if not _is_valid_peptide(mol):
        return None

    # Step 2: Extract residue SMILES by walking the peptide chain
    residue_smiles_list = _extract_residues(mol)
    if residue_smiles_list is None or len(residue_smiles_list) < 2:
        return None

    # Step 3: Identify residues and get stereo prefixes
    named_residues = _identify_residues(residue_smiles_list)
    if named_residues is None:
        return None

    # Step 4: Assemble the peptide name
    return _assemble_peptide_name(named_residues)


def _is_valid_peptide(mol) -> bool:
    """
    Check if molecule is a valid linear peptide.

    Requirements:
    - Has at least one peptide bond (-C(=O)-NH-CH-)
    - Has a terminal primary amine (free NH2)
    - Has a terminal carboxylic acid (-COOH)
    - Has an alpha-amino acid core pattern

    This prevents N-acyl amino acids (e.g., N-acetylglycine) from being
    misrouted: they have a peptide bond pattern but no NH2 on the acyl side.
    """
    peptide_pattern = Chem.MolFromSmarts(_PEPTIDE_BOND_SMARTS)
    if not peptide_pattern or not mol.GetSubstructMatches(peptide_pattern):
        return False

    nh2_pattern = Chem.MolFromSmarts(_TERMINAL_NH2_SMARTS)
    if not nh2_pattern or not mol.GetSubstructMatches(nh2_pattern):
        return False

    cooh_pattern = Chem.MolFromSmarts(_TERMINAL_COOH_SMARTS)
    if not cooh_pattern or not mol.GetSubstructMatches(cooh_pattern):
        return False

    aa_core = Chem.MolFromSmarts(_ALPHA_AA_CORE_SMARTS)
    if not aa_core or not mol.HasSubstructMatch(aa_core):
        return False

    return True


def _extract_residues(mol) -> Optional[List[str]]:
    """
    Extract amino acid residues from a linear peptide by walking from
    N-terminal to C-terminal, cleaving at peptide bonds.

    Returns a list of SMILES strings for each residue (as free amino acids),
    ordered N-terminal to C-terminal.
    """
    # Find the peptide bond C-N single bonds to cleave
    peptide_pat = Chem.MolFromSmarts(_PEPTIDE_BOND_SMARTS)
    if peptide_pat is None:
        return None
    matches = mol.GetSubstructMatches(peptide_pat)
    if not matches:
        return None

    # In the SMARTS [CX3](=O)[NX3;H1][CX4], match indices are:
    # [0] = carbonyl C, [1] = =O oxygen, [2] = amide N, [3] = alpha C on N-side
    # The peptide bond to cleave is between match[0] (C) and match[2] (N)
    peptide_bond_cn_pairs = []
    for m in matches:
        carbonyl_c = m[0]
        amide_n = m[2]
        peptide_bond_cn_pairs.append((carbonyl_c, amide_n))

    # Get the actual bond indices to cleave
    bond_indices_to_cleave = []
    for c_idx, n_idx in peptide_bond_cn_pairs:
        bond = mol.GetBondBetweenAtoms(c_idx, n_idx)
        if bond is not None:
            bond_indices_to_cleave.append(bond.GetIdx())

    if not bond_indices_to_cleave:
        return None

    # Use FragmentOnBonds with dummy atoms
    frag_mol = Chem.FragmentOnBonds(
        mol, bond_indices_to_cleave, addDummies=True,
        dummyLabels=[(i, i) for i in range(len(bond_indices_to_cleave))]
    )

    # Get fragments as individual mol objects
    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
    if not frags or len(frags) < 2:
        return None

    # Reconstruct each fragment as a free amino acid
    # and determine N->C ordering
    residue_data = []
    for frag in frags:
        aa_smi = _reconstruct_free_amino_acid(frag)
        if aa_smi is None:
            return None

        # Classify fragment position based on what dummies it has
        has_c_terminal_dummy = False  # Dummy replacing N (on C-terminal side of bond)
        has_n_terminal_dummy = False  # Dummy replacing C(=O) (on N-terminal side)
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() == 0:  # Dummy
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == 'C':
                        # Check if this C has a double-bond O neighbor (carbonyl)
                        has_carbonyl = any(
                            n2.GetSymbol() == 'O' and
                            frag.GetBondBetweenAtoms(nbr.GetIdx(), n2.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                            for n2 in nbr.GetNeighbors()
                            if n2.GetIdx() != atom.GetIdx() and n2.GetSymbol() == 'O'
                            and frag.GetBondBetweenAtoms(nbr.GetIdx(), n2.GetIdx()) is not None
                        )
                        if has_carbonyl:
                            # Dummy is on the C(=O) side -> this is the C-terminal end of this fragment
                            has_c_terminal_dummy = True
                        else:
                            has_n_terminal_dummy = True
                    elif nbr.GetSymbol() == 'N':
                        has_n_terminal_dummy = True

        # Determine fragment position:
        # N-terminal fragment: has c_terminal_dummy only (original N-term)
        # C-terminal fragment: has n_terminal_dummy only (original C-term)
        # Internal fragment: has both
        if has_c_terminal_dummy and not has_n_terminal_dummy:
            position = 'n_terminal'
        elif has_n_terminal_dummy and not has_c_terminal_dummy:
            position = 'c_terminal'
        elif has_c_terminal_dummy and has_n_terminal_dummy:
            position = 'internal'
        else:
            position = 'unknown'

        residue_data.append({
            'smiles': aa_smi,
            'position': position,
            'frag': frag,
        })

    # Order: N-terminal first, then internal (by original atom index), C-terminal last
    n_terms = [r for r in residue_data if r['position'] == 'n_terminal']
    c_terms = [r for r in residue_data if r['position'] == 'c_terminal']
    internals = [r for r in residue_data if r['position'] == 'internal']

    if len(n_terms) != 1 or len(c_terms) != 1:
        # Fallback: can't determine order
        return None

    ordered = [n_terms[0]] + internals + [c_terms[0]]
    return [r['smiles'] for r in ordered]


def _reconstruct_free_amino_acid(frag: Chem.Mol) -> Optional[str]:
    """
    Reconstruct a free amino acid from a peptide cleavage fragment.

    Each fragment has dummy atoms [*] at cleavage points:
    - Dummy bonded to carbonyl C (replacing N): replace dummy with OH to form -COOH
    - Dummy bonded to N or non-carbonyl C (replacing C=O): remove dummy, N gets H

    Returns canonical SMILES of the reconstructed free amino acid, or None.
    """
    rw_mol = Chem.RWMol(frag)

    # Collect dummy atoms and determine replacement type
    dummies = []
    for atom in rw_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummies.append(atom.GetIdx())

    if not dummies:
        # No dummies: already a complete amino acid (shouldn't happen)
        return Chem.MolToSmiles(frag, canonical=True)

    # Process dummies from highest index to lowest (avoid index shifting on removal)
    dummies.sort(reverse=True)

    for d_idx in dummies:
        dummy = rw_mol.GetAtomWithIdx(d_idx)
        neighbors = list(dummy.GetNeighbors())

        if not neighbors:
            rw_mol.RemoveAtom(d_idx)
            continue

        nbr = neighbors[0]

        if nbr.GetSymbol() == 'C':
            # Check if neighbor C is a carbonyl carbon
            is_carbonyl = False
            for n2 in nbr.GetNeighbors():
                if n2.GetIdx() != d_idx and n2.GetSymbol() == 'O':
                    bond = rw_mol.GetBondBetweenAtoms(nbr.GetIdx(), n2.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                        is_carbonyl = True
                        break

            if is_carbonyl:
                # Dummy replaces amide N: convert to OH for -COOH
                rw_mol.GetAtomWithIdx(d_idx).SetAtomicNum(8)  # O
                rw_mol.GetAtomWithIdx(d_idx).SetNumExplicitHs(1)  # OH
                rw_mol.GetAtomWithIdx(d_idx).SetNoImplicit(True)
            else:
                # Dummy on non-carbonyl C: this shouldn't normally happen
                # for peptide bonds, but handle gracefully
                _remove_dummy_add_h(rw_mol, d_idx, nbr)
        elif nbr.GetSymbol() == 'N':
            # Dummy replaces C(=O): remove dummy, N gets H
            _remove_dummy_add_h(rw_mol, d_idx, nbr)
        else:
            rw_mol.RemoveAtom(d_idx)

    try:
        Chem.SanitizeMol(rw_mol)
        return Chem.MolToSmiles(rw_mol, canonical=True)
    except Exception:
        return None


def _remove_dummy_add_h(rw_mol: Chem.RWMol, dummy_idx: int, neighbor) -> None:
    """Remove a dummy atom and add an implicit H to its neighbor."""
    # Get current explicit H count of neighbor
    if neighbor.GetSymbol() == 'N':
        neighbor.SetNumExplicitHs(neighbor.GetNumExplicitHs() + 1)
    rw_mol.RemoveAtom(dummy_idx)


def _identify_residues(
    residue_smiles_list: List[str],
) -> Optional[List[Dict[str, str]]]:
    """
    Identify each residue by trivial name and determine stereo prefix.

    Returns a list of dicts with keys:
        - 'name': trivial amino acid name (e.g., "glycine")
        - 'acyl': acyl form (e.g., "glycyl")
        - 'stereo': "L-", "D-", or "" (empty for achiral)

    Returns None if any residue cannot be identified as a standard amino acid.
    """
    result = []
    for smi in residue_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        can_smi = Chem.MolToSmiles(mol, canonical=True)

        # Look up trivial name (with stereo)
        aa_name = get_amino_acid_name(can_smi)

        if aa_name is None:
            # Try without stereochemistry (strip isomeric info)
            nostereo_smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            nostereo_mol = Chem.MolFromSmiles(nostereo_smi)
            if nostereo_mol is not None:
                nostereo_can = Chem.MolToSmiles(nostereo_mol, canonical=True)
                aa_name = get_amino_acid_name(nostereo_can)

        if aa_name is None:
            # Non-standard residue: cannot name with acylamino convention
            return None

        # Get acyl form
        acyl = get_amino_acid_acyl_name(aa_name)
        if acyl is None:
            return None

        # Determine stereochemistry prefix
        stereo = _get_stereo_prefix(mol, aa_name)

        result.append({
            'name': aa_name,
            'acyl': acyl,
            'stereo': stereo,
        })

    return result


# Amino acids where L-configuration = R (CIP), not S.
# CIP priority inversion: sulfur (Z=16) or selenium (Z=34) in side chain
# outranks oxygen (Z=8) in COOH, reversing the normal L=S mapping.
_CIP_INVERTED_AMINO_ACIDS = {"cysteine", "cystine", "selenocysteine"}


def _get_stereo_prefix(mol: Chem.Mol, aa_name: str) -> str:
    """
    Get L-/D- stereo prefix for an amino acid residue.

    For standard amino acids:
    - S configuration at alpha-carbon -> "L-"
    - R configuration at alpha-carbon -> "D-"
    - No stereocenter (glycine) -> ""

    Exception: Cysteine family (sulfur/selenium side chain) has inverted
    CIP priorities, so L = R and D = S.
    """
    if aa_name == "glycine":
        return ""  # Glycine is achiral

    # Assign CIP labels (idempotent guard)
    assign_stereochemistry(mol)

    # Find the alpha-carbon: sp3 carbon bonded to both N and C(=O)
    # SMARTS: [NX3][CX4][CX3](=O)
    # Match indices: [0]=N, [1]=alpha-C, [2]=carbonyl-C, [3]=O
    alpha_pattern = Chem.MolFromSmarts("[NX3][CX4][CX3](=O)")
    if alpha_pattern is None:
        return ""

    matches = mol.GetSubstructMatches(alpha_pattern)
    if not matches:
        return ""

    # alpha-carbon is index 1 in the match
    alpha_c_idx = matches[0][1]
    alpha_atom = mol.GetAtomWithIdx(alpha_c_idx)

    cip = alpha_atom.GetPropsAsDict().get('_CIPCode', '')

    # Cysteine family: L = R, D = S (sulfur/selenium outranks oxygen in CIP)
    if aa_name.lower() in _CIP_INVERTED_AMINO_ACIDS:
        if cip == 'R':
            return "L-"
        elif cip == 'S':
            return "D-"
    else:
        # Standard amino acids: L = S, D = R
        if cip == 'S':
            return "L-"
        elif cip == 'R':
            return "D-"
    return ""


def _assemble_peptide_name(named_residues: List[Dict[str, str]]) -> str:
    """
    Assemble the final peptide name from identified residues.

    IUPAC P-66.6.6.2 rules:
    - C-terminal (last) residue: use full amino acid name
    - All other residues: use acyl form (e.g., glycyl-, alanyl-)
    - Add L-/D- stereo prefix before each residue name (P-66.6.6.3)
    - Hyphen ONLY before a residue that has a stereo prefix (L-/D-)
    - No hyphen before achiral residues (concatenate directly)

    Examples:
        glycyl + glycine -> glycylglycine
        L-alanyl + glycine -> L-alanylglycine
        glycyl + L-alanine -> glycyl-L-alanine
        L-alanyl + L-alanine -> L-alanyl-L-alanine
        glycyl + L-alanyl + L-leucine -> glycyl-L-alanyl-L-leucine
    """
    parts = []
    for i, res in enumerate(named_residues):
        is_c_terminal = (i == len(named_residues) - 1)

        if is_c_terminal:
            base = res['name']
        else:
            base = res['acyl']

        stereo = res['stereo']  # "L-", "D-", or ""
        parts.append((stereo, base))

    # Build the name: insert hyphen only before stereo-prefixed residues
    result = parts[0][0] + parts[0][1]  # First residue
    for stereo, base in parts[1:]:
        if stereo:
            # Has stereo prefix (e.g., "L-") -> hyphen before it
            result += "-" + stereo + base
        else:
            # No stereo prefix -> concatenate directly
            result += base

    return result
