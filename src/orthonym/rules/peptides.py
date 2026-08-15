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

Examples (P-103.3.4 omits the L descriptor for Table-10.4 amino acids):
    Gly-Gly         -> glycylglycine
    L-Ala-Gly       -> alanylglycine
    Gly-L-Ala-L-Leu -> glycylalanylleucine
    D-Ala-Gly       -> D-alanylglycine   (D IS cited)

v32 Phase 2 (breadth, phase2e SPY -- ):
two ADDITIVE fallbacks, tried only when the flat acylamino convention above
declines, and gated behind an explicit OPSIN round-trip (``_rt_verified``) so
a table/topology miss degrades to abstention, never a wrong or atom-dropping
name (0-wrong ABSOLUTE):

- Lever A (``_try_n_acyl_cap``): an N-terminus ACYLATED by a plain (nitrogen-
  free) acyl group -- a fatty/simple acyl cap -- in front of an otherwise
  fully standard >=2-residue alpha chain. The cap is named via the general/
  systematic namer and prepended in ACYL form, e.g.
  ``3-hydroxy-11-methyltridecanoylglycylglycine``.
- Lever B (``_try_gamma_link_whole``): a single non-alpha ('gamma'/'beta')
  bond formed from a standard amino acid's OWN side-chain carboxyl (only
  glutamic acid's gamma- and aspartic acid's beta-carboxyl have this shape
  among the standard 20) -- glutathione's linkage type. Builds the fully
  SYSTEMATIC substitutive acyl/amido construction (P-66.1.1.4.3) instead of
  the flat chain shorthand, because OPSIN's flat-chain grammar mis-parses a
  non-retained continuing acyl word (SPY-verified) and a bare 'glutamyl'
  shorthand imposes an L-configuration OPSIN does not know is undefined.
"""

from typing import Optional, List, Dict, Tuple
import re
from rdkit import Chem
from ..perception.stereo import assign_stereochemistry
from ..data.amino_acids import (
    STANDARD_AMINO_ACIDS,
    AMINO_ACID_ACYL_NAMES,
    get_amino_acid_name,
    get_amino_acid_acyl_name,
)


# SMARTS patterns
# Peptide bond: carbonyl_C - amide_N. H1 is the ordinary secondary-amide backbone
# bond; H0 also admits the TERTIARY amide formed when a cyclic imino acid (proline,
# hydroxyproline) is the AMINE component — its ring N loses its only H on bonding,
# so `[NX3;H1]` alone missed every X-Pro / X-Pro-Y peptide (`alanyl-L-proline`,
# `Glu-Pro-Phe`, ...). Proline as the ACYL (N-terminal) residue already worked.
# Broadening to H0 is safe: `_is_alpha_carboxyl_bond` still requires the bond be
# alpha on BOTH sides, and `_extract_residues`/`_identify_residues` fail closed
# unless every cleaved fragment reconstructs to a STANDARD amino acid — so an
# N-methyl backbone or any non-amino-acid tertiary amide declines rather than
# emitting a mis-linked name.
_PEPTIDE_BOND_SMARTS = "[CX3](=O)[NX3;H0,H1][CX4]"

# Terminal FREE amine (not part of an amide C(=O)N). H2 is the ordinary
# alpha-amino N-terminus; H1 also admits a SECONDARY free amine so a cyclic
# imino-acid N-terminus (proline, hydroxyproline) is recognised — its ring N is
# H1, so the H2-only form rejected every Pro-N-terminal peptide. The
# !$([NX3][CX3]=O) exclusion still bars an ACYLATED N, so an N-acyl amino acid
# (N-acetylglycine: its only N is the amide) keeps failing this check; a
# non-peptide secondary amine (sarcosine) is already excluded upstream by the
# peptide-bond requirement.
_TERMINAL_NH2_SMARTS = "[NX3;H1,H2;!$([NX3][CX3]=O)]"

# Terminal carboxylic acid
_TERMINAL_COOH_SMARTS = "[CX3](=O)[OX2H1]"

# Alpha amino acid core: NH2-CH(R)-C(=O)
_ALPHA_AA_CORE_SMARTS = "[NX3;H2,H1][CX4][CX3](=O)"


# Constitution-robust residue identification (InChIKey first block = the skeleton
# layer). Exact-SMILES matching in get_amino_acid_name misses a residue whose
# reconstructed guanidine/imidazole TAUTOMER, or a FragmentOnBonds isotope label,
# differs from the table's spelling: reconstructed arginine
# ``NC(N)=NCCC[C@H](N)C(=O)O`` and the table's ``N=C(N)NCCCC(N)C(=O)O`` are the
# same constitution but different strings, so a genuine arginine-bearing tri+
# peptide failed to name at all. The InChIKey first block ignores tautomer,
# isotope and stereo, so it matches by CONSTITUTION; the L/D descriptor is
# recovered separately from CIP in _get_stereo_prefix. Built lazily once.
_AA_INCHIKEY1_TO_NAME: Optional[Dict[str, str]] = None


def _residue_inchikey1(mol) -> str:
    """First InChIKey block (skeleton) of a residue mol, or '' on failure."""
    from rdkit.Chem import inchi
    try:
        key = inchi.MolToInchiKey(mol)
    except Exception:
        return ""
    return key.split("-")[0] if key else ""


def _aa_inchikey1_map() -> Dict[str, str]:
    """{InChIKey-first-block -> trivial amino-acid name} over the standard set."""
    global _AA_INCHIKEY1_TO_NAME
    if _AA_INCHIKEY1_TO_NAME is None:
        built: Dict[str, str] = {}
        for smi, name in STANDARD_AMINO_ACIDS.items():
            aa_mol = Chem.MolFromSmiles(smi)
            if aa_mol is None:
                continue
            key = _residue_inchikey1(aa_mol)
            if key:
                # A standard amino acid is one constitution, so a first-block
                # collision across DIFFERENT names cannot occur; first-wins is a
                # no-op safety over the alternate-SMILES rows that share a name.
                built.setdefault(key, name)
        _AA_INCHIKEY1_TO_NAME = built
    return _AA_INCHIKEY1_TO_NAME


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
        # v32 Lever A: an acylated (non-free) N-terminus may still be a
        # nameable N-acyl-capped peptide (SPY §3a). Fails closed to None
        # (unchanged behaviour) unless the cap is a genuine plain acyl group
        # AND the residues behind it are a fully standard >=2-residue chain
        # AND the assembled candidate OPSIN-round-trips to this exact mol.
        return _rt_verified(mol, _try_n_acyl_cap(mol))

    # Step 2: Extract residue SMILES by walking the peptide chain
    residue_smiles_list = _extract_residues(mol)
    if residue_smiles_list is None or len(residue_smiles_list) < 2:
        # v32 Lever B: a single non-alpha ('gamma'/'beta') bond from a
        # standard amino acid's own side-chain carboxyl (SPY §3b). Same
        # fail-closed-to-None-unless-round-tripped guarantee as Lever A.
        return _rt_verified(mol, _try_gamma_link_whole(mol))

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


def _bears_carboxyl_carbon(atom, exclude_idx: int) -> bool:
    """True iff ``atom`` neighbours a carbon carrying a double-bonded O (a C=O),
    other than the atom at ``exclude_idx`` — i.e. it is an alpha-carbon bearing a
    carboxyl/amide carbon."""
    for c in atom.GetNeighbors():
        if c.GetIdx() == exclude_idx or c.GetSymbol() != 'C':
            continue
        if any(o.GetSymbol() == 'O'
               and c.GetOwningMol().GetBondBetweenAtoms(c.GetIdx(), o.GetIdx()) is not None
               and c.GetOwningMol().GetBondBetweenAtoms(
                   c.GetIdx(), o.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
               for o in c.GetNeighbors()):
            return True
    return False


def _is_alpha_carboxyl_bond(mol, carbonyl_c: int, amide_n: int) -> bool:
    """True iff the amide bond ``carbonyl_c``-``amide_n`` is an ALPHA-peptide bond
    on BOTH sides — the only bond an alpha-acylamino peptide name may describe.

    Acyl (C=O) side: ``carbonyl_c`` must be a residue's alpha-carboxyl — its
    alpha-carbon (a carbon neighbour that is not the amide N) bears that residue's
    backbone amino nitrogen. A side-chain carboxyl amide (glutathione's
    gamma-glutamyl) fails here.

    Amine (N) side: ``amide_n`` must be a residue's alpha-amino N — its
    alpha-carbon (a carbon neighbour that is not the carbonyl) bears a carboxyl
    carbon (its own alpha-carboxyl or the next peptide carbonyl). A side-chain
    amine amide (an epsilon-lysine isopeptide, ``Gly-eps-Lys``) fails here.

    Either failure => the caller fails closed rather than emit a mis-linked
    alpha name like ``glutamylcysteinylglycine`` / ``glycyllysine``.
    """
    c = mol.GetAtomWithIdx(carbonyl_c)
    acyl_alpha = any(
        nbr.GetSymbol() == 'C' and nbr.GetIdx() != amide_n
        and any(nn.GetSymbol() == 'N' for nn in nbr.GetNeighbors()
                if nn.GetIdx() != carbonyl_c)
        for nbr in c.GetNeighbors()
    )
    if not acyl_alpha:
        return False
    n = mol.GetAtomWithIdx(amide_n)
    amine_alpha = any(
        nbr.GetSymbol() == 'C' and nbr.GetIdx() != carbonyl_c
        and _bears_carboxyl_carbon(nbr, amide_n)
        for nbr in n.GetNeighbors()
    )
    return amine_alpha


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
        # Proline (and any cyclic imino acid) as the AMINE component: its ring N
        # has TWO ring-carbon `[CX4]` neighbours, so the SMARTS matches the SAME
        # C(=O)-N bond TWICE (once per ring carbon). Deduplicate before cleaving —
        # a duplicate bond index passed to FragmentOnBonds crashes RDKit (segfault),
        # and would also double-count the dummy labels.
        if (carbonyl_c, amide_n) in peptide_bond_cn_pairs:
            continue
        # Only an ALPHA-peptide bond is in scope: the carbonyl must be a residue's
        # ALPHA-carboxyl, i.e. its carbon neighbour (the alpha-carbon) itself bears
        # a nitrogen (that residue's backbone amino N). An ISOPEPTIDE bond formed
        # from a side-chain carboxyl (glutathione's gamma-glutamyl) has the carbonyl
        # on a side-chain carbon whose neighbour bears no amino N — naming it with
        # the alpha acylamino convention (`glutamyl...`) is a WRONG constitution, so
        # fail closed on ANY non-alpha bond rather than emit a mis-linked name.
        if not _is_alpha_carboxyl_bond(mol, carbonyl_c, amide_n):
            return None
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

    # Reconstruct each fragment as a free amino acid and record, per fragment, the
    # cleaved-bond LABELS on each side. FragmentOnBonds stamps each cleaved bond's
    # index (dummyLabels=[(i,i)...]) as the ISOTOPE of the two dummies it creates,
    # so the C(=O)-side dummy of residue X and the N-side dummy of residue X+1
    # carry the SAME label i — that shared label is the backbone edge X->X+1.
    residue_data = []
    for frag in frags:
        aa_smi = _reconstruct_free_amino_acid(frag)
        if aa_smi is None:
            return None

        cside_labels = set()  # labels on THIS residue's carbonyl C (its C-terminus)
        nside_labels = set()  # labels on THIS residue's amide N  (its N-terminus)
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() != 0:  # dummy only
                continue
            label = atom.GetIsotope()
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'C':
                    has_carbonyl = any(
                        n2.GetSymbol() == 'O'
                        and frag.GetBondBetweenAtoms(nbr.GetIdx(), n2.GetIdx()) is not None
                        and frag.GetBondBetweenAtoms(
                            nbr.GetIdx(), n2.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                        for n2 in nbr.GetNeighbors()
                        if n2.GetIdx() != atom.GetIdx() and n2.GetSymbol() == 'O'
                    )
                    (cside_labels if has_carbonyl else nside_labels).add(label)
                elif nbr.GetSymbol() == 'N':
                    nside_labels.add(label)

        residue_data.append({
            'smiles': aa_smi,
            'cside': cside_labels,
            'nside': nside_labels,
        })

    # Order residues by WALKING the backbone N->C via the shared bond labels — NOT
    # by fragment/atom index. Atom-index order is SMILES-spelling-dependent, so it
    # scrambled the two internal residues of any tetrapeptide+ (Val-Glu-Ile-Arg was
    # emitted Val-Ile-Glu-Arg on some spellings) — a wrong CONSTITUTION that only
    # SELF-01 stopped. A LINEAR peptide has exactly one N-terminal residue (no
    # N-side label), one C-terminal residue (no C-side label), and <=1 label per
    # side on every residue; a branch/fork (an isopeptide side-chain amide) breaks
    # one of those and fails closed here rather than emitting a mis-linked name.
    if any(len(r['cside']) > 1 or len(r['nside']) > 1 for r in residue_data):
        return None
    n_terms = [r for r in residue_data if not r['nside']]
    c_terms = [r for r in residue_data if not r['cside']]
    if len(n_terms) != 1 or len(c_terms) != 1:
        return None
    by_nside = {}
    for r in residue_data:
        for lbl in r['nside']:
            by_nside[lbl] = r  # each label appears on exactly one N-side (checked above)

    ordered, seen = [], set()
    cur = n_terms[0]
    while cur is not None:
        key = id(cur)
        if key in seen:      # cycle -> not a linear peptide
            return None
        seen.add(key)
        ordered.append(cur)
        if not cur['cside']:
            break            # reached the C-terminus
        cur = by_nside.get(next(iter(cur['cside'])))
    if len(ordered) != len(residue_data):
        return None          # disconnected: the walk did not cover every residue
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
            # Constitution fallback (purely additive: only fires where the
            # string lookups already returned None). Matches the residue by its
            # InChIKey skeleton, so a tautomer/isotope spelling that the exact
            # SMILES table missed (arginine guanidine, a FragmentOnBonds isotope
            # label) still resolves to its standard name. A wrong constitution
            # cannot collide here (distinct skeleton), and the whole-peptide name
            # still faces the downstream SELF-01 validity gate, so 0-wrong holds.
            aa_name = _aa_inchikey1_map().get(_residue_inchikey1(mol))

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

    IUPAC P-103.3.2 / P-103.3.4 rules:
    - C-terminal (last) residue: use full amino acid name
    - All other residues: use acyl form (e.g., glycyl-, alanyl-)
    - P-103.3.4: the stereodescriptor 'L' is NOT indicated for peptides composed
      of Table-10.4 amino acids (the only ones Orthonym names -- non-standard
      residues fail closed in _identify_residues). Only 'D' is cited, at the front
      of each acyl group / name that has that configuration. (BB verbatim:
      "The stereodescriptor 'L' is not indicated in the names ... of peptides
      composed of amino acids listed in Table 10.4. In contrast, the
      stereodescriptor 'D' is indicated at the front of the acyl group or name of
      each component having that configuration.")
    - Hyphen ONLY before a residue that carries a cited (D-) descriptor.
    - The residue's true config is preserved in res['stereo']; the L-omission is a
      display rule applied here, so a standalone amino acid (P-103.1) still shows L.

    Examples (P-103.3.2 / P-103.3.4):
        glycyl + glycine        -> glycylglycine
        L-alanyl + glycine      -> alanylglycine
        glycyl + L-alanine      -> glycylalanine
        L-alanyl + L-alanine    -> alanylalanine
        L-valyl+L-tyrosyl+L-Ile -> valyltyrosylisoleucine
        D-alanyl + glycine      -> D-alanylglycine
        glycyl + D-alanine      -> glycyl-D-alanine
    """
    parts = []
    for i, res in enumerate(named_residues):
        is_c_terminal = (i == len(named_residues) - 1)

        if is_c_terminal:
            base = res['name']
        else:
            base = res['acyl']

        # P-103.3.4: suppress the 'L-' descriptor for display; keep 'D-' (and the
        # achiral "" for glycine). res['stereo'] retains the true configuration.
        display_stereo = "" if res['stereo'] == "L-" else res['stereo']
        parts.append((display_stereo, base))

    # Build the name: insert hyphen only before a cited (D-) descriptor.
    result = parts[0][0] + parts[0][1]  # First residue
    for stereo, base in parts[1:]:
        if stereo:
            # Has a cited descriptor (D-) -> hyphen before it
            result += "-" + stereo + base
        else:
            # No cited descriptor (L omitted, or achiral) -> concatenate directly
            result += base

    return result


# ============================================================================
# v32 Phase 2 breadth: N-acyl cap (Lever A) + gamma/beta side-chain-carboxyl
# donor (Lever B). Both are ADDITIVE fallbacks tried only after the ordinary
# flat acylamino convention above declines, and both are gated by
# ``_rt_verified`` -- a table/topology miss degrades to abstention, never a
# wrong or atom-dropping name. See phase2e SPY for the derivation and the
# corpus-scale sizing ().
# ============================================================================


def _rt_verified(mol, candidate: Optional[str]) -> Optional[str]:
    """0-wrong guarantee for the v32 broadened paths (Lever A / Lever B).

    Returns ``candidate`` unchanged iff it OPSIN-round-trips to the EXACT
    input ``mol`` (full InChI match -- stricter than InChIKey, catches a
    stereo-imposing bare acyl word like a default-L 'glutamyl'). Fails
    CLOSED (returns None) on a mismatch, a parse error, or an unavailable
    OPSIN/Java -- unlike several existing fail-OPEN gates elsewhere in this
    codebase, there is no pre-fix default output to preserve here: the
    pre-Phase-2 behaviour for every path that reaches this function was
    already None, so "cannot verify" and "verified wrong" both correctly
    resolve to the same pre-existing abstention.
    """
    if candidate is None:
        return None
    try:
        from ..validation.opsin_roundtrip import opsin_roundtrip_check
        smi = Chem.MolToSmiles(mol)
        verdict = opsin_roundtrip_check(smi, candidate)
    except Exception:
        return None
    return candidate if verdict.get("passed") else None


def _is_plain_acyl_cap(mol, carbonyl_c: int, amide_n: int) -> bool:
    """True iff ``carbonyl_c``-``amide_n`` is a Lever-A N-cap bond: the AMINE
    side is a genuine peptide N-terminus (alpha, same test as
    ``_is_alpha_carboxyl_bond``'s amine-side check) AND the ACYL side
    contains NO nitrogen atom anywhere -- a plain acyl group (fatty acid,
    acetyl, aroyl...), not itself an amino-acid-shaped residue. This is what
    distinguishes an N-acyl CAP from a gamma/beta-linked DONOR residue
    (glutamic acid's gamma-carboxyl acyl side still carries its own free
    alpha-amino N, so it correctly fails this check and is Lever B's
    territory instead).
    """
    n = mol.GetAtomWithIdx(amide_n)
    amine_alpha = any(
        nbr.GetSymbol() == 'C' and nbr.GetIdx() != carbonyl_c
        and _bears_carboxyl_carbon(nbr, amide_n)
        for nbr in n.GetNeighbors()
    )
    if not amine_alpha:
        return False

    visited = {carbonyl_c, amide_n}
    stack = [carbonyl_c]
    while stack:
        cur = stack.pop()
        for nbr in mol.GetAtomWithIdx(cur).GetNeighbors():
            idx = nbr.GetIdx()
            if idx in visited:
                continue
            if nbr.GetSymbol() == 'N':
                return False  # acyl side reaches a nitrogen -- not a plain cap
            visited.add(idx)
            stack.append(idx)
    return True


def _split_one_bond(mol, carbonyl_c: int, amide_n: int):
    """Cleave exactly the ``carbonyl_c``-``amide_n`` bond, returning
    ``(acyl_side_frag, amine_side_frag)`` as two RDKit Mol fragments (each
    still carrying one dummy atom at its cut point), or ``(None, None)`` on
    any failure. Used by both Lever A and Lever B -- a single-bond special
    case of the multi-bond walk in ``_extract_residues``."""
    bond = mol.GetBondBetweenAtoms(carbonyl_c, amide_n)
    if bond is None:
        return None, None
    try:
        frag_mol = Chem.FragmentOnBonds(
            mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(0, 0)])
        mapping: List = []
        frags = Chem.GetMolFrags(
            frag_mol, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=mapping)
    except Exception:
        return None, None
    if len(frags) != 2:
        return None, None
    acyl_frag = amine_frag = None
    for frag, idxs in zip(frags, mapping):
        if carbonyl_c in idxs:
            acyl_frag = frag
        elif amide_n in idxs:
            amine_frag = frag
    return acyl_frag, amine_frag


def _name_standard_chain(smi: str):
    """Name ``smi`` (a free amino acid / peptide fragment, N reconstructed to
    a free amine) via the ORDINARY, UNCHANGED flat acylamino machinery --
    either as a >=2-residue chain (``_extract_residues`` + ``_identify_residues``)
    or as a single free residue (``_identify_residues`` on the bare SMILES).
    Fails closed (returns None) exactly as that machinery always has: any
    internal non-alpha bond or non-standard residue declines. Shared by
    Lever A's remainder and Lever B's 'flat' acceptor path."""
    frag_mol = Chem.MolFromSmiles(smi)
    if frag_mol is None:
        return None
    residues = _extract_residues(frag_mol)
    if residues is not None and len(residues) >= 2:
        named = _identify_residues(residues)
        if named is None:
            return None
        return _assemble_peptide_name(named)
    named1 = _identify_residues([smi])
    if named1 is None:
        return None
    return _assemble_peptide_name(named1)


def _try_n_acyl_cap(mol) -> Optional[str]:
    """Lever A (SPY §3a): name an N-acyl-capped peptide.

    Scoped to a cap in front of a genuine >=2-residue standard alpha chain
    (NOT a single free N-acyl amino acid, e.g. N-acetylglycine -- that shape
    already names correctly via the general/composer substitutive pipeline
    today, VERIFIED 2026-08-15 ('acetamidoacetic acid'), so this lever must
    not compete with it). Returns an UNVERIFIED candidate string; the caller
    (``name_peptide``) gates it through ``_rt_verified``.
    """
    peptide_pat = Chem.MolFromSmarts(_PEPTIDE_BOND_SMARTS)
    if peptide_pat is None:
        return None
    matches = mol.GetSubstructMatches(peptide_pat)
    if not matches:
        return None
    cooh_pattern = Chem.MolFromSmarts(_TERMINAL_COOH_SMARTS)
    if not cooh_pattern or not mol.GetSubstructMatches(cooh_pattern):
        return None

    seen = set()
    cap_bond = None
    for m in matches:
        carbonyl_c, amide_n = m[0], m[2]
        if (carbonyl_c, amide_n) in seen:
            continue
        seen.add((carbonyl_c, amide_n))
        if _is_plain_acyl_cap(mol, carbonyl_c, amide_n):
            if cap_bond is not None:
                return None  # >1 plain-acyl bond -- ambiguous, decline
            cap_bond = (carbonyl_c, amide_n)
    if cap_bond is None:
        return None
    carbonyl_c, amide_n = cap_bond

    cap_frag, remainder_frag = _split_one_bond(mol, carbonyl_c, amide_n)
    if cap_frag is None or remainder_frag is None:
        return None
    cap_smi = _reconstruct_free_amino_acid(cap_frag)
    remainder_smi = _reconstruct_free_amino_acid(remainder_frag)
    if cap_smi is None or remainder_smi is None:
        return None

    # Require a genuine >=2-residue chain BEHIND the cap (see docstring) --
    # a bare single free amino acid is already handled elsewhere, so this
    # does NOT go through ``_name_standard_chain``'s single-residue fallback.
    remainder_mol = Chem.MolFromSmiles(remainder_smi)
    if remainder_mol is None:
        return None
    residues = _extract_residues(remainder_mol)
    if residues is None or len(residues) < 2:
        return None
    named = _identify_residues(residues)
    if named is None:
        return None
    remainder_name = _assemble_peptide_name(named)

    from ..namer import Orthonym
    from ..errors import is_failure_name
    try:
        cap_acid_name = Orthonym(
            style="pin", _disable_opsin_validity_gate=True).name(cap_smi)
    except Exception:
        return None
    if is_failure_name(cap_acid_name):
        return None

    from ..decomposition.fragment_assembly import _acid_to_acyl
    cap_acyl = _acid_to_acyl(cap_acid_name)
    if cap_acyl is None:
        return None

    if remainder_name.startswith('D-'):
        return f"{cap_acyl}-{remainder_name}"
    return f"{cap_acyl}{remainder_name}"


# Standard amino acids with a second (side-chain) carboxyl -- the only two in
# Table 10.4 -- mapped to the carbon-chain LENGTH from that side-chain
# carboxyl (counted as C1) to the alpha carbon (inclusive of both ends).
# Glutamic acid: C(=O)-CH2-CH2-CH(NH2)(COOH) -> 4 carbons ('butan-').
# Aspartic acid: C(=O)-CH2-CH(NH2)(COOH)      -> 3 carbons ('propan-').
_OMEGA_CARBOXYL_CHAIN_LENGTH = {"glutamic acid": 4, "aspartic acid": 3}

# The 'amino' substituent prefix (optionally locant-prefixed) ANCHORED at the
# start of the name, e.g. matches '1-amino' in
# '1-aminocyclopropane-1-carboxylic acid'. Substituent prefixes FUSE directly
# onto the parent hydride name with no space/hyphen ('aminocyclopropane', not
# 'amino-cyclopropane'), so a token search for a word-bounded 'amino' never
# matches at all -- anchoring at position 0 instead only fires when 'amino'
# is the name's SOLE (or alphabetically-first) substituent, i.e. exactly the
# free-NH2-only non-standard residue shape this lever targets (ACC and
# similar). A residue with additional earlier-sorting substituents declines
# here rather than risk swapping the wrong token.
_AMINO_PREFIX_RE = re.compile(r'^(\d+-)?amino')


def _swap_amino_for_amido(name: str, amido_group: str) -> Optional[str]:
    """Replace the leading 'amino' substituent prefix in ``name`` with the
    parenthesized ``amido_group`` at the same locant (P-16.3.3 complex-
    substituent enclosure), e.g. '1-aminocyclopropane...' + '4-amino-4-
    carboxybutanamido' -> '1-(4-amino-4-carboxybutanamido)cyclopropane...'.
    Declines (returns None) when ``name`` does not start with 'amino' --
    ambiguous, so never guessed."""
    m = _AMINO_PREFIX_RE.match(name)
    if m is None:
        return None
    locant = m.group(1) or ''
    return f"{locant}({amido_group})" + name[m.end():]


def _name_gamma_acceptor(remainder_smi: str) -> Optional[Tuple[str, str]]:
    """Name the ACCEPTOR side of a Lever-B gamma/beta-link bond. Returns
    ``(name, mode)`` where ``mode`` is:
      - 'flat': ``name`` is a standard single residue or a standard >=2-
        residue flat acylamino chain (via ``_name_standard_chain``) -- no
        'amino' token to swap, the acyl group is simply prepended.
      - 'swap': ``name`` is a NON-standard residue named via the general/
        systematic namer (an already-existing capability, SPY-verified),
        with a leading 'amino' substituent prefix to be replaced by the
        donor's acyl-amido group (validated by ``_swap_amino_for_amido``,
        called later by the caller once the acyl-amido text is built).
    Returns None if the remainder cannot be named at all.
    """
    named = _name_standard_chain(remainder_smi)
    if named is not None:
        return named, 'flat'

    from ..namer import Orthonym
    from ..errors import is_failure_name
    try:
        general_name = Orthonym(
            style="pin", _disable_opsin_validity_gate=True).name(remainder_smi)
    except Exception:
        return None
    if is_failure_name(general_name):
        return None
    return general_name, 'swap'


def _find_single_nonalpha_bond(mol) -> Optional[Tuple[int, int]]:
    """Find every peptide-bond-shaped match in the WHOLE molecule and return
    the (carbonyl_c, amide_n) of the ONE that fails ``_is_alpha_carboxyl_bond``,
    PROVIDED every other match passes it. Returns None when there are zero or
    more than one non-alpha match (a mid-chain non-standard residue, e.g.
    chebi 371, produces >=2 non-alpha-adjacent matches or is excluded upstream
    via the N-cap already being handled by Lever A) -- declines rather than
    guess which one is the intended gamma/beta donor link."""
    peptide_pat = Chem.MolFromSmarts(_PEPTIDE_BOND_SMARTS)
    if peptide_pat is None:
        return None
    matches = mol.GetSubstructMatches(peptide_pat)
    if not matches:
        return None
    seen = set()
    nonalpha = []
    for m in matches:
        carbonyl_c, amide_n = m[0], m[2]
        if (carbonyl_c, amide_n) in seen:
            continue
        seen.add((carbonyl_c, amide_n))
        if not _is_alpha_carboxyl_bond(mol, carbonyl_c, amide_n):
            nonalpha.append((carbonyl_c, amide_n))
    if len(nonalpha) != 1:
        return None
    return nonalpha[0]


def _try_gamma_donor_link(mol, carbonyl_c: int, amide_n: int) -> Optional[str]:
    """Lever B core (SPY §3b): build the systematic acyl/amido construction
    for a single gamma/beta side-chain-carboxyl amide bond. Returns an
    UNVERIFIED candidate string; the caller gates it through ``_rt_verified``.
    """
    donor_frag, acceptor_frag = _split_one_bond(mol, carbonyl_c, amide_n)
    if donor_frag is None or acceptor_frag is None:
        return None
    donor_smi = _reconstruct_free_amino_acid(donor_frag)
    acceptor_smi = _reconstruct_free_amino_acid(acceptor_frag)
    if donor_smi is None or acceptor_smi is None:
        return None

    donor_info = _identify_residues([donor_smi])
    if donor_info is None:
        return None  # not a standard amino acid -- Lever B does not apply
    donor_name = donor_info[0]['name']
    donor_stereo = donor_info[0]['stereo']  # "L-" / "D-" / ""

    n = _OMEGA_CARBOXYL_CHAIN_LENGTH.get(donor_name)
    if n is None:
        return None  # only Glu (gamma) / Asp (beta) have a 2nd carboxyl

    # Glu/Asp are not in the cysteine-family CIP-inversion set, so the raw
    # CIP letter is simply what P-103.3.4's L/D mapping already computed.
    cip_letter = {"L-": "S", "D-": "R", "": None}.get(donor_stereo)
    stereo_prefix = f"({n}{cip_letter})-" if cip_letter else ""

    from ..data.chain_names import get_chain_prefix
    from ..assembly.substituent_naming import acyl_carbons_to_amido_prefix
    stem = get_chain_prefix(n)
    amido_stem = acyl_carbons_to_amido_prefix(n)
    if not stem or amido_stem is None:
        return None
    oyl_group = f"{stereo_prefix}{n}-amino-{n}-carboxy{stem}anoyl"
    amido_group = f"{stereo_prefix}{n}-amino-{n}-carboxy{amido_stem}"

    acceptor = _name_gamma_acceptor(acceptor_smi)
    if acceptor is None:
        return None
    acceptor_name, mode = acceptor

    if mode == 'flat':
        if acceptor_name.startswith('D-'):
            return f"{oyl_group}-{acceptor_name}"
        return f"{oyl_group}{acceptor_name}"

    return _swap_amino_for_amido(acceptor_name, amido_group)


def _try_gamma_link_whole(mol) -> Optional[str]:
    """Lever B entry point: locate the sole non-alpha bond in ``mol`` (if
    any) and attempt the systematic donor/acceptor construction. Returns an
    UNVERIFIED candidate string; the caller gates it through ``_rt_verified``.
    """
    bond_pair = _find_single_nonalpha_bond(mol)
    if bond_pair is None:
        return None
    carbonyl_c, amide_n = bond_pair
    return _try_gamma_donor_link(mol, carbonyl_c, amide_n)
