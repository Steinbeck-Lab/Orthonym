"""
Amino acid detection and naming.

Alpha-amino acids have the general structure: H2N-CHR-COOH
SMARTS pattern: [NX3;H2,H1][CX4][CX3](=O)[OX2H1]

For standard amino acids, trivial names are used.
For non-standard, systematic naming applies:
- Carboxylic acid is principal group
- Amine becomes "amino" prefix

Peptide guard: molecules with peptide bonds (-C(=O)-NH-) are NOT simple amino
acids and must NOT be flattened to "2-aminoXXXanoic acid".
"""

from typing import Optional, List, Tuple
from rdkit import Chem

from ..data.amino_acids import get_amino_acid_name, is_standard_amino_acid


# SMARTS for alpha-amino acid pattern
# [NX3;H2,H1] - primary or secondary amine nitrogen
# [CX4] - sp3 carbon (alpha carbon)
# [CX3](=O)[OX2H1] - carboxylic acid
#
# .0: N broadened to also admit H0 -- mirrors
# rules/peptides.py's own `_PEPTIDE_BOND_SMARTS` broadening. A
# peptide's C-TERMINAL residue's own N-Calpha-COOH triad is what this
# pattern is meant to catch (N here can be the free N-terminus of a
# mono-residue "peptide", OR the amide N carried over from the PRECEDING
# residue's peptide bond, hence the pre-existing H1 branch). When that
# C-terminal residue is a cyclic imino acid (proline/hydroxyproline), its
# ring nitrogen has TWO ring substituents already, so acylation by the
# preceding residue leaves it with NO free hydrogen (a tertiary amide, H0)
# -- the H1/H2-only pattern silently missed every X-...-Pro peptide, which
# is why `is_peptide()` (below) returned False even though
# `rules.peptides.name_peptide()` already named these correctly once
# reached directly (sec.4).
# Was previously kept in sync with `peptides.py`'s pattern only by name, not
# by value -- this restores that sync.
ALPHA_AMINO_ACID_SMARTS = "[NX3;H2,H1,H0][CX4][CX3](=O)[OX2H1]"

# SMARTS for peptide bond (secondary amide linkage between amino acids)
# [NX3;H0,H1] - secondary OR tertiary amide nitrogen (NH, or ring N with no H
# left once acylated -- a C-terminal cyclic imino acid such as proline)
# [CX3](=O) - carbonyl carbon
# [CX4] - alpha carbon on the acid side
# This detects -C(=O)-NH-CH- (and -C(=O)-N(ring)-CH- for Pro/Hyp) linkages
# typical of peptide bonds.
#
# .0: broadened to H0,H1 to match
# rules/peptides.py's `_PEPTIDE_BOND_SMARTS` exactly (same rationale as
# ALPHA_AMINO_ACID_SMARTS above) -- a dipeptide whose ONLY bond is an X-Pro
# bond (Pro as the sole C-terminal residue) has no OTHER bond to fall back
# on, so this constant must admit H0 too, not just the amino-acid pattern.
PEPTIDE_BOND_SMARTS = "[CX3](=O)[NX3;H0,H1][CX4]"

# .1 (capped termini): a peptide whose C-TERMINAL residue's
# alpha-carboxyl is a PRIMARY CARBOXAMIDE (-C(=O)NH2, e.g. a bioactive
# peptide's amidated C-terminus) has NO free -COOH anywhere in the whole
# molecule, so ALPHA_AMINO_ACID_SMARTS (which hard-requires `[OX2H1]`) never
# matches and `is_peptide()` returned False -- the dispatcher never called
# `rules.peptides.name_peptide` at all, even once that module's own
# `_is_valid_peptide`/producer were extended to accept this cap
# (sec.3: 71% of the true-
# peptide backlog is exactly this "both termini capped" shape). Deliberately
# a SEPARATE constant rather than broadening `ALPHA_AMINO_ACID_SMARTS`
# itself: that constant also gates `detect_amino_acid`/`name_amino_acid`,
# the STANDALONE free-amino-acid systematic namer, which assumes a genuine
# `[OX2H1]` and would mis-name a standalone amino-acid AMIDE (not a peptide
# at all) if broadened -- this new pattern is consulted ONLY by
# `is_peptide()` below, so that unrelated path is untouched.
_TERMINAL_PRIMARY_AMIDE_ALPHA_SMARTS = "[NX3;H2,H1,H0][CX4][CX3](=O)[NX3H2]"

# (backbone-substitutive ester C-terminus): a peptide whose C-TERMINAL
# residue's alpha-carboxyl is esterified (-C(=O)-O-C, e.g. a Leu-Leu methyl
# ester) has NO free -COOH and NO primary carboxamide anywhere, so neither
# ALPHA_AMINO_ACID_SMARTS (hard-requires `[OX2H1]`) nor
# _TERMINAL_PRIMARY_AMIDE_ALPHA_SMARTS (requires `[NX3H2]`) matches, and
# `is_peptide()` returned False -- so the dispatcher never reached
# `rules.peptides.name_peptide`, even though that module's
# `_try_backbone_substitutive` now names the ester-C-terminus parent
# systematically ('methyl (2S)-2-[(2S)-2-amino-4-methylpentanamido]-4-
# methylpentanoate'). This is the exact structural analogue of the
# primary-amide alternative above. Scoped to an ALPHA-amino-acid ester
# (`[NX3][CX4]` on the ester carbonyl) so it does NOT broadly re-route
# ordinary esters; combined with the `count_peptide_bonds(mol) >= 1`
# requirement in `is_peptide()`, an ester with no peptide linkage (a plain
# amino-acid ester, not a peptide) still does not dispatch here. Like the
# amide constant it is consulted ONLY by `is_peptide()`, so the standalone
# free-amino-acid namer's `ALPHA_AMINO_ACID_SMARTS` path is untouched.
_TERMINAL_ESTER_ALPHA_SMARTS = "[NX3;H2,H1,H0][CX4][CX3](=O)[OX2][CX4]"

# SMARTS patterns for additional functional groups (PEP-02 bailout)
# These detect FGs beyond amino + acid that _name_amino_acid_systematic
# cannot handle. When found, we bail out to the general polyfunctional pipeline.
_EXTRA_FG_SMARTS = [
    '[OX2H1;!$([OX2H1]C=O)]',  # Hydroxy OH (not in COOH)
    '[SX2H1]',                   # Thiol SH
    # (F-THIOETHER-DROP, BB P-103 audit F1): a thioether -S- (both
    # neighbours carbon) is NOT a chain atom — the carbon-count systematic namer
    # would count the S-alkyl carbon(s) as backbone and silently DROP the sulfur
    # (methionine homologues / S-alkyl-cysteines named as plain '2-aminoalkanoic
    # acid' = a different molecule). Bail to the polyfunctional pipeline, which
    # emits the (alkylsulfanyl) prefix (2-amino-3-(ethylsulfanyl)propanoic acid).
    # Disulfides (S has an S neighbour) and ring S (rings already bail above) are
    # excluded by the two-carbon-neighbour requirement.
    # BP-3 C1 (P-63.1.5): the Se/Te ether analogues have the SAME defect — the
    # carbon-count backbone walks through -Se-/-Te- and drops the chalcogen
    # (C[Se]CC(N)C(=O)O was named '2-aminobutanoic acid', a different molecule).
    # Bail so the polyfunctional pipeline emits the (methylselanyl)/(methyltellanyl)
    # prefix (verified: 2-amino-3-(methylselanyl)propanoic acid).
    '[SX2]([#6])[#6]',           # Thioether C-S-C
    '[SeX2]([#6])[#6]',          # Selenoether C-Se-C (P-63.1.5 selanyl)
    '[TeX2]([#6])[#6]',          # Telluroether C-Te-C (P-63.1.5 tellanyl)
    '[F,Cl,Br,I]',               # Halogen
    '[N+](=O)[O-]',              # Nitro
    '[CX3;!$([CX3](=O)[OX2H1]);!$([CX3](=O)[NX3])](=O)',  # Ketone C=O (excludes acid and amide C=O)
    '[CX3](=O)[NX3H2]',         # Primary amide -C(=O)NH2 (e.g., glutamine side chain)
    '[CX2]#[NX1]',              # Nitrile -C#N (e.g., cyano-amino acids)
]


def count_peptide_bonds(mol) -> int:
    """
    Count the number of peptide bonds (-C(=O)-NH-CH-) in a molecule.

    Peptide bonds link amino acid residues. A molecule with >= 1 peptide bond
    is a peptide, not a simple amino acid.

    Args:
        mol: RDKit Mol object

    Returns:
        Number of peptide bond matches found
    """
    pattern = Chem.MolFromSmarts(PEPTIDE_BOND_SMARTS)
    if pattern is None:
        return 0
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)


def is_peptide(mol) -> bool:
    """
    Check if molecule contains peptide bonds (is a di/tri/polypeptide).

    A molecule with one or more -C(=O)-NH-CH- linkages is a peptide,
    not a simple amino acid. Must also have the amino acid pattern
    (terminal NH2 + COOH) to distinguish from random amides.

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule is a peptide (has amino acid pattern AND peptide bonds)
    """
    # Must have amino acid pattern AND peptide bonds..1: OR in the
    # terminal-primary-amide alternative (see the constant's docstring) so a
    # capped-C-terminus peptide still dispatches to `rules.peptides` instead
    # of silently falling through to "unknown organic compound".: OR in the
    # ester-C-terminus alternative too, for the same reason (a '...oate'
    # C-terminus that _try_backbone_substitutive now names systematically).
    amide_pattern = Chem.MolFromSmarts(_TERMINAL_PRIMARY_AMIDE_ALPHA_SMARTS)
    has_amide_alpha = amide_pattern is not None and mol.HasSubstructMatch(amide_pattern)
    ester_pattern = Chem.MolFromSmarts(_TERMINAL_ESTER_ALPHA_SMARTS)
    has_ester_alpha = ester_pattern is not None and mol.HasSubstructMatch(ester_pattern)
    if not (detect_amino_acid(mol) or has_amide_alpha or has_ester_alpha):
        return False
    return count_peptide_bonds(mol) >= 1


def detect_amino_acid(mol) -> bool:
    """
    Check if molecule is an amino acid.

    Detects alpha-amino acid pattern: NH2-CH(R)-COOH

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule contains alpha-amino acid pattern
    """
    pattern = Chem.MolFromSmarts(ALPHA_AMINO_ACID_SMARTS)
    if pattern is None:
        return False
    return mol.HasSubstructMatch(pattern)


def get_amino_acid_atoms(mol) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Get atom indices of alpha-amino acid core.

    The SMARTS pattern [NX3;H2,H1][CX4][CX3](=O)[OX2H1] matches:
    - N (amino nitrogen)
    - alpha_C (alpha carbon attached to N)
    - carbonyl_C (carboxylic acid carbon)
    - carbonyl_O (=O oxygen)
    - acid_O (OH oxygen)

    Returns:
        Tuple of (N, alpha_C, carbonyl_C, carbonyl_O, acid_O) atom indices, or None
    """
    pattern = Chem.MolFromSmarts(ALPHA_AMINO_ACID_SMARTS)
    if pattern is None:
        return None

    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return None

    # Return first match (5 atoms)
    return matches[0]


def _has_extra_functional_groups(mol) -> bool:
    """
    Check if molecule has functional groups beyond amino + carboxylic acid.

    PEP-02: When additional FGs are detected (hydroxy, thiol, halogen, nitro,
    ketone), _name_amino_acid_systematic should bail out and let the general
    polyfunctional pipeline handle the molecule, since it correctly handles
    multi-FG chains with proper prefix ordering, locant assignment, and
    alphabetization.

    Args:
        mol: RDKit Mol object

    Returns:
        True if extra functional groups found
    """
    for smarts in _EXTRA_FG_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None and mol.HasSubstructMatch(pat):
            return True
    return False


def name_amino_acid(mol, canonical_smiles: str) -> Optional[str]:
    """
    Generate name for an amino acid.

    For standard amino acids, returns trivial name.
    For non-standard, returns systematic name.

    -02: SMILES lookup is checked BEFORE the alpha-amino acid SMARTS
    gate. This allows non-alpha amino acids (taurine, creatine, etc.) that are
    in the OPSIN vocabulary to be named correctly via direct SMILES lookup.

    Peptide guard: if the molecule contains peptide bonds (-C(=O)-NH-),
    it is NOT a simple amino acid. Returns None so the molecule falls
    through to the general naming pipeline (or returns None for
    complex peptides beyond current scope).

    Args:
        mol: RDKit Mol object
        canonical_smiles: Canonical SMILES of the molecule

    Returns:
        Amino acid name, or None if not an amino acid or is a peptide
    """
    # PEPTIDE GUARD: check for peptide bonds before naming as amino acid
    # Must run before SMILES lookup to prevent peptides with amino acid substrings
    n_peptide_bonds = count_peptide_bonds(mol)
    if n_peptide_bonds >= 1:
        return None

    # 141-02: Try SMILES lookup BEFORE the SMARTS pattern gate.
    # Many OPSIN amino acid entries (taurine, creatine, etc.) don't match the
    # alpha-amino acid SMARTS but are still valid amino acid trivial names.
    # WSD-07: stereo-aware retained-name lookup — the stereo-strip fallback finds
    # CIP-tagged standard AAs (which miss the stereo-free keys) and emits the L/D
    # configurational descriptor (P-103.1.1.1: L implicit, D explicit); a
    # diastereomer the bare name cannot represent (e.g. allo-isoleucine) returns
    # None and falls through to the systematic namer. This replaces the prior
    # whole-graph CIP-prefix injection (which would emit a non-PIN '(2S)alanine').
    #
    # GATED to TOP-LEVEL naming: at decomposition depth (a peptide/conjugate
    # fragment re-entering naming) the descriptor path is OFF so the call is
    # byte-identical to the original stereo-free exact lookup — this preserves the
    # decomposition/peptide assembly that relies on the prior None-on-stereo-miss
    # behavior (e.g. the prolyl-glutaminyl-cysteine tripeptide must stay a peptide,
    # not a systematic monomer).
    from ..assembly.fragment_naming import is_top_level_naming
    trivial = get_amino_acid_name(
        canonical_smiles, mol=mol, with_descriptor=is_top_level_naming(),
    )
    if trivial:
        # (C2b) STEREO HONESTY: the blanket policy is TOP-LEVEL naming +
        # an undefined real stereocentre -> decline the retained AA name and fall
        # through to the stereo-free systematic name below. Config-fabrication is the
        # PRIMARY reason (a retained AA name implies a specific configuration -- L,
        # unless D-cited -- and asserting one for an input that leaves it undefined
        # fabricates stereo the input never defined, P-92/P-93), but it is not the
        # SOLE one: this same guard also demotes retained names that do not imply a
        # configuration at all, e.g. butyrine (achiral at its would-be alpha centre in
        # some substitution patterns), whenever the top-level input still leaves a
        # real stereocentre elsewhere undefined -- the condition below is a blanket
        # top-level+undefined-stereocentre check, not a per-name config-fabrication
        # test. Scoped to top-level: fragment decomposition keeps the bare name
        # (peptides.py owns that path's guard). Achiral AAs (glycine) have no
        # undefined centre.
        from ..perception.stereo import input_stereo_undefined
        if not (is_top_level_naming() and input_stereo_undefined(mol)):
            return trivial

    # Check if it matches the alpha-amino acid pattern for systematic naming
    if not detect_amino_acid(mol):
        return None

    # Generate systematic name
    return _name_amino_acid_systematic(mol)


def _verify_denoted_alkanoic_backbone(mol, aa_atoms) -> Optional[int]:
    """Prove `mol` IS the structure `2-amino{stem}anoic acid` denotes.

    That name denotes exactly one constitution::

        HOOC-CH(NH2)-(CH2)n-CH3

    i.e. an unbranched, fully saturated, all-carbon backbone running from the acid
    carbon to a terminal methyl, carrying one terminal -NH2 on C2 and no other
    heteroatom, unsaturation, charge or isotope anywhere in the molecule.

    This is an ALLOW-LIST. It replaces a whole-molecule carbon count that could not
    distinguish a chain carbon from a carbon on the far side of a contracted
    heteroatom, and it subsumes every entry of the historical deny-list above
    (thioether/selenoether/telluroether drop, hydroxy, halogen, nitro, ketone,
    amide, nitrile, branching, rings, polyamine, dicarboxylic).

    Args:
        mol: RDKit Mol object.
        aa_atoms: the 5-tuple from:func:`get_amino_acid_atoms` --
            (N, alpha_C, carbonyl_C, carbonyl_O, acid_O).

    Returns:
        The number of backbone carbons (== the stem length) if the molecule is
        provably the denoted structure, else None.
    """
    n_idx, alpha_c, carbonyl_c, carbonyl_o, acid_o = aa_atoms

    # --- composition: only C, N, O; no charges, no isotopes, no radicals --------
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ('C', 'N', 'O'):
            return None
        if atom.GetFormalCharge() != 0:
            return None
        if atom.GetIsotope() != 0:
            return None
        if atom.GetNumRadicalElectrons() != 0:
            return None

    # --- exactly one nitrogen, and it is the terminal alpha -NH2 ---------------
    nitrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'N']
    if nitrogens != [n_idx]:
        return None
    n_atom = mol.GetAtomWithIdx(n_idx)
    if n_atom.GetTotalNumHs() != 2 or n_atom.GetDegree() != 1:
        return None
    if n_atom.GetNeighbors()[0].GetIdx() != alpha_c:
        return None

    # --- exactly two oxygens, and both belong to the one COOH ------------------
    oxygens = {a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'O'}
    if oxygens != {carbonyl_o, acid_o}:
        return None
    for o_idx in (carbonyl_o, acid_o):
        o_atom = mol.GetAtomWithIdx(o_idx)
        if o_atom.GetDegree() != 1 or o_atom.GetNeighbors()[0].GetIdx() != carbonyl_c:
            return None

    # --- saturation: every bond single except the single carbonyl C=O ----------
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE:
            continue
        ends = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
        if bond.GetBondType() == Chem.BondType.DOUBLE and ends == {carbonyl_c, carbonyl_o}:
            continue
        return None  # C=C, C#C, aromatic, or a second C=O

    # --- the carbon skeleton is one unbranched path covering EVERY carbon ------
    # Walk from the acid carbon through the alpha carbon to a terminal methyl.
    def carbon_neighbours(idx):
        return [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
                if nbr.GetSymbol() == 'C']

    if carbon_neighbours(carbonyl_c) != [alpha_c]:
        return None  # acid carbon must be a chain terminus attached to C2

    path = [carbonyl_c, alpha_c]
    prev, current = carbonyl_c, alpha_c
    while True:
        onward = [c for c in carbon_neighbours(current) if c != prev]
        if not onward:
            break  # terminal carbon reached
        if len(onward) > 1:
            return None  # branch point -- not an unbranched chain
        prev, current = current, onward[0]
        if current in path:
            return None  # defensive: a carbocycle (rings already bail above)
        path.append(current)

    all_carbons = {a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'C'}
    if set(path) != all_carbons:
        return None  # carbons exist off the backbone -> something was contracted

    return len(path)


def _name_amino_acid_systematic(mol) -> str:
    """
    Generate systematic IUPAC name for non-standard amino acid.

    Uses: "amino" prefix + acid name
    Example: 2-aminopropanoic acid (systematic for alanine)

    For amino acids with multiple amino groups (e.g., lysine),
    returns None to let the general pipeline handle it.

    PEP-02: Also bails out when additional functional groups are detected
    (hydroxy, thiol, halogen, nitro, ketone) since the general polyfunctional
    pipeline handles multi-FG chains correctly.
    """
    from ..data.chain_names import get_chain_prefix

    # Get amino acid core atoms
    aa_atoms = get_amino_acid_atoms(mol)
    if not aa_atoms:
        return None

    # Count primary amine groups - if more than one, let general pipeline handle
    amine_pattern = Chem.MolFromSmarts('[NX3;H2;!$([NX3][CX3]=O)]')
    if amine_pattern:
        amine_matches = mol.GetSubstructMatches(amine_pattern)
        if len(amine_matches) > 1:
            return None  # Multiple amines - use general naming pipeline

    # AMAC-01: Multiple COOH groups -> polyfunctional pipeline
    # Dicarboxylic amino acids (aspartic, glutamic) need "dioic acid" suffix
    # which the specialized handler can't produce (it hardcodes mono-acid).
    cooh_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    if cooh_pattern:
        cooh_matches = mol.GetSubstructMatches(cooh_pattern)
        if len(cooh_matches) >= 2:
            return None  # Dicarboxylic - use polyfunctional pipeline

    # If the molecule has rings, the simple carbon-count approach would
    # include ring carbons in the chain length (e.g., tyrosine would give
    # "2-aminononanoic acid" instead of falling through to the general
    # pipeline which correctly uses parent selection with ring-atom exclusion).
    ri = mol.GetRingInfo()
    if ri.NumRings() > 0:
        return None  # Let general pipeline handle ring-containing amino acids

    # PEP-02: Check for additional functional groups beyond amino and acid.
    # If present, bail out to general polyfunctional pipeline which handles
    # multi-FG chains correctly with proper prefix ordering and locants.
    if _has_extra_functional_groups(mol):
        return None

    # (P-103.2.3 branched systematic AA, BB audit): the carbon-count
    # stem below assumes a SINGLE unbranched chain (it names every carbon as
    # backbone -> '2-amino{stem}anoic acid'). A branched side chain (a carbon with
    # >2 carbon neighbours, e.g. CC[C@H](C)[C@@H](N)C(=O)O = a 3-methylpentanoic
    # α-amino acid) collapses into a too-long straight chain ('2-aminohexanoic
    # acid' — a DIFFERENT, OPSIN-unparseable molecule that then leaks past the
    # parse-fail-open self-consistency gate). Bail to the polyfunctional pipeline,
    # which selects the parent chain + 3-methyl prefix correctly
    # (2-amino-3-methylpentanoic acid). Straight-chain homologues (Abu/norvaline/
    # norleucine) have no carbon with >2 carbon neighbours and are unaffected.
    if any(sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'C') > 2
           for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'):
        return None  # branched side chain -> general pipeline

    # TASK-I (root cause): the stem used to come from a whole-molecule carbon count
    # carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    # guarded by the DENY-LIST above. A carbon count cannot see whether those carbons
    # are actually one unbranched saturated chain, so every heteroatom or unsaturation
    # the deny-list had not yet enumerated was silently CONTRACTED -- its flanking
    # carbons counted into the backbone and the atom itself dropped from the name.
    # Three holes were open simultaneously, each emitting a DIFFERENT MOLECULE:
    # CNCC(=O)O sarcosine -> '2-aminopropanoic acid' (= alanine)
    # COCC(N)C(=O)O O-Me-serine-> '2-aminobutanoic acid' (ether O dropped)
    # C=CCC(N)C(=O)O allylgly -> '2-aminopentanoic acid' (= norvaline)
    # and CNCCNCCNCC(=O)O dropped TWO nitrogens. A deny-list can only exclude the
    # holes already found; it is the wrong shape for a fail-closed namer.
    #
    # The name this function emits, '2-amino{stem}anoic acid', denotes exactly one
    # constitution: HOOC-CH(NH2)-(CH2)n-CH3. So PROVE the molecule is that, and
    # derive the stem from the proven backbone. Anything else returns None and goes
    # to the general polyfunctional pipeline, which names it correctly.
    backbone_length = _verify_denoted_alkanoic_backbone(mol, aa_atoms)
    if backbone_length is None:
        return None

    # Get stem from the PROVEN backbone length using centralized module
    stem = get_chain_prefix(backbone_length)

    # For alpha-amino acids, the amino group is at position 2
    # (position 1 is the acid carbon)
    name = f"2-amino{stem}anoic acid"

    # STER-09: Inject CIP stereodescriptors (self-contained, runs before perception)
    from ..perception.stereo import assign_stereochemistry
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    assign_stereochemistry(mol)
    locant_map = _build_amino_acid_locant_map(mol)
    descriptors = collect_stereodescriptors(mol, locant_map)
    if descriptors:
        stereo_prefix = format_stereodescriptor_string(descriptors)
        name = f"{stereo_prefix}{name}"

    return name


def _build_amino_acid_locant_map(mol) -> dict:
    """Build atom_to_locant map for amino acid backbone.

    Locant 1 = acid carbon (C(=O)O), then walk along the carbon backbone.
    For alpha-amino acids, the stereocenter is at locant 2.

    Args:
        mol: RDKit Mol object.

    Returns:
        Dict mapping atom index to IUPAC locant number.
    """
    from collections import deque

    # Find the acid carbon: a carbon with a C=O double bond and C-OH single bond
    acid_c = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C':
            continue
        has_double_o = False
        has_single_o = False
        for bond in atom.GetBonds():
            nbr = bond.GetOtherAtom(atom)
            if nbr.GetSymbol() == 'O' and bond.GetBondTypeAsDouble() == 2.0:
                has_double_o = True
            elif nbr.GetSymbol() == 'O' and bond.GetBondTypeAsDouble() == 1.0:
                has_single_o = True
        if has_double_o and has_single_o:
            acid_c = atom.GetIdx()
            break

    if acid_c is None:
        return {}

    # BFS from acid carbon through the carbon backbone
    atom_to_locant = {acid_c: 1}
    visited = {acid_c}
    queue = deque([(acid_c, 1)])

    while queue:
        current, locant = queue.popleft()
        atom = mol.GetAtomWithIdx(current)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in visited:
                continue
            if nbr.GetSymbol() == 'C':
                next_locant = locant + 1
                atom_to_locant[nbr_idx] = next_locant
                visited.add(nbr_idx)
                queue.append((nbr_idx, next_locant))
            else:
                visited.add(nbr_idx)

    return atom_to_locant


def is_n_substituted_amino_acid(mol) -> bool:
    """
    Check if amino acid has N-substituents (like sarcosine = N-methylglycine).
    """
    # Look for secondary amine in amino acid context
    pattern = Chem.MolFromSmarts("[NX3;H1]([CX4])[CX4][CX3](=O)[OX2H1]")
    if pattern is None:
        return False
    return mol.HasSubstructMatch(pattern)
