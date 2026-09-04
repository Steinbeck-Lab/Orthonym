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

, phase2e SPY --):
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

: P-103.1.3.1 "The stereodescriptors 'D'
and 'L'" (BlueBookV2.md:54291) -- "The stereodescriptor 'xi' (Greek letter
xi) indicates unknown configuration" -- and P-103.3.4 itself (:54715) --
"A residue of unknown configuration is indicated by the prefix xi". A bare
retained residue name is therefore NOT a safe default for an undefined
alpha-carbon: the flat acylamino path (``_identify_residues`` with
``strict_stereo=True``, via ``_alpha_stereo_undefined``) now DECLINES the
whole chain rather than silently applying the L-omission convention
(P-103.3.4) to a residue whose configuration the input never defined. Lever
B's donor-residue recovery is deliberately exempt: it builds its own
locant-based descriptor and already treats an unresolved centre as "omit the
descriptor" with no implied configuration.

: Task 2.0 (dispatch SMARTS fix) + Task 2.1 (Lever C, capped
termini -- ``_try_capped_termini``) closed the two largest decline buckets.
Task 2.2 adds ``_try_backbone_substitutive``, the general BACKBONE-
SUBSTITUTIVE producer for the residual (giant / non-standard-residue
chains): the C-terminal residue's own free acid is the parent, and every
other residue is folded in one at a time (PREPEND for a standard/closed
acyl word, SPLICE via ``_swap_amino_for_amido`` when a non-standard
residue's own systematic acid name leaves a leading 'amino' locant open).
Tried LAST, after every path above declines; gated by ``_rt_verified`` like
every other lever here, so an out-of-scope shape (a capped/dual-acid
C-terminus, or a genuine side-chain acid on a non-parent residue -- the
"side-chain-acid trap") ABSTAINS rather than misplace an amidation.
"""

import re
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from ..data.amino_acids import (
    STANDARD_AMINO_ACIDS,
    get_amino_acid_acyl_name,
    get_amino_acid_name,
)
from ..perception.stereo import assign_stereochemistry

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
        # Lever A: an acylated (non-free) N-terminus may still be a
        # nameable N-acyl-capped peptide (SPY §3a). Fails closed to None
        # (unchanged behaviour) unless the cap is a genuine plain acyl group
        # AND the residues behind it are a fully standard >=2-residue chain
        # AND the assembled candidate OPSIN-round-trips to this exact mol.
        result = _rt_verified(mol, _try_n_acyl_cap(mol))
        if result is not None:
            return result
        # Task 2.1 Lever C: a capped-terminus shape `_is_valid_peptide`
        # rejects for a DIFFERENT reason than Lever A covers -- most often a
        # C-terminal PRIMARY AMIDE (no free -COOH anywhere in the molecule,
        # so `_is_valid_peptide`'s COOH check fails and Lever A's own COOH
        # requirement, `:871`, also fails). Self-contained (re-walks the
        # chain itself via `_extract_residues`), so safe to try here too.
        result = _rt_verified(mol, _try_capped_termini(mol))
        if result is not None:
            return result
        # Task 2.2: the general backbone-substitutive producer -- also
        # self-contained (re-walks via `_extract_residues`), tried last.
        return _rt_verified(mol, _try_backbone_substitutive(mol))

    # Step 2: Extract residue SMILES by walking the peptide chain
    residue_smiles_list = _extract_residues(mol)
    if residue_smiles_list is None or len(residue_smiles_list) < 2:
        # Lever B: a single non-alpha ('gamma'/'beta') bond from a
        # standard amino acid's own side-chain carboxyl (SPY §3b). Same
        # fail-closed-to-None-unless-round-tripped guarantee as Lever A.
        return _rt_verified(mol, _try_gamma_link_whole(mol))

    # Step 3: Identify residues and get stereo prefixes. strict_stereo=True:
    # this feeds the FLAT retained-name convention (Step 4), where an omitted
    # descriptor is read as an implicit 'L' (P-103.3.4) -- see
    # ``_identify_residues``'s docstring.
    named_residues = _identify_residues(residue_smiles_list, strict_stereo=True)
    if named_residues is None:
        # Task 2.1 Lever C: the whole-molecule gate passed (free termini
        # SMARTS matched -- a mono-N-methylated N-terminus is still `[NX3;H1]`,
        # so it slips through `_is_valid_peptide`'s free-NH2 check), but a
        # per-residue trivial-name lookup then failed -- typically the
        # N-terminal residue's OWN backbone amino N carrying a free (non-
        # acylated) methyl substituent, a table miss for the modified
        # residue. Self-contained; also independently catches a
        # C-terminal-amide shape reaching this branch some other way.
        result = _rt_verified(mol, _try_capped_termini(mol))
        if result is not None:
            return result
        # Task 2.2: the general backbone-substitutive producer -- also
        # self-contained, tried last (e.g. a non-standard/undefined-stereo
        # residue that made `_identify_residues` decline here).
        return _rt_verified(mol, _try_backbone_substitutive(mol))

    # Step 4: Assemble the peptide name. RT-gated: this
    # was the one candidate-emission site in this file NOT verified against
    # the input mol -- every breadth lever (A/B, below) already routes
    # through `_rt_verified`, but the ORIGINAL flat/acylamino-convention path
    # never did, because before Task 2.0 the dispatch predicate
    # (`amino_acids.is_peptide`) silently protected it from ever reaching a
    # C-terminal cyclic-imino-acid (proline) chain at all. Unblocking that
    # dispatch (Task 2.0's SMARTS fix) now routes 9 more molecules here, and
    # the Phase-0 SPY measured 1/9 of them assembles a WRONG name (an
    # 11-residue chain with an internal Gln + Asp -- likely a stereo/CIP
    # mis-mapping), caught only incidentally by a downstream gate today.
    # `_rt_verified` fails CLOSED on any mismatch/parse-error/no-OPSIN, so a
    # wrong flat candidate now ABSTAINS instead of shipping; every ordinary
    # peptide this path already named correctly continues to round-trip and
    # is therefore unaffected in substance (0-wrong ABSOLUTE).
    candidate = _assemble_peptide_name(named_residues)
    result = _rt_verified(mol, candidate)
    if result is not None:
        return result
    # Task 2.2: last resort when the flat candidate itself failed
    # round-trip (a rare pre-existing bug elsewhere in this file, e.g. a
    # mis-mapped stereo prefix) -- self-contained, re-walks independently.
    return _rt_verified(mol, _try_backbone_substitutive(mol))


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
    *,
    strict_stereo: bool = False,
) -> Optional[List[Dict[str, str]]]:
    """
    Identify each residue by trivial name and determine stereo prefix.

    Returns a list of dicts with keys:
        - 'name': trivial amino acid name (e.g., "glycine")
        - 'acyl': acyl form (e.g., "glycyl")
        - 'stereo': "L-", "D-", or "" (empty for achiral)

    Returns None if any residue cannot be identified as a standard amino acid.

    ``strict_stereo`` (default False -- byte-identical to before this
    parameter existed): when True, ALSO decline (return None) if any
    residue's alpha-carbon configuration is a genuine stereocentre left
    UNDEFINED by the input (see ``_alpha_stereo_undefined``). Scoped to
    callers that feed the result into ``_assemble_peptide_name`` -- the flat
    acylamino/bare-retained-name convention where an omitted descriptor is
    read as an IMPLICIT 'L' (P-103.3.4). Lever B's donor-residue recovery
    (``_try_gamma_donor_link``) deliberately leaves this False: it builds its
    OWN systematic, locant-based stereodescriptor and already treats an
    empty ``stereo`` as "omit the descriptor" with no implied configuration
    (unlike the retained-name convention), so no double-guard is needed
    there.
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

        # STEREO HONESTY (P-103.1.3.1 "The stereodescriptors 'D' and 'L'",
        # BlueBookV2.md:54291, + P-103.3.4 "Indication of configuration in
        # peptides", :54715): a bare retained residue name asserts a SPECIFIC
        # configuration -- P-103.3.4's own text: "A residue of unknown
        # configuration is indicated by the prefix xi (Greek letter xi)".
        # Orthonym does not emit xi-prefixed names, so a residue whose
        # alpha-carbon configuration the INPUT does not define must not be
        # silently folded into the omitted-L convention (that is exactly the
        # implicit-L fabrication this check exists to stop) -- decline the
        # whole peptide instead, so the caller falls through to a systematic/
        # stereo-free name (or an honest abstention). Glycine has no
        # alpha-stereocentre, so it is exempt (never "undefined"). Gated on
        # ``strict_stereo`` -- see the docstring for why Lever B's donor
        # recovery must NOT set it.
        if strict_stereo and _alpha_stereo_undefined(mol, aa_name):
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


def _alpha_stereo_undefined(mol: Chem.Mol, aa_name: str) -> bool:
    """True iff ``aa_name`` (a real, non-achiral standard amino acid) has an
    alpha-carbon stereocentre in ``mol`` whose configuration the INPUT does
    not define -- i.e. no wedge/parity (``CHI_UNSPECIFIED``) at that atom.

    Every standard amino acid other than glycine has four distinct groups at
    its alpha-carbon (H, NH2, COOH-carbon, a distinct side chain), so it is
    ALWAYS a genuine stereocentre; glycine is the sole achiral exception and
    is excluded by name up front. A ``True`` result means: naming this
    residue with its bare retained name (which asserts L, or D if flagged --
    P-103.1.3.1) would assert a configuration the structure does not have.

    Root cause this guards: ``_get_stereo_prefix`` looks up ``_CIPCode`` and
    falls back to ``""`` whenever it is absent, whether that is because the
    residue is TRULY achiral (glycine) or because a genuine stereocentre's
    configuration was simply never specified. P-103.3.4's L-omission display
    rule then makes that "" indistinguishable from a defined-and-omitted L --
    so a flat, stereo-free input silently acquired an implied L it does not
    define. This check runs BEFORE that omission logic and answers only the
    question the omission rule cannot: was a real stereocentre left
    unspecified at all. If the SMARTS locator itself fails, that also means
    "cannot confirm the configuration is defined" -> True (fail closed).
    """
    if aa_name == "glycine":
        return False
    alpha_pattern = Chem.MolFromSmarts("[NX3][CX4][CX3](=O)")
    if alpha_pattern is None:
        return True
    matches = mol.GetSubstructMatches(alpha_pattern)
    if not matches:
        return True
    alpha_c_idx = matches[0][1]
    alpha_atom = mol.GetAtomWithIdx(alpha_c_idx)
    return alpha_atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED


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
# breadth: N-acyl cap (Lever A) + gamma/beta side-chain-carboxyl
# donor (Lever B). Both are ADDITIVE fallbacks tried only after the ordinary
# flat acylamino convention above declines, and both are gated by
# ``_rt_verified`` -- a table/topology miss degrades to abstention, never a
# wrong or atom-dropping name. See phase2e SPY for the derivation and the
# corpus-scale sizing.
# ============================================================================


def _rt_verified(mol, candidate: Optional[str]) -> Optional[str]:
    """0-wrong guarantee for the broadened paths (Lever A / Lever B).

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
    # strict_stereo=True: both branches below feed ``_assemble_peptide_name``,
    # the flat retained-name convention where an omitted descriptor implies
    # 'L' (P-103.3.4) -- see ``_identify_residues``'s docstring.
    if residues is not None and len(residues) >= 2:
        named = _identify_residues(residues, strict_stereo=True)
        if named is None:
            return None
        return _assemble_peptide_name(named)
    named1 = _identify_residues([smi], strict_stereo=True)
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
    # strict_stereo=True: feeds ``_assemble_peptide_name`` (see its docstring).
    named = _identify_residues(residues, strict_stereo=True)
    if named is None:
        return None
    remainder_name = _assemble_peptide_name(named)

    from ..errors import is_failure_name
    from ..namer import Orthonym
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

# The leading, ADDRESSABLE 'amino' substituent prefix of a parent/residue acid
# (or ester) name, anchored at the start, e.g. the 'amino' in
# '1-aminocyclopropane-1-carboxylic acid'. Substituent prefixes FUSE directly
# onto the parent hydride name with no space/hyphen ('aminocyclopropane', not
# 'amino-cyclopropane'), so a token search for a word-bounded 'amino' never
# matches -- anchoring at position 0 instead only fires when 'amino' is the
# name's SOLE (or alphabetically-first) substituent, i.e. exactly the
# free-NH2-only residue shape this lever targets. In front of that token the
# name may legitimately carry (all optional, in this order): an ester O-alkyl
# functional-class word ('methyl '), a stereodescriptor ('(2S)-', '(2S,3R)-'),
# and a locant ('2-'). A defined-stereo systematic acid name
# ('(2S)-2-aminohexanoic acid') or a systematic ester
# ('methyl (2S)-2-amino-4-methylpentanoate') is therefore spliced at its REAL
# 'amino' token with everything before it preserved (P-16.5). ``head`` is that
# preserved prefix; the match end is the position just after 'amino'.
_AMINO_PREFIX_RE = re.compile(
    r"^(?P<head>(?:[a-z]+yl )?(?:\([0-9RSEZ,a-z'*]+\)-)?(?:\d+-)?)amino"
)


def _match_amino_prefix(name: str) -> Optional[Tuple[str, int]]:
    """Return ``(head, tail_index)`` for a leading addressable 'amino'
    substituent prefix in ``name`` (see ``_AMINO_PREFIX_RE``), or None. ``head``
    is everything up to and including any ester O-alkyl word / stereodescriptor
    / locant that precedes the 'amino' token; ``name[tail_index:]`` is what
    follows 'amino'. Used both to DECIDE whether a systematic parent name is
    still 'open' for a further residue to splice into, and to PERFORM that
    splice preserving the ester/stereo context."""
    m = _AMINO_PREFIX_RE.match(name)
    if m is None:
        return None
    return m.group('head'), m.end()


def _swap_amino_for_amido(name: str, amido_group: str) -> Optional[str]:
    """Replace the leading 'amino' substituent prefix in ``name`` with the
    enclosed ``amido_group`` at the same locant (P-16.3.3 complex-substituent
    enclosure), e.g. '1-aminocyclopropane...' + '4-amino-4-carboxybutanamido'
    -> '1-(4-amino-4-carboxybutanamido)cyclopropane...', and (with a stereo
    descriptor on both sides) '(2S)-2-aminohexanoic acid' +
    '(2S)-2-aminopropanamido' -> '(2S)-2-[(2S)-2-aminopropanamido]hexanoic
    acid'. The enclosing marks escalate ( -> [ -> { when ``amido_group`` itself
    already carries brackets (a stereodescriptor's parentheses), per
    P-16.5.4.1. Declines (returns None) when ``name`` has no addressable
    leading 'amino' -- ambiguous, so never guessed."""
    matched = _match_amino_prefix(name)
    if matched is None:
        return None
    head, tail = matched
    from ..assembly.naming_utils import apply_enclosing_marks
    return head + apply_enclosing_marks(amido_group, -1) + name[tail:]


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

    from ..errors import is_failure_name
    from ..namer import Orthonym
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

    from ..assembly.substituent_naming import acyl_carbons_to_amido_prefix
    from ..data.chain_names import get_chain_prefix
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


# ============================================================================
# Task 2.1: Lever C -- capped termini. Extends the flat acylamino
# convention to accept two shapes the ordinary path declines on (SPY:
# sec.3 -- 71% of the true-
# peptide backlog is exactly one or both of these, the single dominant
# blocker):
#   - a C-TERMINAL PRIMARY AMIDE (``-C(=O)NH2``) instead of the free
#     ``-COOH`` the acylamino convention assumes -- rendered as the standard
#     amino-acid-amide suffix ("...amide", e.g. ``glycinamide``).
#   - a mono-N-METHYLATED (free, non-acylated) N-terminus -- rendered as an
#     "N-methyl" substituent prefix on the whole assembled name (the same
#     convention as the well-known ``N-methyl-D-aspartic acid``).
# Both caps are stripped from an ISOLATED COPY of just the one residue
# fragment they live on (never the whole molecule) -- ``_extract_residues``
# already walks the backbone and hands back each residue's OWN reconstructed
# free-amino-acid SMILES untouched at the true termini, so no whole-molecule
# surgery is needed. The stripped fragment is looked up via the EXACT SAME
# trivial-name table as every other residue; the cap is then re-applied
# TEXTUALLY at exactly the token position it structurally came from --
# mutating the RESIDUE DATA that feeds ``_assemble_peptide_name`` (the
# C-terminal residue's ``name`` field) or prepending to the assembled
# string's own start (the N-terminal token, position 0) -- never splicing an
# already-assembled name blindly. ``_rt_verified`` (full-InChI OPSIN
# round-trip) gates every candidate this lever returns, so a mis-rendering
# degrades to abstention, never a wrong name (0-wrong ABSOLUTE).
# ============================================================================


# A lone, free (non-acylated) methyl directly on a secondary amino nitrogen
# that is itself bonded to an sp3 alpha-carbon -- the shape of a MONO-N-
# methylated amino-acid N-terminus (e.g. sarcosine's own N, or N-
# methylalanine's). Applied only to an already-ISOLATED single-residue
# fragment (see module docstring above), so no additional "is this really
# the N-terminus" disambiguation is needed -- `_extract_residues` has
# already done that work.
_MONO_N_METHYL_SMARTS = "[CH3;D1][NX3;H1;!$([NX3][CX3]=O)][CX4]"

# A genuine, unsubstituted primary carboxamide -- ``-C(=O)NH2``. Applied
# only to an already-isolated single-residue fragment; a standard residue
# with its OWN side-chain primary amide (asparagine, glutamine) matches this
# TWICE (once for a genuine C-terminal amide cap, once for the side chain),
# so the "exactly one match" requirement in `_strip_terminal_amide` below
# naturally declines those rather than guessing which one is the cap.
_PRIMARY_CARBOXAMIDE_SMARTS = "[CX3](=O)[NX3H2]"

# Standard amino acids whose SIDE CHAIN is itself a second acid/amide-class
# characteristic group -- P-41 seniority (carboxylic acid > carboxamide)
# means the free side-chain acid, not the amidated alpha-carboxyl, would be
# the true parent suffix, and the flat acylamino convention has no way to
# express that (it would need a `carbamoyl`-prefix construction instead of
# a suffix swap). Explicitly excluded rather than relying solely on the
# RT-gate to catch the resulting wrong-shaped name (the CLAUDE.md-flagged
# "side-chain-acid trap"). Asparagine/glutamine are listed defensively too
# (their OWN side-chain amide already makes `_strip_terminal_amide` decline
# via the two-match ambiguity above; this is belt-and-suspenders).
_COMPETING_SIDE_CHAIN_AMINO_ACIDS = {
    "aspartic acid", "glutamic acid", "asparagine", "glutamine",
}


def _strip_mono_n_methyl(smi: str) -> Tuple[str, bool]:
    """If ``smi`` (a single free-amino-acid residue fragment) has EXACTLY one
    free, non-acylated mono-N-methyl on its backbone amino nitrogen, return
    ``(new_smiles_with_methyl_removed, True)``. Otherwise return
    ``(smi, False)`` UNCHANGED -- ambiguous (0 or >=2 matches) or a
    sanitize failure both decline rather than guess."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, False
    pattern = Chem.MolFromSmarts(_MONO_N_METHYL_SMARTS)
    if pattern is None:
        return smi, False
    matches = mol.GetSubstructMatches(pattern)
    if len(matches) != 1:
        return smi, False
    methyl_idx = matches[0][0]
    rw = Chem.RWMol(mol)
    try:
        rw.RemoveAtom(methyl_idx)
        Chem.SanitizeMol(rw)
    except Exception:
        return smi, False
    return Chem.MolToSmiles(rw, canonical=True), True


def _strip_terminal_amide(smi: str) -> Tuple[str, bool]:
    """If ``smi`` (a single free-amino-acid residue fragment) has EXACTLY one
    genuine primary carboxamide, convert it back to the free acid (the same
    dummy->OH technique ``_reconstruct_free_amino_acid`` already uses) and
    return ``(new_smiles, True)``. Otherwise return ``(smi, False)``
    UNCHANGED -- 0 matches (nothing to strip) or >=2 matches (ambiguous,
    e.g. asparagine/glutamine's own side-chain amide) both decline."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, False
    pattern = Chem.MolFromSmarts(_PRIMARY_CARBOXAMIDE_SMARTS)
    if pattern is None:
        return smi, False
    matches = mol.GetSubstructMatches(pattern)
    if len(matches) != 1:
        return smi, False
    amide_n_idx = matches[0][2]  # [0]=carbonyl C, [1]==O, [2]=amide N
    rw = Chem.RWMol(mol)
    try:
        atom = rw.GetAtomWithIdx(amide_n_idx)
        atom.SetAtomicNum(8)
        atom.SetNumExplicitHs(1)
        atom.SetNoImplicit(True)
        Chem.SanitizeMol(rw)
    except Exception:
        return smi, False
    return Chem.MolToSmiles(rw, canonical=True), True


def _identify_one_residue(smi: str) -> Optional[Dict[str, str]]:
    """Ordinary (unmodified) trivial-name lookup for a SINGLE residue
    fragment, via the exact same table `_identify_residues` uses. Returns
    the residue dict, or None if this fragment is not (as-is) a standard
    amino acid."""
    result = _identify_residues([smi], strict_stereo=True)
    return result[0] if result else None


def _acid_name_to_amide(name: str) -> str:
    """Standard-amino-acid-amide spelling: drop a trailing 'e' and append
    'amide', else append 'amide' directly -- verified against all 16 non-
    excluded Table-10.4 names' known amide forms (glycinamide, alaninamide,
    valinamide, leucinamide, isoleucinamide, prolinamide,
    phenylalaninamide, tryptophanamide, methioninamide, serinamide,
    threoninamide, cysteinamide, tyrosinamide, lysinamide, argininamide,
    histidinamide)."""
    return (name[:-1] + "amide") if name.endswith("e") else (name + "amide")


def _try_capped_termini(mol) -> Optional[str]:
    """Lever C: re-attempt the flat acylamino chain allowing
    the FIRST residue's own backbone amino N to carry a lone free N-methyl,
    and/or the LAST residue's own alpha-carboxyl to be a primary carboxamide
    instead of the free acid the ordinary convention assumes. Returns an
    UNVERIFIED candidate string; the caller gates it through
    ``_rt_verified``.
    """
    residues = _extract_residues(mol)
    if residues is None or len(residues) < 2:
        return None

    working = list(residues)
    methyl_applied = False
    amide_applied = False

    # Only attempt a strip on a residue whose UNMODIFIED form already fails
    # the ordinary lookup -- an already-standard residue (e.g. a genuine,
    # uncapped asparagine sitting at the C-terminus) is left untouched, so
    # this lever can never mistake a residue's own natural side chain for a
    # cap that was never there.
    if _identify_one_residue(working[0]) is None:
        stripped, changed = _strip_mono_n_methyl(working[0])
        if changed:
            working[0] = stripped
            methyl_applied = True

    if _identify_one_residue(working[-1]) is None:
        stripped, changed = _strip_terminal_amide(working[-1])
        if changed:
            working[-1] = stripped
            amide_applied = True

    if not methyl_applied and not amide_applied:
        return None  # nothing this lever can help with -- some other decline

    named = _identify_residues(working, strict_stereo=True)
    if named is None:
        return None

    if amide_applied:
        cterm_name = named[-1]['name']
        if cterm_name in _COMPETING_SIDE_CHAIN_AMINO_ACIDS:
            # Side-chain-acid trap (CLAUDE.md): the free side-chain acid/
            # amide would outrank a suffix-amide swap under P-41 seniority.
            # Abstain rather than misplace.
            return None
        named = list(named)
        named[-1] = dict(named[-1])
        named[-1]['name'] = _acid_name_to_amide(cterm_name)

    candidate = _assemble_peptide_name(named)

    if methyl_applied:
        # The N-terminal token always starts at position 0 of the assembled
        # string in this flat convention; a cited 'D-' descriptor sits
        # before it (P-103.3.4), so 'N-methyl' needs its own hyphen only
        # then (matching the established 'N-methyl-D-aspartic acid' style).
        prefix = "N-methyl-" if candidate.startswith("D-") else "N-methyl"
        candidate = prefix + candidate

    return candidate


# ============================================================================
# Task 2.2: the general BACKBONE-SUBSTITUTIVE producer -- the
# real breadth lever for the ~214-row residual (giant / non-standard-residue
# peptides Levers A/B/C cannot name). Tried LAST, only when every path above
# has declined. See and the
# plan's Phase 2 Task 2.2 for the derivation.
#
# Mechanism (P-66.6.6 generalised to a non-standard/mixed chain): the
# C-TERMINAL residue's own free acid is the parent (P-41 senior principal
# group); every OTHER residue, walking N-terminal-ward one at a time, is
# folded onto whatever has been built so far EXACTLY the way the ordinary
# flat convention already works -- a standard residue's tabulated acyl form
# is simply PREPENDED (this is valid for ANY compound acyl word, standard or
# not, exactly as Lever A's single N-acyl cap already demonstrates); a
# NON-STANDARD residue is named as a free acid via the general/systematic
# namer (on its OWN isolated small fragment only -- never the whole
# remainder, which the general engine does NOT reliably parent-select for a
# multi-amide chain, measured directly this task) and converted to an acyl
# word the same way.
#
# The one extra wrinkle a non-standard residue introduces: its OWN acid name
# carries a VISIBLE, addressable leading 'amino' substituent (P-14
# alphanumeric order almost always puts it first). Once such a word is
# itself prepended onto the chain, that visible 'amino' token is the new
# attachment point for whatever comes next -- so the NEXT residue must be
# SPLICED into it (`_swap_amino_for_amido`, exactly Lever B's mechanism),
# not prepended as a separate word. This file tracks that as `is_open`:
# False after adding a STANDARD (retained/closed) residue, and True after
# adding a NON-STANDARD one whose acid name's leading substituent matched
# `_AMINO_PREFIX_RE` (a defined-stereo systematic name gets a leading CIP
# locant like '(2R)-' that this shared, already-fail-closed regex does not
# match -- so an 'open' non-standard residue immediately followed by
# another residue conservatively ABSTAINS rather than guess; this is the
# SAME conservative behaviour Lever B already has, reused unchanged).
#
# Side-chain-acid trap (CLAUDE.md; 57/226 backlog carry >=2 free -COOH): the
# parent must carry EXACTLY one free -COOH, and every OTHER residue's own
# isolated fragment must ALSO show exactly one (the one `_extract_residues`
# artificially frees at its own alpha-carboxyl cut point) -- an ADDITIONAL
# free acid anywhere is a genuine competing principal group (Glu/Asp side
# chain) this producer must never misplace; it ABSTAINS instead.
#
# Every candidate this returns is UNVERIFIED; `name_peptide` gates it
# through `_rt_verified` (full-InChI OPSIN round-trip) exactly like every
# other lever in this file, so a wrong guess degrades to abstention, never
# a wrong molecule (0-wrong ABSOLUTE).
# ============================================================================

# Work budget (giants): bound the per-residue general-namer fan-out
# rather than a wall-clock alarm -- Phase-0 measured that a per-molecule
# SIGALRM does NOT fire once the JVM is live (HotSpot signal chaining), so
# only an in-algorithm bound is reliable. No real backlog witness has more
# than 35 residues; a chain longer than this has no precedent and simply
# ABSTAINS rather than risk N separate general-namer calls compounding.
_BACKBONE_SUBSTITUTIVE_MAX_RESIDUES = 40

# A STANDARD residue costs one cheap table lookup; only a NON-standard one
# costs a full general-namer call (measured up to ~2s on a defined-stereo
# fragment). This is the real work-budget knob -- a chain where every
# residue is standard (the common shape) is bounded near-instantly by the
# residue cap above regardless of length; a chain needing many separate
# general-namer calls is the one that could compound toward the 40s-class
# verification bound, so cap that count directly.
_BACKBONE_SUBSTITUTIVE_MAX_NONSTANDARD_CALLS = 15


def _resolve_free_residue_acid_name(
    smi: str, systematic: bool = False,
) -> Optional[str]:
    """Name ``smi`` (a SINGLE isolated free-amino-acid fragment -- never a
    multi-residue remainder; the general/composer engine's own parent
    selection does not reliably handle a multi-amide chain, measured
    directly building this producer) as an acid via the general/PIN namer.
    Retained trivial or systematic, whichever the namer itself prefers.
    Fails closed (None) on any error or failure-sentinel output.

    ``systematic=True`` EXCLUDES the retained ``amino_acid`` dispatch class, so
    a standard residue is named by its fully systematic acid form
    ('L-alanine' -> '(2S)-2-aminopropanoic acid') -- required when the result
    must expose an addressable '(stereo)-N-amino' splice point (an amido prefix
    for a backbone residue, or a swap-mode parent) and to match the reference
    PIN form. A residue that has no systematic amino-acid form (achiral glycine
    stays 'glycine') is returned as-is; the caller decides how to fall back.

    ``_extract_residues`` stamps an ISOTOPE label on the cut-point oxygen
    of every fragment (its own internal bond-order bookkeeping --
    ``_identify_residues`` already strips it before its table lookups via
    an isomeric-SMILES-off fallback). The general namer has no such
    fallback and would otherwise try to name a real isotopologue, so clear
    isotopes here FIRST while preserving the genuine stereo descriptors
    (``@``/``@@``) this producer needs kept.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetIsotope():
            atom.SetIsotope(0)
    clean_smi = Chem.MolToSmiles(mol, canonical=True)

    from ..errors import is_failure_name
    from ..namer import Orthonym
    kwargs = dict(style="pin", _disable_opsin_validity_gate=True)
    if systematic:
        kwargs["_seed_excluded_dispatch_classes"] = ["amino_acid"]
    try:
        name = Orthonym(**kwargs).name(clean_smi)
    except Exception:
        return None
    if is_failure_name(name):
        return None
    return name


# A carboxylic-acid ESTER group -C(=O)-O-C: SMARTS atoms are
# [0]=carbonyl C, [1]==O, [2]=ester O, [3]=O-alkyl C.
_ESTER_GROUP_SMARTS = "[CX3](=O)[OX2][CX4]"


def _systematic_amido_form(smi: str) -> Optional[str]:
    """The '-amido' substituent prefix for a backbone residue ``smi`` to be
    SPLICED (P-66.1.1.4.3) into an open parent. Built from the residue's fully
    SYSTEMATIC free-acid name so the prefix carries an addressable
    '(stereo)-N-amino' token -- both to match the reference PIN form
    (alanine -> '(2S)-2-aminopropanamido', not the contracted 'alaninamido')
    and to expose the next splice point. Falls back to the retained standard-
    residue amido for a residue whose PIN acid has no systematic 'oic acid'
    form to transform (achiral glycine -> 'glycinamido'). Returns None when
    neither form is buildable; the caller's ``_rt_verified`` gate is the final
    backstop, so a bad spelling degrades to abstention, never a wrong name."""
    acid = _resolve_free_residue_acid_name(smi, systematic=True)
    if acid is not None:
        from ..assembly.substituent_naming import acid_name_to_amido_prefix
        amido = acid_name_to_amido_prefix(acid)
        if amido is not None:
            return amido
    info = _identify_one_residue(smi)
    if info is not None:
        stereo_disp = "" if info['stereo'] == "L-" else info['stereo']
        return _standard_amido_prefix(info['name'], stereo_disp)
    return None


def _name_ester_parent_systematic(parent_frag) -> Optional[str]:
    """Fully SYSTEMATIC name of a C-terminal ESTER parent (exactly one
    -C(=O)-O-C ester and NO free -COOH) as '<alkyl> ...oate', with an
    addressable leading '(stereo)-N-amino' splice point exposed. Built by
    reconstructing the parent's OWN free acid, naming it systematically, and
    re-esterifying with the ester's O-alkyl word (taken from the ordinary
    retained ester name). Returns None on any failure / out-of-scope shape
    (a non-'oic acid' acid stem, a compound O-alkyl, etc.); the caller's
    ``_rt_verified`` gate is the final backstop (0-wrong ABSOLUTE)."""
    ester_pat = Chem.MolFromSmarts(_ESTER_GROUP_SMARTS)
    if ester_pat is None:
        return None
    ematches = parent_frag.GetSubstructMatches(ester_pat)
    if len(ematches) != 1:
        return None
    carbonyl_c, _dbl_o, ester_o, alkyl_c = ematches[0]

    # Cleave the ester O--alkyl bond and cap the acyl oxygen as -OH, giving the
    # parent's own free carboxylic acid (the acid side keeps the carbonyl C).
    rw = Chem.RWMol(parent_frag)
    rw.RemoveBond(ester_o, alkyl_c)
    o_atom = rw.GetAtomWithIdx(ester_o)
    o_atom.SetNoImplicit(False)
    o_atom.SetNumExplicitHs(1)
    try:
        Chem.SanitizeMol(rw)
        mapping: List = []
        frags = Chem.GetMolFrags(
            rw, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=mapping)
    except Exception:
        return None
    if len(frags) != 2:
        return None
    acid_frag = None
    for frag, idxs in zip(frags, mapping):
        if carbonyl_c in idxs:
            acid_frag = frag
            break
    if acid_frag is None:
        return None

    sys_acid = _resolve_free_residue_acid_name(
        Chem.MolToSmiles(acid_frag), systematic=True)
    if sys_acid is None or not sys_acid.endswith("oic acid"):
        return None

    # O-alkyl word: the leading token of the ordinary (retained) ester name,
    # e.g. 'methyl L-leucinate' -> 'methyl'. A functional-class ester name is
    # '<alkyl> <ate-word>'; a name without that space is not an alkyl ester we
    # can rebuild here.
    retained = _name_gamma_acceptor(Chem.MolToSmiles(parent_frag))
    if retained is None or ' ' not in retained[0]:
        return None
    alkyl = retained[0].split(' ', 1)[0]
    return f"{alkyl} {sys_acid[:-len('oic acid')]}oate"


def _standard_amido_prefix(aa_name: str, stereo_disp: str) -> Optional[str]:
    """AMIDO-prefix form of a STANDARD amino acid's own alpha-carboxyl, for
    use as a SPLICED substituent (P-66.1.1.4.3) when this residue's own
    backbone amino group has already been filled by a non-standard donor's
    contribution further toward the N-terminus. Derived from the SAME
    table-verified amide-name transform ``_acid_name_to_amide`` already
    uses (drop trailing 'e', add 'amide'; final 'e'->'o' for the -amido
    form): 'glycine'->'glycinamido', 'alanine'->'alaninamido'.

    Excludes ``_COMPETING_SIDE_CHAIN_AMINO_ACIDS`` (Asp/Glu/Asn/Gln): their
    own name already encodes a side-chain acid/amide, so the blind
    amide-name transform is ambiguous about WHICH carbonyl it modifies --
    decline rather than guess (the same trap Lever C's docstring already
    flags). Any wrong guess this function makes is still caught by
    ``_rt_verified``'s full-InChI OPSIN round-trip before it can ship, so a
    bad spelling degrades to an abstention, never a wrong molecule.
    """
    if aa_name in _COMPETING_SIDE_CHAIN_AMINO_ACIDS:
        return None
    amide_name = _acid_name_to_amide(aa_name)
    if not amide_name.endswith('e'):
        return None
    amido = amide_name[:-1] + 'o'
    return (stereo_disp + amido) if stereo_disp else amido


def _residue_open_forms(
    smi: str,
) -> Optional[Tuple[Optional[str], Optional[str], bool]]:
    """``(acyl_word, amido_word, is_open)`` for residue ``smi`` viewed in
    ISOLATION (its own free NH2/COOH form, ignoring what precedes it in the
    real molecule). ``acyl_word`` is safe to PREPEND onto a currently-CLOSED
    accumulated name; ``amido_word`` is safe to SPLICE
    (``_swap_amino_for_amido``) into a currently-OPEN one. ``is_open`` says
    whether THIS residue's own contribution, once attached, leaves a
    further-addressable leading 'amino' locant for the NEXT (further
    N-terminal-ward) residue. Returns None when neither form is buildable.
    """
    info = _identify_one_residue(smi)
    if info is not None:
        stereo_disp = "" if info['stereo'] == "L-" else info['stereo']
        acyl = (stereo_disp + info['acyl']) if stereo_disp else info['acyl']
        amido = _standard_amido_prefix(info['name'], stereo_disp)
        return acyl, amido, False  # a retained trivial word is always closed

    acid_name = _resolve_free_residue_acid_name(smi)
    if acid_name is None:
        return None
    from ..assembly.substituent_naming import acid_name_to_amido_prefix
    from ..decomposition.fragment_assembly import _acid_to_acyl
    acyl = _acid_to_acyl(acid_name)
    amido = acid_name_to_amido_prefix(acid_name)
    if acyl is None and amido is None:
        return None
    is_open = _match_amino_prefix(acid_name) is not None
    return acyl, amido, is_open


def _try_backbone_substitutive(mol) -> Optional[str]:
    """.2: name ANY linear alpha-peptide (standard,
    non-standard, or a mix) as one systematic name, tried only when every
    other path in this file has declined. See the module-section comment
    above for the full derivation. Returns an UNVERIFIED candidate string;
    the caller (``name_peptide``) gates it through ``_rt_verified``.
    """
    residues = _extract_residues(mol)
    if residues is None or len(residues) < 2:
        return None
    if len(residues) > _BACKBONE_SUBSTITUTIVE_MAX_RESIDUES:
        return None

    cooh_pattern = Chem.MolFromSmarts(_TERMINAL_COOH_SMARTS)
    ester_pattern = Chem.MolFromSmarts(_ESTER_GROUP_SMARTS)

    # Parent scope: the C-terminal residue is the parent skeleton and must
    # carry EXACTLY one senior acid-class group -- either one free carboxylic
    # acid, OR (no free acid at all) exactly one carboxylic-ESTER C-terminus
    # (a 'methyl ...oate' parent). An aldehyde / primary-amide / dual-acid
    # C-terminus is out of THIS producer's scope (never guess a different
    # principal group / parent skeleton); it declines here and the RT-gate is
    # the final backstop for anything that slips through.
    parent_frag = Chem.MolFromSmiles(residues[-1])
    if parent_frag is None:
        return None
    n_parent_cooh = len(parent_frag.GetSubstructMatches(cooh_pattern))
    parent_is_ester = False
    if n_parent_cooh != 1:
        if n_parent_cooh == 0 and ester_pattern is not None and \
                len(parent_frag.GetSubstructMatches(ester_pattern)) == 1:
            parent_is_ester = True
        else:
            return None

    # Side-chain-acid trap: every OTHER residue's own isolated fragment
    # must show EXACTLY the one free acid its own reconstruction
    # artificially frees (its own alpha-carboxyl cut point) -- an
    # additional free acid anywhere is a genuine competing principal group.
    # Also pre-count how many are NON-standard (a cheap table lookup each)
    # -- ONLY a non-standard residue costs a general-namer call (measured
    # up to ~2s each on a defined-stereo residue), so this is the real work
    # budget: cap it well under the 40s-class verification bound rather
    # than the residue count alone.
    n_nonstandard = 0
    for frag_smi in residues[:-1]:
        frag = Chem.MolFromSmiles(frag_smi)
        if frag is None:
            return None
        if len(frag.GetSubstructMatches(cooh_pattern)) != 1:
            return None
        if _identify_one_residue(frag_smi) is None:
            n_nonstandard += 1
    if n_nonstandard > _BACKBONE_SUBSTITUTIVE_MAX_NONSTANDARD_CALLS:
        return None

    if parent_is_ester:
        # An ESTER C-terminus is always a systematic 'swap' parent: name it
        # fully systematically so its '(stereo)-N-amino' splice point is
        # exposed ('methyl (2S)-2-amino-4-methylpentanoate').
        name = _name_ester_parent_systematic(parent_frag)
        if name is None or _match_amino_prefix(name) is None:
            return None
        is_open = True
    else:
        acceptor = _name_gamma_acceptor(residues[-1])
        if acceptor is None:
            return None
        name, mode = acceptor
        is_open = False
        if mode == 'swap':
            if _match_amino_prefix(name) is None:
                # The retained systematic acid name has no addressable leading
                # 'amino' -- a standard-AA free acid the amino_acid table named
                # 'L-leucine'. Re-derive the fully SYSTEMATIC acid so the
                # '(stereo)-N-amino' splice point is exposed.
                name = _resolve_free_residue_acid_name(
                    residues[-1], systematic=True)
                if name is None or _match_amino_prefix(name) is None:
                    # Still not addressable (e.g. a multiplied 'x,y-diamino...'
                    # parent) -- nothing can be safely attached; decline.
                    return None
            is_open = True

    for smi in reversed(residues[:-1]):
        if is_open:
            # SPLICE this residue into the open parent's 'amino' as a
            # systematic '(...)amido' prefix (P-66.1.1.4.3), which itself
            # re-exposes an 'amino' for the next residue.
            amido = _systematic_amido_form(smi)
            if amido is None:
                return None
            name = _swap_amino_for_amido(name, amido)
            if name is None:
                return None
            is_open = _match_amino_prefix(amido) is not None
        else:
            # PREPEND this residue's acyl form onto the closed accumulated
            # name -- the ordinary flat acylamino convention (retained
            # 'glycyl'/'alanyl' for a standard residue, systematic for a
            # non-standard one).
            forms = _residue_open_forms(smi)
            if forms is None:
                return None
            acyl, _amido, next_is_open = forms
            if acyl is None:
                return None
            if name.startswith('D-'):
                name = f"{acyl}-{name}"
            else:
                name = f"{acyl}{name}"
            is_open = next_is_open

    return name
