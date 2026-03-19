"""Natural product naming rules.

Provides the naming function that bridges perception (scaffold detection)
and data (derivative lookup) to return trivial/retained names for natural
product molecules.

Algorithm:
1. Exact derivative lookup (O(1) dict lookup by canonical SMILES)
2. Scaffold substructure match via perception module
3. Return scaffold name for bare scaffolds
4. For decorated scaffolds (steroid class): enumerate -OH, =O, C=C, esters
   and assemble a systematic name using the scaffold stem
5. Return None for non-natural-product molecules

Decoration enumeration (steroids):
- Hydroxyl groups  -> prefix "hydroxy" with locant
- Ketone groups    -> suffix "-one" with locant
- Double bonds     -> suffix "-ene" with locant
- Triple bonds     -> suffix "-yne" with locant
- Ester groups     -> functional class format: parent-yl acylate (IUPAC P-65.6)
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from ..perception.stereo import assign_stereochemistry
from ..data.natural_products import (
    NATURAL_PRODUCT_DERIVATIVES,
    get_natural_product_name,
    get_scaffold_numbering,
    get_steroid_numbering,
)
from ..perception.natural_products import (
    detect_natural_product,
    get_scaffold_substituents,
)


# ---------------------------------------------------------------------------
# Steroid angular methyl positions for nor-detection
# ---------------------------------------------------------------------------
# For steroid scaffolds that include C-18 and C-19 in their numbering,
# these constants define the ring junction atoms where angular methyls attach.
#
# IUPAC steroid nomenclature:
#   C-18: angular methyl at ring C/D junction (bonded to C-13)
#   C-19: angular methyl at ring A/B junction (bonded to C-10)
#
# Nor-detection: if a scaffold's numbering includes locant 18 or 19, but the
# actual molecule is missing the methyl carbon at that position, it's a nor-
# modification. If the scaffold numbering doesn't include the locant at all
# (e.g., estrane has no C-19), no modification is detected -- the scaffold's
# own name is used (e.g., "estrane" not "19-norandrostane").

_ANGULAR_METHYL_JUNCTION = {
    18: 13,  # C-18 methyl is bonded to C-13 (ring C/D junction)
    19: 10,  # C-19 methyl is bonded to C-10 (ring A/B junction)
}


def detect_np_modifications(
    mol, scaffold_info: Dict, numbering: Dict[int, int]
) -> List[Dict]:
    """Detect nor-, homo-, seco- modifications vs matched NP scaffold.

    Compares the molecule's structure against the matched scaffold to identify
    structural modifications per IUPAC steroid nomenclature rules 3S-7/3S-8.

    Current detection capabilities:
    - nor-: Missing angular methyls (C-18, C-19) in steroid scaffolds.
      Detected by comparing the matched scaffold's numbering against the
      reference expectation for a complete steroid.
    - homo-: Placeholder for ring expansion detection (future).
    - seco-: Placeholder for ring-opening detection (future).

    Args:
        mol: RDKit Mol object.
        scaffold_info: Dict from detect_natural_product() with keys:
            scaffold_name, scaffold_stem, scaffold_class, scaffold_smiles,
            matched_atoms, non_scaffold_atoms.
        numbering: Target atom index -> IUPAC locant mapping from
            _build_target_to_iupac().

    Returns:
        List of modification dicts, each with keys:
            - prefix: str ('nor', 'homo', or 'seco')
            - locants: List[int] (IUPAC locant numbers)
            - ring_letter: Optional[str] (for homo-, e.g., 'D')
        Empty list if no modifications detected.
    """
    if mol is None or scaffold_info is None or numbering is None:
        return []

    modifications: List[Dict] = []

    scaffold_class = scaffold_info.get("scaffold_class", "")

    # --- NOR detection ---
    # For steroid scaffolds, check angular methyl positions (C-18, C-19).
    # Only detect nor- if the scaffold numbering INCLUDES the angular methyl
    # locant but the actual molecule is MISSING the methyl carbon at that
    # position. If the scaffold numbering doesn't include the locant (e.g.,
    # estrane lacks C-19), no modification is detected -- use the scaffold
    # name as-is.
    if scaffold_class == "steroid":
        # Build reverse map: IUPAC locant -> target atom index
        locant_to_target = {v: k for k, v in numbering.items()}
        missing_methyls = []

        for methyl_locant, junction_locant in _ANGULAR_METHYL_JUNCTION.items():
            # Only check if the scaffold numbering includes BOTH the methyl
            # locant and the junction locant
            if methyl_locant not in locant_to_target:
                continue  # Scaffold doesn't have this position (e.g., estrane lacks C-19)
            if junction_locant not in locant_to_target:
                continue

            methyl_target_idx = locant_to_target[methyl_locant]
            junction_target_idx = locant_to_target[junction_locant]

            # The methyl carbon should be bonded to the junction carbon.
            # Check if the methyl atom in the molecule is actually a terminal
            # methyl (CH3) or if it's been removed/replaced.
            methyl_atom = mol.GetAtomWithIdx(methyl_target_idx)
            junction_atom = mol.GetAtomWithIdx(junction_target_idx)

            # Check if the methyl atom has the expected connectivity:
            # a terminal methyl bonded to the junction should have exactly
            # 1 heavy-atom neighbor (the junction) and 3 hydrogens
            heavy_neighbors = [
                n for n in methyl_atom.GetNeighbors()
                if n.GetAtomicNum() > 1
            ]
            if len(heavy_neighbors) == 0:
                # The methyl position maps to a hydrogen-only atom or
                # an atom with no heavy neighbors -- methyl is absent
                missing_methyls.append(methyl_locant)
            elif methyl_atom.GetAtomicNum() != 6:
                # The position is occupied by a non-carbon atom
                missing_methyls.append(methyl_locant)

        if missing_methyls:
            modifications.append({
                "prefix": "nor",
                "locants": sorted(missing_methyls),
                "ring_letter": None,
            })

    # --- HOMO detection ---
    # Ring expansion: a ring in the molecule has one more member than expected.
    # For steroid scaffolds, compare ring sizes in the matched region against
    # the scaffold query molecule's ring sizes.
    # NOTE: Full homo-detection requires comparing ring membership counts
    # between the molecule and scaffold query. This is deferred to a future
    # phase as it requires ring-size analysis on the matched substructure.

    # --- SECO detection ---
    # Ring opening: a scaffold ring bond is absent in the molecule.
    # Seco compounds typically don't match the parent scaffold at all
    # (because the substructure match fails when a ring is broken),
    # so seco-detection requires a different approach (e.g., seco-specific
    # scaffold patterns). This is deferred to a future phase.

    return modifications


def format_np_modification_prefix(modifications: List[Dict]) -> str:
    """Format modification prefixes per IUPAC nomenclature.

    IUPAC steroid nomenclature rules 3S-7, 3S-8:
    - nor-: '{locant}-nor' or '{loc1},{loc2}-dinor' etc.
    - homo-: '{ring_letter}-homo' or '{locant}-homo'
    - seco-: '{loc1},{loc2}-seco'

    Multiple modifications are sorted alphabetically by prefix name
    (homo < nor < seco) and concatenated with hyphens.

    Args:
        modifications: List of modification dicts from detect_np_modifications().

    Returns:
        Combined prefix string (e.g., '19-nor', 'D-homo-19-nor-9,10-seco')
        or empty string if no modifications.
    """
    if not modifications:
        return ""

    # Multiplicative prefixes for nor-
    _NOR_MULTIPLIERS = {
        1: "",
        2: "di",
        3: "tri",
        4: "tetra",
        5: "penta",
    }

    # Sort alphabetically by prefix name
    sorted_mods = sorted(modifications, key=lambda m: m["prefix"])

    parts = []
    for mod in sorted_mods:
        prefix = mod["prefix"]
        locants = sorted(mod.get("locants", []))
        ring_letter = mod.get("ring_letter")

        if prefix == "nor":
            count = len(locants)
            multiplier = _NOR_MULTIPLIERS.get(count, str(count))
            locant_str = ",".join(str(loc) for loc in locants)
            if locant_str:
                parts.append(f"{locant_str}-{multiplier}{prefix}")
            else:
                parts.append(prefix)

        elif prefix == "homo":
            if ring_letter:
                parts.append(f"{ring_letter}-{prefix}")
            elif locants:
                locant_str = ",".join(str(loc) for loc in locants)
                parts.append(f"{locant_str}-{prefix}")
            else:
                parts.append(prefix)

        elif prefix == "seco":
            locant_str = ",".join(str(loc) for loc in locants)
            if locant_str:
                parts.append(f"{locant_str}-{prefix}")
            else:
                parts.append(prefix)

    return "-".join(parts)


def name_natural_product(mol) -> Optional[str]:
    """Name a molecule using natural product recognition.

    Tries exact derivative lookup first, then scaffold substructure matching.
    For steroids with decorations, enumerates functional groups.
    Returns None for molecules that are not recognized natural products.

    Args:
        mol: RDKit Mol object.

    Returns:
        Trivial/retained name if the molecule is a recognized natural product,
        otherwise None.
    """
    if mol is None:
        return None

    # Step 1: Get canonical SMILES for exact lookup
    canonical = Chem.MolToSmiles(mol, canonical=True)

    # Step 2: Exact derivative match (cholesterol, morphine, etc.)
    exact_name = get_natural_product_name(canonical)
    if exact_name is not None:
        return exact_name

    # Step 3: Scaffold substructure match
    scaffold_info = detect_natural_product(mol)
    if scaffold_info is None:
        return None

    # Step 3b: Detect NP modifications (nor-, homo-, seco-)
    # Build numbering early so it can be reused in Steps 4-6
    numbering = _build_target_to_iupac(scaffold_info)
    modification_prefix = ""
    if numbering is not None:
        modifications = detect_np_modifications(mol, scaffold_info, numbering)
        if modifications:
            modification_prefix = format_np_modification_prefix(modifications)

    # Step 4: Check if molecule is just the bare scaffold (no substituents)
    non_scaffold = scaffold_info["non_scaffold_atoms"]
    if not non_scaffold:
        # Bare scaffold -- but check for internal unsaturation
        if numbering:
            scaffold_arom = _build_scaffold_aromatic_atoms(
                scaffold_info.get("scaffold_smiles"),
                scaffold_info.get("matched_atoms"),
            )
            unsaturation = _find_scaffold_unsaturation(
                mol, set(scaffold_info["matched_atoms"]), numbering,
                scaffold_aromatic_atoms=scaffold_arom,
            )
            if unsaturation["ene"] or unsaturation["yne"]:
                stereo_prefix = _collect_np_stereo(mol, numbering)
                return _assemble_np_name(
                    scaffold_info["scaffold_stem"],
                    scaffold_info["scaffold_name"],
                    hydroxyls=[],
                    ketones=[],
                    unsaturation=unsaturation,
                    stereo_prefix=stereo_prefix,
                    modification_prefix=modification_prefix,
                )
        # Coverage gate: reject bare scaffold if it covers too little
        total_heavy = mol.GetNumHeavyAtoms()
        if total_heavy > 10:
            scaffold_coverage = len(scaffold_info["matched_atoms"]) / total_heavy
            if scaffold_coverage < 0.60:
                return None  # Fall through to systematic naming
        if modification_prefix:
            return modification_prefix + scaffold_info["scaffold_name"]
        return scaffold_info["scaffold_name"]

    # Step 5: Check if non-scaffold atoms are only hydrogens
    all_h = all(
        mol.GetAtomWithIdx(idx).GetAtomicNum() == 1
        for idx in non_scaffold
    )
    if all_h:
        # Coverage gate: reject bare scaffold if it covers too little
        total_heavy = mol.GetNumHeavyAtoms()
        if total_heavy > 10:
            scaffold_coverage = len(scaffold_info["matched_atoms"]) / total_heavy
            if scaffold_coverage < 0.60:
                return None  # Fall through to systematic naming
        if modification_prefix:
            return modification_prefix + scaffold_info["scaffold_name"]
        return scaffold_info["scaffold_name"]

    # Step 6: Scaffold with substituents -- enumerate decorations
    return name_natural_product_with_substituents(
        mol, scaffold_info, modification_prefix=modification_prefix
    )


def name_natural_product_with_substituents(
    mol, scaffold_info: Dict, modification_prefix: str = ""
) -> str:
    """Name a natural product with decorations on the scaffold.

    For steroids: enumerates hydroxyl, ketone, unsaturation, and ester
    decorations, assembles a systematic name like "3-hydroxycholest-4-en-17-one"
    or functional class format "3-oxoandrost-4-en-17-yl acetate" for esters.

    For non-steroid scaffolds or when numbering is unavailable, falls back
    to returning just the scaffold name.

    Args:
        mol: RDKit Mol object.
        scaffold_info: Dict from detect_natural_product() with keys:
            - scaffold_name: str
            - scaffold_stem: str
            - scaffold_class: str
            - scaffold_smiles: str
            - matched_atoms: tuple
            - non_scaffold_atoms: set
        modification_prefix: NP modification prefix string (e.g., '19-nor')
            to insert before the scaffold stem. Empty string if no modifications.

    Returns:
        Name string for the natural product.
    """
    scaffold_class = scaffold_info["scaffold_class"]
    scaffold_stem = scaffold_info["scaffold_stem"]
    scaffold_name = scaffold_info["scaffold_name"]

    # Build target atom index -> IUPAC locant mapping
    # Works for any scaffold class (steroid, alkaloid) that has a numbering map
    numbering = _build_target_to_iupac(scaffold_info)
    if numbering is None:
        # No numbering map -> coverage gate, then bare scaffold name
        total_heavy = mol.GetNumHeavyAtoms()
        if total_heavy > 10:
            scaffold_coverage = len(scaffold_info["matched_atoms"]) / total_heavy
            threshold = 0.40 if scaffold_class == "steroid" else 0.50
            if scaffold_coverage < threshold:
                return None  # Fall through to systematic naming
        if modification_prefix:
            return modification_prefix + scaffold_name
        return scaffold_name

    matched_set = set(scaffold_info["matched_atoms"])

    # Collect stereodescriptors from the scaffold numbering
    stereo_prefix = _collect_np_stereo(mol, numbering)

    # 0. Find ester decorations first (they consume atoms that would otherwise
    #    be counted as hydroxyls or ketones)
    esters = _find_ester_decorations(mol, scaffold_info, numbering)
    ester_consumed_atoms = set()
    for e in esters:
        ester_consumed_atoms.update(e["all_atoms"])

    # 0b. Find epoxy bridges (O bridging two scaffold atoms, e.g. morphine 4,5-epoxy)
    epoxy_bridges = _find_epoxy_bridges(mol, matched_set, numbering,
                                        exclude_atoms=ester_consumed_atoms)
    # Collect oxygen atoms consumed by epoxy bridges so they aren't counted as OH
    epoxy_consumed = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 8 or atom.GetIdx() in matched_set:
            continue
        neighbors = [n for n in atom.GetNeighbors()]
        if len(neighbors) == 2:
            n1, n2 = neighbors
            if (n1.GetIdx() in matched_set and n2.GetIdx() in matched_set
                    and n1.GetIdx() in numbering and n2.GetIdx() in numbering):
                epoxy_consumed.add(atom.GetIdx())

    all_exclude = ester_consumed_atoms | epoxy_consumed

    # 0c. Find methoxy groups (-OCH3) before hydroxyls (consumes O atoms)
    methoxys = _find_methoxys(mol, scaffold_info, numbering,
                              exclude_atoms=all_exclude)

    # 1. Find hydroxyl groups (exocyclic -OH on scaffold atoms)
    hydroxyls = _find_hydroxyls(mol, scaffold_info, numbering,
                                exclude_atoms=all_exclude)

    # 2. Find ketone groups (exocyclic =O on scaffold atoms)
    ketones = _find_ketones(mol, scaffold_info, numbering,
                            exclude_atoms=all_exclude)

    # 3. Find unsaturation within the scaffold (C=C, C#C, AROMATIC)
    scaffold_arom = _build_scaffold_aromatic_atoms(
        scaffold_info.get("scaffold_smiles"),
        scaffold_info.get("matched_atoms"),
    )
    unsaturation = _find_scaffold_unsaturation(
        mol, matched_set, numbering,
        scaffold_aromatic_atoms=scaffold_arom,
    )

    # 3b. Find methyl substituents on scaffold (extra CH3 not part of scaffold)
    methyls = _find_methyls(mol, scaffold_info, numbering,
                            exclude_atoms=all_exclude)

    # 3c. Find halogen substituents on scaffold
    halogens = _find_halogens(mol, scaffold_info, numbering,
                              exclude_atoms=all_exclude)

    # 3d. Find N-alkyl groups on scaffold nitrogen atoms
    n_alkyls = _find_n_alkyl(mol, matched_set, numbering,
                             exclude_atoms=all_exclude)

    # 4. If esters found, use functional class format (IUPAC P-65.6)
    if esters:
        return _assemble_np_ester_name(
            scaffold_stem, scaffold_name, hydroxyls, ketones,
            unsaturation, esters, stereo_prefix=stereo_prefix,
            methyls=methyls, halogens=halogens,
        )

    # If no decorations found, return bare scaffold name
    if (not hydroxyls and not ketones and not methyls and not halogens
            and not unsaturation["ene"] and not unsaturation["yne"]
            and not epoxy_bridges and not n_alkyls and not methoxys):
        if modification_prefix:
            return modification_prefix + scaffold_name
        return scaffold_name

    # 5. Assemble the decorated name (substitutive format)
    return _assemble_np_name(
        scaffold_stem, scaffold_name, hydroxyls, ketones, unsaturation,
        stereo_prefix=stereo_prefix, methyls=methyls, halogens=halogens,
        epoxy_bridges=epoxy_bridges, n_alkyls=n_alkyls, methoxys=methoxys,
        modification_prefix=modification_prefix,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_target_to_iupac(scaffold_info: Dict) -> Optional[Dict[int, int]]:
    """Build mapping from target molecule atom indices to IUPAC locants.

    Uses numbering maps from the data module (steroid and alkaloid).

    Args:
        scaffold_info: Dict from detect_natural_product().

    Returns:
        Dict mapping target atom index to IUPAC locant, or None if
        no numbering map is available for this scaffold.
    """
    from ..data.natural_products import get_scaffold_numbering

    scaffold_smiles = scaffold_info["scaffold_smiles"]
    numbering_map = get_scaffold_numbering(scaffold_smiles)
    if numbering_map is None:
        return None

    matched_atoms = scaffold_info["matched_atoms"]
    target_to_iupac = {}
    for query_pos, target_idx in enumerate(matched_atoms):
        if query_pos in numbering_map:
            target_to_iupac[target_idx] = numbering_map[query_pos]

    return target_to_iupac


def _collect_np_stereo(mol, numbering: Dict[int, int]) -> str:
    """Collect stereodescriptors for a natural product using IUPAC locants.

    Uses RDKit CIP labeling and the scaffold numbering map to produce
    a stereodescriptor prefix like "(5R,8S,9S,10R,13S,14S)-".

    Args:
        mol: RDKit Mol object with stereocenters.
        numbering: Dict mapping atom index -> IUPAC locant.

    Returns:
        Stereo prefix string (e.g., "(5R,8S)-") or empty string if
        no stereocenters have defined CIP labels.
    """
    from ..rules.stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    assign_stereochemistry(mol)
    descriptors = collect_stereodescriptors(mol, numbering)
    if descriptors:
        return format_stereodescriptor_string(descriptors)
    return ""


def _find_hydroxyls(
    mol, scaffold_info: Dict, numbering: Dict[int, int],
    exclude_atoms: Optional[set] = None,
) -> List[int]:
    """Find hydroxyl groups (-OH) attached to scaffold atoms.

    Returns a sorted list of IUPAC locants where -OH groups are found.

    Args:
        mol: RDKit Mol object.
        scaffold_info: Dict from detect_natural_product().
        numbering: Target atom index to IUPAC locant mapping.
        exclude_atoms: Set of atom indices consumed by esters (skip these).
    """
    hydroxyls = []
    exclude = exclude_atoms or set()

    subs = get_scaffold_substituents(mol, scaffold_info["matched_atoms"])
    for sub in subs:
        attach = sub["attachment_atom"]
        if attach not in numbering:
            continue

        # Skip atoms consumed by ester detection
        if sub["first_atom"] in exclude:
            continue

        # Single-atom substituent that is oxygen with 1 H -> hydroxyl
        if len(sub["substituent_atoms"]) == 1:
            first_idx = sub["first_atom"]
            atom = mol.GetAtomWithIdx(first_idx)
            if atom.GetAtomicNum() == 8:  # Oxygen
                bond = mol.GetBondBetweenAtoms(attach, first_idx)
                if bond and bond.GetBondType() == Chem.BondType.SINGLE:
                    # Check it has at least 1 H (not ether, not ester)
                    if atom.GetTotalNumHs() >= 1:
                        hydroxyls.append(numbering[attach])

    return sorted(hydroxyls)


def _find_ketones(
    mol, scaffold_info: Dict, numbering: Dict[int, int],
    exclude_atoms: Optional[set] = None,
) -> List[int]:
    """Find ketone groups (=O) on scaffold atoms.

    Returns a sorted list of IUPAC locants where C=O groups are found.

    Args:
        mol: RDKit Mol object.
        scaffold_info: Dict from detect_natural_product().
        numbering: Target atom index to IUPAC locant mapping.
        exclude_atoms: Set of atom indices consumed by esters (skip these).
    """
    ketones = []
    exclude = exclude_atoms or set()

    subs = get_scaffold_substituents(mol, scaffold_info["matched_atoms"])
    for sub in subs:
        attach = sub["attachment_atom"]
        if attach not in numbering:
            continue

        # Skip atoms consumed by ester detection
        if sub["first_atom"] in exclude:
            continue

        # Single-atom substituent that is doubly-bonded oxygen -> ketone
        if len(sub["substituent_atoms"]) == 1:
            first_idx = sub["first_atom"]
            atom = mol.GetAtomWithIdx(first_idx)
            if atom.GetAtomicNum() == 8:  # Oxygen
                bond = mol.GetBondBetweenAtoms(attach, first_idx)
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    ketones.append(numbering[attach])

    return sorted(ketones)


def _find_methyls(
    mol, scaffold_info: Dict, numbering: Dict[int, int],
    exclude_atoms: Optional[set] = None,
) -> List[int]:
    """Find methyl groups (-CH3) attached to scaffold atoms.

    Returns a sorted list of IUPAC locants where extra methyl groups are found.
    Only detects methyl groups that are NOT part of the scaffold (extra methyls).

    Args:
        mol: RDKit Mol object.
        scaffold_info: Dict from detect_natural_product().
        numbering: Target atom index to IUPAC locant mapping.
        exclude_atoms: Set of atom indices consumed by esters (skip these).
    """
    methyls = []
    exclude = exclude_atoms or set()

    subs = get_scaffold_substituents(mol, scaffold_info["matched_atoms"])
    for sub in subs:
        attach = sub["attachment_atom"]
        if attach not in numbering:
            continue

        # Skip atoms consumed by ester detection
        if sub["first_atom"] in exclude:
            continue

        # Single carbon atom substituent with 3 H -> methyl
        # Skip methyls on nitrogen (those are N-alkyls, handled separately)
        if len(sub["substituent_atoms"]) == 1:
            first_idx = sub["first_atom"]
            atom = mol.GetAtomWithIdx(first_idx)
            if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 3:
                attach_atom = mol.GetAtomWithIdx(attach)
                if attach_atom.GetAtomicNum() == 7:
                    continue  # N-alkyl, not C-methyl
                bond = mol.GetBondBetweenAtoms(attach, first_idx)
                if bond and bond.GetBondType() == Chem.BondType.SINGLE:
                    methyls.append(numbering[attach])

    return sorted(methyls)


def _find_halogens(
    mol, scaffold_info: Dict, numbering: Dict[int, int],
    exclude_atoms: Optional[set] = None,
) -> List[Tuple[int, str]]:
    """Find halogen atoms attached to scaffold atoms.

    Returns a sorted list of (IUPAC locant, halogen_prefix) tuples.

    Args:
        mol: RDKit Mol object.
        scaffold_info: Dict from detect_natural_product().
        numbering: Target atom index to IUPAC locant mapping.
        exclude_atoms: Set of atom indices consumed by esters (skip these).
    """
    HALOGEN_PREFIX = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}
    halogens = []
    exclude = exclude_atoms or set()

    subs = get_scaffold_substituents(mol, scaffold_info["matched_atoms"])
    for sub in subs:
        attach = sub["attachment_atom"]
        if attach not in numbering:
            continue

        if sub["first_atom"] in exclude:
            continue

        if len(sub["substituent_atoms"]) == 1:
            first_idx = sub["first_atom"]
            atom = mol.GetAtomWithIdx(first_idx)
            if atom.GetAtomicNum() in HALOGEN_PREFIX:
                halogens.append(
                    (numbering[attach], HALOGEN_PREFIX[atom.GetAtomicNum()])
                )

    return sorted(halogens, key=lambda x: (x[1], x[0]))


def _build_scaffold_aromatic_atoms(
    scaffold_smiles: Optional[str], matched_atoms: Optional[tuple]
) -> set:
    """Build the set of target atom indices that are aromatic in the scaffold pattern.

    For scaffolds whose SMARTS already contain aromatic atoms (e.g. morphinan
    ``c1ccc2c(c1)...``), the returned set marks those atoms so that
    ``_find_scaffold_unsaturation`` can skip inherent aromatic bonds.

    Returns an empty set when no scaffold SMILES is provided or when the
    scaffold contains no aromatic atoms (e.g. steroid scaffolds).
    """
    if not scaffold_smiles or matched_atoms is None:
        return set()

    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
    if scaffold_mol is None:
        return set()

    aromatic_query_indices = {
        atom.GetIdx()
        for atom in scaffold_mol.GetAtoms()
        if atom.GetIsAromatic()
    }
    if not aromatic_query_indices:
        return set()

    # Map query atom positions to target atom indices
    return {
        matched_atoms[qi]
        for qi in aromatic_query_indices
        if qi < len(matched_atoms)
    }


def _find_scaffold_unsaturation(
    mol, matched_set: set, numbering: Dict[int, int],
    scaffold_aromatic_atoms: Optional[set] = None,
) -> Dict[str, List[int]]:
    """Find double and triple bonds within the scaffold.

    Detects explicit DOUBLE and TRIPLE bonds directly.  For aromatic rings
    (IUPAC P-31.1.3.4), Kekulizes a *copy* of the molecule to resolve
    aromatic bonds into canonical alternating DOUBLE/SINGLE pattern, then
    counts the resulting DOUBLE bonds between aromatic C-C atoms.  The
    original molecule is never mutated.

    Aromatic bonds that are *inherent* to the scaffold pattern (e.g.
    morphinan Ring A, whose SMARTS already contains ``c`` atoms) are excluded
    because the base scaffold name already encodes that aromaticity.

    Args:
        mol: RDKit Mol object.
        matched_set: Atom indices belonging to the scaffold.
        numbering: Map from atom index to IUPAC locant.
        scaffold_aromatic_atoms: Atom indices that are aromatic in the
            scaffold SMARTS pattern (from ``_build_scaffold_aromatic_atoms``).
            When ``None`` or empty, all aromatic C-C ene positions are counted.

    Returns a dict with 'ene' and 'yne' keys, each a sorted list of
    IUPAC locants (lower locant of each unsaturated bond).
    """
    if scaffold_aromatic_atoms is None:
        scaffold_aromatic_atoms = set()

    ene_locants = []
    yne_locants = []

    # Determine if we need Kekulized bond orders for aromatic detection.
    # Check whether the molecule has any aromatic scaffold C-C bonds that
    # are NOT inherent to the scaffold pattern.
    has_novel_aromatic = False
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.AROMATIC:
            continue
        b_idx = bond.GetBeginAtomIdx()
        e_idx = bond.GetEndAtomIdx()
        if b_idx not in matched_set or e_idx not in matched_set:
            continue
        b_atom = mol.GetAtomWithIdx(b_idx)
        e_atom = mol.GetAtomWithIdx(e_idx)
        if b_atom.GetAtomicNum() != 6 or e_atom.GetAtomicNum() != 6:
            continue
        # Skip inherent scaffold aromaticity
        if b_idx in scaffold_aromatic_atoms and e_idx in scaffold_aromatic_atoms:
            continue
        has_novel_aromatic = True
        break

    # Kekulize a COPY to resolve aromatic bonds into explicit DOUBLE/SINGLE.
    # clearAromaticFlags=False preserves GetIsAromatic() for filtering.
    kek_mol = None
    if has_novel_aromatic:
        try:
            kek_mol = Chem.RWMol(mol)
            Chem.Kekulize(kek_mol, clearAromaticFlags=False)
        except Exception:
            kek_mol = None  # Fall back: skip aromatic detection

    # Use the Kekulized copy when available, otherwise the original (for
    # explicit DOUBLE/TRIPLE bonds the result is identical).
    scan_mol = kek_mol if kek_mol is not None else mol

    for bond in scan_mol.GetBonds():
        b_idx = bond.GetBeginAtomIdx()
        e_idx = bond.GetEndAtomIdx()

        # Both atoms must be in the scaffold
        if b_idx not in matched_set or e_idx not in matched_set:
            continue

        # Both must have IUPAC locants
        if b_idx not in numbering or e_idx not in numbering:
            continue

        # Only carbon-carbon bonds
        b_atom = scan_mol.GetAtomWithIdx(b_idx)
        e_atom = scan_mol.GetAtomWithIdx(e_idx)
        if b_atom.GetAtomicNum() != 6 or e_atom.GetAtomicNum() != 6:
            continue

        b_loc = numbering[b_idx]
        e_loc = numbering[e_idx]
        lower_loc = min(b_loc, e_loc)

        if bond.GetBondType() == Chem.BondType.DOUBLE:
            # For Kekulized aromatic bonds, check the scaffold-inherent filter
            if b_atom.GetIsAromatic() and e_atom.GetIsAromatic():
                if b_idx in scaffold_aromatic_atoms and e_idx in scaffold_aromatic_atoms:
                    continue  # Inherent scaffold aromaticity
            ene_locants.append(lower_loc)
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            yne_locants.append(lower_loc)

    return {"ene": sorted(ene_locants), "yne": sorted(yne_locants)}


def _find_epoxy_bridges(
    mol, matched_set: set, numbering: Dict[int, int],
    exclude_atoms: Optional[set] = None,
) -> List[Tuple[int, int]]:
    """Find epoxy bridges (oxygen bridging two scaffold atoms).

    An epoxy bridge is an oxygen atom whose ONLY heavy-atom neighbors are both
    scaffold atoms with IUPAC locants.  Phenolic/enol oxygens (aromatic or
    doubly bonded) are excluded.

    Returns sorted list of (lower_locant, higher_locant) tuples.
    """
    exclude = exclude_atoms or set()
    bridges = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 8:
            continue
        if atom.GetIdx() in matched_set:
            continue  # Oxygen that IS part of the scaffold (not a substituent)
        if atom.GetIdx() in exclude:
            continue
        if atom.GetIsAromatic():
            continue

        # Oxygen must have exactly 2 heavy-atom neighbors, both in scaffold
        neighbors = [n for n in atom.GetNeighbors()]
        if len(neighbors) != 2:
            continue

        n1, n2 = neighbors[0], neighbors[1]
        if n1.GetIdx() not in matched_set or n2.GetIdx() not in matched_set:
            continue
        if n1.GetIdx() not in numbering or n2.GetIdx() not in numbering:
            continue

        # Both bonds must be single (not =O or aromatic)
        b1 = mol.GetBondBetweenAtoms(atom.GetIdx(), n1.GetIdx())
        b2 = mol.GetBondBetweenAtoms(atom.GetIdx(), n2.GetIdx())
        if b1.GetBondType() != Chem.BondType.SINGLE:
            continue
        if b2.GetBondType() != Chem.BondType.SINGLE:
            continue

        loc1, loc2 = numbering[n1.GetIdx()], numbering[n2.GetIdx()]
        bridges.append((min(loc1, loc2), max(loc1, loc2)))

    return sorted(bridges)


def _find_n_alkyl(
    mol, matched_set: set, numbering: Dict[int, int],
    exclude_atoms: Optional[set] = None,
) -> List[Tuple[int, str]]:
    """Find N-alkyl groups on scaffold nitrogen atoms.

    Detects alkyl substituents (methyl, ethyl, etc.) attached to nitrogen
    atoms that are part of the scaffold.

    Returns sorted list of (locant, alkyl_name) tuples.
    """
    from ..data.chain_names import get_chain_prefix

    exclude = exclude_atoms or set()
    n_alkyls = []

    for atom_idx in matched_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() != 7:  # Only nitrogen
            continue
        if atom_idx not in numbering:
            continue

        # Find non-scaffold carbon neighbors
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in matched_set:
                continue  # Skip scaffold atoms
            if nbr_idx in exclude:
                continue
            if nbr.GetAtomicNum() != 6:
                continue  # Only carbon substituents

            bond = mol.GetBondBetweenAtoms(atom_idx, nbr_idx)
            if not bond or bond.GetBondType() != Chem.BondType.SINGLE:
                continue

            # Count the alkyl chain length (simple linear alkyl)
            chain_len = _trace_alkyl_chain(mol, nbr_idx, matched_set)
            if chain_len > 0:
                prefix = get_chain_prefix(chain_len)
                n_alkyls.append((numbering[atom_idx], f"{prefix}yl"))

    return sorted(n_alkyls)


def _trace_alkyl_chain(mol, start_idx: int, exclude_set: set) -> int:
    """Trace a simple unbranched alkyl chain starting from start_idx.

    Returns the chain length (number of carbons).  Returns 0 if the
    fragment is not a simple unbranched all-carbon chain.
    """
    visited = set()
    current = start_idx
    length = 0

    while True:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetAtomicNum() != 6:
            break
        visited.add(current)
        length += 1

        # Find the next carbon in the chain (not in scaffold, not visited)
        next_c = None
        branch = False
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in exclude_set or nbr_idx in visited:
                continue
            if nbr.GetAtomicNum() == 6:
                if next_c is not None:
                    branch = True  # Branched -> not simple alkyl
                    break
                next_c = nbr_idx

        if branch:
            return 0  # Not a simple chain
        if next_c is None:
            break  # End of chain
        current = next_c

    return length


def _find_methoxys(
    mol, scaffold_info: Dict, numbering: Dict[int, int],
    exclude_atoms: Optional[set] = None,
) -> List[int]:
    """Find methoxy groups (-OCH3) attached to scaffold atoms.

    Returns sorted list of IUPAC locants where methoxy groups are found.
    Methoxy consumes the OH position (so it should be excluded from hydroxyls).
    """
    methoxys = []
    exclude = exclude_atoms or set()

    subs = get_scaffold_substituents(mol, scaffold_info["matched_atoms"])
    for sub in subs:
        attach = sub["attachment_atom"]
        if attach not in numbering:
            continue
        if sub["first_atom"] in exclude:
            continue

        # Two-atom substituent: O-CH3
        if len(sub["substituent_atoms"]) == 2:
            atoms_list = list(sub["substituent_atoms"])
            first_idx = sub["first_atom"]
            first_atom = mol.GetAtomWithIdx(first_idx)

            # First atom is oxygen, single bond to scaffold
            if first_atom.GetAtomicNum() != 8:
                continue
            bond = mol.GetBondBetweenAtoms(attach, first_idx)
            if not bond or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if first_atom.GetTotalNumHs() != 0:
                continue  # Has H -> it's an OH, not OCH3

            # Guard: if scaffold atom also has =O, this is an ester, not methoxy
            attach_atom = mol.GetAtomWithIdx(attach)
            is_ester_carbonyl = False
            for nbr in attach_atom.GetNeighbors():
                if nbr.GetAtomicNum() == 8 and nbr.GetIdx() != first_idx:
                    b = mol.GetBondBetweenAtoms(attach, nbr.GetIdx())
                    if b and b.GetBondType() == Chem.BondType.DOUBLE:
                        is_ester_carbonyl = True
                        break
            if is_ester_carbonyl:
                continue

            # Second atom must be carbon with 3 H (methyl)
            second_idx = [a for a in atoms_list if a != first_idx][0]
            second_atom = mol.GetAtomWithIdx(second_idx)
            if second_atom.GetAtomicNum() == 6 and second_atom.GetTotalNumHs() == 3:
                methoxys.append(numbering[attach])

    return sorted(methoxys)


def _assemble_np_name(
    stem: str,
    scaffold_name: str,
    hydroxyls: List[int],
    ketones: List[int],
    unsaturation: Dict[str, List[int]],
    stereo_prefix: str = "",
    methyls: Optional[List[int]] = None,
    halogens: Optional[List[Tuple[int, str]]] = None,
    epoxy_bridges: Optional[List[Tuple[int, int]]] = None,
    n_alkyls: Optional[List[Tuple[int, str]]] = None,
    methoxys: Optional[List[int]] = None,
    modification_prefix: str = "",
) -> str:
    """Assemble a decorated natural product name.

    Format: {stereo_prefix}{fg_prefix}{modification_prefix}{stem}{unsat}{suffix}
    Example: "(3R,5R,8R,9S,10S,13S,14S,17R)-androstan-3,17-diol"
    With modification: "17-hydroxy-19-norandrost-4-en-3-one"

    The modification prefix (nor-, homo-, seco-) is inserted directly before
    the scaffold stem, after any functional group prefixes. This follows
    IUPAC convention where modification prefixes are stem modifications.

    Args:
        stem: Scaffold stem (e.g., "cholest", "androst").
        scaffold_name: Full scaffold name for fallback.
        hydroxyls: Sorted IUPAC locants of -OH groups.
        ketones: Sorted IUPAC locants of =O groups.
        unsaturation: Dict with 'ene' and 'yne' locant lists.
        stereo_prefix: Stereodescriptor prefix (e.g., "(5R,8S)-") or "".
        methyls: Optional sorted IUPAC locants of extra methyl groups.
        halogens: Optional sorted list of (locant, halogen_prefix) tuples.
        epoxy_bridges: Optional sorted list of (loc1, loc2) epoxy bridge locants.
        n_alkyls: Optional sorted list of (locant, alkyl_name) tuples.
        methoxys: Optional sorted IUPAC locants of -OCH3 groups.
        modification_prefix: NP modification prefix (e.g., '19-nor') to insert
            before the stem. Empty string if no modifications.

    Returns:
        Assembled IUPAC-style natural product name.
    """
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS

    methyls = methyls or []
    halogens = halogens or []
    epoxy_bridges = epoxy_bridges or []
    n_alkyls = n_alkyls or []
    methoxys = methoxys or []

    # --- Build prefix parts in alphabetical order (IUPAC P-14.5) ---
    # Collect all prefix entries as (sort_key, prefix_str) for alphabetical ordering
    prefix_entries = []

    # Epoxy bridges (e.g., "4,5-epoxy")
    for loc1, loc2 in epoxy_bridges:
        prefix_entries.append(("epoxy", f"{loc1},{loc2}-epoxy"))

    # Halogen prefixes (bromo, chloro, fluoro, iodo)
    if halogens:
        # Group by halogen type
        hal_by_type: Dict[str, List[int]] = defaultdict(list)
        for loc, hal_name in halogens:
            hal_by_type[hal_name].append(loc)
        for hal_name in sorted(hal_by_type.keys()):
            locs = sorted(hal_by_type[hal_name])
            locant_str = ",".join(str(loc) for loc in locs)
            count = len(locs)
            multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
            prefix_entries.append((hal_name, f"{locant_str}-{multiplier}{hal_name}"))

    # Methoxy prefixes (e.g., "3-methoxy")
    if methoxys:
        locant_str = ",".join(str(loc) for loc in methoxys)
        count = len(methoxys)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        prefix_entries.append(("methoxy", f"{locant_str}-{multiplier}methoxy"))

    # Hydroxy prefix: added to prefix_entries for the ketone+hydroxyl path
    # (line 957-965) where hydroxyl is a non-principal group prefix.
    # When hydroxyl is the principal group (no ketones), the hydroxyl-only path
    # at line 942-949 uses non_oh_prefix (which strips hydroxy) and adds -ol suffix.
    # IUPAC P-35.2.1: principal group as suffix only.
    # IUPAC P-59.1: non-principal groups as prefixes only.
    if hydroxyls:
        locant_str = ",".join(str(loc) for loc in hydroxyls)
        count = len(hydroxyls)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        prefix_entries.append(("hydroxy", f"{locant_str}-{multiplier}hydroxy"))

    # Methyl prefix (C-methyl on scaffold carbons)
    if methyls:
        locant_str = ",".join(str(loc) for loc in methyls)
        count = len(methyls)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        prefix_entries.append(("methyl", f"{locant_str}-{multiplier}methyl"))

    # N-alkyl prefixes (e.g., "17-methyl" for N-methyl at position 17)
    for loc, alkyl_name in n_alkyls:
        prefix_entries.append((alkyl_name, f"{loc}-{alkyl_name}"))

    # Sort alphabetically by name (IUPAC P-14.5)
    prefix_entries.sort(key=lambda x: x[0])
    prefix = "-".join(entry[1] for entry in prefix_entries)

    # --- Build unsaturation suffix ---
    # IUPAC P-31.1.3.4: When the total number of unsaturation locants is >= 2,
    # a terminal 'a' is added to the stem for euphony (e.g., cholesta-5,7-dien,
    # not cholest-5,7-dien). For a single locant, no 'a' (cholest-5-en).
    ene_locs = unsaturation.get("ene", [])
    yne_locs = unsaturation.get("yne", [])
    total_unsat_locants = len(ene_locs) + len(yne_locs)

    # Add terminal 'a' to stem when multiple unsaturation locants (IUPAC P-31.1.3.4)
    effective_stem = stem
    if total_unsat_locants >= 2:
        effective_stem = stem + "a"

    unsat_suffix = ""
    if ene_locs or yne_locs:
        parts = []
        if ene_locs:
            ene_locant_str = ",".join(str(loc) for loc in ene_locs)
            ene_count = len(ene_locs)
            if ene_count == 1:
                parts.append(f"-{ene_locant_str}-en")
            else:
                ene_multi = SIMPLE_MULTIPLIERS.get(ene_count, str(ene_count))
                parts.append(f"-{ene_locant_str}-{ene_multi}en")
        if yne_locs:
            yne_locant_str = ",".join(str(loc) for loc in yne_locs)
            yne_count = len(yne_locs)
            if yne_count == 1:
                parts.append(f"-{yne_locant_str}-yn")
            else:
                yne_multi = SIMPLE_MULTIPLIERS.get(yne_count, str(yne_count))
                parts.append(f"-{yne_locant_str}-{yne_multi}yn")
        unsat_suffix = "".join(parts)
    else:
        # Fully saturated -> "an" infix
        unsat_suffix = "an"

    # --- Build ketone suffix ---
    ketone_suffix = ""
    if ketones:
        locant_str = ",".join(str(loc) for loc in ketones)
        count = len(ketones)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        ketone_suffix = f"-{locant_str}-{multiplier}one"

    # --- Combine suffix for -ol (hydroxyl as suffix) when no ketone ---
    # IUPAC convention: if hydroxyl is the only principal group, use -ol suffix
    # But if ketone is present, hydroxyl becomes a prefix
    # For NP names: hydroxyl is always prefix, ketone always suffix
    # If neither ketone nor hydroxyl, and no unsaturation -> bare scaffold name
    if not prefix and not ketone_suffix and not ene_locs and not yne_locs:
        if modification_prefix:
            return stereo_prefix + modification_prefix + scaffold_name
        return stereo_prefix + scaffold_name

    # Build non-OH prefix (methyl, halogen only) for use with -ol suffix
    non_oh_prefix_entries = [
        (key, val) for key, val in prefix_entries if key != "hydroxy"
    ]
    non_oh_prefix = "-".join(entry[1] for entry in non_oh_prefix_entries)

    # If only hydroxyls and no ketone -> use -ol suffix instead of prefix
    # Methyl/halogen prefixes are still included as prefixes
    if hydroxyls and not ketones:
        locant_str = ",".join(str(loc) for loc in hydroxyls)
        count = len(hydroxyls)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        ol_suffix = f"-{locant_str}-{multiplier}ol"
        # non-OH prefix + modification_prefix + effective_stem + unsaturation + -ol
        # e.g., "4-methylcholest-5-en-3-ol" or "cholesta-5,7-dien-3-ol"
        return f"{stereo_prefix}{non_oh_prefix}{modification_prefix}{effective_stem}{unsat_suffix}{ol_suffix}"

    # General case: prefix + modification_prefix + effective_stem + unsaturation + ketone
    # e.g., "17-hydroxy-19-norandr-4-en-3-one" for norethisterone-type
    if unsat_suffix == "an":
        # Saturated: prefix + modification_prefix + stem + "an" + ketone
        # IUPAC: terminal 'e' of "-ane" elided before vowel suffix (-one, -ol, -yl)
        # Keep 'e' only when no suffix follows (bare saturated name)
        if ketone_suffix:
            return f"{stereo_prefix}{prefix}{modification_prefix}{stem}{unsat_suffix}{ketone_suffix}"
        else:
            return f"{stereo_prefix}{prefix}{modification_prefix}{stem}{unsat_suffix}e"
    else:
        # Unsaturated: prefix + modification_prefix + effective_stem + unsaturation + ketone
        # If no suffix follows, add terminal 'e' (IUPAC: "ene"/"yne" not "en"/"yn")
        if ketone_suffix:
            return f"{stereo_prefix}{prefix}{modification_prefix}{effective_stem}{unsat_suffix}{ketone_suffix}"
        else:
            return f"{stereo_prefix}{prefix}{modification_prefix}{effective_stem}{unsat_suffix}e"


def _find_ester_decorations(
    mol, scaffold_info: Dict, numbering: Dict[int, int]
) -> List[Dict]:
    """Find ester groups (-O-C(=O)-R) attached to scaffold atoms.

    Detects esters where:
    - The ester oxygen is single-bonded to a scaffold carbon
    - The ester oxygen has 0 hydrogens (distinguishes from -OH)
    - The ester oxygen is bonded to a carbonyl carbon (C=O)

    For each ester found, identifies the acid fragment and gets its name.

    Args:
        mol: RDKit Mol object.
        scaffold_info: Dict from detect_natural_product().
        numbering: Target atom index to IUPAC locant mapping.

    Returns:
        List of dicts with keys:
            - locant: int (IUPAC locant on scaffold where ester attaches)
            - acylate: str (acylate name, e.g., "acetate", "propanoate")
            - all_atoms: set (all atom indices consumed by this ester group)
    """
    from ..data.trivial_acids import get_systematic_acylate

    esters = []
    matched_set = set(scaffold_info["matched_atoms"])

    subs = get_scaffold_substituents(mol, scaffold_info["matched_atoms"])
    for sub in subs:
        attach = sub["attachment_atom"]
        if attach not in numbering:
            continue

        # Ester starts with oxygen single-bonded to scaffold, no H
        first_idx = sub["first_atom"]
        first_atom = mol.GetAtomWithIdx(first_idx)
        if first_atom.GetAtomicNum() != 8:
            continue

        bond = mol.GetBondBetweenAtoms(attach, first_idx)
        if not bond or bond.GetBondType() != Chem.BondType.SINGLE:
            continue

        if first_atom.GetTotalNumHs() > 0:
            continue  # This is a hydroxyl, not an ester oxygen

        # Check that the ester oxygen connects to a carbonyl carbon
        carbonyl_idx = None
        for nbr in first_atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx == attach:
                continue  # Skip back to scaffold
            if nbr.GetAtomicNum() == 6:
                # Check for C=O (carbonyl)
                for nbr2 in nbr.GetNeighbors():
                    if nbr2.GetIdx() == first_idx:
                        continue
                    if nbr2.GetAtomicNum() == 8:
                        bond2 = mol.GetBondBetweenAtoms(nbr_idx, nbr2.GetIdx())
                        if bond2 and bond2.GetBondType() == Chem.BondType.DOUBLE:
                            carbonyl_idx = nbr_idx
                            break
            if carbonyl_idx is not None:
                break

        if carbonyl_idx is None:
            continue  # Not an ester (no C=O found after oxygen)

        # Count carbon atoms in the acid fragment
        # The acid fragment = carbonyl carbon + everything bonded to it
        # (except the ester oxygen back to scaffold)
        acid_carbons = _count_acid_fragment_carbons(
            mol, carbonyl_idx, first_idx, matched_set
        )

        # Get acylate name based on acid fragment size
        acylate_name = get_systematic_acylate(acid_carbons)

        # For C2 acid (acetic acid), use the common name "acetate"
        if acid_carbons == 2:
            acylate_name = "acetate"
        elif acid_carbons == 1:
            acylate_name = "formate"

        # Collect all atoms in this ester group (for exclusion from other detection)
        all_ester_atoms = set(sub["substituent_atoms"])

        esters.append({
            "locant": numbering[attach],
            "acylate": acylate_name,
            "all_atoms": all_ester_atoms,
        })

    return sorted(esters, key=lambda e: e["locant"])


def _count_acid_fragment_carbons(
    mol, carbonyl_idx: int, ester_oxy_idx: int, scaffold_atoms: set
) -> int:
    """Count carbon atoms in the acid fragment of an ester.

    BFS from the carbonyl carbon, excluding the ester oxygen path back to
    the scaffold. Counts only carbon atoms.

    Args:
        mol: RDKit Mol object.
        carbonyl_idx: Atom index of the carbonyl carbon.
        ester_oxy_idx: Atom index of the ester oxygen (excluded from BFS).
        scaffold_atoms: Set of scaffold atom indices (excluded from BFS).

    Returns:
        Number of carbon atoms in the acid fragment (including the carbonyl C).
    """
    visited = {carbonyl_idx, ester_oxy_idx}
    visited.update(scaffold_atoms)
    queue = deque([carbonyl_idx])
    carbon_count = 0

    while queue:
        current = queue.popleft()
        atom = mol.GetAtomWithIdx(current)
        if atom.GetAtomicNum() == 6:
            carbon_count += 1
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in visited:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    return carbon_count


def _acylate_to_acyloxy(acylate: str) -> str:
    """Convert an acylate name to acyloxy prefix form.

    Transforms the ester suffix (-ate) to the corresponding acyloxy prefix
    (-yloxy) for use in substitutive naming of mixed-acid esters.

    Examples:
        "acetate"    -> "acetyloxy"
        "propanoate" -> "propanoyloxy"
        "benzoate"   -> "benzoyloxy"
        "formate"    -> "formyloxy"

    Args:
        acylate: Acylate name (e.g., "acetate", "propanoate").

    Returns:
        Acyloxy prefix name.
    """
    if acylate.endswith("ate"):
        return acylate[:-3] + "yloxy"
    # Fallback (shouldn't happen for valid acylate names)
    return acylate + "yloxy"


def _assemble_np_ester_name(
    stem: str,
    scaffold_name: str,
    hydroxyls: List[int],
    ketones: List[int],
    unsaturation: Dict[str, List[int]],
    esters: List[Dict],
    stereo_prefix: str = "",
    methyls: Optional[List[int]] = None,
    halogens: Optional[List[Tuple[int, str]]] = None,
) -> str:
    """Assemble a functional class ester name for a natural product.

    Format: {stereo_prefix}{prefix}{stem}{unsaturation}-{locant}-yl {acylate}
    Example: "(5R,8S)-3-oxoandrost-4-en-17-yl acetate"

    In functional class format (IUPAC P-65.6):
    - Ketone groups become "oxo" prefixes (not "-one" suffix)
    - Hydroxyl groups become "hydroxy" prefixes
    - The scaffold ends with "-yl" at the ester attachment locant
    - The acylate name follows as a separate word

    For multiple esters: "parent-diyl diacetate" format.

    Args:
        stem: Scaffold stem (e.g., "androst").
        scaffold_name: Full scaffold name for fallback.
        hydroxyls: Sorted IUPAC locants of -OH groups.
        ketones: Sorted IUPAC locants of =O groups.
        unsaturation: Dict with 'ene' and 'yne' locant lists.
        esters: List of ester dicts with 'locant' and 'acylate' keys.
        stereo_prefix: Stereodescriptor prefix (e.g., "(5R,8S)-") or "".
        methyls: Optional sorted IUPAC locants of extra methyl groups.
        halogens: Optional sorted list of (locant, halogen_prefix) tuples.

    Returns:
        Functional class ester name string.
    """
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS

    methyls = methyls or []
    halogens = halogens or []

    # --- Build prefix (all non-ester decorations become prefixes) ---
    prefix_parts = []

    # Halogen prefixes
    if halogens:
        hal_by_type: Dict[str, List[int]] = defaultdict(list)
        for loc, hal_name in halogens:
            hal_by_type[hal_name].append(loc)
        for hal_name in sorted(hal_by_type.keys()):
            locs = sorted(hal_by_type[hal_name])
            locant_str = ",".join(str(loc) for loc in locs)
            count = len(locs)
            multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
            prefix_parts.append(f"{locant_str}-{multiplier}{hal_name}")

    # Hydroxyl groups -> "hydroxy" prefix
    if hydroxyls:
        locant_str = ",".join(str(loc) for loc in hydroxyls)
        count = len(hydroxyls)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        prefix_parts.append(f"{locant_str}-{multiplier}hydroxy")

    # Methyl prefixes
    if methyls:
        locant_str = ",".join(str(loc) for loc in methyls)
        count = len(methyls)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        prefix_parts.append(f"{locant_str}-{multiplier}methyl")

    # Ketone groups -> "oxo" prefix (NOT "-one" suffix in functional class format)
    if ketones:
        locant_str = ",".join(str(loc) for loc in ketones)
        count = len(ketones)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        prefix_parts.append(f"{locant_str}-{multiplier}oxo")

    # Join multiple prefix parts with hyphen: "3-hydroxy" + "7-oxo" -> "3-hydroxy-7-oxo"
    prefix = "-".join(prefix_parts)

    # --- Build unsaturation suffix ---
    ene_locs = unsaturation.get("ene", [])
    yne_locs = unsaturation.get("yne", [])

    unsat_suffix = ""
    if ene_locs or yne_locs:
        parts = []
        if ene_locs:
            ene_locant_str = ",".join(str(loc) for loc in ene_locs)
            ene_count = len(ene_locs)
            if ene_count == 1:
                parts.append(f"-{ene_locant_str}-en")
            else:
                ene_multi = SIMPLE_MULTIPLIERS.get(ene_count, str(ene_count))
                parts.append(f"-{ene_locant_str}-{ene_multi}en")
        if yne_locs:
            yne_locant_str = ",".join(str(loc) for loc in yne_locs)
            yne_count = len(yne_locs)
            if yne_count == 1:
                parts.append(f"-{yne_locant_str}-yn")
            else:
                yne_multi = SIMPLE_MULTIPLIERS.get(yne_count, str(yne_count))
                parts.append(f"-{yne_locant_str}-{yne_multi}yn")
        unsat_suffix = "".join(parts)
    else:
        unsat_suffix = "an"

    # --- Build yl suffix and acylate word ---
    ester_locants = [e["locant"] for e in esters]
    acylate_names = [e["acylate"] for e in esters]

    if len(esters) == 1:
        # Single ester: stem-unsaturation-locant-yl acylate
        yl_suffix = f"-{ester_locants[0]}-yl"
        acylate_word = acylate_names[0]
    elif len(set(acylate_names)) == 1:
        # Multiple esters with same acid: stem-unsaturation-locant,locant-diyl diacylate
        locant_str = ",".join(str(loc) for loc in ester_locants)
        count = len(esters)
        yl_multi = SIMPLE_MULTIPLIERS.get(count, str(count))
        yl_suffix = f"-{locant_str}-{yl_multi}yl"
        acyl_multi = SIMPLE_MULTIPLIERS.get(count, str(count))
        acylate_word = f"{acyl_multi}{acylate_names[0]}"
    else:
        # Multiple esters with DIFFERENT acids: use acyloxy prefix format
        # OPSIN rejects "diyl acid1 acid2" but accepts "(acyloxy)" prefix format
        # e.g., "3-(acetyloxy)-17-(propanoyloxy)androstane"
        acyloxy_parts = []
        for ester in esters:
            acyloxy = _acylate_to_acyloxy(ester["acylate"])
            acyloxy_parts.append(f"{ester['locant']}-({acyloxy})")

        # Sort alphabetically by acyloxy name for IUPAC ordering
        acyloxy_parts.sort(key=lambda p: p.split("(")[1])

        # Build acyloxy prefix string
        acyloxy_prefix = "-".join(acyloxy_parts)

        # Combine: all prefixes + acyloxy + stem + unsaturation + "e"
        all_prefix = "-".join(p for p in [prefix, acyloxy_prefix] if p)

        if unsat_suffix == "an":
            return f"{stereo_prefix}{all_prefix}{stem}{unsat_suffix}e"
        else:
            return f"{stereo_prefix}{all_prefix}{stem}{unsat_suffix}e"

    # --- Assemble: stereo_prefix + prefix + stem + unsaturation + yl + space + acylate ---
    # IUPAC: terminal 'e' of "-ane" elided before vowel suffix (-yl starts with 'y')
    if unsat_suffix == "an":
        parent = f"{stereo_prefix}{prefix}{stem}{unsat_suffix}{yl_suffix}"
    else:
        parent = f"{stereo_prefix}{prefix}{stem}{unsat_suffix}{yl_suffix}"

    return f"{parent} {acylate_word}"
