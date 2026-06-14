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


# ----------------------------------------------------------------------------
# Phase 160.1 D-19 + D-20 + ADR-19-05: IUPAC-canonical aromatic-ring ene
# locants for retained-NP scaffolds.
#
# Replaces Chem.Kekulize-based aromatic-ring ene detection (which is
# PYTHONHASHSEED-sensitive per Phase 145.2 D-09 — RDKit's Kekulize algorithm
# is deterministic within a process but sensitive to upstream atom-iteration
# order, producing different equally-valid Kekulé forms across processes).
#
# Each entry: list of (lower_locant, higher_locant, indicated_h_locant_or_None)
# tuples — one per canonical aromatic-ring double bond per IUPAC P-31.1.3.4.
# The (indicated_h_locant) is the parenthetical marker rendered per
# IUPAC P-31.1.4.3.4 when a double bond spans a ring-junction atom.
#
# For estrane, the canonical IUPAC PIN form is `1,3,5(10)-trien` — three
# aromatic-ring double bonds at C1=C2, C3=C4, C5=C10 (the (10) marker
# disambiguates between C5-C6 and C5-C10).
#
# Source: IUPAC Blue Book §P-31.1.3.4 + §P-31.1.4.3.4
# Cross-reference: opsin/opsin-core/.../steroidGroupNames.xml
# ----------------------------------------------------------------------------
_NP_AROMATIC_RING_LOCANTS: Dict[str, List[Tuple[int, int, Optional[int]]]] = {
    # estra-1,3,5(10)-triene: aromatic ring A of estrane (estradiol class).
    # Three canonical aromatic double bonds: 1=2, 3=4, 5=10 (the (10) marks
    # the ring-junction double bond per P-31.1.4.3.4; without it, the 5-en
    # is ambiguous between C5-C6 and C5-C10).
    "estrane": [(1, 2, None), (3, 4, None), (5, 10, 10)],
    # Future aromatic-NP scaffolds attach by adding rows here as Phase 161+
    # ships them; e.g.:
    #   "morphinan": [(1, 2, None), (3, 4, None), ...],
    #   "ergoline": [...],
}


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
                scaffold_class=scaffold_info.get("scaffold_name"),
            )
            if unsaturation["ene"] or unsaturation["yne"]:
                stereo_prefix, ring_ab = _collect_np_stereo(mol, numbering, scaffold_info)

                def _assemble_unsat(sp, rab):
                    return _assemble_np_name(
                        scaffold_info["scaffold_stem"],
                        scaffold_info["scaffold_name"],
                        hydroxyls=[],
                        ketones=[],
                        unsaturation=unsaturation,
                        stereo_prefix=sp,
                        modification_prefix=modification_prefix,
                        ring_ab=rab,
                    )

                name_ab = _assemble_unsat(stereo_prefix, ring_ab)
                # Phase 181 D-08: ship α/β only if it OPSIN-round-trips, else whole-graph R/S.
                if ring_ab and not _alpha_beta_rt_ok(mol, name_ab):
                    return _assemble_unsat(_whole_graph_rs_prefix(mol, numbering), {})
                return name_ab
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

    # Collect stereodescriptors from the scaffold numbering (steroid → α/β ring_ab + side R/S)
    stereo_prefix, ring_ab = _collect_np_stereo(mol, numbering, scaffold_info)

    # 0. Find ester decorations first (they consume atoms that would otherwise
    #    be counted as hydroxyls or ketones)
    esters = _find_ester_decorations(mol, scaffold_info, numbering)
    ester_consumed_atoms = set()
    for e in esters:
        ester_consumed_atoms.update(e["all_atoms"])

    # 0a/0b/0c. Phase 182 (WSC-03, D-02/D-03): conjugate decorations — sulfate ester,
    # mono-phosphate ester, glycosyl/uronyl. Each finder delegates fragment classification
    # to conjugate_controller.classify_conjugate and returns ester-shaped dicts carrying a
    # `word` key. Consuming these atoms here automatically dissolves the `androstan-3-olate`
    # mis-assignment (D-05): once the conjugate-O linker is excluded, the OH/oxo finders no
    # longer see it. Fires ONLY when a finder returns a non-empty list (D-09) — non-conjugate
    # inputs reach an empty `conjugates` list and stay byte-identical.
    conjugates = (
        _find_sulfate_conjugates(mol, scaffold_info, numbering)
        + _find_phosphate_conjugates(mol, scaffold_info, numbering)
        + _find_glycosyl_conjugates(mol, scaffold_info, numbering)
    )
    conjugate_consumed_atoms = set()
    for c in conjugates:
        conjugate_consumed_atoms.update(c["all_atoms"])

    # 0d. Find epoxy bridges (O bridging two scaffold atoms, e.g. morphine 4,5-epoxy)
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

    all_exclude = ester_consumed_atoms | epoxy_consumed | conjugate_consumed_atoms

    # 0e. Find methoxy groups (-OCH3) before hydroxyls (consumes O atoms)
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
        scaffold_class=scaffold_info.get("scaffold_name"),
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

    # Phase 182 (WSC-03, D-07): multi-conjugate seniority. The single-conjugate cohort is
    # binding this phase; if >1 conjugate fragment is present (e.g. a disulfate) there is no
    # confidently-nameable prefix form for the non-senior fragments yet, so honest-fail
    # (return None → systematic pipeline) rather than guess. The completeness invariant below
    # would in any case reject a name that drops the second fragment. Multi-conjugate prefix
    # treatment (sulfooxy/phosphonooxy/glycosyloxy) is Phase 183/184.
    if len(conjugates) > 1:
        return None

    # Phase 182 (WSC-03, D-07): a conjugate mixed with a real acyl ester (e.g. a steroid
    # 17-acetate 3-sulfate) is NOT expressible as a single functional-class join — the
    # `_assemble_np_ester_name` mixed-acid `else` branch keys on `ester["acylate"]`, which a
    # conjugate dict (carrying `word`) does not have. There is no confidently-correct PIN for
    # the mixed case yet (the non-senior fragment would need a sulfooxy/acyloxy prefix), so
    # honest-fail (None → systematic pipeline) rather than emit a wrong/partial name or rely on
    # a downstream KeyError. Mixed conjugate+ester is Phase 183/184 scope.
    if conjugates and esters:
        return None

    # If no decorations found, return bare scaffold name. Phase 182 (D-09): also require
    # `not conjugates` so a scaffold carrying ONLY a conjugate (and no other decoration)
    # still enters assembly instead of short-circuiting to the bare scaffold name.
    if (not hydroxyls and not ketones and not methyls and not halogens
            and not unsaturation["ene"] and not unsaturation["yne"]
            and not epoxy_bridges and not n_alkyls and not methoxys
            and not conjugates):
        if modification_prefix:
            return modification_prefix + scaffold_name
        return scaffold_name

    # Phase 182 (WSC-03, D-08): no-silent-drop completeness invariant. Runs ONLY when a
    # conjugate fired (D-09 — non-conjugate inputs are entirely untouched), so it can never
    # false-fail a name that does not go through the conjugate path. Assert every heavy atom
    # in a scaffold-substituent subgraph is claimed by exactly one named feature; any
    # unclaimed heavy atom → honest-fail (return None → systematic pipeline), NEVER emit a
    # name omitting atoms. Asserted over `substituent_atoms` (the get_scaffold_substituents
    # subgraphs), NEVER raw `non_scaffold_atoms` — the cholestane C20-C27 side chain folds
    # into the stem and would otherwise trigger a false honest-fail (Pitfall 3).
    if conjugates:
        # WARNING-4 atom exposure: the OH/ketone/methyl/methoxy/halogen/n_alkyl finders return
        # locants, not atom sets. Recompute each kind's consumed atoms locally from the SAME
        # get_scaffold_substituents subgraphs, claiming a sub's atoms only when it structurally
        # matches a decoration the corresponding finder actually reported. This is additive
        # (the finders' returns are untouched) and reuses the already-excluded membership, so
        # `claimed` can never disagree with `all_exclude`.
        hydroxyl_atoms: set = set()
        ketone_atoms: set = set()
        methyl_atoms: set = set()
        methoxy_atoms: set = set()
        halogen_atoms: set = set()
        n_alkyl_atoms: set = set()
        _hydroxyl_locs = set(hydroxyls)
        _ketone_locs = set(ketones)
        _methyl_locs = set(methyls)
        _methoxy_locs = set(methoxys)
        _halogen_locs = {loc for loc, _ in halogens}
        _n_alkyl_locs = {loc for loc, _ in n_alkyls}
        _HALOGEN_Z = {9, 17, 35, 53}
        all_subs = get_scaffold_substituents(mol, scaffold_info["matched_atoms"])
        for sub in all_subs:
            attach = sub["attachment_atom"]
            loc = numbering.get(attach)
            if loc is None:
                continue
            atoms = set(sub["substituent_atoms"])
            first_idx = sub["first_atom"]
            fa = mol.GetAtomWithIdx(first_idx)
            single_atom = len(atoms) == 1
            if single_atom and fa.GetAtomicNum() == 8 and fa.GetTotalNumHs() >= 1 \
                    and loc in _hydroxyl_locs:
                hydroxyl_atoms |= atoms
            elif single_atom and fa.GetAtomicNum() == 8 and fa.GetTotalNumHs() == 0 \
                    and loc in _ketone_locs:
                ketone_atoms |= atoms
            elif single_atom and fa.GetAtomicNum() == 6 and loc in _methyl_locs:
                methyl_atoms |= atoms
            elif single_atom and fa.GetAtomicNum() in _HALOGEN_Z and loc in _halogen_locs:
                halogen_atoms |= atoms
            elif fa.GetAtomicNum() == 8 and loc in _methoxy_locs:
                methoxy_atoms |= atoms
            elif fa.GetAtomicNum() == 6 and loc in _n_alkyl_locs \
                    and mol.GetAtomWithIdx(attach).GetAtomicNum() == 7:
                n_alkyl_atoms |= atoms

        claimed = (ester_consumed_atoms | epoxy_consumed | conjugate_consumed_atoms
                   | hydroxyl_atoms | ketone_atoms | methyl_atoms | methoxy_atoms
                   | halogen_atoms | n_alkyl_atoms)
        substituent_atoms = set()
        for s in all_subs:
            substituent_atoms |= set(s["substituent_atoms"])
        heavy_unclaimed = {a for a in substituent_atoms - claimed
                           if mol.GetAtomWithIdx(a).GetAtomicNum() > 1}
        if heavy_unclaimed:
            return None  # honest-fail → systematic pipeline; NEVER emit a name omitting atoms

    # 4/5. Assemble the name. `ring_ab` non-empty ⇒ the steroid α/β path fired; we ship α/β
    # ONLY if it OPSIN-round-trips, else fall back to the whole-graph R/S name (Phase 181 D-08).
    # This makes α/β strictly non-regressing: the few decorated steroids whose α/β name OPSIN
    # cannot parse (complex polysubstituted pregnanes) or whose decorated ring-fusion α/β is
    # ambiguous keep the existing R/S name that OPSIN does round-trip. Fail-OPEN when OPSIN is
    # absent (same posture as the SUB-03 validity gate) so the path is verifiable, not mandatory.
    def _assemble_steroid(sp, rab):
        # Phase 182 (WSC-03): a conjugate (sulfate / mono-phosphate / glycosyl) takes the
        # senior `-yl` attachment exactly like an ester (D-07); pass conjugates as
        # ester-shaped dicts carrying a `word` key into the proven `_assemble_np_ester_name`
        # emitter (which already demotes scaffold OH/=O to hydroxy/oxo prefixes and renders
        # α/β). For the binding single-conjugate cohort `esters` is empty, so `joined` is the
        # single conjugate; mixed conjugate+ester is honest-failed upstream (D-07 / >1 guard).
        joined = esters + conjugates
        if joined:
            return _assemble_np_ester_name(
                scaffold_stem, scaffold_name, hydroxyls, ketones,
                unsaturation, joined, stereo_prefix=sp,
                methyls=methyls, halogens=halogens, ring_ab=rab,
            )
        return _assemble_np_name(
            scaffold_stem, scaffold_name, hydroxyls, ketones, unsaturation,
            stereo_prefix=sp, methyls=methyls, halogens=halogens,
            epoxy_bridges=epoxy_bridges, n_alkyls=n_alkyls, methoxys=methoxys,
            modification_prefix=modification_prefix, ring_ab=rab,
        )

    name_ab = _assemble_steroid(stereo_prefix, ring_ab)
    if ring_ab and not _alpha_beta_rt_ok(mol, name_ab):
        # Steroid α/β did not OPSIN-round-trip → fall back to the whole-graph R/S name
        # (Phase 181 D-08). This single check covers the conjugate path too, since the
        # conjugate name carries α/β when ring_ab is present.
        fallback = _assemble_steroid(_whole_graph_rs_prefix(mol, numbering), {})
        # Phase 182 (D-10): for a conjugate we must NEVER ship a name that does not RT —
        # prior behaviour on these rows is already RT-False (the conjugate was dropped), so a
        # non-RT R/S fallback is no better. If the R/S conjugate name also fails RT, honest-fail
        # (None) to the systematic pipeline rather than ship a worse name. Fail-OPEN when OPSIN
        # is absent (the gate above already returned True, so we never reach here).
        if conjugates and not _alpha_beta_rt_ok(mol, fallback):
            return None
        return fallback
    # Phase 182 (D-10): a conjugate with no α/β (ring_ab empty) skipped the gate above; still
    # require the conjugate name to OPSIN-round-trip before shipping, else honest-fail.
    if conjugates and not ring_ab and not _alpha_beta_rt_ok(mol, name_ab):
        return None
    return name_ab


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


def _collect_np_stereo(mol, numbering: Dict[int, int], scaffold_info: Optional[Dict] = None):
    """Collect stereodescriptors for a natural product using IUPAC locants.

    Returns a 2-tuple ``(leading_prefix, ring_ab)``:

    - For a steroid where the P-101.2.6 α/β converter resolves (Phase 181, WSC-02):
      ``leading_prefix`` is the acyclic side-chain R/S block (e.g. ``"(22R)-"`` or "")
      and ``ring_ab`` is ``{locant: 'alpha'/'beta'}`` for the cited ring stereocentres.
      The assembler interleaves ``ring_ab`` INLINE at each locant + on the stem (D-04/D-06).
    - For a non-steroid NP, or a steroid where the converter signals the per-molecule
      no-mix fallback (D-08), ``leading_prefix`` is the existing whole-graph R/S block
      (e.g. ``"(5R,8S,...)-"``, byte-identical to prior behaviour) and ``ring_ab`` is ``{}``.

    Args:
        mol: RDKit Mol object with stereocenters.
        numbering: Dict mapping atom index -> IUPAC locant.
        scaffold_info: detect_natural_product() result; gates the steroid α/β branch.

    Returns:
        (leading_prefix: str, ring_ab: Dict[int, str]).
    """
    from ..rules.stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    assign_stereochemistry(mol)

    # Phase 181 (WSC-02): steroid ring-face α/β branch (D-08 gate). Root-cause at the
    # emitter; the converter round-trips through OPSIN by construction. None ⇒ fall back
    # to the existing whole-graph R/S string (never mix ring α/β with ring R/S).
    if scaffold_info is not None and scaffold_info.get("scaffold_class") == "steroid":
        from .steroid_stereo import collect_steroid_alpha_beta
        ab = collect_steroid_alpha_beta(mol, scaffold_info, numbering)
        if ab is not None:
            side_rs = ab.get("side_rs") or []
            leading = format_stereodescriptor_string(side_rs) if side_rs else ""
            return leading, ab.get("ring_ab", {})

    descriptors = collect_stereodescriptors(mol, numbering)
    leading = format_stereodescriptor_string(descriptors) if descriptors else ""
    return leading, {}


def _whole_graph_rs_prefix(mol, numbering: Dict[int, int]) -> str:
    """Build the legacy whole-graph R/S leading block (the α/β fallback name, Phase 181)."""
    from ..rules.stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    descriptors = collect_stereodescriptors(mol, numbering)
    return format_stereodescriptor_string(descriptors) if descriptors else ""


def _alpha_beta_rt_ok(mol, name: str) -> bool:
    """Return True iff the steroid α/β `name` OPSIN-round-trips to `mol` (Phase 181 D-08).

    Fail-OPEN: when OPSIN/Java is unavailable the round-trip cannot be checked, so we ship
    the α/β name (the same posture as the SUB-03 validity gate, which cannot suppress without
    OPSIN). When OPSIN IS available, an α/β name that does not round-trip is rejected so the
    caller falls back to the whole-graph R/S name — guaranteeing zero RT regression.
    """
    try:
        from ..validation.opsin_roundtrip import (
            opsin_roundtrip_check, _find_opsin_jar, _java_available,
        )
        if _find_opsin_jar() is None or not _java_available():
            return True
        smi = Chem.MolToSmiles(mol)
        return bool(opsin_roundtrip_check(smi, name).get("passed"))
    except Exception:
        return True


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
    scaffold_class: Optional[str] = None,
) -> Dict[str, object]:
    """Find double and triple bonds within the scaffold.

    Detects explicit DOUBLE and TRIPLE bonds directly.  For aromatic rings
    (IUPAC P-31.1.3.4), Kekulizes a *copy* of the molecule to resolve
    aromatic bonds into canonical alternating DOUBLE/SINGLE pattern, then
    counts the resulting DOUBLE bonds between aromatic C-C atoms.  The
    original molecule is never mutated.

    Aromatic bonds that are *inherent* to the scaffold pattern (e.g.
    morphinan Ring A, whose SMARTS already contains ``c`` atoms) are excluded
    because the base scaffold name already encodes that aromaticity.

    Phase 160.1 D-19 + D-20 + ADR-19-05: for retained-NP scaffolds with
    fully-aromatic rings (``scaffold_class in _NP_AROMATIC_RING_LOCANTS``),
    the canonical aromatic-ring ene locants come from the per-scaffold
    canonical-locant table BEFORE Kekulize. This produces IUPAC-canonical
    locants deterministically across all PYTHONHASHSEED values (replacing
    the previous Chem.Kekulize-based detection which was PYTHONHASHSEED-
    sensitive per Phase 145.2 D-09). The Kekulize path is preserved for
    novel aromatic substituents OUTSIDE the scaffold and for non-cataloged
    scaffold classes.

    Args:
        mol: RDKit Mol object.
        matched_set: Atom indices belonging to the scaffold.
        numbering: Map from atom index to IUPAC locant.
        scaffold_aromatic_atoms: Atom indices that are aromatic in the
            scaffold SMARTS pattern (from ``_build_scaffold_aromatic_atoms``).
            When ``None`` or empty, all aromatic C-C ene positions are counted.
        scaffold_class: Scaffold class label (e.g., 'estrane'). When present
            in ``_NP_AROMATIC_RING_LOCANTS``, the aromatic-ring ene locants
            come from the table; Kekulize is bypassed for atoms in
            ``scaffold_aromatic_atoms``.

    Returns a dict with:
      * 'ene' - sorted list of int IUPAC locants (lower locant of each
        unsaturated bond). PRESERVED for backwards compatibility.
      * 'yne' - sorted list of int IUPAC locants (lower locant of each
        triple bond).
      * 'ene_indicated_h' - Dict[int, int] mapping lower-locant to
        indicated-H locant per IUPAC P-31.1.4.3.4 (e.g., {5: 10} for the
        canonical estrane 5(10)-en); only populated for table-sourced
        aromatic-ring ene entries. Used by the locant-string renderer to
        emit the parenthetical marker.
    """
    if scaffold_aromatic_atoms is None:
        scaffold_aromatic_atoms = set()

    ene_locants: List[int] = []
    yne_locants: List[int] = []
    ene_indicated_h: Dict[int, int] = {}

    # Phase 160.1 D-19 + D-20 + ADR-19-05: deterministic aromatic-ring
    # canonical locants from the per-scaffold-class table when the scaffold
    # is cataloged AND the molecule has aromatic atoms inside the scaffold.
    # This bypasses the PYTHONHASHSEED-sensitive Chem.Kekulize call for
    # those atoms; the existing Kekulize path remains for atoms OUTSIDE
    # the scaffold's aromatic region.
    #
    # The trigger condition uses the molecule's RUNTIME aromaticity (any
    # scaffold-matched atom with GetIsAromatic() == True), NOT the SMARTS-
    # inherent aromaticity passed via scaffold_aromatic_atoms. This is
    # because steroid scaffolds (e.g., estrane) use C atoms in their SMARTS
    # (not aromatic c) but the actual molecule can have an aromatic Ring A
    # (estradiol). Runtime aromaticity detection is what determines whether
    # the canonical table should override Kekulize.
    table_handled_atoms: set = set()
    if (
        scaffold_class is not None
        and scaffold_class in _NP_AROMATIC_RING_LOCANTS
    ):
        # Identify scaffold atoms that are aromatic in the actual molecule.
        runtime_aromatic_atoms = {
            idx for idx in matched_set
            if mol.GetAtomWithIdx(idx).GetIsAromatic()
        }
        if runtime_aromatic_atoms:
            # Phase 160.2 Plan-04-02 WR-03 closure: ``high`` is intentionally
            # unused in the loop body (the ``_NP_AROMATIC_RING_LOCANTS`` table
            # schema declares (lower_locant, higher_locant, indicated_h) but
            # this site only consumes ``lower_locant`` and ``indicated_h``).
            # Rename to ``_high`` per Python unused-variable convention.
            for low, _high, indicated_h in _NP_AROMATIC_RING_LOCANTS[scaffold_class]:
                ene_locants.append(low)
                if indicated_h is not None:
                    ene_indicated_h[low] = indicated_h
            # Mark all runtime-aromatic scaffold atoms as handled so the
            # Kekulize loop below does not also count them.
            table_handled_atoms = runtime_aromatic_atoms

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
            # Phase 160.1 D-20: skip aromatic bonds already accounted for
            # by the canonical-locant table (atoms in table_handled_atoms).
            if (
                b_atom.GetIsAromatic() and e_atom.GetIsAromatic()
                and b_idx in table_handled_atoms
                and e_idx in table_handled_atoms
            ):
                continue  # Handled via _NP_AROMATIC_RING_LOCANTS above
            # For Kekulized aromatic bonds, check the scaffold-inherent filter
            if b_atom.GetIsAromatic() and e_atom.GetIsAromatic():
                if b_idx in scaffold_aromatic_atoms and e_idx in scaffold_aromatic_atoms:
                    continue  # Inherent scaffold aromaticity
            ene_locants.append(lower_loc)
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            yne_locants.append(lower_loc)

    return {
        "ene": sorted(ene_locants),
        "yne": sorted(yne_locants),
        "ene_indicated_h": ene_indicated_h,
    }


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
    ring_ab: Optional[Dict[int, str]] = None,
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
    ring_ab = ring_ab or {}

    # Phase 181 (WSC-02): render a ring locant with its ring-face α/β descriptor when one
    # was cited (Latin, no hyphen between locant and greek per P-101.2.6.1.1); else plain.
    def _greek_locant(loc):
        return f"{loc}{ring_ab[loc]}" if loc in ring_ab else str(loc)

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
            locant_str = ",".join(_greek_locant(loc) for loc in locs)
            count = len(locs)
            multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
            prefix_entries.append((hal_name, f"{locant_str}-{multiplier}{hal_name}"))

    # Methoxy prefixes (e.g., "3-methoxy")
    if methoxys:
        locant_str = ",".join(_greek_locant(loc) for loc in methoxys)
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
        locant_str = ",".join(_greek_locant(loc) for loc in hydroxyls)
        count = len(hydroxyls)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        prefix_entries.append(("hydroxy", f"{locant_str}-{multiplier}hydroxy"))

    # Methyl prefix (C-methyl on scaffold carbons)
    if methyls:
        locant_str = ",".join(_greek_locant(loc) for loc in methyls)
        count = len(methyls)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        prefix_entries.append(("methyl", f"{locant_str}-{multiplier}methyl"))

    # N-alkyl prefixes (e.g., "17-methyl" for N-methyl at position 17)
    for loc, alkyl_name in n_alkyls:
        prefix_entries.append((alkyl_name, f"{_greek_locant(loc)}-{alkyl_name}"))

    # Sort alphabetically by name (IUPAC P-14.5)
    prefix_entries.sort(key=lambda x: x[0])
    prefix = "-".join(entry[1] for entry in prefix_entries)

    # --- Build unsaturation suffix ---
    # IUPAC P-31.1.3.4: When the total number of unsaturation locants is >= 2,
    # a terminal 'a' is added to the stem for euphony (e.g., cholesta-5,7-dien,
    # not cholest-5,7-dien). For a single locant, no 'a' (cholest-5-en).
    ene_locs = unsaturation.get("ene", [])
    yne_locs = unsaturation.get("yne", [])
    # Phase 160.1 D-20 + ADR-19-05: indicated-H locant map for ring-junction
    # double bonds per IUPAC P-31.1.4.3.4 (e.g., {5: 10} renders as "5(10)").
    ene_indicated_h = unsaturation.get("ene_indicated_h", {})
    total_unsat_locants = len(ene_locs) + len(yne_locs)

    # Add terminal 'a' to stem when multiple unsaturation locants (IUPAC P-31.1.3.4)
    effective_stem = stem
    if total_unsat_locants >= 2:
        effective_stem = stem + "a"

    def _fmt_ene_locant(loc: int) -> str:
        """Render an ene locant with optional indicated-H marker.

        Per IUPAC P-31.1.4.3.4: ring-junction double bonds emit the
        parenthetical indicated-H locant (e.g., '5(10)'). Plain integer
        for non-junction locants.
        """
        h = ene_indicated_h.get(loc)
        if h is None:
            return str(loc)
        return f"{loc}({h})"

    unsat_suffix = ""
    if ene_locs or yne_locs:
        parts = []
        if ene_locs:
            ene_locant_str = ",".join(_fmt_ene_locant(loc) for loc in ene_locs)
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

    # Phase 181 (WSC-02): free ring-face descriptors not consumed by any substituent/suffix
    # locant (typically C-5, and any cited inverted bridgehead with no decoration) are
    # prepended to the stem as "{loc}{greek}-" — INLINE on the stem, NEVER a leading
    # parenthesised block (the OPSIN-unparseable anti-pattern). The side-chain R/S block
    # (stereo_prefix) stays at the very front (D-06).
    _consumed = (set(hydroxyls) | set(methyls) | set(methoxys) | set(ketones)
                 | {loc for loc, _ in halogens} | {loc for loc, _ in n_alkyls})
    _free = sorted(loc for loc in ring_ab if loc not in _consumed)
    stem_stereo = "".join(f"{loc}{ring_ab[loc]}-" for loc in _free)

    def _stemjoin(pre, body):
        """Splice the stem-stereo prefix between the prefix block and the stem core,
        inserting a hyphen when a non-empty prefix would otherwise abut a digit."""
        if stem_stereo and pre and not pre.endswith("-"):
            return f"{pre}-{stem_stereo}{body}"
        return f"{pre}{stem_stereo}{body}"

    # --- Combine suffix for -ol (hydroxyl as suffix) when no ketone ---
    # IUPAC convention: if hydroxyl is the only principal group, use -ol suffix
    # But if ketone is present, hydroxyl becomes a prefix
    # For NP names: hydroxyl is always prefix, ketone always suffix
    # If neither ketone nor hydroxyl, and no unsaturation -> bare scaffold name
    if not prefix and not ketone_suffix and not ene_locs and not yne_locs:
        # Bare scaffold (+ optional free ring-face config on the stem, e.g. "5alpha-cholestane").
        return f"{stereo_prefix}{_stemjoin(modification_prefix, scaffold_name)}"

    # Build non-OH prefix (methyl, halogen only) for use with -ol suffix
    non_oh_prefix_entries = [
        (key, val) for key, val in prefix_entries if key != "hydroxy"
    ]
    non_oh_prefix = "-".join(entry[1] for entry in non_oh_prefix_entries)

    # If only hydroxyls and no ketone -> use -ol suffix instead of prefix
    # Methyl/halogen prefixes are still included as prefixes
    if hydroxyls and not ketones:
        locant_str = ",".join(_greek_locant(loc) for loc in hydroxyls)
        count = len(hydroxyls)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        ol_suffix = f"-{locant_str}-{multiplier}ol"
        # non-OH prefix + modification_prefix + [stem-stereo] + effective_stem + unsaturation + -ol
        # e.g., "4-methylcholest-5-en-3-ol" or "5alpha-cholestan-3beta-ol"
        return f"{stereo_prefix}{_stemjoin(non_oh_prefix + modification_prefix, effective_stem)}{unsat_suffix}{ol_suffix}"

    # General case: prefix + modification_prefix + [stem-stereo] + effective_stem + unsaturation + ketone
    # e.g., "17beta-hydroxy-5alpha-androstan-3-one"
    if unsat_suffix == "an":
        # Saturated: prefix + modification_prefix + stem + "an" + ketone
        # IUPAC: terminal 'e' of "-ane" elided before vowel suffix (-one, -ol, -yl)
        # Keep 'e' only when no suffix follows (bare saturated name)
        if ketone_suffix:
            return f"{stereo_prefix}{_stemjoin(prefix + modification_prefix, stem)}{unsat_suffix}{ketone_suffix}"
        else:
            return f"{stereo_prefix}{_stemjoin(prefix + modification_prefix, stem)}{unsat_suffix}e"
    else:
        # Unsaturated: prefix + modification_prefix + effective_stem + unsaturation + ketone
        # If no suffix follows, add terminal 'e' (IUPAC: "ene"/"yne" not "en"/"yn")
        if ketone_suffix:
            return f"{stereo_prefix}{_stemjoin(prefix + modification_prefix, effective_stem)}{unsat_suffix}{ketone_suffix}"
        else:
            return f"{stereo_prefix}{_stemjoin(prefix + modification_prefix, effective_stem)}{unsat_suffix}e"


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


# ---------------------------------------------------------------------------
# Phase 182 (WSC-03): conjugate finders — sulfate / mono-phosphate / glycosyl.
#
# Each mirrors `_find_ester_decorations`: it walks `get_scaffold_substituents`,
# skips substituents whose attachment is not in the numbering, requires the linker
# atom to be an ester-type O (no H, so an -OH hydroxyl is never treated as a linker),
# and delegates the fragment classification to the class-agnostic
# `conjugate_controller.classify_conjugate` primitive (D-02). Detection is purely
# structural — never molecule-specific (no CHEBI literals, no name lookups).
#
# Returns `[{locant, kind, word, all_atoms, linker_kind}]` where `all_atoms` is the
# classifier's consumed-atom set (== `set(sub["substituent_atoms"])`), so the
# conjugate-consumed atoms are excluded from the OH/oxo/methyl/halogen finders (the
# same exclusion discipline as `ester_consumed_atoms`).
# ---------------------------------------------------------------------------

def _find_conjugates_of_kind(
    mol, scaffold_info: Dict, numbering: Dict[int, int], kind: str
) -> List[Dict]:
    """Shared conjugate finder body; `kind` ∈ {"sulfate","phosphate","glycoside"}."""
    from .conjugate_controller import classify_conjugate

    out: List[Dict] = []
    matched = set(scaffold_info["matched_atoms"])
    for sub in get_scaffold_substituents(mol, scaffold_info["matched_atoms"]):
        attach, first = sub["attachment_atom"], sub["first_atom"]
        if attach not in numbering:
            continue
        fa = mol.GetAtomWithIdx(first)
        if fa.GetAtomicNum() != 8 or fa.GetTotalNumHs() > 0:
            continue  # linker O, no H (an ester-O, not an -OH)
        info = classify_conjugate(mol, attach, first, matched)
        if info and info["kind"] == kind:
            out.append({"locant": numbering[attach], **info})
    return sorted(out, key=lambda e: e["locant"])


def _find_sulfate_conjugates(
    mol, scaffold_info: Dict, numbering: Dict[int, int]
) -> List[Dict]:
    """Detect `scaffold-O-S(=O)(=O)-O[H/⁻]` sulfate-ester conjugates (RESEARCH Pattern 1)."""
    return _find_conjugates_of_kind(mol, scaffold_info, numbering, "sulfate")


def _find_phosphate_conjugates(
    mol, scaffold_info: Dict, numbering: Dict[int, int]
) -> List[Dict]:
    """Detect `scaffold-O-P(=O)(O[H/⁻])(O[H/⁻])` mono-phosphate conjugates (RESEARCH Pattern 2).

    The classifier returns None for di/tri-phosphate or any P-O-P bridge, so those
    (out-of-scope, Phase 183/184) keep their legacy output.
    """
    return _find_conjugates_of_kind(mol, scaffold_info, numbering, "phosphate")


def _find_glycosyl_conjugates(
    mol, scaffold_info: Dict, numbering: Dict[int, int]
) -> List[Dict]:
    """Detect `scaffold-O-[anomeric C of a recognized sugar ring]` glycosyl/uronyl
    conjugates (RESEARCH Pattern 3); the `word` is the glycoside head
    (e.g. `beta-D-glucopyranosiduronic acid`)."""
    return _find_conjugates_of_kind(mol, scaffold_info, numbering, "glycoside")


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
    ring_ab: Optional[Dict[int, str]] = None,
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
    ring_ab = ring_ab or {}

    # Phase 181 (WSC-02): render a ring locant with its ring-face α/β descriptor (Latin,
    # no locant-greek hyphen) when cited; else plain.
    def _greek_locant(loc):
        return f"{loc}{ring_ab[loc]}" if loc in ring_ab else str(loc)

    # --- Build prefix (all non-ester decorations become prefixes) ---
    prefix_parts = []

    # Halogen prefixes
    if halogens:
        hal_by_type: Dict[str, List[int]] = defaultdict(list)
        for loc, hal_name in halogens:
            hal_by_type[hal_name].append(loc)
        for hal_name in sorted(hal_by_type.keys()):
            locs = sorted(hal_by_type[hal_name])
            locant_str = ",".join(_greek_locant(loc) for loc in locs)
            count = len(locs)
            multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
            prefix_parts.append(f"{locant_str}-{multiplier}{hal_name}")

    # Hydroxyl groups -> "hydroxy" prefix
    if hydroxyls:
        locant_str = ",".join(_greek_locant(loc) for loc in hydroxyls)
        count = len(hydroxyls)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        prefix_parts.append(f"{locant_str}-{multiplier}hydroxy")

    # Methyl prefixes
    if methyls:
        locant_str = ",".join(_greek_locant(loc) for loc in methyls)
        count = len(methyls)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        prefix_parts.append(f"{locant_str}-{multiplier}methyl")

    # Ketone groups -> "oxo" prefix (NOT "-one" suffix in functional class format)
    if ketones:
        locant_str = ",".join(_greek_locant(loc) for loc in ketones)
        count = len(ketones)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        prefix_parts.append(f"{locant_str}-{multiplier}oxo")

    # Join multiple prefix parts with hyphen: "3-hydroxy" + "7-oxo" -> "3-hydroxy-7-oxo"
    prefix = "-".join(prefix_parts)

    # Phase 181 (WSC-02): free ring-face descriptors not consumed by any prefix/ester locant
    # (e.g. "5alpha-") are prepended to the stem INLINE (never a leading parenthesis).
    _ester_consumed_base = (set(hydroxyls) | set(methyls) | set(ketones)
                            | {loc for loc, _ in halogens})

    def _ester_stemjoin(pre, body, ester_locs):
        consumed = _ester_consumed_base | set(ester_locs)
        free = sorted(loc for loc in ring_ab if loc not in consumed)
        stereo = "".join(f"{loc}{ring_ab[loc]}-" for loc in free)
        if stereo and pre and not pre.endswith("-"):
            return f"{pre}-{stereo}{body}"
        return f"{pre}{stereo}{body}"

    # --- Build unsaturation suffix ---
    ene_locs = unsaturation.get("ene", [])
    yne_locs = unsaturation.get("yne", [])
    # Phase 160.1 D-20 + ADR-19-05: indicated-H locant map for ring-junction
    # double bonds per IUPAC P-31.1.4.3.4 (e.g., {5: 10} renders as "5(10)").
    ene_indicated_h = unsaturation.get("ene_indicated_h", {})

    def _fmt_ene_locant_ester(loc: int) -> str:
        """Render an ene locant with optional indicated-H marker (ester path)."""
        h = ene_indicated_h.get(loc)
        if h is None:
            return str(loc)
        return f"{loc}({h})"

    unsat_suffix = ""
    if ene_locs or yne_locs:
        parts = []
        if ene_locs:
            ene_locant_str = ",".join(
                _fmt_ene_locant_ester(loc) for loc in ene_locs
            )
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
    # Phase 182 (WSC-03): conjugate dicts carry a `word` key (e.g. "sulfate",
    # "beta-D-glucopyranosiduronic acid") and have NO "acylate". Prefer `word` so the
    # single-conjugate functional-class join `{parent-yl} {word}` works without KeyError;
    # real ester dicts still resolve via their `acylate` key.
    ester_locants = [e["locant"] for e in esters]
    acylate_names = [e.get("word") or e.get("acylate") for e in esters]

    if len(esters) == 1:
        # Single ester: stem-unsaturation-locant-yl acylate
        yl_suffix = f"-{_greek_locant(ester_locants[0])}-yl"
        acylate_word = acylate_names[0]
    elif len(set(acylate_names)) == 1:
        # Multiple esters with same acid: stem-unsaturation-locant,locant-diyl diacylate
        locant_str = ",".join(_greek_locant(loc) for loc in ester_locants)
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
            acyloxy_parts.append(f"{_greek_locant(ester['locant'])}-({acyloxy})")

        # Sort alphabetically by acyloxy name for IUPAC ordering
        acyloxy_parts.sort(key=lambda p: p.split("(")[1])

        # Build acyloxy prefix string
        acyloxy_prefix = "-".join(acyloxy_parts)

        # Combine: all prefixes + acyloxy + [stem-stereo] + stem + unsaturation + "e"
        all_prefix = "-".join(p for p in [prefix, acyloxy_prefix] if p)
        ester_consumed = set(ester_locants)
        return f"{stereo_prefix}{_ester_stemjoin(all_prefix, stem, ester_consumed)}{unsat_suffix}e"

    # --- Assemble: stereo_prefix + prefix + [stem-stereo] + stem + unsaturation + yl + space + acylate ---
    # IUPAC: terminal 'e' of "-ane" elided before vowel suffix (-yl starts with 'y')
    parent = f"{stereo_prefix}{_ester_stemjoin(prefix, stem, set(ester_locants))}{unsat_suffix}{yl_suffix}"

    return f"{parent} {acylate_word}"
