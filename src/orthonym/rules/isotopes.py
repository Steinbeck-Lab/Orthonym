"""Isotopically substituted compound names (IUPAC P-82.2.1 + P-45.4).

Fail-closed decorator layer. Orthonym's core pipeline strips isotope labels
(RDKit perception ignores GetIsotope for skeleton naming), so a labeled mol
would otherwise emit the UNLABELED name — a wrong PIN. This module runs as an
EARLY branch in Orthonym.name(): it strips the labels, names the skeleton in
systematic style (locanted parents), then re-derives the isotopic descriptor
by an INTERNAL ORACLE (OPSIN-parse a candidate name; compare rdkit canonical
SMILES — with isotopes — to the original). If no candidate round-trips, it
returns None and the label is never mis-placed.

BB P-82.2.1 (BlueBookV2.md:43718): the nuclide symbol(s) in parentheses,
preceded by any necessary locant(s), are inserted before the isotopically
substituted part; polysubstitution count is a right subscript to the symbol.
BB P-45.4.1/.4.2/.4.3 (BlueBookV2.md:22212-22232): lowest locants to modified
positions; then to higher atomic number; then to higher mass number.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from rdkit import Chem


def has_isotopes(mol: Optional[Chem.Mol]) -> bool:
    """True iff any atom of ``mol`` carries a non-zero isotope label."""
    if mol is None:
        return False
    return any(a.GetIsotope() != 0 for a in mol.GetAtoms())


def strip_isotopes(mol: Chem.Mol) -> Tuple[Chem.Mol, Dict[int, int]]:
    """Return an isotope-cleared COPY of ``mol`` and a {atom_idx: mass} map.

    Atom indices are preserved by RWMol copy, so the returned map indexes the
    stripped mol AND the original identically. Used to name the skeleton and
    later to place isotopic descriptors at the correct positions.
    """
    copy = Chem.Mol(mol)
    label_map: Dict[int, int] = {}
    for atom in copy.GetAtoms():
        iso = atom.GetIsotope()
        if iso != 0:
            label_map[atom.GetIdx()] = iso
            atom.SetIsotope(0)
    # Deuterium/tritium ([2H]/[3H]) are EXPLICIT hydrogen atoms; clearing their
    # isotope leaves plain explicit [H] atoms, which change the canonical SMILES
    # ([H]C([H])([H])CO != CCO) and could perturb skeleton perception. Collapse
    # them back into the implicit H count so the skeleton is the true unlabeled
    # parent. RemoveHs is a no-op for heavy-atom labels (14C etc). label_map
    # keys index the ORIGINAL mol (element symbols are read there in Task 3), so
    # the index shift RemoveHs introduces on the returned copy is irrelevant —
    # the copy is used only for its canonical skeleton SMILES + heavy-atom count.
    try:
        copy = Chem.RemoveHs(copy)
    except Exception:
        pass
    return copy, label_map


# ---------------------------------------------------------------------------
# P-82.2.1 — isotopic descriptor formatting
# ---------------------------------------------------------------------------


def nuclide_symbol(mass: int, element: str) -> str:
    """Nuclide symbol with the mass number as a leading integer (ASCII form).

    Orthonym emits ASCII (no <sup>); OPSIN accepts the leading-digit form
    (probe: (2-14C)ethan-1-ol -> C([14CH3])O). P-82.2.1.
    """
    return f"{mass}{element}"


def format_isotope_descriptor(groups) -> str:
    """Build the parenthesized isotopic descriptor from grouped labels.

    ``groups`` = list of (locant, mass, element, count):
      locant None  -> no leading locant (single-position parent / front
                      descriptor, e.g. (2H3)methoxybenzene, (12C1)methane).
      locant int   -> the count locants are repeated then hyphen-joined to the
                      nuclide, e.g. (2,2,2-2H3), (2-14C1).
    Multiple groups at (possibly) the same place are cited alphabetically by
    element then by mass number (P-82.2.1 / P-82.3); groups are comma-joined.
    The count subscript is ALWAYS emitted (P-82.2.1 line 43720 + tritium
    parseability, Task 4).
    """
    def _one(locant, mass, element, count):
        sym = nuclide_symbol(mass, element)
        if locant is None:
            return f"{sym}{count}"
        # BB: the locant is repeated once per substituted atom at that position
        # for a grouped multi-count token (2,2,2-2H3); for count 1 a single
        # locant + hyphen (2-14C1).
        loc_part = ",".join(str(locant) for _ in range(count))
        return f"{loc_part}-{sym}{count}"

    # P-82.2.1: alphabetical by element symbol, then by mass number, then locant.
    ordered = sorted(groups, key=lambda g: (g[2], g[1], g[0] if g[0] is not None else -1))
    inner = ",".join(_one(*g) for g in ordered)
    return f"({inner})"


# ---------------------------------------------------------------------------
# P-45.4.1 / P-82.2.1 — internal-oracle isotopic-descriptor placement
# ---------------------------------------------------------------------------
# The skeleton name fixes the parent numbering, but Orthonym does NOT expose a
# reliable atom->locant map for acyclic parents (retrieve_confidence()
# ['atom_to_locant'] is None for CCO). Rather than re-perceive the chain
# independently (which risks diverging from the name's own numbering — a
# fail-closed violation), the decorator treats the skeleton name's locant space
# as opaque and lets OPSIN adjudicate: it enumerates candidate descriptor
# placements, OPSIN-parses each candidate, and accepts the one whose relabeled
# structure is rdkit-canonical-identical (isotopes retained) to the original.
# Because OPSIN itself renumbers to give the isotope its lowest legal locant,
# the round-trip test IS the P-45.4.1 / P-31.1.4.3.4(i) implementation — the
# wrong-locant candidate simply fails to reproduce the structure.

_ELEMENT_Z = {  # atomic numbers for P-45.4.2 tie-break
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "Cl": 17, "Se": 34, "Br": 35, "I": 53,
}


def _opsin_parse(candidate_name: str) -> Optional[str]:
    """Parse ``candidate_name`` to SMILES via the repo's shared OPSIN CLI
    helper (validation.opsin_roundtrip.opsin_parse — the SAME loader the name()
    validity gate uses). Fail-closed (None) on any error / no JAR."""
    try:
        from ..validation.opsin_roundtrip import opsin_parse
        return opsin_parse(candidate_name)
    except Exception:
        return None


def _isotope_round_trips(candidate_name: str, original_mol: Chem.Mol) -> bool:
    """True iff OPSIN parses ``candidate_name`` to a structure that is rdkit-
    canonical-identical (isotopes retained) to ``original_mol``. Fail-closed
    on parse failure / no JAR."""
    smi = _opsin_parse(candidate_name)
    if not smi:
        return False
    parsed = Chem.MolFromSmiles(smi)
    if parsed is None:
        return False
    try:
        return Chem.MolToSmiles(parsed) == Chem.MolToSmiles(original_mol)
    except Exception:
        return False


def _p4542_p4543_key(groups) -> tuple:
    """P-45.4.2 then P-45.4.3 ordering key for a candidate placement.

    Prefer the placement giving the lowest locant to the HIGHER atomic number
    (P-45.4.2), then to the HIGHER mass number (P-45.4.3). Implemented as: for
    each group sorted by locant ascending, emit (-Z, -mass); the lexicographically
    smallest key is the preferred placement. BB:22220 / BB:22226 / BB:3336(i).
    """
    key = []
    for locant, mass, el, _count in sorted(
        groups, key=lambda g: (g[0] if g[0] is not None else 1 << 30)
    ):
        z = _ELEMENT_Z.get(el, 0)
        key.append((-z, -mass))
    return tuple(key)


def decorate_isotopic_name(smiles: str, style: str, namer) -> Optional[str]:
    """Fail-closed isotopic-substitution PIN (P-82.2.1 + P-45.4).

    1. Parse + strip isotopes; if none, None.
    2. Name the skeleton in systematic style (locanted parents).
    3. Enumerate candidate descriptors (lowest-locant-first, P-45.4.1) and
       accept the first whose OPSIN round-trip reproduces the original mol.
    4. None if nothing round-trips (never a wrong labeled name).
    """
    original = Chem.MolFromSmiles(smiles)
    if original is None or not has_isotopes(original):
        return None
    stripped, label_map = strip_isotopes(original)
    if not label_map:
        return None
    stripped_smiles = Chem.MolToSmiles(stripped)
    # Skeleton in systematic style so parents carry locants (ethan-1-ol, not
    # ethanol) — the descriptor's locant needs a locanted parent.
    skeleton = namer.name(stripped_smiles) if style == "systematic" else None
    if skeleton is None:
        # Re-name in systematic regardless (the descriptor needs locants).
        from ..namer import Orthonym
        skeleton = Orthonym(style="systematic").name(stripped_smiles)
    if not skeleton or "unknown" in skeleton.lower():
        return None

    # Group labels by (mass, element) -> count; element read from the ORIGINAL
    # mol at each labeled atom index (label_map keys index the original).
    by_key: Dict[Tuple[int, str], int] = {}
    for idx, mass in label_map.items():
        el = original.GetAtomWithIdx(idx).GetSymbol()
        by_key[(mass, el)] = by_key.get((mass, el), 0) + 1

    n_pos = stripped.GetNumHeavyAtoms()

    # Alphabetical by element then mass (P-82.2.1 citation order).
    keys = sorted(by_key.items(), key=lambda kv: (kv[0][1], kv[0][0]))

    def _descriptor(locant):
        groups = [(locant, mass, el, count) for (mass, el), count in keys]
        return format_isotope_descriptor(groups)

    # P-82.2.1: the descriptor "is inserted before the part of the compound that
    # is isotopically substituted". For a whole-parent label that is the front of
    # the name; for a labeled part after substituent prefixes (the labeled C of
    # trichloro(12C1)methane) it sits before the parent stem. Orthonym does not
    # parse the skeleton's grammar, so it enumerates every insertion offset and
    # lets the OPSIN round-trip oracle pick the unique correct one (probe: only
    # ONE offset round-trips per (structure, descriptor)). Front (0) is tried
    # first so the front placement is preferred when it is valid.
    insert_offsets = [0] + [
        i for i in range(1, len(skeleton) + 1) if i < len(skeleton) and skeleton[i].isalpha()
    ]
    # dedupe preserving order (front-first)
    seen_off = set()
    offsets = []
    for o in insert_offsets:
        if o not in seen_off:
            seen_off.add(o)
            offsets.append(o)

    # Locant-form order (P-45.4.1 / P-14.3.4): the no-locant form (None) is
    # tried first — locants are omitted when the position is unambiguous; then
    # integer locants ascending (lowest-locant-first). Among round-trippers,
    # P-45.4.2/.4.3 (higher Z then higher mass at the lower locant) breaks ties.
    winners = []
    for locant in [None] + list(range(1, n_pos + 1)):
        desc = _descriptor(locant)
        for off in offsets:
            candidate = skeleton[:off] + desc + skeleton[off:]
            if _isotope_round_trips(candidate, original):
                groups = [(locant, mass, el, count) for (mass, el), count in keys]
                # locant None ranks as lowest for P-45.4.1 (unambiguous omit).
                loc_rank = -1 if locant is None else locant
                winners.append((_p4542_p4543_key(groups), loc_rank, off, candidate))
        if winners:
            # P-45.4.1: the first locant-form that yields ANY round-tripper is
            # the lowest-locant form (None precedes integers, integers ascend);
            # do not consider higher-locant forms once a lower one succeeds.
            break
    if not winners:
        return None
    # Among equally-low-locant round-trippers, P-45.4.2/.4.3 then front-first.
    winners.sort(key=lambda w: (w[0], w[1], w[2]))
    return winners[0][3]
