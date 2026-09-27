"""
IUPAC organometallic seniority + naming rules (a phase).

This module defines ORGM-specific seniority cascade SEPARATE from the
 carbon-organic seniority in src/orthonym/rules/seniority.py per
internal notes hard invariant (ZERO edits to seniority.py).

The two cascades NEVER share data structures; the only common file is
src/orthonym/data/organometallics.py (NEW; ORGM-exclusive).

Anti-patterns to avoid (PATTERNS lines 256-259):
- forbids any import path pointing at rules/seniority (relative
  ..rules dot-seniority or absolute orthonym dot rules dot seniority);
  this enforces ZERO cross-contamination with the carbon-organic
  seniority cascade. The grep gate in Plan-02 task 02-03 asserts neither
  import line exists in this module.
- NEVER add ORGM entries to the existing SENIORITY_ORDER list in
  seniority.py — define a NEW ORGM_LIGAND_ORDER in this file instead.
- NEVER call name_compound(...) recursively from
  assemble_organometallic_name(...) — recursion path is
  OUTER CFR → handler → rules → data; no upward call.

a phase (first scope-expansion phase per -07).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..assembly.name_tree import NameTreeNode

# Import from data layer only; forbids cross-contamination with
# the seniority cascade (no import of rules dot seniority).
from ..data.organometallics import (
    LIGAND_ETA_DEFAULTS,
    LIGAND_NAMES,
    METAL_NAMES,
)
from ..data.organometallics import RETAINED_METALLOCENES  # noqa: F401 re-exported: imported FROM this module by assembly/handlers/organometallic.py, scripts/verify_orgm_canary.py

# a phase Stock-notation Roman numerals (scope -3.. +8 per internal notes)
_ROMAN_NUMERALS: Dict[int, str] = {
    0: '0', 1: 'I', 2: 'II', 3: 'III', 4: 'IV',
    5: 'V', 6: 'VI', 7: 'VII', 8: 'VIII',
    -1: '-I', -2: '-II', -3: '-III',
}


def _to_roman(n: int) -> str:
    """Stock-notation oxidation state in Roman numerals.

    Per internal notes: a phase scope is -3..+8. Raises ValueError outside
    that range (caller catches per internal notes honest-fail-on-data).
    """
    if n not in _ROMAN_NUMERALS:
        raise ValueError(
            f"Stock-notation oxidation state {n} out of Phase 161 scope (-3..+8)"
        )
    return _ROMAN_NUMERALS[n]


# Multiplicative prefixes for simple ligand counts per Salzer
_SIMPLE_MULTIPLICATIVE_PREFIXES: Dict[int, str] = {
    1: '',
    2: 'di',
    3: 'tri',
    4: 'tetra',
    5: 'penta',
    6: 'hexa',
    7: 'hepta',
    8: 'octa',
}


def _multiplicative_prefix(count: int) -> str:
    """Return the multiplicative prefix for a simple ligand count."""
    if count not in _SIMPLE_MULTIPLICATIVE_PREFIXES:
        raise ValueError(f"ligand count {count} outside simple-prefix range (1..8)")
    return _SIMPLE_MULTIPLICATIVE_PREFIXES[count]


# Halide ligand name swaps for Grignard form (chlorido → chloride, etc.)
_HALIDE_LIGAND_TO_HALIDE_WORD: Dict[str, str] = {
    '[F-]':  'fluoride',
    '[Cl-]': 'chloride',
    '[Br-]': 'bromide',
    '[I-]':  'iodide',
}


def _ligand_name_from_atoms(mol: Any, atom_indices: Tuple[int, ...]) -> Optional[str]:
    """Identify the IUPAC name of an alkyl/aryl ligand from its atom indices.

    Recognised ligands (a phase scope; per LIGAND_NAMES + AUDIT):
    - methyl (1 C, all sp³)
    - ethyl (2 C, all sp³)
    - propyl / n-propyl (3 C, all sp³, linear)
    - butyl / n-butyl (4 C, all sp³, linear)
    - phenyl (6 aromatic C in a ring)
    - vinyl / ethenyl (2 C with C=C double bond)

    Returns None if the ligand doesn't match any known pattern.
    """
    n_atoms = len(atom_indices)
    if n_atoms == 0:
        return None

    atoms = [mol.GetAtomWithIdx(i) for i in atom_indices]
    symbols = [a.GetSymbol() for a in atoms]

    # All carbons?
    if not all(s == 'C' for s in symbols):
        return None

    aromatic_count = sum(1 for a in atoms if a.GetIsAromatic())

    # Phenyl: 6 aromatic carbons in a ring
    if n_atoms == 6 and aromatic_count == 6:
        return 'phenyl'

    # Ethenyl (CH2=CH-): a clean 2-carbon σ-ligand joined by a C=C double
    # bond. — 'ethenyl' is the PIN substituent prefix ('vinyl'
    # is retained, general-nomenclature only; both OPSIN-RT). This case was
    # listed in the docstring but the body never implemented it, so C=C[M]
    # silently collapsed to 'ethyl' — a structure-loss bug that dropped the
    # double bond (→ a different molecule, -suppressed in production).
    # Scope is the clean 2-carbon terminal vinyl ONLY; any larger / branched /
    # internal unsaturated ligand stays None (fail-closed) — not yet
    # confidently nameable.
    if n_atoms == 2 and aromatic_count == 0:
        bond = mol.GetBondBetweenAtoms(atom_indices[0], atom_indices[1])
        if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
            return 'ethenyl'

    # Alkyl groups: all sp³ carbons in a linear chain
    if aromatic_count == 0:
        # Check linearity: degree 1 carbons at the ends, degree 2 in middle
        # (the metal-attached carbon will have degree 1 since metal isn't in
        # atom_indices). For n_atoms == 1, it's methyl.
        if n_atoms == 1:
            return 'methyl'

        # Build internal degree map (degree within the ligand subgraph)
        idx_set = set(atom_indices)
        internal_degrees = {}
        for idx in atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            internal_degrees[idx] = sum(
                1 for b in atom.GetBonds()
                if b.GetOtherAtomIdx(idx) in idx_set
            )
        # tert-butyl: a quaternary sp3 carbon bonded to the metal and to three
        # methyls. The retained prefix 'tert-butyl' is a preferred prefix, used
        # unsubstituted in PINs, the Blue Book /:16286
        # '*tert*-butyldi(methyl)phosphane (PIN)'). Fail closed on any multiple
        # bond or other branching (TRIAGE g3 C12, ORG-T4-15 tert-butyllithium).
        if n_atoms == 4:
            _attach4 = [
                idx for idx in atom_indices
                if any(nb.GetAtomicNum() != 1 and nb.GetIdx() not in idx_set
                       for nb in mol.GetAtomWithIdx(idx).GetNeighbors())
            ]
            _all_single = all(
                b.GetBondTypeAsDouble() == 1.0
                for idx in atom_indices
                for b in mol.GetAtomWithIdx(idx).GetBonds()
                if b.GetOtherAtomIdx(idx) in idx_set
            )
            if (_all_single and len(_attach4) == 1
                    and internal_degrees[_attach4[0]] == 3
                    and all(internal_degrees[i] == 1
                            for i in atom_indices if i != _attach4[0])):
                return 'tert-butyl'

        # Linear chain: exactly 2 atoms with degree 1; rest with degree 2
        deg_counts = list(internal_degrees.values())
        if deg_counts.count(1) == 2 and deg_counts.count(2) == n_atoms - 2:
            # 0-wrong guard: the internal-degree multiset alone does NOT tell an
            # n-alkyl attached at a chain END (propyl) apart from a branched
            # isomer attached at an INTERNAL carbon (isopropyl, sec-butyl) — both
            # give the pattern {1, 1, 2,...}. Nor does it inspect bond order, so
            # an allyl / propargyl / butenyl backbone (all internal degree 2) read
            # as a saturated n-alkyl and the C=C / C#C was silently dropped. Both
            # cases fabricated a name for a DIFFERENT molecule. Fail closed unless
            # the fragment is a genuine all-single-bond chain whose metal-attached
            # atom is a terminus. (Any single ligand bond that is not order 1 also
            # excludes an aromatic ring here, but aromatics are handled above.)
            has_multiple_bond = any(
                b.GetBondTypeAsDouble() != 1.0
                for idx in atom_indices
                for b in mol.GetAtomWithIdx(idx).GetBonds()
                if b.GetOtherAtomIdx(idx) in idx_set
            )
            if has_multiple_bond:
                return None
            # The metal-attached ligand atom is the one with a heavy neighbour
            # OUTSIDE the ligand subgraph (a clean σ-alkyl has exactly one — the
            # metal). Require it to be a chain terminus (internal degree 1).
            attach = [
                idx for idx in atom_indices
                if any(nb.GetAtomicNum() != 1 and nb.GetIdx() not in idx_set
                       for nb in mol.GetAtomWithIdx(idx).GetNeighbors())
            ]
            if len(attach) != 1 or internal_degrees[attach[0]] != 1:
                return None
            return {2: 'ethyl', 3: 'propyl', 4: 'butyl',
                    5: 'pentyl', 6: 'hexyl', 7: 'heptyl',
                    8: 'octyl'}.get(n_atoms)

    return None


#: a bare -OH/-NH2/-SH bonded DIRECTLY to a Group-14
# parent-hydride centre (Si/Ge/Sn/Pb) is a principal characteristic group, named
# with a substitutive suffix on the hydride stem (silanol/silanamine/silanethiol)
# — NOT a ligand. element-Z + H-count -> the suffix word.
_GROUP14_PRINCIPAL_SUFFIX: Dict[Tuple[int, int], str] = {
    (8, 1): 'ol',      # -OH (hydroxy)
    (7, 2): 'amine',   # -NH2 (primary amine)
    (16, 1): 'thiol',  # -SH (sulfanyl)
}

#: the substitutive -ol/-amine/-thiol suffix mode is Group-14 ONLY
# (Si/Ge/Sn/Pb). Boron is ALSO a 'hydride_parent' naming system but its hydroxy
# acid is named with the boronic/borinic-acid characteristic group /,
# e.g. phenylboronic acid), NOT 'phenylboranediol' — so B must be EXCLUDED from
# the principal-group diversion and left to cascade to the boronic-acid handler.
_GROUP14_SUFFIX_ELEMENTS = frozenset({'Si', 'Ge', 'Sn', 'Pb'})

# Simple contracted alkoxy prefixes: named -oxy substituents that are
# NOT complex and take NO enclosing marks (methoxysilanetriol, not
# '(methoxy)silanetriol'). Compound -oxy prefixes (acyloxy/aryloxy/cyclyl-oxy such
# as 'acetyloxy'/'phenoxy'/'oxiranylmethoxy') stay complex -> parenthesized. The
# ligand namer's older ``name.endswith('oxy')`` blanket-wrapped these simple
# alkoxy names too; this whitelist restores (simple alkoxy = no marks).
_SIMPLE_ALKOXY_PREFIXES = frozenset({'methoxy', 'ethoxy', 'propoxy', 'butoxy'})


def _is_complex_ligand_oxy(name: str) -> bool:
    """True iff a ligand name is a COMPOUND -oxy prefix that needs enclosing marks
    : ends in 'oxy' but is not one of the simple contracted alkoxy names
    (methoxy/ethoxy/propoxy/butoxy). 'acetyloxy'/'phenoxy'/'oxiranylmethoxy' ->
    True; 'methoxy'/'ethoxy' -> False."""
    return name.endswith('oxy') and name.lower() not in _SIMPLE_ALKOXY_PREFIXES


def _principal_suffix_for_ligand(mol: Any, lg: Any) -> Optional[str]:
    """Return the substitutive suffix ('ol'/'amine'/'thiol') if ``lg`` is a bare
    neutral -OH/-NH2/-SH single-atom ligand, else ``None`` (fail-closed).

    Strictly single-atom O/N/S with exactly one heavy neighbour (the metal) and
    the canonical H-count, so ethers (Si-O-C, fragment has 2 atoms), secondary
    amines (Si-NH-C), metal-oxo (=O, 0 H), and siloxides (charged) all decline.
    """
    idxs = lg.ligand_atom_indices
    if len(idxs) != 1:
        return None
    atom = mol.GetAtomWithIdx(idxs[0])
    if atom.GetFormalCharge() != 0:
        return None
    heavy_deg = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() != 1)
    if heavy_deg != 1:
        return None
    return _GROUP14_PRINCIPAL_SUFFIX.get((atom.GetAtomicNum(), atom.GetTotalNumHs()))


# === ORGM_LIGAND_ORDER (Salzer 1999 alphabetic with multiplicatives ignored) ===
# This is NOT the carbon-organic seniority of — that stays UNTOUCHED
# in src/orthonym/rules/seniority.py per internal notes.
#
# Used for alphabetizing ligands in coordination names per RESEARCH
# Multiplicative prefixes (di-, tri-, tetra-, bis-, tris-, tetrakis-)
# are IGNORED for alphabetization; structural prefixes (cyclo-, η, κ, μ)
# are INCLUDED.
ORGM_LIGAND_ORDER: List[str] = [
    # Alphabetical ligand names per Salzer 1999
    'allyl',
    'benzene',
    'bromido',
    '1,3-butadiene',
    'carbonyl',
    'chlorido',
    'cycloheptatrienyl',
    'cyclooctatetraene',
    'cyclopentadienyl',
    'ethene',
    'ethyne',
    'fluorido',
    'hydrido',
    'iodido',
    'methyl',
    'pentamethylcyclopentadienyl',
    'phenyl',
    'tropylium',
]


# === METAL_OXIDATION_STATE_HINTS (per internal notes + AUDIT) ===
# (metal_symbol, ligand_class) → {default_state, stock_required, iupac_cite}
# Stock notation rule per Salzer 1999 + Red Book.
METAL_OXIDATION_STATE_HINTS: Dict[Tuple[str, str], Dict[str, Any]] = {
    # Tier 1 metallocenes (Cp2 ligand class; Stock required in systematic ONLY)
    ('Fe', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Ru', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Os', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Co', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Ni', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Cr', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('V',  'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Mn', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Fe', 'Cp2_cation'): {'default_state': 3, 'stock_required_systematic': True,
                            'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    # Tier 2 metal carbonyls (always include Stock per Salzer)
    ('Fe', 'CO5'): {'default_state': 0, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §6'},
    ('Ni', 'CO4'): {'default_state': 0, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §6'},
    ('Cr', 'CO6'): {'default_state': 0, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §6'},
    ('Mo', 'CO6'): {'default_state': 0, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §6'},
    ('W',  'CO6'): {'default_state': 0, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §6'},
    # Charged carbonyl anions (ALWAYS Stock per Ewens-Bassett or Stock convention)
    ('V',  'CO6_anion'): {'default_state': -1, 'stock_required_systematic': True,
                          'stock_required_pin': True, 'iupac_cite': 'Salzer §6'},
    ('Co', 'CO4_anion'): {'default_state': -1, 'stock_required_systematic': True,
                          'stock_required_pin': True, 'iupac_cite': 'Salzer §6'},
    ('Mn', 'CO5_anion'): {'default_state': -1, 'stock_required_systematic': True,
                          'stock_required_pin': True, 'iupac_cite': 'Salzer §6'},
    # Tier 3 σ-bonded main-group (NEVER Stock per — single canonical state)
    ('Li', 'alkyl'): {'default_state': 1, 'stock_required_systematic': False,
                      'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('Na', 'alkyl'): {'default_state': 1, 'stock_required_systematic': False,
                      'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('K',  'alkyl'): {'default_state': 1, 'stock_required_systematic': False,
                      'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('Mg', 'alkyl_halide'): {'default_state': 2, 'stock_required_systematic': False,
                              'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('Zn', 'alkyl2'): {'default_state': 2, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('Cd', 'alkyl2'): {'default_state': 2, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('Hg', 'alkyl2'): {'default_state': 2, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('Al', 'alkyl3'): {'default_state': 3, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    # Tier 3 hydride-parent main-group (Si/Ge/Sn/Pb; NEVER Stock — implicit +4)
    ('Si', 'alkyl4'): {'default_state': 4, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.2'},
    ('Ge', 'alkyl4'): {'default_state': 4, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.2'},
    ('Sn', 'alkyl4'): {'default_state': 4, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.2'},
    ('Pb', 'alkyl4'): {'default_state': 4, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.2'},
    # Tier 4 mixed sandwich + arene
    ('Cr', 'bz2'): {'default_state': 0, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Cr', 'bz_CO3'): {'default_state': 0, 'stock_required_systematic': True,
                       'stock_required_pin': False, 'iupac_cite': 'Salzer §5+§6'},
    ('Mn', 'Cp_CO3'): {'default_state': 1, 'stock_required_systematic': True,
                       'stock_required_pin': False, 'iupac_cite': 'Salzer §5+§6'},
    # Cycloheptatrienyl-Mn(CO)3 — Mn in +1 oxidation state per Salzer +
    ('Mn', 'CHT_CO3'): {'default_state': 1, 'stock_required_systematic': True,
                        'stock_required_pin': False, 'iupac_cite': 'Salzer §5+§6'},
}


# === METAL_RANKING_FOR_PARENT_SELECTION (per internal notes) ===
# When multiple metals present, which becomes the central-atom parent:
# defaults to highest-oxidation-state metal; ties broken by atomic number.
# a phase defers polynuclear bridges to a phase — this constant
# is present for forward compatibility; multinuclear compounds return None
# from name_organometallic per Risk R-08 mitigation.
METAL_RANKING_FOR_PARENT_SELECTION: Dict[str, int] = {
    'Li': 3, 'Na': 11, 'K': 19, 'Mg': 12,
    'Zn': 30, 'Cd': 48, 'Hg': 80,
    'B': 5, 'Al': 13, 'Ga': 31, 'In': 49, 'Tl': 81,
    'Si': 14, 'Ge': 32, 'Sn': 50, 'Pb': 82,
    'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Mo': 42, 'W': 74,
    'Ru': 44, 'Rh': 45, 'Pd': 46, 'Os': 76, 'Ir': 77, 'Pt': 78,
}

# W8-P9 Task 9.4: transition metals (Groups 3-12, per METAL_NAMES
# naming_system=='metal_direct') that get the NEW additive sigma-coordination
# branch when bearing both anionic and organic ligands. Deliberately EXCLUDES
# Zn/Cd/Hg (Group 12) -- those already have an established 'alkyl2' ligand
# class (dimethylzinc etc.) with their own Stock-notation hints; this new
# branch only fires for metals that previously fell through to `return None`
# (dead code path) for any halide+organic combination, so it cannot regress
# the existing Zn/Cd/Hg/Mg/Al/Li/Na/K forms.
_TRANSITION_METAL_DIRECT_SYMBOLS: frozenset = frozenset({
    'Ti', 'Zr', 'Hf', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
    'Mo', 'W', 'Ru', 'Os', 'Rh', 'Ir', 'Pd', 'Pt', 'Re',
})


def select_ligand_naming(ligand_group: Any, hapticity: int,
                         style: str = "pin") -> str:
    """Generate the IUPAC name for a single ligand with η-notation.

    STUB BODY (Plan-02): returns empty string. Plan-03 implements per
    RESEARCH + AUDIT SMARTS catalog.

    Algorithm (Plan-03 implementation):
    1. Look up ligand_group.ligand_canonical_smiles in LIGAND_NAMES.
    2. If hapticity > 1 OR ligand in LIGAND_ETA_DEFAULTS: prefix η<N>-.
    3. Apply multiplicative prefix (bis-, tris-, etc.) per outer caller.
    """
    return ""


# (item 4): reference molecular formula for every π-ligand NAME emitted via
# LIGAND_ETA_DEFAULTS, as (element -> atom count) INCLUDING hydrogens. The
# additive/η organometallic PINs ship through the hard-gated carve-out that
# BYPASSES OPSIN (namer._ORGANOMETALLIC_ADDITIVE_PIN_RE), so a lossy
# LIGAND_ETA_DEFAULTS row — one whose ligand NAME implies a different atom count
# than the coordinated FRAGMENT actually has — would silently emit a WRONG
# constitution with no round-trip net to catch it. Two such rows exist (measured,
# PREP-T4): neutral propene `C=CC` (C3H6) named as the η³-`prop-2-en-1-yl` anion
# (C3H5, 1 H short), and `C1=CC=CC=CC=1` (C7H6) named `cycloheptatrienyl` (C7H7).
# A ligand whose fragment formula ≠ its name's reference formula fails the veto,
# so the assembler declines and the cascade abstains ("<metal> compound (not
# supported)") rather than ship a name for atoms it does not describe —
# organometallics are out of scope. Only the π-ligand names can mismatch
# (σ-ligands are named from the fragment itself and conserve by construction, and
# the metal + charge partition is a tautology of metal_complex.formal_charges).
_PI_LIGAND_REFERENCE_FORMULA: Dict[str, Dict[str, int]] = {
    'cyclopentadienyl':            {'C': 5, 'H': 5},
    'methylcyclopentadienyl':      {'C': 6, 'H': 7},
    'pentamethylcyclopentadienyl': {'C': 10, 'H': 15},
    'benzene':                     {'C': 6, 'H': 6},
    '1,3-butadiene':               {'C': 4, 'H': 6},
    'prop-2-en-1-yl':              {'C': 3, 'H': 5},
    'allyl':                       {'C': 3, 'H': 5},
    'ethene':                      {'C': 2, 'H': 4},
    'ethyne':                      {'C': 2, 'H': 2},
    'cycloheptatrienyl':           {'C': 7, 'H': 7},
    'cyclooctatetraene':           {'C': 8, 'H': 8},
    'tropylium':                   {'C': 7, 'H': 7},
}


def _fragment_formula(mol: Any, atom_indices) -> Dict[str, int]:
    """Molecular formula (element -> count, hydrogens included) of the ligand
    fragment at ``atom_indices`` as it sits in ``mol``."""
    from collections import Counter
    c: "Counter[str]" = Counter()
    for i in atom_indices:
        a = mol.GetAtomWithIdx(i)
        c[a.GetSymbol()] += 1
        c['H'] += a.GetTotalNumHs()
    return dict(c)


def _organometallic_conserves(metal_complex: Any, mol: Any) -> bool:
    """0-wrong conservation guard for the additive/η carve-out (item 4).

    True iff every π-ligand (hapticity > 1) named via ``LIGAND_ETA_DEFAULTS`` has
    the SAME molecular formula as its ligand NAME's reference
    (``_PI_LIGAND_REFERENCE_FORMULA``). A mismatch means the emitted name would
    describe a different constitution than the input (measured lossy rows: allyl,
    cycloheptatrienyl) → return False so the assembler declines and the cascade
    abstains. σ-ligands and unrecognised π-ligand names are not vetoed here (an
    unknown name has no reference formula to check and must not force a false
    abstain on a conserving complex). NB the old claim that "σ-ligands conserve
    by construction" is FALSE for the transition-metal additive branch — a
    charged/hydrido metal, a charged/radical σ-carbon, or a 2-point ligand
    mis-named as monodentate all ship a wrong constitution; that branch now
    carries its own guard, ``_sigma_additive_ligands_certified``."""
    for lg in getattr(metal_complex, 'ligand_groups', ()) or ():
        hap = getattr(lg, 'hapticity_n', 1)
        if hap is None or hap <= 1:
            continue
        eta = LIGAND_ETA_DEFAULTS.get(getattr(lg, 'ligand_smarts_key', None))
        if eta is None:
            continue
        want = _PI_LIGAND_REFERENCE_FORMULA.get(eta[1])
        if want is None:
            continue
        if _fragment_formula(mol, lg.ligand_atom_indices) != want:
            return False
    return True


def _sigma_additive_ligands_certified(metal_complex: Any, mol: Any,
                                      organic_ligs) -> bool:
    """0-wrong certificate for the transition-metal additive σ-coordination
    branch (``trichlorido(methyl)titanium``). That name is emitted
    ONLY here and ships UNVERIFIED past OPSIN (the carve-out bypasses the RT
    gate), so every structural feature the name asserts must be proven here or
    the branch declines. This refutes the old ``_organometallic_conserves``
    premise that "σ-ligands conserve by construction" — for a σ-ligand the name
    also asserts the metal has no other bonds, the ligand is a NEUTRAL,
    radical-free, MONODENTATE group, and the metal carries neither charge nor a
    hydride, none of which the additive builder checked. Four measured wrong
    constitutions shipped as a result:
    - metal-bound H (M–H) is never rendered ('hydrido'); RDKit folds an implicit
      metal H into the metal's H-count, so ``C[TiH](Cl)Cl`` shipped
      'dichlorido(methyl)titanium' (a hydrido dropped). Abstain if the metal
      carries any H.
    - a charged metal needs an Ewens-Bassett number we do not emit
      (the IUPAC recommendations' own '...osmium(1+)' additive example), so
      ``C[Ti+](Cl)(Cl)Cl`` dropped the charge. Abstain if the metal is charged.
    - a σ-ligand atom that is charged or a radical is not the neutral group the
      name implies: ``[CH2-][Ti](Cl)(Cl)Cl`` (methanide) and ``[CH2][Ti]...``
      (radical) both shipped as neutral '(methyl)'. Abstain on either.
    - a ligand bonded to the metal at MORE than one atom is chelating/2-point,
      mis-named as monodentate: benzyne ``c1ccc2c(c1)[Ti]2(Cl)Cl`` → '(phenyl)'
      (C6H4 named C6H5), metallacyclopropene ``C1=C[Ti]1(Cl)Cl`` → '(ethenyl)'.
      Abstain unless every σ-ligand touches the metal at exactly one atom.
    abstain on all of these; these organometallics are out of
    scope, so declining (→ cascade abstains) is correct, never a wrong
    constitution."""
    metal_idxs = set(metal_complex.metal_atom_indices)
    for mi in metal_idxs:
        matom = mol.GetAtomWithIdx(mi)
        if matom.GetTotalNumHs() != 0:                       # metal-bound H
            return False
    if any(c != 0 for c in metal_complex.formal_charges):    # charged metal
        return False
    for lg in organic_ligs:
        idxs = list(lg.ligand_atom_indices)
        for i in idxs:                                        # neutral, no radical
            a = mol.GetAtomWithIdx(i)
            if a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() != 0:
                return False
        m_bonds = sum(                                       # monodentate: 1 M–L bond
            1 for i in idxs
            for nb in mol.GetAtomWithIdx(i).GetNeighbors()
            if nb.GetIdx() in metal_idxs
        )
        if m_bonds != 1:
            return False
    return True


def assemble_organometallic_name(metal_complex: Any, mol: Any,
                                  style: str = "pin"
                                  ) -> Optional[Tuple[str, str, List[Any]]]:
    """Assemble systematic IUPAC name for a metal complex per Salzer 1999

    Plan-03 implementation: Tier-1 Cp2 metallocene branch (commit 03-01).
    Subsequent commits 03-02/03/04 add Tier-3/2/4 branches.

    Returns: (full_name, metal_name_part, ligand_tree_nodes) per
    RESEARCH lines 799-815. The handler wraps this in NameTreeNode.

    Returns None to signal cascade-continuation when:
    - metal_complex.is_multimetal (a phase deferred per Risk R-08)
    - Stock-notation lookup fails for the (metal, ligand_class) tuple
    - The compound's topology doesn't match any tier branch yet
    """
    # Risk R-08: defer multinuclear bridges to a phase
    if metal_complex.is_multimetal:
        return None

    # (item 4): 0-wrong conservation veto — decline (→ cascade abstains)
    # rather than ship an η-name whose ligand word describes fewer atoms than the
    # coordinated fragment (the lossy prop-2-en-1-yl / cycloheptatrienyl
    # LIGAND_ETA_DEFAULTS rows). See _organometallic_conserves.
    if not _organometallic_conserves(metal_complex, mol):
        return None

    metal_idx = metal_complex.metal_atom_indices[0]
    metal_symbol = mol.GetAtomWithIdx(metal_idx).GetSymbol()
    metal_charge = metal_complex.formal_charges[0]
    ligand_groups = metal_complex.ligand_groups

    # === dispatch: mononuclear metal carbonyls ===
    if (ligand_groups
            and all(lg.ligand_smarts_key == '[C-]#[O+]' for lg in ligand_groups)):
        n_co = len(ligand_groups)
        co_prefix = _multiplicative_prefix(n_co)

        # Determine ligand_class for Stock lookup (neutral vs anionic)
        if metal_charge == 0:
            ligand_class = f'CO{n_co}'
        else:
            ligand_class = f'CO{n_co}_anion'

        hints = METAL_OXIDATION_STATE_HINTS.get((metal_symbol, ligand_class))
        if hints is None:
            return None

        metal_name_info = METAL_NAMES.get(metal_symbol)
        if metal_name_info is None:
            return None
        metal_name = metal_name_info['direct']

        include_stock = (
            hints['stock_required_systematic'] if style == 'systematic'
            else hints['stock_required_pin']
        )

        # Stock notation per internal notes:
        # - Systematic: always Roman (e.g., (0), (-I))
        # - PIN with anionic metals: Ewens-Bassett charge form (e.g., (1-))
        # - PIN with neutral metals: omit
        if include_stock:
            oxidation = metal_charge if metal_charge != 0 else hints['default_state']
            if metal_charge < 0 and style == 'pin':
                stock_str = f"({abs(metal_charge)}-)"
            elif metal_charge > 0 and style == 'pin':
                stock_str = f"({metal_charge}+)"
            else:
                stock_str = f"({_to_roman(oxidation)})"
        else:
            stock_str = ""

        # Salzer: ligand-first composition order — carbonyl prefix before metal
        full_name = f"{co_prefix}carbonyl{metal_name}{stock_str}"
        metal_name_part = f"{metal_name}{stock_str}"
        co_node = NameTreeNode(
            parent_stem='carbonyl',
            multiplicative_prefix=co_prefix or None,
            class_id='organometallic_ligand',
        )
        return (full_name, metal_name_part, [co_node])

    # === dispatch: bis(η⁵-Cp-class)M sandwich complexes ===
    # Match 2 identical Cp-class ligand groups (hapticity 5, both same key).
    if (len(ligand_groups) == 2
            and all(lg.hapticity_n == 5 for lg in ligand_groups)
            and ligand_groups[0].ligand_smarts_key == ligand_groups[1].ligand_smarts_key
            and ligand_groups[0].ligand_smarts_key is not None
            and ligand_groups[0].ligand_smarts_key in LIGAND_ETA_DEFAULTS):
        # Lookup the Cp variant name (cyclopentadienyl / methylcyclopentadienyl /
        # pentamethylcyclopentadienyl)
        _hap, cp_name = LIGAND_ETA_DEFAULTS[ligand_groups[0].ligand_smarts_key]

        # Cp2 ligand class; metal in +2 (or +3 for ferrocenium) oxidation state
        ligand_class = 'Cp2_cation' if metal_charge == 3 else 'Cp2'
        hints = METAL_OXIDATION_STATE_HINTS.get((metal_symbol, ligand_class))
        if hints is None:
            return None  # cascade to SALT@100

        metal_name_info = METAL_NAMES.get(metal_symbol)
        if metal_name_info is None:
            return None
        metal_name = metal_name_info['direct']

        include_stock = (
            hints['stock_required_systematic'] if style == 'systematic'
            else hints['stock_required_pin']
        )

        oxidation_state = metal_charge if metal_charge != 0 else hints['default_state']
        stock_str = f"({_to_roman(oxidation_state)})" if include_stock else ""

        # Salzer 1999: bis(η⁵-<cp_name>)<metal>(<Stock>)
        full_name = f"bis(η⁵-{cp_name}){metal_name}{stock_str}"
        metal_name_part = f"{metal_name}{stock_str}"

        ligand_node = NameTreeNode(
            parent_stem=f"η⁵-{cp_name}",
            multiplicative_prefix='bis',
            class_id='organometallic_ligand',
        )
        return (full_name, metal_name_part, [ligand_node])

    # === dispatch: σ-bonded main-group organometallics ===
    if all(lg.hapticity_n == 1 for lg in ligand_groups):
        metal_name_info = METAL_NAMES.get(metal_symbol)
        if metal_name_info is None:
            return None
        naming_system = metal_name_info['naming_system']

        # Partition into halide, (Group-14 only) principal-group, and organic
        # ligands.: a bare -OH/-NH2/-SH directly on a Si/Ge/Sn/Pb
        # centre is a principal characteristic group, not a ligand — divert it so
        # Branch A can emit it as a substitutive suffix (-> trimethylsilanol).
        # The diversion is Group-14-scoped so the metal-direct branch keeps its
        # exact prior behaviour (such ligands stay organic -> None -> cascade).
        is_group14 = metal_symbol in _GROUP14_SUFFIX_ELEMENTS
        halide_ligs: List[Any] = []
        pg_ligs: List[Any] = []
        organic_ligs: List[Any] = []
        for lg in ligand_groups:
            if lg.ligand_smarts_key in _HALIDE_LIGAND_TO_HALIDE_WORD:
                halide_ligs.append(lg)
            elif is_group14 and _principal_suffix_for_ligand(mol, lg) is not None:
                pg_ligs.append(lg)
            else:
                organic_ligs.append(lg)

        # Identify organic ligand names from atom indices. Wave-2 C2
        # extensions (hydride-parent Group-14 scope only, fail-closed):
        # (a) an O-attached ligand (R-O-[Si], NOT bare -OH — the G2
        # silanol suffix diversion above keeps priority) names as R-oxy
        # via the substituent chokepoint ('methoxy', 'oxiranylmethoxy');
        # (b) an all-C ligand the simple table rejects (tert-butyl) falls
        # back to the same chokepoint. Anything unresolvable stays None.
        def _extended_ligand_name(lg):
            simple = _ligand_name_from_atoms(mol, lg.ligand_atom_indices)
            if simple is not None:
                return simple
            atoms = list(lg.ligand_atom_indices)
            if not atoms:
                return None
            # find the ligand atom sigma-bonded to the central metal atom
            _central = set(metal_complex.metal_atom_indices)
            attach_idx = None
            for a in atoms:
                if any(nb.GetIdx() in _central
                       for nb in mol.GetAtomWithIdx(a).GetNeighbors()):
                    attach_idx = a
                    break
            if attach_idx is None:
                return None
            from ..assembly.substituent_enumerator import name_substituent
            a0 = mol.GetAtomWithIdx(attach_idx)
            # (a) Group-14 hydride-parent R-O-[Si] -> R-oxy (unchanged scope).
            if (is_group14 and naming_system == 'hydride_parent'
                    and a0.GetSymbol() == 'O' and a0.GetFormalCharge() == 0
                    and a0.GetTotalNumHs() == 0):
                inner = [i for i in atoms if i != attach_idx]
                c_start = [n.GetIdx() for n in a0.GetNeighbors()
                           if n.GetIdx() in inner]
                if len(c_start) != 1 or mol.GetAtomWithIdx(
                        c_start[0]).GetSymbol() != 'C':
                    return None
                nm = name_substituent(mol, set(inner), c_start[0])
                if not nm or nm == 'substituent' or not nm.endswith('yl'):
                    return None
                return nm[:-2] + 'oxy'
            # (b) a carbon-attached ligand the simple table rejects -> the general
            # substituent composer. TWO distinct scopes, deliberately asymmetric:
            # - Group-14 hydride-parent (Si/Ge/Sn/Pb): require the WHOLE ligand to
            # be all-carbon (tert-butyl). This is the ORIGINAL guard, restored:
            # a ligand carrying a heteroatom senior group (e.g. -CH2-COOH on Si)
            # must NOT be claimed here — the senior carboxylic acid parent wins
            # (protect: (trimethylsilyl)acetic acid, NOT carboxymethyl...silane).
            # - Group-12 metal-direct (Zn/Cd/Hg): the mixed-class ligand
            # may contain a WALKED-THROUGH class-2 metalloid (the Sb of a
            # 4-(diphenylstibanyl)phenyl ligand), so allow non-carbon atoms ONLY
            # when every one is itself a metal/metalloid (never an O/N/S senior
            # heteroatom). Gate on the sigma-attach atom being carbon.
            from ..perception.metals import is_metal_element
            _syms = [mol.GetAtomWithIdx(a).GetSymbol() for a in atoms]
            _g14_allcarbon = (
                is_group14 and naming_system == 'hydride_parent'
                and all(s == 'C' for s in _syms))
            _g12_metalloid_ok = (
                naming_system == 'metal_direct'
                and metal_symbol in ('Zn', 'Cd', 'Hg')
                and all(s == 'C' or is_metal_element(s) for s in _syms))
            if a0.GetSymbol() == 'C' and (_g14_allcarbon or _g12_metalloid_ok):
                nm = name_substituent(mol, set(atoms), attach_idx)
                if not nm or nm == 'substituent':
                    return None
                return nm
            return None

        organic_names: List[Optional[str]] = [
            _extended_ligand_name(lg) for lg in organic_ligs
        ]
        if any(n is None for n in organic_names):
            return None  # cascade to SALT@100

        # === Branch A: hydride-parent system (Group 14: Si/Ge/Sn/Pb) ===
        if naming_system == 'hydride_parent':
            parent_name = metal_name_info['hydride_parent']
            if parent_name is None:
                return None
            if halide_ligs:
                # Halide on Si/Ge/Sn/Pb: out of a phase scope (defer)
                return None
            # Group identical ligands
            grouped = _group_ligand_counts(organic_names)
            sorted_groups = _alphabetize_simple_ligands(grouped)
            # Wave-2 C2: a COMPOUND ligand name (compound R-oxy,
            # tert-, or ring-yl forms) takes enclosing marks at the join so
            # the boundary is unambiguous ('tert-butyl(dimethyl)[(oxiran-2-
            # yl)methoxy]silane' BB-style); simple methyl/ethyl stay bare
            # unless multiplied next to a compound neighbour.
            def _ligand_token(count, name):
                from ..assembly.naming_utils import (
                    has_structural_hyphen,
                    multiplier_needs_hyphen,
                    needs_p1634_marks,
                )
                mult = _multiplicative_prefix(count)
                #, SECOND leg of the same rule: a simple multiplier
                # joined to an italicized-prefix-led name keeps the hyphen
                # boundary — 'di-tert-butyl', never the malformed 'ditert-butyl'
                # this site shipped ('ditert-butylmethylsilane' for
                # CC(C)(C)[SiH](C)C(C)(C)C). naming_utils.format_substituent_prefix
                # already had this leg; fixing only the marks leg here in
                # left the two producers disagreeing, so both now call the one
                # primitive (multiplier_needs_hyphen).
                if mult and multiplier_needs_hyphen(name):
                    mult = f'{mult}-'
                # W3-P03-7 (c)/(d), BB 38222): a multiplied alkyl ligand
                # whose NAME begins with a numeric-multiplier syllable (decyl /
                # dodecyl..nonadecyl) takes enclosing marks so the multiplier is
                # not folded into the stem -- 'di(dodecyl)silane' (PIN) -- while
                # KEEPING the basic di/tri multiplier (the alkyl is not otherwise
                # complex, so it is NOT switched to bis). Gated on count > 1.
                if count > 1 and needs_p1634_marks(name):
                    return f'{mult}({name})'
                # /: the hyphen test is the SHARED
                # has_structural_hyphen, not a raw `'-' in name`. A leading
                # italicized 'tert-'/'sec-' is part of a SIMPLE retained prefix and
                # takes NO marks -- BB 16286 '*tert*-butyldi(methyl)phosphane' (PIN)
                # cites tert-butyl bare, as does BB 3465
                # `4-butyl-4-*tert*-butylcyclohexan-1-ol` (PIN) directly after a
                # locant. (An earlier comment here attributed an 'N-tert-butyl'
                # example to; that string does not occur in the Blue Book
                # and is the parentheses rule.) The raw hyphen test made this site the FOURTH
                # divergent copy of the compound predicate and emitted the non-PIN
                # '(tert-butyl)di(methyl)(oxiranylmethoxy)silane' -- the very form
                # the comment above quotes the Blue Book as writing bare.
                # 'tert-butyl-dimethylsilyl' still has a structural hyphen and is
                # still complex.
                def _compound_ligand(nm):
                    return (has_structural_hyphen(nm) or '(' in nm
                            or _is_complex_ligand_oxy(nm))

                if _compound_ligand(name):
                    inner = f'({name})' if '(' not in name else f'[{name}]'
                    return f'{mult}{inner}' if not mult else f'{mult}{inner}'
                if mult and any(_compound_ligand(n) for _c, n in sorted_groups):
                    return f'{mult}({name})'
                return f'{mult}{name}'

            _ligand_tokens = [
                (count, name, _ligand_token(count, name))
                for count, name in sorted_groups
            ]
            #: on a mononuclear parent hydride with 2+ DIFFERENT simple
            # ligands, the first cited is bare and each subsequent one is enclosed --
            # 'methyl(propyl)silanol', mirroring 'ethyl(methyl)(propyl)phosphane (PIN)'
            # (:7282). Only when every ligand is a single (count 1), non-compound
            # group that _ligand_token already left bare (tok == name); compound or
            # multiplied ligands are enclosed inside _ligand_token above and fall to
            # the plain join (keeps di(methyl)/tetramethyl/... canaries byte-identical).
            if (len(_ligand_tokens) >= 2
                    and all(c == 1 and tok == n for c, n, tok in _ligand_tokens)):
                from ..assembly.naming_utils import apply_enclosing_marks
                ligand_prefix = _ligand_tokens[0][2] + ''.join(
                    apply_enclosing_marks(tok, -1) for _c, _n, tok in _ligand_tokens[1:]
                )
            else:
                ligand_prefix = ''.join(tok for _c, _n, tok in _ligand_tokens)
            ligand_tree_nodes = [
                NameTreeNode(
                    parent_stem=name,
                    multiplicative_prefix=_multiplicative_prefix(count) or None,
                    class_id='organometallic_ligand',
                )
                for count, name in sorted_groups
            ]

            if pg_ligs:
                #: emit the bare -OH/-NH2/-SH on the
                # Group-14 centre as a substitutive suffix on the hydride stem
                # (trimethylsilanol, dimethylsilanediol, trimethylsilanamine,
                # trimethylsilanethiol). Every principal group must be the same
                # kind — a mixed -OH/-NH2 centre is ambiguous, so fail closed.
                suffix_kinds = {
                    _principal_suffix_for_ligand(mol, lg) for lg in pg_ligs
                }
                if len(suffix_kinds) != 1:
                    return None
                from ..assembly.naming_utils import (
                    _join_multiplied_suffix,
                    apply_vowel_elision,
                )
                suffix = suffix_kinds.pop()
                mult = _multiplicative_prefix(len(pg_ligs))  # '', 'di', 'tri'
                full_suffix = _join_multiplied_suffix(mult, suffix)
                parent_with_suffix = apply_vowel_elision(parent_name, full_suffix)
                # (BB 26221): the MONO-amine 'silanamine' is the one
                # documented exception to the general locant-omission rule for
                # substituted mononuclear parent hydrides -- (CH3)3Si-NH2 is
                # '1,1,1-trimethylsilanamine' (PIN), corroborated at BB 37493/37495,
                # citing every substituent's locant '1' on the sole skeletal atom.
                # The scope is DELIBERATELY narrow -- (a) (BB 2891) omits
                # '1' "in substituted mononuclear parent hydrides", which is why
                # every sibling omits its locants: 'trimethylsilanol' (BB 27234),
                # 'dimethylsilanediol', and crucially 'methylsilanetriamine'
                # (BB 38180, a TRI-amine) -- so the citation fires ONLY for a single
                # -NH2 principal group (len(pg_ligs) == 1) bearing substituents.
                # This centre is mononuclear (metal_atom_indices[0]), so every
                # ligand is at locant '1'.
                if suffix == 'amine' and len(pg_ligs) == 1 and sorted_groups:
                    locanted = '-'.join(
                        f"{','.join(['1'] * count)}-{_ligand_token(count, name)}"
                        for count, name in sorted_groups
                    )
                    full_name = f"{locanted}{parent_with_suffix}"
                else:
                    full_name = f"{ligand_prefix}{parent_with_suffix}"
                # No Stock notation (hydride-parent system implicit +4)
                return (full_name, parent_name, ligand_tree_nodes)

            full_name = f"{ligand_prefix}{parent_name}"
            # No Stock notation (hydride-parent system implicit +4)
            return (full_name, parent_name, ligand_tree_nodes)

        # === Branch B: metal-direct system (Groups 1/2/12/13) ===
        if naming_system == 'metal_direct':
            metal_name = metal_name_info['direct']

            # W8-P9 Task 9.4: additive sigma-coordination branch for
            # a TRANSITION metal (Groups 3-12, excluding Zn/Cd/Hg which already
            # have their own established alkyl2 form) bearing both anionic
            # ('-ido') ligands and organic ligands directly sigma-bonded.
            # BB verbatim (P6a.pdf: "[Ti(CH3)Cl3] trichlorido
            # (methanido)titanium trichlorido(methyl)titanium" -- ligands
            # (including 'hydrido' for M-H) cited in alphanumerical order,
            # then the metal name; PIN column uses the substitutive organic
            # ligand name ('methyl'), always parenthesized (coordination-
            # nomenclature convention -- every example in parenthesizes
            # the organic/substitutive ligand, even the simple 'methyl'/'ethyl').
            # No Stock number for a neutral complex (the BB Ti(IV) example
            # shows none; Ti(CH3)Cl3 is neutral in SMILES).
            # Fails closed (returns None, falls through to the ligand_class
            # dispatch below -> eventually cascades, backstopped by the Task
            # 9.2 veto) whenever it cannot verify every ligand structurally --
            # never emits a partial/atom-dropping additive name.
            if halide_ligs and organic_ligs and metal_symbol in _TRANSITION_METAL_DIRECT_SYMBOLS:
                if any(lg.ligand_smarts_key not in LIGAND_NAMES for lg in halide_ligs):
                    return None  # fail closed: unrecognised anionic ligand
                # 0-wrong certificate: this additive name bypasses OPSIN, so a
                # charged/hydrido metal, a charged/radical σ-ligand atom, or a
                # 2-point (chelating) ligand mis-named as monodentate would ship
                # a wrong constitution unchecked. Fail closed on any of these.
                if not _sigma_additive_ligands_certified(
                        metal_complex, mol, organic_ligs):
                    return None
                halide_names = [LIGAND_NAMES[lg.ligand_smarts_key] for lg in halide_ligs]
                # organic_names is already fully resolved + None-checked above.
                halide_grouped = _group_ligand_counts(halide_names)
                organic_grouped = _group_ligand_counts(organic_names)
                all_tokens: List[Tuple[str, str]] = []
                for count, name in halide_grouped:
                    all_tokens.append((name, f"{_multiplicative_prefix(count)}{name}"))
                for count, name in organic_grouped:
                    if count > 1:
                        mult = _COMPLEX_MULTIPLICATIVE_PREFIXES.get(count)
                        if mult is None:
                            return None  # fail closed: count outside supported range
                    else:
                        mult = ''
                    all_tokens.append((name, f"{mult}({name})"))
                all_tokens.sort(key=lambda t: t[0])
                ligand_prefix = ''.join(tok for _key, tok in all_tokens)
                full_name = f"{ligand_prefix}{metal_name}"
                ligand_tree_nodes = [
                    NameTreeNode(parent_stem=name, class_id='organometallic_ligand')
                    for name, _tok in all_tokens
                ]
                return (full_name, metal_name, ligand_tree_nodes)

            # Task M1 (v51): route the σ-bonded metal_direct topology to a
            # ligand_class. The pre-M1 dispatch enumerated the metal one by one
            # (Mg-Grignard / Zn|Cd|Hg-alkyl2 / Al-alkyl3 / Li|Na|K-alkyl) and
            # dropped EVERY other metal at `else: return None` -- which is why
            # `C[Mg]C` (dimethylmagnesium), `C[Ca]C`, `C[Be]C` and `C[U]`
            # (methyluranium) abstained though `C[Zn]C` named. The classes are
            # now topology-driven, not metal-hard-coded:
            # * Grignard `alkyl_halide`: exactly one halide + one organic
            # σ-ligand on a NON-transition metal_direct metal (transition
            # metals with halide+organic are owned by the additive branch
            # above; if it declined, such a compound still cascades here);
            # * `alkyl_sigma`: any all-organic σ-complex on a metal_direct
            # metal (Groups 1/2/12/13 + Group-3/lanthanide/actinide added to
            # METAL_NAMES in M1), built as `{multiplied-prefix}{metal}`.
            # Every emission is OPSIN round-trip-gated downstream, so a form that
            # does not round-trip abstains (0-wrong ABSOLUTE) rather than ship.
            if halide_ligs:
                if (len(halide_ligs) == 1 and len(organic_ligs) == 1
                        and metal_symbol not in _TRANSITION_METAL_DIRECT_SYMBOLS):
                    ligand_class = 'alkyl_halide'
                else:
                    # multi-halide / halide-only (MgCl2) / a transition-metal
                    # halide the additive branch declined -> cascade to SALT@100
                    return None
            elif organic_ligs:
                ligand_class = 'alkyl_sigma'
            else:
                # no σ-ligand at all (bare metal atom) -> cascade to SALT@100
                return None

            # Stock notation: every metal_direct row in METAL_OXIDATION_STATE_HINTS
            # is stock-free for BOTH styles (Grignard, dialkylzinc, trialkyl-
            # aluminium, alkyllithium all cite no Stock number), and the M1
            # best-effort metals default to no-Stock too. So include_stock is
            # uniformly False here -- byte-identical to the pre-M1 lookup for the
            # existing metals, which never emitted a Stock number.
            include_stock = False

            # Grignard: space-separated "ethylmagnesium bromide" form
            if ligand_class == 'alkyl_halide':
                organic_name = organic_names[0]  # exactly one organic ligand
                halide_word = _HALIDE_LIGAND_TO_HALIDE_WORD[halide_ligs[0].ligand_smarts_key]
                full_name = f"{organic_name}{metal_name} {halide_word}"
                metal_name_part = metal_name
                ligand_tree_nodes = [
                    NameTreeNode(parent_stem=organic_name,
                                 class_id='organometallic_ligand'),
                    NameTreeNode(parent_stem=halide_word,
                                 class_id='organometallic_ligand'),
                ]
                return (full_name, metal_name_part, ligand_tree_nodes)

            # mixed / complex σ-organic ligands (e.g.
            # [4-(diphenylstibanyl)phenyl](phenyl)mercury). The simple multiplied
            # form (diphenylmercury / dimethylzinc) applies only when the ligands
            # are ALL identical AND simple. Otherwise cite each ligand separately
            # in alphanumerical order, each enclosed — parentheses upgraded to
            # square brackets when the ligand name already contains enclosing
            # marks nesting ORDER (BB 7444; escalation, BB 7509) under the marks requirement (BB 7232)). Neutral Group-12 metal -> no Stock number.
            from ..assembly.naming_utils import (
                alpha_sort_key as _ask,
            )
            from ..assembly.naming_utils import (
                apply_enclosing_marks as _encl,
            )
            from ..assembly.naming_utils import (
                is_complex_substituent as _is_cx,
            )
            _distinct = set(organic_names)
            # The `-` leg of the old char-set test was a SIXTH copy of the compound
            # predicate and it re-opened, in this same file, the exact defect the
            # sibling `_compound_ligand` above was fixed for: 'tert-butyl' matched
            # `-`, so two IDENTICAL SIMPLE ligands were routed away from the
            # multiplied form and came back as '(tert-butyl)(tert-butyl)zinc'
            # instead of (b)/(d)'s 'di-tert-butylzinc'. `_is_cx` already carries
            # the correct hyphen semantics (it calls has_structural_hyphen), so the
            # raw leg is dropped and only the enclosing-mark characters remain.
            _any_complex = any(
                _is_cx(nm) or any(c in nm for c in '()[]') for nm in organic_names)
            if (len(_distinct) > 1 or _any_complex) and not include_stock:
                def _enclose_ligand(nm: str) -> str:
                    # nesting ORDER (BB 7444; escalation, BB 7509) under the marks requirement (BB 7232): a ligand name that ALREADY
                    # contains enclosing marks must be wrapped at the next level
                    # up (depth 1 = square brackets), e.g.
                    # '4-(diphenylstibanyl)phenyl' -> '[4-(diphenylstibanyl)phenyl]'.
                    # apply_enclosing_marks(depth=0) gives '', depth=1 ''.
                    if any(c in nm for c in '()[]'):
                        return _encl(nm, 1)       # nested -> bracket upgrade
                    return f"({nm})"
                ordered = sorted(organic_names, key=lambda nm: (_ask(nm), nm))
                ligand_prefix = ''.join(_enclose_ligand(nm) for nm in ordered)
                full_name = f"{ligand_prefix}{metal_name}"
                ligand_tree_nodes = [
                    NameTreeNode(parent_stem=nm, class_id='organometallic_ligand')
                    for nm in ordered
                ]
                return (full_name, metal_name, ligand_tree_nodes)

            # Single-component multi-alkyl (dimethylzinc, trimethylaluminum, etc.)
            grouped = _group_ligand_counts(organic_names)
            sorted_groups = _alphabetize_simple_ligands(grouped)
            # second leg, the THIRD multiplier-join site in this file:
            # 'di-tert-butylzinc', never the malformed 'ditert-butylzinc'. All three
            # call the one primitive (multiplier_needs_hyphen) so they cannot
            # disagree — fixing only _ligand_token would have left this path
            # emitting the malformed form for exactly the inputs the sibling fix
            # newly routed here.
            from ..assembly.naming_utils import multiplier_needs_hyphen as _mnh

            def _join_mult(count: int, name: str) -> str:
                mult = _multiplicative_prefix(count)
                if mult and _mnh(name):
                    return f'{mult}-{name}'
                return f'{mult}{name}'

            ligand_prefix = ''.join(
                _join_mult(count, name) for count, name in sorted_groups
            )

            # metal_direct σ-organometallics carry no Stock number in this scope
            # (include_stock is uniformly False here -- see the dispatch above).
            stock_str = ""

            full_name = f"{ligand_prefix}{metal_name}{stock_str}"
            metal_name_part = f"{metal_name}{stock_str}"
            ligand_tree_nodes = [
                NameTreeNode(
                    parent_stem=name,
                    multiplicative_prefix=_multiplicative_prefix(count) or None,
                    class_id='organometallic_ligand',
                )
                for count, name in sorted_groups
            ]
            return (full_name, metal_name_part, ligand_tree_nodes)

    # === dispatch: mixed π-ligand sandwich / half-sandwich ===
    # Fires AFTER Tier-1/2/3 (which handle specific topologies). Detected
    # when any ligand has hapticity > 1 (η-bonded).
    if any(lg.hapticity_n > 1 for lg in ligand_groups):
        return _assemble_tier4(metal_complex, mol, style)

    # No tier matched — cascade to SALT@100
    return None


# === Metallacycle skeletal-replacement namer (W8-P9 Task 9.5, =======

_METALLACYCLE_RING_STEM: Dict[int, str] = {
    3: 'cycloprop', 4: 'cyclobut', 5: 'cyclopent',
    6: 'cyclohex', 7: 'cyclohept', 8: 'cyclooct',
}

_METALLACYCLE_HALOGEN_PREFIX: Dict[str, str] = {
    'Cl': 'chloro', 'Br': 'bromo', 'F': 'fluoro', 'I': 'iodo',
}


def _metallacycle_ring_core_name(n: int, double_bond_locants: List[int]) -> Optional[str]:
    """Build the all-carbon von-Baeyer-style ring core name (metal replaced
    by the 'a'-prefix at position 1; ``n`` is the TOTAL ring size including
    the metal, matching the BB's own locant convention: 5-membered Pt ring
    -> 'cyclopenta-2,4-diene', not 'cyclobuta...' for the 4 carbons alone).
    Scope THIS CYCLE: 0-2 ring double bonds (>2 -- e.g. an aromatic
    metallabenzene -- is out of scope, fails closed)."""
    stem = _METALLACYCLE_RING_STEM.get(n)
    if stem is None:
        return None
    if len(double_bond_locants) == 0:
        return f"{stem}ane"
    if len(double_bond_locants) == 1:
        return f"{stem}-{double_bond_locants[0]}-ene"
    if len(double_bond_locants) == 2:
        locs = ','.join(str(x) for x in sorted(double_bond_locants))
        return f"{stem}a-{locs}-diene"
    return None


def _ring_position_substituents(mol: Any, ring_atom_idx: int,
                                 ring_set: Any) -> Optional[List[str]]:
    """Collect simple TERMINAL substituents (methyl / halogen) hanging off a
    single ring position (the metal position 1, or a ring carbon), excluding
    other ring atoms. Returns None (fail closed) if any substituent is not a
    simple recognised terminal group — narrow scope, W8-P9 Task 9.5; a
    fancier substituent (ethyl, aryl, phosphane ligand, etc.) is deliberately
    out of scope this cycle rather than risk a wrong/partial name."""
    names: List[str] = []
    atom = mol.GetAtomWithIdx(ring_atom_idx)
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx in ring_set:
            continue
        heavy_deg = sum(1 for x in nbr.GetNeighbors() if x.GetAtomicNum() != 1)
        if nbr.GetSymbol() == 'C':
            if heavy_deg != 1 or nbr.GetTotalNumHs() != 3 or nbr.GetFormalCharge() != 0:
                return None  # not a terminal methyl -- out of scope
            names.append('methyl')
        elif nbr.GetSymbol() in _METALLACYCLE_HALOGEN_PREFIX:
            if heavy_deg != 1 or nbr.GetFormalCharge() != 0:
                return None
            names.append(_METALLACYCLE_HALOGEN_PREFIX[nbr.GetSymbol()])
        else:
            return None
    return names


def _walk_ring_from(mol: Any, start_idx: int, second_idx: int,
                     ring_set: Any) -> Optional[List[int]]:
    """Traverse a simple (non-fused) ring starting at ``start_idx`` (the
    metal) via ``second_idx``, following the unique in-ring neighbour at
    each step. Returns the full ordered atom-index list (length ==
    len(ring_set)) or None if the ring branches internally (fused/bridged --
    not a simple monocycle, out of this task's scope)."""
    ordered = [start_idx, second_idx]
    prev, cur = start_idx, second_idx
    while True:
        atom = mol.GetAtomWithIdx(cur)
        nbrs_in_ring = [n.GetIdx() for n in atom.GetNeighbors()
                        if n.GetIdx() in ring_set and n.GetIdx() != prev]
        if len(nbrs_in_ring) != 1:
            return None
        nxt = nbrs_in_ring[0]
        if nxt == start_idx:
            break
        ordered.append(nxt)
        prev, cur = cur, nxt
        if len(ordered) > len(ring_set):
            return None
    return ordered if len(ordered) == len(ring_set) else None


def _build_metallacycle_candidate(
    mol: Any, ordered: List[int], ring_set: Any, a_prefix: str,
) -> Optional[Tuple[str, str, List[Any], Tuple[int, ...], Tuple[int, ...]]]:
    """Build one ring-numbering candidate's full name + its locant sets (for
    the lowest-locants tie-break between the two traversal directions).
    Returns (full_name, a_prefix, ligand_tree_nodes, double_bond_locants,
    substituent_locants) or None (fail closed) if this direction's ring
    bonds / substituents fall outside this task's narrow scope."""
    n = len(ordered)
    double_bond_locants: List[int] = []
    for i in range(n):
        a, b = ordered[i], ordered[(i + 1) % n]
        bd = mol.GetBondBetweenAtoms(a, b)
        if bd is None:
            return None
        bt = bd.GetBondTypeAsDouble()
        if bt == 2.0:
            double_bond_locants.append(i + 1)
        elif bt != 1.0:
            return None  # aromatic / triple ring bond -- out of scope

    ring_core = _metallacycle_ring_core_name(n, double_bond_locants)
    if ring_core is None:
        return None

    sub_groups: Dict[str, List[int]] = {}
    for i, atom_idx in enumerate(ordered):
        locant = i + 1
        subs = _ring_position_substituents(mol, atom_idx, ring_set)
        if subs is None:
            return None
        for name in subs:
            sub_groups.setdefault(name, []).append(locant)

    sub_prefix_tokens: List[str] = []
    all_sub_locants: List[int] = []
    for name in sorted(sub_groups.keys()):
        locants = sorted(sub_groups[name])
        all_sub_locants.extend(locants)
        mult = _multiplicative_prefix(len(locants))
        loc_str = ','.join(str(x) for x in locants)
        sub_prefix_tokens.append(f"{loc_str}-{mult}{name}")

    ring_token = f"1-{a_prefix}{ring_core}"
    full_name = '-'.join(sub_prefix_tokens + [ring_token])

    ligand_tree_nodes = [
        NameTreeNode(parent_stem=name, class_id='organometallic_ligand')
        for name in sorted(sub_groups.keys())
    ]
    return (
        full_name, a_prefix, ligand_tree_nodes,
        tuple(sorted(double_bond_locants)), tuple(sorted(all_sub_locants)),
    )


def _assemble_metallacycle(info: Any, mol: Any, style: str = "pin"
                            ) -> Optional[Tuple[str, str, List[Any]]]:
    """ skeletal-replacement metallacycle namer (W8-P9 Task 9.5, BUILT).

    BB verbatim (P6a.pdf: two acceptable names are given for a
    metallacycle — Hantzsch-Widman-type ('...platinole') and skeletal-
    replacement ('...-1-platinacyclopenta-2,4-diene'). Neither is labelled
    (PIN) — explicitly states that PINs for transition-metal
    organometallics await consideration by a future task group. Orthonym
    ships the skeletal-replacement form as the systematic/preferred style,
    consistent with the rest of this project's PIN-style target.

    Scope THIS CYCLE (narrow, conservative — matches ``detect_metallacycle``
    + ``_ring_position_substituents``): monocyclic all-carbon-backbone ring,
    0-2 ring double bonds, exocyclic ligands on the metal are simple
    terminal halides (named as ordinary 'chloro'-style substituent
    prefixes per the BB's own worked example — NOT '-ido' coordination
    ligand names, since this is the skeletal-replacement VIEW where the
    metal is just another numbered ring position), ring-carbon substituents
    are terminal methyl or halogen only. Fails closed (returns None,
    cascading onward — backstopped by the Task 9.2 structure-loss veto)
    outside that scope — e.g. an ethyl/aryl/phosphane ring substituent,
    more than 2 ring double bonds, or a numbering direction this narrow
    builder cannot resolve.
    """
    from ..data.organometallics import METALLACYCLE_A_PREFIX

    metal_idx = info.metal_atom_idx
    metal_symbol = mol.GetAtomWithIdx(metal_idx).GetSymbol()
    a_prefix = METALLACYCLE_A_PREFIX.get(metal_symbol)
    if a_prefix is None:
        return None

    ring_set = set(info.ring_atom_indices)
    metal_atom = mol.GetAtomWithIdx(metal_idx)
    ring_neighbors = [n.GetIdx() for n in metal_atom.GetNeighbors()
                     if n.GetIdx() in ring_set]
    if len(ring_neighbors) != 2:
        return None  # not a simple monocyclic ring position

    candidates = []
    for second_idx in ring_neighbors:
        ordered = _walk_ring_from(mol, metal_idx, second_idx, ring_set)
        if ordered is None:
            continue
        cand = _build_metallacycle_candidate(mol, ordered, ring_set, a_prefix)
        if cand is not None:
            candidates.append(cand)
    if not candidates:
        return None

    # lowest-locants tie-break: unsaturation locants first, then
    # substituent locants.
    candidates.sort(key=lambda c: (c[3], c[4]))
    full_name, metal_name_part, ligand_tree_nodes, _db, _sub = candidates[0]
    return (full_name, metal_name_part, ligand_tree_nodes)


def _assemble_tier4(metal_complex: Any, mol: Any,
                     style: str = "pin") -> Optional[Tuple[str, str, List[Any]]]:
    """Tier-4 assembly for mixed η-bonded / half-sandwich complexes.

    Per Salzer 1999 + the audit lock: CO (carbonyl) ligands are emitted
    BEFORE η-bonded ligands in the prefix sequence, even though strict
    Salzer alphabetic order would place benzene (b) before carbonyl (c).
    The audit-locked output convention reflects established IUPAC practice
    for organometallic carbonyl complexes (cf. Wikipedia "Tricarbonyl
    (η⁶-benzene)chromium").
    """
    metal_idx = metal_complex.metal_atom_indices[0]
    metal_symbol = mol.GetAtomWithIdx(metal_idx).GetSymbol()
    metal_charge = metal_complex.formal_charges[0]
    ligand_groups = metal_complex.ligand_groups

    metal_name_info = METAL_NAMES.get(metal_symbol)
    if metal_name_info is None:
        return None
    metal_name = metal_name_info['direct']

    # Partition ligands: CO + η-bonded
    co_ligands = [lg for lg in ligand_groups if lg.ligand_smarts_key == '[C-]#[O+]']
    pi_ligands = [lg for lg in ligand_groups if lg.ligand_smarts_key != '[C-]#[O+]']

    # Resolve each π-ligand to (hapticity, ligand_iupac_name)
    pi_names: List[Tuple[int, str]] = []
    for lg in pi_ligands:
        if lg.ligand_smarts_key is None or lg.ligand_smarts_key not in LIGAND_ETA_DEFAULTS:
            return None
        hap, name = LIGAND_ETA_DEFAULTS[lg.ligand_smarts_key]
        pi_names.append((lg.hapticity_n, name))

    # Style-aware ligand name overrides (PIN uses different names than systematic
    # for some ligands; per IUPAC + Salzer).
    if style == 'systematic':
        pi_names = [(hap, _SYSTEMATIC_LIGAND_NAME_OVERRIDES.get(name, name))
                    for hap, name in pi_names]

    # Group identical π-ligands (count by (hapticity, name) tuple)
    from collections import Counter
    pi_counter = Counter(pi_names)
    # Sort π-ligands alphabetically by name (ignoring multiplicative prefix)
    pi_sorted = sorted(pi_counter.items(), key=lambda x: x[0][1])

    # Determine if CO uses 'bis/tris/tetrakis'-style or simple multiplicative
    # Per Salzer: simple ligands use di/tri/tetra; complex ligands (with
    # parens/locants/η-prefix) use bis/tris/tetrakis.
    pi_prefix_parts: List[str] = []
    pi_tree_nodes: List[Any] = []
    for (hap, name), count in pi_sorted:
        eta = f"η{_superscript_int(hap)}-"
        ligand_str = f"({eta}{name})"
        if count == 1:
            pi_prefix_parts.append(ligand_str)
        else:
            mult = _COMPLEX_MULTIPLICATIVE_PREFIXES.get(count)
            if mult is None:
                return None
            pi_prefix_parts.append(f"{mult}{ligand_str}")
        pi_tree_nodes.append(NameTreeNode(
            parent_stem=f"{eta}{name}",
            multiplicative_prefix=_COMPLEX_MULTIPLICATIVE_PREFIXES.get(count) if count > 1 else None,
            class_id='organometallic_ligand',
            parenthesization_hint=True,
        ))

    # Carbonyl prefix
    co_prefix_str = ""
    co_tree_nodes: List[Any] = []
    if co_ligands:
        n_co = len(co_ligands)
        co_mult = _multiplicative_prefix(n_co)
        co_prefix_str = f"{co_mult}carbonyl"
        co_tree_nodes.append(NameTreeNode(
            parent_stem='carbonyl',
            multiplicative_prefix=co_mult or None,
            class_id='organometallic_ligand',
        ))

    # Stock notation: use metal's empirical formal charge as oxidation state.
    # For neutral metals (charge 0) bound to neutral π-ligands (benzene,
    # butadiene, ethene, COT), the metal is in 0 oxidation state.
    # For neutral metals bound to anionic ligands (Cp anion), the metal's
    # SMILES form would carry the corresponding positive charge already.
    # For cymantrene CpMn(CO)3 specifically: Mn is neutral in SMILES but the
    # complex has Mn(I); the audit lock applies via METAL_OXIDATION_STATE_HINTS.
    # We use the empirical charge if non-zero; otherwise look up hints.
    hints = None
    # Pick a ligand_class hint key based on the ligand mix
    if len(pi_ligands) == 1 and pi_ligands[0].ligand_smarts_key == 'c1ccccc1' and len(co_ligands) == 3:
        hints = METAL_OXIDATION_STATE_HINTS.get((metal_symbol, 'bz_CO3'))
    elif (len(pi_ligands) == 1 and pi_ligands[0].ligand_smarts_key == 'c1cc[cH-]c1'
          and len(co_ligands) == 3):
        hints = METAL_OXIDATION_STATE_HINTS.get((metal_symbol, 'Cp_CO3'))
    elif (len(pi_ligands) == 2 and len(co_ligands) == 0
          and all(lg.ligand_smarts_key == 'c1ccccc1' for lg in pi_ligands)):
        hints = METAL_OXIDATION_STATE_HINTS.get((metal_symbol, 'bz2'))
    elif (len(pi_ligands) == 1 and pi_ligands[0].ligand_smarts_key == 'C1=CC=CC=CC=1'
          and len(co_ligands) == 3):
        hints = METAL_OXIDATION_STATE_HINTS.get((metal_symbol, 'CHT_CO3'))

    if metal_charge != 0:
        effective_oxidation = metal_charge
    elif hints is not None:
        effective_oxidation = hints['default_state']
    else:
        effective_oxidation = 0  # neutral metal default

    # PIN: no Stock for these by default; systematic: include Stock
    include_stock = (style == 'systematic')
    if include_stock:
        try:
            stock_str = f"({_to_roman(effective_oxidation)})"
        except ValueError:
            stock_str = ""
    else:
        stock_str = ""

    # Compose full name: CO-prefix + π-ligands + metal + Stock
    full_name = f"{co_prefix_str}{''.join(pi_prefix_parts)}{metal_name}{stock_str}"
    metal_name_part = f"{metal_name}{stock_str}"
    return (full_name, metal_name_part, co_tree_nodes + pi_tree_nodes)


def _superscript_int(n: int) -> str:
    """Convert an integer to Unicode superscript digits."""
    digits = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '-': '⁻',
    }
    return ''.join(digits[c] for c in str(n))


# Complex-ligand multiplicative prefixes per Salzer (used when ligand
# contains parens / locants / η-prefix).
_COMPLEX_MULTIPLICATIVE_PREFIXES: Dict[int, str] = {
    2: 'bis', 3: 'tris', 4: 'tetrakis', 5: 'pentakis',
    6: 'hexakis', 7: 'heptakis', 8: 'octakis',
}


# Style-aware ligand name overrides for systematic forms.
# PIN uses the strict IUPAC-2013 substitutive name; systematic uses the
# traditional Salzer 1999 / Red Book organometallic-specific name.
_SYSTEMATIC_LIGAND_NAME_OVERRIDES: Dict[str, str] = {
    'prop-2-en-1-yl': 'allyl',  # T4-06 systematic: bis(η³-allyl)nickel(0)
}


def _group_ligand_counts(ligand_names: List[Optional[str]]) -> List[Tuple[int, str]]:
    """Group identical ligand names with their occurrence counts.

    Returns [(count, name),...]. Names are not yet sorted.
    """
    from collections import Counter
    counter = Counter(n for n in ligand_names if n is not None)
    return [(count, name) for name, count in counter.items()]


def _alphabetize_simple_ligands(
    grouped: List[Tuple[int, str]],
) -> List[Tuple[int, str]]:
    """Sort ligand entries alphabetically per Salzer

    Multiplicative prefixes (di-, tri-, tetra-) are IGNORED — alphabetize
    by the ligand name itself (the input here is already stripped).
    """
    # Wave-2 C2: alphabetize on the real first letter — the italicized structural
    # prefix is ignored: tert-butyl sorts at 'b').: this open-coded
    # `name[5:] if name.startswith('tert-')`, which handled 'tert-' and silently
    # MISSED 'sec-' (sec-butyl sorted at 's'), so it shares the one primitive.
    from ..assembly.naming_utils import strip_italicized_structural_prefix

    def _alpha_key(entry):
        remainder, _ = strip_italicized_structural_prefix(entry[1])
        return remainder

    return sorted(grouped, key=_alpha_key)


__all__ = [
    'ORGM_LIGAND_ORDER',
    'METAL_OXIDATION_STATE_HINTS',
    'METAL_RANKING_FOR_PARENT_SELECTION',
    'select_ligand_naming',
    'assemble_organometallic_name',
]
