"""Substitutive N-nitro / nitramide naming (v36 Milestone B3, Task 4).

BLUE BOOK AUTHORITY
-------------------
``nitramide`` (H2N-NO2) exists in this tree ONLY as an exact-whole-molecule
RETAINED name (``data/opsin_imports/simple_groups.py:5679``, merged into
``ALL_RETAINED_NAMES``). Substitution of its amide nitrogen's hydrogens is
the SAME **P-66.1.1.3.1.1 "N-Substitution"** pattern already built for
sulfonamides (``rules/sulfonamides.py``) and thioimides
(``rules/thioimides.py``):

    "Substituted primary amides, with general structures such as R-CO-NHR'
    and R-CO-NR'R'' ... are named by citing the substituents R' and R'' as
    prefixes preceded by the locant *N* when one amide group is present."

THE DEFECT THIS CLOSES
-----------------------
Any SUBSTITUTED nitramide has a different canonical SMILES than the bare
retained-table key and misses that table entirely -- nothing downstream ever
built the substitutive parent, so every N-substituted nitramide fell through
to GENERAL/decomposition (which named a *fragment*, e.g. ``methanol`` for
``O=[N+]([O-])NCO``, dropping the nitro-amide unit) and was suppressed by
SELF-01 to the ``unknown organic compound`` sentinel.

V36-SPY-B3 section 2 ("N-nitro" rows) verified every target form below
OPSIN-round-trip-exact (full InChIKey match, constitution + charge).

TWO SHAPES BUILT HERE (fails closed on everything else -- never a
per-molecule special case)
------------------------------------------------------------------
1. **``nitramide`` PARENT + N-substituent prefix(es).** Exactly one
   nitro-bearing amide nitrogen in the whole molecule, with 0-2
   carbon-rooted substituents (mirrors ``sulfonamides._collect_n_substituents``'s
   guard: an N-heteroatom substituent is a different construction and is
   refused)::

       O=[N+]([O-])NCO      -> N-(hydroxymethyl)nitramide
       O=[N+]([O-])N(CO)CO  -> N,N-bis(hydroxymethyl)nitramide

2. **``N,N'-dinitro<diamine>``.** EXACTLY two nitro-bearing amide nitrogens,
   each with NO other substituent, both attached to the SAME bridging heavy
   atom which itself carries no other substituent (structurally forced to
   be a plain methylene bridge -- the only real-witness shape verified)::

       O=[N+]([O-])NCN[N+](=O)[O-] -> N,N'-dinitromethanediamine

   The structural guards (degree-2 carbon, both neighbours being the two
   qualifying amide N's, non-aromatic, ring-free) leave exactly ONE possible
   parent hydride -- methanediamine -- so it is returned directly rather
   than re-entering the general naming pipeline (which does not yet build
   ANY gem-diamine parent, a separate pre-existing gap out of scope here;
   see ``_name_dinitro_diamine``). Scoped NARROWLY to this one shape: a
   numeric-locant diamine (``propane-1,3-diamine``) would need
   superscripted ``N^1``/``N^3`` dinitro locants (P-66.1.1.3.1.1) that are
   not built here -- fails closed rather than guess.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from rdkit import Chem

_NITRO_SMARTS = Chem.MolFromSmarts('[NX3+](=O)[O-]')


def _nitro_amide_units(mol) -> List[Dict[str, int]]:
    """Every ``R2N-[N+](=O)[O-]`` unit: a nitro group whose N is directly
    bonded to an otherwise-neutral NITROGEN (the "amide" nitrogen this
    module is about) rather than to carbon (ordinary nitroalkane/nitroarene
    -- a completely different, already-working class this must never
    touch). Returns one ``{'nitro_n': idx, 'amide_n': idx}`` dict per
    independent unit.
    """
    if mol is None or _NITRO_SMARTS is None:
        return []
    units: List[Dict[str, int]] = []
    for match in mol.GetSubstructMatches(_NITRO_SMARTS):
        n_no2 = match[0]
        atom = mol.GetAtomWithIdx(n_no2)
        others = [nb for nb in atom.GetNeighbors() if nb.GetIdx() not in match]
        if len(others) != 1:
            continue
        amide = others[0]
        if amide.GetAtomicNum() != 7 or amide.GetFormalCharge() != 0:
            continue
        units.append({'nitro_n': n_no2, 'amide_n': amide.GetIdx()})
    return units


def _name_single_nitramide(mol, amide_n: int, nitro_n: int, style: str) -> Optional[str]:
    """Shape 1: ``nitramide`` parent + N-substituent prefix(es)."""
    from .sulfonamides import _branch_atoms

    n_atom = mol.GetAtomWithIdx(amide_n)
    branches: List[List[int]] = []
    for nb in n_atom.GetNeighbors():
        if nb.GetIdx() == nitro_n:
            continue
        if nb.GetAtomicNum() != 6:
            # N-heteroatom substituent (N-amino, N-hydroxy, N-N...) is a
            # different construction -- refuse rather than guess.
            return None
        branch = _branch_atoms(mol, nb.GetIdx(), amide_n)
        if nitro_n in branch:
            return None  # loops back through the nitro group -- refuse
        branches.append(branch)

    if not branches:
        return None  # plain nitramide -- owned by the retained-name table
    if len(branches) > 2:
        return None  # a trivalent amide N cannot carry 3 C substituents
    if len(branches) == 2 and set(branches[0]) & set(branches[1]):
        return None  # shared atoms -> a ring through the amide N

    # Structural coverage proof: this nitro-amide unit + its branches must
    # be the WHOLE molecule (never fires when a second, independent
    # nitro-amide unit exists elsewhere -- _name_dinitro_diamine owns that).
    covered = {nitro_n, amide_n}
    for branch in branches:
        covered.update(branch)
    for nb in mol.GetAtomWithIdx(nitro_n).GetNeighbors():
        covered.add(nb.GetIdx())
    if len(covered) != mol.GetNumAtoms():
        return None

    from .amides import _name_n_substituent

    names: List[str] = []
    for branch in branches:
        name = _name_n_substituent(mol, branch, len(branch))
        if not name:
            return None
        names.append(name)

    # Fail-closed splice guard (mirrors format_n_substitution): a refusal
    # sentinel must never be woven into the emitted name.
    from ..errors import is_refusal_sentinel
    if any(is_refusal_sentinel(n) for n in names):
        return None

    # P-67.1.2 (BB:35886, "(chloromethyl)(methyl)nitramide (PIN)"): the amide N
    # of the `nitramide` functional parent is its ONLY substitutable position, so
    # the substituent locants are OMITTED (P-14.3.4.2) -- the PIN is
    # '(chloromethyl)(methyl)nitramide', NOT 'N-(chloromethyl)-N-methylnitramide'
    # (that spelling is the alternative *methanamine*-parent name, where 'N'
    # locants ARE needed). The prefixes are cited without the 'N-' locant, in
    # P-14.5.2 alphanumerical order, enclosed per the mononuclear single-
    # attachment rule P-16.5.1.3.1 (BB:7272): the first cited substituent is bare
    # (unless compound/complex, P-16.5.1.1), the second and further are each
    # parenthesised even when simple -> 'methyl(nitro)nitramide'. Multiplied
    # identical simple substituents keep the multiplier outside the marks.
    from collections import Counter
    from ..assembly.naming_utils import (
        apply_enclosing_marks, enclose_if_compound, multiplied_component,
        prefix_citation_sort_key,
    )
    counts = Counter(names)
    parts = []
    for i, nm in enumerate(sorted(counts, key=prefix_citation_sort_key)):
        marked = enclose_if_compound(nm)          # P-16.5.1.1 compound/complex
        if i > 0 and marked == nm:                # P-16.5.1.3.1 second+ simple
            marked = apply_enclosing_marks(nm, -1)
        parts.append(multiplied_component(counts[nm], nm, marked))
    prefix = ''.join(parts)
    if not prefix:
        return None
    return f"{prefix}nitramide"


def _name_dinitro_diamine(mol, units: List[Dict[str, int]], style: str) -> Optional[str]:
    """Shape 2: ``N,N'-dinitro<diamine>`` for exactly two nitro-amide units
    sharing one plain methylene bridge (the only verified real-witness
    shape). Fails closed on anything wider."""
    amide_ns = [u['amide_n'] for u in units]
    nitro_ns = [u['nitro_n'] for u in units]

    bridges = set()
    for amide_n in amide_ns:
        atom = mol.GetAtomWithIdx(amide_n)
        others = [nb.GetIdx() for nb in atom.GetNeighbors()
                  if nb.GetIdx() not in nitro_ns]
        if len(others) != 1:
            return None  # an amide N with any OTHER substituent is out of scope
        bridges.add(others[0])
    if len(bridges) != 1:
        return None  # the two amide N's must share the SAME bridging atom

    bridge_idx = next(iter(bridges))
    bridge_atom = mol.GetAtomWithIdx(bridge_idx)
    if bridge_atom.GetAtomicNum() != 6:
        return None
    heavy_neighbors = {nb.GetIdx() for nb in bridge_atom.GetNeighbors()}
    if heavy_neighbors != set(amide_ns):
        return None  # the bridge carbon must bear ONLY the two amide N's
    if bridge_atom.GetIsAromatic() or bridge_atom.IsInRing():
        return None

    # This structural shape (a carbon whose ONLY heavy-atom connections are
    # the two qualifying amide nitrogens) has exactly one possible parent
    # hydride: a single carbon carrying two implicit H's plus the two amine
    # groups, i.e. methanediamine. This is NOT a per-molecule string special
    # case -- it is the unique, graph-forced consequence of the structural
    # guards just above (degree-2 heavy connectivity, both neighbours being
    # the two amide N's, carbon, non-aromatic, ring-free) -- and it sidesteps
    # a SEPARATE, pre-existing, out-of-scope gap: `name_compound('NCN')`
    # (bare methanediamine, no nitro at all) itself currently mis-names to
    # 'methanamine' and is suppressed by SELF-01 to the sentinel, because
    # this tree's general amine producer does not yet build a gem-diamine
    # (two -NH2 on the SAME carbon) parent. Fixing that general gap is out
    # of scope for a nitro/nitramide task; the graph proof above lets this
    # narrow branch stay correct without depending on it.
    return "N,N'-dinitromethanediamine"


def name_substituted_nitramide(mol, style: str = 'pin') -> Optional[str]:
    """Return the P-66.1.1.3.1.1 substitutive nitramide/N-nitro PIN, else None.

    ``N-(hydroxymethyl)nitramide``, ``N,N-bis(hydroxymethyl)nitramide``,
    ``N,N'-dinitromethanediamine``. Fails closed on every shape outside the
    two documented in this module's docstring.
    """
    if mol is None:
        return None
    units = _nitro_amide_units(mol)
    if len(units) == 1:
        return _name_single_nitramide(mol, units[0]['amide_n'], units[0]['nitro_n'], style)
    if len(units) == 2:
        return _name_dinitro_diamine(mol, units, style)
    return None
