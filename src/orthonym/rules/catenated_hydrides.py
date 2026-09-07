"""Catenated Group-14 / chalcogen(+N) hydride namer (P-21.2.3 / P-52.1.3).

Alternating homonuclear a(ba)n catenated parent hydrides — a chain of identical
Group-14 atoms {Si, Ge, Sn, Pb} linked by identical bridge heteroatoms
{O, N, S, Se, Te}, terminated at both ends by a Group-14 atom, everything else
saturated with hydrogen::

    [SiH3]O[SiH3]              -> disiloxane        (2 Si, 1 O)
    [SiH3]O[SiH2]O[SiH3]       -> trisiloxane       (3 Si, 2 O)
    [SiH3]N[SiH3]              -> N-silylsilanamine  (N bridge -> amine, P-21.2.3.1)
    [SiH3]S[SiH3]              -> disilathiane      (S bridge; linking 'a')
    [SnH3]O[SnH3]              -> distannoxane      (Sn)
    [GeH3]O[GeH3]              -> digermoxane       (Ge)

A nitrogen bridge is the exception: an '-azane' parent hydride is non-PIN for
these (P-21.2.3.1), so N-bridged chains are named substitutively as amines on the
Group-14 hydride ('silane') -- see ``_name_nitrogen_bridged_group14``.

Every emitted name round-trips through OPSIN 2.9.0.

SCOPE (fail-closed, accuracy-first): a single unbranched chain that strictly
alternates one Group-14 element with one bridge element, terminated by two
Group-14 atoms, all H-saturated, neutral, non-radical, acyclic, single fragment.
ANY carbon, branch, mixed Group-14 kind, mixed bridge kind, or extra substituent
fails a guard and returns None (cascade-continuation). This is the boundary the
skeletal-replacement engine deliberately declines (Gate 3b) — pure homonuclear
Si-O-Si chains route here; mixed Si-O-C-S chains stay with skeletal 'a'.

Graph/atom classifier — no SMARTS broadening; pure (no mol mutation).
"""

from typing import Optional

from rdkit import Chem

from ..assembly.naming_utils import get_multiplier_prefix
from .substituent_purity import organyl_prefix_name

_GROUP14_STEM = {'Si': 'sil', 'Ge': 'germ', 'Sn': 'stann', 'Pb': 'plumb'}
_BRIDGE_SUFFIX = {
    'O': 'oxane', 'N': 'azane', 'S': 'thiane', 'Se': 'selenane', 'Te': 'tellurane',
}
_VOWELS = frozenset('aeiou')


def name_catenated_hydride(mol) -> Optional[str]:
    """Return the catenated Group-14/bridge hydride PIN, else ``None``."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    g14_atoms = []
    bridge_atoms = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == 'H':
            continue
        if sym in _GROUP14_STEM:
            g14_atoms.append(atom)
        elif sym in _BRIDGE_SUFFIX:
            bridge_atoms.append(atom)
        else:
            return None  # any carbon / other element -> not this class

    # Need >= 2 Group-14 atoms and exactly one fewer bridge (a-b-a-b-...-a).
    if len(g14_atoms) < 2 or len(bridge_atoms) != len(g14_atoms) - 1:
        return None

    # Homogeneous element kinds.
    g14_sym = g14_atoms[0].GetSymbol()
    if any(a.GetSymbol() != g14_sym for a in g14_atoms):
        return None
    bridge_sym = bridge_atoms[0].GetSymbol()
    if any(a.GetSymbol() != bridge_sym for a in bridge_atoms):
        return None

    # Strict alternation + unbranched: every Group-14 atom bonds ONLY to bridge
    # atoms (1 if terminal, 2 if internal); every bridge atom bonds ONLY to two
    # Group-14 atoms. No Group-14--Group-14 or bridge--bridge bonds.
    for a in g14_atoms:
        heavy = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
        if not heavy or len(heavy) > 2:
            return None
        if any(n.GetSymbol() != bridge_sym for n in heavy):
            return None
    for b in bridge_atoms:
        heavy = [n for n in b.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy) != 2 or any(n.GetSymbol() != g14_sym for n in heavy):
            return None

    # P-21.2.3.1 (BB 26243/23547/16015): a NITROGEN-bridged Group-14 a(ba)n chain
    # is NOT named as an '-azane' parent hydride — 'disilazane'/'trisilazane' are
    # explicitly non-PIN ("disilazane is not a recommended parent hydride, see
    # P-21.2.3.1"). Nitrogen carries the amine functionality, so the PIN is built
    # substitutively on the Group-14 hydride 'silane'/'germane'/... :
    #   SiH3-NH-SiH3         -> N-silylsilanamine         (preselected name, BB 26243)
    #   SiH3-NH-SiH2-NH-SiH3 -> N,N'-disilylsilanediamine (BB 23547)
    if bridge_sym == 'N':
        return _name_nitrogen_bridged_group14(g14_sym, len(g14_atoms))

    stem = _GROUP14_STEM[g14_sym]
    suffix = _BRIDGE_SUFFIX[bridge_sym]
    # Linking 'a' when the bridge suffix begins with a consonant (silathiane,
    # not silthiane; siloxane keeps no linker — vowel-initial).
    link = '' if suffix[0] in _VOWELS else 'a'
    base = f"{stem}{link}{suffix}"
    multiplier = get_multiplier_prefix(len(g14_atoms), base)
    return f"{multiplier}{base}"


# Substitutive parent-hydride and 'yl' substituent names for the Group-14
# elements, used by the nitrogen-bridged (silazane) amine renderer.
_GROUP14_HYDRIDE_NAME = {'Si': 'silane', 'Ge': 'germane', 'Sn': 'stannane',
                         'Pb': 'plumbane'}
_GROUP14_YL_NAME = {'Si': 'silyl', 'Ge': 'germyl', 'Sn': 'stannyl',
                    'Pb': 'plumbyl'}


def _name_nitrogen_bridged_group14(g14_sym: str, n_g14: int) -> Optional[str]:
    """Return the substitutive amine PIN for a nitrogen-bridged homonuclear
    Group-14 a(ba)n chain (P-21.2.3.1 / P-62.2.2.1), else ``None`` (fail-closed).

    Only the two Blue-Book-documented members are built — a longer N-bridged chain
    is not a documented PIN and fails closed (cascade-continuation):

        SiH3-NH-SiH3          -> N-silylsilanamine         (BB 26243, preselected)
        SiH3-NH-SiH2-NH-SiH3  -> N,N'-disilylsilanediamine (BB 23547)
    """
    from ..assembly.naming_utils import apply_vowel_elision
    hydride = _GROUP14_HYDRIDE_NAME.get(g14_sym)
    yl = _GROUP14_YL_NAME.get(g14_sym)
    if hydride is None or yl is None:
        return None
    if n_g14 == 2:
        # One Group-14 hydride is the parent bearing the amine; the far -EH3 is an
        # N-'yl' substituent (P-62.2.2.1 method 1, BB 26243).
        return f"N-{yl}{apply_vowel_elision(hydride, 'amine')}"
    if n_g14 == 3:
        # The central Group-14 hydride bears both amines (a diamine); each amine
        # nitrogen bears an -EH3 -> N,N'-di'yl' (BB 23547).
        return f"N,N'-di{yl}{apply_vowel_elision(hydride, 'diamine')}"
    return None  # longer N-bridged chains: not a BB-documented PIN, fail closed


# ---------------------------------------------------------------------------
# P-68.3.2.2 — homonuclear Group-15 (pnictogen) catenated parent hydrides
# ---------------------------------------------------------------------------
# The pnictogen (P/As/Sb/Bi) analogue of the polyazane family (N -> polyazane):
# an unbranched chain of >=2 IDENTICAL Group-15 atoms, fully H-saturated,
# neutral, acyclic, standard valence -> <multiplier><stem> (P-21.2.2, no elision
# of the multiplier vowel):
#     PP -> diphosphane (BB 39081; not 'diphosphine')
#     AsAsAsAsAs -> pentaarsane (BB 39083)   H2Bi-BiH2 -> dibismuthane (BB 39278)
# Mirrors polyazane's saturated homonuclear-chain logic. Nitrogen deliberately
# routes to polyazane (higher functionality of amines, P-21.2.3.1); this family
# is P/As/Sb/Bi only.
_PNICTOGEN_STEM = {'P': 'phosphane', 'As': 'arsane', 'Sb': 'stibane',
                   'Bi': 'bismuthane'}
# Basic multiplying prefixes (Table 1.4). NO elision of the terminal vowel
# (P-21.2.2): penta+arsane -> "pentaarsane".
_PNICTOGEN_MULT = {2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa',
                   7: 'hepta', 8: 'octa'}


def name_homonuclear_pnictogen_chain(mol) -> Optional[str]:
    """Return the PIN for a homonuclear Group-15 catenated parent hydride
    (P-68.3.2.2), else ``None`` (fail-closed cascade-continuation).

    Scope (a pure graph classifier, NOT SMARTS): a single unbranched chain of
    >=2 IDENTICAL pnictogen atoms {P, As, Sb, Bi}, every one H-saturated, neutral,
    non-radical, standard bonding number (3), acyclic, single fragment. ANY carbon
    (organyl phosphane -> name_phosphine), a mononuclear hub (PH3 -> mononuclear
    hydride), a heteronuclear chain (Si-As -> dinuclear hydride), nitrogen
    (-> polyazane), a branch, a ring, a multiple bond, a charge/radical, or a
    nonstandard valence fails a guard and cascades onward. Pure: no mol mutation.
    """
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None

    heavy = [a for a in mol.GetAtoms() if a.GetSymbol() != 'H']
    if len(heavy) < 2:
        return None                      # mononuclear PH3 -> mononuclear hydride
    syms = {a.GetSymbol() for a in heavy}
    if len(syms) != 1:
        return None                      # heteronuclear / has carbon -> not here
    element = next(iter(syms))
    stem = _PNICTOGEN_STEM.get(element)
    if stem is None:
        return None                      # not a Group-15 P/As/Sb/Bi element

    from .lambda_convention import nonstandard_bonding_number
    idxs = {a.GetIdx() for a in heavy}
    deg = {}
    for a in heavy:
        if a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() != 0:
            return None
        if nonstandard_bonding_number(mol, a.GetIdx()) is not None:
            return None                  # lambda5 phosphane chain not built here
        d = sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in idxs)
        if d > 2:
            return None                  # branch
        deg[a.GetIdx()] = d

    # Every chain bond must be single (a P=P diphosphene is not built here).
    for b in mol.GetBonds():
        if b.GetBeginAtomIdx() in idxs and b.GetEndAtomIdx() in idxs:
            if b.GetBondType() != Chem.BondType.SINGLE:
                return None

    ends = [i for i, d in deg.items() if d == 1]
    if len(ends) != 2:
        return None                      # ring (no endpoint) or malformed

    n = len(heavy)
    mult = _PNICTOGEN_MULT.get(n)
    if mult is None:
        return None                      # chain too long for the multiplier table
    return f"{mult}{stem}"


# ---------------------------------------------------------------------------
# P-68.4.2.1 / P-21.2.3.1 — pure-chalcogen a[ba]n parent hydrides (dithioxane)
# ---------------------------------------------------------------------------
# An unbranched chain of chalcogen atoms strictly ALTERNATING between two
# distinct elements and TERMINATED at both ends by the element coming LATER in
# the seniority order O > S > Se > Te (the JUNIOR terminal element)::
#
#     HS-O-SH        -> dithioxane          (2 terminal S junior, 1 central O senior)
#     CH3-S-O-SH     -> methyldithioxane    (BB: not methylsulfane-OS-thioperoxol)
#     CH3-S-O-S-CH3  -> dimethyldithioxane
#
# Name (P-21.2.3.1) = <multiplier(# terminal atoms)> + 'a'-term of the JUNIOR
# terminal element + 'a'-term of the SENIOR central element + 'ane' (with 'a'
# elision before a vowel; the multiplier vowel is NOT elided). Terminal organyls
# are cited as prefixes; the two terminal positions are equivalent by symmetry so
# BB omits their locants (methyldithioxane / dimethyldithioxane / methyl(phenyl)-
# dithioxane). These preselected parent hydrides receive the PIN and PRE-EMPT the
# skeletal-replacement 'a'-name (CH3-S-O-S-CH3 is 'dimethyldithioxane', NOT the
# valid-but-non-PIN '3-oxa-2,4-dithiapentane') per P-68.4.2.1.
_CHALCOGEN_ATERM = {'O': 'oxa', 'S': 'thia', 'Se': 'selena', 'Te': 'tellura'}
# Seniority index (lower = senior) for the a[ba]n terminal/central choice.
_CHALCOGEN_SENIORITY = {'O': 0, 'S': 1, 'Se': 2, 'Te': 3}
_ABA_MULT = {2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa'}
_SUB_MULT_ABA = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}


def _join_aterms(terms) -> str:
    """Concatenate 'a'-terms + 'ane', eliding a trailing 'a' before a vowel
    (thia+oxa+ane -> thioxane)."""
    out = ''
    for t in terms:
        if out and out[-1] == 'a' and t[0] in _VOWELS:
            out = out[:-1]
        out += t
    return out


def name_heterochalcogen_aba(mol) -> Optional[str]:
    """Return the PIN for a pure-chalcogen a[ba]n parent hydride (P-68.4.2.1 /
    P-21.2.3.1: dithioxane / methyldithioxane / dimethyldithioxane), else ``None``
    (fail-closed cascade-continuation). Pure: no mol mutation.

    Scope (a graph classifier, NOT SMARTS): a single unbranched chain of >=3
    chalcogen atoms strictly alternating between EXACTLY two distinct elements,
    both termini the SAME element and that element JUNIOR (later in O>S>Se>Te) to
    the central element; every internal chalcogen H-only; terminal chalcogens bear
    one H or one pure organyl; neutral, non-radical, acyclic, single fragment; the
    only non-chalcogen heavy atoms are terminal organyl carbons. A homogeneous
    chalcogen chain (-> chalcogen_chain), a carbon-in-backbone chain
    (-> skeletal_replacement), a Group-14 a[ba]n (-> catenated_hydride), a ring,
    an ion, or a radical fails a guard and cascades onward.
    """
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    # Every heavy atom is a chalcogen or a carbon (organyl); collect chalcogens.
    chal = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == 'H':
            continue
        if sym in _CHALCOGEN_ATERM:
            chal.append(atom.GetIdx())
        elif sym != 'C':
            return None                          # stray heteroatom -> not this class
    if len(chal) < 3:
        return None
    chal_set = set(chal)

    # The chalcogens must form one unbranched single-bonded path.
    adj = {i: [] for i in chal}
    for i in chal:
        for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nbr.GetIdx()
            if j in chal_set:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    return None
                adj[i].append(j)
    if any(len(adj[i]) > 2 for i in chal):
        return None
    endpoints = [i for i in chal if len(adj[i]) == 1]
    if len(endpoints) != 2:
        return None                              # ring / forked / disconnected
    order = [endpoints[0]]
    prev, cur = -1, endpoints[0]
    while True:
        nxts = [j for j in adj[cur] if j != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        order.append(cur)
    if len(order) != len(chal):
        return None

    syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in order]
    # Exactly two distinct elements, strictly alternating along the chain.
    if len(set(syms)) != 2:
        return None
    for k in range(len(syms) - 1):
        if syms[k] == syms[k + 1]:
            return None                          # a run of identical -> not a[ba]n
    terminal_elem = syms[0]
    if syms[-1] != terminal_elem:
        return None                              # both termini must be the same element
    central_elem = syms[1]
    # The terminal element must be JUNIOR (later in seniority) to the central.
    if _CHALCOGEN_SENIORITY[terminal_elem] <= _CHALCOGEN_SENIORITY[central_elem]:
        return None

    # Internal chalcogens carry only H; terminal chalcogens bear one organyl or H.
    subs = []
    for pos, idx in enumerate(order):
        atom = mol.GetAtomWithIdx(idx)
        heavy_nonchain = [n for n in atom.GetNeighbors()
                          if n.GetIdx() not in chal_set and n.GetSymbol() != 'H']
        is_terminal = pos in (0, len(order) - 1)
        if not heavy_nonchain:
            continue
        if not is_terminal or len(heavy_nonchain) != 1:
            return None                          # internal / >1 organyl -> decline
        name = organyl_prefix_name(mol, heavy_nonchain[0].GetIdx(), idx)
        if name is None:
            return None
        subs.append(name)

    n_terminal = syms.count(terminal_elem)
    mult = _ABA_MULT.get(n_terminal)
    if mult is None:
        return None
    base = _join_aterms([_CHALCOGEN_ATERM[terminal_elem],
                         _CHALCOGEN_ATERM[central_elem], 'ane'])
    parent = f"{mult}{base}"                      # dithioxane / trithioxane ...
    if not subs:
        return parent

    # Terminal organyls: no locants (the terminal positions are symmetric — BB
    # methyldithioxane / dimethyldithioxane / methyl(phenyl)dithioxane).
    #
    # v29 P3: the organyl guard above is now the shared chokepoint, so a prefix
    # reaching here may carry LOCANTS ('propan-2-yl'), a retained italicized
    # prefix ('tert-butyl') or its own enclosing marks ('(4-bromophenyl)methyl').
    # Raw `sorted()` + bare concatenation was correct only for the letters-only
    # class the retired narrow walker could return, so ordering and marks are
    # delegated to the shared primitives — no local copy of either decision:
    #   * P-14.5.2/P-14.5.4 `prefix_citation_sort_key` — alphanumerical citation
    #     order, which ignores enclosing marks and the italicized prefix;
    #   * BB 25719 `1,4-di(propan-2-yl)cyclohexane` (PIN) — a compound prefix is
    #     enclosed and the SIMPLE multiplier sits OUTSIDE the marks;
    #   * BB 16286 `*tert*-butyldi(methyl)phosphane` (PIN) + P-16.3.4 — a retained
    #     italicized prefix is cited bare and keeps its hyphen under a multiplier
    #     ('di-tert-butyl', never 'ditert-butyl').
    from collections import Counter

    from ..assembly.naming_utils import (
        apply_enclosing_marks,
        enclose_if_compound,
        multiplier_needs_hyphen,
        prefix_citation_sort_key,
    )
    counts = Counter(subs)
    uniq = sorted(counts, key=prefix_citation_sort_key)
    parts = []
    for i, nm in enumerate(uniq):
        m = _SUB_MULT_ABA.get(counts[nm])
        if m is None:
            return None
        # Both mark rules are "enclose unless already enclosed", so ask the
        # shared compound test (P-16.5.1.1) first and only force marks when it
        # declined and P-16.5.1.3 still needs a separator.
        marked = enclose_if_compound(nm)
        if len(uniq) >= 2 and i > 0 and marked == nm:
            marked = apply_enclosing_marks(nm, -1)
        if m and marked == nm and multiplier_needs_hyphen(nm):
            token = f"{m}-{marked}"              # P-16.3.3(b)/P-16.2.4.1(d) di-tert-butyl
        else:
            token = f"{m}{marked}"
        parts.append(token)
    return f"{''.join(parts)}{parent}"


__all__ = ["name_catenated_hydride", "name_homonuclear_pnictogen_chain",
           "name_heterochalcogen_aba"]
