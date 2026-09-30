"""Catenated Group-14/Group-15 / chalcogen(+N) hydride namer /.

Alternating homonuclear a(ba)n catenated parent hydrides — a chain of identical
Group-14 {Si, Ge, Sn, Pb} OR Group-15 {P, As, Sb, Bi} hub atoms linked by
identical bridge heteroatoms {O, N, S, Se, Te}, terminated at both ends by a hub
atom, everything else saturated with hydrogen::

    [SiH3]O[SiH3] -> disiloxane (2 Si, 1 O)
    [SiH3]O[SiH2]O[SiH3] -> trisiloxane (3 Si, 2 O)
    [SiH3]N[SiH3] -> N-silylsilanamine (N bridge -> amine,
    [SiH3]S[SiH3] -> disilathiane (S bridge; linking 'a')
    [SnH3]O[SnH3] -> distannoxane (Sn)
    [GeH3]O[GeH3] -> digermoxane (Ge)
    P[Se]P -> diphosphaselenane (Group-15 hub, BB 8020 PIN)

A nitrogen bridge is the exception: an '-azane' parent hydride is non-PIN for
these, so N-bridged chains are named substitutively as amines on the
hub hydride ('silane') -- see ``_name_nitrogen_bridged_group14``. That amine
renderer is documented only for the Group-14 family, so a Group-15 hub with an
N bridge falls through it and returns ``None`` (fail-closed, cascade-continuation).

Every emitted name round-trips through OPSIN 2.9.0.

SCOPE (fail-closed, accuracy-first): a single unbranched chain that strictly
alternates one hub element (Group-14 or Group-15) with one bridge element,
terminated by two hub atoms, all H-saturated, standard bonding number, neutral,
non-radical, acyclic, single fragment. ANY carbon, branch, mixed hub kind, mixed
bridge kind, nonstandard valence (lambda), or extra substituent fails a guard and
returns None (cascade-continuation). This is the boundary the skeletal-replacement
engine deliberately declines (Gate 3b) — pure homonuclear Si-O-Si chains route
here; mixed Si-O-C-S chains stay with skeletal 'a'.

Graph/atom classifier — no SMARTS broadening; pure (no mol mutation).
"""

from typing import Optional

from rdkit import Chem

from ..assembly.naming_utils import get_multiplier_prefix
from .substituent_purity import organyl_prefix_name

_GROUP14_STEM = {'Si': 'sil', 'Ge': 'germ', 'Sn': 'stann', 'Pb': 'plumb'}
# Group-15 (pnictogen) hub stems, bare (trailing 'a' of the full 'a' term
# phospha/arsa/stiba/bisma stripped) so the SAME stem+link+suffix elision
# scheme below applies unmodified BB 8020: PH2-Se-PH2 ->
# diphosphaselenane; 'phospha' + 'selenane' with no elision needed since
# 'selenane' begins with a consonant -> stem 'phosph' + link 'a' + 'selenane').
_GROUP15_STEM = {'P': 'phosph', 'As': 'ars', 'Sb': 'stib', 'Bi': 'bism'}
_HUB_STEM = {**_GROUP14_STEM, **_GROUP15_STEM}
_BRIDGE_SUFFIX = {
    'O': 'oxane', 'N': 'azane', 'S': 'thiane', 'Se': 'selenane', 'Te': 'tellurane',
}
_VOWELS = frozenset('aeiou')


def name_catenated_hydride(mol) -> Optional[str]:
    """Return the catenated Group-14/Group-15 hub + bridge hydride PIN, else
    ``None``."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    hub_atoms = []
    bridge_atoms = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == 'H':
            continue
        if sym in _HUB_STEM:
            hub_atoms.append(atom)
        elif sym in _BRIDGE_SUFFIX:
            bridge_atoms.append(atom)
        else:
            return None  # any carbon / other element -> not this class

    # Need >= 2 hub atoms and exactly one fewer bridge (a-b-a-b-...-a).
    if len(hub_atoms) < 2 or len(bridge_atoms) != len(hub_atoms) - 1:
        return None

    # Homogeneous element kinds.
    hub_sym = hub_atoms[0].GetSymbol()
    if any(a.GetSymbol() != hub_sym for a in hub_atoms):
        return None
    bridge_sym = bridge_atoms[0].GetSymbol()
    if any(a.GetSymbol() != bridge_sym for a in bridge_atoms):
        return None

    # Strict alternation + unbranched: every hub atom bonds ONLY to bridge
    # atoms (1 if terminal, 2 if internal); every bridge atom bonds ONLY to two
    # hub atoms. No hub--hub or bridge--bridge bonds.
    for a in hub_atoms:
        heavy = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
        if not heavy or len(heavy) > 2:
            return None
        if any(n.GetSymbol() != bridge_sym for n in heavy):
            return None
    for b in bridge_atoms:
        heavy = [n for n in b.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy) != 2 or any(n.GetSymbol() != hub_sym for n in heavy):
            return None

    # Standard bonding number only — a lambda-convention hub (e.g. a
    # hypervalent P) is not a plain a(ba)n hydride; fail closed rather than
    # mis-stem it. (Only the Group-15 hubs are checked: this dict is empty of
    # Group-14 symbols, so the long-standing Group-14 anchor is untouched.)
    if hub_sym in _GROUP15_STEM:
        from .lambda_convention import nonstandard_bonding_number
        for a in hub_atoms:
            if nonstandard_bonding_number(mol, a.GetIdx()) is not None:
                return None

    # (BB 26243/23547/16015): a NITROGEN-bridged hub a(ba)n chain is
    # NOT named as an '-azane' parent hydride — 'disilazane'/'trisilazane' are
    # explicitly non-PIN ("disilazane is not a recommended parent hydride, see
    # "). Nitrogen carries the amine functionality, so the PIN is built
    # substitutively on the hub hydride 'silane'/'germane'/...:
    # SiH3-NH-SiH3 -> N-silylsilanamine (preselected name, BB 26243)
    # SiH3-NH-SiH2-NH-SiH3 -> N,N'-disilylsilanediamine (BB 23547)
    # (Documented only for the Group-14 family; a Group-15 hub falls through
    # this renderer's dict lookups and returns None, cascade-continuation.)
    if bridge_sym == 'N':
        return _name_nitrogen_bridged_group14(hub_sym, len(hub_atoms))

    stem = _HUB_STEM[hub_sym]
    suffix = _BRIDGE_SUFFIX[bridge_sym]
    # Linking 'a' when the bridge suffix begins with a consonant (silathiane,
    # not silthiane; siloxane keeps no linker — vowel-initial).
    link = '' if suffix[0] in _VOWELS else 'a'
    base = f"{stem}{link}{suffix}"
    # The prefix counts the hub atoms of the chain, 'disiloxane',
    # 'diphosphaselenane'): a basic numerical term, never the 'bis' of a
    # multiplied component (c) concerns multiplied components).
    from ..assembly.naming_utils import simple_multiplier_word
    multiplier = simple_multiplier_word(len(hub_atoms)) or get_multiplier_prefix(
        len(hub_atoms), base)
    return f"{multiplier}{base}"


# Substitutive parent-hydride and 'yl' substituent names for the Group-14
# elements, used by the nitrogen-bridged (silazane) amine renderer.
_GROUP14_HYDRIDE_NAME = {'Si': 'silane', 'Ge': 'germane', 'Sn': 'stannane',
                         'Pb': 'plumbane'}
_GROUP14_YL_NAME = {'Si': 'silyl', 'Ge': 'germyl', 'Sn': 'stannyl',
                    'Pb': 'plumbyl'}


def _name_nitrogen_bridged_group14(g14_sym: str, n_g14: int) -> Optional[str]:
    """Return the substitutive amine PIN for a nitrogen-bridged homonuclear
    Group-14 a(ba)n chain /, else ``None`` (fail-closed).

    Only the two Blue-Book-documented members are built — a longer N-bridged chain
    is not a documented PIN and fails closed (cascade-continuation):

        SiH3-NH-SiH3 -> N-silylsilanamine (BB 26243, preselected)
        SiH3-NH-SiH2-NH-SiH3 -> N,N'-disilylsilanediamine (BB 23547)
    """
    from ..assembly.naming_utils import apply_vowel_elision
    hydride = _GROUP14_HYDRIDE_NAME.get(g14_sym)
    yl = _GROUP14_YL_NAME.get(g14_sym)
    if hydride is None or yl is None:
        return None
    if n_g14 == 2:
        # One Group-14 hydride is the parent bearing the amine; the far -EH3 is an
        # N-'yl' substituent method 1, BB 26243).
        return f"N-{yl}{apply_vowel_elision(hydride, 'amine')}"
    if n_g14 == 3:
        # The central Group-14 hydride bears both amines (a diamine); each amine
        # nitrogen bears an -EH3 -> N,N'-di'yl' (BB 23547).
        return f"N,N'-di{yl}{apply_vowel_elision(hydride, 'diamine')}"
    return None  # longer N-bridged chains: not a BB-documented PIN, fail closed


# ---------------------------------------------------------------------------
# — homonuclear Group-15 (pnictogen) catenated parent hydrides
# ---------------------------------------------------------------------------
# The pnictogen (P/As/Sb/Bi) analogue of the polyazane family (N -> polyazane):
# an unbranched chain of >=2 IDENTICAL Group-15 atoms, fully H-saturated,
# neutral, acyclic, standard valence -> <multiplier><stem>, no elision
# of the multiplier vowel):
# PP -> diphosphane (BB 39081; not 'diphosphine')
# AsAsAsAsAs -> pentaarsane (BB 39083) H2Bi-BiH2 -> dibismuthane (BB 39278)
# Mirrors polyazane's saturated homonuclear-chain logic. Nitrogen deliberately
# routes to polyazane (higher functionality of amines,; this family
# is P/As/Sb/Bi only.
_PNICTOGEN_STEM = {'P': 'phosphane', 'As': 'arsane', 'Sb': 'stibane',
                   'Bi': 'bismuthane'}
# Basic multiplying prefixes (Table 1.4). NO elision of the terminal vowel
#: penta+arsane -> "pentaarsane".
_PNICTOGEN_MULT = {2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa',
                   7: 'hepta', 8: 'octa'}


def name_homonuclear_pnictogen_chain(mol) -> Optional[str]:
    """Return the PIN for a homonuclear Group-15 catenated parent hydride
    , else ``None`` (fail-closed cascade-continuation).

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
# / — pure-chalcogen a[ba]n parent hydrides (dithioxane)
# ---------------------------------------------------------------------------
# An unbranched chain of chalcogen atoms strictly ALTERNATING between two
# distinct elements and TERMINATED at both ends by the element coming LATER in
# the seniority order O > S > Se > Te (the JUNIOR terminal element)::
#
# HS-O-SH -> dithioxane (2 terminal S junior, 1 central O senior)
# CH3-S-O-SH -> methyldithioxane (BB: not methylsulfane-OS-thioperoxol)
# CH3-S-O-S-CH3 -> dimethyldithioxane
#
# Name = <multiplier(# terminal atoms)> + 'a'-term of the JUNIOR
# terminal element + 'a'-term of the SENIOR central element + 'ane' (with 'a'
# elision before a vowel; the multiplier vowel is NOT elided). Terminal organyls
# are cited as prefixes; the two terminal positions are equivalent by symmetry so
# BB omits their locants (methyldithioxane / dimethyldithioxane / methyl(phenyl)-
# dithioxane). These preselected parent hydrides receive the PIN and PRE-EMPT the
# skeletal-replacement 'a'-name (CH3-S-O-S-CH3 is 'dimethyldithioxane', NOT the
# valid-but-non-PIN '3-oxa-2,4-dithiapentane') per.
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


def _try_aba_parent(mol, order, chal_set) -> Optional[str]:
    """Attempt the direct a[ba]n PARENT match / over an
    already-ordered single unbranched chalcogen chain: exactly two distinct
    elements strictly alternating, both termini the same JUNIOR element,
    terminal organyls only (internal atoms H-only). Returns the parent name
    (dithioxane / methyldithioxane / dimethyldithioxane...), or ``None`` if
    ``order`` does not itself describe a bare or organyl-substituted a[ba]n
    parent. ``chal_set`` is every chalcogen atom index in the WHOLE molecule
    (not just ``order``) so a chain-internal chalcogen bonded to a
    caller-truncated neighbour (see the ``-ol`` peel in
    :func:`name_heterochalcogen_aba`) is never mistaken for an organyl
    substituent. Pure: no mol mutation.
    """
    if len(order) < 3:
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
    parent = f"{mult}{base}"                      # dithioxane / trithioxane...
    if not subs:
        return parent

    # Terminal organyls: no locants (the terminal positions are symmetric — BB
    # methyldithioxane / dimethyldithioxane / methyl(phenyl)dithioxane).
    #
    #: the organyl guard above is now the shared chokepoint, so a prefix
    # reaching here may carry LOCANTS ('propan-2-yl'), a retained italicized
    # prefix ('tert-butyl') or its own enclosing marks ('(4-bromophenyl)methyl').
    # Raw `sorted` + bare concatenation was correct only for the letters-only
    # class the retired narrow walker could return, so ordering and marks are
    # delegated to the shared primitives — no local copy of either decision:
    # * / `prefix_citation_sort_key` — alphanumerical citation
    # order, which ignores enclosing marks and the italicized prefix;
    # * BB 25719 `1,4-di(propan-2-yl)cyclohexane` (PIN) — a compound prefix is
    # enclosed and the SIMPLE multiplier sits OUTSIDE the marks;
    # * BB 16286 `*tert*-butyldi(methyl)phosphane` (PIN) + — a retained
    # italicized prefix is cited bare and keeps its hyphen under a multiplier
    # ('di-tert-butyl', never 'ditert-butyl').
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
        # shared compound test first and only force marks when it
        # declined and still needs a separator.
        marked = enclose_if_compound(nm)
        if len(uniq) >= 2 and i > 0 and marked == nm:
            marked = apply_enclosing_marks(nm, -1)
        from ..assembly.naming_utils import multiplied_component as _mc
        token = _mc(counts[nm], nm, marked)     # (b)/(d) di-tert-butyl
        parts.append(token)
    return f"{''.join(parts)}{parent}"


def name_heterochalcogen_aba(mol) -> Optional[str]:
    """Return the PIN for a pure-chalcogen a[ba]n parent hydride /
    : dithioxane / methyldithioxane / dimethyldithioxane), or that
    same parent bearing a terminal ``-ol`` suffix: HO-S-O-SH ->
    dithioxanol) when exactly one terminal chalcogen is a bare ``-OH``
    principal group, else ``None`` (fail-closed cascade-continuation). Pure: no
    mol mutation.

    Scope (a graph classifier, NOT SMARTS): a single unbranched chain of >=3
    chalcogen atoms strictly alternating between EXACTLY two distinct elements,
    both termini the SAME element and that element JUNIOR (later in O>S>Se>Te) to
    the central element; every internal chalcogen H-only; terminal chalcogens bear
    one H or one pure organyl; neutral, non-radical, acyclic, single fragment; the
    only non-chalcogen heavy atoms are terminal organyl carbons. A homogeneous
    chalcogen chain (-> chalcogen_chain), a carbon-in-backbone chain
    (-> skeletal_replacement), a Group-14 a[ba]n (-> catenated_hydride), a ring,
    an ion, or a radical fails a guard and cascades onward.

    The ``-ol`` suffix fires only for a BARE parent (no organyl
    substituent anywhere) with exactly one terminal ``-OH`` and no other
    decoration: the OH's own oxygen is itself a chalcogen so it walks straight
    into the contiguous-chalcogen chain above (HO-S-O-SH is the 4-atom chain
    O-S-O-S, termini O/S mismatched), so a single terminal atom is peeled and
    the remainder re-tested as a bare a[ba]n parent. A second ``-OH`` leaves an
    even-length remainder whose mismatched termini are rejected the same way; a
    combined organyl+``-OH`` decoration is deferred (fails closed), not built
    here — see -noncarbon Task 7.
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

    direct = _try_aba_parent(mol, order, chal_set)
    if direct is not None:
        return direct

    # -ol suffix: exactly ONE terminal chalcogen of a BARE a[ba]n
    # parent bears -OH as the principal group (HO-S-O-SH -> dithioxanol). The
    # OH's own oxygen is itself a chalcogen, so it already walked into `order`
    # above as an extra link (HO-S-O-SH is the 4-atom chain O-S-O-S, termini
    # O/S mismatched -> `_try_aba_parent` above declines it). Peel ONE end and
    # re-test the remainder as a bare a[ba]n parent. Fail closed -- single
    # peel only, `O` element only, a bare -OH (no organyl on that atom), and NO
    # organyl anywhere on the remaining parent -- on any other decoration: a
    # second -OH leaves an even-length remainder whose mismatched termini
    # `_try_aba_parent` rejects the same way, and a combined organyl+-OH
    # decoration is deferred (not attempted) rather than guessed at.
    for strip_left in (True, False):
        if strip_left:
            endpt, neigh, remaining = 0, 1, order[1:]
        else:
            endpt, neigh, remaining = len(order) - 1, len(order) - 2, order[:-1]
        if len(remaining) < 3:
            continue
        oh_atom = mol.GetAtomWithIdx(order[endpt])
        if oh_atom.GetSymbol() != 'O':
            continue                              # only -ol (O) is in scope here
        if oh_atom.GetSymbol() == mol.GetAtomWithIdx(order[neigh]).GetSymbol():
            continue                              # part of a run, not a bare suffix
        if oh_atom.GetTotalNumHs() != 1:
            continue                              # not a bare -OH
        if any(n.GetIdx() not in chal_set and n.GetSymbol() != 'H'
               for n in oh_atom.GetNeighbors()):
            continue                              # organyl on the OH oxygen -> decline
        if any(any(n.GetIdx() not in chal_set and n.GetSymbol() != 'H'
                   for n in mol.GetAtomWithIdx(remaining[p]).GetNeighbors())
               for p in (0, len(remaining) - 1)):
            continue                              # substituent + -ol combo -> deferred
        parent = _try_aba_parent(mol, remaining, chal_set)
        if parent is None:
            continue
        from ..assembly.naming_utils import apply_vowel_elision
        return apply_vowel_elision(parent, 'ol')   # dithioxane -> dithioxanol
    return None


__all__ = ["name_catenated_hydride", "name_homonuclear_pnictogen_chain",
           "name_heterochalcogen_aba"]
