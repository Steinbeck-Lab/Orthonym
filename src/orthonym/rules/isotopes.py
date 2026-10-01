"""Isotopically substituted compound names (IUPAC +.

Fail-closed decorator layer. Orthonym's core pipeline strips isotope labels
(RDKit perception ignores GetIsotope for skeleton naming), so a labeled mol
would otherwise emit the UNLABELED name — a wrong PIN. This module runs as an
EARLY branch in Orthonym.name: it strips the labels, names the skeleton in
systematic style (locanted parents), then re-derives the isotopic descriptor
by an INTERNAL ORACLE (OPSIN-parse a candidate name; compare rdkit canonical
SMILES — with isotopes — to the original). If no candidate round-trips, it
returns None and the label is never mis-placed.

BB (the Blue Book): the nuclide symbol(s) in parentheses,
preceded by any necessary locant(s), are inserted before the isotopically
substituted part; polysubstitution count is a right subscript to the symbol.
BB /.4.2/.4.3 (the Blue Book-22232): lowest locants to modified
positions; then to higher atomic number; then to higher mass number.
"""
from __future__ import annotations

import contextlib
import contextvars
import re
from typing import Dict, List, Optional, Tuple

from rdkit import Chem


def has_isotopes(mol: Optional[Chem.Mol]) -> bool:
    """True iff any atom of ``mol`` carries a non-zero isotope label."""
    if mol is None:
        return False
    return any(a.GetIsotope() != 0 for a in atoms_of(mol))


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
# — isotopic descriptor formatting
# ---------------------------------------------------------------------------


def nuclide_symbol(mass: int, element: str) -> str:
    """Nuclide symbol with the mass number as a leading integer (ASCII form).

    Orthonym emits ASCII (no <sup>); OPSIN accepts the leading-digit form
    (probe: (2-14C)ethan-1-ol -> C([14CH3])O)..
    """
    return f"{mass}{element}"


def _max_atoms_at_position(atom: Chem.Atom) -> int:
    """Atoms of this element that could occupy ``atom``'s position in the
    UNMODIFIED parent -- the "polysubstitution at a single position"
    quantity that decides whether the count subscript is shown.

    A labelled HEAVY atom (13C, 18O, 15N, 131I,...) IS a skeleton position and a
    covalent position holds exactly one such atom -> 1. A labelled H (D/T) shares
    its single heavy neighbour with every other H that neighbour can bear, so the
    governing quantity is that neighbour's total-H capacity in the parent
    (``GetTotalNumHs(includeNeighbors=True)`` -- counts the D/T themselves, which
    RDKit keeps as explicit isotope-labelled H, plus any implicit H, so it reads
    the true parent complement directly off the ORIGINAL mol without a
    stripped-index remap). ``atom`` is indexed into the ORIGINAL (label-bearing)
    mol -- the ``label_map`` keys index there.
    """
    if atom.GetSymbol() == "H":
        nbrs = atom.GetNeighbors()
        if not nbrs:
            return 1
        return nbrs[0].GetTotalNumHs(includeNeighbors=True)
    return 1


def format_isotope_descriptor(groups, force_show: bool = False,
                              bracket: bool = False) -> str:
    """Build the parenthesized isotopic descriptor from grouped labels.

    ``bracket=True`` (FIX-F, / renders the SPECIFICALLY-LABELLED
    form ``[2H1]`` / ``[13C]`` in square brackets instead of the substituted
    form ``(2H1)`` / ``(13C)``. It is OPT-IN and OFF on the default single-call
    path by design: a labelled compound is STRUCTURALLY IDENTICAL to its
    substituted counterpart (``[13CH4]`` is one graph), so nothing in a bare SMILES
    selects between the two conventions -- the whole of chapter is therefore a
    permanent RIGHT_MOL_NONPIN ceiling under the default parenthetical form, not a
    reclaimable gap. The renderer exists and is testable so a caller that KNOWS it
    wants the labelled convention (e.g. a future explicit-mode API) can request it;
    the swap of the enclosing marks around a nested complex prefix is a
    documented extension point, not built here. See internal notes

    ``force_show=True`` restores the pre-FIX-A always-emit-the-subscript form
    regardless of ``max_at_pos``. The placement search uses it as a FALLBACK: the
    BB-preferred omitted form ``(2H)`` / ``(13C)`` is tried first, but OPSIN 2.9.0
    does not parse every omitted spelling in every position (e.g. an O-bound
    single ``(2H)`` before an ``-oic acid`` suffix), so when the omitted form does
    not round-trip the forced form ``(2H1)`` is tried before abstaining -- the
    right molecule at a non-preferred spelling beats silence (never a wrong name;
    every candidate stays OPSIN-RT gated).

    ``groups`` = list of (locant, mass, element, count[, max_at_pos]):
      locant None -> no leading locant (single-position parent / front
                      descriptor, e.g. (2H3)methoxybenzene, (12C)methane).
      locant int -> the count locants are repeated then hyphen-joined to the
                      nuclide, e.g. (2,2,2-2H3), (2-14C).
      max_at_pos -> OPTIONAL 5th field, the polysubstitution quantity
                      (:func:`_max_atoms_at_position`). Absent (legacy 4-tuple) ->
                      the count subscript is kept unconditionally (the pre-FIX-A
                      behaviour), so only a caller that supplies max_at_pos gets
                      the omission.

    Multiple groups at (possibly) the same place are cited alphabetically by
    element then by mass number /; groups are comma-joined.

     (the Blue Book-43720): the count subscript is shown "when polysubstitution
    at a single position is possible". So it is emitted iff ``count > 1`` (more
    than one atom of the nuclide is cited together) OR ``max_at_pos > 1`` (the
    single position could carry a second atom of that element) -- e.g. a lone D on
    a CH3 keeps ``(2H1)`` (3 H can occupy) but a lone 13C keeps no subscript
    ``(13C)`` (a carbon position holds one carbon) and a lone D on a CH keeps none
    ``(2H)`` (1 H can occupy). Verbatim witnesses: ``trichloro(12C)methane``,
    ``(2-13C)ethan-1-ol``, ``(2S)-(2-2H)butan-2-ol``.
    """
    def _one(group):
        locant, mass, element, count = group[0], group[1], group[2], group[3]
        max_at_pos = group[4] if len(group) > 4 else None
        sym = nuclide_symbol(mass, element)
        show_sub = force_show or count > 1 or max_at_pos is None or max_at_pos > 1
        sub = str(count) if show_sub else ""
        if locant is None:
            return f"{sym}{sub}"
        if isinstance(locant, (tuple, list)):
            #: a nuclide labelled at several DISTINCT positions cites each
            # locant once, ascending -> (2,4-2H2). (A scalar locant instead repeats
            # once per substituted atom at ONE shared position -> (2,2,2-2H3).) The
            # tuple form is what the mixed-descriptor builder passes for a count>1
            # group whose atoms sit at distinct positions.
            loc_part = ",".join(str(l) for l in locant)
        else:
            # BB: the locant is repeated once per substituted atom at that position
            # for a grouped multi-count token (2,2,2-2H3); for count 1 a single
            # locant + hyphen (2-14C).
            loc_part = ",".join(str(locant) for _ in range(count))
        return f"{loc_part}-{sym}{sub}"

    def _loc_sort_val(loc):
        # element/mass are the primary keys; this tie-break only matters when two
        # groups share both. Map None -> -1 and a distinct-locant tuple -> its
        # first (lowest) locant so an int/tuple/None mix stays orderable.
        if loc is None:
            return -1
        if isinstance(loc, (tuple, list)):
            return loc[0] if loc else -1
        return loc

    #: alphabetical by element symbol, then by mass number, then locant.
    ordered = sorted(groups, key=lambda g: (g[2], g[1], _loc_sort_val(g[0])))
    inner = ",".join(_one(g) for g in ordered)
    open_mark, close_mark = ("[", "]") if bracket else ("(", ")")
    return f"{open_mark}{inner}{close_mark}"


# ---------------------------------------------------------------------------
# / — internal-oracle isotopic-descriptor placement
# ---------------------------------------------------------------------------
# The skeleton name fixes the parent numbering, but Orthonym does NOT expose a
# reliable atom->locant map for acyclic parents (retrieve_confidence
# ['atom_to_locant'] is None for CCO). Rather than re-perceive the chain
# independently (which risks diverging from the name's own numbering — a
# fail-closed violation), the decorator treats the skeleton name's locant space
# as opaque and lets OPSIN adjudicate: it enumerates candidate descriptor
# placements, OPSIN-parses each candidate, and accepts the one whose relabeled
# structure is rdkit-canonical-identical (isotopes retained) to the original.
# Because OPSIN itself renumbers to give the isotope its lowest legal locant,
# the round-trip test IS the / (i) implementation — the
# wrong-locant candidate simply fails to reproduce the structure.

_ELEMENT_Z = {  # atomic numbers for tie-break
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "Cl": 17, "Se": 34, "Br": 35, "I": 53,
}


def _opsin_parse(candidate_name: str) -> Optional[str]:
    """Parse ``candidate_name`` to SMILES via the repo's shared OPSIN CLI
    helper (validation.opsin_roundtrip.opsin_parse — the SAME loader the name
    validity gate uses). Fail-closed (None) on any error / no JAR."""
    try:
        from ..validation.opsin_roundtrip import opsin_parse
        return opsin_parse(candidate_name)
    except Exception:
        return None


def _same_fixed_h_inchi(a: Chem.Mol, b: Chem.Mol) -> bool:
    """True iff ``a`` and ``b`` have the same fixed-H InChI (isotopic layers
    included); False when either has none.

    The isomeric canonical SMILES compare in ``_isotope_round_trips`` depends on
    the Kekule form of a conjugated ring RDKit does not perceive as aromatic: the
    14-atom periphery of '10b,10c-dihydropyren-1-yl' comes back from OPSIN with its
    double bonds in the other alternation than the input's, the same compound (one
    standard InChI) spelt as two SMILES, so a correctly placed descriptor failed
    its round trip (PubChem 1M: a (2H5)phenyl on such a skeleton). InChI does not
    depend on the Kekule form, and its fixed-H layer, unlike the standard key,
    still fixes the position of a nuclide on an exchangeable hydrogen
    ('[2H]OC(=O)CN' and 'OC(=O)CN[2H]' share the standard key, not the fixed-H
    InChI), which is what the SMILES compare guards. The caller still requires
    the full standard InChIKey to match as well."""
    try:
        from rdkit.Chem import inchi as _inchi
        ia = _inchi.MolToInchi(a, options="/FixedH")
        ib = _inchi.MolToInchi(b, options="/FixedH")
    except Exception:
        return False
    return bool(ia) and ia == ib


#: Set when ``_isotope_round_trips`` refuses a candidate whose parse is a
#: bond-shift isomer of the input (same constitution, a 4n alternating circuit).
_BOND_SHIFT_PARSE_REFUSED = contextvars.ContextVar(
    "orthonym_isotope_bond_shift_refused", default=False)


#: How many atom mappings ``_kekule_forms_of_one_compound`` tries before it
#: answers False (fail closed: the parse is then judged by its SMILES alone).
_KEKULE_MAPPING_CAP = 1000


def _single_bond_graph(mol: Chem.Mol) -> Chem.Mol:
    """``mol`` with every bond single and every atom's hydrogen count fixed as
    drawn (isotopes, charges and stereo tags kept): its constitution without its
    bond orders."""
    m = Chem.RWMol(mol)
    for atom in m.GetAtoms():
        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(False)
    for bond in m.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    m.UpdatePropertyCache(strict=False)
    return m.GetMol()


def _alternating_parts(diff: Dict[frozenset, Tuple[bool, bool]]
                       ) -> Optional[Tuple[List[int], List[Tuple[int, int, int]]]]:
    """The parts that ``diff`` -- the bonds that are double in exactly one of two
    drawings, ``{atom pair: (double in the first, double in the second)}`` --
    consists of: ``(circuit sizes, paths)``, each path ``(end, end, bond count)``;
    or None when it is not a set of disjoint circuits and paths whose bonds
    alternate between the two drawings."""
    adj: Dict[int, List[Tuple[int, bool]]] = {}
    for pair, (in_first, _in_second) in diff.items():
        x, y = tuple(pair)
        adj.setdefault(x, []).append((y, in_first))
        adj.setdefault(y, []).append((x, in_first))
    if any(len(v) > 2 or (len(v) == 2 and v[0][1] == v[1][1]) for v in adj.values()):
        return None
    sizes: List[int] = []
    paths: List[Tuple[int, int, int]] = []
    seen: set = set()
    # Paths first: each starts at an atom with one differing bond.
    for start in [x for x, v in adj.items() if len(v) == 1]:
        if start in seen:
            continue
        bonds, prev, cur = 0, None, start
        while True:
            seen.add(cur)
            nxt = [y for y, _ in adj[cur] if y != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            bonds += 1
        paths.append((start, cur, bonds))
    for start in adj:
        if start in seen:
            continue
        size, prev, cur = 0, None, start
        while True:
            seen.add(cur)
            size += 1
            nxt = [y for y, _ in adj[cur] if y != prev]
            if not nxt:
                return None
            prev, cur = cur, nxt[0]
            if cur == start:
                break
        sizes.append(size)
    return sizes, paths


def _charge_free(mol: Chem.Mol) -> Chem.Mol:
    """``mol`` with every formal charge set to 0: the graph the atom mapping of
    ``_kekule_forms_of_one_compound`` is searched on (RDKit matches a query atom's
    charge exactly, so two drawings that place a delocalized charge on different
    atoms would have no complete mapping, and an exhaustive search for one)."""
    m = Chem.RWMol(mol)
    for atom in m.GetAtoms():
        atom.SetFormalCharge(0)
    m.UpdatePropertyCache(strict=False)
    return m.GetMol()


def _kekule_forms_of_one_compound(a: Chem.Mol, b: Chem.Mol) -> bool:
    """True iff ``a`` and ``b`` are ONE compound drawn as two of its resonance
    structures -- two Kekule alternations of a ring system, or one charge placed on
    different atoms of a delocalized group -- and not two bond-shift (valence)
    isomers.

    The fixed-H InChI compare (``_same_fixed_h_inchi``) accepts a parse whose
    double bonds sit elsewhere than the input's, because InChI records no bond
    orders. That is right for the 14-atom periphery of '10b,10c-dihydropyren-1-yl'
    (a 4n+2 circuit: two Kekule structures of one aromatic ring system, which OPSIN
    writes in the other alternation), and wrong for a ring whose alternations are
    distinct compounds: a 4n circuit -- cycloocta-1,3,5,7-tetraene (8 atoms), the
    [4n]annulenes, the pentalene and heptalene peripheries (8, 12) -- has two
    bond-shift isomers, which a name cites by its ene locants: 'cycloocta-1,3,5,7-
    tetraene (PIN)', the Blue Book, "In monocyclic homogeneous
    unsaturated compounds, one double or triple bond is always allocated the locant
    '1'";:16554), 'cyclododeca-1,3,5,7,9,11-hexaene (PIN)' for [12]annulene
    ,:8081). There '1-methyl(2-2H)cycloocta-1,3,5,7-tetraene' (D on the
    carbon double-bonded to C1) is not the input '1-methyl(8-2H)...' (D on the
    carbon single-bonded to C1), although the fixed-H InChIs are equal.

    A delocalized charge is one species whatever atom a drawing puts it on:
    DELOCALIZED RADICALS AND IONS (:43540), "Delocalization in names involving one
    radical or ionic center in an otherwise conjugated double bonds structure is
    denoted by the appropriate suffix without locants" (:43542). So the 18O of
    '[18O-][N+](=O)c1ccccc1' and of '[18O]=[N+]([O-])c1ccccc1' is one label of
    one nitrobenzene, and a D-tropylium with the charge drawn at another carbon is
    the same cation. The two drawings then differ by an alternating PATH of an even
    number of bonds whose two end atoms exchange the charge.

    Decided on the atoms: the two drawings' single-bond graphs, charges set aside,
    are matched atom for atom (element, isotope and hydrogen count equal); for some
    matching, the bonds whose order differs must form disjoint alternating circuits
    of 4n+2 atoms each and alternating paths of an even number of bonds whose two
    ends, and only those atoms, carry opposite changes of formal charge. Up to
    ``_KEKULE_MAPPING_CAP`` matchings are tried; False when none qualifies (fail
    closed). The graphs are compared by canonical SMILES before the search, so the
    search only runs on two isomorphic graphs, where every matching it counts is
    complete and the cap bounds the work."""
    try:
        ka, kb = Chem.Mol(a), Chem.Mol(b)
        Chem.Kekulize(ka, clearAromaticFlags=True)
        Chem.Kekulize(kb, clearAromaticFlags=True)
        ga, gb = _single_bond_graph(ka), _single_bond_graph(kb)
        qa, qb = _charge_free(ga), _charge_free(gb)
    except Exception:
        return False
    if (ga.GetNumAtoms() != gb.GetNumAtoms()
            or ga.GetNumBonds() != gb.GetNumBonds()):
        return False
    try:
        if Chem.MolToSmiles(qa) != Chem.MolToSmiles(qb):
            return False
    except Exception:
        return False
    double = Chem.BondType.DOUBLE
    try:
        matches = qa.GetSubstructMatches(
            qb, uniquify=False, useChirality=False,
            maxMatches=_KEKULE_MAPPING_CAP)
    except Exception:
        return False
    for match in matches:
        if any(ka.GetAtomWithIdx(j).GetAtomicNum() != kb.GetAtomWithIdx(i).GetAtomicNum()
               or ka.GetAtomWithIdx(j).GetIsotope() != kb.GetAtomWithIdx(i).GetIsotope()
               or ga.GetAtomWithIdx(j).GetTotalNumHs() != gb.GetAtomWithIdx(i).GetTotalNumHs()
               for i, j in enumerate(match)):
            continue
        charge_change = {}
        for i, j in enumerate(match):
            dq = (ka.GetAtomWithIdx(j).GetFormalCharge()
                  - kb.GetAtomWithIdx(i).GetFormalCharge())
            if dq:
                charge_change[j] = dq
        diff: Dict[frozenset, Tuple[bool, bool]] = {}
        orders_agree = True
        for bond_b in kb.GetBonds():
            i, j = match[bond_b.GetBeginAtomIdx()], match[bond_b.GetEndAtomIdx()]
            bond_a = ka.GetBondBetweenAtoms(i, j)
            if bond_a is None:
                orders_agree = False
                break
            ta, tb = bond_a.GetBondType(), bond_b.GetBondType()
            if ta == tb:
                continue
            if {ta, tb} != {double, Chem.BondType.SINGLE}:
                orders_agree = False
                break
            diff[frozenset((i, j))] = (ta == double, tb == double)
        if not orders_agree:
            continue
        parts = _alternating_parts(diff)
        if parts is None:
            continue
        sizes, paths = parts
        if not all(n % 4 == 2 for n in sizes):
            continue
        ends = set()
        paths_ok = True
        for x, y, n_bonds in paths:
            dx, dy = charge_change.get(x, 0), charge_change.get(y, 0)
            if n_bonds % 2 or not dx or dx != -dy:
                paths_ok = False
                break
            ends.update((x, y))
        if paths_ok and ends == set(charge_change):
            return True
    return False


def _bond_order_free_key(mol: Chem.Mol) -> Optional[str]:
    """A canonical SMILES of ``mol`` with every bond made single and every atom's
    hydrogen count fixed as drawn (isotopes and stereo tags kept), or None.

    It names WHICH atom carries a label without depending on the Kekule form of
    a conjugated ring RDKit does not perceive as aromatic (``_same_fixed_h_inchi``):
    ``_decorate_distinct_multi_locant`` maps each probe locant to the atom it
    labels by this key when the plain canonical SMILES of OPSIN's parse and of the
    input disagree only in that alternation. Only a search key: every candidate
    it leads to is still judged by ``_isotope_round_trips``."""
    try:
        return Chem.MolToSmiles(_single_bond_graph(mol))
    except Exception:
        return None


def _isotope_round_trips(candidate_name: str, original_mol: Chem.Mol,
                         stereo_blind: bool = False) -> bool:
    """True iff OPSIN parses ``candidate_name`` to a structure that is rdkit-
    canonical-identical (isotopes retained) to ``original_mol``. Fail-closed
    on parse failure / no JAR.

    ``stereo_blind=True`` compares CONSTITUTION only (stereochemistry stripped
    from both sides, InChIKey layer not consulted). Used by the isotope-induced
    stereocentre path (FIX-D, to find the descriptor placement that
    reproduces the connectivity before the stereodescriptor prefix is chosen --
    the full-stereo compare is then re-applied to the prefixed candidate, so no
    isotopologue or stereoisomer is ever shipped on a stereo-blind pass alone.
    """
    smi = _opsin_parse(candidate_name)
    if not smi:
        return False
    parsed = Chem.MolFromSmiles(smi)
    if parsed is None:
        return False
    try:
        if stereo_blind:
            a, b = Chem.Mol(parsed), Chem.Mol(original_mol)
            Chem.RemoveStereochemistry(a)
            Chem.RemoveStereochemistry(b)
            return (Chem.MolToSmiles(a) == Chem.MolToSmiles(b)
                    or (_same_fixed_h_inchi(a, b)
                        and _kekule_forms_of_one_compound(a, b)))
        # A parse whose SMILES differs from the input's is accepted only as the
        # other Kekule alternation of the same compound: equal fixed-H InChI and
        # every alternating circuit 4n+2 (``_kekule_forms_of_one_compound``), so
        # a bond-shift isomer ('1-methyl(2-2H)cycloocta-1,3,5,7-tetraene' for the
        # input's '(8-2H)') is refused. The refusal of a bond-shift isomer is
        # recorded (``_BOND_SHIFT_PARSE_REFUSED``): the name accepted after it
        # carries a higher locant than the one of the same constitution.
        if Chem.MolToSmiles(parsed) != Chem.MolToSmiles(original_mol):
            if not _same_fixed_h_inchi(parsed, original_mol):
                return False
            if not _kekule_forms_of_one_compound(parsed, original_mol):
                _BOND_SHIFT_PARSE_REFUSED.set(True)
                return False
        # (a review P4 related note): RDKit CanonSmiles can call a charged /
        # aromatic multi-component pair equal while InChI distinguishes them (the
        # same weakness fixed in ``namer._rt_match`` — measured on a cyanine dye).
        # Back the isomeric-CanonSmiles compare with the full InChIKey (isotope
        # layer included) so a wrong isotopologue on such a species cannot pass.
        # Strictly tightening; fall back to the CanonSmiles match only when InChI is
        # genuinely unavailable for a side (no regression).
        from rdkit.Chem import inchi as _inchi
        _ik_p = _inchi.MolToInchiKey(parsed)
        _ik_o = inchikey_of(original_mol)
        if _ik_p and _ik_o:
            return _ik_p == _ik_o
        return True
    except Exception:
        return False


#: A parent hydride's indicated-hydrogen prefix, e.g. the ``1H-`` of
#: ``1H-pyrrole``. The hyphen is part of the match because the descriptor goes
#: *in place of* it -- see ``_insertion_offsets``.
_INDICATED_H_PREFIX_RE = re.compile(r"-\d+H-")

#: An indicated hydrogen at the START of a bracketed sub-name, e.g. the ``1H-`` of
#: ``(1H-indol-2-yl)…`` / ``[1H-inden-…]``. Here the ``\d+H-`` follows an OPENING
#: BRACKET, not a hyphen, so ``_INDICATED_H_PREFIX_RE`` misses it and the descriptor
#: slot (right after the bracket, before the digit) was never offered -- a labelled
#: fused-ring substituent (indol-2-yl carrying ¹³C) had nowhere legal to place its
#: descriptor and failed closed (a review review 2026-09-08).
_BRACKET_INDICATED_H_RE = re.compile(r"[(\[{]\d+H-")

#: An interior sub-parent introduced by its OWN locant set, e.g. the ``-1,4-`` of
#: ``3,6-dimethyl-1,4-dioxane-2,5-dione`` or the ``-1,3-`` of
#: ``2,4-dioxo-1,3-diazaspiro[4.7]dodecane``. A nuclide ON that sub-parent puts its
#: descriptor immediately before the sub-parent: "inserted before the
#: part... that is isotopically substituted"), replacing the introducing hyphen:
#: ``3,6-dimethyl`` + ``(2H2)`` + ``-1,4-dioxane-2,5-dione``. That slot begins with
#: a digit, so it was covered by neither the alphabetic-start rule nor the
#: indicated-H rules, and a labelled interior ring failed closed. More targeted
#: than "every hyphen before a digit": the locant set must itself be followed by a
#: hyphen and a stem letter (``[\d,]+-[a-z]``), so a plain substituent locant like
#: the ``-2-`` of ``prop-2-en`` is not offered. Every candidate stays OPSIN-RT
#: gated by the caller, so any surplus offset can only be discarded, never shipped.
_INTERIOR_LOCANT_STEM_RE = re.compile(r"-(?=[\d,]+-[a-z])")

#: The opening bracket of a ring-assembly name inside a name, e.g. the ``[`` of
#: ``([1,1'-biphenyl]-4-yl)`` or ``[2,2'-bipyridin]-3-yl``.
#: (the Blue Book, "They also enclose ring assembly names when these are
#: followed by a principal group suffix or a cumulative suffix") puts the whole
#: ring-assembly stem in brackets when a suffix follows, so the slot "before the
#: part of the compound that is isotopically substituted" is the
#: bracket itself: ``[(2',3',4',5',6'-2H5)[1,1'-biphenyl]-4-yl]``,
#::7469: those brackets are ignored in the nesting order of the enclosing marks).
#: The stem begins with a locant set, not a letter, and the
#: bracket follows ``(`` or ``-``, so none of the rules above offered it and a
#: labelled ring of a biphenyl-yl prefix failed closed (PubChem 1M: five
#: deuterated OLED hosts the paper named). Every candidate stays OPSIN-RT gated.
_RING_ASSEMBLY_STEM_RE = re.compile(r"\[(?=\d+'*(?:[,:]\d+'*)+-[a-z])")

#: A SUFFIX that names a labelled HETEROATOM whose nuclide is placed immediately
#: BEFORE the suffix it modifies (``propane-1-(34S)thiol``), rather than floated to
#: the front of the parent -- "position not normally denoted by a locant"
#: / ``the Blue Book Blue Book`` ("Specific positions of nuclides
#: must be indicated... preceding the nuclide symbol"). Matched narrowly at a token
#: boundary: a hyphen must introduce the suffix (``skel[off-1] == "-"``, so
#: ``methan|ol`` / ``phen|ol`` / ``hexan|oic acid`` are NOT slots) and a non-letter
#: must follow the stem (so the ``ol`` inside ``indole`` is not a slot).
#:
#: ⚠ ``-ol`` (alcohol O) and ``-amine`` (amine N) are DELIBERATELY EXCLUDED. The
#: gold protect pin ``CC[18OH]`` -> ``(18O)ethan-1-ol`` (W2F-P5-P4, "parent-front
#: descriptor") keeps a locant-free O nuclide at the FRONT of the parent, and a
#: co-labelled amine N must COMBINE with a positional carbon nuclide into one
#: front descriptor -- ``C[13CH2][15NH2]`` -> ``(1-13C,15N)ethan-1-amine``
#:, ``:44202``; gold V47-/03), which the suffix-adjacent split
#: ``(1-13C)ethan-1-(15N)amine`` broke. Placing the ``ol``/``amine`` nuclide at a
#: distinct suffix slot let ``_decorate_multi_position`` emit that split instead of
#: falling through to the combined ``_decorate_mixed_locant`` form. Only the
#: chalcogen analogues of the alcohol suffix -- ``-thiol`` (S), ``-selenol`` (Se),
#: ``-tellurol`` (Te) -- take the suffix-adjacent slot (``propane-1-(34S)thiol``,
#: ``propane-1-(77Se)selenol``, ``propane-1-(125Te)tellurol``; all OPSIN-round-trip).
_HETEROATOM_SUFFIX_STEM_RE = re.compile(
    r"(?:carboxylic|carbonitrile|carbaldehyde|nitrile|thiol|selenol"
    r"|tellurol|one|oic|al)(?![a-z])"
)

#: The "carbo-affix" suffixes -- ``carboxylic`` / ``carbonitrile`` /
#: ``carbaldehyde`` -- name the functional heteroatom (the carboxyl O, the
#: nitrile N, the aldehyde O) as part of a suffix that attaches DIRECTLY to the
#: ring/parent name with NO hyphen (``benzenecarbaldehyde``,
#: ``benzenecarboxylic acid``). So their heteroatom slot is preceded by a letter,
#: not the ``-`` the chain suffixes above require. A locant-free heteroatom
#: descriptor there (``benzene(18O)carbaldehyde``) belongs immediately before the
#: suffix, not floated to the parent front (``(18O)benzene...``).
_CARBO_AFFIX_SUFFIX_RE = re.compile(
    r"(?:carboxylic|carbonitrile|carbaldehyde)(?![a-z])"
)


#: A locant set as it opens a name part: numerals with an optional letter / primes
#: (``2``, ``3a``, ``4'``) or an italic element locant (``N``, ``N1``), comma-joined,
#: closed by the hyphen that introduces what it locates.
_LEADING_LOCANT_SET_RE = re.compile(
    r"(?:\d+[a-z]?'*|[A-Z][a-z]?\d*'*)(?:,(?:\d+[a-z]?'*|[A-Z][a-z]?\d*'*))*-")

#: A basic or derived multiplying prefix that can open a multiplied substituent
#: (``2,4-dimethyl``, ``3,5-bis(...)``, ``2,6-di-tert-butyl``).
_OPENING_MULTIPLIER_RE = re.compile(
    r"(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca|bis|tris|tetrakis)"
    r"(?=[a-z(\[{]|-(?:tert|sec)-)")


def _slot_precedes_substituent_locant(skel: str, off: int) -> bool:
    """Does an insertion at ``off`` sit in front of the attachment locant of a
    DETACHABLE substituent prefix (the slot before ``-2-phenyl`` in
    ``...-2-phenyl-1-benzofuran``, or the front of ``3-phenylpropanoic acid``)?

     (the Blue Book): the nuclide symbols, "preceded by any necessary
    locant(s)", go "before the part of the compound that is isotopically
    substituted". A substituent prefix's attachment locant belongs to the parent it
    locates, not to the prefix, so a descriptor for the prefix's own atoms follows
    that locant: '4-(2-14C)ethylbenzoic acid (PIN)' (:43818), '2-(13C)methyl-3-
    methylpyridine (PIN)' (:43764), '1-(13C)methyl(2-13C)benzene (PIN)' (:43736);
    and a descriptor for the parent's atoms goes before the parent stem, after the
    prefixes: '1-phenyl(1,2-13C2)ethan-1-one (PIN)' (:43732). The slot before a
    locant is right only when that locant belongs to the labelled part itself -- an
    indicated hydrogen ('(15N)-1H-indole (PIN)',:43790), the heteroatom locants of
    a replacement or Hantzsch-Widman parent ('...(2H2)-1,4-dioxane', the
    ``_INTERIOR_LOCANT_STEM_RE`` slot).

    OPSIN reads the descriptor in front of a substituent locant leniently as
    labelling that substituent, so the round trip accepts
    '(2,3,4,5,6-2H5)-2-phenyl-1-benzofuran' as well as the Blue Book's
    '2-(2,3,4,5,6-2H5)phenyl-1-benzofuran'; the oracle cannot choose, the grammar
    must. True when what follows the locant set opens a substituent prefix: an
    enclosing mark, a multiplier, an italicized structural prefix, or a prefix
    component the shared prefix vocabulary recognises (``_is_recognised_prefix_
    component``) that ends at a hyphen / mark or, as an ``...yl`` prefix, runs
    straight into the parent ('phenylpropanoic acid'). A replacement or ring stem
    ('azatricyclo', 'dioxane', 'triazin', 'benzofuran') is not a prefix component,
    and a component that only BEGINS a stem ('oxo' in 'oxolane') is not accepted,
    so the legitimate sub-parent slots keep their rank.
    """
    if off < 0 or off >= len(skel):
        return False
    if off == 0:
        text = skel
    elif skel[off] == "-":
        text = skel[off + 1:]
    elif skel[off].isupper() and skel[off - 1] in "-([{":
        # the slot right before an italic element locant ('-N-phenyl')
        text = skel[off:]
    else:
        return False
    m = _LEADING_LOCANT_SET_RE.match(text)
    if not m:
        return False
    after = text[m.end():]
    if not after:
        return False
    if after[0] in "({":
        return True
    if after[0] == "[" and not after[1:2].isdigit():
        return True
    mm = _OPENING_MULTIPLIER_RE.match(after)
    if mm:
        rest = after[mm.end():]
        if rest[:1] in "([{-":
            return True
        after = rest
    from ..assembly.naming_utils import (
        _is_recognised_prefix_component,
        strip_italicized_structural_prefix,
    )
    if strip_italicized_structural_prefix(after)[1]:
        return True
    for k in range(len(after), 2, -1):
        head = after[:k]
        if not head.isalpha() or not _is_recognised_prefix_component(head):
            continue
        nxt = after[k:k + 1]
        if not nxt or nxt in "-([{":
            return True
        if nxt.isalpha() and head.endswith(("yl", "ylidene", "ylidyne")):
            return True
    return False


def _insertion_offsets(skel: str) -> List[int]:
    """Offsets in ``skel`` at which an isotopic descriptor may be inserted.

    : the descriptor "is inserted before the part of the compound that is
    isotopically substituted". The enumeration does not parse the skeleton's
    grammar -- it offers every plausible offset and lets the OPSIN round-trip
    oracle pick the one that is right -- so the only thing that matters here is
    that no legal placement is left out of the candidate set.

    Alphabetic starts cover a bare parent stem. They do NOT cover a parent
    carrying an indicated hydrogen: the slot before ``1H-pyrrole`` begins with a
    digit, so a labelled molecule that also has substituent prefixes had nowhere
    legal to put the descriptor and failed closed -- ``[2H]C1=C(N(C=C1)C)[N+](=O)[O-]``
    named nothing at all, though its unlabelled skeleton names fine.

     prints the construction four times, every one a PIN
    (``the Blue Book Blue Book``-``:43796``)::

        (15N)-1H-indole (PIN)
        2,3-dihydro(15N)-1H-indole (PIN)
        2,3-dihydro(2,3-2H2,15N)-1H-indole (PIN)

    -- descriptor, then a hyphen, then the indicated-hydrogen prefix. So the
    descriptor replaces the hyphen that introduces that prefix, and the offset to
    offer is the hyphen's own index: ``1-methyl-2-nitro`` + ``(3-2H1)`` +
    ``-1H-pyrrole``.

    A third gap is an interior sub-parent introduced by its own locant set
    (``-1,4-dioxane``): a nuclide on that ring needs the slot before its locant
    set, which begins with a digit -- see ``_INTERIOR_LOCANT_STEM_RE``.

    Every candidate built from these offsets is still OPSIN-round-trip gated by
    the caller, so a surplus offset can only ever be discarded, never shipped.
    """
    offsets = [i for i in range(1, len(skel)) if skel[i].isalpha()]
    offsets.extend(m.start() for m in _INDICATED_H_PREFIX_RE.finditer(skel))
    # A bracketed indicated-H opening: offer the slot right AFTER the bracket
    # (before the digit), so '(1H-indol-2-yl)…' can take '((desc)-1H-indol-2-yl)…'
    # -> escalated to '[(desc)-1H-indol-2-yl]…'.
    offsets.extend(m.start() + 1 for m in _BRACKET_INDICATED_H_RE.finditer(skel))
    # An interior sub-parent introduced by its own locant set: offer the slot at
    # the introducing hyphen so a nuclide on that ring can place its descriptor
    # (e.g. '3,6-dimethyl' + '(2H2)' + '-1,4-dioxane-2,5-dione'). RT-gated.
    offsets.extend(m.start() for m in _INTERIOR_LOCANT_STEM_RE.finditer(skel))
    # A bracketed ring-assembly stem ('[1,1'-biphenyl]-4-yl'): the slot is its
    # opening bracket (``_RING_ASSEMBLY_STEM_RE``). RT-gated.
    offsets.extend(m.start() for m in _RING_ASSEMBLY_STEM_RE.finditer(skel))
    # A bracketed substituent ring opening with skeletal-replacement locants
    # ('[9,24-dioxahexacyclo...'): the slot right after the mark
    # (``_BRACKET_SKELETAL_LOCANT_RE``). RT-gated.
    offsets.extend(m.start() + 1
                   for m in _BRACKET_SKELETAL_LOCANT_RE.finditer(skel))
    return sorted(set(offsets))


def _p4542_p4543_key(groups) -> tuple:
    """ then ordering key for a candidate placement.

    Prefer the placement giving the lowest locant to the HIGHER atomic number
    , then to the HIGHER mass number. Implemented as: for
    each group sorted by locant ascending, emit (-Z, -mass); the lexicographically
    smallest key is the preferred placement. the Blue Book / the Blue Book / the Blue Book(i).
    """
    key = []
    for group in sorted(
        groups, key=lambda g: (g[0] if g[0] is not None else 1 << 30)
    ):
        # groups are (locant, mass, element, count[, max_at_pos]) -- index the
        # first three so a 4- or 5-tuple both unpack.
        _locant, mass, el = group[0], group[1], group[2]
        z = _ELEMENT_Z.get(el, 0)
        key.append((-z, -mass))
    return tuple(key)


def _bracket_depth(skel: str, off: int) -> int:
    """Enclosing-mark nesting depth at ``off`` in ``skel`` -- the count of ``(``
    or ``[`` opened before ``off`` and not yet closed. Depth 0 means the
    descriptor modifies the PARENT hydride; depth > 0 means it is nested inside a
    SUBSTITUENT's own enclosing marks (``(1,1,2,2,2-pentafluoro(13C1)ethyl)…``).
    The parent-scope locant gate applies only at depth 0 -- a
    substituent-scoped descriptor is a separate rule this fix leaves
    untouched (PREP-T3: the ``(13C1)ethyl`` rows are out of scope)."""
    head = skel[:off]
    return (head.count("(") - head.count(")")
            + head.count("[") - head.count("]"))


def _is_heteroatom_suffix_slot(skel: str, off: int) -> bool:
    """True when ``off`` is the slot immediately before a hyphen-introduced
    heteroatom SUFFIX (the ``ol`` of ``cyclopentan-1-ol``), i.e. the place a
    locant-free O/N descriptor belongs /. Narrow: the
    suffix must be hyphen-introduced (``methanol``/``phenol``/``hexanoic acid``
    are excluded, their stem does not follow a ``-``) and the stem must end at a
    token boundary (the ``ol`` inside ``indole`` is excluded).

    A ``carbo``-affix suffix (``carboxylic``/``carbonitrile``/``carbaldehyde``)
    is the one heteroatom suffix that attaches with NO hyphen -- fused straight
    onto the ring name (``benzenecarbaldehyde``) -- so it is recognised when
    preceded by a LETTER instead. This slot is only consulted for a LOCANT-FREE
    descriptor, and a locant-free heteroatom is reached only when its position is
    unambiguous, i.e. exactly one such suffix group -- never a multiplied
    ``...dicarbaldehyde`` (whose single-atom label would require a locant), so
    the ``di``/``tri`` prefix cannot be mis-split."""
    if off <= 0:
        return False
    if skel[off - 1] == "-":
        return _HETEROATOM_SUFFIX_STEM_RE.match(skel[off:]) is not None
    if skel[off - 1].isalpha():
        return _CARBO_AFFIX_SUFFIX_RE.match(skel[off:]) is not None
    return False


#: A name part opening with a skeletal replacement ('a') prefix or a
#: Hantzsch-Widman stem (the (c) set of ``naming_utils``).
from ..assembly.naming_utils import _REPLACEMENT_FRONT_RE as _REPLACEMENT_PART_RE  # noqa: E402


#: A stereodescriptor block closing the text before a slot: '(2R,4S)', '(1r,4r)',
#: '(E)', '(2Z,4E)'.
_STEREO_BLOCK_END_RE = re.compile(
    r"\((?:\d*[a-z']*[RSEZrs](?:\*)?|rel|cis|trans)"
    r"(?:,(?:\d*[a-z']*[RSEZrs](?:\*)?))*\)$")


def _replacement_locant_follows(rest: str) -> bool:
    """Does ``rest`` open with a numeral locant set (after an optional hyphen)
    that introduces a replacement prefix or Hantzsch-Widman stem ('-1,4-dioxane',
    '2-azatricyclo')?"""
    m = re.match(r"-?((?:\d+[a-z]?'*)(?:,\d+[a-z]?'*)*)-", rest)
    return bool(m) and bool(_REPLACEMENT_PART_RE.match(rest[m.end():]))


def _placement_quality(skel: str, off: int, locant_free: bool = False) -> int:
    """ placement rank for an isotope descriptor spliced at ``off``:
    ``-1`` when the descriptor is a locant-free label sitting directly before the
    heteroatom SUFFIX it modifies (``cyclopentan-1-(18O)ol``); ``0`` when it sits
    directly before the AFFIX / parent name it modifies (an alphabetic character,
    or the ``-nH-`` indicated-hydrogen prefix); ``1`` when it is DETACHED from that
    affix by an intervening locant (or stereo prefix) at the front of the name.

    OPSIN parses both ``5-(81Br)bromo…`` (rank 0) and the front-detached
    ``(81Br)5-bromo…`` (rank 1) to the same structure, so the round-trip oracle
    cannot separate them; the Blue Book uses only the former -- the Blue Book
    ``(2R)-1-(131I)iodo-3-iodopropan-2-ol`` is locant, then descriptor, then
    affix, with the stereo descriptor kept at the front. Ranking the detached
    form BELOW the adjacent one lets the correct placement win the offset
    tie-break without EXCLUDING the detached form, so a molecule whose only
    round-tripper is the detached form still emits rather than failing closed.
    (The legitimate front-of-parent hyphen-joined placement for a
    skeletal-replacement locant, ``_front_hyphen_candidate``, is scored 0
    explicitly by its caller -- it is the parent-scope placement.)

    The ``-1`` suffix-adjacent rank is the / fix: a heteroatom
    named by a suffix is "not normally denoted by a locant", so its (locant-free)
    descriptor is spliced immediately before the suffix -- ``(1²H₁)ethan-1-(²H)ol
    (PIN)`` ``the Blue Book Blue Book`` -- rather than floated to the front
    of the parent. That slot is ALREADY offered by the alphabetic-start rule (the
    suffix begins with a letter), so both ``…-1-(18O)ol`` and the front-detached
    ``(18O)…-1-ol`` round-trip; the front splice used to win only because it has the
    lower offset. It is gated on ``locant_free`` so a LOCANTED carbon label keeps
    its front placement (``(2-13C)ethan-1-ol``, not ``ethan-1-(2-13C)ol``)."""
    if locant_free and _is_heteroatom_suffix_slot(skel, off):
        return -1
    rest = skel[off:]
    # A replacement ('a') prefix or Hantzsch-Widman stem carries its heteroatom
    # locants as part of its own name ('2-azatricyclo[...]', '1,4-dioxane',
    # '1-oxaethan-1-yl'): the descriptor goes before that locant set, hyphen-joined
    #, the Blue Book, "when the name, or a part of a name, includes
    # a preceding locant, a hyphen is inserted"; the slot of '(15N)-1H-indole
    # (PIN)',:43790). Splicing it between the locants and the prefix
    # ('2-(2H8)azatricyclo') detaches the locants from what they locate.
    # (Not at the hyphen after a stereodescriptor block, which stays:
    # '(2R,4S)-(2-2H)-3-oxatricyclo[...]'.)
    if _replacement_locant_follows(rest) and not (
            rest.startswith("-") and _STEREO_BLOCK_END_RE.search(skel[:off])):
        return 0
    if off > 0 and _REPLACEMENT_PART_RE.match(rest) and re.search(
            r"(?:^|[-(\[{])(?:\d+[a-z]?'*)(?:,\d+[a-z]?'*)*-$", skel[:off]):
        return 2
    rest = skel[off:]
    if rest[:1].isupper() and _LEADING_LOCANT_SET_RE.match(rest):
        # An italic element locant ('N-methyl', 'N,N-dimethyl', 'S-methyl') is a
        # locant, not the affix: the descriptor follows it,
        # (the Blue Book, nuclide symbols "preceded by any necessary
        # locant(s)"): 'N-(2H3)methylbenzamide', not '(2H3)N-methylbenzamide'.
        return 1
    if rest[:1].isalpha():
        return 0
    if _INDICATED_H_PREFIX_RE.match(rest):
        return 0
    return 1


def _isotopomer_key(cand: str) -> Optional[str]:
    """The distinct-structure key OPSIN assigns a candidate name (isotope-aware
    InChIKey, SMILES fallback); None when it does not parse. Shared by the
    uniqueness oracle so the seed and the probes are compared at one layer."""
    smi = _opsin_parse(cand)
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        from rdkit.Chem import inchi as _inchi
        return _inchi.MolToInchiKey(m) or Chem.MolToSmiles(m)
    except Exception:
        return Chem.MolToSmiles(m)


def _isotope_carrier(atom: Chem.Atom) -> int:
    """The heavy-atom POSITION a nuclide occupies: itself for a heavy nuclide,
    else the heavy atom that bears the labelled H. A descriptor's locant names
    this heavy position, so is evaluated on carriers (the methyl's D
    are all on one carrier carbon)."""
    if atom.GetSymbol() == "H":
        for nb in atom.GetNeighbors():
            if nb.GetSymbol() != "H":
                return nb.GetIdx()
    return atom.GetIdx()


def _element_scope(mol: Chem.Mol, start: int, el: str) -> set:
    """The named-unit SCOPE of a carrier: the connected run of element-``el``
    atoms reachable from ``start`` through ``el``-``el`` bonds that stay WITHIN
    one named unit (one substituent or one parent). Two boundaries stop the run:

      * a heteroatom -- so the O-methyl of ``methyl acetate`` scopes to its lone
        carbon (``(13C)methyl``,, and an ester's alkyl stays clear of
        the acyl (``(1-14C)ethyl propanoate``);
      * an EXOCYCLIC bond onto a ring (a substituent-attachment bond) -- so the
        ``aminomethyl`` CH2 of ``1-[amino(14C)methyl]cyclopentan-1-ol`` scopes to
        its lone carbon (locant omitted) rather than merging into the ring, while
        ``ethane``'s two chain carbons and ``benzene``'s six ring carbons each
        stay in one scope.

    A C-C bond is therefore traversed only when it lies inside a ring (fused/ring
    positions) or joins two acyclic (chain) atoms."""
    seen = {start}
    stack = [start]
    while stack:
        i = stack.pop()
        ai = mol.GetAtomWithIdx(i)
        for nb in ai.GetNeighbors():
            j = nb.GetIdx()
            if nb.GetSymbol() != el or j in seen:
                continue
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond.IsInRing() or (not ai.IsInRing() and not nb.IsInRing()):
                seen.add(j)
                stack.append(j)
    return seen


def _single_label_is_immaterial(original: Chem.Mol, mass: int, el: str,
                                carrier: int, orbit: set) -> bool:
    """For the -clause-1 case (ONE label on a symmetric multi-atom
    orbit): True iff which orbit member carries it is genuinely immaterial --
    placing the same label on every other member yields the SAME molecule
    (isotope- AND stereo-aware ``InChIKey``).

    The stripped-parent orbit test alone omits ``(1-2H1)`` from
    ``(2R)-(1-2H1)propan-2-ol``, because C1 and C3 (the two methyls) are one
    orbit in achiral propan-2-ol. But labelling ONE of them makes C2 a
    stereocentre "possibility of isomers": the C1 and C3 labellings
    are enantiomers), so the locant must be cited. That induced desymmetrisation
    is invisible to ``FindPotentialStereo`` on the stripped parent, so it is
    caught here by actually moving the label. Cheap: reached only in the rare
    single-label symmetric-orbit branch, and uses RDKit alone (no OPSIN)."""
    try:
        oh = Chem.AddHs(original)
        key0 = Chem.MolToInchiKey(oh)
    except Exception:
        return False
    if el == "H":
        lab = [nb.GetIdx() for nb in oh.GetAtomWithIdx(carrier).GetNeighbors()
               if nb.GetSymbol() == "H" and nb.GetIsotope() == mass]
    else:
        lab = [carrier]
    if not lab:
        return False
    for other in orbit:
        if other == carrier:
            continue
        moved = Chem.RWMol(oh)
        for i in lab:
            moved.GetAtomWithIdx(i).SetIsotope(0)
        if el == "H":
            avail = [nb.GetIdx() for nb in oh.GetAtomWithIdx(other).GetNeighbors()
                     if nb.GetSymbol() == "H"]
        else:
            avail = [other]
        if len(avail) < len(lab):
            return False
        for i in avail[:len(lab)]:
            moved.GetAtomWithIdx(i).SetIsotope(mass)
        try:
            if Chem.MolToInchiKey(moved.GetMol()) != key0:
                return False
        except Exception:
            return False
    return True


def _split_yields_distinct_isomer(original: Chem.Mol, mass: int, el: str,
                                  carrier: int, orbit: set) -> bool:
    """ ("Locants are not omitted when there is a possibility of
    isomers", ``the Blue Book Blue Book``): True iff SPLITTING a
    count>=2 nuclide group that all sits on ONE carrier across its symmetric
    orbit -- moving one nuclide to a symmetric partner (1,1 -> 1,2) -- yields
    an isotopomer with a DISTINCT isotope-aware ``InChIKey`` from the
    all-on-carrier arrangement.

    :func:`_single_label_is_immaterial` only tests moving the WHOLE group to a
    partner (2,0 -> 0,2), which reproduces the molecule on a symmetric orbit and
    so wrongly licensed omission for e.g. ``(2H2)ethane-1,2-diyl``. But
    ``(1,1-2H2)`` and ``(1,2-2H2)`` are DISTINCT isotopomers, so more than one
    isotopomer of that count exists and the locant is required. A gem-vs-vicinal
    split is a constitutional isotope difference, so this fires for the diol /
    linker / dibromoethane family while an all-collapse split (which never
    produces a distinct key) is NOT over-cited -- the decision is gated on the
    actual InChIKey comparison, never on the mere presence of a symmetric partner.

    A count-1 group cannot be split (there is nothing to move partially), so this
    returns False for it and the caller's whole-move test governs the single-label
    case. Heavy nuclides (13C, 18O,...) hold one atom per position, so their
    carrier count is 1 and this is likewise a no-op for them. RDKit only (no OPSIN).
    """
    try:
        oh = Chem.AddHs(original)
        key0 = Chem.MolToInchiKey(oh)
    except Exception:
        return False
    if el == "H":
        lab = [nb.GetIdx() for nb in oh.GetAtomWithIdx(carrier).GetNeighbors()
               if nb.GetSymbol() == "H" and nb.GetIsotope() == mass]
    else:
        lab = [carrier]
    if len(lab) < 2:
        return False                              # nothing to split (count 1)
    for other in orbit:
        if other == carrier:
            continue
        moved = Chem.RWMol(oh)
        moved.GetAtomWithIdx(lab[0]).SetIsotope(0)   # remove one nuclide from carrier
        if el == "H":
            avail = [nb.GetIdx()
                     for nb in oh.GetAtomWithIdx(other).GetNeighbors()
                     if nb.GetSymbol() == "H" and nb.GetIsotope() == 0]
        else:
            avail = [other]
        if not avail:
            continue
        moved.GetAtomWithIdx(avail[0]).SetIsotope(mass)  # place it on the partner
        try:
            if Chem.MolToInchiKey(moved.GetMol()) != key0:
                return True                       # 1,1 vs 1,2 distinct -> cite
        except Exception:
            continue
    return False


def _placement_unambiguous(original: Chem.Mol,
                           complete_sub_omission: bool = False) -> bool:
    """: may the isotope descriptor's position-locant(s) be OMITTED?

    ``complete_sub_omission`` enables the single-carrier
    complete-substitution omission (``(2H3)acetonitrile``). It is True ONLY on the
    single-descriptor whole-molecule enumeration; the multi-position path leaves it
    False, because a name carrying several isotope descriptors must cite ALL locants
    once ANY is required whole-molecule: the per-D-glycine alpha keeps
    ``(2,2-2H2)``), and the masked mol it evaluates cannot see the other descriptors.

    This is a FORMAL rule, not the structural-ambiguity test the prior version
    used ("no OTHER isotopomer can be expressed by moving the locant"). That test
    wrongly omitted on a *valence-blocked* parent: ``1,1,1-trifluoroethane`` has H
    only on C2, so ``(2H1)`` is structurally unique -- yet still requires
    the ``2-`` locant, because C2 is a distinguishable position of the parent
    hydride, not one of a symmetric set. That over-omission dropped the locant on
    the whole Sub-pattern-A class (``…(2-2H1)ethane``, ``2-methoxy(3,4,5,6-3H4)phenol``).

    Per " Omission of locants" (``the Blue Book Blue Book``),
    evaluated per nuclide group within the group's NAMED-UNIT scope
    (:func:`_element_scope` -- "a unit of structure as defined by its
    appropriate enclosing marks"); the deny default is (``:44202``
    "Locants are not omitted when there is a possibility of isomers"):

      * ** clause 1** (``:44182`` "locants are omitted if no locants are
        necessary in unmodified names"): a single label on a symmetric multi-atom
        orbit whose placement is immaterial (:func:`_single_label_is_immaterial`)
        -- any choice gives one molecule, so no locant is necessary
        (``(2H1)benzene``, ``(2H1)methane``); a label that instead DESYMMETRISES
        the orbit (``(2R)-(1-2H1)propan-2-ol``) keeps its locant.
      * **** (``:44190`` "only one atom of a given element"): the scope
        holds exactly one atom of the carrier's element (``(15N)…indole``,
        ``(13C)methyl acetate``, the ``(81Br)`` of a bromo group).
      * **** (``:44196`` "all positions... completely isotopically
        substituted or modified in the same way"): the labelled carriers are one
        complete canonical-rank orbit (size >= 2) of the stripped parent, entirely
        labelled (``(2H6)benzene``, the ``(2H3)methoxy`` methyl).

    Returns True (omit) iff EVERY nuclide group qualifies; otherwise the whole
    descriptor keeps its locants deny-by-default). Symmetry is read from
    the STRIPPED parent (a labelled and an unlabelled equivalent atom share a
    rank); ``AddHs`` normalises H so a partly-labelled methyl's carrier is a
    genuine orbit member, and preserves ``original``'s heavy-atom indices so the
    labels read straight off ``original`` map into the ranked mol. The OPSIN
    round-trip in:func:`_find_best_placement` remains the 0-wrong gate; this
    predicate only decides omit-vs-cite.
    """
    # Labelled atoms of ``original`` grouped by (mass, element).
    groups: Dict[Tuple[int, str], set] = {}
    for atom in original.GetAtoms():
        iso = atom.GetIsotope()
        if iso:
            groups.setdefault((iso, atom.GetSymbol()), set()).add(atom.GetIdx())
    if not groups:
        return True
    try:
        base = Chem.Mol(original)
        for atom in base.GetAtoms():
            atom.SetIsotope(0)
        base_h = Chem.AddHs(base)  # heavy indices preserved; H made explicit
        ranks = list(Chem.CanonicalRankAtoms(base_h, breakTies=False))
        orig_h = Chem.AddHs(Chem.Mol(original))  # isotopes kept, H explicit
    except Exception:
        # Pathological RDKit failure: fall open to the already-round-tripping
        # omitted candidate (0-wrong, breadth-preserving) rather than abstain.
        return True
    for (mass, el), labelled in groups.items():
        carriers = {_isotope_carrier(original.GetAtomWithIdx(i)) for i in labelled}
        car_el = base_h.GetAtomWithIdx(next(iter(carriers))).GetSymbol()
        if any(base_h.GetAtomWithIdx(c).GetSymbol() != car_el for c in carriers):
            return False  # carriers span elements -> multi-position path decides
        scope = set()
        for c in carriers:
            scope |= _element_scope(base_h, c, car_el)
        # (the Blue Book, heading "Locants are not omitted when there is a
        # possibility of isomers"): a locant-free descriptor is unambiguous only
        # if the labelled element has ONE symmetry class of host positions in the
        # scope, OR the labels occupy every host class. When a SECOND, distinct
        # canonical-rank class of the same element exists in scope -- one that
        # could bear the same nuclide COUNT and yield a non-isomorphic molecule --
        # the bare descriptor is ambiguous and the locant MUST be cited, even
        # though the carrier's own orbit is symmetric. Naphthalene has two
        # ring-carbon classes (alpha 1,4,5,8; beta 2,3,6,7); a single 13C on an
        # alpha vs a beta carbon are distinct isotopologues, so "(13C)naphthalene"
        # is ambiguous and the PIN is "(1-13C)naphthalene". The carrier-orbit test
        # below sees only the label's OWN rank, so it passed here; the omitted form
        # then round-tripped only because OPSIN's default placement coincidentally
        # lands on the labelled class -- a spelling error invisible to the RT gate.
        # Host positions: any scope atom for a heavy nuclide; only H-bearing scope
        # atoms for a D/T nuclide (a carbon with no H cannot host it, so it does
        # not widen the ambiguity -- e.g. (2H3)acetonitrile's nitrile carbon stays
        # out, and (2H6)benzene keeps its single ring-CH class). Ranks are read
        # from the STRIPPED parent, so atoms in different classes are genuinely
        # non-equivalent (no automorphism maps one onto the other) and labelling
        # them gives real isomers. This restores the intent of the removed
        # round-trip-alternative probe without its OPSIN cost.
        if el == "H":
            hostable = {c for c in scope
                        if any(nb.GetSymbol() == "H"
                               for nb in base_h.GetAtomWithIdx(c).GetNeighbors())}
        else:
            hostable = set(scope)
        host_ranks = {ranks[c] for c in hostable}
        labelled_ranks = {ranks[c] for c in carriers}
        if len(host_ranks) > 1 and host_ranks != labelled_ranks:
            return False                          # possibility of isomers -> cite
        if len(scope) == 1:                       #: sole atom in scope
            continue
        # (the Blue Book "all positions... completely isotopically
        # substituted"): a SINGLE labelled-H carrier whose EVERY hydrogen is the
        # nuclide, and which is the sole hydrogen-bearing atom of its element in
        # the scope, is a position completely substituted -- the count fills the
        # only site it can, so no locant is necessary (``(2H3)acetonitrile``
        # the Blue Book: the methyl is the sole H-carbon of the two-carbon nitrile
        # scope). A PARTIAL fill (``1,1,1-trifluoro(2-2H1)ethane``, 1 of 3 methyl H)
        # or a SECOND H-carbon in scope (``(2-2H2)propane``: C1/C3 also bear H)
        # leaves a possibility of isomers and still cites. Read from
        # ``orig_h`` (isotopes kept, H explicit) so implicit H count too. Gated to
        # the single-descriptor whole-molecule call (``complete_sub_omission`` +
        # exactly one nuclide group) so the multi-descriptor whole-molecule rule
        # still cites every locant in per-D-glycine and friends.
        if (complete_sub_omission and len(groups) == 1
                and el == "H" and len(carriers) == 1):
            (carrier,) = tuple(carriers)
            scope_h_carbons = {
                c for c in scope
                if any(nb.GetSymbol() == "H"
                       for nb in base_h.GetAtomWithIdx(c).GetNeighbors())}
            carrier_h_all_labelled = all(
                nb.GetIsotope() == mass
                for nb in orig_h.GetAtomWithIdx(carrier).GetNeighbors()
                if nb.GetSymbol() == "H")
            if scope_h_carbons == {carrier} and carrier_h_all_labelled:
                continue
        lab_ranks = {ranks[c] for c in carriers}
        if len(lab_ranks) != 1:                   # >1 orbit -> distinguishable
            return False
        orbit = {c for c in scope if ranks[c] == next(iter(lab_ranks))}
        if len(orbit) < 2:                        # unique singleton position -> cite
            return False
        if len(carriers) == 1:                    # all labels on ONE carrier of a symmetric orbit
            carrier = next(iter(carriers))
            # cl.1: a label whose orbit member is immaterial (moving the
            # WHOLE group to a symmetric partner reproduces the molecule) may omit
            # its locant; a label that DESYMMETRISES the orbit must cite it.
            if not _single_label_is_immaterial(original, mass, el, carrier, orbit):
                return False                      # whole-group move desymmetrises -> cite
            # (the Blue Book): even when the whole-group move collapses, a
            # count>=2 group SPLIT across the orbit (1,1 -> 1,2) can give a DISTINCT
            # isotopomer -- a possibility of isomers that requires the locant. Gated
            # on an actual distinct-InChIKey test so all-collapse cases are not
            # over-cited. ``(2H2)ethane-1,2-diyl`` -> ``(1,1-2H2)ethane-1,2-diyl``.
            if _split_yields_distinct_isomer(original, mass, el, carrier, orbit):
                return False                      # split gives a distinct isomer -> cite
            continue
        if orbit != carriers:                     # partial fill of orbit -> isomers
            return False
        # else: complete orbit, entirely labelled ->
    return True


def _element_singleton_in(mol: Chem.Mol, el: str) -> bool:
    """ (``the Blue Book Blue Book``, heading "Locants are
    omitted when there is only one atom of a given element"): True iff element
    ``el`` occurs EXACTLY ONCE in ``mol`` — the whole-parent scope of a front
    descriptor. With only one atom of the element the nuclide can go nowhere else,
    so its locant is omitted even inside a COMBINED descriptor whose other groups
    cite locants (verbatim PIN ``(2,4-2H2,15N)pyridine``, ``:44194``: the sole ring
    N drops its locant while the two D keep ``2,4-``).

    Read per nuclide GROUP by the mixed-descriptor builder; a group whose element
    is NOT a singleton falls through to the explicit-locant search. H is counted
    over explicit hydrogens so a lone-H molecule is judged correctly."""
    try:
        if el == "H":
            m = Chem.AddHs(Chem.Mol(mol))
            return sum(1 for a in m.GetAtoms() if a.GetSymbol() == "H") == 1
        return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == el) == 1
    except Exception:
        return False


#: A parent hydride opening with a skeletal-replacement / added locant, e.g. the
#: ``3-`` of ``3-oxatricyclo…`` or ``1-`` of ``1-azabicyclo…``, optionally behind
#: leading stereo / configuration prefixes (``(2R,4S)-``, ``rel-``, ``rac-``). The
#: isotope descriptor for the WHOLE parent goes at the front, hyphen-joined to that
#: locant: "inserted before the part … isotopically substituted") --
#: ``(2R,4S)-(2-2H1)-3-oxatricyclo…``, NOT the mid-parent ``…-3-(2-2H1)oxa…`` a
#: pure alpha-offset splice yields. The ``\dH`` indicated-H opening is EXCLUDED --
#: ``_INDICATED_H_PREFIX_RE`` already reuses that hyphen.
_LEADING_STEREO_PREFIX_RE = re.compile(r"^(?:\([^)]*\)-|rel-|rac-|cis-|trans-)*")

# A leading locant is a SKELETAL-REPLACEMENT locant (front-hyphen placement
# applies) only when a nondetachable 'a'-prefix stem follows it -- ``3-oxa…``,
# ``1-aza…``, ``2,4-dioxa…`` -- never a detachable-substituent locant such as
# ``1-methyl-`` or ``4-amino-`` (a review I2). The former regex ``^\d+-(?!\d*H-)``
# matched EVERY leading locant, so a wrong front candidate
# (``(3-2H1)-1-methyl-2-nitro-1H-pyrrole``) was built and RT-tested; it lost only
# because OPSIN 2.9 rejects it, i.e. correctness rested on the grammar rather than
# the rule -- exactly the fragility this fix exists to remove. The 'a'-prefix stem
# set is taken from HW_PREFIXES so it stays in sync with the data (all end in 'a',
# so they cannot false-match the common detachable prefixes oxo-/azido-/thio-).
from ..data.hw_heteroatoms import HW_PREFIXES as _HW_PREFIXES
from ..perception.molcache import atoms_of, inchikey_of
_A_PREFIX_ALT = "|".join(sorted(set(_HW_PREFIXES.values()), key=len, reverse=True))
_LEADING_SKELETAL_LOCANT_RE = re.compile(
    r"^\d+(?:,\d+)*-(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?"
    r"(?:" + _A_PREFIX_ALT + r")")

#: The same skeletal-replacement locant opening a SUBSTITUENT's ring name inside
#: its enclosing marks, e.g. the ``9,24-`` of ``[9,24-dioxahexacyclo[...]...-yl]``.
#: The part isotopically substituted is that ring, and its name begins with the
#: locant set, so the descriptor goes right after the opening mark, hyphen-joined
#: to the locant as at the front of a parent (``_LEADING_SKELETAL_LOCANT_RE``):
#: ``[(3,4,...-2H11)-9,24-dioxahexacyclo[...]...-yl]``. No other rule offered that
#: slot, so a labelled heteroatom von Baeyer substituent failed closed (PubChem 1M:
#: a deuterated dibenzofuran-fused OLED host). Used by ``_insertion_offsets``
#: (the slot is the index after the mark); every candidate stays OPSIN-RT gated.
_BRACKET_SKELETAL_LOCANT_RE = re.compile(
    r"[(\[{](?=\d+(?:,\d+)*-(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?"
    r"(?:" + _A_PREFIX_ALT + r"))")


def _front_hyphen_candidate(skel: str, desc: str) -> Optional[Tuple[int, str]]:
    """``(insertion_point, candidate)`` placing ``desc`` at the front of the
    parent hydride, hyphen-joined, when ``skel`` opens (after any leading stereo
    prefix) with a skeletal-replacement locant; else ``None``. See
    ``_LEADING_SKELETAL_LOCANT_RE``. The candidate is OPSIN-RT gated by the
    caller, so a wrong guess is discarded, never shipped."""
    pre = _LEADING_STEREO_PREFIX_RE.match(skel).group(0)
    rest = skel[len(pre):]
    if not _LEADING_SKELETAL_LOCANT_RE.match(rest):
        return None
    p = len(pre)
    return p, skel[:p] + desc + "-" + skel[p:]


def _distinct_hydroxyl_o_classes(original: Chem.Mol) -> int:
    """Number of DISTINGUISHABLE hydroxyl oxygens in ``original``'s parent
    skeleton -- the "possibility of isomers" quantity for an ``O-H ->
    O-D`` hydroxyl label.

    A hydroxyl oxygen is a single-bonded, one-heavy-neighbour (a carbon) oxygen
    (an ``-OH``; the carbonyl ``=O`` is excluded by the double bond). Two such
    oxygens are the SAME position -- deuterating either yields ONE molecule -- iff
    they share a constitutional symmetry class. The count is taken on the
    ISOTOPE-STRIPPED, hydrogen-normalised skeleton: isotopes are cleared and the
    mol is round-tripped through canonical SMILES so the labelled O's now-plain
    explicit H collapses to implicit and it ranks identically to an unlabelled
    ``-OH`` (without this the residual explicit H would split an otherwise
    symmetric diol into two classes). Returns 0 when there is no hydroxyl O.

    ``ethane-1,2-diol`` -> 1 (the two ``-OH`` are one orbit, no isomer, bare
    ``O``); ``propane-1,2-diol`` -> 2 (primary vs secondary ``-OH`` are distinct,
    the numeral is required); a mono-ol or an ``-OD`` carboxylic acid -> 1.
    """
    skel = Chem.Mol(original)
    for a in skel.GetAtoms():
        a.SetIsotope(0)
    skel = Chem.MolFromSmiles(Chem.MolToSmiles(skel))
    if skel is None:
        return 0
    ranks = list(Chem.CanonicalRankAtoms(skel, breakTies=False))
    classes = set()
    for a in skel.GetAtoms():
        if a.GetSymbol() != "O":
            continue
        heavy = [n for n in a.GetNeighbors() if n.GetSymbol() != "H"]
        if len(heavy) != 1 or heavy[0].GetSymbol() != "C":
            continue
        bond = skel.GetBondBetweenAtoms(a.GetIdx(), heavy[0].GetIdx())
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        classes.add(ranks[a.GetIdx()])
    return len(classes)


#: A parent that cites SEVERAL nitrogens of its own suffix groups: a multiplied
#: amine / imine / amide-type suffix ('propane-1,2-diamine', 'benzene-1,3-
#: dicarboxamide'), or a name that already carries a numbered nitrogen locant
#: ('N1-methylpropane-1,2-diamine'). (the Blue Book): "Superscript
#: arabic numbers, which are the locants of the parent structure, are used to
#: differentiate the nitrogen atoms of di- and polyamines, di- and polyimines, di-
#: and polyamides, except for geminal amines, imines, and amides."
_PARENT_NUMBERS_ITS_NITROGENS_RE = re.compile(
    r"(?:di|tri|tetra|tetr|penta|hexa|hepta|octa|bis\(|tris\()"
    r"(?:amine|imine|amide|carboxamide|sulfonamide|sulfinamide|carbothioamide"
    r"|carboximidamide|imidamide|amidine|hydrazonamide|amidrazone)"
    r"|(?<![A-Za-z])N\d")


def _parent_numbers_its_nitrogens(skeleton: Optional[str]) -> bool:
    """: does the skeleton's parent cite more than one nitrogen of its own
    characteristic groups (so each needs its numbered ``N1`` / ``N2`` locant)?

    ``None`` (no skeleton known) keeps the old answer, True, for a caller that
    has not passed one."""
    if skeleton is None:
        return True
    return _PARENT_NUMBERS_ITS_NITROGENS_RE.search(skeleton) is not None


def _letter_locant_candidates(original: Chem.Mol, label_map: Dict[int, int],
                              n_pos: int,
                              skeleton: Optional[str] = None) -> List[str]:
    """Letter locants the placement search should offer in addition to the integer
    locants -- the amide/amine nitrogen ``N`` and the carboxyl-oxygen
    ``O``.

    A labelled H (D/T) on a non-ring nitrogen sits on a position the parent's
    suffix cites by the letter locant ``N`` (``acetamide`` -> ``(N-2H1)acetamide``),
    not by a ring/chain number, so the integer-only search cannot place it and the
    row abstains. Offering ``N`` is safe: every candidate is still OPSIN-RT gated,
    so a spurious offer (a D on a nitrogen the parent does NOT letter-locant) is
    discarded, never shipped.

     (``the Blue Book Blue Book``, "Italicized nuclide symbols and/or
    italic capital letters are used to distinguish between different nuclides of the
    same element"): the two oxygens of a carboxylic acid -- the carbonyl ``=O`` and
    the hydroxyl ``-OH`` -- are the same element in one suffix, so an ``18O`` on the
    HYDROXYL oxygen (witness ``(18O-2H, 18O)acetic acid (PIN)``, "the O is a locant")
    needs the italic ``O`` locant to distinguish it from the carbonyl one. The
    integer search cannot place it (``(1-18O)…`` does not parse) and the bare
    front descriptor ``(18O)acetic acid`` denotes the CARBONYL oxygen (a different
    isotopomer, discarded by the round-trip gate), so the row abstained. Offer the
    ``O`` letter locant ONLY when a labelled O is the single-bonded, degree-1
    hydroxyl oxygen of a carboxyl carbon; every candidate stays OPSIN-RT gated so a
    spurious offer is discarded, never shipped. The descriptor is then ``(O-18O)``,
    placed immediately before the parent stem (``(O-18O)acetic acid``,
    ``4-methyl-2-propyl(O-18O)pentanoic acid``) by the enumeration's insertion
    offsets. (Isotope path only; no non-isotope name is reached here.)

     / (Task 7b): when the parent carries MORE THAN ONE atom of
    that element, the bare letter locant ``N`` is ambiguous (the diamine
    ``propane-1,2-diamine`` has two nitrogens) and the SPECIFIC numbered locant the
    parent assigns must be cited -- ``N1`` for the D on the nitrogen borne by C1,
    exactly as the non-isotope substituent path spells ``N1-methylpropane-1,2-
    diamine``: a numeral distinguishes among several like heteroatoms;
    a single one keeps the bare italic letter,. The OPSIN round-trip
    gate is BLIND to ``N`` vs ``N1`` (both denote the same nuclide position, so both
    reproduce the structure), so the numeral is NOT resolved by the gate but chosen
    BY CONSTRUCTION here: for a multi-nitrogen parent we drop the bare ``N`` and
    offer the numbered forms ``N1..Nk`` ascending. The gate is NOT blind to WHICH
    numeral -- ``N1`` and ``N2`` denote DIFFERENT nitrogens, hence different
    structures -- so it discards a wrong numeral (``N2`` on an unsymmetrical
    diamine) and keeps the correct one; the placement search's lowest-locant-first
    ``break`` then selects the lowest round-tripping numeral (the parent's own
    numbering, which the module treats as opaque -- see the header note). The
    numeral is the labelled heteroatom's attachment-atom locant, bounded above by
    ``n_pos`` (the parent's heavy-atom count). The bare ``N`` is appended LAST as a
    no-regression fallback for a parent whose second nitrogen is NOT itself cited by
    a letter locant (e.g. an aromatic ring N alongside a lone amide N): no numbered
    form round-trips there, so ``N`` still wins. Every candidate stays OPSIN-RT
    gated (defence in depth); correctness of the numeral-vs-bare choice comes from
    the atom count, and of the numeral value from the RT-distinguishable molecules.

    Two corrections (PIN spelling, PubChem 1M job): the numbered forms were offered
    whenever the MOLECULE had several nitrogens, so a ring nitrogen of a
    substituent made the lone amide nitrogen of an acetamide '(N1,2,2-2H3)
    acetamide'. (the Blue Book, "In a name consisting of one word,
    the isotopic descriptor is placed before the name, with an appropriate
    locant") writes that nitrogen '(N-2H1)acetamide (PIN)' (:43828), and
    (:7739) numbers the nitrogens of di- and polyamines, -imines and -amides only.
    So the numbered forms are offered only when the SKELETON's parent cites several
    nitrogens of its own groups (``_parent_numbers_its_nitrogens``); otherwise the
    bare ``N`` alone. And a RING nitrogen is a skeletal atom of the ring, located
    by its ring numeral, never by an italic letter: the D on N-3 of an
    imidazolidine is '(3-2H)imidazolidine', not '(N3-2H)imidazolidine', so no
    letter locant is offered for it (the integer sweep places it).
    """
    n_count = sum(1 for a in original.GetAtoms() if a.GetSymbol() == "N")
    numbered = n_count > 1 and _parent_numbers_its_nitrogens(skeleton)
    out: List[str] = []
    for idx in label_map:
        atom = original.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "H":
            continue
        nbrs = atom.GetNeighbors()
        if (nbrs and nbrs[0].GetSymbol() == "N" and not nbrs[0].GetIsAromatic()
                and not nbrs[0].IsInRing()):
            if not numbered:
                #: a single nitrogen keeps the bare italic ``N``.
                if "N" not in out:
                    out.append("N")
            else:
                #: several nitrogens -> cite the numbered locant. Offer
                # every numeral the parent could assign (1..n_pos), ascending, so
                # the lowest round-tripping one wins; bare ``N`` last as the
                # no-regression fallback (see docstring).
                for i in range(1, n_pos + 1):
                    tag = f"N{i}"
                    if tag not in out:
                        out.append(tag)
                if "N" not in out:
                    out.append("N")
    # italic ``O`` locant: a labelled O on the HYDROXYL oxygen of a
    # carboxyl group (the single-bonded, degree-1 ``-OH``, NOT the ``=O``). It is
    # the same element as the carbonyl O in one suffix, so the bare descriptor is
    # ambiguous and the italic ``O`` locant is required (see docstring). RT-gated.
    for idx in label_map:
        atom = original.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "O":
            continue
        nbrs = atom.GetNeighbors()
        if len(nbrs) != 1:
            continue
        carbon = nbrs[0]
        bond = original.GetBondBetweenAtoms(idx, carbon.GetIdx())
        if (bond is not None
                and bond.GetBondType() == Chem.BondType.SINGLE
                and _is_carboxyl_carbon(original, carbon)):
            if "O" not in out:
                out.append("O")
    # italic ``O`` locant for a labelled HYDROGEN (D/T) on a HYDROXYL
    # oxygen -- the ``O-H -> O-D`` case of a mono-ol / di-ol / poly-ol (and an
    # ``-OD`` carboxylic acid). The labelled atom is the H (mass 2/3), whose sole
    # neighbour is a hydroxyl-type O (single bond, one heavy neighbour = C). The
    # bare front descriptor ``(2H)ethane-1,2-diol`` does NOT parse in OPSIN 2.9.0,
    # and an integer locant ``(1-2H)`` places the D on a CARBON (a different
    # isotopomer, discarded by the round-trip gate), so the position can only be
    # cited by the italic ``O`` letter locant (the Blue Book ``(O-2H, 18O)acetic acid``
    # -- "the O is a locant"). OPSIN parses ``(O-2H)`` / ``(O1-2H)`` to the O-D.
    #
    # Bare ``O`` vs numbered ``O1..Ok`` is chosen BY CONSTRUCTION, not by the RT
    # gate: OPSIN resolves a bare ``O`` to the LOWEST-locant oxygen (O1), so on an
    # asymmetric diol ``(O-2H)propane-1,2-diol`` round-trips to the O1 isotopomer
    # and the gate cannot tell it from ``(O1-2H)`` -- exactly the ``N`` vs ``N1``
    # situation above. (the Blue Book, "locants are not omitted when there is
    # a possibility of isomers"): when the hydroxyl oxygens are DISTINGUISHABLE
    # (propane-1,2-diol: primary vs secondary ``-OH``, 2 symmetry classes) the
    # numeral is required -> offer ``O1..O(n_pos)`` ascending so the lowest
    # round-tripping numeral wins (the parent's own numbering, treated as opaque),
    # bare ``O`` LAST as a no-regression fallback. When the hydroxyl oxygens are
    # EQUIVALENT or there is a single one (ethane-1,2-diol, a mono-ol, an ``-OD``
    # acid) there is no isomer possibility /.4) -> bare ``O``. Every
    # candidate stays OPSIN-RT gated (defence in depth); the numeral's VALUE is
    # then RT-resolved (``O1`` vs ``O2`` denote different oxygens).
    labelled_hydroxyl_h = False
    for idx in label_map:
        atom = original.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "H":
            continue
        nbrs = atom.GetNeighbors()
        if len(nbrs) != 1 or nbrs[0].GetSymbol() != "O":
            continue
        o_atom = nbrs[0]
        heavy = [n for n in o_atom.GetNeighbors() if n.GetSymbol() != "H"]
        if len(heavy) != 1 or heavy[0].GetSymbol() != "C":
            continue
        bond = original.GetBondBetweenAtoms(o_atom.GetIdx(), heavy[0].GetIdx())
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        labelled_hydroxyl_h = True
    if labelled_hydroxyl_h:
        if _distinct_hydroxyl_o_classes(original) <= 1:
            if "O" not in out:
                out.append("O")
        else:
            for i in range(1, n_pos + 1):
                tag = f"O{i}"
                if tag not in out:
                    out.append(tag)
            if "O" not in out:
                out.append("O")
    return out


#: A simple, unbranched ALKYL substituent prefix at the START of a name tail
#: (the ``methyl`` of ``2-methylpropanal``, where it is concatenated DIRECTLY with
#: the parent stem ``propanal``). Used by the parent-chain promotion
#: below to recognise a symmetric alkyl BRANCH whose isotope belongs in the
#: principal chain. Restricted to unbranched C1-C12 stems on purpose: a multiplied
#: (``di``/``tri``) or complex substituent is handled by the de-multiplication
#: paths, and the promotion is RT-gated + fail-closed regardless -- so a surplus
#: match can only fail its round-trip, never ship a wrong name. No trailing
#: boundary is asserted: the parent stem follows the prefix immediately
#: (``methylpropanal``), so a ``(?![a-z])`` lookahead would wrongly reject it.
_ALKYL_SUBSTITUENT_PREFIX_RE = re.compile(
    r"(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl"
    r"|undecyl|dodecyl)"
)


def _promote_isotope_to_parent_chain(
    skel: str, keys, original: Chem.Mol, n_pos: int,
    current_best: str, stereo_blind: bool = False,
) -> Optional[str]:
    """ criterion (k) (``the Blue Book Blue Book``, heading
    " The senior ring, ring system, or principal chain that has one or
    more isotopically modified atoms [criterion (k) in ]"; the decisive
    sentence ``:21477``): "The senior parent structure contains the
    greater number of isotopically modified atoms or groups."

    The skeleton is named from the isotope-STRIPPED molecule, so chain selection
    cannot see the label. For a molecule with a SYMMETRIC alkyl branch
    (isobutyraldehyde ``[2H]C([2H])([2H])C(C=O)C``: two equivalent methyls, one =
    CD3) the stripped parent is a genuine tie and the descriptor lands on the
    substituent -- ``2-(1,1,1-2H3)methylpropanal`` -- when criterion (k) requires
    the isotope-bearing methyl to be the CHAIN terminus: ``2-methyl(3,3,3-2H3)propanal``.

    When the winning SINGLE-nuclide descriptor sits immediately before a simple
    ALKYL substituent prefix, re-place the SAME nuclide at the parent-stem slot
    just after that substituent, lowest-locant-first, and prefer it iff it
    round-trips. A parent placement round-trips ONLY when its atom is symmetry-
    equivalent to the substituent atom (identical InChIKey) -- i.e. exactly when
    every higher criterion ties by graph automorphism and criterion (k)
    governs. So an asymmetric (wrong) promotion cannot round-trip and fails closed
    to ``current_best`` (0-wrong is preserved). Isotope-path ONLY: this lives in the
    decorator, is reached only for a labelled molecule, and touches no non-isotope
    chain-selection code.
    """
    if len(keys) != 1:
        return None
    # Recover the descriptor's insertion offset in ``skel`` by a common-affix diff
    # against ``current_best`` (cheap; no OPSIN). Any mark escalation / hyphenation
    # that makes the splice non-recoverable bails -> fail closed to current_best.
    pre = 0
    while (pre < len(skel) and pre < len(current_best)
           and skel[pre] == current_best[pre]):
        pre += 1
    suf = 0
    while (suf < len(skel) - pre and suf < len(current_best) - pre
           and skel[len(skel) - 1 - suf]
           == current_best[len(current_best) - 1 - suf]):
        suf += 1
    inserted = current_best[pre:len(current_best) - suf]
    if not inserted or skel[:pre] + inserted + skel[pre:] != current_best:
        return None
    # The descriptor must sit immediately before a simple alkyl substituent.
    m = _ALKYL_SUBSTITUENT_PREFIX_RE.match(skel[pre:])
    if not m:
        return None
    parent_off = pre + m.end()
    # Re-place the SAME nuclide at the parent-stem slot, exactly as
    # _find_best_placement builds a descriptor lowest-locant-first: the
    # locant-free form first, then integer locants ascending). Every candidate is
    # OPSIN-RT gated, so only a symmetry-equivalent (criterion-k tie) parent atom
    # can win; anything else fails closed.
    for locant in [None] + list(range(1, n_pos + 1)):
        groups_desc = [(locant, mass, el, count, maxpos)
                       for (mass, el), count, maxpos in keys]
        desc_pref = format_isotope_descriptor(groups_desc)
        desc_force = format_isotope_descriptor(groups_desc, force_show=True)
        variants = [desc_pref] + ([desc_force] if desc_force != desc_pref else [])
        for desc in variants:
            candidate = skel[:parent_off] + desc + skel[parent_off:]
            if _isotope_round_trips(candidate, original, stereo_blind=stereo_blind):
                #: only OMIT the locant when the position is genuinely
                # unambiguous (or the unit is completely labelled,;
                # otherwise fall through to the explicit-locant form.
                if (locant is None and not _placement_unambiguous(original, True)
                        and not _unit_completely_labelled(
                            skel, parent_off, keys, original)):
                    continue
                if desc != desc_pref:
                    # The always-show subscript is the parse fallback, not the
                    # preferred spelling (see _find_best_placement).
                    from ..metrics.provenance import record_non_pin_fragment
                    record_non_pin_fragment(candidate)
                    record_non_pin_fragment(_descriptor_fragment(candidate, desc))
                return candidate
    return None


def _nitrogen_hydrogen_needs_locant(original: Chem.Mol, skel: str, off: int) -> bool:
    """ (the Blue Book, "Locants are not omitted when there is a
    possibility of isomers") for a hydrogen nuclide on a NITROGEN of the parent: a
    locant-free descriptor at the parent level (enclosing-mark depth 0) is ambiguous
    when the parent unit of that nitrogen -- its ring system for a ring nitrogen,
    the nitrogen and the carbon chain or ring it is bonded to for an amine or amide
    nitrogen -- has another hydrogen-bearing position. '(N-2H1)acetamide (PIN)'
    ,:43828), '(N-2H2)aniline (PIN)' (:43830); the ring nitrogen takes its
    ring numeral ('(1-2H)piperidine'). OPSIN puts a locant-free D on an N-H by
    default, so the round trip cannot see the ambiguity.

    True only when every label is a hydrogen isotope on one nitrogen carrier and
    the slot does not open a substituent prefix: a descriptor before 'amino'
    ('4-(2H2)aminobenzoic acid') labels that prefix, whose one position needs no
    locant."""
    if _bracket_depth(skel, off) != 0:
        return False
    from ..assembly.naming_utils import _is_recognised_prefix_component
    run = re.match(r"[a-z]+", skel[off:])
    if run and any(_is_recognised_prefix_component(run.group(0)[:k])
                   for k in range(3, len(run.group(0)))):
        return False
    carriers = set()
    for atom in original.GetAtoms():
        if atom.GetIsotope():
            if atom.GetSymbol() != "H":
                return False
            carriers.add(_isotope_carrier(atom))
    if len(carriers) != 1:
        return False
    (carrier,) = tuple(carriers)
    try:
        base = Chem.Mol(original)
        for atom in base.GetAtoms():
            atom.SetIsotope(0)
        base_h = Chem.AddHs(base)
        ranks = list(Chem.CanonicalRankAtoms(base_h, breakTies=False))
    except Exception:
        return False
    x = base_h.GetAtomWithIdx(carrier)
    if x.GetSymbol() != "N":
        return False
    ri = base_h.GetRingInfo()
    unit = set()
    if x.IsInRing():
        rings = [set(r) for r in ri.AtomRings()]
        system = set()
        frontier = [r for r in rings if carrier in r]
        while frontier:
            r = frontier.pop()
            if r <= system:
                continue
            system |= r
            frontier.extend(q for q in rings if len(q & system) >= 2 and not q <= system)
        unit = system
    else:
        unit = {carrier}
        for nb in x.GetNeighbors():
            if nb.GetSymbol() == "C":
                unit |= _element_scope(base_h, nb.GetIdx(), "C")
    hosts = {i for i in unit
             if any(nb.GetSymbol() == "H" for nb in base_h.GetAtomWithIdx(i).GetNeighbors())}
    return any(ranks[i] != ranks[carrier] for i in hosts)


def _locant_names_the_labelled_position(skel: str, off: int, locant, keys,
                                        original: Chem.Mol) -> bool:
    """Does the shared locant of a hydrogen-isotope descriptor with count >= 2
    ('(1,1,1-2H3)') name the position that carries the labels?

    A descriptor cites one locant per nuclide at that position,
    the Blue Book, "all locants are placed before the nuclide that is
    multiplied"), so the position must hold that many hydrogens. OPSIN 2.9.0 reads
    '(1,1,1-2H3)ethylbenzene' leniently as the CD3 compound although C1 of ethyl,
    the atom with the free valence (1),:15813), bears two hydrogens only;
    the PIN is '(2,2,2-2H3)ethylbenzene'. Checked by reading the same slot with ONE
    nuclide at that locant: it must label an atom equivalent to the input's
    labelled carrier. True when the check does not apply or cannot be made."""
    if locant is None or len(keys) != 1:
        return True
    (mass, el), count, maxpos = keys[0]
    if el != "H" or count < 2:
        return True
    one = format_isotope_descriptor([(locant, mass, el, 1, maxpos)])
    smi = _opsin_parse(skel[:off] + one + skel[off:])
    if not smi:
        return True
    parsed = Chem.MolFromSmiles(smi)
    if parsed is None:
        return True
    try:
        ref = Chem.RWMol(Chem.AddHs(original))
        kept = False
        for atom in ref.GetAtoms():
            if atom.GetIsotope() == mass and atom.GetSymbol() == el:
                if kept:
                    atom.SetIsotope(0)
                kept = True
        want = Chem.MolToSmiles(Chem.RemoveHs(ref.GetMol()))
        got = Chem.MolToSmiles(parsed)
    except Exception:
        return True
    return want == got


def _whole_compound_labelled(keys, original: Chem.Mol) -> bool:
    """Is every position of each nuclide's element in the compound labelled with
    that nuclide: every hydrogen atom for a hydrogen nuclide, every atom of the
    element for a heavier one?"""
    try:
        mol = Chem.AddHs(original)
    except Exception:
        return False
    for (mass, el), _count, _maxpos in keys:
        atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == el]
        if not atoms or any(a.GetIsotope() != mass for a in atoms):
            return False
    return True


def _slot_opens_substituent_prefix(skel: str, off: int) -> bool:
    """Does a descriptor spliced at ``off`` label a substituent group: inside a
    substituent's enclosing marks, or in front of a substituent prefix that the
    shared prefix vocabulary recognises and that is followed by more name
    ('(2H5)ethylbenzene', '2-(2H5)phenyl-1-benzofuran', '(2H5)ethyl acetate')?
    A component that only begins a stem ('oxo' in 'oxolane') is not a prefix, as
    in:func:`_slot_precedes_substituent_locant`."""
    if off < 0 or off >= len(skel):
        return False
    if _bracket_depth(skel, off) > 0:
        return True
    from ..assembly.naming_utils import _is_recognised_prefix_component
    text = skel[off:]
    for k in range(len(text), 2, -1):
        head = text[:k]
        if not head.isalpha() or not _is_recognised_prefix_component(head):
            continue
        nxt = text[k:k + 1]
        if nxt and nxt in "-([{ ":
            return True
        # an 'yl' prefix, or an 'oxy' prefix, running straight into the parent
        # ('(2H5)ethylbenzene', '(2H5)ethoxybenzene')
        if nxt.isalpha() and head.endswith(("yl", "ylidene", "ylidyne", "oxy")):
            return True
    return False


def _scope_completely_labelled(original: Chem.Mol, keys) -> bool:
    """Every hydrogen position of the element-connected unit that carries the
    labels (:func:`_element_scope`) is labelled, for a hydrogen nuclide; every
    atom of the unit, for a heavier one."""
    if len(keys) != 1:
        return False
    (mass, el), _count, _maxpos = keys[0]
    try:
        mol = Chem.AddHs(original)
    except Exception:
        return False
    labelled = [a for a in mol.GetAtoms() if a.GetIsotope() == mass and a.GetSymbol() == el]
    if not labelled:
        return False
    carriers = {_isotope_carrier(a) for a in labelled}
    scope = set()
    for c in carriers:
        scope |= _element_scope(mol, c, mol.GetAtomWithIdx(c).GetSymbol())
    if el == "H":
        hs = [nb for i in scope for nb in mol.GetAtomWithIdx(i).GetNeighbors()
              if nb.GetSymbol() == "H"]
        return bool(hs) and all(h.GetIsotope() == mass for h in hs)
    return all(mol.GetAtomWithIdx(i).GetIsotope() == mass for i in scope)


def _label_keys(original: Chem.Mol):
    """``[((mass, element), count, max_at_pos)]`` of every label of ``original``."""
    by_key: Dict[Tuple[int, str], int] = {}
    maxpos: Dict[Tuple[int, str], int] = {}
    for atom in original.GetAtoms():
        if atom.GetIsotope():
            k = (atom.GetIsotope(), atom.GetSymbol())
            by_key[k] = by_key.get(k, 0) + 1
            maxpos[k] = max(maxpos.get(k, 0), _max_atoms_at_position(atom))
    return sorted(((k, by_key[k], maxpos[k]) for k in by_key),
                  key=lambda kv: (kv[0][1], kv[0][0]))


def _complete_unit_spelling_unreadable(skel: str, off: int, keys,
                                        original: Chem.Mol) -> bool:
    """The locant-free spelling gives a completely labelled substituent
    group at ``off`` is not read by OPSIN 2.9.0 as the input's constitution
    ('(2H7)propylbenzene' reads as the propan-2-yl isomer), so the locanted
    spelling that ships is not the PIN spelling. Graph check of the unit
    (:func:`_scope_completely_labelled`), then the parse of the locant-free form."""
    if not _slot_opens_substituent_prefix(skel, off):
        return False
    if not _scope_completely_labelled(original, keys):
        return False
    try:
        want = Chem.MolToInchiKey(original).split('-')[0]
    except Exception:
        return False
    groups = [(None, mass, el, count, maxpos) for (mass, el), count, maxpos in keys]
    smi = _opsin_parse(skel[:off] + format_isotope_descriptor(groups) + skel[off:])
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return True
    try:
        return Chem.MolToInchiKey(mol).split('-')[0] != want
    except Exception:
        return True


def _unit_completely_labelled(skel: str, off: int, keys,
                              original: Chem.Mol) -> bool:
    """: does the unit that a locant-free descriptor spliced at
    ``off`` modifies hold no further position for any of its nuclides?

    "Locants are omitted in compounds or substituent groups in which all
    positions are completely isotopically substituted or modified in the same
    way.", the Blue Book; '(2H6)benzene (PIN)',:44200.)
    :func:`_placement_unambiguous` reads symmetry classes of the graph, so a
    unit whose positions fall in SEVERAL classes and are all labelled
    ('(2H5)ethyl': CH2 and CH3) fails it although nothing is left to place.

    The unit is the one OPSIN reads the descriptor onto, so OPSIN is asked: the
    same slot with ONE more nuclide of a kind ('(2H6)' where the candidate has
    '(2H5)') must not describe the input's constitution. When it does, the unit
    had a free position, the candidate fills it only partly, and a possibility
    of isomers remains ("Locants are not omitted when there is a possibility of
    isomers",,:44202): the caller then cites the locants.
    '(2H5)ethylbenzene' passes (ethyl has five hydrogen atoms); a phenyl-d5 of
    '1,1'-biphenyl' or of 'benzyl' does not (ten and seven). A hydrogen on a
    chalcogen atom counts as a position, as in '(2-2H1)acetic acid (PIN)'
    (:43804), so a fully labelled ethyl of ethanol keeps its locants.

    The rule names two kinds of unit, "compounds or substituent groups": the
    whole compound, when every position of the nuclide's element in it is
    labelled ('(2H8)propane', '(2H5)glycine'), or a substituent group, when the
    descriptor opens a substituent prefix (:func:`_slot_opens_substituent_prefix`).
    The parent of a substituted compound is neither: a fully labelled acetamide
    parent keeps its locants, '2-(pyridin-4-yl)-N-phenyl(N,2,2-2H3)acetamide',
    as the unmodified analogue cites 'N,2,2-trichloro' "as in
    unmodified names";,:3007, omits locants only in compounds or
    substituent groups that are completely substituted).

    A control parse of the candidate's own count, with its subscript shown,
    must succeed, so an OPSIN refusal of the spelling itself is never read as
    a full unit. This only licenses omission for a candidate that already
    reads back to the input (the caller's round trip).
    """
    if not keys:
        return False
    if not (_slot_opens_substituent_prefix(skel, off)
            or _whole_compound_labelled(keys, original)):
        return False
    try:
        want = Chem.MolToInchiKey(original).split('-')[0]
    except Exception:
        return False
    if not want:
        return False

    def _constitution(name: str) -> Optional[str]:
        smi = _opsin_parse(name)
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            return None
        try:
            return Chem.MolToInchiKey(mol).split('-')[0] or None
        except Exception:
            return None

    control = [(None, mass, el, count, maxpos)
               for (mass, el), count, maxpos in keys]
    if _constitution(skel[:off] + format_isotope_descriptor(control, force_show=True)
                     + skel[off:]) != want:
        return False
    for i in range(len(keys)):
        bumped = [(None, mass, el, count + (1 if j == i else 0), maxpos)
                  for j, ((mass, el), count, maxpos) in enumerate(keys)]
        name = (skel[:off] + format_isotope_descriptor(bumped, force_show=True)
                + skel[off:])
        if _constitution(name) == want:
            return False            # room for one more -> isomers possible
    return True


def _find_best_placement(
    skel: str, keys, original: Chem.Mol, n_pos: int,
    allow_front_hyphen: bool = False,
    extra_locants: Optional[List] = None,
    stereo_blind: bool = False,
    complete_sub_omission: bool = False,
    multi_descriptor: bool = False,
) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[int]]:
    """Best ``(candidate, offset, descriptor, loc_rank)`` for ONE skeleton
    spelling and ONE ``(mass, element) -> count`` key list, OPSIN-RT gated
    against ``original``; ``(None, None, None, None)`` if nothing round-trips.

    Extracted (A3) from what was a same-shaped closure named ``_enumerate``
    inside ``_decorate_isotopic_name_inner`` so BOTH the single-attachment
    path there (unchanged behaviour -- it now calls this with its own
    ``keys``/``original``) and the multi-attachment placement below
    (:func:`_decorate_multi_position`, called once per attachment GROUP with a
    masked ``original``) share ONE oracle-driven search, instead of two
    independently-maintained copies that could drift apart.

    ``loc_rank`` is -1 when the winning descriptor needs NO locant
    lowest-locant-first: the locant-free form is tried before any integer),
    else the locant that turned out to be required.

    ``allow_front_hyphen`` enables the front-of-parent hyphen-joined placement
    (:func:`_front_hyphen_candidate`) for a LOCANTED descriptor on a parent that
    opens with a skeletal-replacement locant. It is OFF for the multi-attachment
    caller, whose ``(off, desc)`` pair is re-spliced verbatim and cannot express
    the added hyphen.

    ``multi_descriptor`` is True for the multi-attachment caller. There
    ``original`` is a masked copy carrying one group's labels only, so the
     unit licence (:func:`_unit_completely_labelled`) cannot see the
    other descriptors of the name, and a name with several descriptors cites
    every locant once any is required, the Blue Book); the
    licence is not consulted then.
    """
    offs_raw = [0] + _insertion_offsets(skel)
    seen: set = set()
    offs: List[int] = []
    for x in offs_raw:
        if x not in seen:
            seen.add(x)
            offs.append(x)
    won = []
    # letter locants (e.g. amide ``N``) are tried after the locant-free
    # form and before the integer locants -- an amide/amine D belongs on ``N``,
    # which no integer locant can express. RT-gated like every other candidate.
    #
    # A locant here is ONE position shared by every labelled atom of a nuclide
    # (``format_isotope_descriptor``: '(2,2,2-2H3)', three D at position 2). A
    # nuclide with more labelled atoms than one position can bear (count >
    # max_at_pos: four D on four aromatic CH, two 13C) has no such position, so
    # every locanted candidate fails its round trip; only the locant-free form is
    # tried then, and the other decorators (distinct positions, several parts)
    # place those labels. The outcome is the one the full sweep reached; the
    # sweep itself cost up to (positions x offsets x 2) OPSIN parses (PubChem 1M:
    # a D4 mixture spent 58,000 of them, 60 s, on candidates that cannot parse
    # to the drawn structure).
    _shared_position_possible = all(
        count <= maxpos for (_mass, _el), count, maxpos in keys)
    _locants = [None] + ((list(extra_locants or []) + list(range(1, n_pos + 1)))
                         if _shared_position_possible else [])
    # Locant-free candidates for a D on a parent nitrogen with other hydrogen-
    # bearing positions (``_nitrogen_hydrogen_needs_locant``) and locanted ones
    # whose locant does not name the labelled position
    # (``_locant_names_the_labelled_position``): kept aside and used only when no
    # other candidate round-trips (then labelled below the PIN), so no name is
    # lost.
    held = []
    for locant in _locants:
        groups_desc = [(locant, mass, el, count, maxpos)
                       for (mass, el), count, maxpos in keys]
        p4542 = _p4542_p4543_key(groups_desc)
        loc_rank = -1 if locant is None else locant
        # Subscript axis (FIX-A): the BB-preferred omitted form (sub_rank 0) is
        # tried first; the forced always-show form (sub_rank 1) is a parseability
        # fallback so an OPSIN-unparseable omission degrades to the right molecule
        # at a non-preferred spelling rather than abstaining. Identical strings
        # (the predicate already keeps the subscript) collapse to one variant.
        desc_pref = format_isotope_descriptor(groups_desc)
        desc_force = format_isotope_descriptor(groups_desc, force_show=True)
        variants = [(0, desc_pref)]
        if desc_force != desc_pref:
            variants.append((1, desc_force))
        for sub_rank, desc in variants:
            for off in offs:
                candidate = skel[:off] + desc + skel[off:]
                if _isotope_round_trips(candidate, original, stereo_blind=stereo_blind):
                    #: a locant-free descriptor may only be OMITTED when
                    # its position is genuinely unambiguous. A round-trip alone is
                    # not proof -- OPSIN's default placement can coincidentally match
                    # the true atom on an asymmetric parent. Verify uniqueness at this
                    # slot before accepting the omission; otherwise fall through to
                    # the explicit-locant search (and the forced_locant_scope
                    # re-render below), which produces the required locanted form.
                    # applies at ANY enclosing-mark depth: a locant-free
                    # descriptor inside a substituent bracket
                    # ('[(13C6)naphthalen-2-yl]…' -- 6 of 10 ring C, no locants) is
                    # just as ambiguous as one on the parent, and round-trips only by
                    # OPSIN's default-placement coincidence. Run the uniqueness oracle
                    # regardless of depth; when ambiguous, fall through to the explicit
                    # locant search (and, for distinct multi-position labels, to
                    # _decorate_distinct_multi_locant). (Was gated to depth 0, which
                    # shipped the ambiguous nested form -- a review review 2026-09-08.)
                    if (locant is None
                            and not _placement_unambiguous(
                                original, complete_sub_omission)
                            and (multi_descriptor
                                 or not _unit_completely_labelled(
                                     skel, off, keys, original))):
                        continue
                    if (locant is None
                            and _nitrogen_hydrogen_needs_locant(original, skel, off)):
                        held.append((p4542, loc_rank, sub_rank,
                                     _placement_quality(skel, off, locant_free=True),
                                     off, desc, candidate))
                        continue
                    if not _locant_names_the_labelled_position(
                            skel, off, locant, keys, original):
                        held.append((p4542, loc_rank, sub_rank,
                                     _placement_quality(skel, off, locant_free=False),
                                     off, desc, candidate))
                        continue
                    won.append((p4542, loc_rank, sub_rank,
                                _placement_quality(skel, off,
                                                   locant_free=(locant is None)),
                                off, desc, candidate))
            # front placement: a REQUIRED locant on a parent that opens
            # with a skeletal-replacement locant belongs at the front, hyphen-
            # joined, not spliced mid-parent. Offer that candidate (RT-gated) with
            # the lowest possible offset so it wins the tie-break over the
            # mid-parent splice.
            if allow_front_hyphen and locant is not None:
                fh = _front_hyphen_candidate(skel, desc)
                if fh is not None and _isotope_round_trips(
                        fh[1], original, stereo_blind=stereo_blind):
                    # front-hyphen is the parent-scope placement -- it is
                    # adjacent to the part it modifies, so it is quality 0.
                    won.append((p4542, locant, sub_rank, 0, fh[0], desc, fh[1]))
        if won:
            #: the first locant-form that yields ANY round-tripper is
            # the lowest-locant form; do not consider higher-locant forms.
            break
    below_pin = False
    if not won and held:
        won, below_pin = held, True
    if not won:
        return None, None, None, None
    # Tie-break order: /.3 (w[0]) -> locant (w[1]) -> subscript form
    # (w[2]) -> placement quality (w[3], adjacent-to-affix beats
    # front-detached) -> insertion offset (w[4], deterministic final key).
    won.sort(key=lambda w: (w[0], w[1], w[2], w[3], w[4]))
    _, loc_rank, _sub_rank, _q, off, desc, candidate = won[0]
    # (the Blue Book "1,2-di[(13C)methyl]benzene"): a descriptor spliced
    # INSIDE a substituent's own enclosing marks makes that substituent compound,
    # so the marks step up  ->  -> { } ('1-(amino(14C)methyl)cyclopentan-
    # 1-ol' -> '1-[amino(14C)methyl]cyclopentan-1-ol'). Escalate for THIS
    # single-descriptor placement; the multi-attachment caller
    # (:func:`_decorate_multi_position`) discards this candidate and runs its own
    # cascade after splicing every group, so it is unaffected. A descriptor at
    # parent scope (no enclosing pair) leaves the name untouched.
    candidate = _escalate_marks_for_descriptor(candidate, off)
    # (the Blue Book) / the Blue Book "(15N)-1*H*-indole (PIN)": a parenthetical
    # isotope descriptor placed at the FRONT of a parent that opens with a locant
    # digit (an indicated-hydrogen locant '1H-', a leading skeletal locant) is
    # separated from that locant by a hyphen -- a name-part is always separated
    # from a following locant by '-'. The bare front concatenation
    # '(15N)1H-indole' is a spelling defect even though it round-trips. Re-render
    # hyphenated when the descriptor is the plain front splice (off==0) and the
    # next char is a digit, RT-gated so a non-parsing hyphenation falls back to
    # the bare form (the front-hyphen candidates above already carry their '-').
    # The same holds inside a name: a descriptor spliced in front of a
    # substituent or parent part that opens with a locant ('[(2H11)-9,24-
    # dioxahexacyclo[...]...-6-yl]'), (:43718, "when the name, or a
    # part of a name, includes a preceding locant, a hyphen is inserted").
    _end = off + len(desc) if desc else -1
    if (desc and candidate[off:_end] == desc
            and candidate[_end:_end + 1].isdigit()):
        hyphenated = candidate[:_end] + "-" + candidate[_end:]
        if _isotope_round_trips(hyphenated, original, stereo_blind=stereo_blind):
            candidate = hyphenated
    # Below the PIN: a locant-free form kept for want of a locanted one that OPSIN
    # reads, and a letter locant repeated for one heteroatom ('(N,N-2H2)aniline'),
    # where the Blue Book cites the letter once ('(N-2H2)aniline (PIN)',
    # the Blue Book), a form OPSIN 2.9.0 cannot parse.
    # And a count subscript the preferred form omits ('ethan-1-(2H1)ol' where the
    # Blue Book writes 'ethan(2H)ol (PIN)',, the Blue Book, and
    # '(1-2H1)ethan-1-(2H)ol (PIN)' when a locant is required,:44214): the
    # always-show form is only the fallback for an omitted spelling OPSIN 2.9.0
    # cannot parse, so the name it builds is labelled below the PIN.
    # And a descriptor in front of a D/L configurational affix: "Stereochemical
    # affixes (for example D and L) are added according to the rules of special
    # classes" and precede the descriptor, 'L-(4-13C,35S)methionine',
    # the Blue Book-44128), a spelling OPSIN 2.9.0 does not read; the
    # readable '(3-13C)L-alanine' is kept below the PIN.
    # And a completely labelled substituent group whose locant-free spelling
    #,:44196) OPSIN 2.9.0 misreads keeps its locants, below the PIN
    # ('(1,1,2,2,3,3,3-2H7)propylbenzene').
    if (below_pin or _sub_rank == 1
            or re.match(r"(?:D|L|DL)-", candidate[off + len(desc):] if desc else "")
            or (isinstance(loc_rank, int) and loc_rank >= 1
                and _complete_unit_spelling_unreadable(skel, off, keys, original))
            or (isinstance(loc_rank, str) and desc
                and f"{loc_rank},{loc_rank}-" in desc)):
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(candidate)
        record_non_pin_fragment(_descriptor_fragment(candidate, desc))
    return candidate, off, desc, loc_rank


def _descriptor_fragment(name: str, desc: str) -> str:
    """The descriptor ``desc`` in ``name`` together with the name part it modifies
    (up to the end of that part's first alphabetic run): '(2H1)ol' in
    'ethan-1-(2H1)ol'. A below-PIN mark recorded on this fragment still finds the
    name when other descriptors are spliced in around it (the multi-position
    decorator combines per-group candidates into one name)."""
    start = name.find(desc) if desc else -1
    if start < 0:
        return ""
    j = start + len(desc)
    while j < len(name) and not name[j].isalpha():
        j += 1
    while j < len(name) and name[j].isalpha():
        j += 1
    return name[start:j]


def _mask_isotopes(mol: Chem.Mol, keep_indices) -> Chem.Mol:
    """A COPY of ``mol`` with every isotope label cleared except at
    ``keep_indices``. Used to test ONE attachment-group's placement in
    isolation -- see:func:`_decorate_multi_position`.

    A cleared D/T is an EXPLICIT plain-1H atom (same reasoning as
    ``strip_isotopes``): left alone, its canonical SMILES prints a literal
    ``[H]`` that can never match an OPSIN-parsed candidate (which has an
    ordinary IMPLICIT hydrogen there), so every group but the last would
    spuriously fail its round-trip. ``RemoveHs`` collapses exactly the
    now-plain explicit H atoms into implicit H count while leaving any
    STILL-labelled (isotope != 0) explicit H untouched -- RDKit treats an
    isotopically-labelled H as "special" and keeps it explicit.
    """
    copy = Chem.Mol(mol)
    keep = set(keep_indices)
    for atom in copy.GetAtoms():
        if atom.GetIdx() not in keep and atom.GetIsotope() != 0:
            atom.SetIsotope(0)
    try:
        copy = Chem.RemoveHs(copy)
    except Exception:
        pass
    return copy


def _partition_by_attachment(
    original: Chem.Mol, label_map: Dict[int, int]
) -> List[Dict[int, int]]:
    """Partition ``label_map`` into groups that share ONE skeleton attachment
    point, so each group's descriptor can be placed at its OWN insertion
    offset independently: the descriptor is inserted "before the
    part of the compound that is isotopically substituted" -- different
    PARTS of the compound take different insertion points).

    An explicit-H label (D/T, ``strip_isotopes`` never removes them from
    ``label_map`` even though ``RemoveHs`` collapses the STRIPPED copy) is
    attached to exactly one heavy neighbour; every D/T sharing that neighbour
    (the two D's of an NH2, the two of a CH2,...) is one group. A labelled
    HEAVY atom (13C, 18F, 11C,...) is its own attachment point.

    A single-group molecule (everything shares one neighbour, or there is
    only one labelled atom) returns a length-1 list, so the existing
    single-descriptor path is exercised exactly as before for every
    previously-working case -- this function only SPLITS when the labelled
    atoms sit at structurally distinct positions, and the OPSIN round-trip
    oracle still has the final say on every candidate it enables.
    """
    groups: Dict[int, Dict[int, int]] = {}
    for idx, mass in label_map.items():
        atom = original.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "H":
            nbrs = atom.GetNeighbors()
            key = nbrs[0].GetIdx() if nbrs else idx
        else:
            key = idx
        groups.setdefault(key, {})[idx] = mass
    return list(groups.values())


def _enclosing_pairs(name: str) -> List[Tuple[int, int]]:
    """Every matched enclosing-mark pair ``(open_idx, close_idx)`` in ``name``,
    across ````, ```` and ``{}`` alike (they interleave by nesting level, not
    by type). Unmatched marks are ignored -- a name string is always balanced by
    construction."""
    stack: List[int] = []
    pairs: List[Tuple[int, int]] = []
    for i, ch in enumerate(name):
        if ch in "([{":
            stack.append(i)
        elif ch in ")]}" and stack:
            pairs.append((stack.pop(), i))
    return pairs


def _escalate_marks_for_descriptor(name: str, desc_start: int) -> str:
    """: an isotope descriptor spliced INSIDE a substituent's
    existing enclosing marks makes that substituent compound, so its marks must
    step up ```` -> ```` -> ``{}`` (the Blue Book ``1,2-di[(13C)methyl]benzene``).

    ``desc_start`` is where the descriptor's own ``(`` now sits in ``name``. The
    immediately-enclosing detachable pair (the one with the largest open index
    strictly before ``desc_start`` whose close is after it) is re-enclosed by its
    content's nesting depth via the shared ``apply_enclosing_marks`` primitive;
    the step-up may over-fill THAT group's own parent, so the cascade repeats
    outward until an ancestor is already at the right mark. A descriptor at parent
    scope (no enclosing pair) leaves ``name`` untouched.

    OPSIN-RT is blind to the bracket LEVEL, so this must be deterministically
    correct rather than oracle-selected. Each rewrite only swaps the two bracket
    characters of one pair, so string length -- and therefore every other index
    -- is preserved across the cascade.
    """
    from ..assembly.naming_utils import apply_enclosing_marks

    pos = desc_start
    while True:
        enclosing = [(o, c) for (o, c) in _enclosing_pairs(name)
                     if o < pos <= c]
        if not enclosing:
            return name
        o, c = max(enclosing, key=lambda p: p[0])
        rewrapped = apply_enclosing_marks(name[o + 1:c], -1)
        if rewrapped == name[o:c + 1]:
            return name
        name = name[:o] + rewrapped + name[c + 1:]
        pos = o  # cascade to the parent of the group just stepped up


def _parent_carries_positional_isotope(mol: Optional[Chem.Mol]) -> bool:
    """True when a labelled atom sits on the PARENT skeleton rather than on a
    purely-acyclic-carbon substituent branch.

    A ring atom, or any labelled heteroatom, is taken to be part of the parent
    hydride, so ``1-(79Br)bromo(2-13C)benzene`` (¹³C on a ring carbon) restores the
    ring's monosubstituted locant while ``(1,1,2,2,2-pentafluoro(1-13C)ethyl)benzene``
    (¹³C on a substituent sp3 carbon) does not. This is a conservative structural
    proxy for the parent/substituent boundary the isotope module deliberately does not
    compute grammatically; it gates the ``isotope_parent_positional_scope`` entered at
    each forced-locant re-render below, and every emission stays OPSIN-round-trip-gated
    so a mis-classification can never ship a wrong molecule.

    A labelled HYDROGEN (D/T) is not itself a skeleton atom, so its
    parent/substituent status is read off its heavy NEIGHBOUR: a D on a RING carbon
    (``Brc1c([2H])cccc1`` -> ``1-bromo(2-2H)benzene``) labels a parent ring position
    and breaks its monosubstituted symmetry, exactly as a ¹³C on that carbon would;
    but a D on a substituent methyl (``[2H]C([2H])([2H])c1ccccc1`` -> ``(2H3)toluene``)
    leaves the ring symmetric and must NOT restore a ring locant. The ring-neighbour
    test captures precisely that boundary (v52 a phase Task N2,."""
    if mol is None:
        return False
    for a in mol.GetAtoms():
        if not a.GetIsotope():
            continue
        if a.IsInRing() or a.GetSymbol() not in ("C", "H"):
            return True
        # A labelled H (D/T): a parent-positional label iff its heavy neighbour is a
        # ring (parent) atom -- a substituent-carbon D leaves the parent symmetric.
        if a.GetSymbol() == "H":
            for nb in a.GetNeighbors():
                if nb.IsInRing():
                    return True
    return False


@contextlib.contextmanager
def _isotope_locant_scopes(original: Chem.Mol):
    """Enter ``forced_locant_scope``, every re-render) and, when the parent
    skeleton itself carries a positional isotope, ALSO
    ``isotope_parent_positional_scope``, the parent-own-locant licences)."""
    from ..assembly.locant_omission import (
        forced_locant_scope, isotope_labelled_original_scope,
        isotope_parent_positional_scope)
    with forced_locant_scope("isotope"):
        if _parent_carries_positional_isotope(original):
            with isotope_parent_positional_scope("isotope"), \
                    isotope_labelled_original_scope(original):
                yield
        else:
            yield


def _forced_locant_skeleton(namer, stripped_smiles: Optional[str],
                            original: Chem.Mol, skeleton: str) -> str:
    """Re-render the isotope-STRIPPED skeleton with its -omitted
    substituent locants RESTORED, when (and only when) the parent skeleton itself
    carries a positional isotope, "possibility of isomers"): a ring D /
    ¹³C / ¹⁵N that breaks the monosubstituted parent's symmetry restores the
    substituent's own locant (``1-bromo(2,4-2H2)benzene``, not
    ``bromo(2,4-2H2)benzene``). Returns ``skeleton`` unchanged when the parent is
    unlabelled (a substituent-buried label leaves the parent symmetric) or the
    re-render fails to build -- the caller RT-gates the result, so a wrong or
    unbuildable re-render never ships. Shares the exact scope/oracle the
    single-descriptor restoration (``_place_on_skeleton``) uses."""
    if namer is None or stripped_smiles is None:
        return skeleton
    if not _parent_carries_positional_isotope(original):
        return skeleton
    with _isotope_locant_scopes(original):
        skel2 = _flagged_systematic_namer(namer).name(stripped_smiles)
    if skel2 and "unknown" not in skel2.lower():
        return skel2
    return skeleton


def _decorate_multi_position(
    skeleton: str, groups: List[Dict[int, int]], original: Chem.Mol, n_pos: int,
    namer=None, stripped_smiles: Optional[str] = None,
    merge_collisions: bool = False,
) -> Optional[str]:
    """Place ONE descriptor per attachment GROUP, each at its own insertion
    offset + /.2/.3 -- different labelled parts of a
    compound each get their own descriptor scoped to that part, e.g. BB's own
    ``benzene(13C)carbonitrile`` / ``3-[ethyl(2-34S)trisulfanyl]propanoic
    acid`` pattern of several independently-scoped parenthetical descriptors
    in one name).

    A single combined descriptor (the existing enumeration, one shared
    locant for every labelled atom of a nuclide) cannot express labelled
    atoms sitting at MORE than one structurally distinct position -- e.g.
    per-deuterated glycine has D on the amino N, the alpha C, and the
    carboxyl O, three unrelated "parts". Each group's own winning
    ``(offset, descriptor)`` is found INDEPENDENTLY by:func:`_find_best_placement`
    against an isotope-MASKED copy of ``original`` carrying ONLY that group's
    labels -- valid because OPSIN scopes each parenthesized descriptor to the
    substring it precedes, independent of any other descriptor elsewhere in
    the name (verified 2026-08-23: the three-way glycine split round-trips
    when the three independently-found descriptors are combined). The
    combined candidate is still OPSIN-RT gated against the TRUE original
    before being returned, so any compositionality failure (e.g. two groups
    wanting the identical offset) simply fails closed -- never a wrong name.
    """
    if len(groups) < 2:
        return None

    def _place(skel: str) -> Tuple[Optional[List[Tuple[int, str]]], bool]:
        """Per-group ``(offset, descriptor)`` on ONE skeleton spelling, plus a flag
        for whether ANY group cites a positional (integer) isotope locant.

        Returns ``(None, False)`` if any group cannot be placed on ``skel``."""
        placements: List[Tuple[int, str]] = []
        any_positional = False
        for group in groups:
            by_key: Dict[Tuple[int, str], int] = {}
            maxpos_by_key: Dict[Tuple[int, str], int] = {}
            for idx, mass in group.items():
                atom = original.GetAtomWithIdx(idx)
                el = atom.GetSymbol()
                by_key[(mass, el)] = by_key.get((mass, el), 0) + 1
                maxpos_by_key[(mass, el)] = max(
                    maxpos_by_key.get((mass, el), 0), _max_atoms_at_position(atom))
            group_keys = sorted(
                (((mass, el), by_key[(mass, el)], maxpos_by_key[(mass, el)])
                 for (mass, el) in by_key),
                key=lambda kv: (kv[0][1], kv[0][0]))
            masked = _mask_isotopes(original, group.keys())
            # letter locants (italic ``O`` on a labelled hydroxyl/carboxyl
            # oxygen, ``N`` on an amide/amine D) scoped to THIS group's own atoms, so
            # the multi-descriptor composition can place a group whose only legal
            # position is a letter locant -- e.g. an 18O on the carboxyl -OH beside a
            # 34S inside a trisulfanyl substituent (``3-[ethyl(1-34S)trisulfanyl]
            # (O-18O)propanoic acid``). The single-descriptor enumeration already
            # passes these (``_enumerate``); the multi-position path did not, so such
            # a group returned ``off is None`` and the whole row abstained. Computed
            # per group (masked to that group's indices) so each group offers only the
            # letter locants ITS atoms justify; still OPSIN-RT gated (0-wrong).
            # ``complete_sub_omission`` stays False here (see ``_placement_unambiguous``:
            # a multi-descriptor name must cite all locants once any is required), and
            # ``multi_descriptor`` keeps the unit licence off, since it
            # would read only this group's masked copy (a propanoyl whose CH3 and CH2
            # are two groups: OPSIN reads the bumped '(3H4)propanoylamino' of the
            # CH3 group as another constitution, the licence then placed that group
            # locant-free, and the two groups could not be joined at their slot).
            group_letter_locants = _letter_locant_candidates(
                original, group, n_pos, skeleton=skel)
            _cand, off, desc, loc_rank = _find_best_placement(
                skel, group_keys, masked, n_pos,
                extra_locants=group_letter_locants, multi_descriptor=True)
            if off is None:
                return None, False
            if isinstance(loc_rank, int) and loc_rank >= 1:
                any_positional = True
            placements.append((off, desc, loc_rank, group_keys))
        return placements, any_positional

    def _merged(placements) -> Optional[List[Tuple[int, str]]]:
        """One descriptor per insertion offset. Groups whose descriptors land at the
        SAME offset label the same part of the name. (the Blue Book)
        places the nuclide symbols, "preceded by any necessary locant(s), letters,
        and/or numerals, before the part of the compound that is isotopically
        substituted", and "When two or more nuclide symbols appear at the same place
        in the name they are cited first in alphabetical order and then by their
        mass number" -- one descriptor for that place, citing every labelled
        position of the part: the D of a CD2 (locant 2) and of the amide NH (locant
        N) of one acetamide give '(N,2,2-2H3)acetamide',:43824, with its
        PIN '(N-2H1)acetamide',:43828; locant order,:3195, "Italic
        capital and lower-case letter locants are lower than Greek letter locants,
        which, in turn, are lower than numerals"). The labels of one nuclide pool
        their locants (a scalar locant once per atom at it); different nuclides stay
        separate groups in that order (``format_isotope_descriptor``). Collisions are
        merged only on the last-resort call (``merge_collisions``, made after every
        other placement strategy declined, so no name that ships without it
        changes); otherwise, and when a colliding group is placed without a locant
        (it cannot share a descriptor that cites locants), a collision fails closed
        (None) as before."""
        by_off: Dict[int, list] = {}
        for off, desc, loc_rank, group_keys in placements:
            by_off.setdefault(off, []).append((desc, loc_rank, group_keys))
        out: List[Tuple[int, str]] = []
        for off, items in by_off.items():
            if len(items) == 1:
                out.append((off, items[0][0]))
                continue
            if not merge_collisions or any(
                    loc_rank == -1 for _d, loc_rank, _k in items):
                return None
            pooled: Dict[Tuple[int, str], Tuple[List, int]] = {}
            for _d, loc_rank, group_keys in items:
                for (mass, el), count, maxpos in group_keys:
                    locs, mp = pooled.get((mass, el), ([], 0))
                    pooled[(mass, el)] = (locs + [str(loc_rank)] * count,
                                          max(mp, maxpos))
            groups_desc = [
                (tuple(sorted(locs, key=_locant_sort_key)), mass, el, len(locs), mp)
                for (mass, el), (locs, mp) in pooled.items()]
            out.append((off, format_isotope_descriptor(groups_desc)))
        return out

    def _assemble(skel: str, placements) -> Optional[str]:
        # Descriptors that land at one offset are merged into one (``_merged``);
        # a collision it cannot merge fails closed.
        placements = _merged(placements)
        if placements is None:
            return None
        # (the Blue Book): "Immediately after the parentheses there
        # is neither space nor hyphen, except that when the name, or a part of a
        # name, includes a preceding locant, a hyphen is inserted" -- on the merging
        # call (``merge_collisions``, the last resort) a descriptor spliced right
        # before a locant digit is hyphen-joined to it ('(15N)-1H-indole'), as the
        # single-descriptor and distinct-position paths do; the bare splice stays
        # the fallback when OPSIN does not read the hyphenated one. The first call
        # keeps its bare splice, so no name it already builds changes.
        variants = []
        for hyphenate in ((True, False) if merge_collisions else (False,)):
            inserts = [(off, desc + ("-" if hyphenate and skel[off:off + 1].isdigit()
                                     else "")) for off, desc in placements]
            if inserts not in variants:
                variants.append(inserts)
        for inserts in variants:
            # Splice from the HIGHEST offset down so an earlier offset is never
            # shifted by a later insertion.
            out = skel
            for off, ins in sorted(inserts, key=lambda p: -p[0]):
                out = out[:off] + ins + out[off:]
            #: step up the enclosing marks of any substituent that just
            # received a descriptor INSIDE its own marks (``(1-bromopropyl)`` +
            # ``(81Br)`` -> ``[1-(81Br)bromopropyl]``). The final start index of each
            # descriptor in ``out`` is its own offset plus the total length of every
            # insert spliced at a LOWER offset (each rewrite below preserves length,
            # so these indices stay valid through the cascade).
            shift = 0
            for off, ins in sorted(inserts):
                out = _escalate_marks_for_descriptor(out, off + shift)
                shift += len(ins)
            if _isotope_round_trips(out, original):
                # One merged descriptor for a completely labelled substituent group
                # whose locant-free spelling, the Blue Book) OPSIN
                # 2.9.0 misreads: the locanted name ships below the PIN
                # ('(1,1,2,2,3,3,3-2H7)propylbenzene').
                if len(placements) == 1:
                    _all_keys = _label_keys(original)
                    if (_all_keys and _complete_unit_spelling_unreadable(
                            skel, placements[0][0], _all_keys, original)):
                        from ..metrics.provenance import record_non_pin_fragment
                        record_non_pin_fragment(out)
                        record_non_pin_fragment(
                            _descriptor_fragment(out, placements[0][1]))
                return out
        return None

    placements, any_positional = _place(skeleton)
    if placements is None:
        return None

    # (the Blue Book) / (the Blue Book, "possibility of isomers"): when
    # ANY part carries a positional isotope locant, that label breaks the parent's
    # symmetry, so the parent restores its own licence-omitted locants too --
    # ``1-(79Br)bromo(2-13C)benzene``, not ``(79Br)bromo(2-13C)benzene``. The parent
    # here was named from the isotope-STRIPPED skeleton, so its licences
    # elided freely; re-name it inside the ambient forced-locant scope (as
    # ``_decorate_mixed_locant`` does) and re-place on the locanted spelling.
    # RT-gated, falling back to the free skeleton if the re-render fails to build or
    # parse (never trade a verified name for an unverified one). Gated on
    # ``any_positional`` so an all-locant-free multi-group label (per-D glycine)
    # keeps its licensed omissions.
    if any_positional and namer is not None and stripped_smiles is not None:
        with _isotope_locant_scopes(original):
            skel_forced = _flagged_systematic_namer(namer).name(stripped_smiles)
        if (skel_forced and "unknown" not in skel_forced.lower()
                and skel_forced != skeleton):
            fplacements, _ = _place(skel_forced)
            if fplacements is not None:
                forced_out = _assemble(skel_forced, fplacements)
                if forced_out is not None:
                    return forced_out
    return _assemble(skeleton, placements)


#: Heavy-atom cap for the distinct-position multi-locant sweep. The sweep does
#: up to (offsets x n) single-label OPSIN parses; 60 bounds the cost on a large
#: cage/ring parent while covering every realistic labelled skeleton.
_MULTI_LOCANT_SWEEP_CAP = 60
#: Bound on the number of candidate locant-SETS the sweep RT-gates per offset.
#: An asymmetric parent yields ~1 (each atom's environment is unique); a highly
#: symmetric one can combinatorially blow up, so a set count over this fails
#: closed rather than spending unbounded OPSIN parses.
_MULTI_LOCANT_COMBO_CAP = 4000


def _decorate_distinct_multi_locant(
    skeleton: str, keys, original: Chem.Mol,
    label_map: Dict[int, int], n_pos: int,
) -> Optional[str]:
    """ /: ONE nuclide labelled at MULTIPLE structurally
    DISTINCT positions on ONE parent -> a single ascending multi-locant
    descriptor, e.g. D at cage/ring positions 1,3,7 -> ``(1,3,7-2H3)``.

    Neither existing path expresses this. ``_find_best_placement`` (the single
    combined enumeration) repeats ONE shared locant ``count`` times
    (``2,2,2-2H3`` -- three atoms all at position 2, via
    :func:`format_isotope_descriptor`), and:func:`_decorate_multi_position`
    splits by distinct ATTACHMENT GROUP, not by distinct position within one
    parent. So a partially-labelled von Baeyer cage or a multiply-labelled ring
    (the residual isotope-abstain mass, measured 2026-09-08) fell through to
    ``None``.

    Orthonym exposes no atom->locant map for these parents, so -- exactly as the
    single-descriptor enumeration does -- this treats the skeleton's locant space
    as opaque and lets OPSIN adjudicate. For each insertion offset it sweeps a
    single-label probe ``(L-<sym>1)`` over every candidate locant TOKEN ``L``
    (parse -> canonical SMILES), building a locant->structure map, then groups
    locants by the single-label isotopomer they produce and the labelled atoms by
    their own single-label reference (:func:`_mask_isotopes`). Each reference class
    must be filled from its matching locant class; the labelled SET (not the
    individual atom, which is indistinguishable on a symmetric parent) is what the
    isotopomer fixes, so it enumerates the candidate locant sets in lowest-locant
    order and returns the first that OPSIN-RT gates against the true
    original.

    The candidate locants include PRIMED forms (``4'``, ``1''``) when the skeleton
    itself carries a prime (a ring assembly ``1,1'-biphenyl``, a multiparent fused
    system): those parents number across sub-units with primed locants, which a
    plain integer sweep cannot reach, so a D on the second ring of biphenyl had no
    placeable locant. The prime depth is bounded by the skeleton's own prime count,
    so a skeleton with NO prime sweeps the identical integer set as before (no
    behaviour change for those parents).

    Fail-closed (None) outside its scope: exactly one nuclide, count >= 2, every
    reference class fillable from its locant class, the candidate-set count within
    :data:`_MULTI_LOCANT_COMBO_CAP`, and a round-tripping candidate. A wrong
    locant set simply fails to reproduce the structure, so no wrong or
    label-dropping name is ever emitted.
    """
    import itertools

    if len(keys) != 1:
        return None
    (mass, el), count, _maxpos = keys[0]
    my_atoms = [idx for idx, m in label_map.items()
                if m == mass and original.GetAtomWithIdx(idx).GetSymbol() == el]
    if count < 2 or len(my_atoms) != count:
        return None
    # Single-label canonical reference (isotopes kept) for each labelled atom,
    # grouped: {reference_smiles: how many atoms carry it}. Symmetric atoms share
    # one reference and must be filled from the SAME locant class.
    #
    # Two keys name the labelled atom: the canonical SMILES (as before), and, when
    # that leaves a reference class unfilled, the bond-order-free key
    # (``_bond_order_free_key``): OPSIN can write a conjugated ring RDKit does not
    # perceive as aromatic in the other Kekule alternation than the input, so no
    # probe's SMILES equals any reference although each probe labels a real atom.
    try:
        ref_counts: Tuple[Dict[str, int], Dict[str, int]] = ({}, {})
        for idx in my_atoms:
            masked = _mask_isotopes(original, [idx])
            for kind, r in enumerate((Chem.MolToSmiles(masked),
                                      _bond_order_free_key(masked))):
                ref_counts[kind][r] = ref_counts[kind].get(r, 0) + 1
    except Exception:
        return None
    sym = nuclide_symbol(mass, el)
    n = min(n_pos, _MULTI_LOCANT_SWEEP_CAP)
    # Prime depth: 0 when the skeleton carries no prime (then the token set is the
    # identical integer sweep as before -> no behaviour change). A ring assembly /
    # multiparent parent numbers across sub-units with primed locants; sweep up to
    # its own prime count (bounded to 3 to keep the probe count sane).
    maxprime = min(3, skeleton.count("'"))
    loc_tokens = [f"{base}{chr(39) * p}"
                  for p in range(maxprime + 1) for base in range(1, n + 1)]
    # Fusion / ring-junction carbons carry a LETTER-suffixed locant (indole 3a, 7a;
    # a labelled fused-ring bridgehead has no plain integer locant). Add '<base>a'
    # tokens when the molecule has a ring-fusion atom (a carbon shared by >=2 rings);
    # OPSIN rejects the invalid ones (they never enter loc_canon) and the RT gate
    # keeps 0-wrong. A non-fused parent keeps the identical integer sweep.
    ri = original.GetRingInfo()
    if any(ri.NumAtomRings(a) >= 2 for a in range(original.GetNumAtoms())):
        loc_tokens += [f"{base}a" for base in range(1, n + 1)]
    seen: set = set()
    offs = [o for o in ([0] + _insertion_offsets(skeleton))
            if not (o in seen or seen.add(o))]
    #: a slot in front of a substituent prefix's attachment locant is not
    # a place the Blue Book puts a descriptor (``_slot_precedes_substituent_
    # locant``), but OPSIN reads it back, and the first offset that round-trips
    # wins below. Try those slots last (stable), so the Blue Book slot -- after
    # the locant, before the prefix or the parent stem it labels -- is found first
    # ('2-(2,3,4,5,6-2H5)phenyl-1-benzofuran', not '(2,3,4,5,6-2H5)-2-phenyl-1-
    # benzofuran'); a molecule whose only round-tripper is such a slot still names.
    offs.sort(key=lambda o: _slot_precedes_substituent_locant(skeleton, o))
    for off in offs:
        # locant token -> single-label isotopomer at this offset
        loc_canons: Tuple[Dict[str, str], Dict[str, str]] = ({}, {})
        for tok in loc_tokens:
            probe = f"({tok}-{sym}1)"
            s = _opsin_parse(skeleton[:off] + probe + skeleton[off:])
            if not s:
                continue
            pm = Chem.MolFromSmiles(s)
            if pm is not None:
                loc_canons[0][tok] = Chem.MolToSmiles(pm)
                loc_canons[1][tok] = _bond_order_free_key(pm)
        if not loc_canons[0]:
            continue
        # invert: reference key -> locant tokens that reproduce it (lowest-first);
        # each reference class must be fillable from its own locant class, under the
        # SMILES key or else under the bond-order-free key.
        for loc_canon, ref_count in zip(loc_canons, ref_counts):
            by_ref: Dict[str, List[str]] = {}
            for tok in sorted(loc_canon, key=_locant_sort_key):
                by_ref.setdefault(loc_canon[tok], []).append(tok)
            if not any(ref is None or ref not in by_ref or len(by_ref[ref]) < need
                       for ref, need in ref_count.items()):
                break
        else:
            continue
        # count candidate locant-set combinations; fail closed if unbounded
        combos = 1
        for ref, need in ref_count.items():
            combos *= _n_choose_k(len(by_ref[ref]), need)
            if combos > _MULTI_LOCANT_COMBO_CAP:
                break
        if combos > _MULTI_LOCANT_COMBO_CAP:
            continue
        # per-reference choices; the full locant set is one pick from each class
        per_ref_choices = [
            list(itertools.combinations(by_ref[ref], need))
            for ref, need in ref_count.items()
        ]
        candidate_sets = []
        for pick in itertools.product(*per_ref_choices):
            locs = sorted((tok for group in pick for tok in group),
                          key=_locant_sort_key)
            candidate_sets.append(tuple(locs))
        # lowest-locant-first, unprimed before primed
        candidate_sets.sort(key=lambda s: [_locant_sort_key(t) for t in s])
        # /: a descriptor spliced immediately before a locant
        # digit (a front skeletal locant '1,1'-biphenyl', or a bracketed/leading
        # indicated-H '1H-indol-...') is hyphen-separated from it:
        # '(2,4',5-2H3)-1,1'-biphenyl', '(15N)-1H-indole'. Try the hyphenated form
        # first when the splice point is a digit; RT-gated, so a non-parsing
        # hyphenation falls back to the bare splice that already round-tripped.
        before_digit = off < len(skeleton) and skeleton[off:off + 1].isdigit()
        for locs in candidate_sets:
            loc_part = ",".join(locs)
            # count-bearing subscript first, bare form as a parseability fallback.
            for desc in (f"({loc_part}-{sym}{count})", f"({loc_part}-{sym})"):
                joins = ["-", ""] if before_digit else [""]
                for join in joins:
                    cand = skeleton[:off] + desc + join + skeleton[off:]
                    if _isotope_round_trips(cand, original):
                        cand = _escalate_marks_for_descriptor(cand, off)
                        # A completely labelled substituent group cites no locants
                        #, the Blue Book); when OPSIN 2.9.0 misreads
                        # that spelling ('(2H7)propylbenzene' as the propan-2-yl
                        # isomer) the locanted name ships below the PIN.
                        if _complete_unit_spelling_unreadable(skeleton, off, keys,
                                                              original):
                            from ..metrics.provenance import record_non_pin_fragment
                            record_non_pin_fragment(cand)
                            record_non_pin_fragment(_descriptor_fragment(cand, desc))
                        return cand
    return None


def _locant_sort_key(tok: str) -> tuple:
    """ (the Blue Book) locant order: 'Primed locants are placed
    immediately after the corresponding unprimed locants in a set arranged in
    ascending order; locants consisting of a number and a lower-case letter...
    are placed immediately after the corresponding numeric locant'.

    So order by NUMERIC value first, then by (letter-suffix, prime-count) as a
    tie-break -- i.e. {2,5,4'} -> 2,4',5 (not 2,5,4'); 4a immediately after 4;
    4' immediately after 4. Key = (base_number, has_letter, letter, nprime).
    ``"2"`` -> (2,0,'',0); ``"4'"`` -> (4,0,'',1); ``"3a"`` -> (3,1,'a',0);
    ``"4'a"`` -> (4,1,'a',1)."""
    nprime = tok.count("'")
    core = tok.replace("'", "")
    # split trailing lower-case letter(s) (fusion locant like 3a / 4a)
    i = len(core)
    while i > 0 and core[i - 1].isalpha():
        i -= 1
    num_part, letter = core[:i], core[i:]
    try:
        base = int(num_part)
    except ValueError:
        base = 0
    return (base, 1 if letter else 0, letter, nprime)


def _n_choose_k(n: int, k: int) -> int:
    """C(n, k), 0 for k > n. Small local helper for the combination bound."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = 1
    for i in range(k):
        num = num * (n - i) // (i + 1)
    return num


# ---------------------------------------------------------------------------
# (the Blue Book) -- systematic-parent fallback
# ---------------------------------------------------------------------------
# "When the nuclide is located at a position in a retained name that is not
# numbered[,] a systematic name that identifies separately the relevant atom
# is used for the IUPAC preferred name." style='systematic' does not always
# deliver that: measured 2026-08-23, on a substituted 2-carbon carboxylic
# acid it still keeps the RETAINED "acetic acid" stem ("aminoacetic acid",
# not "aminoethanoic acid") -- omits the substituent's own locant and
# the retained "acid" suffix has no "-oic acid" slot to insert a descriptor
# before, unlike the fully systematic form. The amino-acid class already
# carries its own fully systematic namer for exactly this shape
# (`_name_amino_acid_systematic`); this fallback is scoped to it. When it
# does not apply, this returns None and the caller's fallback is a no-op --
# never a forced or guessed alternate parent.
def _systematic_parent_fallback(
        stripped: Chem.Mol,
        original: Optional[Chem.Mol] = None,
        label_map: Optional[Dict[int, int]] = None,
        retained_name: Optional[str] = None) -> Optional[str]:
    """An alternate, more fully systematic parent name for ``stripped``.

    Two modes, selected by whether the isotope-label context is supplied:

    * **Label-context mode** (``original`` and ``label_map`` given) — the
       (the Blue Book) ring-carbon parent for a nuclide that sits on a
      position a RETAINED name does not number (see
      :func:`_ring_carbon_systematic_parent`). Consulted PRE-emptively by the
      decorator, because such a retained name (``benzonitrile``) still
      round-trips and so is never abandoned by the ordinary skeleton loop.
      Returns None (never the amino-acid form) when no ring-carbon class applies.
    * **Plain mode** (no label context) — the amino-acid systematic name (the
      verified per-D-glycine gap), used as a POST-retained fallback below the
      skeleton loop. Returns None when inapplicable.
    """
    if original is not None and label_map is not None:
        return _ring_carbon_systematic_parent(original, label_map, retained_name)
    try:
        from .amino_acids import _name_amino_acid_systematic, detect_amino_acid
    except Exception:
        return None
    try:
        if not detect_amino_acid(stripped):
            return None
        return _name_amino_acid_systematic(stripped)
    except Exception:
        return None


def _aromatic_benzene(mol: Chem.Mol, atom_idx: int):
    """Ring tuple for a benzene (6-membered aromatic all-carbon) ring that
    ``atom_idx`` belongs to, else None."""
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) == 6 and atom_idx in ring and all(
                mol.GetAtomWithIdx(i).GetIsAromatic()
                and mol.GetAtomWithIdx(i).GetSymbol() == "C" for i in ring):
            return ring
    return None


def _saturated_carbocycle(mol: Chem.Mol, atom_idx: int, size: int = 6):
    """Ring tuple for a ``size``-membered all-carbon SATURATED (single-bonded)
    ring that ``atom_idx`` belongs to, else None."""
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != size or atom_idx not in ring:
            continue
        if not all(mol.GetAtomWithIdx(i).GetSymbol() == "C"
                   and not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        n = len(ring)
        if all(
                (b := mol.GetBondBetweenAtoms(ring[k], ring[(k + 1) % n])) is not None
                and b.GetBondType() == Chem.BondType.SINGLE for k in range(n)):
            return ring
    return None


def _ring_monosubstituted(mol: Chem.Mol, ring) -> bool:
    """True iff exactly one ring atom carries a heavy-atom substituent outside
    the ring (so the ``benzene``/``cyclohexane`` parent needs no ring locant)."""
    ring_set = set(ring)
    exo = sum(1 for i in ring
              for nb in mol.GetAtomWithIdx(i).GetNeighbors()
              if nb.GetIdx() not in ring_set and nb.GetSymbol() != "H")
    return exo == 1


def _is_carboxyl_carbon(mol: Chem.Mol, atom: Chem.Atom) -> bool:
    """True iff ``atom`` is a carboxylic-acid carbon C(=O)-OH (the single-bonded
    O is degree 1, i.e. a real -OH, NOT an ester/anhydride O)."""
    if atom.GetSymbol() != "C":
        return False
    dbl_o = oh = False
    for nb in atom.GetNeighbors():
        if nb.GetSymbol() != "O":
            continue
        b = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
        if b.GetBondType() == Chem.BondType.DOUBLE:
            dbl_o = True
        elif b.GetBondType() == Chem.BondType.SINGLE and nb.GetDegree() == 1:
            oh = True
    return dbl_o and oh


def _is_aldehyde_carbon(mol: Chem.Mol, atom: Chem.Atom) -> bool:
    """True iff ``atom`` is an aldehyde carbon -CHO: a carbon double-bonded to a
    terminal (degree-1) oxygen, bearing exactly ONE hydrogen and exactly one
    OTHER heavy neighbour (the part it is attached to). An explicit D/T hydrogen
    counts toward the single H (an isotope-labelled aldehyde hydrogen). A
    carboxyl carbon (which also carries an -OH heavy neighbour and no H) is
    excluded by the ``heavy_other == 1 and h == 1`` test."""
    if atom.GetSymbol() != "C":
        return False
    dbl_o = False
    heavy_other = 0
    h = atom.GetTotalNumHs()
    for nb in atom.GetNeighbors():
        b = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
        if (nb.GetSymbol() == "O" and nb.GetDegree() == 1
                and b.GetBondType() == Chem.BondType.DOUBLE):
            dbl_o = True
        elif nb.GetSymbol() == "H":
            h += 1
        else:
            heavy_other += 1
    return dbl_o and heavy_other == 1 and h == 1


def _aldehyde_carbon_for_label(
        mol: Chem.Mol, atom: Chem.Atom) -> Optional[Chem.Atom]:
    """The aldehyde carbon of the -CHO group a labelled ``atom`` belongs to --
    when ``atom`` IS that carbon, is its doubly-bonded terminal oxygen, or is its
    hydrogen (D/T) -- else None. Lets a nuclide on ANY of the three atoms of the
    unnumbered aldehyde group route to the systematic ring-carbon parent."""
    if _is_aldehyde_carbon(mol, atom):
        return atom
    if atom.GetSymbol() in ("O", "H"):
        for nb in atom.GetNeighbors():
            if _is_aldehyde_carbon(mol, nb):
                return nb
    return None


def _ring_carbon_systematic_parent(
        original: Chem.Mol, label_map: Dict[int, int],
        retained_name: Optional[str]) -> Optional[str]:
    """ (the Blue Book): the systematic ring-carbon parent spelling for a
    nuclide that sits on a position a RETAINED name does not number.

    Verbatim, under the heading " Location of nuclides on positions not
    normally denoted by locants": "**** When the nuclide is located at
    a position in a retained name that is not numbered a systematic name that
    identifies separately the relevant atom is used for the IUPAC preferred
    name." The BB PIN examples (the Blue Book-44259) are ``benzene(13C)carbonitrile``
    (not ``(cyano-13C)benzonitrile``), ``benzene(13C)carboxylic acid`` and
    ``[phenyl(13C)methyl]hydrazine``.

    The retained parents ``benzonitrile`` / ``benzoic acid`` and the retained
    substituent ``benzyl`` do NOT number their nitrile / carboxyl / benzylic
    carbon, so a label there gets no locant; the decorator's retained placement
    then emits an unlocanted, BB-illegitimate ``(13C)benzonitrile`` that
    nonetheless round-trips. This returns the systematic spelling that DOES
    express the atom, so the decorator's placement cascade lands the descriptor
    before the suffix (``benzene(13C)carbonitrile``). Every candidate stays
    ``_isotope_round_trips``-gated at the call site, so a wrong skeleton is
    discarded (0-wrong); this fires ONLY when a labelled atom IS such an
    unnumbered functional carbon and the ring is monosubstituted, so the
    retained/systematic specs are untouched whenever the rule does not apply.

    Covers exactly the skeleton classes on the isotope target set:
    monosubstituted benzene + carbonitrile / + carboxylic acid / + carbaldehyde,
    a labelled benzyl substituent (respelled ``benzyl`` -> ``(phenylmethyl)``),
    and a cyclohexane-1,1-diyl gem-diacid (one -COOH becomes the suffix, the
    other a ``carboxy`` prefix). A cyclohexane gem-diacid bearing TWO DISTINCT
    nuclides (one per carboxyl) is beyond the single-combined-descriptor
    placement and is left to abstain rather than emit a degraded name."""
    if not label_map:
        return None
    for idx in label_map:
        atom = original.GetAtomWithIdx(idx)

        # (5) benzaldehyde's aldehyde group: the retained name ``benzaldehyde``
        # does NOT number its -CHO carbon, oxygen or hydrogen, so a nuclide on
        # ANY of the three gets no locant and yields an unlocanted,
        # BB-illegitimate ``(13C)benzaldehyde`` that nonetheless round-trips.
        # Switch to the systematic ring-carbon parent ``benzenecarbaldehyde``
        #, the Blue Book), which numbers that carbon, so the descriptor
        # lands before the suffix -> ``benzene(13C)carbaldehyde``. Checked before
        # the ``C``-only guard below because the labelled atom may be the -CHO
        # oxygen (``(18O)``) or its D/T. RT-gated at the call site, fired ONLY
        # when a label is on that unnumbered aldehyde group and the ring is
        # monosubstituted.
        ald_c = _aldehyde_carbon_for_label(original, atom)
        if ald_c is not None:
            ring_nb = next((nb for nb in ald_c.GetNeighbors()
                            if nb.GetSymbol() == "C" and nb.GetIsAromatic()), None)
            if ring_nb is not None:
                ring = _aromatic_benzene(original, ring_nb.GetIdx())
                if ring and _ring_monosubstituted(original, ring):
                    return "benzenecarbaldehyde"

        if atom.GetSymbol() != "C":
            continue
        nbrs = list(atom.GetNeighbors())

        # (1) benzene nitrile: labelled C triple-bonded to N and single-bonded
        # to an aromatic benzene carbon -> benzene(13C)carbonitrile (the Blue Book).
        nit_n = ar_c = None
        others = 0
        for nb in nbrs:
            b = original.GetBondBetweenAtoms(idx, nb.GetIdx())
            if nb.GetSymbol() == "N" and b.GetBondType() == Chem.BondType.TRIPLE:
                nit_n = nb
            elif nb.GetSymbol() == "C" and nb.GetIsAromatic():
                ar_c = nb
            else:
                others += 1
        if nit_n is not None and ar_c is not None and others == 0:
            ring = _aromatic_benzene(original, ar_c.GetIdx())
            if ring and _ring_monosubstituted(original, ring):
                return "benzenecarbonitrile"

        # (2)/(4) carboxyl carbon: on a benzene ring -> benzene(13C)carboxylic
        # acid (the Blue Book); on a cyclohexane gem-diacid ring carbon ->
        # 1-carboxycyclohexane-1-carboxylic acid.
        if _is_carboxyl_carbon(original, atom):
            ring_nb = next((nb for nb in nbrs if nb.GetSymbol() == "C"), None)
            if ring_nb is not None and ring_nb.GetIsAromatic():
                ring = _aromatic_benzene(original, ring_nb.GetIdx())
                if ring and _ring_monosubstituted(original, ring):
                    return "benzenecarboxylic acid"
            elif ring_nb is not None:
                ring = _saturated_carbocycle(original, ring_nb.GetIdx(), 6)
                gem = sum(1 for nb in ring_nb.GetNeighbors()
                          if _is_carboxyl_carbon(original, nb))
                if ring and gem == 2:
                    return "1-carboxycyclohexane-1-carboxylic acid"

        # (3) benzyl: labelled CH2 bonded to a monosubstituted benzene carbon and
        # exactly one non-aromatic heavy atom -> respell the retained substituent
        # benzyl -> (phenylmethyl), giving [phenyl(13C)methyl]... (the Blue Book). A
        # re-anchored (structurally gated), OPSIN-RT-audited respell, not a blind
        # rewrite: the negative lookahead keeps benzyloxy/benzylidene/... intact.
        if retained_name and atom.GetTotalNumHs() == 2 and atom.GetDegree() == 2:
            ar = [nb for nb in nbrs if nb.GetSymbol() == "C" and nb.GetIsAromatic()]
            nonar = [nb for nb in nbrs
                     if not (nb.GetSymbol() == "C" and nb.GetIsAromatic())]
            if len(ar) == 1 and len(nonar) == 1:
                ring = _aromatic_benzene(original, ar[0].GetIdx())
                if ring and _ring_monosubstituted(original, ring):
                    respelled = re.sub(
                        r"benzyl(?!oxy|idene|sulfan|amino|thio|carbon)",
                        "(phenylmethyl)", retained_name, count=1)
                    if respelled != retained_name:
                        return respelled
    return None


_MULT_TO_COUNT = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6}
_COUNT_TO_MULT = {v: k for k, v in _MULT_TO_COUNT.items()}

#: A repeated IDENTICAL letter locant (``N``, ``O``,...) followed by one of
#: the simple multiplier stems, e.g. the ``N,N,N-tri`` of
#: ``N,N,N-trimethylethanaminium``. Anchored to start-of-string or a
#: preceding hyphen so it cannot match mid-word.
_LETTER_MULT_RE = re.compile(
    r"(?:^|-)(?P<letter>[A-Z])(?:,(?P=letter))+-(?P<mult>di|tri|tetra|penta|hexa)"
)


def _decorate_letter_multiplier(skeleton, keys, original, stripped,
                                stereo_blind: bool = False) -> Optional[str]:
    """ de-multiplication for a REPEATED-LETTER-LOCANT substituent
    cluster (e.g. ``N,N,N-trimethyl``), found ANYWHERE in the skeleton -- not
    only a leading multiplier at the very front of the whole name (that
    narrower shape, with a NUMERIC locant list, is:func:`_decorate_demultiplied`).

    A letter locant (``N``, ``O``,...) cites several substituents on the
    SAME heteroatom; unlike a numeric gem-locant pair there is no alternative
    numbered position to disambiguate, so the copies are chemically
    INTERCHANGEABLE before labelling -- splitting off any ONE of them for the
    label yields the unique labelled structure, and no permutation search
    (unlike:func:`_decorate_demultiplied`'s distinct-numeric-position search)
    is needed.

    Scoped to a single labelled copy of a single nuclide (the verified
    witness, 11C-choline's ``N,N,N-trimethyl`` -> one 11C-methyl); every
    candidate is still OPSIN-RT gated against the true original AND the
    intermediate re-worded-but-unlabelled split is gated against the
    STRIPPED skeleton (to find the correct substituent-word boundary), so a
    wrong split point or a wrong final structure can only be discarded, never
    shipped.
    """
    if len(keys) != 1:
        return None
    (mass, el), count, maxpos = keys[0]
    if count != 1:
        return None
    for m in _LETTER_MULT_RE.finditer(skeleton):
        letter = m.group("letter")
        mult_count = _MULT_TO_COUNT[m.group("mult")]
        prefix = skeleton[: m.start("letter")]
        after = skeleton[m.end():]
        remaining = mult_count - 1
        for k in range(1, len(after)):
            base, tail = after[:k], after[k:]
            if remaining == 0:
                bare_seg = ""
            elif remaining == 1:
                bare_seg = f"{letter}-{base}-"
            else:
                bare_seg = (
                    ",".join([letter] * remaining) + "-"
                    + _COUNT_TO_MULT[remaining] + base + "-"
                )
            unlabelled_split = f"{prefix}{bare_seg}{letter}-{base}{tail}"
            if not _isotope_round_trips(unlabelled_split, stripped):
                continue
            # Locant None: the descriptor sits immediately after the "N-"
            # substituent citation already spliced into ``candidate`` below,
            # so 's "before the part... substituted" scope is already
            # unambiguous without also repeating the letter INSIDE the
            # descriptor itself (that would be a different, unverified
            # spelling -- e.g. ``(N-11C1)`` -- not the one this function is
            # built to produce).
            desc = format_isotope_descriptor([(None, mass, el, 1, maxpos)])
            # (the Blue Book): the labelled copy is cited before
            # the unlabelled ones ("The isotopically modified substituent is
            # preferred alphabetically to the unmodified substituent").
            candidate = (f"{prefix}{letter}-{desc}{base}"
                         f"{('-' + bare_seg[:-1]) if bare_seg else ''}{tail}")
            # ``stereo_blind`` caller): compare CONSTITUTION only so the
            # stereodescriptor-less split reproduces the connectivity; the caller
            # re-gates the stereodescriptor-prefixed candidate full-InChIKey. The
            # intermediate ``unlabelled_split`` probe stays FULL (it locates the
            # substituent-word boundary against the already-achiral ``stripped``).
            if _isotope_round_trips(candidate, original, stereo_blind=stereo_blind):
                return candidate
    return None


def _decorate_demultiplied(skeleton, keys, original, stripped,
                           stereo_blind: bool = False) -> Optional[str]:
    """Generalized de-multiplication (single OR mixed nuclides).

    Split a leading-locant simple multiplier (e.g. ``1,2-dimethoxyethane``) and
    scope a DISTINCT nuclide descriptor to each labeled copy, leaving at most one
    copy bare. Handles both:
      * single-group — one labeled copy among bare copies
                         +, e.g. 1-(13C1)methoxy-2-methoxyethane)
      * mixed — DISTINCT nuclides on DISTINCT copies, where — because the
                        parent is symmetric — BOTH numbering directions round-trip
                        and /.4.3 must choose (e.g.
                        1-(18O1)methoxy-2-(13C1)methoxyethane: 18O Z=8 > 13C Z=6).

    Every candidate is OPSIN-RT gated with isotopes retained; among the
    round-trippers, (lowest locants to the modified copies) then
    /.4.3 (higher atomic number then higher mass number at the lower
    locant) selects deterministically. The single-combined-descriptor enumeration
    in decorate_isotopic_name CANNOT place distinct nuclides on distinct copies,
    and would exhaust its JVM-per-candidate search fruitlessly on a de-mult
    skeleton, so this runs FIRST and that enumeration is the fall-through.

    Fail-closed (None) — never a wrong OR non-PIN separated name — when: the
    skeleton is not a simple-multiplier form; locants repeat (gem copies, which a
    locant-keyed map cannot tell apart); more than one copy is left bare or a
    nuclide repeats across copies (the unmodified/identical copies must stay
    grouped under a reduced multiplier — a DIFFERENT name shape, not this split,
    e.g. `3,5-dimethyl` not `3-methyl-5-methyl`; NOT built here); the candidate
    budget would blow up; or nothing round-trips. This is a DELIBERATE PIN-safe
    scope, not a superset of the single-group-only fallback it replaces (that
    fallback exploded every copy — non-PIN for >1 bare copy).
    """
    import re
    from itertools import combinations, permutations
    from math import comb, factorial

    m = re.match(
        r"^(?P<locs>\d+(?:,\d+)+)-(?P<mult>di|tri|tetra|penta|hexa)(?P<tail>[a-z].+)$",
        skeleton,
    )
    if not m:
        return None
    mult_count = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6}[m.group("mult")]
    locs = sorted(int(x) for x in m.group("locs").split(","))
    if len(locs) != mult_count:
        return None
    if len(set(locs)) != len(locs):
        # Repeated locants (gem-disubstitution, e.g. 1,1,2-...): a locant-keyed
        # copy map (loc_to_label below) cannot distinguish two copies sharing a
        # locant, so a single labeled gem-copy is not constructible -> fail closed.
        return None
    tail = m.group("tail")

    # One nuclide instance per labeled copy (count 1 each). ``maxpos_of`` carries
    # the polysubstitution quantity per nuclide out to the descriptor
    # build below WITHOUT entering ``labels`` -- the set-distinctness / repeated-
    # nuclide guards must key on (mass, el) alone (two copies of one nuclide could
    # differ in max_at_pos yet must still be treated as identical here).
    maxpos_of = {(mass, el): maxpos for (mass, el), count, maxpos in keys}
    labels = []
    for (mass, el), count, maxpos in keys:
        labels.extend([(mass, el)] * count)
    # Scope (PIN-safe de-multiplication): every labeled copy carries a DISTINCT
    # nuclide and AT MOST ONE copy is left bare. Then no two copies share a
    # descriptor, so nothing needs re-multiplying and the split is a genuine PIN
    # de-multiplication. Repeated identical labels / >1 bare copy would require
    # re-collecting copies under a multiplier (a different name shape) -> fail
    # closed rather than emit a non-PIN separated form.
    if (not labels
            or len(set(labels)) != len(labels)
            or (mult_count - len(labels)) not in (0, 1)):
        return None
    # Cost cap (bounded oracle budget): the assignment search is
    # C(mult_count, len(labels)) * len(labels)! candidates, EACH OPSIN-RT gated
    # via a fresh JVM (~1.5-2s). Reject the combinatorial blow-up of high
    # multipliers with many distinct nuclides (penta/hexa reach 720) -> fail
    # closed rather than spend minutes; 24 covers di/tri/tetra, the verified
    # target being di.
    if comb(mult_count, len(labels)) * factorial(len(labels)) > 24:
        return None

    # Find the substituent/parent split point(s) against the UNLABELED skeleton
    # FIRST (bounded: <= len(tail) probes). A non-de-mult skeleton or a
    # parent-labeled skeleton exits here cheaply instead of exhausting the oracle.
    valid_ks = []
    for k in range(1, len(tail)):
        base = tail[:k]
        bare = "-".join(f"{L}-{base}" for L in locs) + tail[k:]
        if _isotope_round_trips(bare, stripped):
            valid_ks.append(k)
    if not valid_ks:
        return None

    winners = []  # (p4541_key, p4542_key, k, candidate)
    # labels are all distinct (guard above), so permutations yields no duplicates
    # and need not be de-duped; iteration order does not affect the RESULT because
    # the final sort key ends in the candidate string (see the sort comment).
    for chosen in combinations(locs, len(labels)):
        for perm in permutations(labels):
            loc_to_label = dict(zip(chosen, perm))
            for k in valid_ks:
                base, parent = tail[:k], tail[k:]
                # (the Blue Book): "The isotopically modified
                # substituent is preferred alphabetically to the unmodified
                # substituent": the labelled copies are cited first ('N-[7-(131I)
                # iodo-6-iodo-9H-fluoren-2-yl]acetamide (PIN)',:43762).
                lab_segs, bare_segs = [], []
                for L in locs:
                    if L in loc_to_label:
                        mass, el = loc_to_label[L]
                        desc = format_isotope_descriptor(
                            [(None, mass, el, 1, maxpos_of[(mass, el)])])
                        lab_segs.append(f"{L}-{desc}{base}")
                    else:
                        bare_segs.append(f"{L}-{base}")
                cand = "-".join(lab_segs + bare_segs) + parent
                # ``stereo_blind`` caller): compare CONSTITUTION only so a
                # de-multiplied constitution that lacks the stereodescriptor still
                # passes; the caller re-gates the stereo-prefixed candidate on the
                # full InChIKey. The ``bare`` vs ``stripped`` probe above stays FULL
                # (``stripped`` is achiral -- it only fixes the word boundary).
                if _isotope_round_trips(cand, original, stereo_blind=stereo_blind):
                    groups = [(L, loc_to_label[L][0], loc_to_label[L][1], 1,
                               maxpos_of[loc_to_label[L]])
                              for L in sorted(loc_to_label)]
                    p4541 = tuple(sorted(loc_to_label))
                    winners.append((p4541, _p4542_p4543_key(groups), k, cand))
    if not winners:
        return None
    # (lowest locants to modified copies) then /.4.3; the
    # trailing candidate STRING term (w[3]) makes the winner fully deterministic
    # regardless of assignment-iteration order -- it is load-bearing, not a
    # decorative tiebreak. Do not drop it.
    winners.sort(key=lambda w: (w[0], w[1], w[2], w[3]))
    return winners[0][3]


def _decorate_reduced_multiplier(skeleton, keys, original, stripped,
                                 stereo_blind: bool = False) -> Optional[str]:
    """ de-multiplication that RE-COLLECTS the unlabelled copies under a
    REDUCED multiplier — the "different name shape":func:`_decorate_demultiplied`
    declines by design.

    :func:`_decorate_demultiplied` splits a leading-locant simple multiplier
    (``1,2-dimethoxyethane``) and leaves AT MOST ONE copy bare. When TWO OR MORE
    copies stay unlabelled they must be regrouped under a reduced multiplier, e.g.
    ``1,3,5-trimethylbenzene`` with ONE ¹³C methyl → ``1-(13C)methyl-3,5-dimethyl-
    benzene``.

    Numbering — "Priority between isotopically substituted and unmodified
    atoms or groups" (``the Blue Book Blue Book``): "the starting point and
    the direction of numbering of the analogous isotopically substituted compound
    are chosen so as to give lowest locants to the modified atoms or groups
    considered together in one series in increasing numerical order." Verbatim
    example ``(2-14C)butane (PIN) [not (3-14C)butane]``. So among the numberings the
    round-trip oracle accepts, the LABELLED copy takes the LOWEST locant of the
    parent's fixed locant set — NOT the highest. On a NON-symmetric parent only the
    physically-correct assignment round-trips, so 's fixed numbering
    (``:44149``, "Numbering... is not changed from that of... the unmodified
    compound") is honoured automatically.

    Citation order — (``:43758``): "The isotopically modified
    substituent is preferred alphabetically to the unmodified substituent", so the
    labelled segment is cited first whatever its locant ('N-[7-(131I)iodo-6-iodo-
    9H-fluoren-2-yl]acetamide (PIN)', ``:43762``).

    Scope (bounded, PIN-safe, isotope-path only): ONE labelled copy of ONE nuclide
    with TWO OR MORE bare copies. Every candidate is OPSIN-RT gated (0-wrong);
    anything outside this scope, or that does not round-trip, fails closed to None
    (→ the row keeps abstaining, never a wrong or non-PIN name). Repeated locants
    (gem copies a locant-keyed split cannot distinguish) also fail closed. The
    0-bare / 1-bare shapes belong to:func:`_decorate_demultiplied` and the
    all-labelled uniform shape to:func:`_decorate_uniform_multiplier`; both run
    ahead of this so their behaviour is unchanged.
    """
    import re
    m = re.match(
        r"^(?P<locs>\d+(?:,\d+)+)-(?P<mult>di|tri|tetra|penta|hexa)(?P<tail>[a-z].+)$",
        skeleton,
    )
    if not m:
        return None
    mult_count = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6}[m.group("mult")]
    locs = sorted(int(x) for x in m.group("locs").split(","))
    if len(locs) != mult_count or len(set(locs)) != len(locs):
        # A wrong count, or gem/repeated locants a locant-keyed copy map cannot
        # tell apart -> fail closed.
        return None
    tail = m.group("tail")

    # Exactly ONE labelled copy of ONE nuclide; the rest (>= 2) stay bare.
    labels = []
    maxpos_of = {(mass, el): maxpos for (mass, el), count, maxpos in keys}
    for (mass, el), count, maxpos in keys:
        labels.extend([(mass, el)] * count)
    if len(labels) != 1:
        return None
    n_bare = mult_count - len(labels)
    if n_bare < 2:
        # 0/1 bare copies are _decorate_demultiplied's job (runs first).
        return None
    (mass, el) = labels[0]
    desc = format_isotope_descriptor([(None, mass, el, 1, maxpos_of[(mass, el)])])

    # Word boundary (substituent stem | parent) probed against the UNLABELLED
    # skeleton first: a non-de-mult / parent-labelled skeleton exits cheaply.
    valid_ks = []
    for k in range(1, len(tail)):
        base = tail[:k]
        bare = "-".join(f"{L}-{base}" for L in locs) + tail[k:]
        if _isotope_round_trips(bare, stripped):
            valid_ks.append(k)
    if not valid_ks:
        return None

    _RED = {2: "di", 3: "tri", 4: "tetra", 5: "penta"}

    def _assemble(lab_loc: int, k: int) -> str:
        base, parent = tail[:k], tail[k:]
        bare_locs = [L for L in locs if L != lab_loc]
        lab_seg = f"{lab_loc}-{desc}{base}"
        bare_seg = f"{','.join(str(L) for L in bare_locs)}-{_RED[len(bare_locs)]}{base}"
        # (the Blue Book): "The isotopically modified substituent
        # is preferred alphabetically to the unmodified substituent", whatever the
        # locants ('N-[7-(131I)iodo-6-iodo-9H-fluoren-2-yl]acetamide (PIN)',:43762).
        return f"{lab_seg}-{bare_seg}{parent}"

    # Enumerate the labelled copy over the parent's fixed locant set; RT-gate each;
    # keeps the lowest labelled locant. The trailing candidate string makes
    # the winner fully deterministic regardless of iteration order.
    winners = []  # (lab_loc, k, candidate)
    for lab_loc in locs:
        for k in valid_ks:
            cand = _assemble(lab_loc, k)
            if _isotope_round_trips(cand, original, stereo_blind=stereo_blind):
                winners.append((lab_loc, k, cand))
    if not winners:
        return None
    winners.sort(key=lambda w: (w[0], w[1], w[2]))
    return winners[0][2]


#: One locant token of a multiplied prefix group: a numeral with an optional
#: letter and primes ('2', '4a', "4'") or an italic element locant with an
#: optional superscript numeral and primes ('N', 'N2', "N'", 'O').
_GROUP_LOCANT = r"(?:\d+[a-z]?'*|[A-Z][a-z]?\d*'*)"
#: A multiplied simple prefix group anywhere in a name: its locant set, a basic
#: multiplying prefix and the prefix word ('2,7-diphenyl', 'N2,N4-dimethyl',
#: "N,N'-dimethyl", 'N,N-dimethyl', '6,7-diiodo' inside a substituent).
_MULT_GROUP_RE = re.compile(
    rf"(?:^|(?<=[-(\[{{]))(?P<locs>{_GROUP_LOCANT}(?:,{_GROUP_LOCANT})+)-"
    r"(?P<mult>di|tri|tetra|penta|hexa)(?P<word>[a-z]+)")


def _group_locant_key(tok: str) -> tuple:
    """ (the Blue Book) order of the locants of one group: "Italic
    capital and lower-case letter locants are lower than... numerals"; primes
    follow the unprimed locant."""
    primes = tok.count("'")
    core = tok.replace("'", "")
    if core[:1].isdigit():
        num = int(re.match(r"\d+", core).group())
        return (1, "", num, core[len(str(num)):], primes)
    m = re.match(r"([A-Z][a-z]?)(\d*)", core)
    return (0, m.group(1), int(m.group(2) or 0), "", primes)


def _decorate_one_labelled_copy(skeleton, keys, original, stripped,
                                stereo_blind: bool = False) -> Optional[str]:
    """: a multiplied prefix group ONE of whose copies carries every
    label is cited as two groups, the labelled copy first.

    "When two substituent groups are isotopically modified in different ways so
    that they cannot be combined together using multiplicative terms such as
    'di-', 'bis-', etc., they are cited separately. The isotopically modified
    substituent is preferred alphabetically to the unmodified substituent."
    , the Blue Book; '2-(13C)methyl-3-methylpyridine (PIN)',
    :43764; 'N-[7-(131I)iodo-6-iodo-9H-fluoren-2-yl]acetamide (PIN)',:43762.)

    The splitters above take a group at the front of the name with numeral
    locants and one nuclide atom per copy. This one takes the group anywhere
    (inside a substituent's marks too), with numeral or italic locants
    ('N2,N4-dimethyl', "N,N'-dimethyl", 'N,N-dimethyl') and any number of labels
    on the one copy ('(2H5)phenyl', '(2H3)methyl'). The copy's descriptor carries
    no locant, which is licensed only when the copy is completely labelled
    ,:func:`_unit_completely_labelled`); otherwise it declines. The
    other copies keep a reduced multiplier. Among the numberings OPSIN reads back,
    the labelled copy takes the lowest locant,:44172). Every candidate
    is OPSIN-RT gated against the input (0-wrong); nothing that does not read back
    is returned."""
    from ..assembly.naming_utils import _SIMPLE_PREFIX_VOCABULARY
    desc_groups = [(None, mass, el, count, maxpos)
                   for (mass, el), count, maxpos in keys]
    desc = format_isotope_descriptor(desc_groups)
    winners = []
    for m in _MULT_GROUP_RE.finditer(skeleton):
        locs = m.group("locs").split(",")
        n = _MULT_TO_COUNT[m.group("mult")]
        if len(locs) != n:
            continue
        word = m.group("word")
        bases = [word[:k] for k in range(len(word), 0, -1)
                 if word[:k] in _SIMPLE_PREFIX_VOCABULARY]
        if not bases:
            continue
        base = bases[0]
        pre, rest = skeleton[:m.start()], skeleton[m.start("word") + len(base):]
        if not _isotope_round_trips(
                pre + "-".join(f"{L}-{base}" for L in locs) + rest, stripped):
            continue                     # the word boundary is not 'base'
        for i, lab in enumerate(locs):
            others = locs[:i] + locs[i + 1:]
            bare = (f"{others[0]}-{base}" if len(others) == 1 else
                    f"{','.join(others)}-{_COUNT_TO_MULT[len(others)]}{base}")
            head = pre + f"{lab}-"
            off = len(head)
            plain = head + f"{base}-{bare}{rest}"
            if not _unit_completely_labelled(plain, off, keys, original):
                continue
            cand = _escalate_marks_for_descriptor(
                head + f"{desc}{base}-{bare}{rest}", off)
            if _isotope_round_trips(cand, original, stereo_blind=stereo_blind):
                winners.append((_group_locant_key(lab), cand))
        if winners:
            break
    if not winners:
        return None
    winners.sort()
    return winners[0][1]


def _decorate_uniform_multiplier(skeleton, keys, original, stripped,
                                 stereo_blind: bool = False) -> Optional[str]:
    """ + — a UNIFORM multiplied substituent: every copy of a
    de-multipliable prefix carries the IDENTICAL nuclide descriptor.

    ``_decorate_demultiplied`` deliberately DECLINES the repeated-identical-nuclide
    case (its docstring: identical copies must stay GROUPED under the multiplier, a
    DIFFERENT name shape). This builds that shape. It keeps the ``di``/``tri``/...
    multiplier and scopes ONE descriptor to the repeated substituent, then steps the
    enclosing marks up ```` -> ```` because the substituent already carries the
    descriptor's parentheses, the Blue Book; verbatim example the Blue Book
    ``1,2-di[(13C)methyl]benzene``). The step-up reuses the shared
    ``apply_enclosing_marks`` primitive -- no string post-processing.

    ⚠ SCOPE — NON-hydrogen nuclides ONLY. ``apply_enclosing_marks`` routes a leading
    ``(2H)``/``(3H)`` through ``_INDICATED_H_RE`` (naming_utils.py:1127) and returns
    ``((2H)methyl)`` (parentheses, NO step-up) instead of ``[(2H)methyl]``; OPSIN-RT
    is blind to that bracket-level spelling error and would NOT reject it. A
    deuterium/tritium uniform multiplier therefore fails closed here and falls
    through to the enumeration path (an RT-valid, non-bracket-stepped spelling),
    never shipping a wrong bracket.

    Every candidate is OPSIN-RT gated (isotopes retained), so a wrong split fails
    closed. Fail-closed (None) when the skeleton is not a leading-locant simple
    multiplier, the labels are not ONE uniform non-hydrogen nuclide with exactly one
    atom per copy (``count == mult_count``), or nothing round-trips.
    """
    from ..assembly.naming_utils import apply_enclosing_marks

    m = re.match(
        r"^(?P<locs>\d+(?:,\d+)+)-(?P<mult>di|tri|tetra|penta|hexa)(?P<tail>[a-z].+)$",
        skeleton,
    )
    if not m:
        return None
    mult_count = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6}[m.group("mult")]
    locs = sorted(int(x) for x in m.group("locs").split(","))
    if len(locs) != mult_count or len(set(locs)) != len(locs):
        # gem/repeated locants: not the distinct-copy uniform shape -> fail closed.
        return None
    tail = m.group("tail")

    # UNIFORM scope: exactly one distinct nuclide, one atom on EACH of the
    # mult_count copies (count == mult_count), and NOT hydrogen (bracket hazard
    # above). A single labeled copy among bare ones, or count != mult_count, is a
    # different (or non-uniform) shape -> fail closed.
    if len(keys) != 1:
        return None
    (mass, el), count, maxpos = keys[0]
    if el == "H" or count != mult_count:
        return None

    # One descriptor for the repeated copy (count 1 per copy). ``maxpos`` decides
    # the subscript (FIX-A): a carbon position holds one carbon -> ``(13C)``.
    desc = format_isotope_descriptor([(None, mass, el, 1, maxpos)])

    # Split ``tail`` into the repeated substituent (``methyl``) and the parent
    # (``benzene``). The grammar is opaque to this module, so every split is offered
    # and the OPSIN round-trip oracle keeps only the one that reproduces the mol
    # descriptor scope). ``apply_enclosing_marks(..., -1)`` steps ->
    #. Deterministic: lowest split index, then candidate string.
    winners = []
    for k in range(1, len(tail)):
        base, parent = tail[:k], tail[k:]
        sub = apply_enclosing_marks(f"{desc}{base}", -1)
        cand = f"{m.group('locs')}-{m.group('mult')}{sub}{parent}"
        # ``stereo_blind`` caller): constitution-only gate; the caller
        # re-gates the stereo-prefixed candidate full-InChIKey (0-wrong preserved).
        if _isotope_round_trips(cand, original, stereo_blind=stereo_blind):
            winners.append((k, cand))
    if not winners:
        return None
    winners.sort()
    return winners[0][1]


# ---------------------------------------------------------------------------
# (BB:~43855) -- isotope-INDUCED stereocentre
# ---------------------------------------------------------------------------
def _cip_labeled_count(mol: Chem.Mol) -> int:
    """Number of atoms carrying an assigned CIP label under the project's
    canonical stereo path (``perception.stereo.assign_stereochemistry`` -- the
    ``centres`` engine reads isotope mass as CIP Rule 1b, confirmed by spike).

    The mol is RE-PARSED from its canonical SMILES first: an isotope-stripped copy
    keeps the ORIGINAL's chiral tag on a carbon that is no longer a stereocentre
    (two H's now identical), and the canonical writer drops exactly those phantom
    tags -- so the difference original-minus-stripped counts only the centres the
    nuclide actually induces. Fail-soft to 0 (a missing JVM must never hard-fail a
    name)."""
    try:
        m = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if m is None:
            return 0
        from ..perception.stereo import assign_stereochemistry
        assign_stereochemistry(m)
    except Exception:
        return 0
    return sum(1 for a in m.GetAtoms() if a.HasProp("_CIPCode"))


_MIXED_LOCANT_COMBO_CAP = 256  # bound on the per-group locant-spec product; fail closed above


def _decorate_mixed_locant(skeleton: str, keys, original: Chem.Mol,
                           n_pos: int, stripped_smiles: str,
                           namer) -> Optional[str]:
    """ / mixed descriptor: a multi-nuclide label whose
    groups take DIFFERENT locant policies. Two shapes neither prior path builds:

      * ``C[13CH2][15NH2]`` -> ``(1-13C,15N)`` — ¹³C ambiguous among the two
        carbons (cites), ¹⁵N the sole nitrogen, omits);
      * ``[2H]c1cc[15n]c([2H])c1`` -> ``(2,4-2H2,15N)pyridine`` — the two D sit at
        DISTINCT ring positions 2 and 4, so the H group needs the multi-locant
        token ``2,4-2H2`` (which ``_find_best_placement`` cannot build — it shares
        ONE locant across a whole group), while the sole ring N omits its locant.

    Per nuclide GROUP the locant policy is decided by the omission
    predicate (:func:`_element_singleton_in`, heading "Locants are omitted when
    there is only one atom of a given element", ``the Blue Book``; verbatim PIN
    ``(2,4-2H2,15N)pyridine`` ``:44194``): a group whose element occurs once in the
    parent omits its locant even inside a combined descriptor whose other groups
    cite. Every non-singleton group CITES; its locant(s) are found by the OPSIN
    round-trip oracle — a single locant for count 1, and a candidate SET of
    ``count`` distinct (or one shared) positions for count>1.
    :func:`format_isotope_descriptor` renders each group's ``locant`` None /
    scalar / distinct-tuple and orders the groups alphabetically by element then
    mass, so ``(2,4-2H2,15N)`` falls out once each group's locant is set.

    Every candidate stays OPSIN-RT gated (0-wrong). Candidate locant SETS are
    tried lowest-first and, within the first set that round-trips, the
     adjacent-to-affix placement is preferred over a front-detached one
    (:func:`_placement_quality`), exactly as:func:`_find_best_placement`.
    Bounded / fail-closed on a large product.

    Isolated fallback: only reached from the ``best is None`` cascade AFTER the
    single-descriptor and distinct-multi-locant paths return None, and only for a
    MULTI-nuclide descriptor, so it cannot change any row those paths already
    name."""
    import itertools
    if len(keys) < 2:
        return None
    # A mixed descriptor ALWAYS cites at least one locant, so forces
    # the parent to restore its own omitted locants too (…ethan-1-amine, not
    # …ethanamine). Re-name the skeleton inside the forced-locant scope, as the
    # single-descriptor path does at the restoration block below; keep
    # the free-elided skeleton as a fallback if the re-render fails.
    with _isotope_locant_scopes(original):
        skel_forced = _flagged_systematic_namer(namer).name(stripped_smiles)
    if skel_forced and "unknown" not in skel_forced.lower():
        skeleton = skel_forced

    # Per-group candidate locant-specs. A spec is None omit) or a tuple
    # of ``count`` locants (ascending); a count-1 cite group is a 1-tuple.
    n = min(n_pos, _MULTI_LOCANT_SWEEP_CAP)
    positions = list(range(1, n + 1))
    per_group_specs = []
    for (mass, el), count, _maxpos in keys:
        if _element_singleton_in(original, el):
            per_group_specs.append([None])              #: omit locant
        elif count == 1:
            per_group_specs.append([(L,) for L in positions])
        else:
            # distinct positions first (2,4-2H2), then one shared position
            # (2,2-2H2) as a fallback for a genuinely co-located group.
            specs = [tuple(c) for c in itertools.combinations(positions, count)]
            specs += [(L,) * count for L in positions]
            per_group_specs.append(specs)

    # A genuinely MIXED descriptor needs at least one cite group; an all-omit form
    # is already covered by _find_best_placement's locant-free path.
    if all(s == [None] for s in per_group_specs):
        return None
    # Bound the product (fail closed rather than spend minutes on a large sweep).
    total = 1
    for s in per_group_specs:
        total *= len(s)
        if total > _MIXED_LOCANT_COMBO_CAP:
            return None

    seen_off: set = set()
    offs = [o for o in ([0] + _insertion_offsets(skeleton))
            if not (o in seen_off or seen_off.add(o))]

    def _combo_cited(combo):
        return sorted(L for spec in combo if spec is not None for L in spec)

    #: try candidate locant SETS lowest-first; the first set that yields
    # ANY round-tripper is the answer, and within it the best-placement (lowest
    # placement_quality, then lowest offset) candidate is emitted.
    combos = sorted(itertools.product(*per_group_specs), key=_combo_cited)
    for combo in combos:
        groups_desc = [(loc, mass, el, count, maxpos)
                       for loc, ((mass, el), count, maxpos) in zip(combo, keys)]
        # Subscript axis (as in _find_best_placement): the BB-preferred omitted-
        # subscript form (sub_rank 0, `(2,4-2H2,15N)`) is tried first; the
        # force-show form (sub_rank 1, `(2,4-2H2,15N1)`) is a parseability fallback.
        desc_pref = format_isotope_descriptor(groups_desc)
        desc_force = format_isotope_descriptor(groups_desc, force_show=True)
        variants = [(0, desc_pref)]
        if desc_force != desc_pref:
            variants.append((1, desc_force))
        winners = []
        for sub_rank, desc in variants:
            for off in offs:
                before_digit = off < len(skeleton) and skeleton[off:off + 1].isdigit()
                for join in (["-", ""] if before_digit else [""]):
                    cand = skeleton[:off] + desc + join + skeleton[off:]
                    if _isotope_round_trips(cand, original):
                        winners.append(
                            (sub_rank,
                             _placement_quality(skeleton, off, locant_free=False),
                             off, _escalate_marks_for_descriptor(cand, off)))
        if winners:
            winners.sort(key=lambda w: (w[0], w[1], w[2], w[3]))
            if winners[0][0] == 1:
                # The always-show subscript is the parse fallback, not the
                # preferred spelling (see _find_best_placement).
                from ..metrics.provenance import record_non_pin_fragment
                record_non_pin_fragment(winners[0][3])
                record_non_pin_fragment(_descriptor_fragment(winners[0][3], desc_force))
            return winners[0][3]
    return None


_STEREO_TOKEN_RE = re.compile(r"(\d*)([a-z]?)('{0,2})([RSrsEZez])$")


def _stereo_token_key(tok: str):
    """Total order over stereodescriptor tokens by ASCENDING locant
    'cited according to the ascending order of their corresponding locants').
    Key: (int locant, letter suffix e.g. 'a' in 7a, prime count, descriptor)."""
    m = _STEREO_TOKEN_RE.match(tok)
    if not m:
        return (float("inf"), "", 0, tok)
    num, letter, primes, desc = m.groups()
    return (int(num) if num else 0, letter, len(primes), desc)


def _merge_leading_stereo(induced, base: str) -> str:
    """: merge the isotope-INDUCED stereodescriptors with any genuine
    stereodescriptor group already leading ``base`` into ONE parenthesised group,
    cited in ascending locant order — ``(1S)-`` + ``(2R)-…`` -> ``(1S,2R)-…``.
    ``induced`` is a list of ``(int_locant, 'R'|'S')``. When ``base`` carries no
    leading stereo group the result is byte-identical to the old ``prefix+base``,
    so single-induced rows (``(1R)-(1-2H1)ethan-1-ol``) are unchanged. The isotope
    descriptor ``(1-2H1)`` is NOT a stereodescriptor and never matches
    ``_STEREO_PREFIX_RE``, so it is never swept into the merge."""
    from .stereochemistry import _STEREO_PREFIX_RE  # lazy: avoid import cycle
    m = _STEREO_PREFIX_RE.match(base)
    base_tokens, rest = [], base
    if m:
        base_tokens = m.group(0)[1:-2].split(",")  # strip '(' and ')-'
        rest = base[m.end():]
    induced_tokens = [f"{loc}{cfg}" for loc, cfg in induced]
    merged = sorted(base_tokens + induced_tokens, key=_stereo_token_key)
    return "(" + ",".join(merged) + ")-" + rest


def _decorate_isotope_stereo(skeleton, keys, original, stripped, n_pos,
                             extra_locants) -> Optional[str]:
    """: an isotope-INDUCED stereocentre -- a centre that is a stereocentre
    only because a nuclide breaks a local symmetry (``(1R)-(1-2H1)ethan-1-ol``,
    achiral once the D is collapsed). The skeleton is named from the STRIPPED mol
    and carries no such descriptor, and the isotope descriptor alone reproduces
    the constitution but not the stereo, so the full-InChIKey oracle rejects it
    and the row abstains.

    Fully oracle-driven (no atom->locant map needed, consistent with the rest of
    this module):
      1. find the isotope-descriptor placement reproducing the CONSTITUTION
         (``stereo_blind`` -- e.g. ``(1-2H1)ethan-1-ol``);
      2. enumerate stereodescriptor prefixes over (locant-subset x R/S) for the
         ``k`` induced centres, cost-capped;
      3. accept the first (lowest-locant, deterministic) prefixed candidate whose
         FULL InChIKey (stereo + isotope) reproduces the original.

    : "the stereodescriptors are cited first" -- the prefix leads the name.
    Every emission is FULL-oracle gated, so a wrong config/locant fails closed;
    fails closed (None) when there is no induced centre, no constitution placement,
    the enumeration would exceed the cap, or nothing round-trips.
    """
    k = _cip_labeled_count(original) - _cip_labeled_count(stripped)
    if k < 1:
        return None
    base, _off, _desc, _lr = _find_best_placement(
        skeleton, keys, original, n_pos, allow_front_hyphen=True,
        extra_locants=extra_locants, stereo_blind=True)
    if base is None:
        # The single-descriptor placement above cannot SPLIT a symmetry-broken
        # multiplied substituent -- e.g. the two iodomethyl arms of
        # 1,3-diiodopropan-2-ol, made distinct by one heavy-iodine nuclide, which
        # is exactly what induces the C2 centre. Obtain that de-multiplied
        # constitution stereo-blind (connectivity only) from the demux helpers so
        # it passes the constitution gate; the stereodescriptor prefix and the
        # full-InChIKey acceptance below still apply, so 0-wrong is preserved.
        for _demux_fn in (_decorate_demultiplied,
                          _decorate_uniform_multiplier,
                          _decorate_letter_multiplier):
            base = _demux_fn(skeleton, keys, original, stripped, stereo_blind=True)
            if base is not None:
                break
    if base is None:
        return None
    from itertools import combinations, product
    from math import comb
    # Cost cap (bounded oracle budget): C(n_pos, k) * 2**k fresh OPSIN parses.
    # 24 covers the single-centre rows (2*n_pos) comfortably; a multi-centre
    # symmetric parent that blows it up fails closed rather than spend minutes.
    if comb(n_pos, k) * (2 ** k) > 24:
        return None
    winners = []
    for locs in combinations(range(1, n_pos + 1), k):
        for cfg in product("RS", repeat=k):
            #: merge the induced descriptors with any genuine
            # stereo group `base` already carries into ONE ascending-locant
            # group (`(1S,2R)-`), not two adjacent `(1S)-(2R)-` groups.
            cand = _merge_leading_stereo(list(zip(locs, cfg)), base)
            if _isotope_round_trips(cand, original):  # FULL stereo + isotope
                winners.append((locs, cfg, cand))
    if not winners:
        return None
    # lowest-locant-first; deterministic tiebreak on the candidate string.
    winners.sort(key=lambda w: (w[0], w[1], w[2]))
    return winners[0][2]


# ─────────────────────────────────────────────────────────────────────────────────────
# WITHDRAWN 2026-07-28: a whole-molecule isotopomer-ambiguity test.
#
# It was built and wired here, and it REGRESSED a correct name, so it is recorded rather
# than kept. It asked "can this nuclide multiset be placed on the skeleton in more than
# one non-isomorphic way?" and forced a locant when so. That ignores ****
# (``:44182``, verbatim): the descriptor is inserted *"before the part of the compound
# that is isotopically substituted"* -- so the descriptor's POSITION carries its scope,
# and ambiguity has to be judged inside that scope, not over the whole molecule.
#
# The witness that refuted it: ``[13CH3]OC(C)=O`` -> ``(13C1)methyl acetate``. Methyl
# acetate has three distinct carbons, so a whole-molecule test says "ambiguous" and forces
# ``(1-13C1)methyl acetate``. But the descriptor sits before ``methyl``, whose scope is a
# single carbon, so the name is already unique. ``:7492`` settles it:
# ``1,2-di[(13C)methyl]benzene (PIN. `` -- descriptor inside the brackets,
# scoped to ``methyl``, no locant on the label.
#
# Judging scope correctly needs awareness of the name's grammar, which this module does
# not have (it enumerates insertion offsets and lets the round-trip oracle choose). So the
# general rule is NOT implemented. What IS implemented below is the half with a verbatim
# Blue Book anchor: when the descriptor itself needs a locant, restores the
# parent's locants. See the open-defect note in
# internal notes
# ─────────────────────────────────────────────────────────────────────────────────────

#: Retained parent-hydride names whose ring / side-chain positions are NOT numbered
#: by the retained spelling and which marks "no substitution" for PINs, so
#: an isotopic modification anywhere on them CANNOT be sited on a legal numbered
#: locant in the retained name. Per (the Blue Book, "When the nuclide is
#: located at a position in a retained name that is not numbered a systematic name
#: that identifies separately the relevant atom is used for the IUPAC preferred
#: name") the systematic parent is the PIN once such a name is isotopically labelled.
#: The worked example (the Blue Book) is decisive: ``1-(13C)methyl(2-13C)benzene
#: (PIN)`` with ``(α,2-13C2)-toluene`` shown only as the GENERAL-nomenclature form.
#: This gate is consulted ONLY by the isotope decorator (a label is always present),
#: so the non-isotope namer that emits plain ``toluene`` is untouched.
_ISOTOPE_NONSUBSTITUTABLE_RETAINED = frozenset({"toluene"})


def decorate_isotopic_name(smiles: str, style: str, namer) -> Optional[str]:
    """Fail-closed isotopic-substitution PIN +.

    1. Parse + strip isotopes; if none, None.
    2. Name the skeleton in systematic style (locanted parents).
    3. Enumerate candidate descriptors (lowest-locant-first, and
       accept the first whose OPSIN round-trip reproduces the original mol.
    4. None if nothing round-trips (never a wrong labeled name).
    """
    original = Chem.MolFromSmiles(smiles)
    if original is None or not has_isotopes(original):
        return None
    stripped, label_map = strip_isotopes(original)
    if not label_map:
        return None
    # (``:44180``). Everything below names the isotope-STRIPPED molecule,
    # so no structural test further down can see that a label exists -- measured: at
    # the substituent licences every ``GetIsotope`` in scope reads 0 even
    # for a 13C input. Declare it ambiently instead, UNCONDITIONALLY, so a licence
    # about to empty a scope of all its locants can decline. This is deliberately
    # weaker than ``forced_locant_scope`` (entered further down only once a locant is
    # known to be REQUIRED): the weaker flag must not gag the omissions that stay
    # correct under, i.e. ``(13C1)benzenehexol`` and ``(2H6)benzene``.
    from ..assembly.locant_omission import isotopic_naming_scope
    _shift_token = _BOND_SHIFT_PARSE_REFUSED.set(False)
    try:
        with isotopic_naming_scope("isotope"):
            result = _decorate_isotopic_name_inner(
                smiles, style, namer, original, stripped, label_map)
            if result is None:
                result = _decorate_with_identical_prefixes_apart(
                    smiles, style, namer, original, stripped, label_map)
        if (result and _BOND_SHIFT_PARSE_REFUSED.get()
                and _has_fused_localized_alternation(original)):
            # A fused mancude parent names no ene locants, so for the drawing
            # whose lower-locant name reads back as its bond-shift isomer the
            # name has a higher locant ('(3-2H)pentalene' against '(1-2H)'), and
            # the PIN of that drawing carries a Delta descriptor,
            # "Localized double bonds" (the Blue Book,:14597 "this
            # differentiation is indicated by the use of the Greek letter Delta";
            # '1,6-dimethyl-Delta1(10a)-heptalene (PIN)':14601), which OPSIN 2.9.0
            # cannot read. The name is kept and labelled below the PIN.
            from ..metrics.provenance import record_non_pin_fragment
            record_non_pin_fragment(result)
        return result
    finally:
        _BOND_SHIFT_PARSE_REFUSED.reset(_shift_token)


def _has_fused_localized_alternation(mol: Chem.Mol) -> bool:
    """True iff ``mol`` has a fused ring system (two or more rings sharing a bond)
    whose ring atoms are all unsaturated (aromatic, or in a ring double bond) and
    that carries a ring double bond RDKit does not perceive as aromatic: a mancude
    fused parent named by fusion nomenclature, which cites no ene locants
    (pentalene, heptalene, s-indacene). A monocycle cites its ene locants
    ('cycloocta-1,3,5,7-tetraene', and is not such a system."""
    try:
        ri = mol.GetRingInfo()
        rings = [set(r) for r in ri.BondRings()]
    except Exception:
        return False
    if len(rings) < 2:
        return False
    # fused systems: rings that share a bond
    systems: List[set] = []
    for r in rings:
        merged = [sys_ for sys_ in systems if sys_ & r]
        new = set(r)
        for m in merged:
            new |= m
            systems.remove(m)
        systems.append(new)
    double = Chem.BondType.DOUBLE
    for bonds in systems:
        n_rings = sum(1 for r in rings if r <= bonds)
        if n_rings < 2:
            continue
        bs = [mol.GetBondWithIdx(i) for i in bonds]
        if not any(b.GetBondType() == double and not b.GetIsAromatic() for b in bs):
            continue
        atoms = {a for b in bs for a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())}
        if all(mol.GetAtomWithIdx(a).GetIsAromatic()
               or any(b.GetBondType() == double and b.IsInRing()
                      for b in mol.GetAtomWithIdx(a).GetBonds())
               for a in atoms):
            return True
    return False


def _decorate_with_identical_prefixes_apart(smiles, style, namer, original, stripped,
                                            label_map) -> Optional[str]:
    """The decoration of a skeleton named with its identical prefixes of different
    atoms cited apart, when the grouped skeleton could not carry the labels.

     'Different isotopic modifications on otherwise identical
    substituents' (the Blue Book): "When two substituent groups are
    isotopically modified in different ways so that they cannot be combined
    together using multiplicative terms such as 'di-', 'bis-', etc., they are cited
    separately" (:43758). The skeleton is named without its labels, so an
    N-substituent and a ring prefix of one name form one group ('N,N,6-triphenyl-
    ...-9H-carbazol-3-amine'), and a label on one member ('6-(2H5)phenyl') has no
    place in it. Inside ``identical_prefixes_cited_apart`` the producers cite each
    source's groups on their own ('N,N-diphenyl-6-phenyl-...'), and the same
    enumeration and round trip run on that skeleton. Only when that skeleton is
    spelled differently. The name also separates groups whose members carry the
    same labels, so it is recorded non-PIN (labelled below the PIN)."""
    from ..assembly.composition_primitives import identical_prefixes_cited_apart
    stripped_smiles = Chem.MolToSmiles(stripped)
    try:
        grouped = (namer.name(stripped_smiles),
                   _flagged_systematic_namer(namer).name(stripped_smiles))
        with identical_prefixes_cited_apart():
            apart = (namer.name(stripped_smiles),
                     _flagged_systematic_namer(namer).name(stripped_smiles))
            if apart == grouped:
                return None
            result = _decorate_isotopic_name_inner(
                smiles, style, namer, original, stripped, label_map)
    except Exception:  # noqa: BLE001 - an optional second spelling
        return None
    if result:
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(result)
    return result


def _flagged_systematic_namer(namer):
    """A fresh ``style="systematic"`` engine that INHERITS the caller's best-effort
    flags.

     (p4-trace): the isotope decorator named the isotope-STRIPPED skeleton with a
    FLAG-LESS ``Orthonym(style="systematic")``. The default top-level style is
    ``"pin"``, so the flag-carrying ``namer.name`` path below is bypassed and this
    fresh engine ran WITHOUT the caller's ``general_fallback`` /
    ``general_fallback_unverified`` / ``allow_aromatic_general`` — so a skeleton the
    general engine could only name at the best-effort tier returned
    ``'unknown organic compound'`` and the decoration failed closed
    (``isotope_decorator_failed``), dropping ~26% of the reclaimable isotope rows.
    The skeleton must be named at the SAME tier the caller runs at. Best-effort tier
    only (the flags are False at the PIN default), so PIN/complete output is
    byte-identical; ``_isotope_round_trips`` still gates every emission (0-wrong)."""
    from ..namer import Orthonym
    return Orthonym(
        style="systematic",
        general_fallback=getattr(namer, "_general_fallback", False),
        general_fallback_unverified=getattr(
            namer, "_general_fallback_unverified", False),
        allow_aromatic_general=getattr(namer, "_allow_aromatic_general", False),
    )


def _is_same_parent_locanted(retained_skel: str, systematic_skel: str) -> bool:
    """True iff ``systematic_skel`` is the SAME parent as ``retained_skel``,
    differing ONLY by locants (``ethanol`` / ``ethan-1-ol``), not a different
    parent word (``acetic acid`` / ``ethanoic acid``).

     decides between a retained and a systematic parent by WHERE the
    nuclide sits, not by spelling: its converse keeps the retained PIN when the
    label is on a numbered position (``(2-2H1)acetic acid``, ``(2H3)acetonitrile``
    the Blue Book). But 's own example (the Blue Book) shows a retained *shorthand*
    that merely elides a suffix locant must STILL show it when the isotope forces a
    locant: ``(2-13C)ethan-1-ol [not (2-13C)ethanol]``. Those are the same parent,
    just locanted, so the systematic re-render is accepted; a genuinely different
    parent word (``ethanoic acid`` for ``acetic acid``) is not. Comparison strips
    every locant digit, its ``,`` separators and the ``-`` that delimit it; a
    surplus/false match can only pick a non-preferred (still OPSIN-RT-verified)
    spelling, never a wrong molecule (0-wrong)."""
    def _strip(s: str) -> str:
        return re.sub(r"[\d,]", "", s).replace("-", "")
    return _strip(retained_skel) == _strip(systematic_skel)


def _decorate_isotopic_name_inner(smiles, style, namer, original, stripped,
                                  label_map) -> Optional[str]:
    """Body of:func:`decorate_isotopic_name`, run inside the ambient isotopic scope."""
    stripped_smiles = Chem.MolToSmiles(stripped)
    from ..errors import is_failure_name
    # ── Skeleton selection and its converse) ─────────────────────────
    # The descriptor is spliced into a SKELETON named from the isotope-STRIPPED mol.
    # Which spelling is PREFERRED is a question: a retained name is
    # abandoned for a systematic one ONLY when the nuclide sits on a position the
    # retained name does not number. Its CONVERSE -- the label IS on a numbered
    # position -- keeps the retained PIN (``(2H3)acetonitrile (PIN)`` the Blue Book;
    # ``(2-2H1)acetic acid`` by /. So the retained/PIN spelling
    # (``namer.name`` at the caller's style, e.g. ``acetic acid``) is tried FIRST,
    # and the always-locanted SYSTEMATIC spelling (``ethanoic acid``) is the
    # guaranteed fallback with UNCHANGED behaviour when the retained parent places
    # nothing (benzonitrile -> Task 6; ``anisole`` cannot site the label so
    # ``methoxybenzene`` wins). ``systematic_name`` is named at the caller's tier
    # (p4-trace) so a best-effort skeleton stays reachable. Every candidate below
    # is ``_isotope_round_trips``-gated, so a wrong skeleton choice fails closed.
    # Each skeleton naming records its own provenance (the producer that named
    # it); the shipped name then carries the provenance of the skeleton it was
    # built on. Without the snapshots the systematic skeleton's producer (the
    # general engine at the best-effort tier) stayed recorded when the name was
    # built on the PIN skeleton: '(1-2H)pentalene' read systematic_verified at the
    # best-effort tier and pin_verified at the PIN tier.
    from ..metrics.provenance import get_provenance, restore_provenance
    _prov_before = get_provenance()
    systematic_name = (namer.name(stripped_smiles) if style == "systematic"
                       else _flagged_systematic_namer(namer).name(stripped_smiles))
    _prov_systematic = get_provenance()
    if style == "systematic":
        pin_name = systematic_name
        _prov_pin = _prov_systematic
    else:
        restore_provenance(_prov_before)
        pin_name = namer.name(stripped_smiles)
        _prov_pin = get_provenance()

    def _usable(s: Optional[str]) -> bool:
        return bool(s) and "unknown" not in s.lower() and not is_failure_name(s)

    # (the Blue Book) + (the Blue Book): a retained arene whose
    # positions the retained spelling does not number and which allows "no
    # substitution" for PINs (``toluene``) is NOT substitutable once isotopically
    # modified -- the label has no legal numbered position in the retained name, so
    # the systematic parent (``methylbenzene``) is the PIN parent. The retained
    # ``toluene`` otherwise WINS the skeleton loop below because ``(13C)toluene``
    # round-trips; drop it from the candidates when a usable systematic spelling
    # exists so the descriptor lands on ``methylbenzene`` and, for a ring label,
    # ``_place_on_skeleton``'s re-render restores ``1-methylbenzene``
    # (-> ``1-methyl(2-13C)benzene``). Isotope-path only (a label is always present
    # here); the non-isotope namer that emits plain ``toluene`` is never reached.
    drop_retained = (pin_name in _ISOTOPE_NONSUBSTITUTABLE_RETAINED
                     and _usable(systematic_name) and pin_name != systematic_name)

    # Ordered ``(skeleton, is_pin_pass)`` list: retained/PIN spelling first, but ONLY
    # when it genuinely differs from the systematic one; otherwise the single deduped
    # skeleton runs the unchanged systematic path. ``is_pin_pass`` gates the
    # re-render's parent-switch (see ``_place_on_skeleton``).
    #
    # (the Blue Book, ``(2-13C)ethan-1-ol [not (2-13C)ethanol]``): when the
    # retained/PIN spelling is the SAME PARENT as the systematic one and differs from
    # it ONLY by an elided locant (``ethanol`` vs ``ethan-1-ol``), an isotopic
    # modification withdraws that omission licence -- the locanted spelling is
    # required. Do NOT offer the unlocanted retained pass in that case, so the label
    # lands on the locanted parent (``CC[18OH]`` -> ``(18O)ethan-1-ol``, gold protect
    # W2F-P5-P4) rather than the unlocanted ``(18O)ethanol``. A genuinely different
    # retained parent word (``acetic acid`` vs ``ethanoic acid``) is unaffected and
    # keeps its PIN pass. Every candidate stays OPSIN-RT gated (0-wrong).
    specs: List[Tuple[str, bool]] = []
    if (not drop_retained and _usable(pin_name) and _usable(systematic_name)
            and pin_name != systematic_name
            and not _is_same_parent_locanted(pin_name, systematic_name)):
        specs.append((pin_name, True))
    if _usable(systematic_name):
        specs.append((systematic_name, False))
    elif _usable(pin_name):
        specs.append((pin_name, False))
    if not specs:
        return None

    # Group labels by (mass, element) -> count; element read from the ORIGINAL
    # mol at each labeled atom index (label_map keys index the original).
    # ``maxpos_by_key`` carries the polysubstitution quantity per nuclide
    # (:func:`_max_atoms_at_position`), taken as the MAX over the group's atoms so
    # a count-1 nuclide contributes its single atom's capacity -- it decides
    # whether the count subscript is shown (FIX-A). ``keys`` entries are enriched
    # to ((mass, el), count, max_at_pos).
    by_key: Dict[Tuple[int, str], int] = {}
    maxpos_by_key: Dict[Tuple[int, str], int] = {}
    for idx, mass in label_map.items():
        atom = original.GetAtomWithIdx(idx)
        el = atom.GetSymbol()
        by_key[(mass, el)] = by_key.get((mass, el), 0) + 1
        maxpos_by_key[(mass, el)] = max(
            maxpos_by_key.get((mass, el), 0), _max_atoms_at_position(atom))

    n_pos = stripped.GetNumHeavyAtoms()

    # Alphabetical by element then mass citation order).
    keys = sorted(
        (((mass, el), by_key[(mass, el)], maxpos_by_key[(mass, el)])
         for (mass, el) in by_key),
        key=lambda kv: (kv[0][1], kv[0][0]))

    # The insertion-offset / locant enumeration, reused per skeleton spelling.
    #
    #: the descriptor "is inserted before the part of the compound that
    # is isotopically substituted". For a whole-parent label that is the front of
    # the name; for a labeled part after substituent prefixes (the labeled C of
    # trichloro(12C1)methane) it sits before the parent stem. Orthonym does not
    # parse the skeleton's grammar, so it enumerates every insertion offset and
    # lets the OPSIN round-trip oracle pick the unique correct one (probe: only
    # ONE offset round-trips per (structure, descriptor)). Front (0) is tried
    # first so the front placement is preferred when it is valid.
    #
    # Locant-form order /: the no-locant form (None) is
    # tried first — locants are omitted when the position is unambiguous; then
    # integer locants ascending (lowest-locant-first). Among round-trippers,
    # /.4.3 (higher Z then higher mass at the lower locant) breaks ties.
    # (A3: the search itself now lives in the module-level
    # ``_find_best_placement`` so the multi-attachment path below can reuse it
    # unchanged; this closure is a thin adapter preserving the existing
    # ``(candidate, loc_rank)`` contract used by the rest of this function.)

    def _enumerate(skel):
        """Best (candidate, loc_rank) for one skeleton spelling, or (None, None).

        ``loc_rank`` is -1 when the winning descriptor needs NO locant, else the
        locant. That value is the trigger: see below.
        """
        # The letter locants depend on the skeleton spelled: numbered
        # N locants only for a parent citing several nitrogens of its own groups).
        candidate, _off, _desc, loc_rank = _find_best_placement(
            skel, keys, original, n_pos, allow_front_hyphen=True,
            extra_locants=_letter_locant_candidates(
                original, label_map, n_pos, skeleton=skel),
            complete_sub_omission=True)
        return candidate, loc_rank

    def _place_on_skeleton(skeleton: str, is_pin_pass: bool) -> Optional[str]:
        """Run the de-mult / uniform / letter-mult / single-descriptor
        enumeration (+ the locant restoration) on ONE skeleton spelling,
        returning the first round-tripping candidate or None.

        The heavy multi-position / distinct-multi-locant / mixed-locant / stereo
        fallbacks are NOT run here — they execute ONCE, below the skeleton loop, on
        the systematic skeleton after every ``specs`` entry has placed nothing, so
        their behaviour is byte-for-byte unchanged from before PIN-first ordering."""
        # de-multiplication FIRST (single OR mixed nuclides). This is the
        # ONLY path that can place DISTINCT nuclides on DISTINCT copies of a
        # de-multiplied substituent (the mixed 18O/13C case), and it is bounded, so
        # routing it ahead of the single-combined-descriptor enumeration below both
        # (a) fixes that case and (b) avoids the fruitless JVM-per-candidate
        # exhaustion that enumeration performs on a de-mult skeleton. Fail-through to
        # the enumeration when the skeleton is not a de-mult form.
        demux = _decorate_demultiplied(skeleton, keys, original, stripped)
        if demux is not None:
            return demux

        # +: a UNIFORM multiplied substituent, every copy carrying
        # the IDENTICAL nuclide descriptor (which _decorate_demultiplied declines by
        # design -- identical copies stay grouped under the multiplier). Keep the
        # multiplier and scope one bracket-stepped descriptor to the repeated copy:
        # 1,2-di[(13C)methyl]benzene, NOT 1,2-di(1,1-13C2)methylbenzene. Runs before the
        # single-combined-descriptor enumeration (which would place a whole-molecule
        # (1,1-13C2) descriptor before the multiplier). Non-hydrogen nuclides only
        # (deuterium bracket hazard -- see the function's docstring); fails through
        # otherwise.
        uniform = _decorate_uniform_multiplier(skeleton, keys, original, stripped)
        if uniform is not None:
            return uniform

        # A3 Task 3, letter-locant class): a repeated IDENTICAL
        # letter-locant multiplier ANYWHERE in the skeleton (e.g. the
        # ``N,N,N-trimethyl`` of a quaternary ammonium salt cation) -- distinct
        # from the numeric leading-multiplier shape ``_decorate_demultiplied``
        # already handles. See that function's docstring for why.
        letter_demux = _decorate_letter_multiplier(skeleton, keys, original, stripped)
        if letter_demux is not None:
            return letter_demux

        # +: ONE labelled copy among the copies of a simple
        # multiplied substituent, with >= 2 copies left bare. The bare copies must
        # regroup under a REDUCED multiplier (1,3,5-trimethylbenzene + one 13C ->
        # 1-(13C)methyl-3,5-dimethylbenzene) -- the "different name shape"
        # _decorate_demultiplied declines (it leaves at most one copy bare). Runs
        # after the 0/1-bare (_decorate_demultiplied) and all-labelled uniform
        # (_decorate_uniform_multiplier) shapes, whose scopes are disjoint from this.
        reduced = _decorate_reduced_multiplier(skeleton, keys, original, stripped)
        if reduced is not None:
            return reduced

        best, loc_rank = _enumerate(skeleton)
        if best is None:
            return _decorate_one_labelled_copy(skeleton, keys, original, stripped)

        # ── criterion (k) (the Blue Book), the senior-chain isotope
        # tie-break ──────────────────────────────────────────────────────────────────
        # The skeleton was chosen from the isotope-STRIPPED mol, so a SYMMETRIC alkyl
        # branch (isobutyraldehyde's two methyls, one = CD3) is a genuine tie and the
        # descriptor lands on the substituent (``2-(1,1,1-2H3)methylpropanal``). The
        # senior parent chain must instead carry the isotope
        # (``2-methyl(3,3,3-2H3)propanal``). Promote the label into the chain iff the
        # parent placement round-trips (which proves the two positions are symmetry-
        # equivalent, so criterion (k) governs); RT-gated + fail-closed. The promoted
        # parent placement already carries its full locants, so the
        # substituent re-render below does not apply -- return it directly.
        promoted = _promote_isotope_to_parent_chain(
            skeleton, keys, original, n_pos, best)
        if promoted is not None:
            return promoted

        # ── (the Blue Book), the conditional locant restoration ────────────────
        # Verbatim: "In preferred IUPAC names, locants are omitted if no locants are
        # necessary in unmodified names. However, if isotopic modification requires a locant
        # to specify its position, then all locants must be specified and none are omitted."
        # Its own example prints the elided form as the REJECTED one (the Blue Book):
        # 13CH3-CH2-OH (2-13C)ethan-1-ol [not (2-13C)ethanol]
        #
        # `loc_rank >= 1` is precisely "the isotopic modification requires a locant": the
        # enumeration tries the locant-FREE descriptor first, so a locant is only reached
        # when the locant-free form failed to round-trip -- i.e. when the position genuinely
        # has to be stated. When `loc_rank == -1` no locant was needed, the condition in the
        # rule is not met, and the parent keeps its licensed omission. That is what makes
        # `(13C1)benzenehexol` correct, the Blue Book -- all six ring positions are one
        # orbit, so there is only one isotopomer) while `(13C2)benzenehexol` is not.
        #
        # The skeleton above was named from the isotope-STRIPPED molecule, so the
        # licences could not see the label and elided freely. Re-name it inside the ambient
        # scope so they decline, then re-enumerate against the locanted spelling.
        # A letter locant amide ``N``) is a non-int ``loc_rank``; it is
        # already forced INTO the descriptor, so the parent-relocanting re-render below
        # (which handles integer positions on the parent chain/ring) does not apply --
        # skip it for non-int ranks.
        if isinstance(loc_rank, int) and loc_rank >= 1:
            # The re-render records its own producer; the name built on it carries
            # that record, a name kept on ``skeleton`` does not.
            _prov_skeleton = get_provenance()
            with _isotope_locant_scopes(original):
                skel2 = _flagged_systematic_namer(namer).name(stripped_smiles)
            _prov_skel2 = get_provenance()
            restore_provenance(_prov_skeleton)
            # (converse): on the retained/PIN pass the systematic
            # re-render can be a DIFFERENT parent word (``ethanoic acid`` for
            # ``acetic acid``); the converse keeps the retained PIN, so accept the
            # systematic re-render only when it is the SAME parent merely locanted
            # (``ethanol`` -> ``ethan-1-ol``, the /the Blue Book example). The
            # systematic pass (``is_pin_pass`` False) always relocates -- unchanged.
            if (skel2 and "unknown" not in skel2.lower() and skel2 != skeleton
                    and (not is_pin_pass
                         or _is_same_parent_locanted(skeleton, skel2))):
                best2, _ = _enumerate(skel2)
                # Fail toward the locanted spelling only if it still round-trips; never
                # trade a verified name for an unverified one.
                if best2 is not None:
                    restore_provenance(_prov_skel2)
                    return best2
        return best

    # ── (the Blue Book) systematic ring-carbon parent, PRE-emptive ─────────
    # "When the nuclide is located at a position in a retained name that is not
    # numbered a systematic name that identifies separately the relevant atom is
    # used for the IUPAC preferred name." When the label sits on the nitrile /
    # carboxyl / benzylic carbon of benzonitrile / benzoic acid / benzyl-, the
    # retained placement below emits an unlocanted, BB-illegitimate
    # ``(13C)benzonitrile`` -- and it ROUND-TRIPS, so the retained/systematic specs
    # loop never abandons it and the post-loop systematic fallback is unreachable.
    # Offer the systematic ring-carbon parent (``benzene...carbonitrile`` /
    # ``...carboxylic acid`` / ``(phenylmethyl)...``) HERE so the descriptor lands
    # before the suffix (``benzene(13C)carbonitrile``). RT-gated via ``_enumerate``
    # and fired ONLY when a label is on that unnumbered functional carbon, so the
    # specs loop is byte-for-byte unchanged whenever does not apply.
    ring_c_parent = _systematic_parent_fallback(
        stripped, original, label_map, pin_name)
    if (ring_c_parent and _usable(ring_c_parent)
            and ring_c_parent not in (pin_name, systematic_name)):
        rc_best, _rc_loc = _enumerate(ring_c_parent)
        if rc_best is not None:
            return rc_best
        # A label spanning MORE than one part on that ring-carbon parent (e.g. a
        # 13C on the unnumbered carboxyl carbon AND a ring D) needs one
        # descriptor per attachment part, which the single combined enumeration
        # above cannot place. The ordinary multi-position fallback below runs on
        # the RETAINED skeleton (``benzoic acid``), which still cannot number the
        # functional carbon, so run it HERE on the ring-carbon parent
        # (``benzenecarboxylic acid``) -> ``(4-2H)benzene(13C)carboxylic acid``.
        # RT-gated inside ``_decorate_multi_position`` (0-wrong).
        rc_groups = _partition_by_attachment(original, label_map)
        if len(rc_groups) > 1:
            rc_multi = _decorate_multi_position(
                ring_c_parent, rc_groups, original, n_pos,
                namer=namer, stripped_smiles=stripped_smiles)
            if rc_multi is not None:
                return rc_multi

    # Retained/PIN skeleton first, systematic second converse). Every
    # candidate is OPSIN-RT-gated inside ``_place_on_skeleton`` (0-wrong).
    for _skel, _is_pin_pass in specs:
        restore_provenance(_prov_pin if _skel == pin_name else _prov_systematic)
        placed = _place_on_skeleton(_skel, _is_pin_pass)
        if placed is not None:
            return placed

    # ── Heavy fallbacks (UNCHANGED): run once, on the systematic skeleton ─────────
    # Reached only when NO skeleton spelling in ``specs`` placed a single-descriptor
    # candidate above. ``skeleton`` is the systematic spelling (or the sole usable
    # one), i.e. the same parent the pre-loop code fell back to.
    skeleton = specs[-1][0]
    restore_provenance(_prov_pin if skeleton == pin_name else _prov_systematic)
    # A3 Task 1 + multi-position placement): the single
    # combined descriptor above shares ONE locant across every labelled
    # atom of a nuclide, which cannot express labels sitting at more than
    # one structurally distinct position (per-D-glycine: D on the amino
    # N, the alpha C, AND the carboxyl O). Try, in order:
    # (a) one descriptor per attachment group, on THIS skeleton;
    # (b) if the skeleton itself has no slot at all for some part of
    # the label (a retained parent with an unnumbered position,
    #, retry every placement strategy against a MORE
    # fully systematic alternate parent.
    groups = _partition_by_attachment(original, label_map)
    if len(groups) > 1:
        multi = _decorate_multi_position(
            skeleton, groups, original, n_pos,
            namer=namer, stripped_smiles=stripped_smiles)
        if multi is not None:
            return multi
    # (b') ONE nuclide at several DISTINCT positions on this parent
    # (1,3,7-2H3 on a von Baeyer cage / a multiply-labelled ring) -- neither
    # the single-shared-locant enumeration nor the attachment-group split
    # above expresses it.: when the parent skeleton itself carries the
    # positional label (a ring D breaking the monosubstituted symmetry), restore
    # its substituent locant too -- ``1-bromo(2,4-2H2)benzene`` -- by re-placing on
    # the forced-locant re-render first; RT-gated, falling back to the free skeleton.
    forced_skel = _forced_locant_skeleton(namer, stripped_smiles, original, skeleton)
    ml = None
    if forced_skel != skeleton:
        ml = _decorate_distinct_multi_locant(
            forced_skel, keys, original, label_map, n_pos)
    if ml is None:
        ml = _decorate_distinct_multi_locant(
            skeleton, keys, original, label_map, n_pos)
    if ml is not None:
        return ml
    # (b'') MIXED per-nuclide locants: an ambiguous nuclide that
    # needs a locant beside a unique one that does not -- (1-13C,15N) -- which
    # neither the single-shared-locant enumeration nor the same-nuclide
    # distinct-multi-locant path above can express.
    mx = _decorate_mixed_locant(skeleton, keys, original, n_pos,
                                stripped_smiles, namer)
    if mx is not None:
        return mx
    alt_skeleton = _systematic_parent_fallback(stripped)
    if alt_skeleton and alt_skeleton != skeleton:
        alt_demux = _decorate_demultiplied(alt_skeleton, keys, original, stripped)
        if alt_demux is not None:
            return alt_demux
        alt_letter_demux = _decorate_letter_multiplier(
            alt_skeleton, keys, original, stripped)
        if alt_letter_demux is not None:
            return alt_letter_demux
        alt_best, _alt_loc_rank = _enumerate(alt_skeleton)
        if alt_best is not None:
            return alt_best
        if len(groups) > 1:
            alt_multi = _decorate_multi_position(alt_skeleton, groups, original, n_pos)
            if alt_multi is not None:
                return alt_multi
        alt_ml = _decorate_distinct_multi_locant(
            alt_skeleton, keys, original, label_map, n_pos)
        if alt_ml is not None:
            return alt_ml
    # isotope-INDUCED stereocentre: the constitution round-trips but
    # the full-stereo oracle does not, because a nuclide made an otherwise
    # achiral centre chiral. Enumerate the stereodescriptor prefix (oracle-
    # driven, full-InChIKey gated). Tried last so a plain (non-stereo) row is
    # never charged its cost.
    stereo = _decorate_isotope_stereo(
        skeleton, keys, original, stripped, n_pos,
        _letter_locant_candidates(original, label_map, n_pos, skeleton=skeleton))
    if stereo is not None:
        return stereo
    # Last resort: one descriptor per attachment group as in (a), with the groups
    # that land at the same insertion slot merged into one descriptor for that
    # part: nuclide symbols "at the same place in the name" are cited
    # together) -- a CD2 and the amide ND of one acetamide, '(N,2,2-2H3)acetamide'
    # (PubChem 1M). Tried after every other strategy so no name they build
    # changes; RT-gated like every candidate.
    if len(groups) > 1:
        merged = _decorate_multi_position(
            skeleton, groups, original, n_pos,
            namer=namer, stripped_smiles=stripped_smiles, merge_collisions=True)
        if merged is not None:
            return merged
    # De-multiplication (single AND mixed nuclides) was already attempted
    # ahead of this enumeration by _decorate_demultiplied; nothing else to
    # try -> fail closed (never a wrong labeled name).
    return None

