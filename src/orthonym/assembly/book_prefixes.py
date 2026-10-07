"""Shared spellings for the best-effort writers: the book's form of a substituent prefix.

The last-resort writers (``rules/terminal_fragment``, the floor in
``assembly/universal_substituent``) assemble names mechanically. The helpers here give
them the Blue Book's spelling of the same group, built from the parts the writer has
already named; they never read or rewrite an emitted name.

Locants (roadmap N5f):

* (a) (the Blue Book-2893, heading ' The locant '1' is
  omitted:'): "(a) in substituted mononuclear parent hydrides" -- a methyl group cites
  no locant on its prefixes ('chloromethyl', the Blue Book; '2-(hydroxymethyl)
  benzene-1,4-diol (PIN)',:6802), and (:7272) encloses the second and
  further substituents of a mononuclear parent ('chloro(methyl)silane (PIN)').
* (:3007): "All locants are omitted in compounds or substituent groups in
  which all substitutable positions are completely substituted or modified... in the
  same way" ('1-chloro-2-(pentafluoroethyl)benzene (PIN)',:3023); "In case of partial
  substitution or modification, all numerical prefixes must be indicated" (:3009).
* (1) (:15813): "The atom with the free valence terminates a chain and always
  has the locant '1', which is omitted from the name" -- 'ethyl', '2-chloroethyl', not
  'ethan-1-yl'; 'methyl (preferred prefix) methanyl' (:15864).

Acyl groups (roadmap N5e):

* (2) (:29451): "carbonyl groups in position 1 of a side chain, i.e., -CO-R, are
  described by the appropriate acyl group name"; (3) (:29452) 'formyl'.
* (:32922-:32928): -CO-NH2 on a ring, a heterogeneous chain or a
  nonterminal carbon is 'carbamoyl', "substituted in the normal way"
  ('3-(dimethylcarbamoyl)pentanedioic acid (PIN)',:32934); never 'amino... oxomethyl'.
* (:30624): the acid suffix 'carboxylic acid' gives the acyl suffix
  'carbonyl' ('pyrrolidine-1-carboxylic acid (PIN)',:29892 -> 'pyrrolidine-1-carbonyl').
* (:31698): an ester group -CO-OR' cited as a prefix is 'alkoxycarbonyl'
  ('3-(ethoxycarbonyl)phenoxy',:31813).
* (:30446): 'benzoyl (preferred prefix) benzenecarbonyl oxo(phenyl)methyl'
  -- the enclosed 'R(oxo)methyl' is the general spelling when no acyl name is built;
   (:30851): a monovalent group concatenated with 'carbonyl',
  '(azetidin-1-yl)carbonyl', when the acid name of a ring-N carbonyl is not built.
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Iterable, Optional, Sequence

__all__ = [
    "A_CHAIN_ENDS",
    "CHALCOGEN_HEAD",
    "HETERO_ROOT_VALENCE",
    "METHYL_HEADS",
    "MONONUCLEAR_HEAD",
    "a_chain_licensed",
    "acyl_derivation",
    "acyl_group_prefix",
    "acyl_side_atoms",
    "book_forms_enabled",
    "hetero_roots_composable",
    "mechanical_forms",
    "methyl_group_name",
    "note_book_form",
    "positions_alike_prefix",
    "retained_group_prefix",
    "watch_book_forms",
]

#: True (the default): the writers spell the book's form. A test (or a retry that
#: wants the earlier spelling back) switches it off with ``mechanical_forms``. The
#: variable is ``assembly.memo.book_forms_var``, so every memo key of a naming path
#: carries the switch (a value computed with one spelling is never served to the other).
from .memo import book_forms_var as _BOOK_FORMS  # noqa: E402


def book_forms_enabled() -> bool:
    return _BOOK_FORMS.get()


#: Set by ``watch_book_forms``: a one-item list that ``note_book_form`` sets to True.
_FIRED = contextvars.ContextVar("orthonym_book_forms_fired", default=None)


def note_book_form() -> None:
    """A writer records that it gave a book spelling in place of its mechanical one
    (read by ``watch_book_forms``; a no-op outside one)."""
    box = _FIRED.get()
    if box is not None:
        box[0] = True


@contextlib.contextmanager
def watch_book_forms():
    """``with watch_book_forms as fired:`` -- ``fired[0]`` is True after the block when
    a writer inside it gave a book spelling. The public naming calls use it to decide
    whether a molecule the book spellings left without a verified name is named again
    with ``mechanical_forms`` (``namer.Orthonym.name`` / ``name_tiered``)."""
    box = [False]
    token = _FIRED.set(box)
    try:
        yield box
    finally:
        _FIRED.reset(token)


@contextlib.contextmanager
def mechanical_forms():
    """Inside the block every writer gives its mechanical spelling (the spelling
    before roadmap N5)."""
    token = _BOOK_FORMS.set(False)
    try:
        yield
    finally:
        _BOOK_FORMS.reset(token)


#: the mononuclear carbon parent and its substituent prefixes by free-valence bond order
METHYL_HEADS = {0: "methane", 1: "methyl", 2: "methylidene", 3: "methylidyne"}


def methyl_group_name(branch_names: Sequence[str], bond_order: int = 1) -> Optional[str]:
    """'trifluoromethyl', '(4-chlorophenyl)methyl', 'chlorodi(fluoro)methyl',
    'hydroxy(phenyl)methyl', 'trichloromethane' (``bond_order`` 0) -- a one-carbon
    parent with its prefixes, no locants (a)), the second and further
    prefixes enclosed. ``branch_names`` holds one bare prefix name per
    substituent (a name repeated k times is cited once with its multiplier). ``None``
    when a name has no multiplier or marks the shared assembler accepts."""
    head = METHYL_HEADS.get(bond_order)
    if head is None or not branch_names:
        return None
    if any(not n or not n.strip() for n in branch_names):
        return None
    from .composer import _assemble_decorated_amino_prefix
    out = _assemble_decorated_amino_prefix(
        [(n, False) for n in branch_names], enclose=False, head=head)
    if out:
        note_book_form()
    return out


#: (the Blue Book 'The following retained names are used as preferred
#: prefixes for which no substitution is recommended'): C6H5-CH2- 'benzyl' (:24414),
#: C6H5-CH= 'benzylidene' (:24416), C6H5-C≡ 'benzylidyne' (:24418)
_BENZYL_BY_BOND_ORDER = {1: "benzyl", 2: "benzylidene", 3: "benzylidyne"}


def retained_group_prefix(mol, group_atoms, attach_idx: int, bond_order: int) -> Optional[str]:
    """The retained preferred prefix of an UNSUBSTITUTED benzyl or tert-butyl group, decided
    from its atoms, or ``None``.

    * (the Blue Book) '-C(CH3)3 tert-butyl (preferred prefix)
      1,1-dimethylethyl'; (:24414-:24418) 'benzyl', 'benzylidene', 'benzylidyne'.
    * (:16272) "... retained preferred prefixes, but are not to be substituted":
      any other heavy neighbour on the group, an isotope label (also on a hydrogen), a
      charge or a radical makes it another group ('(4-chlorophenyl)methyl').

    ``group_atoms`` is every heavy atom of the group, ``attach_idx`` its free-valence atom,
    ``bond_order`` the order of the bond to the parent. A radical centre is not a group on a
    parent (1),:40376); the radical writer never calls this."""
    from rdkit import Chem
    group = set(group_atoms)
    if attach_idx not in group or bond_order not in (1, 2, 3):
        return None
    for a in group:
        atom = mol.GetAtomWithIdx(a)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() or atom.GetIsotope()
                or atom.GetNumRadicalElectrons()):
            return None
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() == 1 and nb.GetIsotope():
                return None
            if a != attach_idx and nb.GetAtomicNum() > 1 and nb.GetIdx() not in group:
                return None
    c = mol.GetAtomWithIdx(attach_idx)
    if c.IsInRing():
        return None
    outside = [nb.GetIdx() for nb in c.GetNeighbors()
               if nb.GetAtomicNum() > 1 and nb.GetIdx() not in group]
    inside = [nb.GetIdx() for nb in c.GetNeighbors() if nb.GetIdx() in group]
    if len(outside) > 1 or any(
            mol.GetBondBetweenAtoms(attach_idx, j).GetBondType() != Chem.BondType.SINGLE
            for j in inside):
        return None
    if len(group) == 4 and len(inside) == 3:
        if bond_order != 1 or c.GetTotalNumHs() != 0:
            return None
        if any(mol.GetAtomWithIdx(j).GetTotalNumHs() != 3
               or sum(1 for nb in mol.GetAtomWithIdx(j).GetNeighbors()
                      if nb.GetAtomicNum() > 1) != 1 for j in inside):
            return None
        return "tert-butyl"
    if len(group) == 7 and len(inside) == 1:
        if c.GetTotalNumHs() != 3 - bond_order:
            return None
        if not _is_unsubstituted_benzene(mol, group - {attach_idx}, inside[0]):
            return None
        return _BENZYL_BY_BOND_ORDER[bond_order]
    return None


def _is_unsubstituted_benzene(mol, ring, ipso: int) -> bool:
    """``ring`` is six carbons forming one ring of ``mol`` (in no other ring), aromatic or
    with three alternating double bonds, one hydrogen on every atom but ``ipso``."""
    from rdkit import Chem
    ring = set(ring)
    if len(ring) != 6:
        return False
    ri = mol.GetRingInfo()
    if not any(set(r) == ring for r in ri.AtomRings()):
        return False
    if any(ri.NumAtomRings(a) != 1 for a in ring):
        return False
    if any(mol.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in ring):
        return False
    bonds = [b for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring]
    kinds = [b.GetBondType() for b in bonds]
    aromatic = all(k == Chem.BondType.AROMATIC for k in kinds)
    kekule = (kinds.count(Chem.BondType.DOUBLE) == 3
              and kinds.count(Chem.BondType.SINGLE) == 3
              and all(sum(1 for b in bonds if b.GetBondType() == Chem.BondType.DOUBLE
                          and a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())) == 1
                      for a in ring))
    if not (aromatic or kekule):
        return False
    return all(mol.GetAtomWithIdx(a).GetTotalNumHs() == (0 if a == ipso else 1)
               for a in ring)


def positions_alike_prefix(mol, spine_atoms: Iterable[int],
                           branch_names: Sequence[str], *,
                           unit_cites_locant: bool) -> Optional[str]:
    """The prefix block without locants when every substitutable position of the spine
    carries the same prefix:3007): 'pentafluoro' for -CF2-CF3, 'pentachloro'
    for C6Cl5-. ``None`` when the substitution is partial (a hydrogen is left on a spine
    atom) or when two kinds of prefix are present (:3009, all locants are then cited).

    ``unit_cites_locant`` is the writer's fact that the unit the block is cited on (the
    parent, the substituent prefix) cites a locant, not read from its text.
    (the Blue Book): "if any locants are essential for defining the structure of the
    parent structure or of a unit of structure... then all locants must be cited for the
    parent structure or that structural unit" -- a unit that cites a locant ('propan-2-yl',
    '1,3,4-oxadiazol-2-yl', '1H-imidazole') keeps every locant
    ('(1,1,1,3,3,3-hexafluoropropan-2-yl)oxy',:46359), so ``None``."""
    if unit_cites_locant:
        return None
    names = list(branch_names)
    if not names or len(set(names)) != 1:
        return None
    spine = list(spine_atoms)
    if not spine:
        return None
    for a in spine:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetTotalNumHs() != 0:
            return None
    from .naming_utils import enclose_if_compound, multiplied_component
    name = names[0]
    marked = enclose_if_compound(name)
    note_book_form()
    if len(names) == 1:
        return marked
    return multiplied_component(len(names), name, marked)


def acyl_side_atoms(mol, carbonyl_c: int, exclude: int) -> set:
    """The atoms of the acyl group R-CO- that hold ``carbonyl_c``: everything reachable from
    it without passing through ``exclude`` (the atom the group is attached to)."""
    seen, stack = set(), [carbonyl_c]
    while stack:
        a = stack.pop()
        if a in seen or a == exclude:
            continue
        seen.add(a)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors())
    return seen


def acyl_derivation(mol, carbonyl_c: int, acyl_atoms):
    """How the prefix of the acyl group R-CO- is built, from its atoms:

    * ``substituted`` (c),:7035): R carries an atom outside the acid's parent --
      the unbranched carbon chain that ends at the carbonyl carbon ('acetyl', the
      alkanoyls;:30442,:30608) or the ring system / ring
      assembly the carbonyl carbon is bonded to:30624);
    * ``enclosed``,:7232): a substituted acyl; an acyl named by the suffix
      'carbonyl' on a ring ('4-(cyclohexanecarbonyl)benzene-1-carbothioic acid (PIN)',
      :7322; '3-[(pyridine-3-carbonyl)oxy]propanoic acid (PIN)',:31723); not the
      retained 'benzoyl' ('3-(benzoyloxy)propanoic acid (PIN)',:31711) nor a saturated
      alkanoyl; ``None`` (main's predicates decide) for an unsaturated alkanoyl."""
    from rdkit import Chem
    from .prefix_derivation import PrefixDerivation
    acyl = set(acyl_atoms)
    c = mol.GetAtomWithIdx(carbonyl_c)
    oxo = [nb.GetIdx() for nb in c.GetNeighbors()
           if nb.GetIdx() in acyl and nb.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(carbonyl_c, nb.GetIdx()).GetBondType()
           == Chem.BondType.DOUBLE]
    rest = acyl - {carbonyl_c} - set(oxo[:1])
    if not rest:
        return PrefixDerivation(substituted=False, enclosed=False)        # formyl
    roots = [nb.GetIdx() for nb in c.GetNeighbors() if nb.GetIdx() in rest]
    if len(roots) != 1 or any(mol.GetAtomWithIdx(a).GetIsotope()
                              or mol.GetAtomWithIdx(a).GetFormalCharge() for a in rest):
        return PrefixDerivation(substituted=True, enclosed=True)
    y = roots[0]
    if _is_carboxy_group(mol, y, rest):
        # (the Blue Book) 'HO-CO-CO- oxalo (preferred prefix)', a
        # retained name,:29910): simple and bare, 'oxalooxy (preferred
        # prefix)' (:30544)
        return PrefixDerivation(substituted=False, enclosed=False)
    if mol.GetAtomWithIdx(y).IsInRing():
        from ..perception.rings import get_ring_systems
        systems = [set(s) for s in get_ring_systems(mol) if set(s) <= rest]
        own = next((s for s in systems if y in s), None)
        if own is None:
            return PrefixDerivation(substituted=True, enclosed=True)
        if rest == own:
            simple = True
        elif (len(systems) > 1 and sum(len(s) for s in systems) == len(rest)):
            from ..rules.ring_assemblies import detect_ring_assembly
            simple = detect_ring_assembly(mol, systems) is not None
        else:
            simple = False
        if not simple:
            return PrefixDerivation(substituted=True, enclosed=True)
        benzoyl = rest == own and _is_unsubstituted_benzene(mol, own, y)
        return PrefixDerivation(substituted=False, enclosed=not benzoyl)
    # acyclic: an unbranched carbon chain that starts at the carbonyl carbon
    prev, cur, seen, unsaturated = carbonyl_c, y, set(), False
    while True:
        atom = mol.GetAtomWithIdx(cur)
        if atom.GetAtomicNum() != 6 or atom.IsInRing():
            return PrefixDerivation(substituted=True, enclosed=True)
        seen.add(cur)
        if mol.GetBondBetweenAtoms(prev, cur).GetBondType() != Chem.BondType.SINGLE:
            unsaturated = True
        onward = [nb.GetIdx() for nb in atom.GetNeighbors()
                  if nb.GetIdx() in rest and nb.GetIdx() != prev]
        if not onward:
            break
        if len(onward) > 1:
            return PrefixDerivation(substituted=True, enclosed=True)
        prev, cur = cur, onward[0]
    if seen != rest:
        return PrefixDerivation(substituted=True, enclosed=True)
    return PrefixDerivation(substituted=False, enclosed=None if unsaturated else False)


def _is_carboxy_group(mol, c_idx: int, atoms) -> bool:
    """``atoms`` is exactly the carboxy group -CO-OH on carbon ``c_idx``: the carbon, one
    double-bonded O and one OH, no label and no charge."""
    from rdkit import Chem
    atoms = set(atoms)
    c = mol.GetAtomWithIdx(c_idx)
    if c.GetAtomicNum() != 6 or c.IsInRing() or len(atoms) != 3 or c_idx not in atoms:
        return False
    kinds = []
    for o in atoms - {c_idx}:
        at = mol.GetAtomWithIdx(o)
        bond = mol.GetBondBetweenAtoms(c_idx, o)
        if (at.GetAtomicNum() != 8 or bond is None or at.GetIsotope()
                or at.GetFormalCharge() or at.GetDegree() != 1):
            return False
        kinds.append((bond.GetBondType(), at.GetTotalNumHs()))
    return sorted(kinds) == sorted([(Chem.BondType.DOUBLE, 0), (Chem.BondType.SINGLE, 1)]) \
        and not c.GetIsotope() and not c.GetFormalCharge()


def acyl_group_prefix(mol, carbonyl_c: int, parent_idx: int, acyl_atoms,
                      name_part) -> Optional[str]:
    """The book's prefix of an acyl group (``_acyl_group_prefix``), recorded with
    ``note_book_form`` when one is given."""
    out = _acyl_group_prefix(mol, carbonyl_c, parent_idx, acyl_atoms, name_part)
    if out:
        note_book_form()
    return out


def _acyl_group_prefix(mol, carbonyl_c: int, parent_idx: int, acyl_atoms,
                       name_part) -> Optional[str]:
    """The book's prefix of the acyl group whose carbonyl carbon ``carbonyl_c`` (one
    ``=O``, not in a ring) is bonded to the parent atom ``parent_idx``, or ``None``.

    ``acyl_atoms`` is the whole group (carbonyl C, its =O and the rest). ``name_part(atoms,
    attach)`` is the calling writer's own namer of a smaller part as a substituent prefix
    (it returns the bare prefix name or ``None``); the parts are the amide N's
    substituents, the ester's R' group and a ring R.

    * -CHO 'formyl'; -CO-OH 'carboxy';
    * -CO-NH2 'carbamoyl', -CO-NHR / -CO-NRR' '(R)carbamoyl' / 'R(R')carbamoyl' through the
      shared N-prefix assembler marks); -CO-N< of a ring '<ring>-<n>-carbonyl'
      from the OPSIN-verified acid name ('azetidine-1-carbonyl');
    * -CO-OR' '(alkoxy)carbonyl' ('methoxycarbonyl', '(benzyloxy)carbonyl');
    * -CO-R (R carbon) the acyl name of the OPSIN-verified acid ('acetyl', 'benzoyl',
      'cyclohexanecarbonyl'), else the enclosed general form 'R(oxo)methyl'.
    """
    from rdkit import Chem

    from .prefix_derivation import built, with_derivation
    acyl = set(acyl_atoms)
    if carbonyl_c not in acyl or parent_idx in acyl:
        return None
    c = mol.GetAtomWithIdx(carbonyl_c)
    if (c.GetSymbol() != "C" or c.GetFormalCharge() or c.IsInRing()
            or mol.GetBondBetweenAtoms(carbonyl_c, parent_idx) is None
            or mol.GetBondBetweenAtoms(carbonyl_c, parent_idx).GetBondType()
            != Chem.BondType.SINGLE):
        return None
    oxo = [nb.GetIdx() for nb in c.GetNeighbors()
           if nb.GetSymbol() == "O" and nb.GetIdx() in acyl and nb.GetDegree() == 1
           and mol.GetBondBetweenAtoms(carbonyl_c, nb.GetIdx()).GetBondType()
           == Chem.BondType.DOUBLE]
    if len(oxo) != 1:
        return None
    others = [nb for nb in c.GetNeighbors()
              if nb.GetIdx() not in (parent_idx, oxo[0]) and nb.GetAtomicNum() > 1]
    if not others:
        return (built("formyl", substituted=False, enclosed=False)
                if c.GetTotalNumHs() == 1 and acyl == {carbonyl_c, oxo[0]} else None)
    if len(others) != 1:
        return None
    y = others[0]
    yi = y.GetIdx()
    if yi not in acyl or y.GetFormalCharge() or y.GetIsotope():
        return None
    if mol.GetBondBetweenAtoms(carbonyl_c, yi).GetBondType() != Chem.BondType.SINGLE:
        return None
    rest = acyl - {carbonyl_c, oxo[0]}

    def parts_of(root):
        """The substituent parts hanging off ``root`` inside ``rest`` (root excluded)."""
        out = []
        claimed = {root}
        for nb in mol.GetAtomWithIdx(root).GetNeighbors():
            j = nb.GetIdx()
            if j not in rest or j in claimed:
                continue
            comp, stack = set(), [j]
            while stack:
                cur = stack.pop()
                if cur in comp or cur == root:
                    continue
                comp.add(cur)
                for nb2 in mol.GetAtomWithIdx(cur).GetNeighbors():
                    k = nb2.GetIdx()
                    if k in rest and k not in comp and k != root:
                        stack.append(k)
            claimed |= comp
            out.append((j, comp))
        return out

    sym = y.GetSymbol()
    if sym == "O":
        if y.IsInRing():
            return None
        if y.GetDegree() == 1:
            return "carboxy" if y.GetTotalNumHs() == 1 and rest == {yi} else None
        parts = parts_of(yi)
        if len(parts) != 1 or y.GetDegree() != 2:
            return None
        r_attach, r_atoms = parts[0]
        if mol.GetAtomWithIdx(r_attach).GetSymbol() != "C" or _is_acyl_carbon(mol, r_attach):
            return None  # an anhydride or an O-hetero ester: not this producer's
        r_name = name_part(r_atoms, r_attach)
        if not r_name:
            return None
        from .substituent_enumerator import (
            alkoxy_prefix_from_substituent, cite_organyl_in_composed_prefix)
        alkoxy = alkoxy_prefix_from_substituent(r_name)
        if not alkoxy:
            return None
        cited = cite_organyl_in_composed_prefix(alkoxy)
        # (the Blue Book) '3-(ethoxycarbonyl)phenoxy': compound
        return built(f"{cited}carbonyl", substituted=True, enclosed=True) if cited else None
    if sym == "N":
        if y.GetTotalValence() != 3 or any(
                b.GetBondType() != Chem.BondType.SINGLE for b in y.GetBonds()):
            return None
        if y.IsInRing():
            named = _ring_nitrogen_carbonyl(mol, carbonyl_c, parent_idx, acyl)
            if named:
                return with_derivation(named, acyl_derivation(mol, carbonyl_c, acyl))
            # (:30851): a monovalent group added to the divalent acyl group
            # 'carbonyl' by concatenation, '(azetidin-1-yl)carbonyl'
            ring_name = name_part(rest, yi)
            if not ring_name:
                return None
            from ..metrics.provenance import record_non_pin_fragment
            from .naming_utils import apply_enclosing_marks
            # (the Blue Book) '(azetidin-1-yl)carbonyl': the ring-N prefix
            # cites its locant (2),:15814), so it is enclosed; method (2) is never
            # part of a PIN, method (1) names are (:30853)
            out = built(f"{apply_enclosing_marks(ring_name, -1)}carbonyl",
                        substituted=True, enclosed=True)
            record_non_pin_fragment(out)
            return out
        parts = parts_of(yi)
        if not parts:
            return (built("carbamoyl", substituted=False, enclosed=False)
                    if rest == {yi} else None)
        names = []
        for p_attach, p_atoms in parts:
            nm = name_part(p_atoms, p_attach)
            if not nm:
                return None
            names.append(nm)
        from .composer import _assemble_decorated_amino_prefix
        return _assemble_decorated_amino_prefix(
            [(n, False) for n in names], enclose=False, head="carbamoyl")
    if sym == "C":
        from .substituent_naming import acyl_prefix_from_branch
        try:
            named = acyl_prefix_from_branch(mol, carbonyl_c, parent_idx, acyl)
        except Exception:  # noqa: BLE001 - the acid route is a producer; decline
            named = None
        if named:
            return with_derivation(named, acyl_derivation(mol, carbonyl_c, acyl))
        r_name = name_part(rest, yi)
        if not r_name:
            return None
        out = methyl_group_name(["oxo", r_name], 1)
        if out:
            # (the Blue Book), (:30628): the acyl group
            # spelled as a methyl group is a general name, never part of a PIN
            from ..metrics.provenance import record_non_pin_fragment
            record_non_pin_fragment(out)
        return out
    return None


def _is_acyl_carbon(mol, idx: int) -> bool:
    a = mol.GetAtomWithIdx(idx)
    return a.GetSymbol() == "C" and any(
        nb.GetSymbol() in ("O", "S") and mol.GetBondBetweenAtoms(idx, nb.GetIdx())
        .GetBondTypeAsDouble() == 2.0 for nb in a.GetNeighbors())


def _ring_nitrogen_carbonyl(mol, carbonyl_c: int, parent_idx: int, acyl) -> Optional[str]:
    """'<ring>-<n>-carbonyl' for -CO-N< of a ring:30624 on the acid
    '<ring>-<n>-carboxylic acid', 'pyrrolidine-1-carboxylic acid (PIN)':29892), named as
    the acid R-N-CO-OH and kept only when OPSIN reads the acid name back to it
    (``substituent_naming.fragment_acid_name_verified``)."""
    from .substituent_naming import (
        acid_name_to_acyl_prefix, fragment_acid_name_verified)
    from .fragment_naming import name_fragment_recursively
    from rdkit import Chem
    try:
        rw = Chem.RWMol(mol)
        oh = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(carbonyl_c, oh, Chem.BondType.SINGLE)
        hh = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(oh, hh, Chem.BondType.SINGLE)
        acid = Chem.MolFragmentToSmiles(rw, sorted(set(acyl) | {oh, hh}), canonical=True)
        if not acid:
            return None
        for style in ("pin", "systematic"):
            acid_name = name_fragment_recursively(acid, style=style)
            if not acid_name or not acid_name.endswith("carboxylic acid"):
                continue
            prefix = acid_name_to_acyl_prefix(acid_name)
            if prefix and fragment_acid_name_verified(acid_name, acid):
                return prefix
    except Exception:  # noqa: BLE001 - a producer that raises is a decline
        return None
    return None


# --- heteroatom-rooted groups and 'a' chains (roadmap N5c) ------------------------------

#: (the Blue Book, 'Skeletal replacement ('a') nomenclature for acyclic
#: parent hydrides'): "The chain must be terminated by a C atom or one of the following
#: heteroatoms: P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, or Tl.",:23430)
A_CHAIN_ENDS = frozenset({"C", "P", "As", "Sb", "Bi", "Si", "Ge", "Sn", "Pb", "B", "Al",
                          "Ga", "In", "Tl"})


def a_chain_licensed(mol, path) -> bool:
    """True when the atom ``path`` is a carbon chain, or an 'a' chain the book allows: four
    or more heterounits, at least one carbon atom, both ends C or an ``A_CHAIN_ENDS``
    element the Blue Book,:6465). A heterounit "is a set of
    heteroatoms having a name of its own such as, -SS-, disulfanediyl":23350),
    counted by the strict path's own rule (``skeletal_replacement._count_chain_heterounits``:
    -SS-, -OO-, -SiH2-O-SiH2- are one unit each); a run that rule cannot count (a
    trisulfane,:23385) licenses no 'a' chain."""
    syms = [mol.GetAtomWithIdx(a).GetSymbol() for a in path]
    if all(x == "C" for x in syms):
        return True
    if "C" not in syms or syms[0] not in A_CHAIN_ENDS or syms[-1] not in A_CHAIN_ENDS:
        return False
    from ..rules.skeletal_replacement import _count_chain_heterounits
    units = _count_chain_heterounits(mol, list(path))
    return units is not None and units >= 4


#: the heteroatom roots a writer composes as a substituent prefix ('methoxy',
#: '(R)sulfanyl', '(R)amino', 'trimethylsilyl'), with their standard bonding number
#:
HETERO_ROOT_VALENCE = {"O": 2, "S": 2, "Se": 2, "Te": 2, "N": 3, "P": 3, "As": 3, "B": 3,
                       "Si": 4, "Ge": 4, "Sn": 4, "Pb": 4}
#: (:27633) 'R-oxy'; (:27649) '(R)sulfanyl', '(R)selanyl',
#: '(R)tellanyl'
CHALCOGEN_HEAD = {"O": "oxy", "S": "sulfanyl", "Se": "selanyl", "Te": "tellanyl"}
#: substituent prefixes of the mononuclear parent hydrides,: 'silyl',
#: 'phosphanyl', 'boranyl',...
MONONUCLEAR_HEAD = {"Si": "silyl", "Ge": "germyl", "Sn": "stannyl", "Pb": "plumbyl",
                    "P": "phosphanyl", "As": "arsanyl", "B": "boranyl"}


def hetero_roots_composable(mol, path) -> bool:
    """True when every heteroatom of the atom ``path`` can root a composed substituent
    prefix ('methoxy', '(R)amino', 'azaniumyl', 'trimethylsilyl'): an element of
    ``HETERO_ROOT_VALENCE`` at its standard bonding number with no charge and no isotope
    label, or an ammonium nitrogen (four bonds, +1). A charged or hypervalent heteroatom
    (the S+ of a sulfoxide written as S+-O-, the N+ of a nitrone, a sulfate sulfur) has
    no composed prefix here, so a writer keeps its chain spelling for such a path rather
    than cut the chain into a branch it can only spell as another 'a' chain. An -OO- link
    likewise: an oxygen roots 'R-oxy' from a carbon R only, and the book's 'R-peroxy'
     the Blue Book "'R′-peroxy' (not R′-dioxy)"; '(methylperoxy)ethane
    (PIN)',:27870) is not built here. (An -SS- link is cut: '[(R)sulfanyl]sulfanyl' reads
    back, the book's 'R-disulfanyl' of the same rule is not built either; residual.)"""
    on_path = set(path)
    for a in path:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetAtomicNum() == 6:
            continue
        sym = atom.GetSymbol()
        if sym == "O" and any(nb.GetAtomicNum() == 8 and nb.GetIdx() in on_path
                              for nb in atom.GetNeighbors()):
            return False
        if sym not in HETERO_ROOT_VALENCE or atom.GetIsotope():
            return False
        ammonium = (sym == "N" and atom.GetFormalCharge() == 1
                    and atom.GetTotalValence() == 4
                    and all(b.GetBondTypeAsDouble() == 1.0 for b in atom.GetBonds()))
        if atom.GetFormalCharge() and not ammonium:
            return False
        if not ammonium and atom.GetTotalValence() != HETERO_ROOT_VALENCE[sym]:
            return False
    return True
