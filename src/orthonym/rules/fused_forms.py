"""The book's names for a fusable ring system, with their numbering (roadmap N5b).

The best-effort writers named every polycyclic ring system by the von Baeyer system
('3,4-diazabicyclo[4.4.0]deca-1(10),2,6,8-tetraene', '7-azabicyclo[4.3.0]nona-...'). For a
system the book names by fusion nomenclature that is not a book spelling:

* 'Five-membered ring requirement' (the Blue Book;:23710): "Fusion
  nomenclature gives preferred IUPAC names only to compounds having at least two rings of
  at least five or more members.... When fusion names are not allowed, unsaturated von
  Baeyer ring system names are preferred IUPAC names".
* (:24221): "Retained fusion names are used for the fully unsaturated compounds
  (see; they are the preferred IUPAC names. Preferred IUPAC names for the partially
  saturated and fully saturated compounds are formed by using 'hydro' prefixes." --
  'decahydronaphthalene (PIN) bicyclo[4.4.0]decane' (:24233).
* 'Prefix nomenclature' (:24862,:24864): "After the introduction of indicated
  and 'added indicated hydrogen' atoms, all substituent groups not expressed as suffixes
  are cited as prefixes" -- a ring C=O cited as 'oxo' is a saturated position of the
  parent ('1-oxo-1,2-dihydrophthalazine').

``fused_system_form`` returns the fusion name of one ortho- or ortho- and peri-fused ring
system with two or more rings of five or more members, built from parts the engine
already certifies: the mancude parent and its catalogue numbering
(``partial_saturation._resolve_oxo_parent``: a bond-order-agnostic match against the fused
heterocycle and polycyclic catalogues, mancude entries only) and the indicated hydrogen
and hydro prefixes of the structure under that numbering (``ring_hydrogen.ring_hydrogen``
with no suffix group;,. The atom of a free valence that has no
hydrogen in the mancude parent takes the indicated hydrogen,:24768;
'1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl', Example 5,:25477); otherwise, among
the numberings the catalogue allows (its automorphisms), the lowest locants go to
,:3246-:3301) (b) indicated hydrogen, (c) the free valence, (e) hydro prefixes,
(f) the prefixes; then a canonical-rank tie-break, so the result does not depend on the
input atom order.

Out of scope (``None``; the caller keeps its von Baeyer spelling): bridged and spiro
systems; systems with fewer than two rings of five or more members (von Baeyer is the
book's there,:23710); charged, radical or isotope-labelled ring atoms; a ring atom of
nonstandard bonding number (a lambda atom,:2758); skeletons no
catalogue parent matches (fusion names the engine does not build yet, the fusion lane's);
a substituent or a free valence on a fusion atom (a lettered locant); an ylidene free
valence.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)

__all__ = ["FusedForm", "fused_system_form", "is_fusable_system", "locant_key"]

#: a leading indicated-hydrogen block of a catalogue name: '1H-', '9H-', '2H,6H-'
_IH_PREFIX = re.compile(r"^\d+[a-z]?H(?:,\d+[a-z]?H)*-")
_LOCANT = re.compile(r"^(\d+)([a-z]*)(['′]*)$")


def locant_key(loc) -> Tuple[int, str, str]:
    """A total order over ring locants, int or lettered str: 4 < 4a < 4b < 5
    , interior and fusion atoms take the letters of the preceding locant)."""
    if isinstance(loc, int):
        return (loc, "", "")
    m = _LOCANT.match(str(loc))
    if not m:
        return (10 ** 6, str(loc), "")
    return (int(m.group(1)), m.group(2), m.group(3))


@dataclass(frozen=True)
class FusedForm:
    """One fusable ring system spelled by its fusion name.

    ``parent`` the parent hydride with its hydro prefixes and indicated hydrogen
                   ('1,2,3,4-tetrahydronaphthalene', '1H-indole', '1,2-dihydrophthalazine').
    ``numbering`` ring atom index -> locant (int, or a lettered str for a fusion atom).
    ``kind`` 'fused' (the writers read ``MonocycleForm.kind`` the same way).
    """

    parent: str
    numbering: Dict[int, object]
    kind: str = "fused"
    parent_cites_locant: bool = False

    def cites_locant(self, free_valence_locant: Optional[int] = None) -> bool:
        """The parent or its prefix at ``free_valence_locant`` cites a locant,
        the Blue Book)."""
        return self.parent_cites_locant or free_valence_locant is not None

    def prefix(self, free_valence_atom: int, bond_order: int = 1) -> Optional[str]:
        """The substituent prefix with the free valence at ``free_valence_atom``."""
        return self.prefix_at_locant(self.numbering.get(free_valence_atom), bond_order)

    def prefix_at_locant(self, loc, bond_order: int = 1) -> Optional[str]:
        """'naphthalen-2-yl', '1H-indol-3-yl', '1,2,3,4-tetrahydronaphthalen-1-yl'
         (2), the Blue Book: the final 'e' of the parent is elided before
        'y' and the locant is cited). A single free valence on a numbered peripheral atom
        only: an ylidene changes the hydrogen of the parent, which this form does not
        recompute."""
        if bond_order != 1 or not isinstance(loc, int):
            return None
        stem = self.parent[:-1] if self.parent.endswith("e") else self.parent
        return f"{stem}-{loc}-yl"


def is_fusable_system(mol, ring_atoms: Iterable[int]) -> bool:
    """True for one whole ortho- or ortho- and peri-fused ring system with two or more
    rings of five or more members, the Blue Book): every ring of the
    molecule that touches ``ring_atoms`` lies inside it, two rings share either nothing
    or exactly one bond (no spiro atom, no bridge), and the smallest rings span the cycle
    space (a bridged system has more independent cycles than smallest rings)."""
    ring = set(ring_atoms)
    all_rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    if any(r & ring and not r <= ring for r in all_rings):
        return False
    rings = [r for r in all_rings if r <= ring]
    if len(rings) < 2:
        return False
    n_bonds = sum(1 for b in mol.GetBonds()
                  if b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring)
    if n_bonds - len(ring) + 1 != len(rings):
        return False
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared = rings[i] & rings[j]
            if len(shared) == 1 or len(shared) > 2:
                return False
            if len(shared) == 2:
                a, b = shared
                if mol.GetBondBetweenAtoms(a, b) is None:
                    return False
    return sum(1 for r in rings if len(r) >= 5) >= 2


def fused_system_form(mol, ring_atoms: Iterable[int], free_valence_atom: Optional[int] = None,
                      branch_atoms: Sequence[int] = ()) -> Optional[FusedForm]:
    """The fusion name and numbering of the ring system ``ring_atoms``, or ``None``.

    ``branch_atoms`` lists the ring atoms that carry a substituent prefix, once per prefix
     (f)); ``free_valence_atom`` is the ring atom of the free valence of a
    substituent prefix, ``None`` for a parent. Every atom of either kind must take a
    numbered (not lettered) locant."""
    from ..assembly.book_prefixes import book_forms_enabled
    if not book_forms_enabled():
        return None
    ring = set(ring_atoms)
    if free_valence_atom is not None and free_valence_atom not in ring:
        return None
    if not is_fusable_system(mol, ring):
        return None
    from .lambda_convention import nonstandard_bonding_number
    for a in ring:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumRadicalElectrons():
            return None
        if nonstandard_bonding_number(mol, a) is not None:
            # (the Blue Book) and (:12452): a ring atom of
            # nonstandard bonding number takes a lambda descriptor the catalogue's parent
            # name does not carry ('[1,3,2]benzodioxathiole' is the divalent-S ring, not
            # the ring of c1ccc2c(c1)O[SH2]O2); the caller keeps its von Baeyer spelling
            return None
    try:
        from .partial_saturation import _resolve_oxo_parent
        candidates = _resolve_oxo_parent(mol, ring, monocyclic=False)
    except Exception:  # noqa: BLE001 - a resolver that raises is a decline
        logger.info("fused_forms: the parent resolver raised; decline")
        return None
    from . import ring_hydrogen as rh_rule
    from .monocycle_forms import free_valence_carries_indicated_hydrogen
    needed = list(branch_atoms)
    if free_valence_atom is not None:
        needed.append(free_valence_atom)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    best = None
    for cand in candidates:
        if len(cand) != 3:
            continue  # an alternative-indicated-hydrogen parent: asked for a ketone only
        pname, pairs, _ih = cand
        if not pname or "hydro" in pname or " " in pname:
            continue
        stem = _IH_PREFIX.sub("", pname, count=1)
        if not stem or not (stem[0].isalpha() or stem[0].isdigit() or stem[0] == "["):
            continue
        for a2l, _parent_h in pairs:
            if set(a2l) != ring or any(not isinstance(a2l.get(a), int) for a in needed):
                continue
            try:
                rh = rh_rule.ring_hydrogen(mol, ring, (), a2l)
            except Exception:  # noqa: BLE001 - a structure the rule cannot read declines
                rh = None
            if rh is None or rh.added:
                continue
            if (free_valence_atom is not None and free_valence_atom not in rh.indicated
                    and free_valence_carries_indicated_hydrogen(rh.state, free_valence_atom)):
                # (the Blue Book): the free valence is put on the
                # mancude parent first and the indicated hydrogen goes on its atom
                # Example 5,:25462-:25477, '1,3-dioxo-1,3-dihydro-2H-isoindol-
                # 2-yl'), the hydro prefixes after
                try:
                    rh_fv = rh_rule.ring_hydrogen(mol, ring, (free_valence_atom,), a2l)
                except Exception:  # noqa: BLE001 - keep the lowest-locant placement
                    rh_fv = None
                if (rh_fv is not None and not rh_fv.added
                        and free_valence_atom in rh_fv.indicated):
                    rh = rh_fv
            pre = rh_rule.parent_prefix(rh, a2l)
            if pre is None:
                continue
            ih_key, _suffix_key, _added_key, hydro_key = rh_rule.numbering_key(rh, a2l, ())
            fv_key = ((locant_key(a2l[free_valence_atom]),)
                      if free_valence_atom is not None else ())
            prefix_key = tuple(sorted(locant_key(a2l[a]) for a in branch_atoms))
            tie = tuple(ranks[a] for a in sorted(ring, key=lambda x: locant_key(a2l[x])))
            key = (ih_key, fv_key, hydro_key, prefix_key, stem, tie)
            if best is None or key < best[0]:
                # (:16880) the hydro prefixes precede the indicated hydrogen;
                # a hyphen only before a stem that starts with a locant
                # ('4,5,6,7-tetrahydro-1,3-benzoxazole'), none before a bracket
                # ('3,4-dihydro[1,3]dioxolo[4′,5′:3,4]matridine', the Blue Book)
                glue = "-" if pre and not pre.endswith("-") and stem[:1].isdigit() else ""
                # `stem` is the catalogue's parent name (data, not an emitted name): a
                # digit in it is a fusion locant the name cites,:2869)
                cites_locant = bool(pre) or any(ch.isdigit() for ch in stem)
                best = (key, f"{pre}{glue}{stem}", dict(a2l), cites_locant)
    if best is None:
        return None
    from ..assembly.book_prefixes import note_book_form
    note_book_form()
    return FusedForm(parent=best[1], numbering=best[2],
                     parent_cites_locant=best[3])
