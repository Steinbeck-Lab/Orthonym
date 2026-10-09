"""Substituent prefixes for mononuclear P and S oxoacid groups, read from the structure.

A group whose central atom is phosphorus or sulfur and that carries a terminal ``=O``
(``=S`` on phosphorus) is cited as a substituent prefix built from the acyl group of the
acid and the prefixes of its ligands "Compound and complex substituent
groups", the Blue Book;,:36484; "Substituent groups derived
from polyacids",:36937). The central atom is never a skeletal 'a' atom of a chain: a
phosphate or sulfonate written as '2-oxa-1-phosphaethyl' is a valid but unpreferred
spelling,:6465: a chain is not terminated by O).

``oxoacid_group_prefix(mol, frag_atoms, attach_idx, name_r)`` returns the prefix or ``None``
(fail closed: every atom of ``frag_atoms`` must be consumed by the group and the R
subtrees its ligands carry).

Acyl groups built (central atom X, ligands L):

    S(=O)(=O)(OH) sulfo S(=O)(=O)(O-) sulfonato
    S(=O)(OH) sulfino S(=O)(O-) sulfinato
    S(=O)(=O)(NH2) sulfamoyl S(=O)(=O)(OR) (alkoxy)sulfonyl
    S(=O)(=O)(hal) (halo)sulfonyl S(=O)(OR) (alkoxy)sulfinyl
    P(=O)(OH)2 phosphono P(=O)(O-)2 phosphonato
    P(=O)(L)(L') L(L')phosphoryl P(=S)(L)(L') L(L')phosphorothioyl

The attachment may be the central atom (the acyl prefix itself) or an O or NH that links it
to the parent ('sulfooxy', 'phosphonooxy', '[(methoxysulfonyl)oxy]', 'sulfoamino'). A ligand
that is an -O-P or -O-S group is cited as that group's acyl prefix plus 'oxy'
method (1),:36943).
"""
from __future__ import annotations

import contextvars
import re
from typing import Callable, List, NamedTuple, Optional, Set

from rdkit import Chem

#: True while a wider-tier call (complete / best-effort: ``Orthonym(general_fallback=True)``) runs.
#: ``general_fallback_ctx`` is reset inside the isolated naming sessions of the fragment
#: recursion, so the P/S prefixes would stop at the first nested fragment; this variable is
#: published with it (``namer.Orthonym.name``) and is not reset there. A default-tier call
#: and the strict twin publish False, so the PIN path never gives these prefixes.
ps_tier_ctx = contextvars.ContextVar("orthonym_ps_tier", default=False)

_HAL = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}
_CONTRACTED = {"sulfo", "sulfino", "sulfonato", "sulfinato", "sulfamoyl",
               "phosphono", "phosphonato"}
_MULT = {2: "di", 3: "tri", 4: "tetra"}
_MULT_BIS = {2: "bis", 3: "tris", 4: "tetrakis"}
_MAX_DEPTH = 8
_SIMPLE_LIGANDS = frozenset({"hydroxy", "oxido", "amino", "sulfanyl", "fluoro", "chloro",
                            "bromo", "iodo"})
_A_CHAIN = re.compile(r"\d(?:λ\d)?-(?:di|tri|tetra|penta)?(?:oxa|thia|aza|phospha)")


def has_replacement_chain(name: str) -> bool:
    """True if ``name`` carries a skeletal ('a') replacement term with its locant
    ('2-oxa', '1-phospha', '3,5-dioxa')."""
    return bool(_A_CHAIN.search(name or ""))


def _centre_candidate(mol, j: int) -> bool:
    """Phosphorus or sulfur, neutral, acyclic, carrying a terminal ``=O`` (``=S`` on P)
    and at least one ligand that is O, N or a halogen (an acid derivative, not a
    phosphine oxide, sulfoxide or sulfone)."""
    a = mol.GetAtomWithIdx(j)
    sym = a.GetSymbol()
    if sym not in ("P", "S") or a.GetFormalCharge() or a.GetNumRadicalElectrons():
        return False
    if a.IsInRing() or a.GetIsotope():
        return False
    oxo = 0
    hetero_lig = 0
    for nb in a.GetNeighbors():
        b = mol.GetBondBetweenAtoms(j, nb.GetIdx())
        dbl = b.GetBondType() == Chem.BondType.DOUBLE
        if dbl and nb.GetDegree() == 1 and nb.GetFormalCharge() == 0 and (
                nb.GetSymbol() == "O" or (sym == "P" and nb.GetSymbol() == "S")):
            oxo += 1
        elif b.GetBondType() == Chem.BondType.SINGLE and nb.GetSymbol() in (
                "O", "N", "F", "Cl", "Br", "I"):
            hetero_lig += 1
    return oxo >= 1 and hetero_lig >= 1


def _dummy_name_r(atoms, root):
    return "methyl"


def group_shape_ok(mol, j: int) -> bool:
    """True if the centre ``j`` has the exact shape ``_acyl`` can name, whichever of its
    single-bonded neighbours is the bond towards the parent: no H, no stereo tag, only
    terminal neutral ``=O`` (``=S`` on P), the oxo and ligand counts of the acid, and ligands
    from the set ``_ligand`` names (an N-N, isocyano or isocyanato ligand is not one). The
    answer is cached on the molecule."""
    cache = getattr(mol, "_oxoacid_shape", None)
    if cache is None:
        cache = {}
        try:
            mol._oxoacid_shape = cache
        except AttributeError:                                  # pragma: no cover
            pass
    if j in cache:
        return cache[j]
    ok = False
    if _centre_candidate(mol, j):
        everything = {a.GetIdx() for a in mol.GetAtoms()}
        for nb in mol.GetAtomWithIdx(j).GetNeighbors():
            if mol.GetBondBetweenAtoms(j, nb.GetIdx()).GetBondType() != Chem.BondType.SINGLE:
                continue
            ctx = _Ctx(mol, everything, _dummy_name_r)
            if _acyl(ctx, j, nb.GetIdx(), 0) is not None:
                ok = True
                break
    cache[j] = ok
    return ok


def is_oxoacid_centre(mol, j: int) -> bool:
    """A P or S oxoacid centre whose group the leaf can name (``group_shape_ok``): the
    one predicate of the spine exclusion and of the namer, so an atom is kept off the chain
    spine only where a prefix names it."""
    return group_shape_ok(mol, j)


def is_oxoacid_linker(mol, j: int) -> bool:
    """A divalent O, S or NH joining an oxoacid centre to one other heavy atom."""
    a = mol.GetAtomWithIdx(j)
    if a.GetSymbol() not in ("O", "N", "S") or a.GetFormalCharge() or a.IsInRing():
        return False
    if a.GetDegree() != 2 or a.GetIsotope():
        return False
    if a.GetSymbol() == "N" and a.GetTotalNumHs() != 1:
        return False
    return any(is_oxoacid_centre(mol, n.GetIdx()) for n in a.GetNeighbors()
               if mol.GetBondBetweenAtoms(j, n.GetIdx()).GetBondType()
               == Chem.BondType.SINGLE)


def _subtree(mol, start: int, block: int, frag: Set[int]) -> Optional[Set[int]]:
    """Atoms reached from ``start`` without crossing ``block``; ``None`` if the walk
    leaves ``frag`` or reaches ``block`` through a ring."""
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            k = nb.GetIdx()
            if k == block and cur != start:
                return None
            if k == block or k in seen:
                continue
            if k not in frag:
                return None
            seen.add(k)
            stack.append(k)
    return seen


def _enclose(tok: str) -> str:
    from ..assembly.naming_utils import apply_enclosing_marks
    return apply_enclosing_marks(tok, -1)


def _alkoxy(yl: Optional[str]) -> Optional[str]:
    if not yl or yl == "substituent" or yl.endswith("ylyl"):
        return None
    from ..assembly.substituent_enumerator import alkoxy_prefix_from_substituent
    return alkoxy_prefix_from_substituent(yl)


class _Lig(NamedTuple):
    """One ligand prefix: its ``token``, whether it is a compound prefix (enclosed and
    multiplied with bis/tris,, the Blue Book;,:7232) and, for an
    N-substituted amino ligand, the ``nsub`` string that goes in front of 'sulfamoyl'."""
    token: str
    compound: bool
    nsub: Optional[str] = None


def _simple_word(tok: str) -> bool:
    return tok in _SIMPLE_LIGANDS


def _lig_of(tok: str) -> _Lig:
    """A ligand token with its compound flag: the fixed one-word ligands are simple, any
    other token is compound when the shared predicate says so."""
    from ..assembly.naming_utils import is_complex_substituent
    return _Lig(tok, False if _simple_word(tok) else bool(is_complex_substituent(tok)))


class _Ctx:
    def __init__(self, mol, frag, name_r):
        self.mol, self.frag, self.name_r = mol, frag, name_r
        self.used: Set[int] = set()
        #: False once the prefix uses a spelling that is not the preferred one (see
        #: ``oxoacid_group_prefix_ex``)
        self.preferred = True


def _yl(ctx: _Ctx, atoms: Set[int], root: int) -> Optional[str]:
    yl = ctx.name_r(sorted(atoms), root)
    if not yl or yl == "substituent" or yl.endswith("ylyl"):
        return None
    return yl


def _ligand(ctx: _Ctx, x_idx: int, lig_idx: int, depth: int) -> Optional[_Lig]:
    """The prefix of one single-bonded ligand of the centre ``x_idx``."""
    mol = ctx.mol
    a = mol.GetAtomWithIdx(lig_idx)
    if lig_idx not in ctx.frag or a.GetIsotope() or a.GetNumRadicalElectrons():
        return None
    z = a.GetAtomicNum()
    on_p = mol.GetAtomWithIdx(x_idx).GetSymbol() == "P"
    if z in _HAL and a.GetDegree() == 1 and a.GetFormalCharge() == 0:
        ctx.used.add(lig_idx)
        ctx.preferred = ctx.preferred and not on_p      # phosphorodichloridoyl,:36166
        return _lig_of(_HAL[z])
    if a.GetSymbol() == "O":
        heavy = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != x_idx]
        if not heavy:
            if a.GetFormalCharge() == -1 and a.GetTotalNumHs() == 0:
                ctx.used.add(lig_idx)
                return _lig_of("oxido")
            if a.GetFormalCharge() == 0 and a.GetTotalNumHs() == 1:
                ctx.used.add(lig_idx)
                return _lig_of("hydroxy")
            return None
        if a.GetFormalCharge() or a.GetTotalNumHs() or len(heavy) != 1:
            return None
        nxt = mol.GetAtomWithIdx(heavy[0])
        if nxt.GetSymbol() in ("P", "S") and heavy[0] in ctx.frag:
            if depth >= _MAX_DEPTH:
                return None
            ctx.used.add(lig_idx)
            acyl = _acyl(ctx, heavy[0], lig_idx, depth + 1)
            if acyl is None:
                return None
            if nxt.GetSymbol() == "P":
                ctx.preferred = False                   # method (1) of (:36949)
            return _Lig(_oxy(acyl), True)
        if nxt.GetSymbol() == "C":
            sub = _subtree(mol, heavy[0], lig_idx, ctx.frag)
            if sub is None:
                return None
            alk = _alkoxy(_yl(ctx, sub, heavy[0]))
            if alk is None:
                return None
            ctx.used |= sub | {lig_idx}
            return _lig_of(alk)
        return None
    if a.GetSymbol() == "N":
        if a.GetFormalCharge() != 0 or a.IsInRing():
            return None
        ctx.preferred = ctx.preferred and not on_p      # phosphoramidoyl forms,:36178
        if a.GetDegree() == 1 and a.GetTotalNumHs() == 2:
            ctx.used.add(lig_idx)
            return _lig_of("amino")
        # an N-substituted amino ligand: 'methylamino', 'dimethylamino'
        heavy = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != x_idx]
        if (not heavy or len(heavy) > 2 or a.GetTotalNumHs() != 2 - len(heavy)
                or any(mol.GetAtomWithIdx(k).GetSymbol() != "C" for k in heavy)):
            return None
        subs = []
        used = {lig_idx}
        for k in heavy:
            sub = _subtree(mol, k, lig_idx, ctx.frag)
            if sub is None or sub & used:
                return None
            yl = _yl(ctx, sub, k)
            if yl is None:
                return None
            subs.append(_lig_of(yl))
            used |= sub
        joined = _join_ligands(subs)
        if joined is None:
            return None
        ctx.used |= used
        return _Lig(joined + "amino", True, joined)
    if (a.GetSymbol() == "S" and a.GetDegree() == 2 and a.GetFormalCharge() == 0
            and a.GetTotalNumHs() == 0 and mol.GetAtomWithIdx(x_idx).GetSymbol() == "P"):
        heavy = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != x_idx]
        if len(heavy) == 1 and mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "C":
            sub = _subtree(mol, heavy[0], lig_idx, ctx.frag)
            if sub is None:
                return None
            yl = _yl(ctx, sub, heavy[0])
            if yl is None:
                return None
            ctx.used |= sub | {lig_idx}
            ctx.preferred = False                       # a thio ester of the acid: infix form
            yl_c = _lig_of(yl)
            return _Lig((_enclose(yl) if yl_c.compound else yl) + "sulfanyl", True)
        return None
    if a.GetSymbol() == "S" and a.GetDegree() == 1 and a.GetFormalCharge() == 0 \
            and a.GetTotalNumHs() == 1 and mol.GetAtomWithIdx(x_idx).GetSymbol() == "P":
        ctx.used.add(lig_idx)
        return _lig_of("sulfanyl")
    if a.GetSymbol() == "C":
        # P-C: a phosphonoyl / phosphinoyl group, not the preferred spelling (:36244); S-C: an
        # 'R-sulfonyl' prefix (the sulfonamide named from its N side),
        sub = _subtree(mol, lig_idx, x_idx, ctx.frag)
        if sub is None:
            return None
        yl = _yl(ctx, sub, lig_idx)
        if yl is None:
            return None
        ctx.used |= sub
        if mol.GetAtomWithIdx(x_idx).GetSymbol() == "P":
            ctx.preferred = False
        return _lig_of(yl)
    return None


def _oxy(acyl: str) -> str:
    if acyl in _CONTRACTED:
        return acyl + "oxy"
    return _enclose(acyl) + "oxy"


def _join_ligands(ligs: List[_Lig]) -> Optional[str]:
    """Ligand prefixes in alphanumerical order, identical ones multiplied: simple ones
    with di/tri/tetra, compound ones with bis/tris/tetrakis and enclosing marks; every
    compound prefix is enclosed, the first one included, the Blue Book:
    'bis(dimethylamino) (preferred prefix)';,:7232). A simple prefix after the
    first is enclosed too ('hydroxy(sulfanyl)phosphorothioyl',:36337). ``None`` when a
    multiplicity has no multiplier."""
    from ..assembly.naming_utils import alpha_sort_key
    counts: dict = {}
    for l in ligs:
        counts[l] = counts.get(l, 0) + 1
    parts = []
    for l in sorted(counts, key=lambda x: alpha_sort_key(x.token)):
        n = counts[l]
        if n == 1:
            parts.append((l.token, l.compound))
        elif l.compound:
            if n not in _MULT_BIS:
                return None
            parts.append((_MULT_BIS[n] + _enclose(l.token), False))
        else:
            if n not in _MULT:
                return None
            parts.append((_MULT[n] + l.token, False))
    out = ""
    for k, (tok, compound) in enumerate(parts):
        out += _enclose(tok) if (compound or k > 0) else tok
    return out


def _acyl(ctx: _Ctx, x_idx: int, from_idx: Optional[int], depth: int) -> Optional[str]:
    """The acyl prefix of the centre ``x_idx`` (``from_idx`` = the bond towards the
    parent, left out)."""
    mol = ctx.mol
    if depth > _MAX_DEPTH or not _centre_candidate(mol, x_idx) or x_idx not in ctx.frag:
        return None
    x = mol.GetAtomWithIdx(x_idx)
    if x.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
        return None
    ctx.used.add(x_idx)
    oxo: List[int] = []
    ligs: List[int] = []
    for nb in x.GetNeighbors():
        k = nb.GetIdx()
        if k == from_idx:
            continue
        b = mol.GetBondBetweenAtoms(x_idx, k)
        if b.GetBondType() == Chem.BondType.DOUBLE:
            if not (nb.GetDegree() == 1 and nb.GetFormalCharge() == 0
                    and nb.GetTotalNumHs() == 0
                    and nb.GetSymbol() in ("O", "S")):
                return None
            oxo.append(k)
        elif b.GetBondType() == Chem.BondType.SINGLE:
            ligs.append(k)
        else:
            return None
    if x.GetTotalNumHs():
        return None
    sym = x.GetSymbol()
    if sym == "S":
        if any(mol.GetAtomWithIdx(k).GetSymbol() != "O" for k in oxo):
            return None
        if len(oxo) not in (1, 2) or len(ligs) != 1:
            return None
        ctx.used.update(oxo)
        lig = _ligand(ctx, x_idx, ligs[0], depth)
        if lig is None:
            return None
        if len(oxo) == 2:
            if lig.nsub is not None:
                return lig.nsub + "sulfamoyl"
            if lig.token in ("hydroxy", "oxido", "amino"):
                return {"hydroxy": "sulfo", "oxido": "sulfonato",
                        "amino": "sulfamoyl"}[lig.token]
            return (_enclose(lig.token) if lig.compound else lig.token) + "sulfonyl"
        if lig.token in ("hydroxy", "oxido"):
            return {"hydroxy": "sulfino", "oxido": "sulfinato"}[lig.token]
        return (_enclose(lig.token) if lig.compound else lig.token) + "sulfinyl"
    # phosphorus
    if len(oxo) != 1 or len(ligs) != 2:
        return None
    ctx.used.update(oxo)
    thio = mol.GetAtomWithIdx(oxo[0]).GetSymbol() == "S"
    lg = []
    for k in ligs:
        t = _ligand(ctx, x_idx, k, depth)
        if t is None:
            return None
        lg.append(t)
    toks = sorted(l.token for l in lg)
    if not thio and toks == ["hydroxy", "hydroxy"]:
        return "phosphono"
    if not thio and toks == ["oxido", "oxido"]:
        return "phosphonato"
    joined = _join_ligands(lg)
    if joined is None:
        return None
    if thio and toks == ["hydroxy", "hydroxy"]:
        ctx.preferred = False                           # thiophosphono,
    if any(l.token in ("hydroxy", "oxido") for l in lg):
        # an acid function left on the centre: whether the group may be cited as a prefix
        # depends on the parent's principal group,:36327), which this
        # producer does not see
        ctx.preferred = False
    return joined + ("phosphorothioyl" if thio else "phosphoryl")


def oxoacid_group_prefix(mol, frag_atoms, attach_idx: int,
                         name_r: Optional[Callable] = None) -> Optional[str]:
    """The prefix of the group (see ``oxoacid_group_prefix_ex``), or ``None``."""
    return oxoacid_group_prefix_ex(mol, frag_atoms, attach_idx, name_r)[0]


def oxoacid_group_prefix_ex(mol, frag_atoms, attach_idx: int,
                            name_r: Optional[Callable] = None):
    """``(prefix, preferred)``: the prefix of the P/S oxoacid group, or ``(None, True)``.

    A compound prefix that carries no enclosing mark of its own ('methoxysulfonyl',
    'sulfonatooxy') is returned enclosed, as the Blue Book cites it:
    '3-[(dimethoxyphosphoryl)sulfanyl]propanoic acid (PIN)', the Blue Book. The
    retained one-word prefixes stay bare.

    ``preferred`` is False when the spelling is not the preferred one, so the caller
    labels a name that holds it below the PIN (``record_non_pin_label``): a P-C, P-halogen
    or P-N ligand (phosphonoyl, phosphinoyl, phosphorodichloridoyl, phosphoramidoyl
    prefixes, -.5,:36166,:36178,:36244), a thio-ester ligand, '-P(S)(OH)2'
    (thiophosphono), method (1) of where method (2) applies (:36949), and an
    acid function left on the centre of a compound prefix (whether such a group may be a
    prefix depends on the parent's principal group,,:36327)."""
    tok, preferred = _group_prefix(mol, frag_atoms, attach_idx, name_r)
    if tok is None:
        return None, True
    if tok in _CONTRACTED or (tok[0] in "([{" and _fully_enclosed(tok)):
        return tok, preferred
    if any(m in tok for m in "([{"):
        return tok, preferred
    return _enclose(tok), preferred


def _fully_enclosed(tok: str) -> bool:
    from ..assembly.naming_utils import _is_fully_enclosed
    return _is_fully_enclosed(tok)


def _has_anhydride_ligand(mol, centre: int, from_idx: int) -> bool:
    """True if the P centre carries an -O-P ligand."""
    c = mol.GetAtomWithIdx(centre)
    if c.GetSymbol() != "P":
        return False
    for nb in c.GetNeighbors():
        if nb.GetIdx() == from_idx or nb.GetSymbol() != "O":
            continue
        if any(x.GetSymbol() == "P" and x.GetIdx() != centre for x in nb.GetNeighbors()):
            return True
    return False


def _group_prefix(mol, frag_atoms, attach_idx: int,
                  name_r: Optional[Callable] = None):
    """The prefix of the P/S oxoacid group ``frag_atoms`` attached at ``attach_idx``, or
    ``None``. ``name_r(atoms, attach)`` names an R subtree as a '-yl' prefix."""
    frag = set(frag_atoms)
    if attach_idx is None or attach_idx not in frag:
        return None, True
    if name_r is None:
        def name_r(atoms, att):                                  # noqa: E306
            from ..assembly.substituent_enumerator import name_substituent
            return name_substituent(mol, set(atoms), att, True)
    ctx = _Ctx(mol, frag, name_r)
    a = mol.GetAtomWithIdx(attach_idx)
    outside = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() not in frag]
    if len(outside) != 1:
        return None, True
    try:
        if is_oxoacid_centre(mol, attach_idx):
            bond = mol.GetBondBetweenAtoms(attach_idx, outside[0])
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None, True
            tok = _acyl(ctx, attach_idx, outside[0], 0)
        elif is_oxoacid_linker(mol, attach_idx):
            inner = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() in frag]
            if len(inner) != 1:
                return None, True
            if a.GetSymbol() == "O" and _has_anhydride_ligand(mol, inner[0], attach_idx):
                # (the Blue Book): method (2), the diphosphoxane parent, is
                # the PIN (:36949 '3-[(1,3,3-trihydroxy-1,3-dioxo-1lambda5,3lambda5-
                # diphosphoxan-1-yl)oxy]propanoic acid (PIN)'); method (1) below is the
                # alternative the book prints beside it
                from .phosphorus import name_phosphoxane_oxy_substituent
                pin = name_phosphoxane_oxy_substituent(mol, attach_idx, outside[0])
                if pin is not None:
                    return pin, True
            ctx.used.add(attach_idx)
            acyl = _acyl(ctx, inner[0], attach_idx, 0)
            if acyl is None:
                return None, True
            if a.GetSymbol() == "O":
                tok = _oxy(acyl)
            else:
                end = "amino" if a.GetSymbol() == "N" else "sulfanyl"
                tok = (acyl + end if acyl in _CONTRACTED else _enclose(acyl) + end)
        else:
            return None, True
    except RecursionError:
        return None, True
    if tok is None or ctx.used != frag:
        return None, True
    return tok, ctx.preferred


def note_ps_form(token: str, preferred: bool) -> None:
    """A producer gave this P/S group prefix in place of its 'a' chain: record the book
    form (so the mechanical retry exists) and, when the spelling is not the preferred one,
    the label-only non-PIN record (a name that holds ``token`` is labelled below the PIN).
    The one place the lane reports a form to the retry ladder."""
    from ..assembly.book_prefixes import note_book_form
    note_book_form('ps')
    if not preferred:
        from ..metrics.provenance import record_non_pin_label
        record_non_pin_label(
            token, rule="P-67.1.4.1.1",
            detail="the preferred prefix of an acid group is derived from the preferred "
                   "name of the acid (P-67.1.4.1.1.3, BlueBookV2.md:36058); this spelling "
                   "is a valid systematic one")
