"""functional-group perception fix (a phase) hydroxylamine handler.

Names substituted hydroxylamines on the retained parent hydride ``hydroxylamine``
(H2N-OH,. Substituents on the nitrogen take the ``N-`` locant; those
on the oxygen take ``O-``:

    CCCNO -> N-propylhydroxylamine
    CCCN(O)C -> N-methyl-N-propylhydroxylamine
    CON -> O-methylhydroxylamine
    CN(C)O -> N,N-dimethylhydroxylamine

Without this handler the perceived ``hydroxylamine`` FG (added to
``functional_groups.py`` in 169.7) is dropped and the molecule names as the bare
carbon chain (``CCCNO`` -> ``propane``), silently losing both heteroatoms.

The substituent names come from the shared ``name_substituent_fragment`` namer;
multiplying affixes + alphanumeric ordering reuse the standard prefix helpers.

IUPAC cite:.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, List, Optional, Set, Tuple

from ..name_tree import NameTreeNode, NamingResult
from ..naming_utils import strip_italicized_structural_prefix
from ...perception.molcache import atoms_of, bonds_of

logger = logging.getLogger(__name__)

# Sentinel returned by ``_name_n_carbon_hydroxylamine_as_amine`` for an N-carbon
# hydroxylamine core (R-NH-OH / RR'N-OH) it RECOGNISES but must fail closed on
# (a base amine name carrying a leading stereodescriptor -- D1). It means
# "abstain the WHOLE handler", distinct from ``None`` ("try the next branch"):
# returning ``None`` would fall through to the functional-class branch, which
# ships the same malformed / double-descriptor stereo name the fix removes.
_ABSTAIN = object()


def _is_hydroxylamine(features: Any) -> bool:
    """Fire when hydroxylamine is the principal characteristic group, OR when the
    ONLY functional group is 'aminooxy' (H2N-O-R) and the whole molecule is a
    hydroxylamine derivative with no more-senior parent: the
    O-substituted-N-bare form's PIN is O-substituted hydroxylamine, e.g.
    CON -> O-methylhydroxylamine, NOT the prefix 'aminooxy...' which applies only
    when a senior parent is present), OR when the molecule is an N,O-disubstituted
    hydroxylamine, whose PIN is an O-substituted AMINE."""
    if getattr(features, "principal_group", None) == "hydroxylamine":
        return True
    if _is_pure_aminooxy_hydroxylamine(features):
        return True
    mol = getattr(features, "mol", None)
    return mol is not None and no_disub_hydroxylamine_core(mol) is not None


def no_disub_hydroxylamine_core(mol) -> Optional[Tuple[int, int]]:
    """Return ``(n_idx, o_idx)`` for an N,O-DISUBSTITUTED hydroxylamine core
    (R-NH-O-R' or R2N-O-R'), else None. The PIN of such a compound is an
    O-substituted AMINE, and skeletal ('a') replacement is
    explicitly forbidden (BB note).

    Fail-closed to None off this exact class: a single acyclic neutral N-O single
    bond where the N bears >=1 carbon (besides O) AND the O bears exactly one
    carbon (no O-H), with N/O the ONLY heteroatoms and every other heavy atom a
    carbon (pure hydrocarbyl substituents), no multiple bond on N/O. This keeps
    the simpler O-substituted (H2N-O-R) and N-substituted (R-NH-OH) forms — whose
    PIN is the hydroxylamine parent — out (there the O bears H, or the N bears
    only H)."""
    from rdkit import Chem
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    n_atoms = [a for a in atoms_of(mol) if a.GetSymbol() == "N"]
    o_atoms = [a for a in atoms_of(mol) if a.GetSymbol() == "O"]
    if len(n_atoms) != 1 or len(o_atoms) != 1:
        return None
    # N and O are the only heteroatoms; everything else must be carbon.
    for a in atoms_of(mol):
        if a.GetSymbol() not in ("C", "N", "O"):
            return None
        if a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() != 0:
            return None
    n, o = n_atoms[0], o_atoms[0]
    if n.IsInRing() or o.IsInRing():
        return None
    bond = mol.GetBondBetweenAtoms(n.GetIdx(), o.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    # No double bond anywhere touching N or O (would be oxime/nitroso/etc.).
    for b in bonds_of(mol):
        if b.GetBondType() != Chem.BondType.SINGLE:
            syms = {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()}
            if syms & {"N", "O"}:
                return None
    # N bears >=1 carbon (besides O); O bears exactly one carbon (no O-H).
    n_carbons = [nb for nb in n.GetNeighbors()
                 if nb.GetSymbol() == "C"]
    o_carbons = [nb for nb in o.GetNeighbors()
                 if nb.GetSymbol() == "C"]
    if len(n_carbons) < 1 or len(o_carbons) != 1:
        return None
    if o.GetTotalNumHs() != 0:
        return None
    return (n.GetIdx(), o.GetIdx())


def _is_pure_aminooxy_hydroxylamine(features: Any) -> bool:
    """True when the molecule is exactly a hydroxylamine derivative perceived
    only as 'aminooxy' (H2N-O-R): a single N-O bond, N neutral acyclic (the
    'aminooxy' SMARTS [NX3H2][OX2][#6] fixes N as bare -NH2, so ALL
    substitution is on the O side), O itself acyclic, and NO other functional
    group / no more-senior PCG. Fail-closed for anything richer.

     (BB:5005) keeps the hydroxylamine parent for the
    O-substituted-only form even when R is an aryl group (O-phenylhydroxylamine
    for H2N-O-C6H5) -- so a ring reachable ONLY through the O-substituent is
    allowed, provided it is a genuine AROMATIC benzo ring (checked via
    ``GetIsAromatic``, not merely 'in a ring'), never the non-aromatic
    Kekule-drawn fallback some upstream SMILES could produce."""
    if getattr(features, "principal_group", None) is not None:
        return False
    fgs = getattr(features, "functional_groups", None) or {}
    active = [k for k, v in fgs.items() if v]
    if active != ["aminooxy"]:
        return False
    mol = getattr(features, "mol", None)
    if mol is None:
        return False
    # Exactly one N and one O forming the hydroxylamine core, both neutral;
    # no other hetero.
    n_atoms = [a for a in atoms_of(mol) if a.GetSymbol() == "N"]
    o_atoms = [a for a in atoms_of(mol) if a.GetSymbol() == "O"]
    if len(n_atoms) != 1 or len(o_atoms) != 1:
        return False
    for a in atoms_of(mol):
        if a.GetSymbol() not in ("C", "N", "O"):
            return False
        if a.GetFormalCharge() != 0:
            return False
    n_atom, o_atom = n_atoms[0], o_atoms[0]
    # N (bare -NH2 per the SMARTS) and O themselves are never ring atoms; any
    # ring present belongs entirely to the O-substituent R (e.g. O-phenyl).
    if n_atom.IsInRing() or o_atom.IsInRing():
        return False
    n_idx, o_idx = n_atom.GetIdx(), o_atom.GetIdx()
    bond = mol.GetBondBetweenAtoms(n_idx, o_idx)
    from rdkit import Chem
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    # Every ring present must be a genuine aromatic benzo ring -- the
    # O-substituent namer (_named_substituents -> name_substituent_fragment)
    # only knows how to spell a real aromatic phenyl/aryl, never a
    # non-aromatic Kekule-drawn carbocycle.
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) != 6 or not all(
            mol.GetAtomWithIdx(i).GetSymbol() == "C"
            and mol.GetAtomWithIdx(i).GetIsAromatic()
            for i in ring
        ):
            return False
    # No C=O / C=N (would be a senior oxime/amide territory).
    for b in bonds_of(mol):
        if b.GetBondType() == Chem.BondType.DOUBLE:
            syms = {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()}
            if syms & {"N", "O"}:
                return False
    return True


def _collect_substituent_fragment(mol, start_idx: int, block_idx: int) -> List[int]:
    """BFS the substituent fragment rooted at ``start_idx``, never crossing
    ``block_idx`` (the hydroxylamine N or O)."""
    seen: Set[int] = set()
    queue = deque([start_idx])
    while queue:
        idx = queue.popleft()
        if idx in seen or idx == block_idx:
            continue
        seen.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx != block_idx and nidx not in seen:
                queue.append(nidx)
    return sorted(seen)


def _named_substituents(mol, center_idx: int, other_center_idx: int) -> List[str]:
    """Name every carbon-rooted substituent on the hydroxylamine center atom
    ``center_idx`` (N or O), excluding the bond to ``other_center_idx``."""
    from ..substituent_naming import name_substituent_fragment

    names: List[str] = []
    center = mol.GetAtomWithIdx(center_idx)
    for nbr in center.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == other_center_idx:
            continue
        if nbr.GetSymbol() != "C":
            continue  # only carbon substituents take a locant; H is the parent
        frag = _collect_substituent_fragment(mol, nidx, center_idx)
        sub_name = name_substituent_fragment(mol, frag, nidx, [center_idx])
        if sub_name:
            names.append(sub_name)
    return names


def _format_locant_block(names: List[str], locant: str) -> List[Tuple[str, str]]:
    """Group identical substituents into ``(sort_key, "locant,locant-prefixname")``
    tuples with the simple multiplying affix (di/tri/...).

    : a COMPOUND/substituted substituent prefix (``4-methylphenyl``)
    takes enclosing marks; a SIMPLE one (``phenyl``, ``methyl``) does not --
    reuse the shared ``enclose_if_compound`` primitive rather than hand-roll
    the distinction (BB ``O-(4-methylphenyl)hydroxylamine``, not
    ``O-4-methylphenylhydroxylamine``). The sort key stays the bare,
    unenclosed name."""
    from ..naming_utils import SIMPLE_MULTIPLIERS  # {2: 'di', 3: 'tri',...}
    from ..naming_utils import enclose_if_compound

    out: List[Tuple[str, str]] = []
    # Count duplicates preserving first-seen order.
    counts: dict = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    for sub_name, count in counts.items():
        enclosed = enclose_if_compound(sub_name)
        locs = ",".join([locant] * count)
        if count > 1:
            mult = SIMPLE_MULTIPLIERS.get(count, "")
            # (d) (the Blue Book, 'di-*tert*-butyl':6964): a
            # hyphen after the multiplier before a bare italicized prefix, none
            # before an enclosed one,:6968) -- the shared primitive.
            from ..naming_utils import multiplier_needs_hyphen
            if mult and multiplier_needs_hyphen(enclosed):
                mult += "-"
            term = f"{locs}-{mult}{enclosed}"
        else:
            term = f"{locant}-{enclosed}"
        out.append((sub_name, term))
    return out


# verbatim, the complete retained R-O– contraction list ("Some
# contracted names are retained for R-O– substituent groups... they are used both
# as preferred IUPAC prefixes"): methoxy, ethoxy, propoxy, butoxy, phenoxy, and
#
# (CH3)3C-O– *tert*-butoxy (preferred prefix) (no substitution)
#
# so the (CH3)3C-O– prefix is 'tert-butoxy', NOT 'tert-butyloxy' (the index at
# the Blue Book spells the rejection out: "tert-butoxy* (unsubstituted) =
# (2-methylpropan-2-yl)oxy = 1,1-dimethylethoxy (not tert-butyloxy)", and:55671
# "tert-butyloxy: see tert-butoxy*"), and certainly not the over-enclosed
# '(tert-butyl)oxy' this module used to emit.
#
# Every key is matched EXACTLY and every value is 's "no substitution" /
# fully-substitutable form as cited, so a substituted R (which arrives spelled with
# locants, e.g. '2-methylpropan-2-yl') can never reach a contraction it is not
# entitled to.
_CONTRACTED_ALKOXY = {
    "methyl": "methoxy", "ethyl": "ethoxy", "propyl": "propoxy",
    "butyl": "butoxy", "phenyl": "phenoxy",
    "tert-butyl": "tert-butoxy",
}


def _alkoxy_prefix(mol, o_idx: int, r_c_idx: int) -> Optional[str]:
    """Name the ``-O-R'`` group as an alkoxy/aryloxy substituent prefix
    : methyl -> methoxy, ethyl -> ethoxy,...; longer / complex R'
    -> ``{R'}oxy`` or ``({R'})oxy``. None if R' cannot be named."""
    from ..substituent_naming import name_substituent_fragment
    frag = _collect_substituent_fragment(mol, r_c_idx, o_idx)
    alkyl = name_substituent_fragment(mol, frag, r_c_idx, [o_idx])
    if not alkyl or not alkyl.endswith("yl"):
        return None
    if alkyl in _CONTRACTED_ALKOXY:
        return _CONTRACTED_ALKOXY[alkyl]
    # / carve-out, via the SHARED primitive rather than the raw
    # `"-" in alkyl` this line used to carry: the hyphen of a leading italicized
    # structural prefix is not a compound boundary, so it must not draw enclosing
    # marks. But grants a retained -oxy contraction to exactly the six
    # groups in _CONTRACTED_ALKOXY above and REVOKES the others by name —
    # "The prefixes '*sec*-butoxy' and 'isobutoxy' are no longer recommended", with
    # the PIN being '(butan-2-yl)oxy'. So an italicized-led R that is NOT in the
    # table has no spelling this function is entitled to emit ('sec-butyloxy' is
    # rejected verbatim at the Blue Book) and the only correct one,
    # '(butan-2-yl)oxy', requires a different name for R than the one handed in.
    # Fail closed rather than invent a rejected contraction; in practice the
    # substituent namer already returns the locanted 'butan-2-yl', which falls
    # through to the enclosed form below and is the PIN.
    _, _had_italicized = strip_italicized_structural_prefix(alkyl)
    if _had_italicized:
        return None
    if any(ch.isdigit() for ch in alkyl) or "-" in alkyl or "(" in alkyl:
        return f"({alkyl})oxy"
    return f"{alkyl}oxy"


def _name_no_disub_hydroxylamine(features: Any) -> Optional[NamingResult]:
    """: an N,O-disubstituted hydroxylamine (R-NH-O-R') is named
    as an O-substituted AMINE — the N-carbon skeleton is the amine parent and the
    ``-O-R'`` is an N-(R'-oxy) substituent: CNOC -> N-methoxymethanamine;
    C6H5-NH-O-CH2CH3 -> N-ethoxyaniline. Skeletal ('a') replacement is forbidden
    for these (BB note), so this must outrank it.

    SCOPE (fail-closed -> None): the N bears EXACTLY ONE carbon (the clean
    R-NH-O-R' parent-amine case); R2N-O-R' (a second N-carbon needing N-locant
    disambiguation among N-substituents) is deferred."""
    from rdkit import Chem
    mol = features.mol
    core = no_disub_hydroxylamine_core(mol)
    if core is None:
        return None
    n_idx, o_idx = core
    n = mol.GetAtomWithIdx(n_idx)
    o = mol.GetAtomWithIdx(o_idx)
    n_carbons = [nb.GetIdx() for nb in n.GetNeighbors() if nb.GetSymbol() == "C"]
    if len(n_carbons) != 1:
        return None                                  # R2N-O-R' -> defer
    r_c = [nb.GetIdx() for nb in o.GetNeighbors() if nb.GetSymbol() == "C"][0]

    alkoxy = _alkoxy_prefix(mol, o_idx, r_c)
    if not alkoxy:
        return None

    # Build the parent amine (R-NH2) by deleting the O atom (which severs -O-R'),
    # then name the N-bearing fragment via the full namer.
    try:
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(o_idx)
        frag_mol = rw.GetMol()
        Chem.SanitizeMol(frag_mol)
    except Exception:
        return None
    pieces = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
    amine_piece = None
    for p in pieces:
        if any(a.GetSymbol() == "N" for a in p.GetAtoms()):
            amine_piece = p
            break
    if amine_piece is None:
        return None
    from ...namer import name_compound as _name_compound
    try:
        base = _name_compound(Chem.MolToSmiles(amine_piece))
    except Exception:
        return None
    # The fragment is a clean primary amine (R-NH2), so its PIN is an amine parent
    # ('methanamine', 'ethanamine', 'aniline',...). Reject only a failure sentinel.
    if not base or "unknown" in base or "not supported" in base:
        return None
    name = f"N-{alkoxy}{base}"

    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    pool = get_current_pool()
    pool.add(name, "hydroxylamine", features)
    final_name = _inject_stereo_if_missing(features, pool.best().name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=base, fragment_legacy=final_name,
            class_id="hydroxylamine", iupac_section_cite="P-68.3.1.1.1.3",
        ),
        atom_to_locant_hint=None,
    )


def _name_n_carbon_hydroxylamine_as_amine(features: Any):
    """ (BB:38308): R-NH-OH or RR'N-OH (the N bears >=1 carbon --
    the perceived 'hydroxylamine' FG SMARTS [OX2H1][NX3...][#6] guarantees this
    for every match) is named as an *N*-derivative of the senior AMINE, NOT by
    hydroxylamine functional-class nomenclature:

        CNO -> N-hydroxymethanamine (:38314, PIN)
        CN(C)O -> N-hydroxy-N-methylmethanamine (:38316, PIN)
        ONCCl -> chloro-N-hydroxymethanamine alpha order)

    Returns a ``NamingResult``, or ``None`` ("not my case -> try the next
    branch"), or the module ``_ABSTAIN`` sentinel ("this IS an N-carbon
    hydroxylamine but no well-formed PIN can be built -- abstain the whole
    handler, never fall through to the functional-class branch"; see the
    leading-stereodescriptor case below).

    Built by the SAME recursive re-entry the sibling builder
    (``_name_no_disub_hydroxylamine`` above) and the oxime handler's
    substitutive builder (``oxime.py::_substitutive_oxime_name``) already use:
    delete the -OH oxygen, re-derive the PIN of the resulting plain amine
    through the FULL namer (only it knows chain/ring parent selection), then
    insert the 'N-hydroxy' prefix -- RE-ANCHORED against an independently
    computed candidate substituent name, never a blind string edit.

    Routing this through the general amine N-substituent walker instead
    (``composer.py::_walk_amine_n_substituents``) is NOT safe (a project rule
    probe): that walker silently DROPS any N-substituent fragment with zero
    carbons -- a bare -OH on N vanishes from its returned list with no ``None``
    sentinel (measured: chain_set=set on 'CNO' returns ``['methyl']``, the
    oxygen branch simply missing) -- which would emit the structure-wrong
    'methanamine' for CNO with the -OH silently gone. A wrong-direction
    hydroxylamine name is bad; a silent atom drop is worse, so this handler
    builds the amine name itself rather than declining into that walker.

    FAIL-CLOSED scope (returns None -> caller's existing, non-worse fallback):
    more than one hydroxylamine FG match; the O does not carry exactly 1 H;
    N bears 0 or >2 carbons (excluded by the trivalent-N/single-O-H bond
    structure, kept as defense); the O-deleted fragment fails to (re-)name or
    names as a failure sentinel; a lone-carbon (primary-amine) base with a
    leading locant (e.g. '2-chloroethanamine' -- longer chain / ring, ordering
    deferred) or a compound / enclosed-mark C-prefix; or (2-carbon,
    secondary-amine base) neither independently-named candidate N-substituent is
    found as the base name's leading token.

    ABSTAIN scope (returns ``_ABSTAIN`` -> whole handler abstains): a lone-carbon
    base carrying a LEADING (or nested) stereodescriptor -- prepending 'N-hydroxy'
    there yields a malformed / unparseable / double-descriptor name, and the
    functional-class fallback ships the same, so neither may fire (D1).
    """
    matches = features.functional_groups.get("hydroxylamine", [])
    if not matches:
        return None
    # SMARTS [OX2H1][NX3...][#6] binds once per carbon neighbour of N, so an
    # RR'N-OH core yields TWO raw matches (one per R) sharing one (O, N) pair
    # -- dedupe on that pair and require exactly one hydroxylamine core.
    on_pairs = {(m[0], m[1]) for m in matches}
    if len(on_pairs) != 1:
        return None
    mol = features.mol
    o_idx, n_idx = next(iter(on_pairs))
    o = mol.GetAtomWithIdx(o_idx)
    n = mol.GetAtomWithIdx(n_idx)
    if o.GetSymbol() != "O" or n.GetSymbol() != "N" or o.GetTotalNumHs() != 1:
        return None
    n_carbons = [nb.GetIdx() for nb in n.GetNeighbors() if nb.GetSymbol() == "C"]
    if len(n_carbons) not in (1, 2):
        return None

    from rdkit import Chem
    try:
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(o_idx)
        frag_mol = rw.GetMol()
        Chem.SanitizeMol(frag_mol)
    except Exception:
        return None
    pieces = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
    amine_piece = None
    for p in pieces:
        if any(a.GetSymbol() == "N" for a in p.GetAtoms()):
            amine_piece = p
            break
    if amine_piece is None:
        return None
    from ...namer import name_compound as _name_compound
    try:
        base = _name_compound(Chem.MolToSmiles(amine_piece))
    except Exception:
        return None
    if not base or "unknown" in base or "not supported" in base:
        return None

    if len(n_carbons) == 1:
        from ..naming_utils import alpha_sort_key
        from ...rules.stereochemistry import strip_stereo
        # D1 / 0-wrong: a base carrying a leading (or nested) stereodescriptor
        # cannot be spelled by prepending 'N-hydroxy' without a malformed /
        # OPSIN-unparseable / double-descriptor name -- the correct
        # '(nS)-N-hydroxy-...' PIN needs proper stereo placement (a broad
        # refactor, deferred). Fail closed for the WHOLE handler (via _ABSTAIN):
        # returning None would fall through to the functional-class branch, which
        # ships the same malformed stereo. ON[C@@H](C)CC, ON[C@@H](C)c1ccccc1.
        if strip_stereo(base) != base:
            return _ABSTAIN
        # (the Blue Book): 'N-hydroxy' (an N-locanted detachable prefix) and a
        # leading C-substituent prefix are cited in alphanumerical order. The
        # only substituted primary amine whose C-prefix locant is fully omitted
        # (a)) is 'methanamine' -- every longer chain / ring gives the
        # base a leading locant -- so a non-bare base of the form
        # '{simple-prefix}methanamine' is merged here (ONCCl -> chloro < hydroxy
        # -> 'chloro-N-hydroxymethanamine'); anything richer defers.
        _PARENT = "methanamine"
        if base != _PARENT and base.endswith(_PARENT):
            cprefix = base[:-len(_PARENT)]
            # The amine piece keeps its N-H, so its C-prefix may cite the locant '1'
            #, the Blue Book; '1-hydrazinylmethanamine (PIN)', the Blue Book):
            # '1,1,1-trichloromethanamine'. The locant set of a one-carbon parent is
            # all '1's; it stays with its prefix ('1,1,1-trichloro-N-hydroxy...').
            cloc = ""
            _head, _sep, _tail = cprefix.partition("-")
            if _sep and _head and set(_head.split(",")) == {"1"}:
                cloc, cprefix = f"{_head}-", _tail
            if (not cprefix[:1].isalpha()) or any(
                    ch in "([{0123456789" for ch in cprefix):
                return None  # compound C-prefix -- ordering deferred
            if alpha_sort_key(cprefix) < alpha_sort_key("hydroxy"):
                # C-prefix cited first; the following 'N-' locant takes a hyphen.
                name = f"{cloc}{cprefix}-N-hydroxy{_PARENT}"
            elif cloc:
                # 'N-hydroxy' cited first, then the locanted C-prefix
                # ('N-hydroxy-1-phenylmethanamine').
                name = f"N-hydroxy-{cloc}{cprefix}{_PARENT}"
            else:
                # 'N-hydroxy' cited first; the C-prefix starts with a letter and
                # juxtaposes with no hyphen (N-hydroxyphenylmethanamine).
                name = f"N-hydroxy{cprefix}{_PARENT}"
        elif base[:1].isdigit():
            return None  # leading locant (longer chain / ring) -- ordering deferred
        else:
            name = f"N-hydroxy{base}"
    else:
        from ..naming_utils import _wrap_n_substituent, alpha_sort_key, enclose_if_compound

        name = None
        for cand in dict.fromkeys(_named_substituents(mol, n_idx, o_idx)):
            token = f"N-{_wrap_n_substituent(enclose_if_compound(cand))}"
            if base.startswith(token):
                parent = base[len(token):]
                if alpha_sort_key("hydroxy") <= alpha_sort_key(cand):
                    name = f"N-hydroxy-{token}{parent}"
                else:
                    name = f"{token}-N-hydroxy{parent}"
                break
        if name is None:
            return None

    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    pool = get_current_pool()
    pool.add(name, "hydroxylamine", features)
    final_name = _inject_stereo_if_missing(features, pool.best().name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=base, fragment_legacy=final_name,
            class_id="hydroxylamine", iupac_section_cite="P-68.3.1.1.1.1",
        ),
        atom_to_locant_hint=None,
    )


def name_hydroxylamine(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Name a substituted hydroxylamine on the ``hydroxylamine`` parent, or an
    N,O-disubstituted hydroxylamine as an O-substituted amine,
    or an N-carbon-substituted hydroxylamine as an N-hydroxy amine derivative
    ."""
    #: N,O-disubstituted hydroxylamine -> O-substituted amine
    # (skeletal 'a'-replacement forbidden). Try this first; fail-closed otherwise.
    _no_disub = _name_no_disub_hydroxylamine(features)
    if _no_disub is not None:
        return _no_disub
    #: R-NH-OH / RR'N-OH -> N-hydroxy derivative of the senior
    # amine, NOT the hydroxylamine functional class. Try before the
    # functional-class branch below (which would otherwise wrongly claim this
    # same FG match and emit the non-PIN 'N-methylhydroxylamine' direction).
    _amine_form = _name_n_carbon_hydroxylamine_as_amine(features)
    if _amine_form is _ABSTAIN:
        return None  # N-carbon core with a stereo-descriptor base: abstain, do
        # NOT fall through to the functional-class branch (which ships malformed
        # / double-descriptor stereo -- D1).
    if _amine_form is not None:
        return _amine_form
    m = features.mol
    matches = features.functional_groups.get("hydroxylamine", [])
    if matches:
        # SMARTS [OX2H1][NX3...][#6] -> match[0]=O, match[1]=N.
        match = matches[0]
        o_idx, n_idx = match[0], match[1]
    else:
        # Pure aminooxy whole-molecule case (H2N-O-R -> O-substituted
        # hydroxylamine,: locate the single N-O core.
        amx = features.functional_groups.get("aminooxy", [])
        if not amx or not _is_pure_aminooxy_hydroxylamine(features):
            return None
        # SMARTS [NX3H2][OX2][#6] -> match[0]=N, match[1]=O.
        match = amx[0]
        n_idx, o_idx = match[0], match[1]
    if m.GetAtomWithIdx(o_idx).GetSymbol() != "O" or m.GetAtomWithIdx(n_idx).GetSymbol() != "N":
        return None

    n_subs = _named_substituents(m, n_idx, o_idx)   # N-locant
    o_subs = _named_substituents(m, o_idx, n_idx)   # O-locant
    if not n_subs and not o_subs:
        return None  # bare hydroxylamine has no carbon substituents — leave to default

    # Alphanumeric order on the substituent name; each term already
    # carries its leading O-/N- locant, so adjacent terms hyphen-join and the parent
    # attaches directly (e.g. "N-ethyl-N-methylhydroxylamine", "N-propylhydroxylamine").
    terms = _format_locant_block(o_subs, "O") + _format_locant_block(n_subs, "N")
    terms.sort(key=lambda t: t[0].lstrip("([{").lower())
    name = "-".join(t[1] for t in terms) + "hydroxylamine"

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "HANDLER_COVERAGE: handler=hydroxylamine coverage=NA accounted=NA/%d name=%s",
            m.GetNumHeavyAtoms(), name[:60],
        )

    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    pool = get_current_pool()
    pool.add(name, "hydroxylamine", features)
    final_name = _inject_stereo_if_missing(features, pool.best().name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem="hydroxylamine", fragment_legacy=final_name,
            class_id="hydroxylamine", iupac_section_cite="P-68.3.1.2.1",
        ),
        atom_to_locant_hint=None,
    )


__all__ = [
    "name_hydroxylamine", "_is_hydroxylamine", "no_disub_hydroxylamine_core",
]
