"""a phase oxime handler — Tier B retained-name (gate 0.40).

Verbatim lift of the oxime dispatch logic from composer.py:771-782
(inline branch) + composer.py:1906-2042 (body of _name_oxime_or_hydrazone).
The body itself STAYS in composer.py per internal notes incremental-migration
discipline — this handler module is a thin wrapper that invokes the
existing composer.py logic via lazy import. Plan-03 commit 03-10
(composer.py thinning) deletes the inline body from composer.py once
the inner-dispatch substrate is fully wired.

Byte-identical lock per internal notes (DECOMP-03): the handler's behavior
on every canary fixture MUST equal the inline branch's behavior bit-for-bit;
verified by `python scripts/verify_decomp_byte_identical.py --mode delta`
at the atomic commit gate.

IUPAC cite: (oxime functional class naming).

References:
- composer.py:771-782 (inline dispatch branch; REMOVED at this commit).
- composer.py:1906-2042 (_name_oxime_or_hydrazone body; STAYS until 03-10).
- internal notes-DECOMP.md row 'oxime' + predicate purity proof.
- internal notes (Tier-B lift handler pattern).
- 160-internal notes (atomic-commit byte-identical canary lock).
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_oxime(features: Any) -> bool:
    """Mirrors composer.py:771 (``features.principal_group == 'oxime'``).

    Pure read-only per internal notes / -26: reads
    ``features.principal_group`` attribute set by perception layer; no
    mutation of features, mol, or module-global state.
    """
    return getattr(features, 'principal_group', None) == 'oxime'


def name_oxime(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """a phase Tier-B oxime handler.

    Verbatim semantics of composer.py:771-782 (inline branch). Returns
    ``NamingResult(name=<final string>, tree=None, atom_to_locant_hint=None)``
    on success; ``None`` on gate-fail / not-applicable (pool gate
    threshold 0.40 per HANDLER_POLICIES['oxime']).

    Wave-1 strategy per internal notes: lazy-import the composer.py body
    (``_name_oxime_or_hydrazone``) and the enrichment helper
    (``_enrich_handler_name``); call them with the same arguments the
    inline branch used; route through the same ``pool.add`` call so
    a phase byte-identical lock methodology is preserved.

    The ``mol`` parameter is accepted for API uniformity per internal notes
     but not used here (composer.py:_name_oxime_or_hydrazone reads
    features.mol directly). The ``style`` parameter is similarly unused
    for first-wave Tier-B handlers (style only affects Pass-2 assembly,
    not Tier-B handlers).

    Args:
        features: MolecularFeatures object.
        mol: RDKit Mol object (not used in first-wave; reserved per).
        style: Naming style (not used in first-wave Tier-B; reserved per).

    Returns:
        NamingResult on success, or None on gate-fail. Per internal notes
        the ``tree`` field is None for first-wave handlers; the ``name``
        field is the byte-identical contract per DECOMP-03.
    """
    # Lazy imports per PATTERNS § Lazy Import (avoid composer.py -> handlers
    # -> composer.py cycle at module load).
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _enrich_handler_name,
        _inject_stereo_if_missing,
        _name_oxime_or_hydrazone,
    )

    # Wave2 (f), BB VERBATIM): 'Oximes are named substitutively
    # as N-hydroxy derivatives of imines and not by functional class
    # nomenclature as in previous recommendations' — 'CH3-CH2-CH=N-OH
    # propanal oxime... N-hydroxypropan-1-imine (PIN)'. Try the substitutive
    # PIN first; the functional-class form remains the fallback (and the
    # --trivial rendering) when the bounded substitutive builder declines.
    if style == "pin":
        _subst = _substitutive_oxime_name(features)
        if _subst:
            pool = get_current_pool()
            cand = pool.add(_subst, "oxime", features)
            if cand is not None:
                return NamingResult(
                    name=cand.name,
                    tree=NameTreeNode(
                        parent_stem=cand.name, class_id="oxime",
                        iupac_section_cite="P-66.6.5", fragment_legacy=cand.name,
                    ),
                    atom_to_locant_hint=None,
                )

    oxime_name = _name_oxime_or_hydrazone(features, 'oxime')
    if not oxime_name:
        return None

    oxime_name = _enrich_handler_name(features, oxime_name, "oxime")

    # a phase: route through pool.add — returns None on gate-fail.
    pool = get_current_pool()
    cand = pool.add(oxime_name, "oxime", features)
    if cand is None:
        return None

    # Per internal notes layering: this handler's inline branch at
    # composer.py:781 wrapped the name in _inject_stereo_if_missing — we
    # preserve that byte-identical behavior here. The NamingResult.name
    # field is the FINAL name (post-stereo-injection); the dispatch caller
    # returns it directly without further processing per the Plan-02
    # dispatch contract.
    final_name = _inject_stereo_if_missing(
        features, cand.name, atom_to_locant=None,
    )
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="oxime", iupac_section_cite="P-68.3.1.2", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


import re as _re

_PLAIN_IMINE_NAME_RE = _re.compile(r"^[a-z]+an(?:imine|-\d+-imine)$")

# W3-P15: a SUBSTITUTED alkane-imine parent — a block of
# numeric-locant detachable prefixes followed by the '<stem>an[-N-]imine' parent
# (e.g. '1-nitropropan-1-imine'). The stem alternation (longest-first) anchors
# the parent so a prefix ending in an alkane-like fragment is not mis-split.
_ALKANE_STEM_ALT = "|".join(
    ["pentadec", "tetradec", "tridec", "dodec", "undec", "dec", "non", "oct",
     "hept", "hex", "pent", "but", "prop", "eth", "meth"]
)
_SUBST_IMINE_RE = _re.compile(
    r"^(?P<pre>\d.*?)(?P<par>(?:cyclo)?(?:" + _ALKANE_STEM_ALT
    + r")an(?:-\d+-)?imine)$"
)
_SIMPLE_MULT_BY_COUNT = {2: "di", 3: "tri", 4: "tetra", 5: "penta"}


def _insert_n_hydroxy(sub: str) -> Optional[str]:
    """Insert the italic-N prefix 'N-hydroxy' alphanumerically among the
    detachable prefixes of a substituted alkane-imine name:
    '1-nitropropan-1-imine' -> 'N-hydroxy-1-nitropropan-1-imine'.

    Fail-closed (returns None) for any imine name that is not a plain block of
    numeric-locant prefixes on an '<stem>an-imine' parent (complex/bracketed
    prefixes, non-alkane parents, etc.) — those keep the functional-class
    oxime fallback."""
    from ..naming_utils import alpha_sort_key

    def _join(parts):
        # Insert a hyphen between adjacent prefix tokens when the previous ends
        # in a letter / closing bracket and the next starts with a LOCANT — a
        # digit (numeric locant, '1-nitro') OR an uppercase italic locant letter
        # ('N-hydroxy'). Substituent names are lowercase, so an uppercase start
        # unambiguously marks an italic locant.
        out = []
        for i, p in enumerate(parts):
            if i > 0 and out:
                prev, nxt = out[-1][-1], p[:1]
                if (prev.isalpha() or prev in ")]}") and (
                        nxt.isdigit() or (nxt.isalpha() and nxt.isupper())):
                    out.append("-")
            out.append(p)
        return "".join(out)

    m = _SUBST_IMINE_RE.match(sub)
    if not m:
        return None
    pre, par = m.group("pre"), m.group("par")
    # Tokenize the prefix block: a new token starts at a locant digit that
    # follows a letter or a closing bracket (mirror of _joined_prefix_parts).
    tokens, cur = [], ""
    for ch in pre:
        if ch.isdigit() and cur and (cur[-1].isalpha() or cur[-1] in ")]}"):
            tokens.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        tokens.append(cur)

    def _name_of(tok: str) -> Optional[str]:
        mm = _re.match(r"^([\dN,]+)-(.+)$", tok)
        if not mm:
            return None
        locs, rest = mm.group(1).split(","), mm.group(2)
        if "(" in rest or "[" in rest:
            return None                        # complex substituent -> defer
        mult = _SIMPLE_MULT_BY_COUNT.get(len(locs))
        if mult and rest.startswith(mult):
            rest = rest[len(mult):]
        return rest

    entries = []
    for tok in tokens:
        nm = _name_of(tok)
        if nm is None:
            return None
        entries.append((alpha_sort_key(nm), tok))
    entries.append((alpha_sort_key("hydroxy"), "N-hydroxy"))
    entries.sort(key=lambda e: e[0])
    return _join([e[1] for e in entries]) + par


def _substitutive_oxime_name(features: Any) -> Optional[str]:
    """Bounded substitutive-oxime builder: R2C=N-OH -> 'N-hydroxy<imine PIN>'.

    Derives the imine parent by deleting the oxime O (RWMol surgery) and
    re-entering the full namer — the route_charged / acyl-amido re-entry
    pattern — then prefixes 'N-hydroxy'.

    FAIL-CLOSED scope (returns None -> functional-class fallback, today's
    RT-valid behavior): exactly one oxime; no stereocenters / double-bond
    stereo (the surgered imine would mis-derive the descriptor: the C=N-OH
    E/Z priorities differ from C=N-H); and the recursive imine name must be
    a PLAIN unsubstituted parent ('ethanimine', 'propan-1-imine',
    'cyclohexan-1-imine') — a substituted imine name would need N-hydroxy
    alphabetized among its prefixes, deferred with the oxime-ether class.
    """
    mol = getattr(features, 'mol', None)
    matches = (getattr(features, 'functional_groups', None) or {}).get('oxime', [])
    if mol is None or len(matches) != 1:
        return None
    if getattr(features, 'stereocenters', None) or \
            getattr(features, 'double_bond_stereo', None):
        return None
    match = matches[0]  # SMARTS [CX3]=[NX2][OX2H] -> (C, N, O)
    if len(match) != 3:
        return None
    o_idx = match[2]
    if mol.GetAtomWithIdx(o_idx).GetSymbol() != 'O':
        return None
    from rdkit import Chem
    try:
        em = Chem.RWMol(mol)
        em.RemoveAtom(o_idx)
        imine_mol = em.GetMol()
        Chem.SanitizeMol(imine_mol)
        imine_smiles = Chem.MolToSmiles(imine_mol)
    except Exception:
        return None
    from ...namer import name_compound as _name_compound
    try:
        sub = _name_compound(imine_smiles)
    except Exception:
        return None
    if not sub:
        return None
    if _PLAIN_IMINE_NAME_RE.match(sub):
        joiner = "-" if sub[:1].isdigit() else ""
        return f"N-hydroxy{joiner}{sub}"
    # W3-P15: a SUBSTITUTED imine parent (nitrolic/nitrosolic
    # acids -> '1-nitropropan-1-imine' etc.) — insert 'N-hydroxy' alphanumerically
    # among its prefixes. Fail-closed to the functional-class oxime otherwise.
    return _insert_n_hydroxy(sub)


__all__ = ["name_oxime", "_is_oxime", "_substitutive_oxime_name"]
