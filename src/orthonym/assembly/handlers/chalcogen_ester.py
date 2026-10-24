"""a phase chalcogen-ester handler — Tier FRN-E functional-class naming.

Handles selenoester (R-C(=O)-Se-R') and telluroester (R-C(=O)-Te-R') per
IUPAC + functional-replacement nomenclature. Emits names of the
form "X-{alkyl} {chain}X{oate}" where X is "Se" or "Te":

    CC(=O)[Se]C -> "Se-methyl ethaneselenoate"
    CCC(=O)[Se]C -> "Se-methyl propaneselenoate"
    CCC(=O)[Se]CC -> "Se-ethyl propaneselenoate"
    CCC(=O)[Te]C -> "Te-methyl propanetelluroate"

Structurally mirrors handlers/imidate.py:
- Predicate consults features.principal_group so the handler defers to higher-
  seniority groups (acid / amide / etc.).
- Universal substituent enumeration for the alkyl side.
- Local longest-carbon-chain DFS + composer._integrate_universal_prefixes for
  the chain (acyl) side with branched/substituted support.

References:
- handlers/imidate.py (Tier FRN-D predicate pattern)
- handlers/isothiocyanate.py (Tier-B functional-class direct-return template)
- functional_groups.py selenoester / telluroester SMARTS (Plan-02 commit 98e46cbb)
- internal notes-FRN.md + + (audit baseline; a phase closure
  ships the dedicated handler that was originally backlogged).
"""
from __future__ import annotations

from typing import Any, Optional

from rdkit import Chem

from ..name_tree import NameTreeNode, NamingResult

# Map principal-group FG name -> ("X" prefix token, suffix token).
_CHALCOGEN_ESTER_TOKENS = {
    # W3-P08: thioester R-C(=O)-S-R' -> "S-alkyl alkanethioate".
    # Reuses the identical engine as selenoester/telluroester (SMARTS
    # "[CX3](=O)[SX2][#6]" has the same (C,O,S,C_alkyl) match order). The
    # thioic-S-acid suppression in functional_groups.py keeps C(=O)SH out.
    "thioester":     ("S", "thioate"),
    "selenoester":   ("Se", "selenoate"),
    "telluroester":  ("Te", "telluroate"),
}


def _is_chalcogen_ester(features: Any) -> bool:
    """Predicate: selenoester OR telluroester is the principal group.

    Mirrors handlers/imidate._is_imidate predicate contract. Fires when
    features.principal_group identifies a chalcogen ester AND no higher-
    seniority group outranks it. Side-effect-free per.
    """
    pg = getattr(features, "principal_group", None)
    if pg not in _CHALCOGEN_ESTER_TOKENS:
        return False
    fg = getattr(features, "functional_groups", None) or {}
    return bool(fg.get(pg))


def name_chalcogen_ester(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """a phase selenoester / telluroester functional-class handler.

    Algorithm:
    1. Identify (C_carbonyl, O, X_chalcogen, C_alkyl) from the SMARTS match.
    2. Render the alkyl side via substituent_enumerator.name_substituent so
       branched/substituted alkyls (isopropyl, benzyl,...) render correctly.
    3. Render the chain side by walking the longest carbon chain through
       C_carbonyl and enumerating branch substituents via
       composer._integrate_universal_prefixes (same path as anhydrides + ester
       chain renderers).
    4. Emit "{X}-{alkyl} {chain-prefix}{chain-stem}{X-suffix}".
    """
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _enrich_handler_name,
        _inject_stereo_if_missing,
    )

    pg = features.principal_group
    tokens = _CHALCOGEN_ESTER_TOKENS.get(pg)
    if tokens is None:
        return None
    x_prefix, x_suffix = tokens

    matches = features.functional_groups.get(pg, [])
    if not matches:
        return None
    if mol is None:
        mol = getattr(features, "mol", None)
    if len(matches) == 2 and mol is not None:
        name = _name_bis_chalcogen_diester(features, mol, matches, x_prefix, x_suffix)
        if name is not None:
            pool = get_current_pool()
            cand = pool.add(name, "chalcogen_ester", features)
            if cand is None:
                return None
            return NamingResult(
                name=cand.name,
                tree=NameTreeNode(parent_stem=cand.name, class_id="chalcogen_ester",
                                  iupac_section_cite="P-65.6.3.3.7.1",
                                  fragment_legacy=cand.name),
                atom_to_locant_hint=None,
            )
    match = matches[0]
    if len(match) < 4:
        return None
    c_carbonyl_idx, o_idx, x_idx, c_alkyl_idx = match[0], match[1], match[2], match[3]

    if mol is None:
        mol = getattr(features, "mol", None)
    if mol is None:
        return None

    # Branch 1: alkyl word ({X}-{alkyl}) - render the R' on the chalcogen side.
    alkyl_atoms = _collect_subgraph(mol, c_alkyl_idx, exclude={x_idx})
    alkyl_word = _name_alkyl_side(mol, alkyl_atoms, anchor=c_alkyl_idx)
    if alkyl_word is None:
        return None

    # Branch 2: chain stem with {X-suffix} - render the acyl (chain) side.
    stem_word = _name_chalcogen_chain(
        mol, c_carbonyl_idx, exclude={o_idx, x_idx, c_alkyl_idx},
        x_suffix=x_suffix,
    )
    if stem_word is None:
        return None

    # Compose: "{X}-{alkyl} {chain}{X-suffix}" - e.g., "Se-methyl propaneselenoate".
    # A compound or locant-bearing alkyl word is enclosed after the element locant
    #, the Blue Book; '*S*-(2-cyanoethyl) cyclohexanesulfinothioate
    # (PIN)':31765, '*S*-(trimethylgermyl) ethanesulfonothioate (PIN)':31659), never
    # 'S-2-cyanoethyl'; a simple one ('S-ethyl hexanethioate (PIN)',:31989) is bare.
    from ..naming_utils import enclose_if_compound
    # Stereodescriptors stand "immediately at the front of the part of the name to
    # which they relate"; for a substituent group "at the front of the
    # corresponding prefix" "NAMING OF STEREOISOMERS", the Blue Book).
    # An ester names its acid and its organyl group as separate words, each with
    # its own descriptors: '(2S)-butan-2-yl (4S)-4-chlorohexanoate (PIN)' (:46715),
    # 'ethyl (R)-4-nitrobenzene-1-sulfinate (PIN)' (:46268). So the acyl chain's
    # descriptors go in front of the acid word ('S-ethyl (2E)-but-2-enethioate'),
    # and the organyl word carries its own (name_substituent cites them:
    # 'S-[(3S)-3-hydroxybutyl]'); nothing goes in front of the whole name
    # ('(2E)-S-[(3S)-3-hydroxybutyl] but-2-enethioate' and '(2R)-S-[butan-2-yl]
    # ethanethioate' were built so). When the organyl word cites fewer
    # descriptors than it has stereo elements, the front block of the whole name
    # is kept (as before).
    acid_side = set(_collect_subgraph(mol, c_carbonyl_idx,
                                      {o_idx, x_idx, c_alkyl_idx}))
    acid_centres = [sc for sc in (features.stereocenters or [])
                    if sc['idx'] in acid_side]
    acid_bonds = [db for db in (getattr(features, 'double_bond_stereo', None) or [])
                  if set(db.get('atoms') or ()) <= acid_side]
    organyl_elements = (
        len(features.stereocenters or []) - len(acid_centres)
        + len(getattr(features, 'double_bond_stereo', None) or []) - len(acid_bonds))
    import re as _re
    organyl_cited = organyl_elements <= sum(
        len(block.split(',')) for block in _re.findall(
            r"\((\d*[RSrsEZ](?:,\d*[RSrsEZ])*)\)-", alkyl_word))
    acid_stereo_placed = bool(acid_centres or acid_bonds or organyl_elements) and organyl_cited
    if acid_stereo_placed and (acid_centres or acid_bonds):
        from types import SimpleNamespace
        acid_features = SimpleNamespace(
            mol=mol, stereocenters=acid_centres, double_bond_stereo=acid_bonds)
        stem_word = _inject_stereo_if_missing(
            acid_features, stem_word,
            atom_to_locant={a: i + 1 for i, a in enumerate(
                _find_longest_carbon_chain(mol, c_carbonyl_idx, acid_side))},
            parent_scope='chain')
    name = f"{x_prefix}-{enclose_if_compound(alkyl_word)} {stem_word}"
    name = _enrich_handler_name(features, name, "chalcogen_ester")

    pool = get_current_pool()
    cand = pool.add(name, "chalcogen_ester", features)
    if cand is None:
        return None

    final_name = cand.name if acid_stereo_placed else _inject_stereo_if_missing(
        features, cand.name, atom_to_locant=None,
    )
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="chalcogen_ester", iupac_section_cite="P-65.6", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


def _name_bis_chalcogen_diester(features: Any, mol: Any, matches: list,
                                x_prefix: str, x_suffix: str) -> Optional[str]:
    """The diester of an unbranched alkanedicarbothioic (-seleno, -telluro) S-acid
    with two identical organyl groups: (the Blue Book)
    'CH3-S-CO-CO-S-CH3 S,S-dimethyl ethanebis(thioate) (PIN)' (:32005). The acid
    is 'ethanebis(thioic S-acid)': (b) (:7106) cites 'thioic acid' and its
    Se and Te analogues with 'bis', as in 'butanebis(thioic acid) (PIN)' (:30237).

    Fail closed (None, the single-ester path then runs) unless: the two ester
    carbons end one unbranched saturated acyclic carbon chain, the two organyl
    groups are separate branches with the same name, every heavy atom is one of
    those, and the molecule has no stereo element."""
    from ..naming_utils import enclose_if_compound, multiplied_component

    if getattr(features, 'stereocenters', None) or getattr(
            features, 'double_bond_stereo', None):
        return None
    (c1, o1, x1, a1), (c2, o2, x2, a2) = (tuple(m[:4]) for m in matches)
    alk1 = set(_collect_subgraph(mol, a1, exclude={x1}))
    alk2 = set(_collect_subgraph(mol, a2, exclude={x2}))
    if c1 in alk1 or c2 in alk2 or alk1 & alk2:
        return None
    acid = set(_collect_subgraph(mol, c1, exclude={o1, x1, o2, x2}))
    if c2 not in acid or acid & (alk1 | alk2):
        return None
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if heavy != acid | alk1 | alk2 | {o1, x1, o2, x2}:
        return None
    # one unbranched saturated carbon chain from c1 to c2
    for idx in acid:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C' or atom.IsInRing() or atom.GetFormalCharge():
            return None
        n_in = sum(1 for nb in atom.GetNeighbors() if nb.GetIdx() in acid)
        if n_in != (1 if idx in (c1, c2) else 2):
            return None
    for bond in mol.GetBonds():
        if (bond.GetBeginAtomIdx() in acid and bond.GetEndAtomIdx() in acid
                and bond.GetBondType() != Chem.BondType.SINGLE):
            return None
    w1 = _name_alkyl_side(mol, tuple(sorted(alk1)), anchor=a1)
    w2 = _name_alkyl_side(mol, tuple(sorted(alk2)), anchor=a2)
    if not w1 or w1 != w2:
        return None
    try:
        from ...data.chain_names import get_chain_prefix
        stem = get_chain_prefix(len(acid))
    except (ValueError, KeyError, ImportError):
        return None
    if not stem:
        return None
    organyl = multiplied_component(2, w1, enclose_if_compound(w1))
    return f"{x_prefix},{x_prefix}-{organyl} {stem}anebis({x_suffix})"


def _collect_subgraph(
    mol: Any, anchor_idx: int, exclude: "set[int]",
) -> "tuple[int, ...]":
    """BFS-walk a fragment from anchor, halting at excluded atoms. Pure."""
    visited: "set[int]" = set()
    stack = [anchor_idx]
    while stack:
        idx = stack.pop()
        if idx in visited or idx in exclude:
            continue
        visited.add(idx)
        for bond in mol.GetAtomWithIdx(idx).GetBonds():
            other_idx = bond.GetOtherAtomIdx(idx)
            if other_idx not in visited and other_idx not in exclude:
                stack.append(other_idx)
    return tuple(sorted(visited))


def _name_alkyl_side(
    mol: Any, atoms: "tuple[int, ...]", anchor: int,
) -> Optional[str]:
    """Render the {X}-alkyl-side via substituent_enumerator.name_substituent.

    Handles retained names (methyl, ethyl, isopropyl, benzyl, phenyl,...) +
    systematic + recursive compound names automatically.
    """
    if not atoms:
        return None
    try:
        from ..substituent_enumerator import name_substituent
        return name_substituent(mol, set(atoms), anchor)
    except Exception:
        return None


def _name_chalcogen_chain(
    mol: Any, anchor_idx: int, exclude: "set[int]", x_suffix: str,
) -> Optional[str]:
    """Render the acyl (chain) side with {x_suffix} (selenoate / telluroate).

    Walks the longest carbon chain from C_carbonyl excluding the =O / chalcogen
    / alkyl-side atoms. Enumerates branch substituents via the composer's
    universal prefix integrator so branched + substituted chains render
    correctly (mirrors handlers/imidate._name_chain_with_imidate_suffix).
    """
    # Collect the candidate fragment (chain side atoms only).
    frag = _collect_subgraph(mol, anchor_idx, exclude)
    if not frag:
        return None

    principal_chain = _find_longest_carbon_chain(mol, anchor_idx, set(frag))
    chain_len = len(principal_chain)
    if chain_len < 1:
        return None

    # An acyl carbon bonded to a ring heteroatom is the carboxy group of a ring parent
    #, the Blue Book: "The carboxy group can be attached to any atom, carbon
    # or heteroatom, of any parent hydride"; the chalcogen analogues take the 'carbothioic',
    # 'carboselenoic' and 'carbotelluroic' suffixes with the name of the parent hydride,,
    #:30270), not a methane substituted by the ring.
    ring_on_acyl = _ring_heteroatom_on(mol, anchor_idx, set(frag)) if chain_len == 1 else None
    if ring_on_acyl is not None:
        stem = _ring_carbo_stem(mol, anchor_idx, set(frag), ring_on_acyl, x_suffix)
        if stem is not None:
            return stem

    # Chain prefix: methane / ethane / propane / butane / pentane /...
    try:
        from ...data.chain_names import get_chain_prefix
        chain_prefix = get_chain_prefix(chain_len)
    except (ValueError, KeyError, ImportError):
        return None

    # Chain multiple bonds are part of the parent name: the
    # 'dodeca-2,8-diene' of an unsaturated acyl chain. It was always written
    # '-ane' ('7,11-dihydroxydodecanethioate' for a dodeca-2,8-dienethioate,
    # a different molecule the round trip refused). C1 is the ester carbon.
    from rdkit import Chem
    double_locants, triple_locants = [], []
    for pos in range(chain_len - 1):
        bond = mol.GetBondBetweenAtoms(principal_chain[pos], principal_chain[pos + 1])
        if bond is None:
            return None
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            double_locants.append(pos + 1)
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            triple_locants.append(pos + 1)
        elif bond.GetBondType() != Chem.BondType.SINGLE:
            return None  # aromatic / other: not an acyclic chain
    from ..composition_primitives import _build_unsaturation_infix
    infix = _build_unsaturation_infix(double_locants, triple_locants)

    # IUPAC: 'e' of '{prefix}ane' / '-ene' / '-yne' preserved because
    # '-thioate' / '-selenoate' / '-telluroate' start with consonants.
    base = f"{chain_prefix}{infix}e{x_suffix}"

    chain_set = set(principal_chain)
    has_branches = any(
        sum(
            1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
            if n.GetIdx() in set(frag) and n.GetIdx() not in chain_set
        ) > 0
        for idx in principal_chain
    )
    if not has_branches:
        return base

    atom_to_locant = {idx: pos + 1
                      for pos, idx in enumerate(principal_chain)}
    all_atom_idxs = set(range(mol.GetNumAtoms()))
    exclude_atoms = all_atom_idxs - set(frag)

    try:
        from ..composer import _integrate_universal_prefixes
        prefix_str = _integrate_universal_prefixes(
            mol, chain_set,
            parent_type="chain",
            principal_chain=principal_chain,
            atom_to_locant=atom_to_locant,
            exclude_atoms=exclude_atoms,
        )
    except Exception:
        prefix_str = ""

    result = f"{prefix_str}{base}" if prefix_str else base
    if ring_on_acyl is not None:
        # the methane name is valid but is not the name of the ring parent: kept for the wider
        # tiers, recorded as not the preferred name
        from ...metrics.provenance import record_non_pin_label
        record_non_pin_label(result, rule="P-65.1.2.2",
                             detail="the carboxy group on a ring heteroatom is a suffix of the ring parent")
    return result


def _ring_heteroatom_on(
    mol: Any, carbonyl_idx: int, frag_atoms: "set[int]",
) -> Optional[int]:
    """The ring heteroatom bonded to the acyl carbon ``carbonyl_idx`` within the acyl side, or
    ``None``."""
    atom = mol.GetAtomWithIdx(carbonyl_idx)
    for nbr in atom.GetNeighbors():
        k = nbr.GetIdx()
        if k in frag_atoms and nbr.IsInRing() and nbr.GetAtomicNum() not in (1, 6):
            return k
    return None


def _ring_carbo_stem(
    mol: Any, carbonyl_idx: int, frag_atoms: "set[int]", ring_atom: int, x_suffix: str,
) -> Optional[str]:
    """'aziridine-1-carbothioate' for an acyl carbon on the heteroatom ``ring_atom`` of a bare
    single ring: the name of the parent hydride, the locant of the free valence and the 'carbo'
    suffix, the Blue Book: 'pyrrolidine-1-carboxylic acid (PIN)'). The parent
    and its numbering are those of ``monocycle_form``. ``None`` for a ring that is not a bare
    heteromonocycle: the caller keeps the methane name and records it."""
    from ...rules.monocycle_forms import monocycle_form
    ring = next((tuple(r) for r in mol.GetRingInfo().AtomRings() if ring_atom in r), None)
    if ring is None or set(ring) | {carbonyl_idx} != set(frag_atoms):
        return None
    form = monocycle_form(mol, ring, ring_atom)
    if form is None or form.kind != "heteromonocycle":
        return None
    loc = form.numbering.get(ring_atom)
    if loc is None or loc == form.fv_hydro_locant:
        return None
    return f"{form.parent}-{loc}-carbo{x_suffix}"


def _find_longest_carbon_chain(
    mol: Any, start: int, frag_atoms: "set[int]",
) -> list:
    """DFS longest simple carbon path from `start` within `frag_atoms`. Pure."""
    carbon_set = {
        i for i in frag_atoms
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
    }
    best_path = [start]

    def dfs(current: int, visited: "set[int]", path: list) -> None:
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = list(path)
        for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in visited or nidx not in carbon_set:
                continue
            visited.add(nidx)
            path.append(nidx)
            dfs(nidx, visited, path)
            path.pop()
            visited.discard(nidx)

    dfs(start, {start}, [start])
    return best_path


__all__ = ["name_chalcogen_ester", "_is_chalcogen_ester"]
