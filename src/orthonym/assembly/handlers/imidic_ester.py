"""Esters of imidic acids that the =NH oxygen-ester handler (``imidate``) does not own.

IUPAC cite: "Esters of acids modified by functional replacement nomenclature",
 (the Blue Book-31999): 'methyl ethanimidate (PIN)', 'ethyl propanehydrazonate
(PIN)', 'methyl benzenecarboximidothioate (PIN)', 'ethyl N-sulfanylbenzenecarboximidothioate (PIN)';
 'ethyl N,N-dimethylcarbamimidothioate (PIN)'. (the Blue Book):
imidic and hydrazonic acids follow the carboxylic acids in class 7a, so their esters are class 9, senior
to every group that is not an acid, an anhydride or an ester.

What owned these molecules before: ``ester_family`` (the sulfur ester came out as
'1-(methylsulfanyl)ethan-1-imine'), the ``imine`` handler (an N-substituted ester came out as
'1-ethoxy-N-methylethan-1-imine') and the ``hydrazone`` handler ('(1-ethoxypropylidene)hydrazine'),
each shipped as ``pin_verified``. A name for the same structure that puts the ester oxygen or sulfur on
the imine carbon as a prefix is a systematic name, not the PIN.

Scope (anything else returns None and the old path runs unchanged, so no row loses a name):
one acyclic ``R-C(=N-X)-Y-R'`` group, Y = O or S, no other principal group, a neutral molecule whose
stereo elements (if any) all lie in the organyl group R' (it cites them, 'N-... (2S)-2-chloropropyl
cyclohexanecarboximidothioate'); R = H, an acyclic carbon chain whose first carbon is saturated (its unsaturation is
cited by 'ene'/'yne'; the chain is the longest,, an unsubstituted benzene ring, or any
other ring, whose stem is the name of its acid with '-ic acid' turned into '-ate' ('methyl cyclohexane-
carboximidothioate', 'ethyl 4-chlorobenzene-1-carboximidothioate'; ``_stem_from_acid``), or, where the
engine has no name for that acid (a heteroaromatic or fused ring), the name of the carboxylic acid with the
same ring and group position, whose '-carboxylic acid' becomes the suffix of the ester ('methyl pyridine-4-
carboximidothioate'; ``_stem_from_carboxylic_acid``)
(R = N gives the S-ester of carbamimidic acid); X = H, a carbon group, NH2 (the O-ester hydrazonate), or
OH "Carbohydroximic acids": 'N-hydroxyethanimidic acid (PIN)', so 'methyl
N-hydroxyethanimidate'). The =NH O-esters and the O-carbamimidates stay with ``imidate`` (priority
2900). Registered at priority 90, ahead of ``oxime`` (100), ``hydrazone`` (200) and every handler
listed above.

PURE per: predicate and builder only read ``mol``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

from ...rules.locanted_prefixes import locanted_prefix_block
from ..name_tree import NameTreeNode, NamingResult
from .imidate import _collect_subgraph, _name_alkyl_fragment, _name_carbamimidate

# (C, =N, O or S, alkyl C): the imidic ester group; every atom outside a ring
_IMIDIC_ESTER = Chem.MolFromSmarts("[CX3;!R](=[NX2;!R])[OX2,SX2;!R][#6]")
_C_DOUBLE_N = Chem.MolFromSmarts("[#6]=[#7]")

# stem endings by the acid they come from: 'imidic', 'imidothioic', 'hydrazonic' (the 'e' of the
# alkane is elided before a vowel only: ethanimidate, propanehydrazonate)
_CHAIN_SUFFIX = {"imidate": "animidate", "imidothioate": "animidothioate", "hydrazonate": "anehydrazonate"}
_RING_SUFFIX = {"imidate": "carboximidate", "imidothioate": "carboximidothioate",
                "hydrazonate": "carbohydrazonate"}


def _heavy_neighbors(atom, exclude: "set[int]") -> list:
    return [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in exclude]


def _parts(mol: Any) -> Optional[Dict[str, Any]]:
    """The one imidic ester group of ``mol`` classified, or None when it is not this handler's."""
    if mol is None or Chem.GetFormalCharge(mol) != 0:
        return None
    if any(a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms()):
        return None
    hits = mol.GetSubstructMatches(_IMIDIC_ESTER)
    if len(hits) != 1 or len(mol.GetSubstructMatches(_C_DOUBLE_N)) != 1:
        return None                       # a second C=N would be left unnamed
    c, n, y, alkyl = hits[0]
    stereo = {a.GetIdx() for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED}
    for b in mol.GetBonds():
        if b.GetStereo() != Chem.BondStereo.STEREONONE:
            stereo.update((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    if stereo:
        # only the organyl group R' names its own stereo elements; one in the stem or on the N would be cited
        # apart from the acid's descriptors, which this builder does not compose
        alkyl_atoms = set(_collect_subgraph(mol, alkyl, {y}))
        if c in alkyl_atoms or not stereo <= alkyl_atoms:
            return None
    c_at, n_at, y_sym = mol.GetAtomWithIdx(c), mol.GetAtomWithIdx(n), mol.GetAtomWithIdx(y).GetSymbol()
    others = _heavy_neighbors(c_at, {n, y})
    if len(others) > 1:
        return None
    r = others[0] if others else None
    stem = "H" if r is None else ("N" if r.GetSymbol() == "N" else ("C" if r.GetSymbol() == "C" else None))
    if stem is None:
        return None
    nx = _heavy_neighbors(n_at, {c})
    if not nx:
        x = "H"
    elif len(nx) == 1 and nx[0].GetSymbol() == "C":
        x = "C"
    elif len(nx) == 1 and nx[0].GetSymbol() == "N" and nx[0].GetTotalNumHs() == 2 and nx[0].GetDegree() == 1:
        x = "N"
    elif len(nx) == 1 and nx[0].GetSymbol() == "O" and nx[0].GetTotalNumHs() == 1 and nx[0].GetDegree() == 1:
        x = "O"                           # N-hydroxy "Carbohydroximic acids")
    else:
        return None
    if stem == "N":
        # the S-ester of carbamimidic acid; the amino N must be acyclic; no hydrazonate
        if y_sym != "S" or x in ("N", "O") or r.IsInRing() or r.GetIsAromatic():
            return None
    elif y_sym == "O" and x == "H" and not (
            stem == "C" and r.IsInRing() and not _plain_benzene(mol, r)):
        return None                       # the =NH O-ester: imidate names it (a ring stem it has no name for stays here)
    elif y_sym == "S" and x == "N":
        return None                       # 'hydrazonothioate' has no example in the book: the old path runs
    if stem == "C" and not _stem_ok(mol, r):
        return None
    return {"c": c, "n": n, "y": y, "alkyl": alkyl, "r": None if r is None else r.GetIdx(),
            "stem": stem, "x": x, "n_sub": nx[0].GetIdx() if x == "C" else None,
            "y_sym": y_sym}


def _plain_benzene(mol: Any, r) -> bool:
    """R is a carbon of an unsubstituted benzene ring."""
    if not r.GetIsAromatic():
        return False
    ring = [list(rg) for rg in mol.GetRingInfo().AtomRings() if r.GetIdx() in rg]
    if len(ring) != 1 or len(ring[0]) != 6:
        return False
    for idx in ring[0]:
        at = mol.GetAtomWithIdx(idx)
        if at.GetSymbol() != "C" or not at.GetIsAromatic() or at.GetDegree() != (3 if idx == r.GetIdx() else 2):
            return False
    return True


def _stem_ok(mol: Any, r) -> bool:
    """R is a saturated acyclic carbon (a chain) or a ring carbon: the plain benzene ring has its stem here, any
    other ring is named from its acid (:func:`_stem_from_acid`) or the builder declines."""
    return r.IsInRing() or r.GetHybridization() == Chem.HybridizationType.SP3


def _name_acid_strict(acid_smiles: str) -> Optional[str]:
    """``name_compound`` of an acid that is the stem of an ester, kept only where the strict PIN path built it.

    At the best-effort tier the call falls to the general engine for an acid the strict path cannot name, and
    that engine stamps its source on the provenance of the whole call: the ester, which this handler builds from
    the strict name alone, then shipped labelled as a general-engine name ('methyl naphthalene-2-
    carboximidothioate' was 'systematic_verified' at the best-effort tier and 'pin_verified' at the default
    tier). The source, OPSIN and gate records of the nested call are undone; the name-scoped records stay (the
    stem carries what the acid name carries). A name of the general engine is no stem: None."""
    from ...errors import is_failure_name
    from ...metrics import provenance as pv
    from ...namer import name_compound

    before = pv.get_provenance()
    keys = ("source", "opsin", "gate_outcome", "gate_outcome_name", "stereo_unexpressed")
    pv.restore_provenance(dict(before, source=None))
    try:
        name = name_compound(acid_smiles)
        built_by = pv.get_provenance().get("source")
    finally:
        pv.restore_provenance(dict(pv.get_provenance(), **{k: before.get(k) for k in keys}))
    if not name or is_failure_name(name) or built_by == "general_engine":
        return None
    return name


def _stem_from_acid(mol: Any, c: int, y: int, alkyl: int, suffix: str) -> Optional[str]:
    """The '-ate' stem of a ring-stem imidic ester from the name of its acid.

     "Esters of acids modified by functional replacement nomenclature" (the Blue Book),
     (:31983): "names of esters are all derived from acids modified by functional replacement
    whose substitutive names are systematically formed" ('methyl benzenecarboximidothioate (PIN)',:31997);
     (:31663) "All preferred IUPAC names for esters are named by functional class nomenclature".
    The acid is the ester with its organyl group R' replaced by H; it is named by the strict PIN path
    ('4-chlorobenzene-1-carboximidothioic acid') and its '-ic acid' becomes '-ate'. None (the builder declines)
    when the organyl group is not a separate branch, when the molecule has a stereo element (the acid name
    would cite it apart from the organyl group's), when the acid has no name, or when its name does not end in
    the one acid word of ``suffix``."""
    from ...rules.esters import _acid_name_to_ate

    alkyl_atoms = set(_collect_subgraph(mol, alkyl, {y}))
    if c in alkyl_atoms:
        return None
    rw = Chem.RWMol(mol)
    for idx in sorted(alkyl_atoms, reverse=True):
        rw.RemoveAtom(idx)
    try:
        acid = rw.GetMol()
        Chem.SanitizeMol(acid)
        acid_smiles = Chem.MolToSmiles(acid)
    except Exception:  # noqa: BLE001 -- an unbuildable acid is no stem
        return None
    acid_name = _name_acid_strict(acid_smiles)
    if not acid_name:
        return None
    acid_word = suffix[: -len("ate")] + "ic acid"          # 'imidothioate' -> 'imidothioic acid'
    if not acid_name.endswith(acid_word) or acid_name.count(acid_word[: -len(" acid")]) != 1:
        return None
    return _acid_name_to_ate(acid_name)


def _stem_from_carboxylic_acid(mol: Any, p: Dict[str, Any], suffix: str) -> Optional[str]:
    """The '-ate' stem of a ring-stem imidic ester from the name of the carboxylic acid of the same ring.

     "Imidic acids" / "Hydrazonic acids" (carboximidic and carbohydrazonic acids take the
    place of the carboxylic acid on the same carbon of the same ring, and
    (the Blue Book): the ring, its prefixes and the position of the group are those of the acid
    'pyridine-4-carboxylic acid', and the suffix of the ester is '-carboximidothioate'
    ('methyl pyridine-4-carboximidothioate'). Used where the engine has no name for the imidic or hydrazonic acid
    itself (:func:`_stem_from_acid`). Only the group without a substituent on the N (=NH, or the NH2 of the
    hydrazonate): an N group is cited with the ring's prefixes in one alphanumerical order, which this
    stem does not compose. None (the builder declines) when the acid has no name or its name does not end in
    the one word 'carboxylic acid'."""
    if p["x"] not in ("H", "N"):
        return None
    c, n, y, alkyl = p["c"], p["n"], p["y"], p["alkyl"]
    drop = set(_collect_subgraph(mol, alkyl, {y}))
    if c in drop:
        return None
    if p["x"] == "N":
        drop.update(a.GetIdx() for a in mol.GetAtomWithIdx(n).GetNeighbors() if a.GetIdx() != c)
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(y).SetAtomicNum(8)
    rw.GetAtomWithIdx(n).SetAtomicNum(8)
    for idx in sorted(drop, reverse=True):
        rw.RemoveAtom(idx)
    try:
        acid = rw.GetMol()
        for a in (acid.GetAtomWithIdx(i) for i in range(acid.GetNumAtoms())):
            if a.GetSymbol() == "O" and a.GetDegree() == 1:      # the =O and the -OH: their hydrogens are implicit
                a.SetNoImplicit(False)
                a.SetNumExplicitHs(0)
        Chem.SanitizeMol(acid)
        acid_smiles = Chem.MolToSmiles(acid)
    except Exception:  # noqa: BLE001 -- an unbuildable acid is no stem
        return None
    acid_name = _name_acid_strict(acid_smiles)
    word = "carboxylic acid"
    if not acid_name or not acid_name.endswith(word) or acid_name.count(word) != 1:
        return None
    return acid_name[: -len(word)] + _RING_SUFFIX[suffix]


# What the seniority cascade reports for a molecule whose most senior group is the C=N of the ester
# (nothing senior to it survives: an acid, anhydride, other ester, amide, alcohol or amine would be the
# principal group instead).
_PG_OF_THE_C_N = (None, "imine", "hydrazone", "oxime", "amidine", "iminoester")
# The esters of the oxoacids of phosphorus are junior to the ester of an imidic acid (class 7a before 7c, Table
# 4.1, the Blue Book; the esters keep the order of their acids, ASSUMED). The cascade reports them as the
# principal group at the best-effort tier and the imine at the default tier, so they are named the same at both: the
# group is bonded to the chain through O or P and is cited as a prefix ('(dimethoxyphosphoryl)oxy').
_PG_PHOSPHORUS_ESTER = ("phosphate_triester", "phosphate_diester", "phosphate_monoester", "phosphite_triester",
                        "phosphonate_diester")


def imidic_ester_parent_expected(features: Any) -> bool:
    """True when the PIN of the molecule is the ester of an imidic or hydrazonic acid: one acyclic
    ``R-C(=N-X)-Y-R'`` group and no group senior to it, the Blue Book, class 9; the
    amides, nitriles, aldehydes, ketones, alcohols, amines and imines are junior to it). A name of such a molecule that cites the group as an imine ('1-(methylsulfanyl)-1-
    (pyridin-4-yl)methanimine') is correct and is not the PIN,:31983)."""
    from ...rules.seniority import SENIORITY_ORDER

    mol = getattr(features, "mol", None)
    pg = getattr(features, "principal_group", None)
    if mol is None or Chem.GetFormalCharge(mol) != 0 or len(mol.GetSubstructMatches(_IMIDIC_ESTER)) != 1:
        return False
    # the cascade has no class for the S-ester, so a group junior to the esters (amide, nitrile, aldehyde,
    # ketone, alcohol,...) is what it reports for the molecule: the group senior to it is the ester. The esters
    # of sulfonic and sulfinic acids are junior to it too: class 7a (:18172) puts the imidic acids of the
    # carboxylic acids before the sulfonic and the sulfinic acids, and the esters keep the order of their acids
    # (ASSUMED: class 9 (:18182) states no order; classes 10 to 12 say "in the order of the corresponding acids")
    return pg in _PG_OF_THE_C_N or (
        pg in SENIORITY_ORDER and SENIORITY_ORDER.index(pg) >= SENIORITY_ORDER.index("sulfonic_ester"))


def _is_imidic_ester(features: Any) -> bool:
    """Predicate (pure): one imidic ester group and no group senior to it, class 9)."""
    if getattr(features, "principal_group", None) not in _PG_OF_THE_C_N + _PG_PHOSPHORUS_ESTER:
        return False
    return _parts(getattr(features, "mol", None)) is not None


def _chain(mol: Any, start: int) -> Tuple[List[int], List[int], List[int]]:
    """The principal chain from ``start``: (atoms, double-bond locants, triple-bond locants).

     "Greater number of skeletal atoms" first, then "Seniority of... the principal
    chain": (a) the greater number of multiple bonds, (b) the greater number of double bonds
    , (j) the lower locants for 'ene' and 'yne' endings; what remains goes to
    the lowest substituent locants. The suffix sits at C-1, so criterion (h) never
    separates two chains.
    """
    def carbon_ok(idx: int) -> bool:
        at = mol.GetAtomWithIdx(idx)
        return at.GetSymbol() == "C" and not at.IsInRing() and not at.GetIsAromatic()

    paths: List[List[int]] = []
    best = [0]

    def walk(path: List[int]) -> None:
        ext = False
        for nb in mol.GetAtomWithIdx(path[-1]).GetNeighbors():
            j = nb.GetIdx()
            if j in path or not carbon_ok(j):
                continue
            ext = True
            walk(path + [j])
        if not ext:
            if len(path) > best[0]:
                best[0], paths[:] = len(path), [path]
            elif len(path) == best[0]:
                paths.append(path)

    walk([start])

    def multiple(path: List[int]) -> Tuple[List[int], List[int]]:
        dbl: List[int] = []
        trp: List[int] = []
        for k in range(len(path) - 1):
            order = mol.GetBondBetweenAtoms(path[k], path[k + 1]).GetBondType()
            if order == Chem.BondType.DOUBLE:
                dbl.append(k + 1)
            elif order == Chem.BondType.TRIPLE:
                trp.append(k + 1)
        return dbl, trp

    def key(path: List[int]):
        dbl, trp = multiple(path)
        on = {a: i + 1 for i, a in enumerate(path)}
        locs = sorted(on[i] for i in path for a in mol.GetAtomWithIdx(i).GetNeighbors()
                      if a.GetIdx() not in on and a.GetAtomicNum() > 1)
        return (-(len(dbl) + len(trp)), -len(dbl), sorted(dbl + trp), dbl, locs, [on[i] for i in path[1:]])

    chosen = min(paths, key=key)
    dbl, trp = multiple(chosen)
    return chosen, dbl, trp


def _usable(word: Optional[str]) -> bool:
    return bool(word) and word != "substituent" and " " not in word


def build_imidic_ester_name(mol: Any) -> Optional[str]:
    """The PIN of one imidic ester group, or None when this builder does not own the molecule."""
    p = _parts(mol)
    if p is None:
        return None
    c, n, y, alkyl = p["c"], p["n"], p["y"], p["alkyl"]
    if p["stem"] == "N":
        return _name_carbamimidate(mol, (c, n, p["r"], y, alkyl), thio=True)
    alkyl_word = _name_alkyl_fragment(mol, _collect_subgraph(mol, alkyl, {y}), alkyl)
    if not _usable(alkyl_word):
        return None
    suffix = "hydrazonate" if p["x"] == "N" else ("imidothioate" if p["y_sym"] == "S" else "imidate")
    pieces: "List[Tuple[str, str]]" = []
    if p["x"] == "C":
        from ..substituent_enumerator import name_substituent
        frag = _collect_subgraph(mol, p["n_sub"], {n})
        nm = name_substituent(mol, set(frag), p["n_sub"])
        if not _usable(nm):
            return None
        pieces.append((nm, "N"))
    elif p["x"] == "O":
        pieces.append(("hydroxy", "N"))
    if p["stem"] == "H":
        from ...data.chain_names import get_chain_prefix
        stem = get_chain_prefix(1) + _CHAIN_SUFFIX[suffix]
    elif _plain_benzene(mol, mol.GetAtomWithIdx(p["r"])):
        stem = "benzene" + _RING_SUFFIX[suffix]
    elif mol.GetAtomWithIdx(p["r"]).IsInRing():
        stem = _stem_from_acid(mol, c, y, alkyl, suffix) or _stem_from_carboxylic_acid(mol, p, suffix)
        if stem is None:
            return None
        return f"{alkyl_word} {stem}"          # the N groups, if any, are part of the acid name
    else:
        from ...data.chain_names import get_chain_prefix
        from ..composition_primitives import _build_unsaturation_infix
        from ..naming_utils import format_suffix_with_locants
        from ..substituent_enumerator import name_substituent
        chain, dbl, trp = _chain(mol, c)
        try:
            stem = format_suffix_with_locants(get_chain_prefix(len(chain)), _build_unsaturation_infix(dbl, trp),
                                              suffix, [])
        except (ValueError, KeyError):
            return None
        on = {a: i + 1 for i, a in enumerate(chain)}
        ring_or_chain = set(chain) | {n, y}
        for i in chain:
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                j = nb.GetIdx()
                if j in on or j in ring_or_chain or nb.GetAtomicNum() <= 1:
                    continue
                frag = _collect_subgraph(mol, j, {i})
                if set(frag) & set(chain):
                    return None
                nm = name_substituent(mol, set(frag), j)
                if not _usable(nm):
                    return None
                pieces.append((nm, str(on[i])))
    block = locanted_prefix_block(pieces) if pieces else ""
    if block is None:
        return None
    return f"{alkyl_word} {block}{stem}"


def name_imidic_ester(features: Any, mol: Any = None, style: str = "pin") -> Optional[NamingResult]:
    """Handler: the imidic ester PIN, or None (gate-fail, -04) so the next handler runs."""
    from ..candidate_pool import get_current_pool

    mol = mol if mol is not None else getattr(features, "mol", None)
    name = build_imidic_ester_name(mol)
    if not name:
        return None
    cand = get_current_pool().add(name, "imidate", features)
    if cand is None:
        return None
    return NamingResult(
        name=cand.name,
        tree=NameTreeNode(parent_stem=cand.name, class_id="imidate",
                          iupac_section_cite="P-65.6.3.3.7.1", fragment_legacy=cand.name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_imidic_ester", "_is_imidic_ester", "build_imidic_ester_name", "imidic_ester_parent_expected"]
