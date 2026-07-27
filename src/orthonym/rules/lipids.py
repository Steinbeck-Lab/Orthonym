"""P-107 lipid backbone-aware Form-B assembler (Phase 180, WSC-01).

Consumes the structured ``BackboneMatch`` from ``perception.lipids.detect_lipid_backbone``
and builds the fully systematic substitutive / functional-class (Form B) name — the
empirically OPSIN-round-trip-verified target form (180-CONTEXT.md D-01). Every gate
failure returns ``None`` so the molecule cascade-continues to the general pipeline
(D-06/D-11, fail-safe → zero non-lipid regression).

Acyl groups are named by the robust acid-fragment-reuse strategy: isolate the fatty
acid, name it with the proven acid pipeline (systematic + E/Z), then convert
``-oic acid`` → ``-oate`` / ``-oyl``. This handles saturated, unsaturated,
polyunsaturated, branched, and hydroxy acyls uniformly.
"""

from __future__ import annotations

import logging
from typing import Optional

from rdkit import Chem

from ..assembly.naming_utils import get_multiplier_prefix, is_complex_substituent
from ..data.chain_names import get_chain_prefix
from ..data.sugar_names import sugar_to_glycosyloxy_prefix
from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Acyl naming — robust acid-fragment reuse
# --------------------------------------------------------------------------- #
def _acid_fragment_name(mol, carbonyl_idx: int, ester_o_idx: int) -> Optional[str]:
    """Isolate the acid fragment of an ``R-C(=O)-O-`` ester and name it as a free acid.

    Returns the systematic acid name (e.g. ``'(9Z)-octadec-9-enoic acid'``) or None.
    """
    rw = Chem.RWMol(mol)
    rw.RemoveBond(carbonyl_idx, ester_o_idx)
    new_o = rw.AddAtom(Chem.Atom(8))           # the acid -OH oxygen
    rw.AddBond(carbonyl_idx, new_o, Chem.BondType.SINGLE)
    m2 = rw.GetMol()
    try:
        Chem.SanitizeMol(m2)
    except Exception:
        return None
    # the fragment containing the carbonyl carbon IS the acid
    for frag in Chem.GetMolFrags(m2):
        if carbonyl_idx in frag:
            try:
                acid_smiles = Chem.MolFragmentToSmiles(
                    m2, atomsToUse=list(frag), canonical=True, isomericSmiles=True)
            except Exception:
                return None
            from orthonym import name_compound  # lazy: re-entrant on a non-lipid fragment
            try:
                name = name_compound(acid_smiles)
            except Exception:
                return None
            if not name or name == "unknown organic compound" or "unknown" in name:
                return None
            return name
    return None


def _acid_to_oate(acid_name: str) -> Optional[str]:
    s = acid_name[:-5] if acid_name.endswith(" acid") else acid_name
    if s.endswith("oic"):
        return s[:-3] + "oate"
    if s.endswith("ic"):
        return s[:-2] + "ate"
    return None


def _acid_to_acyl(acid_name: str) -> Optional[str]:
    s = acid_name[:-5] if acid_name.endswith(" acid") else acid_name
    if s.endswith("oic"):
        return s[:-3] + "oyl"
    if s.endswith("ic"):
        return s[:-2] + "yl"
    return None


def _acylate_for_site(mol, site) -> Optional[str]:
    """site = ('acyl', carbonyl_idx, ester_o_idx) → '<acyl>oate' or None."""
    acid = _acid_fragment_name(mol, site[1], site[2])
    return _acid_to_oate(acid) if acid else None


def _acyloxy_for_site(mol, site) -> Optional[str]:
    """site = ('acyl', carbonyl_idx, ester_o_idx) → '<acyl>oyloxy' (acyloxy prefix) or None."""
    acid = _acid_fragment_name(mol, site[1], site[2])
    acyl = _acid_to_acyl(acid) if acid else None
    return f"{acyl}oxy" if acyl else None


def _multiplied_acylate(acylate: str, count: int) -> str:
    if count == 1:
        return acylate
    mult = get_multiplier_prefix(count, acylate)
    if is_complex_substituent(acylate):
        return f"{mult}[{acylate}]"
    return f"{mult}{acylate}"


# --------------------------------------------------------------------------- #
# Glycerol numbering helpers
# --------------------------------------------------------------------------- #
def _central_carbon(mol, core_atoms):
    """The glycerol C-2: the backbone carbon bonded to both other backbone carbons."""
    coreset = set(core_atoms)
    for c in core_atoms:
        nbrs = {n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors()}
        if len(nbrs & coreset) == 2:
            return c
    return core_atoms[1]  # fallback (SMARTS middle)


def _glyceride_numbering(mol, match):
    """Return {atom_idx: iupac_locant} numbering propane so the ester (PCG) sites get
    the lowest locants; central carbon is always 2."""
    core = match.core_atoms
    central = _central_carbon(mol, core)
    terminals = [c for c in core if c != central]
    atom_site = {}
    loc_to_atom = {v: k for k, v in match.position_locants.items()}
    for loc, site in match.sites.items():
        atom_site[loc_to_atom[loc]] = site
    # Ester (acyl) is the PCG → lowest locants; phospho/glyco/OH become prefixes.
    acyl_atoms = {a for a, s in atom_site.items() if s[0] == "acyl"}
    t1, t2 = terminals
    candidates = [{t1: 1, central: 2, t2: 3}, {t2: 1, central: 2, t1: 3}]

    def key(numbering):
        return sorted(numbering[a] for a in acyl_atoms) or [99]

    best = min(candidates, key=key)
    return best, central, atom_site


# --------------------------------------------------------------------------- #
# Glyceride assembler (P-107.2)
# --------------------------------------------------------------------------- #
def _assemble_glyceride(mol, match, style) -> Optional[str]:
    num, central, atom_site = _glyceride_numbering(mol, match)

    acyl_atoms = sorted((a for a, s in atom_site.items() if s[0] == "acyl"),
                        key=lambda a: num[a])
    oh_atoms = [a for a, s in atom_site.items() if s[0] == "free_oh"]
    glyco_atoms = [a for a, s in atom_site.items() if s[0] == "glycosyl"]
    phospho_atoms = [a for a, s in atom_site.items() if s[0] == "phospho"]

    if not acyl_atoms:
        return None  # no ester PCG → not a glyceride (glycerol itself / phospho handled elsewhere)

    # --- acylate names per acyl position ---
    acylate_by_loc = {}
    for a in acyl_atoms:
        acylate = _acylate_for_site(mol, atom_site[a])
        if acylate is None:
            return None
        acylate_by_loc[num[a]] = acylate
    acyl_locants = sorted(acylate_by_loc)
    n_acyl = len(acyl_locants)

    # --- attachment (glycerol -yl) descriptor ---
    if n_acyl == 3:
        attach = "propane-1,2,3-triyl"
    elif n_acyl == 2:
        attach = f"propane-{','.join(str(x) for x in acyl_locants)}-diyl"
    else:  # 1
        loc = acyl_locants[0]
        attach = "propyl" if loc == 1 else f"propan-{loc}-yl"

    # --- ester suffix ---
    acylates = [acylate_by_loc[loc] for loc in acyl_locants]
    if len(set(acylates)) != 1 and n_acyl > 1:
        # W3-P07 (P-65.6.3.3.3.2 method (1), the PIN): DIFFERENT anions are cited
        # in alphanumerical order, EACH preceded by its attachment locant(s); a
        # multiplicative prefix (di/tri) denotes a multiplicity of identical
        # anions — 'propane-1,2,3-triyl 1,3-diacetate 2-propanoate'. This clean
        # two-word-list form applies only to the FULLY-esterified backbone (no
        # free-OH / glyco / phospho prefix); otherwise defer to the general
        # (acyloxy)-prefix pipeline (method (2), acceptable in general nomenclature).
        # The name is a valid BB PIN even though OPSIN cannot parse this
        # functional-class multi-anion syntax (BB is the sole PIN authority).
        if oh_atoms or glyco_atoms or phospho_atoms:
            return None
        from collections import defaultdict as _defaultdict
        _groups = _defaultdict(list)
        for _loc in acyl_locants:
            _groups[acylate_by_loc[_loc]].append(_loc)
        _ordered = sorted(_groups, key=_alpha_key)
        _parts = []
        for _aname in _ordered:
            _locs = sorted(_groups[_aname])
            _loc_str = ",".join(str(x) for x in _locs)
            _parts.append(f"{_loc_str}-{_multiplied_acylate(_aname, len(_locs))}")
        suffix = " ".join(_parts)
    else:
        suffix = _multiplied_acylate(acylates[0], n_acyl)

    # --- prefixes (free-OH, glycosyl, phosphoryloxy) on the non-acyl positions ---
    prefix = _build_glycerol_prefixes(num, oh_atoms, glyco_atoms, atom_site, phospho_atoms)
    if prefix is None:
        return None  # unrecognized neutral phospho head group → defer (D-11)

    name = f"{prefix}{attach} {suffix}"

    # --- stereo: backbone C-2 R/S (acyl-chain E/Z already inside the acylate strings;
    # sugar stereo inside 'beta-D-...'). Build the descriptor directly from the
    # authoritative {central: 2} locant map (D-09) and prepend it; the coarse
    # needs_stereo_injection gate is bypassed because the backbone descriptor is
    # absent from the assembled name even when sugar/acyl descriptors are present. ---
    descriptors = collect_stereodescriptors(mol, {central: 2}, include_near_parent_ez=False)
    if descriptors:
        name = format_stereodescriptor_string(descriptors) + name
    return name


def _alpha_key(acylate: str) -> str:
    """Alphabetization key: strip enclosing marks, locants, and stereo blocks."""
    import re
    s = re.sub(r"^\([0-9EZRSdlDL,\s]+\)-?", "", acylate)   # leading (9Z)- block
    s = re.sub(r"[\[\]()]", "", s)
    s = re.sub(r"[0-9,\-]", "", s)
    return s.lower()


# Head-group → alkoxy substituent for the substitutive (neutral) phosphoryloxy prefix.
_HEAD_GROUP_ALKOXY = {
    "ethanolamine": "2-aminoethoxy",
}


def _build_glycerol_prefixes(num, oh_atoms, glyco_atoms, atom_site, phospho_atoms=()) -> Optional[str]:
    """Build the detachable-prefix string (hydroxy / glycosyloxy / phosphoryloxy)
    preceding the attachment. Returns None if a phospho head group is unrecognized
    in the substitutive (neutral) regime (honest-gate, D-11)."""
    subs = []  # (alpha_key, rendered)
    if oh_atoms:
        locs = sorted(num[a] for a in oh_atoms)
        mult = {1: "", 2: "di", 3: "tri"}[len(locs)]
        rendered = f"{','.join(str(x) for x in locs)}-{mult}hydroxy"
        subs.append(("hydroxy", rendered))
    for a in glyco_atoms:
        anomer, config, base = atom_site[a][2]
        gly = sugar_to_glycosyloxy_prefix(anomer, config, base)
        subs.append((_alpha_key(gly), f"{num[a]}-({gly})"))
    for a in phospho_atoms:
        head_desc = atom_site[a][3]
        head_alkoxy = _HEAD_GROUP_ALKOXY.get(head_desc)
        if head_alkoxy is None:
            return None  # neutral substitutive form not RT-verified for this head → defer (D-11)
        # P-16.5.4 nesting ORDER (BB 7444; escalation P-16.5.4.1.5, BB 7509) under the P-16.5.1.1 marks requirement (BB 7232): {[(<head-alkoxy>)hydroxyphosphoryl]oxy}
        rendered = f"{num[a]}-{{[({head_alkoxy})hydroxyphosphoryl]oxy}}"
        subs.append(("phosphoryl", rendered))
    if not subs:
        return ""
    subs.sort(key=lambda t: t[0])  # alphabetical by substituent name
    return "".join(r for _, r in subs)


# --------------------------------------------------------------------------- #
# Public dispatcher
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Phospholipid assembler (P-107.3) — functional-class phosphate diester
# --------------------------------------------------------------------------- #
# Cationic head groups that use the P-68 functional-class "phosphate" (zwitterion) form.
# Neutral heads (ethanolamine, ...) route to the substitutive 'hydroxyphosphoryl-oxy'
# form instead (see _build_glycerol_prefixes / _HEAD_GROUP_ALKOXY).
_HEAD_GROUP_ALKYL = {
    "choline": "2-(trimethylazaniumyl)ethyl",
}


def _bis_enclose(inner: str, mult: str) -> str:
    """Enclose a multiplied compound substituent: brackets if it has parens, else parens."""
    if "(" in inner or "[" in inner:
        return f"{mult}[{inner}]"
    return f"{mult}({inner})"


# Multiplier word for free acidic -OH on the functional-class phosphate ester.
_HYDROGEN_WORD = {1: "hydrogen", 2: "dihydrogen"}


def _collect_head_skeleton(mol, head_c, head_o_idx) -> list:
    """BFS the head-group skeleton from ``head_c``, blocked at the phospho-ester
    oxygen (so the phosphate is excluded). Returns the atom-index list naming the
    head substituent (P-107.3.3)."""
    from collections import deque
    seen = {head_o_idx, head_c}
    out = [head_c]
    dq = deque([head_c])
    while dq:
        cur = dq.popleft()
        for n in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = n.GetIdx()
            if ni in seen:
                continue
            seen.add(ni)
            out.append(ni)
            dq.append(ni)
    return out


def _count_free_phosphate_oh(mol, p_idx) -> int:
    """Count free acidic -OH oxygens on the phosphorus (single-bonded, degree 1,
    >=1 H) — the 'hydrogen'/'dihydrogen' multiplier for the phosphate ester."""
    n = 0
    p = mol.GetAtomWithIdx(p_idx)
    for nb in p.GetNeighbors():
        if nb.GetAtomicNum() != 8:
            continue
        b = mol.GetBondBetweenAtoms(p_idx, nb.GetIdx())
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        if nb.GetDegree() == 1 and nb.GetTotalNumHs() >= 1:
            n += 1
    return n


def _name_phospho_head_substituent(mol, head_o_idx) -> Optional[str]:
    """Name a neutral organic phospho head group (e.g. serine) as a substituent
    prefix FROM STRUCTURE, via the (structure-loss-free) substituent namer. The
    head carbon is the carbon neighbour of the phospho-ester oxygen; the skeleton
    is collected blocked at that oxygen. Fail-closed: returns None on a degenerate
    / unnameable result so the caller defers to the substitutive glyceride form."""
    from ..assembly.substituent_enumerator import name_substituent
    head_o = mol.GetAtomWithIdx(head_o_idx)
    head_c = next((n.GetIdx() for n in head_o.GetNeighbors()
                   if n.GetAtomicNum() == 6), None)
    if head_c is None:
        return None
    skeleton = _collect_head_skeleton(mol, head_c, head_o_idx)
    name = name_substituent(mol, skeleton, head_c)
    if (not name or name in ("substituent", "unknown organic compound")
            or "unknown" in name):
        return None
    return name


def _assemble_phospholipid(mol, match, style) -> Optional[str]:
    central = _central_carbon(mol, match.core_atoms)
    loc_to_atom = {v: k for k, v in match.position_locants.items()}
    atom_site = {loc_to_atom[loc]: site for loc, site in match.sites.items()}

    phospho_atoms = [a for a, s in atom_site.items() if s[0] == "phospho"]
    acyl_atoms = [a for a, s in atom_site.items() if s[0] == "acyl"]
    if len(phospho_atoms) != 1 or len(acyl_atoms) != 2:
        return None  # only the diacyl-phospho (canonical phospholipid) regime here
    phospho_atom = phospho_atoms[0]
    head_desc = atom_site[phospho_atom][3]

    # Cationic head (choline): the internal N+/[O-] zwitterion is captured by the
    # P-68 functional-class "phosphate" form. Neutral heads (ethanolamine, ...) need
    # the substitutive 'hydroxyphosphoryl-oxy' prefix form → route to the glyceride
    # assembler (phospho handled as a C-3 prefix). RESOLVED: OPSIN's functional-class
    # "phosphate" yields the anionic/zwitterion form (matches PC, not neutral PE).
    head_alkyl = _HEAD_GROUP_ALKYL.get(head_desc)
    if head_alkyl is None:
        # v23 Phase 14: a FREE phosphatidic acid (no head group beyond the
        # phosphate) is named as the functional-class phosphate monoester
        # '<2,3-bis(acyloxy)propyl> dihydrogen phosphate' (P-107.3.1). The
        # detector tags the bare -OPO(OH)2 site head_desc='phosphate'.
        if head_desc == "phosphate":
            glyceryl = _diacyl_glyceryl(mol, match, central, phospho_atom, acyl_atoms, atom_site)
            if glyceryl is None:
                return None
            return f"{glyceryl} dihydrogen phosphate"
        # v23 SL: a neutral organic head (serine -> phosphatidylserine, P-107.3.3)
        # named FROM STRUCTURE via the structure-loss-free substituent namer ->
        # functional-class hydrogen-phosphate diester '<glyceryl> <head> hydrogen
        # phosphate'. inositol (stereo-entangled ring head), ethanolamine/glycerol
        # (substitutive form) keep their existing paths. Fail-closed: declines to
        # the substitutive glyceride form when the head is not cleanly nameable.
        if head_desc not in ("choline", "ethanolamine", "glycerol", "inositol"):
            # W8-P7c.3 (P-107.3.3): phosphatidylserine's PIN is on the L-serine
            # parent (the serine carboxylic acid outranks the phosphorus oxoacid,
            # P-41) -> O-{[<glyceryl-propoxy>]hydroxyphosphoryl}-<L|D>-serine, NOT
            # the phosphate-ester-parent fallback below. The head-config + form are
            # fixed by a HARD OPSIN round-trip (opsin_parse fails-CLOSED without
            # Java -> the input falls through to the valid phosphate-ester form).
            glyceryl_ser = _diacyl_glyceryl(
                mol, match, central, phospho_atom, acyl_atoms, atom_site)
            if glyceryl_ser is not None and glyceryl_ser.endswith("propyl"):
                glyoxy = glyceryl_ser[: -len("propyl")] + "propoxy"
                from ..validation.opsin_roundtrip import opsin_parse
                target = Chem.MolToSmiles(mol)
                for chir in ("L", "D"):
                    cand = (f"O-{{[{glyoxy}]hydroxyphosphoryl}}-{chir}-serine")
                    parsed = opsin_parse(cand)
                    if parsed:
                        pm = Chem.MolFromSmiles(parsed)
                        if pm is not None and Chem.MolToSmiles(pm) == target:
                            return cand
            head_o_idx = atom_site[phospho_atom][4]
            p_idx = atom_site[phospho_atom][1]
            head_name = (
                _name_phospho_head_substituent(mol, head_o_idx)
                if head_o_idx is not None else None
            )
            hword = _HYDROGEN_WORD.get(_count_free_phosphate_oh(mol, p_idx))
            if head_name is not None and hword is not None:
                glyceryl = _diacyl_glyceryl(
                    mol, match, central, phospho_atom, acyl_atoms, atom_site)
                if glyceryl is not None:
                    return f"{glyceryl} {head_name} {hword} phosphate"
        return _assemble_glyceride(mol, match, style)  # neutral substitutive form (or None if unrecognized)

    glyceryl = _diacyl_glyceryl(mol, match, central, phospho_atom, acyl_atoms, atom_site)
    if glyceryl is None:
        return None
    # P-68 functional-class diester: [<glyceryl>] <head-alkyl> phosphate (glyceryl bracketed first)
    return f"[{glyceryl}] {head_alkyl} phosphate"


def _diacyl_glyceryl(mol, match, central, phospho_atom, acyl_atoms, atom_site) -> Optional[str]:
    """Build the diacyl glyceryl substituent '(2R)-2,3-bis(acyloxy)propyl' for a
    phospholipid: local C1 = the phospho-attached carbon, C2 = central, C3 = the
    other terminal; the two acyls become (acyloxy) prefixes at locals 2,3."""
    other_terminal = [a for a in match.core_atoms if a not in (central, phospho_atom)]
    if len(other_terminal) != 1:
        return None
    other_terminal = other_terminal[0]
    local = {phospho_atom: 1, central: 2, other_terminal: 3}

    acyloxy_by_loc = {}
    for a in acyl_atoms:
        ax = _acyloxy_for_site(mol, atom_site[a])
        if ax is None:
            return None
        acyloxy_by_loc[local[a]] = ax
    acyl_locs = sorted(acyloxy_by_loc)
    locant_str = ",".join(str(x) for x in acyl_locs)
    names = [acyloxy_by_loc[l] for l in acyl_locs]
    if len(set(names)) == 1:
        gly_subs = f"{locant_str}-{_bis_enclose(names[0], 'bis')}"
    else:
        parts = sorted(((l, acyloxy_by_loc[l]) for l in acyl_locs), key=lambda t: _alpha_key(t[1]))
        gly_subs = "".join(
            f"{l}-{('[' + ax + ']') if ('(' in ax or '[' in ax) else '(' + ax + ')'}"
            for l, ax in parts
        )
    glyceryl = f"{gly_subs}propyl"
    # backbone C-2 R/S (local locant 2 = central) inside the glyceryl substituent
    descriptors = collect_stereodescriptors(mol, {central: 2}, include_near_parent_ez=False)
    if descriptors:
        glyceryl = format_stereodescriptor_string(descriptors) + glyceryl
    return glyceryl


# --------------------------------------------------------------------------- #
# Sphingolipid / ceramide assembler (P-107.4.3) — amide PCG, sphingoid N-substituent
# --------------------------------------------------------------------------- #
def _acid_to_amide(acid_name: str) -> Optional[str]:
    s = acid_name[:-5] if acid_name.endswith(" acid") else acid_name
    if s.endswith("oic"):
        return s[:-3] + "amide"
    if s.endswith("ic"):
        return s[:-2] + "amide"
    return None


def _sphingoid_prefixes(match) -> str:
    """Build the C1/C3 detachable prefixes (hydroxy / glycosyloxy) for the sphingoid -yl."""
    ohs, others = [], []
    for loc in (1, 3):
        s = match.sites.get(loc)
        if s is None:
            continue
        if s[0] == "free_oh":
            ohs.append(loc)
        elif s[0] == "glycosyl":
            anomer, config, base = s[2]
            gly = sugar_to_glycosyloxy_prefix(anomer, config, base)
            others.append((_alpha_key(gly), f"{loc}-({gly})"))
    parts = []
    if len(ohs) == 2:
        parts.append(("hydroxy", f"{ohs[0]},{ohs[1]}-dihydroxy"))
    elif len(ohs) == 1:
        parts.append(("hydroxy", f"{ohs[0]}-hydroxy"))
    parts.extend(others)
    parts.sort(key=lambda t: t[0])
    return "-".join(r for _, r in parts)


def _assemble_ceramide(mol, match, style) -> Optional[str]:
    n_acyl = next((s for s in match.sites.values() if s[0] == "n_acyl"), None)
    if n_acyl is None:
        return None
    _, acyl_c, n_idx = n_acyl

    # amide parent = the fatty acid named then converted -oic acid -> -amide
    acid = _acid_fragment_name(mol, acyl_c, n_idx)
    amide = _acid_to_amide(acid) if acid else None
    if amide is None:
        return None

    # sphingoid N-substituent: <prefixes><stem><ene>-2-yl
    clen = match.extra.get("chain_length")
    dbs = match.extra.get("double_bonds", [])
    chain_pos = match.extra.get("chain_pos")
    if not clen or chain_pos is None:
        return None
    stem = get_chain_prefix(clen)
    yl = "-2-yl"  # sphingoid attaches to the amide N at C-2
    if not dbs:
        sub_core = f"{stem}an{yl}"
    elif len(dbs) == 1:
        sub_core = f"{stem}-{dbs[0]}-en{yl}"
    else:
        mult = {2: "di", 3: "tri", 4: "tetra"}.get(len(dbs), "")
        sub_core = f"{stem}a-{','.join(map(str, dbs))}-{mult}en{yl}"

    prefixes = _sphingoid_prefixes(match)
    substituent = f"{prefixes}{sub_core}" if prefixes else sub_core

    # stereo block: R/S at C2,C3 + E/Z of chain double bonds, from the ceramide mol
    descriptors = collect_stereodescriptors(mol, dict(chain_pos), include_near_parent_ez=True)
    if descriptors:
        substituent = format_stereodescriptor_string(descriptors) + substituent

    return f"N-[{substituent}]{amide}"


def name_lipid(mol, style: str = "pin") -> Optional[str]:
    """Name a lipid on its detected backbone (Form B), or None (cascade-continue)."""
    from ..perception.lipids import detect_lipid_backbone
    match = detect_lipid_backbone(mol)
    if match is None:
        return None
    if match.family == "glyceride":
        return _assemble_glyceride(mol, match, style)
    if match.family == "phospholipid":
        return _assemble_phospholipid(mol, match, style)
    if match.family == "sphingolipid":
        return _assemble_ceramide(mol, match, style)
    return None
