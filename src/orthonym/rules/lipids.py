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
    acyl_atoms = {a for a, s in atom_site.items() if s[0] in ("acyl", "phospho")}
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
    if len(set(acylates)) == 1:
        suffix = _multiplied_acylate(acylates[0], n_acyl)
    else:
        # mixed acyls: alphabetized locant-prefixed list (Blue Book P-107.2 form)
        parts = sorted(
            ((loc, acylate_by_loc[loc]) for loc in acyl_locants),
            key=lambda t: _alpha_key(t[1]),
        )
        suffix = " ".join(
            f"{loc}-[{ac}]" if is_complex_substituent(ac) else f"{loc}-{ac}"
            for loc, ac in parts
        )

    # --- prefixes (free-OH, glycosyl) on the non-acyl positions ---
    prefix = _build_glycerol_prefixes(num, oh_atoms, glyco_atoms, atom_site)

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


def _build_glycerol_prefixes(num, oh_atoms, glyco_atoms, atom_site) -> str:
    """Build the detachable-prefix string (hydroxy / glycosyloxy) preceding the attachment."""
    subs = []  # (sorted_locants_tuple, alpha_key, rendered)
    if oh_atoms:
        locs = sorted(num[a] for a in oh_atoms)
        mult = {1: "", 2: "di", 3: "tri"}[len(locs)]
        rendered = f"{','.join(str(x) for x in locs)}-{mult}hydroxy"
        subs.append((locs[0], "hydroxy", rendered))
    for a in glyco_atoms:
        anomer, config, base = atom_site[a][2]
        gly = sugar_to_glycosyloxy_prefix(anomer, config, base)
        rendered = f"{num[a]}-({gly})"
        subs.append((num[a], _alpha_key(gly), rendered))
    if not subs:
        return ""
    subs.sort(key=lambda t: t[1])  # alphabetical by substituent name
    return "".join(r for _, _, r in subs)


# --------------------------------------------------------------------------- #
# Public dispatcher
# --------------------------------------------------------------------------- #
def name_lipid(mol, style: str = "pin") -> Optional[str]:
    """Name a lipid on its detected backbone (Form B), or None (cascade-continue)."""
    from ..perception.lipids import detect_lipid_backbone
    match = detect_lipid_backbone(mol)
    if match is None:
        return None
    if match.family == "glyceride":
        return _assemble_glyceride(mol, match, style)
    # TODO Plan 04: phospholipid functional-class phosphate diester
    # TODO Plan 05: sphingolipid amide PCG (ceramide)
    return None
