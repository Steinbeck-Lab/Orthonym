""" lipid backbone-aware Form-B assembler (a phase, -01).

Consumes the structured ``BackboneMatch`` from ``perception.lipids.detect_lipid_backbone``
and builds the fully systematic substitutive / functional-class (Form B) name — the
empirically OPSIN-round-trip-verified target form (180-internal notes). Every gate
failure returns ``None`` so the molecule cascade-continues to the general pipeline
(/, fail-safe → zero non-lipid regression).

Acyl groups are named by the robust acid-fragment-reuse strategy: isolate the fatty
acid, name it with the proven acid pipeline (systematic + E/Z), then convert
``-oic acid`` → ``-oate`` / ``-oyl``. This handles saturated, unsaturated,
polyunsaturated, branched, and hydroxy acyls uniformly.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from rdkit import Chem

from ..assembly.naming_utils import apply_enclosing_marks, get_multiplier_prefix, is_complex_substituent
from ..data.chain_names import get_chain_prefix
from ..data.sugar_names import sugar_to_glycosyloxy_prefix
from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Acyl naming — robust acid-fragment reuse
# --------------------------------------------------------------------------- #
def _isolated_acid_smiles(mol, carbonyl_idx: int, ester_o_idx: int,
                          anionic: bool = False) -> Optional[str]:
    """SMILES of the acid fragment of an ``R-C(=O)-O-`` ester site: the ester bond is
    cut and the site carbonyl gets a fresh ``-OH`` (``-O(-)`` when ``anionic``).
    None when the cut fragment does not sanitize."""
    rw = Chem.RWMol(mol)
    rw.RemoveBond(carbonyl_idx, ester_o_idx)
    new_o = rw.AddAtom(Chem.Atom(8))           # the acid -OH oxygen
    if anionic:
        rw.GetAtomWithIdx(new_o).SetFormalCharge(-1)
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
                return Chem.MolFragmentToSmiles(
                    m2, atomsToUse=list(frag), canonical=True, isomericSmiles=True)
            except Exception:
                return None
    return None


def _name_fragment(smiles: Optional[str]) -> Optional[str]:
    """The engine's name for an isolated fragment, or None (unnamed / unknown).

    The fragment is named for a part of a larger name (an acyl or an '-oate'),
    often speculatively: a classifier asks for an acyloxy prefix that the ester
    route then does not use. The producer that named the fragment is therefore
    not the producer of the outer name, so its record (``source``/``opsin``) is
    not left on the outer call: 'bis[4-(butan-2-yl)phenyl] oxalate' read
    systematic_verified at the best-effort tier, where a discarded acyloxy probe
    had reached the general engine, and pin_verified at the PIN tier. A fragment
    name the general engine built is recorded instead as a non-PIN fragment, which
    labels below the PIN only a name that carries it (and its acyl / '-oate' forms,
    ``_acid_to_acyl`` / ``_acid_to_oate``)."""
    if not smiles:
        return None
    from orthonym import name_compound  # lazy: re-entrant on a non-lipid fragment
    from ..metrics.provenance import (get_provenance, record_non_pin_fragment,
                                      restore_provenance)
    before = get_provenance()
    try:
        name = name_compound(smiles)
    except Exception:
        name = None
    after = get_provenance()
    nested_source = after.get("source")
    after["source"], after["opsin"] = before.get("source"), before.get("opsin")
    restore_provenance(after)
    if not name or name == "unknown organic compound" or "unknown" in name:
        return None
    if nested_source == "general_engine":
        record_non_pin_fragment(name)
    return name


def _acid_fragment_name(mol, carbonyl_idx: int, ester_o_idx: int) -> Optional[str]:
    """Isolate the acid fragment of an ``R-C(=O)-O-`` ester and name it as a free acid.

    Returns the systematic acid name (e.g. ``'(9Z)-octadec-9-enoic acid'``) or None.
    """
    return _name_fragment(_isolated_acid_smiles(mol, carbonyl_idx, ester_o_idx))


# A neutral carboxy group, -C(=O)-OH.
_CARBOXY = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")


def _site_acid_is_polycarboxylic(mol, carbonyl_idx: int, ester_o_idx: int) -> bool:
    """True when the isolated acid of this ester site carries more than one carboxy
    group (the site's own, re-formed, plus at least one free -COOH of the acyl side).

    The acid name of such a fragment cites every carboxy group it can as a suffix
    ('butanedioic acid', 'cyclohexane-1,4-dicarboxylic acid') or puts the suffix on
    whichever carboxy group ranks senior, which need not be the ester site ('4-
    (carboxymethyl)benzoic acid' for either site). Its '-oyl' form then names the
    acyl group left by removing the -OH from EACH carboxy group,
    the Blue Book, 'butanedioyl' is divalent) or the wrong carbonyl, never
    the monovalent acyl of this one site."""
    smi = _isolated_acid_smiles(mol, carbonyl_idx, ester_o_idx)
    frag = Chem.MolFromSmiles(smi) if smi else None
    return frag is not None and len(frag.GetSubstructMatches(_CARBOXY)) > 1


def _polycarboxylic_site_acyl(mol, carbonyl_idx: int, ester_o_idx: int) -> Optional[str]:
    """The monovalent acyl of ONE carboxy site of a polycarboxylic acid, every other
    carboxy group cited as a 'carboxy' prefix: HOOC-CH2-CH2-CO- is '3-carboxypropanoyl'.

    Named through the site's carboxylate anion, the method (1) form
    that carries the '-oate' suffix on the anionic carboxy group and cites the
    free ones as 'carboxy' ('ammonium 3-carboxypropanoate (PIN)',
    the Blue Book; 'potassium 6-carboxyhexanoate (PIN)',:31602); '-oate' /
    '-carboxylate' then become '-oyl' / '-carbonyl':30607,
     :30624).

    The engine's anion name is checked for WHERE it puts the charge: its own
    verification is the InChIKey, which does not locate the charge among the
    carboxy groups (measured: it names the anion of either site of 4-(carboxy-
    methyl)benzoic acid '4-(carboxymethyl)benzoate'). So the name is kept only
    when its OPSIN structure is the site anion itself (charge-located canonical
    SMILES equal); otherwise, and whenever OPSIN cannot say, None.

    Fail closed (None) as well when the anion fragment is charged elsewhere, is
    not named, is not a single '-ate' word citing a 'carboxy' prefix (a retained
    anion name such as 'dihydrocitrate' has no '-oyl' form), or carries any
    stereo element: the site O(-) and the in-situ ester O-R rank differently
    under CIP, so a descriptor of the fragment (C-3 of a 3-hydroxyglutaryl, say)
    can be the opposite of the one in the ester; the caller's own path cites
    such stereo in situ instead.

    HOOC-CO- is 'oxalo': "The prefix 'oxalo' is recommended as the preferred
    prefix for -CO-CO-OH", the Blue Book), 'oxalooxy
    (preferred prefix)' (:30540)."""
    neutral = _isolated_acid_smiles(mol, carbonyl_idx, ester_o_idx)
    neutral_mol = Chem.MolFromSmiles(neutral) if neutral else None
    if neutral_mol is not None and Chem.MolToSmiles(neutral_mol) == "O=C(O)C(=O)O":
        return "oxalo"
    smi = _isolated_acid_smiles(mol, carbonyl_idx, ester_o_idx, anionic=True)
    frag = Chem.MolFromSmiles(smi) if smi else None
    if frag is None:
        return None
    if sum(abs(a.GetFormalCharge()) for a in frag.GetAtoms()) != 1:
        return None
    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in frag.GetAtoms()):
        return None
    if any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in frag.GetBonds()):
        return None
    name = _name_fragment(smi)
    if not name or " " in name or not re.search(r"carboxy(?!l)", name):
        return None
    if name.endswith("carboxylate"):
        acid = name[:-len("carboxylate")] + "carboxylic acid"
    elif name.endswith("ate"):
        acid = name[:-len("ate")] + "ic acid"
    else:
        return None
    try:
        from ..validation.atom_coverage import _parse_name_with_opsin, find_opsin_jar
        parsed = _parse_name_with_opsin(name, find_opsin_jar())
    except Exception:
        return None
    if not parsed or parsed != Chem.MolToSmiles(frag):
        return None  # the '-oate' sits on another carboxy group (or no parse)
    return _acid_to_acyl(acid)


def _acid_to_oate(acid_name: str) -> Optional[str]:
    out = _acid_to_oate_form(acid_name)
    if out:
        # The '-oate' keeps the acid name's non-PIN record (_name_fragment).
        from ..metrics.provenance import record_derived_non_pin_fragment
        record_derived_non_pin_fragment(acid_name, out)
    return out


def _acid_to_oate_form(acid_name: str) -> Optional[str]:
    s = acid_name[:-5] if acid_name.endswith(" acid") else acid_name
    if s.endswith("oic"):
        return s[:-3] + "oate"
    if s.endswith("ic"):
        return s[:-2] + "ate"
    return None


def _acid_to_acyl(acid_name: str) -> Optional[str]:
    out = _acid_to_acyl_form(acid_name)
    if out:
        # The acyl keeps the acid name's non-PIN record (_name_fragment).
        from ..metrics.provenance import record_derived_non_pin_fragment
        record_derived_non_pin_fragment(acid_name, out)
    return out


def _acid_to_acyl_form(acid_name: str) -> Optional[str]:
    # '-amic' / '-imidic' acids take '-oyl' ('carbamoyl (preferred prefix)',
    # the Blue Book); the table and its rules live with the shared converter.
    from ..decomposition.fragment_assembly import ACID_ENDINGS_TO_OYL_ACYL
    for end, acyl in ACID_ENDINGS_TO_OYL_ACYL:
        if acid_name.endswith(end):
            return acid_name[:-len(end)] + acyl
    s = acid_name[:-5] if acid_name.endswith(" acid") else acid_name
    if s.endswith("oic"):
        return s[:-3] + "oyl"
    # (the Blue Book): "Acyl groups derived from an acid named
    # by means of the suffix 'carboxylic acid' are named by changing the
    # 'carboxylic acid' suffix to the suffix 'carbonyl'", 'cyclohexanecarbonyl
    # (preferred prefix)' (:30628), '3-[(pyridine-3-carbonyl)oxy]propanoic acid
    # (PIN)' (:31723). The generic '-ic' -> '-yl' step below would spell
    # 'cyclohexanecarboxylyl', which OPSIN 2.9.0 does not parse.
    if s.endswith("carboxylic"):
        return s[:-len("carboxylic")] + "carbonyl"
    if s.endswith("ic"):
        return s[:-2] + "yl"
    return None


def _acylate_for_site(mol, site) -> Optional[str]:
    """site = ('acyl', carbonyl_idx, ester_o_idx) → '<acyl>oate' or None."""
    acid = _acid_fragment_name(mol, site[1], site[2])
    return _acid_to_oate(acid) if acid else None


def _acyloxy_for_site(mol, site, carboxy_prefixed_polyacid: bool = False) -> Optional[str]:
    """site = ('acyl', carbonyl_idx, ester_o_idx) → '<acyl>oyloxy' (acyloxy prefix) or None.

    ``carboxy_prefixed_polyacid`` (opt-in; the best-effort universal floor): when
    the site's acid is polycarboxylic, name the monovalent acyl of THIS site with
    the other carboxy groups as 'carboxy' prefixes ('(3-carboxypropanoyl)oxy')
    or fail closed, instead of converting the polyacid's name ('butanedioyloxy',
    a divalent acyl). Off, the historic conversion is unchanged."""
    if carboxy_prefixed_polyacid and _site_acid_is_polycarboxylic(mol, site[1], site[2]):
        acyl = _polycarboxylic_site_acyl(mol, site[1], site[2])
    else:
        acid = _acid_fragment_name(mol, site[1], site[2])
        acyl = _acid_to_acyl(acid) if acid else None
    if not acyl:
        return None
    # Lane L2: the acyl group's multiplier and marks, from its atoms
    from ..assembly.book_prefixes import acyl_derivation, acyl_side_atoms
    from ..assembly.prefix_derivation import built, with_derivation
    acyl = with_derivation(acyl, acyl_derivation(
        mol, site[1], acyl_side_atoms(mol, site[1], site[2])))
    #: a compound acyl is cited inside its own marks with 'oxy' outside,
    # '4-[(3-ethoxy-3-oxopropanoyl)oxy]phenyl...' (the Blue Book),
    # '3-[(pyridine-3-carbonyl)oxy]propanoic acid (PIN)' (:31723); a simple acyl
    # stays bare, '3-(benzoyloxy)propanoic acid (PIN)' (:31711). So
    # '(4-hydroxybenzoyl)oxy', never '4-hydroxybenzoyloxy'.
    from ..assembly.substituent_enumerator import cite_organyl_in_composed_prefix
    cited = cite_organyl_in_composed_prefix(acyl)
    # (the Blue Book): an acyl-oxy prefix is compound,
    # '2-(acetyloxy)ethane-1-sulfonic acid (PIN)' (:31713), 'bis(acetyloxy)' (:32382)
    return built(f"{cited}oxy", substituted=True, enclosed=True) if cited else None


def _multiplied_acylate(acylate: str, count: int) -> str:
    if count == 1:
        return acylate
    mult = get_multiplier_prefix(count, acylate)
    if is_complex_substituent(acylate):
        # (the Blue Book): the mark follows the acylate's own depth
        return f"{mult}{apply_enclosing_marks(acylate, -1)}"
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


def _glyceride_numbering(mol, match, acylate_of=None):
    """Return {atom_idx: iupac_locant} numbering propane so the ester (PCG) sites get
    the lowest locants; central carbon is always 2.

    ``acylate_of`` (acyl atom -> anion name), given when the anions differ: with the
    ester locant set tied, the anion cited first in the name (alphanumerical order,
     method (1), the Blue Book) takes the lowest locants, as a
    prefix cited first does (g),:3307): 'propane-1,2,3-triyl 1,2-diacetate
    3-propanoate (PIN)' (:31840), '... 2-acetate 1-hexadecanoate 3-[(9Z)-octadec-9-
    enoate] (PIN)' (:31846)."""
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
        primary = sorted(numbering[a] for a in acyl_atoms) or [99]
        if not acylate_of:
            return (primary,)
        groups = {}
        for a in acyl_atoms:
            groups.setdefault(acylate_of[a], []).append(numbering[a])
        return (primary, [sorted(groups[g]) for g in sorted(groups, key=_alpha_key)])

    best = min(candidates, key=key)
    return best, central, atom_site


# --------------------------------------------------------------------------- #
# Glyceride assembler
# --------------------------------------------------------------------------- #
def _assemble_glyceride(mol, match, style) -> Optional[str]:
    num, central, atom_site = _glyceride_numbering(mol, match)
    _acylate_of = {a: _acylate_for_site(mol, s) for a, s in atom_site.items() if s[0] == "acyl"}
    if None not in _acylate_of.values() and len(set(_acylate_of.values())) > 1:
        num, central, atom_site = _glyceride_numbering(mol, match, _acylate_of)

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
        # W3-P07 method (1), the PIN): DIFFERENT anions are cited
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
            _word = _multiplied_acylate(_aname, len(_locs))
            if len(_locs) == 1 and _aname.startswith("("):
                # A single anion that carries its own stereodescriptor is enclosed
                # after its locant, the mark escalating past the descriptor's
                # parentheses: 'propane-1,2,3-triyl 2-acetate 1-hexadecanoate
                # 3-[(9Z)-octadec-9-enoate] (PIN)',
                # the Blue Book). Was '3-(11Z)-octadec-11-enoate' (TRIAGE g5 C12).
                _word = apply_enclosing_marks(_aname, -1)
            _parts.append(f"{_loc_str}-{_word}")
        suffix = " ".join(_parts)
    else:
        suffix = _multiplied_acylate(acylates[0], n_acyl)

    # --- prefixes (free-OH, glycosyl, phosphoryloxy) on the non-acyl positions ---
    prefix = _build_glycerol_prefixes(num, oh_atoms, glyco_atoms, atom_site, phospho_atoms)
    if prefix is None:
        return None  # unrecognized neutral phospho head group → defer

    name = f"{prefix}{attach} {suffix}"

    # --- stereo: backbone C-2 R/S (acyl-chain E/Z already inside the acylate strings;
    # sugar stereo inside 'beta-D-...'). Build the descriptor directly from the
    # authoritative {central: 2} locant map  and prepend it; the coarse
    # needs_stereo_injection gate is bypassed because the backbone descriptor is
    # absent from the assembled name even when sugar/acyl descriptors are present. ---
    descriptors = collect_stereodescriptors(mol, {central: 2}, include_near_parent_ez=False)
    if descriptors:
        name = format_stereodescriptor_string(descriptors) + name
    return name


def _alpha_key(acylate: str) -> str:
    """Alphabetization key: strip enclosing marks, locants, and the
    non-alphabetising stereo/config noise.

    : route the descriptor strip through the SHARED ``strip_alphanumerical_noise``
    primitive, which removes the Greek α/β/ξ stereodescriptors and the D/L
    configuration that the old bare regex kept — so a sugar substituent
    alphabetises by its stem regardless of its descriptor SPELLING. The old form
    left the leading ``β`` in the key (Greek β = U+03B2 sorts AFTER ASCII letters),
    so switching the anomer emit from ``beta-`` to ``β-`` flipped a glycosyloxy vs
    hydroxy citation order (the glycosphingolipid greek-emit regression).
    """
    import re
    from ..assembly.naming_utils import strip_alphanumerical_noise
    s = strip_alphanumerical_noise(acylate)
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
    in the substitutive (neutral) regime (honest-gate,)."""
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
            return None  # neutral substitutive form not RT-verified for this head → defer
        # nesting ORDER (BB 7444; escalation, BB 7509) under the marks requirement (BB 7232): {[(<head-alkoxy>)hydroxyphosphoryl]oxy}
        rendered = f"{num[a]}-{{[({head_alkoxy})hydroxyphosphoryl]oxy}}"
        #: a composed prefix sorts at the first letter of its COMPLETE
        # name ('aminoethoxy...'), not at 'phosphoryl'.
        from ..assembly.naming_utils import alpha_sort_key as _ask
        subs.append((_ask(f"[({head_alkoxy})hydroxyphosphoryl]oxy"), rendered))
    if not subs:
        return ""
    subs.sort(key=lambda t: t[0])  # alphabetical by substituent name
    return "".join(r for _, r in subs)


# --------------------------------------------------------------------------- #
# Public dispatcher
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Phospholipid assembler — functional-class phosphate diester
# --------------------------------------------------------------------------- #
# Cationic head groups that use the functional-class "phosphate" (zwitterion) form.
# Neutral heads (ethanolamine,...) route to the substitutive 'hydroxyphosphoryl-oxy'
# form instead (see _build_glycerol_prefixes / _HEAD_GROUP_ALKOXY).
_HEAD_GROUP_ALKYL = {
    "choline": "2-(trimethylazaniumyl)ethyl",
}


def _bis_enclose(inner: str, mult: str) -> str:
    """Enclose a multiplied compound substituent at its nesting level
    (the Blue Book): '(' for a bare one, then '[', '{' as the inner marks
    deepen -- 'bis{[(9Z)-hexadec-9-enoyl]oxy}', never 'bis[[(9Z)-...]oxy]'."""
    return f"{mult}{apply_enclosing_marks(inner, -1)}"


# Multiplier word for free acidic -OH on the functional-class phosphate ester.
_HYDROGEN_WORD = {1: "hydrogen", 2: "dihydrogen"}


def _collect_head_skeleton(mol, head_c, head_o_idx) -> list:
    """BFS the head-group skeleton from ``head_c``, blocked at the phospho-ester
    oxygen (so the phosphate is excluded). Returns the atom-index list naming the
    head substituent."""
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
    # functional-class "phosphate" form. Neutral heads (ethanolamine,...) need
    # the substitutive 'hydroxyphosphoryl-oxy' prefix form → route to the glyceride
    # assembler (phospho handled as a C-3 prefix). RESOLVED: OPSIN's functional-class
    # "phosphate" yields the anionic/zwitterion form (matches PC, not neutral PE).
    head_alkyl = _HEAD_GROUP_ALKYL.get(head_desc)
    if head_alkyl is None:
        # a phase: a FREE phosphatidic acid (no head group beyond the
        # phosphate) is named as the functional-class phosphate monoester
        # '<2,3-bis(acyloxy)propyl> dihydrogen phosphate'. The
        # detector tags the bare -OPO(OH)2 site head_desc='phosphate'.
        if head_desc == "phosphate":
            glyceryl = _diacyl_glyceryl(mol, match, central, phospho_atom, acyl_atoms, atom_site)
            if glyceryl is None:
                return None
            return f"{glyceryl} dihydrogen phosphate"
        # SL: a neutral organic head (serine -> phosphatidylserine,
        # named FROM STRUCTURE via the structure-loss-free substituent namer ->
        # functional-class hydrogen-phosphate diester '<glyceryl> <head> hydrogen
        # phosphate'. inositol (stereo-entangled ring head), ethanolamine/glycerol
        # (substitutive form) keep their existing paths. Fail-closed: declines to
        # the substitutive glyceride form when the head is not cleanly nameable.
        if head_desc not in ("choline", "ethanolamine", "glycerol", "inositol"):
            # W8-P7c.3: phosphatidylserine's PIN is on the L-serine
            # parent (the serine carboxylic acid outranks the phosphorus oxoacid,
            # -> O-{[<glyceryl-propoxy>]hydroxyphosphoryl}-<L|D>-serine, NOT
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
    if head_desc == "choline":
        # The choline head '2-(trimethylazaniumyl)ethyl' cites a carbon-substituted
        # N+ as azaniumyl; its PIN form is '2-(N,N-dimethylmethanaminiumyl)ethyl'
        # method (1),; prints no PIN for the
        # zwitterion), which cannot be verified -- the same rule and the same
        # label as every other such prefix.
        from ..assembly.substituent_naming import record_amine_cation_prefix
        record_amine_cation_prefix(head_alkyl)
    # functional-class diester: [<glyceryl>] <head-alkyl> phosphate (glyceryl bracketed first)
    return f"{apply_enclosing_marks(glyceryl, -1)} {head_alkyl} phosphate"


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
        # marks at each prefix's own depth (the Blue Book), and
        # (b) (:6944): a hyphen after the closing mark when a locant
        # follows -- '3-(hexadecanoyloxy)-2-{...}propyl', never '...oxy)2-[...'.
        gly_subs = "-".join(
            f"{l}-{apply_enclosing_marks(ax, -1)}" for l, ax in parts
        )
    glyceryl = f"{gly_subs}propyl"
    # backbone C-2 R/S (local locant 2 = central) inside the glyceryl substituent
    descriptors = collect_stereodescriptors(mol, {central: 2}, include_near_parent_ez=False)
    if descriptors:
        glyceryl = format_stereodescriptor_string(descriptors) + glyceryl
    return glyceryl


# --------------------------------------------------------------------------- #
# Sphingolipid / ceramide assembler — amide PCG, sphingoid N-substituent
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
