"""Biaryl stereogenic axes (atropisomers): the configuration the standard InChIKey cannot see.

 "Stereogenic axis" (the Blue Book,:44824): "Stereodescriptors 'M' and 'P' are used in preferred
IUPAC names. The lowest locant on the axis is cited before the stereodescriptor." "Specification of
stereogenic axes" (:49797) gives the PINs '(1M)-6,6′-dinitro[1,1′-biphenyl]-2,2′-dicarboxylic acid' (:49803) and
'(2P)-2-[2-(hydroxymethyl)naphthalen-1-yl]-3,5-dimethylphenol' (:49828).

An axis reaches Orthonym only in a CXSMILES string (``|wU:..|``/``|wD:..|``, RDKit's ``STEREOATROPCW/CCW``); RDKit drops
an allene configuration at parse, and a Mol with an atropisomer tag is declined at input (``input_mol``). The M and P
forms share one standard InChIKey, so a name is checked here atom by atom:

*:func:`axis_verdict` -- 'n/a' (the input has no axis: every other molecule), 'match' (the name's read-back has the
  input's one axis with the same M/P), 'omission' (the name cites no axial descriptor: it names the input without its
  axis and is not the PIN,:44624), 'mismatch' (anything else, fail closed).
*:func:`axis_decorated` -- the descriptor-free name with the axis cited at the front, kept only on 'match'.
*:func:`axial_locant_twin` -- the spelling OPSIN reads: OPSIN main reads '(M)-' and no locant before an axial
  descriptor (dan2097/opsin#356); OPSIN 2.9.0 reads no axial descriptor, so there no name gets one.

Labels come from ``rdCIPLabeler`` on a fresh parse of a string, never from a renumbered Mol (a renumbering can flip
RDKit's atropisomer label), and the same labeller reads both sides. OPSIN's extended SMILES puts the wedge on a bond
that RDKit may read as a Kekule double bond, where it drops it without an error;:func:`_read_extended` moves the wedge
onto RDKit's aromatic writing of the same atoms and fails closed when any wedge is lost.
"""
import re
from typing import Dict, Optional, Tuple

from rdkit import Chem

_ATROP = (Chem.BondStereo.STEREOATROPCW, Chem.BondStereo.STEREOATROPCCW)
_CX_WEDGE = re.compile(r"\|[^|]*\bw[UD]:[^|]*\|")
#: a parenthesised group with no enclosing mark inside it, and one stereodescriptor item of such a group
_GROUP = re.compile(r"\(([^()\[\]{}]*)\)")
_ITEM = re.compile(r"(\d+[a-z]?['′]*)?(Ra|Sa|Rp|Sp|[RSEZMPrs])\*?")
#: mass numbers that make '(32P)' or '(226Ra)' an isotope descriptor, not a locanted axial descriptor
_NUCLIDE_MASSES = {"P": range(26, 48), "Ra": range(200, 236)}
#: a token of a SMILES string, for the bond order of OPSIN's CXSMILES (a ring bond counts at its closing digit)
_TOKEN = re.compile(r"\[[^\]]+\]|Cl|Br|[BCNOPSFIbcnops]|%\d\d|\d|[-=#:/\\$]|\(|\)|\.")


def _axis_mol(smiles: Optional[str]) -> Optional[Chem.Mol]:
    """A fresh parse of ``smiles`` when it configures at least one axis, else None."""
    if not smiles or not isinstance(smiles, str) or not _CX_WEDGE.search(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not any(b.GetStereo() in _ATROP for b in mol.GetBonds()):
        return None
    return mol


def _axes(mol: Chem.Mol) -> Optional[Dict[Tuple[int, int], str]]:
    """{(stereo-free canonical ranks of the two axis atoms): 'M' or 'P'}; None when an axis has no label or two axes
    share a key."""
    from rdkit.Chem import rdCIPLabeler
    labelled = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(labelled)
    bare = Chem.Mol(mol)
    Chem.RemoveStereochemistry(bare)
    for bond in bare.GetBonds():            # RemoveStereochemistry keeps the atropisomer flag, and the ranks see it
        if bond.GetStereo() in _ATROP:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    ranks = list(Chem.CanonicalRankAtoms(bare, breakTies=False))
    out: Dict[Tuple[int, int], str] = {}
    for bond in labelled.GetBonds():
        if bond.GetStereo() not in _ATROP:
            continue
        label = bond.GetProp("_CIPCode") if bond.HasProp("_CIPCode") else None
        key = tuple(sorted((ranks[bond.GetBeginAtomIdx()], ranks[bond.GetEndAtomIdx()])))
        if label not in ("M", "P") or key in out:
            return None
        out[key] = label
    return out


def _cx_bonds(plain: str):
    """The bonds of ``plain`` in CXSMILES order as (atom, atom), and the atom count."""
    bonds, stack, prev, n, rings = [], [], None, -1, {}
    for tok in _TOKEN.findall(plain):
        if tok == "(":
            stack.append(prev)
        elif tok == ")":
            prev = stack.pop()
        elif tok == ".":
            prev = None
        elif tok[0] in "-=#:/\\$":
            pass
        elif tok[0].isdigit() or tok[0] == "%":
            if tok in rings:
                bonds.append((rings.pop(tok), prev))
            else:
                rings[tok] = prev
        else:
            n += 1
            if prev is not None:
                bonds.append((prev, n))
            prev = n
    return bonds, n + 1


def _read_extended(ext: Optional[str]) -> Optional[Chem.Mol]:
    """RDKit's reading of OPSIN's extended SMILES with every wedge kept, or None."""
    if not ext:
        return None
    plain, _, cx = ext.strip().partition(" ")
    mol = Chem.MolFromSmiles(plain)
    if mol is None:
        return None
    wedges = re.findall(r"(w[UD]):([\d.,]+)", cx)
    if not wedges:
        return mol
    bonds, atoms = _cx_bonds(plain)
    if atoms != mol.GetNumAtoms() or len(bonds) != mol.GetNumBonds() or any(
            mol.GetBondBetweenAtoms(*pair) is None for pair in bonds):
        return None
    written = Chem.MolToSmiles(mol, canonical=False)
    props = mol.GetPropsAsDict(True, True)
    atom_order, bond_order = list(props["_smilesAtomOutputOrder"]), list(props["_smilesBondOutputOrder"])
    parts, count = [], 0
    for kind, items in wedges:
        for item in items.strip(",").split(","):
            atom, bond = map(int, item.split("."))
            if bond >= len(bonds) or atom not in bonds[bond]:
                return None
            count += 1
            rd_bond = mol.GetBondBetweenAtoms(*bonds[bond]).GetIdx()
            parts.append(f"{kind}:{atom_order.index(atom)}.{bond_order.index(rd_bond)}")
    out = Chem.MolFromSmiles(f"{written} |{','.join(parts)}|")
    if out is None or sum(b.GetStereo() in _ATROP for b in out.GetBonds()) != count:
        return None
    return out


def _axial_items(name: Optional[str]):
    """(start, end) of the locant (empty span when there is none) of each axial descriptor (M, P, Ra, Sa) that a
    stereo group of ``name`` cites: a group every comma-separated item of which is a stereodescriptor; '(32P)' and
    '(226Ra)' are isotope descriptors and do not count."""
    spans = []
    for group in _GROUP.finditer(name or ""):
        items, offset, found = group.group(1).split(","), group.start(1), []
        for item in items:
            m = _ITEM.fullmatch(item)
            if m is None:
                found = None
                break
            locant, desc = m.group(1) or "", m.group(2)
            nuclide = desc in _NUCLIDE_MASSES and locant.isdigit() and int(locant) in _NUCLIDE_MASSES[desc]
            if desc in ("M", "P", "Ra", "Sa") and not nuclide:
                found.append((offset, offset + len(locant)))
            offset += len(item) + 1
        spans.extend(found or [])
    return spans


def axial_descriptors(name: Optional[str]) -> int:
    """How many axial descriptors (M, P, Ra, Sa) the stereo groups of ``name`` cite."""
    return len(_axial_items(name))


def axial_locant_twin(name: str) -> str:
    """``name`` with the locant of its one axial descriptor left out ('(1M)-' -> '(M)-'), the form OPSIN main reads;
    ``name`` unchanged when it cites no axial descriptor with a locant, or more than one axial descriptor."""
    found = _axial_items(name)
    if len(found) != 1 or found[0][0] == found[0][1]:
        return name
    return name[:found[0][0]] + name[found[0][1]:]


def _read_back(name: str) -> Optional[Chem.Mol]:
    from .opsin_roundtrip import _extended_smiles
    read = _read_extended(_extended_smiles(name))
    twin = axial_locant_twin(name)
    if read is None and twin != name:
        read = _read_extended(_extended_smiles(twin))
    return read


def axis_verdict(input_smiles: Optional[str], name: Optional[str]) -> str:
    """'n/a' | 'match' | 'omission' | 'mismatch' for ``name`` of ``input_smiles`` (see the module docstring)."""
    inp = _axis_mol(input_smiles)
    if inp is None:
        return "n/a"
    if not name or not axial_descriptors(name):
        return "omission"
    want = _axes(inp)
    read = _read_back(name)
    got = _axes(read) if read is not None else None
    if want is None or got is None or len(want) != 1 or want != got:
        return "mismatch"
    return "match"


def record_axis_omission(name: Optional[str]) -> None:
    """Label a name that leaves the input's axis out: it names the input and is not the PIN
    "Stereodescriptors used in substitutive nomenclature", the Blue Book: "In preferred IUPAC names,
    stereodescriptors, preceded by a locant, must be cited to specify each stereogenic unit")."""
    from ..metrics.provenance import record_non_pin_label, record_stereo_unexpressed
    record_stereo_unexpressed(True)
    record_non_pin_label(name, rule="P-91.2.1.2.1", detail="the configuration of a stereogenic axis is not cited")


def key_blind_verdict(input_smiles: Optional[str], opsin_smiles: Optional[str], name: Optional[str]) -> str:
    """The verdict of ``validation.key_blind_double_bonds`` with the axis checked as well: 'mismatch' when either
    says so; an axis left out is labelled here (:func:`record_axis_omission`); else the double-bond verdict."""
    from .key_blind_double_bonds import key_blind_double_bond_verdict
    axis = axis_verdict(input_smiles, name)
    if axis == "mismatch":
        return "mismatch"
    if axis == "omission":
        record_axis_omission(name)
    return key_blind_double_bond_verdict(input_smiles, opsin_smiles)


def _token(locant) -> Optional[str]:
    """A locant of ``opsin_atom_locant_map`` as written in a name: 1, '4a', (1, "'") -> "1'", (4, "'", 'a') -> "4'a"."""
    if locant is None:
        return None
    if isinstance(locant, (tuple, list)):
        return "".join(str(part) for part in locant)
    return str(locant)


def _locant_key(token: str):
    m = re.match(r"(\d+)(['′]*)([a-z]?)(['′]*)$", token)
    return (0, int(m.group(1)), len(m.group(2)), m.group(3), len(m.group(4))) if m else (1, 0, 0, token, 0)


def axis_decorated(input_smiles: Optional[str], name: Optional[str]) -> Optional[str]:
    """``name`` with the one axis of ``input_smiles`` cited at the front ('(1M)-[1,1'-binaphthalene]-2,2'-diol'), when
    OPSIN reads that name (or its:func:`axial_locant_twin`) back to the input's InChIKey and axis; else None.

    The locant is the lowest of the axis atoms' locants in OPSIN's numbering of ``name``; an unprimed
    locant before its primed one,:3193); a leading group of stereodescriptors takes it in locant order."""
    inp = _axis_mol(input_smiles)
    if inp is None or not name or axial_descriptors(name):
        return None
    axes = _axes(inp)
    atrop = [b for b in inp.GetBonds() if b.GetStereo() in _ATROP]
    if axes is None or len(axes) != 1 or len(atrop) != 1:
        return None
    from .opsin_roundtrip import opsin_atom_locant_map
    locants = opsin_atom_locant_map(name, inp)
    if not locants:
        return None
    tokens = [t for t in (_token(locants.get(atrop[0].GetBeginAtomIdx())),
                          _token(locants.get(atrop[0].GetEndAtomIdx()))) if t]
    if not tokens:
        return None
    item = f"{min(tokens, key=_locant_key)}{next(iter(axes.values()))}"
    from ..assembly.stereo_composed_offer import stereo_groups
    lead = re.match(r"\(([^()]*)\)-", name)
    if lead and stereo_groups(lead.group(0)[:-1]):
        items = lead.group(1).split(",") + [item]
        items.sort(key=lambda it: _locant_key(re.match(r"\d+['′]*[a-z]?['′]*", it).group(0))
                   if re.match(r"\d", it) else (0, 0, 0, "", 0))
        decorated = f"({','.join(items)})-{name[lead.end():]}"
    else:
        decorated = f"({item})-{name}"
    read = _read_back(decorated)
    if read is None or Chem.MolToInchiKey(read) != Chem.MolToInchiKey(inp):
        return None
    return decorated if axis_verdict(input_smiles, decorated) == "match" else None


def caller_axis_decorated(mol: Optional[Chem.Mol], name: Optional[str]) -> Optional[str]:
    """:func:`axis_decorated` for the caller's own input, when ``mol`` (a producer's molecule) is that whole input;
    None otherwise (a nested naming of a part, an input with no axis). The axis and its label are read from the
    caller's string, never from ``mol``."""
    if mol is None or not name:
        return None
    from ..namer import _caller_input
    caller = _caller_input()
    inp = _axis_mol(caller)
    if inp is None or inp.GetNumAtoms() != mol.GetNumAtoms():
        return None
    try:
        if Chem.MolToInchiKey(inp) != Chem.MolToInchiKey(mol):
            return None
    except Exception:  # noqa: BLE001 -- a molecule without a key is not compared
        return None
    return axis_decorated(caller, name)


def exit_check(engine, name: str, smiles: str) -> str:
    """The axis check of a public exit (``Orthonym._exit_round_trip_check``), for the caller's own string: a name that
    leaves the axis out ships labelled (:func:`record_axis_omission`); a name whose axis is not the input's is
    withdrawn. Every other name, and every input with no axis, passes unchanged."""
    from ..errors import is_failure_name
    if not name or is_failure_name(name) or _axis_mol(smiles) is None:
        return name
    from ..assembly.fragment_naming import is_top_level_naming
    from ..namer import _caller_input
    caller = _caller_input()
    if not is_top_level_naming() or (caller is not None and smiles != caller):
        return name
    try:
        verdict = axis_verdict(smiles, name)
    except Exception:  # noqa: BLE001 -- fails closed below
        verdict = "mismatch" if axial_descriptors(name) else "omission"
    if verdict == "omission":
        record_axis_omission(name)
    elif verdict == "mismatch":
        return engine._withdraw_at_exit(name, smiles, "exit_axis_configuration")
    return name
