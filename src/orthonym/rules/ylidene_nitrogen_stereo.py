"""The E/Z of a C=N bond that joins an '-ylidene' substituent to a nitrogen of the parent: method (b), below the PIN.

An acylhydrazone R'CO-NH-N=CR2 is the hydrazide 'N'-(R-ylidene)...hydrazide', a semicarbazone the
'2-(R-ylidene)hydrazine-1-carboxamide'. The Blue Book gives no PIN form for the configuration of that C=N:
"Descriptors 'E' and 'Z' to denote double bonds linked to heteroatoms" (the Blue Book) makes the carbon-parent
'(Z)-N-hydroxy(4-chlorophenyl)(phenyl)methanimine (PIN)' and lists the nitrogen-parent
'(Z)-N-[(4-chlorophenyl)(phenyl)methylidene]hydroxylamine' as another name. By user decision 2026-11-23 such a name
ships below the PIN, with the descriptor at the front of the '-ylidene' prefix (1)(b), the Blue Book:
48271: "the stereodescriptor is cited at the front of the name of the substituent group"): 'N'-[(1E)-2-phenyl-
ethylidene]heptanehydrazide', never a front '(1E)' that cites the ylidene group's own carbon, nor '(N'E)'.

The placement is the scope placer's (``rules.prefix_stereo._placed``), re-anchored on OPSIN's numbering and read back
exactly; it is shipped only when the molecule's only stereo units are such bonds (any other descriptor keeps the place
the name gave it), and the name is recorded below the PIN whenever the placer finds such a bond.
"""
import re
from typing import Optional, Tuple

from rdkit import Chem

#: a bare '-ylidene' prefix directly before the hydrazine/hydrazide word ('2-propylidenehydrazine'), which the placer
#: can only reach inside enclosing marks
_BARE_YLIDENE = re.compile(r"(^|[-\s,])([a-z]+ylidene)(?=[a-z]*(?:hydrazin|hydrazid))")


#: an '-ylidene' prefix cited on the hydrazine or hydrazide of the parent ('2-(benzylidene)hydrazine-1-carboxylate',
#: "N'-(propylidene)benzohydrazide")
_YLIDENE_ON_HYDRAZINE = re.compile(r"ylidene[)\]}]*[a-z]*(?:hydrazine|hydrazide)")


def ylidene_nitrogen_name(name: str, smiles: str) -> str:
    """``name`` with the E/Z of such a bond cited in its '-ylidene' prefix, recorded below the PIN; ``name`` itself
    when it has none (``namer``'s stereo backstop)."""
    found = respell_ylidene_nitrogen(name, smiles) if isinstance(name, str) else None
    if found is None:
        return name
    from ..metrics.provenance import record_non_pin_fragment
    record_non_pin_fragment(found[0])
    return found[0]


def _has_ylidene_nitrogen_bond(mol) -> bool:
    """A stereo C=N whose nitrogen (acyclic, two neighbours) leads on to another nitrogen or an oxygen."""
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        atoms = (bond.GetBeginAtom(), bond.GetEndAtom())
        if {a.GetSymbol() for a in atoms} != {"C", "N"}:
            continue
        nitrogen = atoms[0] if atoms[0].GetSymbol() == "N" else atoms[1]
        if (not nitrogen.IsInRing() and nitrogen.GetDegree() == 2
                and any(nb.GetSymbol() in ("N", "O") for nb in nitrogen.GetNeighbors())):
            return True
    return False


def respell_ylidene_nitrogen(name: str, smiles: str) -> Optional[Tuple[str, bool]]:
    """``(name to ship, below the PIN)`` for a name with such a bond, or None (no such bond, or the placer finds none)."""
    if not name or "ylidene" not in name:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not _has_ylidene_nitrogen_bond(mol):
        return None
    from .prefix_stereo import _placed
    from .stereochemistry import strip_stereo
    flat = _BARE_YLIDENE.sub(lambda m: f"{m.group(1)}({m.group(2)})", strip_stereo(name, words=True))
    kept = set()
    try:
        placed = _placed(flat, smiles, kept_inside=kept)
    except Exception:
        return None
    if not kept:
        # a retained '-ylidene' group whose carbon OPSIN gives no locant ('benzylidene'): the placer cannot confirm the
        # attachment, the name keeps its spelling, and a hydrazine/hydrazide parent of the prefix still puts it below
        # the PIN (the same decision)
        return (name, True) if _YLIDENE_ON_HYDRAZINE.search(flat) else None
    stereo_bonds = {b.GetIdx() for b in mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE}
    centres = any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
    if isinstance(placed, str) and not centres and stereo_bonds == kept:
        return placed, True
    return name, True
