"""Acyl halides on a ring named from the acid they come from: '4-methylbenzoyl chloride'.

 'Acid halides' (the Blue Book): "Acyl halides are named by citing the name of the corresponding acyl
group followed by the class name(s) of the halide(s)"; (:30624) gives the acyl group of an acid named
with the suffix 'carboxylic acid' ('-carbonyl') and the '-ic acid' -> '-yl' acyl groups ('benzoyl').
'2-carbonocyanidoyl-5-methylbenzoyl chloride (PIN)' (:31543) is a substituted one. The ring acid halide producer
wrote only 'benzoyl chloride' or 'cyclohexanecarbonyl chloride' and so named no substituted ring: the substituents
fell to the general engine ('1-[chloro(oxo)methyl]-4-methylbenzene', no PIN). Here the acid with every acyl halide
group drawn as a carboxy group is named by the engine, and its suffix is changed to the acyl form
(``lipids._acid_to_acyl``, which keeps a non-PIN record of the acid name on the acyl). Acid halides rank below acids,
anhydrides and esters, so the input has no carboxy group of its own and every carboxy group of the acid is one
of its acyl halide groups; the multiplied suffix must count all of them, or the name is not built.
"""
import re
from typing import List, Optional, Tuple

from rdkit import Chem

#: the halide word of each halogen
_WORD = {"F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide"}
#: the multiplied acid suffix of a name and its multiplier ('-1,4-dicarboxylic acid')
_MULTIPLIED_ACID = re.compile(r"-[\d,]+-(di|tri|tetra)(?:carboxylic|oic) acid$")
_ACID = re.compile(r"\bcarboxy\b|carboxylic acid|oic acid|ic acid$")


def acyl_halide_from_acid(mol, combined: List[Tuple[tuple, str]]) -> Optional[str]:
    """'4-methylbenzoyl chloride' for ``mol`` whose acyl halide groups ``combined`` (``((C, O, X), symbol)`` each) carry
    halogen each; the groups of the senior halogen on ring carbons are the suffix, any others are prefixes
    ('3-chloro-3-oxopropyl', '4-carbonobromidoylbenzoyl chloride') when no branch holds more of them than the ring.
    None for any other case (a carboxy group of the input's own, an acid name whose suffix does not count every ring
    group)."""
    from .isotope_alternative import alternative_skeleton_active
    if not combined or alternative_skeleton_active():
        return None                                  # the second skeleton of a labelled molecule (isotope_alternative)
    # class 10 (the Blue Book): acid halides "in the order of the halogen atoms (-F > -Cl > -Br > -I)"
    senior = min((sym for _m, sym in combined), key=list(_WORD).index)
    combined = [(m, sym) for m, sym in combined if sym == senior]
    if len(combined) > 4:
        return None
    if any(mol.GetAtomWithIdx(m[2]).GetIsotope() for m, _s in combined):
        return None                                  # a labelled halogen: the acid would lose the label
    if mol.HasSubstructMatch(_CARBOXY):
        return None                                  # an acid is senior; another producer names it
    ring_groups = [m for m, _s in combined if _on_ring_atom(mol, m)]
    if not ring_groups or _most_groups_on_one_branch(mol, [m for m, _s in combined if m not in ring_groups]) > len(
            ring_groups):
        return None                                  # a chain with more groups is the parent
    acid = _acid_smiles(mol, [m[2] for m in ring_groups])
    if acid is None:
        return None
    from .lipids import _acid_to_acyl, _name_fragment
    acid_name = _name_fragment(acid)
    if not acid_name or not acid_name.endswith(" acid"):
        return None
    found = _MULTIPLIED_ACID.search(acid_name)
    count = {"di": 2, "tri": 3, "tetra": 4}[found.group(1)] if found else 1
    if count != len(ring_groups) or len(_ACID.findall(acid_name)) != 1:
        return None                                  # a 'carboxy' prefix or another acid word is a group not converted
    acyl = _acid_to_acyl(acid_name)
    if not acyl:
        return None
    from ..assembly.naming_utils import get_multiplier_prefix
    word = _WORD[combined[0][1]]
    return f"{acyl} {get_multiplier_prefix(count, word)}{word}"


_CARBOXY = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")


def _on_ring_atom(mol, match) -> bool:
    """True when the acyl carbon of ``match`` is joined to exactly one other atom, a ring carbon or ring nitrogen:
    'Carboxy groups attached to cyclic parent hydrides... are always named by using the suffix 'carboxylic acid''
    , the Blue Book), 'pyrrolidine-1-carboxylic acid (PIN)' (:29892)."""
    others = [nb for nb in mol.GetAtomWithIdx(match[0]).GetNeighbors() if nb.GetIdx() not in match]
    return len(others) == 1 and others[0].IsInRing() and others[0].GetSymbol() in ("C", "N")


def _most_groups_on_one_branch(mol, matches) -> int:
    """The largest number of acyl halide groups ``matches`` on one acyclic branch (ring atoms removed), 0 for none.
    Within one class the parent has the most principal groups, and a ring wins a tie: 'for the same number of
    characteristic groups cited as the principal characteristic group, a ring is always selected as the parent
    hydride', the Blue Book); '4-(3-chloro-3-oxopropyl)benzoyl chloride'."""
    if any(mol.GetAtomWithIdx(m[0]).IsInRing() for m in matches):
        return len(matches) + 1                      # an acyl carbon in a ring: not this producer's shape
    seen, best = set(), 0
    for match in matches:
        if match[0] in seen:
            continue
        branch, stack = set(), [match[0]]
        while stack:
            x = stack.pop()
            if x in branch or mol.GetAtomWithIdx(x).IsInRing():
                continue
            branch.add(x)
            stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(x).GetNeighbors())
        seen |= branch
        best = max(best, sum(1 for m in matches if m[0] in branch))
    return best


def _acid_smiles(mol, halogen_atoms: List[int]) -> Optional[str]:
    """SMILES of ``mol`` with each atom of ``halogen_atoms`` replaced by a hydroxy oxygen, or None."""
    rw = Chem.RWMol(mol)
    for idx in halogen_atoms:
        atom = rw.GetAtomWithIdx(idx)
        atom.SetAtomicNum(8)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    return Chem.MolToSmiles(rw)
