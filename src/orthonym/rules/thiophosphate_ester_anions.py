"""The anion of an ester of a phosphorothioic acid: 'O-(2,5-dioctylphenyl) phosphorodithioate'.

 "Acid esters of organic acids" (the Blue Book): "Preferred IUPAC names of acid esters of inorganic acids
as discussed in are formed by the method of 'hydrogen salts'" ('phenyl hydrogen phosphate (PIN)',:41005); the anion
of such an ester is the name of the ester without 'hydrogen' ('phenyl phosphate'), which the producer of the phosphate ester
anions (``acid_ester_anion.name_acid_ester_anion``) builds for the oxygen acid. The same anion of a thio acid has an 'S-' for an
'O-' or more ('phosphorothioate', 'phosphorodithioate'), the thio acids being named by the functional replacement of
"Mononuclear noncarbon oxoacids modified by functional replacement nomenclature" (:35592; 'diethylphosphinodithioic acid (PIN)',
:35606), and no producer built it: the thio acid ester is named by the replacement ladders as
'[(1-oxo-1-sulfanyl-2-thia-1-phosphaethoxy)]benzene' where it is named at all.

The centre is one phosphorus atom, neutral or the cation of a semipolar pair (``P+-O-`` is the drawing of ``P=O``), with
one or two ester oxygens on carbon groups and every other neighbour a terminal chalcogen: an oxo or thioxo group, or an anionic
O- or S-. The groups are named by the substituent-prefix namer and cited as the words of an ester, each with the locant ``O``
of the oxygen it stands on, in alphanumerical order, ``alpha_sort_key_total``), 'O,O-di'/'O,O-bis' for two of one; the
name ends in 'phosphorothioate', 'phosphorodithioate' or 'phosphorotrithioate' for the number of sulfur atoms.

The locant is essential, not decoration. An ester of a thio acid has two constitutions, the ester on an oxygen or on a
sulfur ('O-phenyl phosphorodithioate' is PhO-P, 'S-phenyl phosphorodithioate' is PhS-P, and OPSIN 2.9.0 reads the second as
``[O-]P([O-])(=S)Sc1ccccc1``); a bare 'phenyl phosphorodithioate' reads back only by OPSIN's default of the oxygen, so a name
without the locant leaves the structure to a reader's default, which "Citation of locants" (:2867, sentence:2869)
forbids where a locant is essential. The Blue Book writes it so: (:46016) 'sodium (R)-(O-methyl O-phenyl
phosphorothioate) (PIN)' and 'sodium (R)-(O-propan-2-yl methylphosphonothioate) (PIN)' (the alternative 'O-(1-methylethyl)'
shows a compound word enclosed after its locant), and "Esters of mononuclear noncarbon oxoacids" (:35916)
'S-ethyl O-methyl phenylphosphonothioate (PIN)' (:35954).

The label stays below PIN. The book's PIN names are those of a SALT of the diester anion (sodium (R)-(O-methyl O-phenyl
phosphorothioate)); it states no PIN for the free anion, none for the monoester, none for a dithioate, and the anion drawn
here is delocalised (a standard InChIKey cannot tell ``P(=O)([S-])`` from ``P(=S)([O-])``), so the chalcogens are not cited.
The form of the diester anion is the anion part of the book's PIN, and that is the only claim made. The name ships only when
OPSIN reads it back to the input's full InChIKey with the hydrons and the charge where they were: the drawings
``P(=O)([S-])`` and ``P(=S)([O-])`` of one anion share a key and are one anion.
"""
from rdkit import Chem

_ENDING = {1: 'phosphorothioate', 2: 'phosphorodithioate', 3: 'phosphorotrithioate'}


def _terminal_chalcogens(mol, phosphorus):
    """``(terminals, esters)`` of the phosphorus atom: its terminal chalcogen neighbours as ``(symbol, atom index)`` and its ester
    oxygens as ``(oxygen index, carbon index)``; None for a neighbour of another kind (an N, a C, a ring bond, an OH)."""
    terminals, esters = [], []
    for nb in phosphorus.GetNeighbors():
        sym = nb.GetSymbol()
        if sym not in ('O', 'S') or nb.GetIsotope():
            return None
        bond = mol.GetBondBetweenAtoms(phosphorus.GetIdx(), nb.GetIdx())
        if nb.GetDegree() == 1 and nb.GetTotalNumHs() == 0:
            single = bond.GetBondType() == Chem.BondType.SINGLE
            if single and nb.GetFormalCharge() == 0:
                return None                         # a hydrogen-free single-bonded terminal atom: no valence
            terminals.append((sym, nb.GetIdx()))
        elif sym == 'O' and nb.GetDegree() == 2 and bond.GetBondType() == Chem.BondType.SINGLE and not nb.IsInRing():
            other = next(x for x in nb.GetNeighbors() if x.GetIdx() != phosphorus.GetIdx())
            if other.GetSymbol() != 'C' or nb.GetFormalCharge():
                return None
            esters.append((nb.GetIdx(), other.GetIdx()))
        else:
            return None                             # an OH, an S-S, a ring bond, a hydroperoxy
    return terminals, esters


def thiophosphate_ester_anion(mol, style: str = 'pin') -> str:
    """The name of ``mol``, the anion of an ester of a phosphorothioic acid (see the module docstring); '' on any decline."""
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return ''
    centres = [a for a in mol.GetAtoms() if a.GetSymbol() == 'P']
    if len(centres) != 1 or centres[0].IsInRing() or centres[0].GetDegree() != 4 or centres[0].GetNumRadicalElectrons():
        return ''
    phosphorus = centres[0]
    found = _terminal_chalcogens(mol, phosphorus)
    if found is None:
        return ''
    terminals, esters = found
    sulfur = sum(1 for sym, _ in terminals if sym == 'S')
    if not 1 <= len(esters) <= 2 or len(terminals) != 4 - len(esters) or not 1 <= sulfur <= 3:
        return ''
    charged = {a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()}
    if not charged <= {phosphorus.GetIdx()} | {i for _, i in terminals} or any(
            a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms()):
        return ''
    if Chem.GetFormalCharge(mol) != -(len(terminals) - 1):
        return ''           # the anion of the ester with one of its terminal atoms double bonded, the others each -1
    from ..assembly.naming_utils import alpha_sort_key_total, enclose_if_compound, multiplied_component
    from .hetero_group_anions import _group_atoms, _group_prefix
    words = []
    for oxygen, carbon in esters:
        atoms = _group_atoms(mol, oxygen, carbon)
        prefix = _group_prefix(mol, atoms, carbon, oxygen) if atoms else None
        if not prefix:
            return ''
        words.append(prefix)
    # (:46016) 'O-methyl O-phenyl phosphorothioate': each word is cited with the locant O of its oxygen, in
    # order; the two words of one name are 'O,O-di...'. A compound word is enclosed after its locant ('O-(propan-2-yl)').
    words.sort(key=alpha_sort_key_total)
    if len(words) == 2 and words[0] == words[1]:
        cited = ['O,O-' + multiplied_component(2, words[0], enclose_if_compound(words[0]))]
    else:
        cited = ['O-' + enclose_if_compound(word) for word in words]
    name = ' '.join(cited + [_ENDING[sulfur]])
    from .charged_router import _cation_name_rt_ok
    if not _cation_name_rt_ok(name, mol):
        return ''
    from ..metrics.provenance import record_non_pin_fragment
    record_non_pin_fragment(name)
    return name
