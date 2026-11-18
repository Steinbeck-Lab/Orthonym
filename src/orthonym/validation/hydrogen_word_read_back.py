"""Read back an adduct name with a hydrogen component that OPSIN does not read.

 "Names for hydrogen atoms and ions", Table 8.1 (the Blue Book) names the hydrogen cation 'hydron'
(natural composition), 'proton' (1H), 'deuteron' (2H) and 'triton' (3H), and the anion 'hydride', 'protide',
'deuteride' and 'tritide'. The isotopically modified dihydrogens take the nuclide descriptors of
("nuclide descriptors", 'dichloro(2H2)methane (PIN)',:43728) on 'dihydrogen': '(2H1)dihydrogen', '(2H2)dihydrogen'
(the book gives no name of its own for them). OPSIN reads none of these words, and a dihydrogen beside a radical
component makes it read the whole adduct as one component.

A adduct name ("Organic adducts",:3765) is the names of its individual compounds joined by em dashes, with
the proportions after them. Its structure is the structure of each component, so the reader here reads each
component on its own: a hydrogen word from:data:`HYDROGEN_WORDS`, every other component by OPSIN (the radical setting
on, as the validity gate reads). The name ships through this reader only when it reproduces the drawn structure
exactly, every hydrogen an atom (:func:`hydrogen_word_round_trip`): a standard InChIKey does not see a hydron (it
goes into the protonation layer: 'acetate—hydron (1/1)' and acetic acid share one key).
"""
import re
from typing import List, Optional, Tuple

from rdkit import Chem

#: the hydrogen words a component of an adduct name may be, with their structure
HYDROGEN_WORDS = {
    "hydron": "[H+]",
    "proton": "[1H+]",
    "deuteron": "[2H+]",
    "triton": "[3H+]",
    "hydride": "[H-]",
    "protide": "[1H-]",
    "deuteride": "[2H-]",
    "tritide": "[3H-]",
    "dihydrogen": "[H][H]",
    "(2H1)dihydrogen": "[2H][H]",
    "(2H2)dihydrogen": "[2H][2H]",
    "(3H1)dihydrogen": "[3H][H]",
    "(3H2)dihydrogen": "[3H][3H]",
    "(2H1,3H1)dihydrogen": "[2H][3H]",
}

#: the words that make a name need this reader: every hydrogen word but 'hydride', which OPSIN reads beside a metal
#: cation ('caesium(1+)—hydride'); reading a 'hydride' name here as well let the adduct namer take another spelling of
#: a large mixture and search its isotope placements again and again (a 32 s row ran past 300 s)
_UNREAD_WORDS = frozenset(HYDROGEN_WORDS) - {"hydride"}


#: the rule id of a read through this reader (``metrics.provenance.reconstructed_outcome``)
HYDROGEN_WORDS_RULE = "hydrogen_words"

_ADDUCT = re.compile(r"(?P<body>[^—]+(?:—[^—]+)+) \((?P<ratio>\d+(?:/\d+)+)\)")


def _components(name: str) -> Optional[List[Tuple[str, int]]]:
    m = _ADDUCT.fullmatch(name)
    if m is None:
        return None
    parts, ratio = m.group("body").split("—"), m.group("ratio").split("/")
    if len(parts) != len(ratio):
        return None
    return [(p, int(r)) for p, r in zip(parts, ratio)]


#: a multiplied element word ('dioxygen', 'trioxygen', 'dinitrogen'): no locant, a multiplying prefix and letters
_MULTIPLIED_WORD = re.compile(r"(di|tri|tetra|penta|hexa|hepta|octa|nona|deca)[a-z]+")

#: a word OPSIN reads in place of each hydrogen word it does not read, with the same charge and no other part in the
#: balance of the charges (``StructureBuilder.balanceChargeIfPossible``: a neutral noble gas has no typical charge, a
#: cation with its charge stated keeps it, the hydride ion is no acid atom), and the atom of the stand-in
_STAND_IN = {0: ("neon", "Ne"), 1: ("francium(1+)", "Fr"), -1: ("hydride", "H")}


def _stand_in_read(comps) -> Optional[Chem.Mol]:
    """OPSIN's reading of the adduct ``comps`` with each unread hydrogen word read as its stand-in, the stand-in atoms
    removed; None when OPSIN reads no such name (OPSIN 2.9.0 reads many adduct names in one component order only)."""
    from ..namer import _validity_gate_name_to_smiles
    words, removed = [], {}
    for word, n in comps:
        if word in _UNREAD_WORDS:
            charge = Chem.GetFormalCharge(Chem.MolFromSmiles(HYDROGEN_WORDS[word]))
            stand_in, element = _STAND_IN[charge]
            word = stand_in
            removed[element] = removed.get(element, 0) + n
        words.append((word, n))
    read = _validity_gate_name_to_smiles("—".join(w for w, _n in words) + " ("
                                         + "/".join(str(n) for _w, n in words) + ")")
    mol = Chem.MolFromSmiles(read) if read else None
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in removed and a.GetDegree() == 0
                       and (a.GetSymbol() != "H" or a.GetFormalCharge() == -1)), reverse=True):
        symbol = rw.GetAtomWithIdx(idx).GetSymbol()
        if removed.get(symbol, 0) > 0:
            removed[symbol] -= 1
            rw.RemoveAtom(idx)
    return rw.GetMol() if not any(removed.values()) else None


def read_back_with_hydrogen_words(name: str) -> Optional[str]:
    """The SMILES of the adduct ``name`` as OPSIN would read it if it read the hydrogen words, every hydrogen an
    explicit atom; None when ``name`` is not an adduct name with a hydrogen word, or a component does not read.

    OPSIN reads an adduct name as a whole and balances its charges (``StructureBuilder.balanceChargeIfPossible``): an
    element word without a charge or an oxidation number takes its typical charge ('ethane—yttrium' beside a carbanion
    reads as Y+), and in a negative mixture of several components the acid atoms of the suffixes are neutralised
    ('acetate—dihydrogen (1/1)' reads as acetic acid in an OPSIN that reads 'dihydrogen'); it also reads some words
    otherwise inside an adduct ('ethanol—dioxygen (1/1)' is [O].[O]). So the name is read by OPSIN as a whole with each
    unread hydrogen word replaced by a word of the same charge that takes no part in that balance (:data:`_STAND_IN`),
    and the hydrogens put in place of the stand-ins. OPSIN 2.9.0 reads many adduct names in one component order only
    ('ethanol—acetaldehyde (1/1)' fails, 'acetaldehyde—ethanol (1/1)' reads); where it reads none, the components are
    read one by one and the reading is refused where the balance could change it: a bare element word beside
    unbalanced ions, a negative mixture with an acid anion named by its suffix, or a multiplied element word."""
    comps = _components(name)
    if comps is None or not any(word in _UNREAD_WORDS for word, _n in comps):
        return None
    from ..namer import _validity_gate_name_to_smiles
    pieces = [HYDROGEN_WORDS[w] for w, n in comps if w in _UNREAD_WORDS for _ in range(n)]
    whole = _stand_in_read(comps)
    if whole is not None:
        mol = Chem.MolFromSmiles(".".join(pieces + [Chem.MolToSmiles(whole)]))
        return None if mol is None else Chem.MolToSmiles(Chem.AddHs(mol))
    bare_atom, split_word, acid_anion, alone = False, False, False, []
    for word, n in comps:
        if word in _UNREAD_WORDS:
            continue
        smi = _validity_gate_name_to_smiles(word)
        piece = Chem.MolFromSmiles(smi) if smi else None
        if piece is None:
            return None
        # 'dioxygen': a multiplied element word, read as separate atoms inside an adduct
        split_word = split_word or (len({a.GetAtomicNum() for a in piece.GetAtoms()}) == 1 and piece.GetNumAtoms() > 1
                                    and not any(a.GetTotalNumHs() for a in piece.GetAtoms())   # not 'trioxidane'
                                    and _MULTIPLIED_WORD.fullmatch(word) is not None)
        # the acid atom of a suffix ('acetate', 'ethanolate', 'sulfite'), which OPSIN neutralises in a negative mixture;
        # an anionic prefix ('oxido', 'sulfinato') or an ion word ('chloride', 'methanide') it leaves
        acid_anion = acid_anion or (Chem.GetFormalCharge(piece) < 0 and word.endswith(("ate", "ite")))
        alone += [smi] * n
        bare_atom = bare_atom or (piece.GetNumAtoms() == 1 and "(" not in word
                                  and piece.GetAtomWithIdx(0).GetFormalCharge() == 0)
    mol = Chem.MolFromSmiles(".".join(pieces + alone))
    if mol is None or split_word:
        return None
    net = Chem.GetFormalCharge(mol)
    if (bare_atom and net != 0) or (len(comps) > 1 and net < 0 and acid_anion):
        return None  # a reader that balances charges could change the bare element or an acid atom
    return Chem.MolToSmiles(Chem.AddHs(mol))


def hydrogen_word_round_trip(name: str, smiles: Optional[str]) -> Optional[str]:
    """:data:`HYDROGEN_WORDS_RULE` when OPSIN does not read ``name`` and:func:`read_back_with_hydrogen_words` gives
    exactly the drawn structure ``smiles`` (the same graph with every hydrogen an atom: the same atoms, charges,
    isotopes, radical electrons and component counts); else None."""
    if not smiles:
        return None
    read = read_back_with_hydrogen_words(name)
    if read is None:
        return None
    from ..namer import _validity_gate_name_to_smiles
    if _validity_gate_name_to_smiles(name) is not None:
        return None  # OPSIN reads the name itself: its own gates decide
    drawn = Chem.MolFromSmiles(smiles)
    if drawn is None or Chem.MolToSmiles(Chem.AddHs(drawn)) != read:
        return None
    return HYDROGEN_WORDS_RULE
