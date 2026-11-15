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


def read_back_with_hydrogen_words(name: str) -> Optional[str]:
    """The SMILES of the adduct ``name`` read component by component (hydrogen words from:data:`HYDROGEN_WORDS`, the
    others by OPSIN), every hydrogen an explicit atom; None when ``name`` is not an adduct name with a hydrogen word,
    or a component does not read."""
    comps = _components(name)
    if comps is None or not any(word in _UNREAD_WORDS for word, _n in comps):
        return None
    from ..namer import _validity_gate_name_to_smiles
    pieces = []
    for word, n in comps:
        smi = HYDROGEN_WORDS.get(word) or _validity_gate_name_to_smiles(word)
        if not smi:
            return None
        pieces += [smi] * n
    mol = Chem.MolFromSmiles(".".join(pieces))
    return None if mol is None else Chem.MolToSmiles(Chem.AddHs(mol))


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
