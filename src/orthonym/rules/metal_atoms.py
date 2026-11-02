"""Names of the lone atom, the monoatomic ions and the isotopes of every element, and of the hydride ion beside a metal.

One home for the element names of the metals. A metal atom takes its element name; a metal cation
adds its charge number in parentheses, the notation of 'pentaammine(ethanido)osmium(1+) chloride'
, the Blue Book); a nuclide goes in parentheses before the element name, "enclosed in
parentheses... before the part of the compound that is isotopically substituted",
:43718). Two callers read the names:

* the adduct of ``rules/adducts.py``, which cites a one-atom metal component by them at the
  best-effort tier (``metal_atom_name``): "preferred IUPAC names cannot be assigned to mixed adducts"
  (:4667), so such a name is never a PIN;
* the class ``ELEMENT_ATOM`` of the dispatch table, for a molecule that is one atom without hydrogen
  of any element (``lone_atom_name``): a metal admitted to the mechanism ``ELEMENT`` of
  ``data/metal_scope.py``, a metalloid, a non-metal or a noble gas, as an atom, an ion or an isotope.
  The name is a verified systematic name and no PIN: PREFERRED IUPAC NAMES (:2056) labels
  "the names of compounds whose parent hydride contains at least one of the following elements: B,
  Si,... At, and that also contain at least one carbon atom in their structure", and a lone atom
  holds no such pair (``rules/metal_labels.is_lone_atom_name_without_pin`` sets the label).

``inorganic_anion_word`` gives the word of a one-atom or oxo anion in a salt: the table of retained
inorganic anions, and 'hydride' for the hydride ion beside metals admitted to ``CARBANION_SALTS``.
``ammine_component_names`` gives the one-atom component 'ammonia' of an adduct beside metals admitted
to ``AMMINE``.
"""
from typing import Dict, Optional

from ..data.ion_retained_names import INORGANIC_ANIONS, INORGANIC_CATIONS
from ..data.metal_scope import AMMINE, CARBANION_SALTS, ELEMENT, METAL_SCOPE, has_mechanism

# The monoatomic metal components of a disconnected depiction -- 'CC(N)C(=O)O.CC(N)C(=O)O.[Ni]', six
# '[C-]#N' with '[Fe+2]'. "Mixed organic - inorganic adducts" (the Blue Book):
# "organic components in order as described in, inorganic components in order as described
# in Ref 12"; and (:4667) "preferred IUPAC names cannot be assigned to mixed adducts because
# preferred IUPAC names have not yet been determined for inorganic components". (:39735)
# notes no PIN for the Group 1-12 metals either. Such a name is therefore never a PIN.
#
# A metal atom takes its element name; a metal cation adds its charge number in parentheses, the
# notation of 'pentaammine(ethanido)osmium(1+) chloride',:39793). The element set is
# perception.metals.METAL_ELEMENT_SYMBOLS; every spelling below was read back by OPSIN 2.9.0 to the
# bare atom and to its 1+ and 2+ ions (all 94).
METAL_ELEMENT_NAMES: Dict[str, str] = {
    'Ac': 'actinium', 'Ag': 'silver', 'Al': 'aluminium', 'Am': 'americium',
    'As': 'arsenic', 'Au': 'gold', 'B': 'boron', 'Ba': 'barium',
    'Be': 'beryllium', 'Bh': 'bohrium', 'Bi': 'bismuth', 'Bk': 'berkelium',
    'Ca': 'calcium', 'Cd': 'cadmium', 'Ce': 'cerium', 'Cf': 'californium',
    'Cm': 'curium', 'Cn': 'copernicium', 'Co': 'cobalt', 'Cr': 'chromium',
    'Cs': 'caesium', 'Cu': 'copper', 'Db': 'dubnium', 'Ds': 'darmstadtium',
    'Dy': 'dysprosium', 'Er': 'erbium', 'Es': 'einsteinium', 'Eu': 'europium',
    'Fe': 'iron', 'Fm': 'fermium', 'Fr': 'francium', 'Ga': 'gallium',
    'Gd': 'gadolinium', 'Ge': 'germanium', 'Hf': 'hafnium', 'Hg': 'mercury',
    'Ho': 'holmium', 'Hs': 'hassium', 'In': 'indium', 'Ir': 'iridium',
    'K': 'potassium', 'La': 'lanthanum', 'Li': 'lithium', 'Lr': 'lawrencium',
    'Lu': 'lutetium', 'Md': 'mendelevium', 'Mg': 'magnesium', 'Mn': 'manganese',
    'Mo': 'molybdenum', 'Mt': 'meitnerium', 'Na': 'sodium', 'Nb': 'niobium',
    'Nd': 'neodymium', 'Ni': 'nickel', 'No': 'nobelium', 'Np': 'neptunium',
    'Os': 'osmium', 'Pa': 'protactinium', 'Pb': 'lead', 'Pd': 'palladium',
    'Pm': 'promethium', 'Po': 'polonium', 'Pr': 'praseodymium', 'Pt': 'platinum',
    'Pu': 'plutonium', 'Ra': 'radium', 'Rb': 'rubidium', 'Re': 'rhenium',
    'Rf': 'rutherfordium', 'Rg': 'roentgenium', 'Rh': 'rhodium', 'Ru': 'ruthenium',
    'Sb': 'antimony', 'Sc': 'scandium', 'Sg': 'seaborgium', 'Si': 'silicon',
    'Sm': 'samarium', 'Sn': 'tin', 'Sr': 'strontium', 'Ta': 'tantalum',
    'Tb': 'terbium', 'Tc': 'technetium', 'Te': 'tellurium', 'Th': 'thorium',
    'Ti': 'titanium', 'Tl': 'thallium', 'Tm': 'thulium', 'U': 'uranium',
    'V': 'vanadium', 'W': 'tungsten', 'Y': 'yttrium', 'Yb': 'ytterbium',
    'Zn': 'zinc', 'Zr': 'zirconium',
}

#: the elements that are not in ``METAL_ELEMENT_NAMES``: hydrogen, carbon, the non-metals of Groups 15-17, the
#: noble gases and the superheavy elements. A lone atom of one of them takes its element name, as a metal does,
#: unless it is the radical of a mononuclear parent hydride (``RADICAL_PARENT_HYDRIDES``).
NONMETAL_ELEMENT_NAMES: Dict[str, str] = {
    'H': 'hydrogen', 'He': 'helium', 'C': 'carbon', 'N': 'nitrogen', 'O': 'oxygen', 'F': 'fluorine',
    'Ne': 'neon', 'P': 'phosphorus', 'S': 'sulfur', 'Cl': 'chlorine', 'Ar': 'argon', 'Se': 'selenium',
    'Br': 'bromine', 'Kr': 'krypton', 'I': 'iodine', 'Xe': 'xenon', 'At': 'astatine', 'Rn': 'radon',
    'Nh': 'nihonium', 'Fl': 'flerovium', 'Mc': 'moscovium', 'Lv': 'livermorium', 'Ts': 'tennessine',
    'Og': 'oganesson',
}

#: the mononuclear parent hydrides of Table 2.1, the Blue Book-7932) on which a bare atom or a bare
#: monoatomic ion of the element is named, with the standard bonding number of the element. The neutral atom is
#: the radical, (:40426) "named by adding the suffix 'yl' to the name of the parent hydride, eliding
#: the final letter 'e'", 'ylidene' and 'ylidyne' for two and three hydrogen atoms (:40448): 'chloranyl',
#: 'oxidanylidene', 'azanylidyne'; an ion combines the cumulative suffixes 'ide', 'ium', 'ylium' and 'yl'
#: Suffixes,:4777: "Cumulative suffixes are used to designate radicals, ions, radical ions... They can be combined
#: with each other"): 'phosphanetriide', 'azanidylidene', 'tellaniumylidyne', 'silyliumylidyne'. Silicon and germanium
#: are here for their ions only (a neutral atom with four radical electrons takes the element name). Astatine is
#: not here: OPSIN 2.9.0 reads no 'astanyl', so the atom takes the element name 'astatine'. The metals of Groups
#: 13 and 14 (alumane... plumbane) take the element name of the coordination nomenclature.
RADICAL_PARENT_HYDRIDES: Dict[str, tuple] = {
    'B': ('borane', 3), 'Si': ('silane', 4), 'Ge': ('germane', 4), 'N': ('azane', 3), 'P': ('phosphane', 3),
    'As': ('arsane', 3), 'Sb': ('stibane', 3), 'Bi': ('bismuthane', 3), 'O': ('oxidane', 2), 'S': ('sulfane', 2),
    'Se': ('selane', 2), 'Te': ('tellane', 2), 'Po': ('polane', 2), 'F': ('fluorane', 1), 'Cl': ('chlorane', 1),
    'Br': ('bromane', 1), 'I': ('iodane', 1),
}
#: the Group 14 hydrides whose single 'yl' replaces the ending 'ane',:40414: 'silyl', not 'silanyl')
_GROUP_14_HYDRIDES = frozenset({'silane', 'germane'})
_RADICAL_SUFFIX = {0: '', 1: 'yl', 2: 'ylidene', 3: 'ylidyne', 4: 'tetrayl'}
_MULTIPLIER = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}


def _suffixed(hydride: str, suffix: str) -> str:
    """The hydride name with a suffix: its final 'e' is elided before a vowel or 'y' ('chloranyl', 'oxidanide')
    and kept before a consonant ('oxidanediide')."""
    return hydride[:-1] + suffix if suffix[:1] in tuple('aeiouy') else hydride + suffix


def _hydride_ion_name(hydride: str, bonding: int, charge: int, radicals: int) -> Optional[str]:
    """The name of a bare monoatomic ion on its parent hydride (all its hydrogen atoms taken away), or None when
    the charge and the radical electrons do not account for them.

    An anion loses ``-charge`` hydrons ('ide') and the rest of its hydrogen atoms as radicals: 'phosphanetriide',
    'azanidylidene', 'oxidanidyl',:40900). A cation adds hydrons ('ium',, and then loses
    every hydrogen atom as radicals: 'tellaniumylidyne') or loses hydride ions ('ylium',,:41553,
    and then the rest as radicals: 'silyliumylidyne', 'silanetetraylium'); the radical electrons decide which."""
    if radicals not in _RADICAL_SUFFIX:
        return None
    tail = _RADICAL_SUFFIX[radicals]
    if charge < 0:
        k = -charge
        if k not in _MULTIPLIER or k + radicals != bonding:
            return None
        return _suffixed(hydride, _MULTIPLIER[k] + 'id' + (tail or 'e'))
    if charge == 1 and radicals == bonding + 1:
        return _suffixed(hydride, 'ium' + tail)
    if charge in _MULTIPLIER and radicals == bonding - charge:
        if charge == 1 and hydride in _GROUP_14_HYDRIDES:
            return hydride[:-3] + 'yl' + 'ium' + tail
        return _suffixed(hydride, _MULTIPLIER[charge] + 'ylium' + tail)
    return None


#: Table 8.1 Names of the hydrogen atoms and ions, the Blue Book), by (mass number, charge); the
#: mass number 0 is the natural composition
HYDROGEN_NAMES: Dict[tuple, str] = {
    (0, 0): 'hydrogen', (1, 0): 'protium', (2, 0): 'deuterium', (3, 0): 'tritium',
    (0, -1): 'hydride', (1, -1): 'protide', (2, -1): 'deuteride', (3, -1): 'tritide',
    (0, 1): 'hydron', (1, 1): 'proton', (2, 1): 'deuteron', (3, 1): 'triton',
}

#: the canonical SMILES of the hydride ion, the fragment ``inorganic_anion_word`` cites as 'hydride'
HYDRIDE_ION = '[H-]'

#: the neutral one-atom components that an adduct of an ammine complex cites besides the table of
#: ``rules/adducts.SINGLE_ATOM_COMPONENT_NAMES``, by the canonical SMILES of the fragment.
#: Retained names (the Blue Book): "The common names water, ammonia,... are used in these
#: recommendations"; the table of the other components cites 'water' and 'hydrogen chloride' alike.
AMMINE_COMPONENT_NAMES: Dict[str, str] = {'N': 'ammonia'}


def metal_atom_name(frag_mol) -> Optional[str]:
    """The name of a one-atom metal component, or None.

    'nickel' for [Ni], 'iron(2+)' for [Fe+2], '(99Tc)technetium' for [99Tc]
    (the nuclide descriptor of in front of the element name). None for
    a non-metal, a metal carrying hydrogen ([NaH] is a hydride, not a metal atom)
    or a negative metal ion (an '-ide' of a metal is not built here).
    """
    if frag_mol.GetNumAtoms() != 1:
        return None
    atom = frag_mol.GetAtomWithIdx(0)
    name = METAL_ELEMENT_NAMES.get(atom.GetSymbol())
    if name is None or atom.GetTotalNumHs() != 0:
        return None
    charge = atom.GetFormalCharge()
    if charge < 0:
        return None
    if charge > 0:
        name = f"{name}({charge}+)"
    if atom.GetIsotope():
        name = f"({atom.GetIsotope()}{atom.GetSymbol()}){name}"
    return name


def admitted_metal_atom_name(mol) -> Optional[str]:
    """The name of ``mol`` when it is one atom of a metal admitted to ``ELEMENT`` in
    ``data/metal_scope.py`` ('zinc', 'zinc(2+)', '(65Zn)zinc'), else None (the metal case of
    ``lone_atom_name``)."""
    if mol is None or mol.GetNumAtoms() != 1:
        return None
    if not has_mechanism(mol.GetAtomWithIdx(0).GetSymbol(), ELEMENT):
        return None
    return metal_atom_name(mol)


def _charge_number(charge: int) -> str:
    return f"({abs(charge)}{'+' if charge > 0 else '-'})"


def lone_atom_name(mol) -> Optional[str]:
    """The name of a lone atom, ion or isotope (``lone_atom_name_and_basis``), else None."""
    return lone_atom_name_and_basis(mol)[0]


def lone_atom_name_and_basis(mol) -> tuple:
    """``(name, on_parent_hydride)``: the name of ``mol`` when it is one atom without hydrogen (an atom, a monoatomic
    ion or an isotope of either), else ``(None, False)``; ``on_parent_hydride`` is True when the name is built on the
    mononuclear parent hydride of the element ('chloranyl', 'phosphanetriide') and False for an element name or a
    hydrogen name ('neon', 'sodium(1+)', 'hydride').

    * a neutral atom of an element of ``RADICAL_PARENT_HYDRIDES`` whose radical electrons are the bonding number
      of the element: the radical of its parent hydride: 'chloranyl', 'oxidanylidene';
    * hydrogen: Table 8.1: 'hydrogen', 'deuteride', 'hydron';
    * any other neutral atom: the element name ('neon', 'thallium', 'carbon');
    * an ion of an element of ``RADICAL_PARENT_HYDRIDES``: its name on the parent hydride when the charge and the
      radical electrons account for every hydrogen atom of the hydride (``_hydride_ion_name``): 'phosphanetriide',
      'oxidanidyl', 'chloraniumylidene', 'silanetetraylium', '(36Cl)chloranide';
    * any other ion: the element name with the charge number ('sodium(1+)', 'selenium(6+)'), the notation of
      'pentaammine(ethyl)osmium(1+)', the Blue Book).

    None for an ion of the table of retained cations whose name carries an oxidation number ('iron(II)': the class
    CATION_RETAINED, and the isotope decorator for '(59Fe)iron(II)'); a bare cation word of that table ('sodium',
    'silver') is the word of a salt and names no free ion. None for a closed-shell anion of the table of retained
    anions or of a parent hydride ('chloride', 'phosphanetriide'), which the anion classes name; its isotope
    ('(36Cl)chloranide') and its radical forms ('oxidanidyl') are named here.

    An element of ``data/metal_scope.py`` (every true metal is there) takes a name only when the table admits it
    to ``ELEMENT``; the metalloids and the non-metals are not in that table. The nuclide symbol goes in
    parentheses in front of the name, the Blue Book). The name is never a PIN: PREFERRED
    IUPAC NAMES (:2056) gives the label to names "that also contain at least one carbon atom", and a lone atom is
    no salt (``rules/metal_labels.is_lone_atom_name_without_pin``). A pure function of the structure: the predicate
    and the handler of the dispatch class ``ELEMENT_ATOM`` both call it; the namer's gate reads the name back."""
    if mol is None or mol.GetNumAtoms() != 1:
        return None, False
    atom = mol.GetAtomWithIdx(0)
    if atom.GetTotalNumHs() != 0:
        return None, False
    symbol, charge, isotope = atom.GetSymbol(), atom.GetFormalCharge(), atom.GetIsotope()
    if symbol == 'H':
        return HYDROGEN_NAMES.get((isotope, charge)), False
    if symbol == 'C' and charge:
        return None, False          # 'methanide', 'methylium' (PINs,: the ion classes and the isotope decorator
    if symbol in METAL_SCOPE and not has_mechanism(symbol, ELEMENT):
        return None, False
    element = METAL_ELEMENT_NAMES.get(symbol) or NONMETAL_ELEMENT_NAMES.get(symbol)
    if element is None:
        return None, False
    plain = f"[{symbol}{'+' if charge > 0 else '-' if charge < 0 else ''}{abs(charge) if abs(charge) > 1 else ''}]"
    radicals = atom.GetNumRadicalElectrons()
    hydride = RADICAL_PARENT_HYDRIDES.get(symbol)
    if charge > 0 and '(' in (INORGANIC_CATIONS.get(plain) or ''):
        return None, False          # 'iron(II)': CATION_RETAINED (and the isotope decorator for '(59Fe)iron(II)')
    if charge < 0 and not isotope and not radicals and (plain in INORGANIC_ANIONS or hydride):
        return None, False          # 'chloride', 'phosphanetriide': the anion classes name a closed-shell anion
    if charge == 0:
        name = (_suffixed(hydride[0], _RADICAL_SUFFIX[radicals])
                if hydride and radicals == hydride[1] and radicals <= 3 else None)
    else:
        name = hydride and _hydride_ion_name(hydride[0], hydride[1], charge, radicals)
    on_hydride = name is not None
    if name is None:
        name = element + (_charge_number(charge) if charge else '')
    return (f"({isotope}{symbol}){name}" if isotope else name), on_hydride


def inorganic_anion_word(smiles: str, cation_frags) -> Optional[str]:
    """The word of the anion fragment ``smiles`` (canonical SMILES) in a salt, or None.

    The table of retained inorganic anions (``data/ion_retained_names.INORGANIC_ANIONS``: chloride,
    sulfate,...), and 'hydride' for the hydride ion when every cation of the salt is one atom of a
    metal admitted to ``CARBANION_SALTS`` ('zinc dihydride'): (the Blue Book),
    "binary names formed by citing the name of the cation followed by that of the anion". A hydride
    beside any other cation has no word here, so the salts of the other metals keep their names."""
    word = INORGANIC_ANIONS.get(smiles)
    if word is not None:
        return word
    if smiles == HYDRIDE_ION and cation_frags and all(
            frag['mol'].GetNumAtoms() == 1
            and has_mechanism(frag['mol'].GetAtomWithIdx(0).GetSymbol(), CARBANION_SALTS)
            for frag in cation_frags):
        return 'hydride'
    return None


def ammine_component_names(mol) -> Dict[str, str]:
    """The one-atom components that ``mol`` may cite beside its metals, besides the table of
    ``rules/adducts.SINGLE_ATOM_COMPONENT_NAMES``: ``AMMINE_COMPONENT_NAMES`` when ``mol`` holds a true
    metal and every true metal of it is admitted to ``AMMINE`` in ``data/metal_scope.py``, else ``{}``.

    A bare nitrogen is the ammonia of an ammine complex only beside a metal that the table admits;
    beside any other metal, and in an assembly with no metal, it stays a fragment that no producer
    names, as before. The guard reads the whole assembly: one metal outside the table keeps the rest
    of the assembly outside it."""
    from ..perception.metals import _TRUE_METAL_SYMBOLS_FOR_VETO
    metals = {a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO}
    if metals and all(has_mechanism(symbol, AMMINE) for symbol in metals):
        return dict(AMMINE_COMPONENT_NAMES)
    return {}
