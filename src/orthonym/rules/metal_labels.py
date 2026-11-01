"""Which names of metal compounds the Blue Book gives no PIN, read for the tier row of the namer.

Label only: each predicate answers for the name a producer built, and a name it holds is shipped
unchanged at a label below the PIN (``Orthonym._tier_row_and_pin_form``). Kept out of ``namer.py``
so that file stays within its size budget; the namer imports these names back."""
from typing import Optional

from rdkit import Chem

from ..data.metal_scope import BONDED_NAMES, ELEMENT, has_mechanism


def is_carbanion_salt_of_group_3_12_metal(smiles: str, name: Optional[str]) -> bool:
    """True iff ``name`` is the compositional name of a salt of a carbanion with a Group 3-12
    metal cation ('zinc bis(ethan-1-ide)', 'copper(II) benzenide'), whose names the Blue Book
    gives without PIN status.

     INTRODUCTION (the Blue Book): "neither preferred IUPAC names or preselected
    names (see for organometallic compounds involving the transition elements (including
    the Group 3 elements) and Groups 1 and 2 elements, except for 'ocene' compounds, are noted";
    the heading of (:39751), "ORGANOMETALLIC COMPOUNDS INVOLVING THE ELEMENTS IN GROUPS 3
    THROUGH 12", puts Group 12 with them; (:40173) calls 'sodium ethenide' "also
    acceptable; it is a compositional name analogous to sodium chloride". The structure has no
    metal-carbon bond (that case is ``namer._is_p69_organometallic_without_pin``) and a carbanion
    carbon beside the metal (``perception.metals.holds_carbanion_salt_of_group_3_12_metal``; a
    cyanide anion, the anion of an acid, is none). A name is outside the rule when

    * it is a name of the exact-structure list (``namer._coordination_list_name``): the label of
      its path (user decision 2026-09-30);
    * it is a retained metallocene name or an ocene name (``RETAINED_METALLOCENES``,
      ``namer._OCENE_STEMS``): "except for 'ocene' compounds" and "Ocenes"
      (:40129), whose examples carry "(PIN)" (:40133);
    * it is an eta-, kappa- or mu-hapto additive name of the organometallic assembler
      (``namer._ORGANOMETALLIC_ADDITIVE_PIN_RE``): its label is not this rule's to change.

    Label only: the name is unchanged."""
    if not smiles or not name:
        return False
    try:
        from ..namer import _OCENE_STEMS, _ORGANOMETALLIC_ADDITIVE_PIN_RE, _coordination_list_name
        from ..perception.metals import holds_carbanion_salt_of_group_3_12_metal
        mol = Chem.MolFromSmiles(smiles)
        if not holds_carbanion_salt_of_group_3_12_metal(mol):
            return False
        if _coordination_list_name(smiles, name):
            return False
        from ..data.organometallics import RETAINED_METALLOCENES
        if name in RETAINED_METALLOCENES.values() or any(stem in name for stem in _OCENE_STEMS):
            return False
        return _ORGANOMETALLIC_ADDITIVE_PIN_RE.match(name) is None
    except Exception:
        return False


def is_beta_diketonate_salt_of_group_3_12_metal(smiles: str, name: Optional[str]) -> bool:
    """True iff ``name`` is the name of an ionic drawing of a Group 3-12 metal with the anion of a beta-diketone
    ('copper bis(4-oxopent-2-en-2-olate)'): the compound is a coordination compound, the Blue Book) and the
    Blue Book gives its name no PIN,:39735). The salts of Group 1 and 2 metals and the carve-out metals keep the
    label of a salt of a hydroxy compound ('sodium propan-2-olate',,:28178): this predicate reads the Group 3-12
    metals only (``perception.metals.GROUP_3_12_METAL_SYMBOLS``). The maintainers' ruling of 2026-11-01. Label only."""
    if not smiles or not name:
        return False
    try:
        from ..perception.metals import holds_beta_diketonate_salt_of_group_3_12_metal
        return holds_beta_diketonate_salt_of_group_3_12_metal(Chem.MolFromSmiles(smiles))
    except Exception:
        return False


def is_bonded_metal_name_without_pin(smiles: str, name: Optional[str]) -> bool:
    """True iff ``name`` names a drawing in which a metal admitted to ``BONDED_NAMES``
    (``data/metal_scope.py``) is bonded to another atom or carries hydrogen ('oxozinc',
    'dihydridozinc', 'ethyl(hydroxy)zinc', 'trimethylzinc(1-)'), whose names the Blue Book gives
    without PIN status. The bond to carbon is not required: ``namer._is_p69_organometallic_without_pin``
    holds the names of the compounds with one, and this predicate holds the others.

    It covers every bonded zinc drawing, whatever the producer of the name: the structure decides, not
    the route. The ligand producer of ``BONDED_NAMES`` (``rules.organometallics``), the organyl branches
    of the assembler ('diethylzinc'), the salt, adduct and macrocycle producers reached through a drawing
    with a bond from the metal, and the name that ``assembly.metal_offer`` gives the rewritten drawing of
    ``Cl[Zn]Cl`` ('zinc dichloride') all name a drawing this predicate holds for.

     (the Blue Book), under the heading "ORGANOMETALLIC COMPOUNDS INVOLVING THE
    ELEMENTS IN GROUPS 3 THROUGH 12": "Coordination nomenclature is the primary nomenclature
    method used to name organometallic compounds containing elements of Groups 3 through 12", so a
    name of the ligands of the metal is no preferred name; PREFERRED IUPAC NAMES (:2056)
    gives the label "PIN" to "the names of compounds whose parent hydride contains at least one
    of the following elements: B, Si, Ge, Sn, Pb, N, P, As, Sb, Bi, O, S, Se, Te, Po, F, Cl, Br, I,
    At, and that also contain at least one carbon atom in their structure": zinc is none of them,
    and 'oxozinc' has no carbon. A name of the exact-structure list (``namer._coordination_list_name``)
    keeps the label of its path (user decision 2026-09-30).

    Label only: the name is unchanged."""
    if not smiles or not name:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not any(
                has_mechanism(a.GetSymbol(), BONDED_NAMES) and (a.GetDegree() > 0 or a.GetTotalNumHs() > 0)
                for a in mol.GetAtoms()):
            return False
        from ..namer import _coordination_list_name
        return not _coordination_list_name(smiles, name)
    except Exception:
        return False


def is_metal_atom_or_ion_name_without_pin(smiles: str, name: Optional[str]) -> bool:
    """True iff ``name`` names a drawing that is one atom of a metal admitted to ``ELEMENT``
    (``data/metal_scope.py``): the atom, an ion or an isotope ('zinc', 'zinc(2+)', '(65Zn)zinc'),
    whose names the Blue Book gives without PIN status.

     PREFERRED IUPAC NAMES (the Blue Book): "the label 'PIN' is added to the names of
    compounds whose parent hydride contains at least one of the following elements: B, Si, Ge, Sn,
    Pb, N, P, As, Sb, Bi, O, S, Se, Te, Po, F, Cl, Br, I, At, and that also contain at least one
    carbon atom in their structure": zinc is none of those elements and the drawing has no carbon;
    the project keeps the path label of a carbon-free SALT (user decision 2026-09-30), and a lone
    atom or ion is no salt. The structure alone decides: the name is the one the producer of
    ``rules/metal_atoms.py`` built.

    Label only: the name is unchanged."""
    if not smiles or not name:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return (mol is not None and mol.GetNumAtoms() == 1
                and has_mechanism(mol.GetAtomWithIdx(0).GetSymbol(), ELEMENT))
    except Exception:
        return False


def is_macrocycle_complex_name_without_pin(smiles: str, name: Optional[str]) -> bool:
    """True iff ``name`` names an assembly that holds a porphyrin ion of a metal admitted to
    ``MACROCYCLE`` (``data/metal_scope.py``): the salt or the adduct of the metal with the porphyrin
    anion ('zinc 5,10,15,20-tetraphenylporphyrin-21,23-diide'), whose names the Blue Book gives
    without PIN status, whichever parent the name of the ligand is built on.

     (the Blue Book), under the heading "ORGANOMETALLIC COMPOUNDS INVOLVING THE
    ELEMENTS IN GROUPS 3 THROUGH 12": "Coordination nomenclature is the primary nomenclature
    method used to name organometallic compounds containing elements of Groups 3 through 12", so a
    name that cites the ligand and the metal as the words of a salt is no preferred name;
    INTRODUCTION (:50943): "Preferred IUPAC names (PINs) are not identified for the compounds in this
    Chapter", the chapter of the porphyrin parent. A name of the exact-structure list
    (``namer._coordination_list_name``) keeps the label of its path (user decision 2026-09-30).

    The structure decides (``metal_macrocycles.macrocycle_ion_fragments``), not the producer of the
    ligand name: that producer names a fused-ring substituent of the ion in the best-effort call only,
    and the label is read after the call. Label only: the name is unchanged."""
    if not smiles or not name:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        from .metal_macrocycles import macrocycle_ion_fragments
        if not macrocycle_ion_fragments(mol):
            return False
        from ..namer import _coordination_list_name
        return not _coordination_list_name(smiles, name)
    except Exception:
        return False


def is_metal_name_without_pin(smiles: str, name: Optional[str]) -> bool:
    """True iff ``name`` is a name of a compound of a Group 3-12 metal that the Blue Book gives no
    PIN: the compositional name of a salt of a carbanion (``is_carbanion_salt_of_group_3_12_metal``),
    a name of a bonded drawing that lists its ligands (``is_bonded_metal_name_without_pin``), the
    name of an atom, an ion or an isotope (``is_metal_atom_or_ion_name_without_pin``) or the name of
    a salt of a metal with a macrocyclic ligand (``is_macrocycle_complex_name_without_pin``).
    The tier row of ``Orthonym._tier_row_and_pin_form`` reads this one predicate. Label only."""
    return (is_carbanion_salt_of_group_3_12_metal(smiles, name)
            or is_beta_diketonate_salt_of_group_3_12_metal(smiles, name)
            or is_bonded_metal_name_without_pin(smiles, name)
            or is_metal_atom_or_ion_name_without_pin(smiles, name)
            or is_macrocycle_complex_name_without_pin(smiles, name))
