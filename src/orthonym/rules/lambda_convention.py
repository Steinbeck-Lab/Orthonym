"""Shared λ-convention (lambda) infrastructure — IUPAC / Table 2.8.

The λ-convention cites the *non-standard* bonding number of a skeletal atom
immediately after its locant, e.g. a tetravalent ring sulfur at locant 4 ->
``4lambda4-thia`` (``4lambda4-thiaspiro[3.5]nonane``), or a hexavalent sulfur
mononuclear hydride -> ``hexafluoro-lambda6-sulfane`` (SF6).

This module is the single source of truth for the standard-bonding-number table
and the "is this valence non-standard?" decision. It was promoted verbatim from
``rules/spiro.py``  so that the spiro numberer, the acyclic
skeletal-replacement namer, and the mononuclear-hydride / chalcogen /
halogen namers all share one fail-closed implementation rather than each
re-deriving Table 2.8.

References:
    IUPAC 2013 Blue Book, /3 (standard / nonstandard bonding number)
    IUPAC 2013 Blue Book, (λ placement: after the locant, no hyphen)
    IUPAC 2013 Blue Book, Table 1.5 (standard bonding numbers) [BBv2:6436]
    IUPAC 2013 Blue Book, (λ in chains / replacement nomenclature)
    IUPAC 2013 Blue Book, / (mononuclear hydrides bearing λ)

Fail-closed contract: a λ is emitted ONLY for a neutral atom present in
``STANDARD_BONDING_NUMBER`` whose actual valence differs from its standard
value. Charged atoms and elements absent from the table are treated as standard
(no spurious λ — never invent a descriptor the namer cannot justify).
"""

from typing import Optional

# Greek small letter lambda (IUPAC. The Blue Book spells the
# lambda-convention descriptor with the Greek letter (533 occurrences in the
# 2013 Blue Book text), never the ASCII word "lambda" (0 occurrences inside a
# name). This is the single source of truth for the on-the-wire spelling;
# every emitter in this codebase must route through it rather than hardcoding
# either spelling.
LAMBDA = "λ"

# IUPAC standard bonding numbers for skeletal-replacement ('a') atoms. The values
# are the column headers of **Table 1.5** [BBv2:6436-6443], which
# prints each 'a' prefix under its standard bonding number; Table 2.4
# [BBv2:8236-8248] repeats them in a "Bonding Number (Valence)"
# column and agrees on every shared element. (Earlier revisions cited
# "Table 2.8" under a rule id the book does not have; Table 2.8 is the retained heterocyclic-parent name
# table [BBv2:11511] and has no bonding numbers. Corrected against the book;
# ``rules/ring_replacement.TABLE_1_5`` now carries the same numbers as data and a
# test locks the two against each other.)
#
# A skeletal heteroatom whose ACTUAL bonding number differs from its standard
# value carries the lambda convention: e.g. a tetravalent sulfur ->
# "lambda4". Elements absent from this table are treated as standard (no lambda)
# — fail-closed, never a spurious lambda.
#
# Table 1.5 has 25 rows; this table deliberately carries 23. **C and At are
# omitted on purpose.** Neither is emittable as a ring 'a' prefix (both are in
# ``ring_replacement.VB_INADMISSIBLE``) so an entry would buy nothing, while
# adding ``'C': 4`` would let every neutral carbon this helper is ever handed --
# a radical carbon has total valence 3 -- acquire a spurious ``lambda3``. Adding
# a row here is therefore NOT a documentation change; it changes emitted names.
STANDARD_BONDING_NUMBER = {
    'O': 2, 'S': 2, 'Se': 2, 'Te': 2, 'Po': 2,
    'N': 3, 'P': 3, 'As': 3, 'Sb': 3, 'Bi': 3,
    'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4,
    'B': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3,
    'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,
}


def nonstandard_bonding_number(mol, atom_idx: int) -> Optional[int]:
    """Return the lambda bonding number for a skeletal atom whose
    valence is non-standard, or None when the valence is standard.

    The bonding number is the total count of skeletal/H bonds (RDKit
    ``GetTotalValence``). It is emitted as ``lambda<n>`` immediately after the
    atom's locant in the 'a'-replacement prefix, e.g. a tetravalent ring sulfur
    at locant 4 -> ``4lambda4-thia`` spiro example
    ``4lambda4-thiaspiro[3.5]nonane``), or on a mononuclear hydride parent
    (no locant) -> ``hexafluoro-lambda6-sulfane`` (SF6, /.

    Fail-closed: only neutral atoms present in ``STANDARD_BONDING_NUMBER`` can
    carry a lambda; charged/exotic atoms return None (no spurious lambda).
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetFormalCharge() != 0:
        return None
    standard = STANDARD_BONDING_NUMBER.get(atom.GetSymbol())
    if standard is None:
        return None
    bonding_number = atom.GetTotalValence()
    return bonding_number if bonding_number != standard else None


#: Elements whose divalent or trivalent radical centre the lambda convention describes
#: (``radical_lambda_bonding_number``): those of ``STANDARD_BONDING_NUMBER`` that can lose two or three
#: hydrogens from one atom, and carbon (standard bonding number 4).
_RADICAL_LAMBDA_STANDARD = {**{sym: n for sym, n in STANDARD_BONDING_NUMBER.items() if n >= 3}, 'C': 4}


def radical_lambda_bonding_number(mol, atom_idx: int) -> Optional[int]:
    """The bonding number ``n`` of the ``λn`` that describes the radical centre at ``atom_idx``, or None.

     "The λ-convention" (the Blue Book, sentence:40562): "Divalent and trivalent radical centers in a parent
    hydride formally derived by the removal of two or three hydrogen atoms from the same skeletal atom in its
    standard valence state may be described by the λ-convention (see. Locants for the radical centers
    are followed by the symbol λ^n, where 'n' is the bonding number of the skeletal atom (see. This
    method is only for general nomenclature." (dichloro-λ2-methane, fluoro-λ1-methane, phenyl-λ1-azane).
    The sentence licenses two and three removed hydrogens only: a monovalent radical centre (one electron)
    has no lambda here and returns None, as does a charged, aromatic or non-listed atom, and an atom whose
    bonds, hydrogens and unpaired electrons do not make up its standard bonding number (the atom is then not
    "in its standard valence state" less the hydrogens removed)."""
    atom = mol.GetAtomWithIdx(atom_idx)
    n_e = atom.GetNumRadicalElectrons()
    standard = _RADICAL_LAMBDA_STANDARD.get(atom.GetSymbol())
    if (n_e not in (2, 3) or standard is None or atom.GetFormalCharge() or atom.GetIsAromatic()
            or atom.GetIsotope()):
        return None
    bonding_number = atom.GetTotalValence()
    return bonding_number if bonding_number + n_e == standard and bonding_number >= 1 else None


def format_lambda_token(locant, lam: Optional[int]) -> str:
    """Format a locant (+ optional lambda) into the citation token.

    ``format_lambda_token(4, 4)`` -> ``"4λ4"``: the lambda follows
    the locant with no separator); ``format_lambda_token(4, None)`` -> ``"4"``.
    Shared by the spiro numberer and the acyclic skeletal-replacement namer so
    the on-the-wire spelling is identical everywhere.
    """
    return f"{locant}{LAMBDA}{lam}" if lam is not None else str(locant)


# Ring halogens admitted to the lambda-convention of a fused ring system, the Blue Book,
# and,:39660: "For cyclic and acyclic parent hydrides the lambda convention is used to
# indicate nonstandard bonding numbers"). One home for the admission, read by the benzo-heterocycle
# producer (``rules/halogen_ring_fusion.py``), by the mancude-parent valence of the ring-hydrogen
# rule (``rules/bridged_fused_pin/hydro.py``) and by the ring 'a' table of the von Baeyer, spiro and
# replacement-monocycle names (``rules/ring_replacement.py``, lane L3). A halogen is admitted when a
# measurement showed its names read back; an element not listed keeps its old behaviour in all three.
FUSED_RING_LAMBDA_HALOGENS = frozenset({'I'})

# The ring elements whose ion of the simplest kind (``ring_halogen_ion``) is named from the lambda hydride of its
# ring atom: the admitted ring halogens, and silicon for the anion R4Si- of a ring ('1,1-dimethyl-1lambda5-silolan-1-ide'
# is the 'ide' of the lambda5 hydride, method (2), the Blue Book; the book prints
# 'tetramethyl-λ5-phosphanide' for the phosphorus analogue,:41106, and 'diphenyl-λ3-iodanide',:41110). Silicon is
# no halogen: the rest of the admission (the ring 'a' table, the mancude valence, the non-PIN record of a halogen
# name) is the halogens' and stays theirs.
RING_ION_LAMBDA_ELEMENTS = FUSED_RING_LAMBDA_HALOGENS | frozenset({'Si'})


def ring_lambda_valence(mol, atom_idx: int) -> Optional[int]:
    """The bonding number of an admitted ring halogen, or of a ring silicon (``RING_ION_LAMBDA_ELEMENTS``), in the lambda
    state, else None.

    The mancude parent of a ring with a lambda halogen counts the atom as a ring atom with its
    own bonding number (``1H-1lambda3-iodole``,, the Blue Book ``1H-1λ3-benziodole
    (PIN)``): a ring double bond or a hydrogen takes the valences its two ring bonds leave.
    ``None`` for every other atom (an acyclic halogen too), so a caller keeps the default valence of the periodic table
    there."""
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() not in RING_ION_LAMBDA_ELEMENTS or not atom.IsInRing():
        return None
    return nonstandard_bonding_number(mol, atom_idx)


def ring_halogen_ion(mol, atom_idx: int) -> Optional[str]:
    """``'anion'`` or ``'cation'`` for an admitted ring halogen that is an ion of the simplest
    kind, else ``None``: a ring atom of ``FUSED_RING_LAMBDA_HALOGENS`` whose bonds are all single, with
    no hydrogen, a formal charge of -1 or +1 and an even number of bonds, the two of the ring and the
    substituents of the centre (the iodanuide ``[I-]1CCCCC1``, ``F[I-]1(F)CCCCC1``; the iodonium
    ``c1ccc2c(c1)[I+]c1ccccc1-2``, ``C[I+]1(C)CCCCC1``).

    Both ions are named from the same ring hydride, the one whose atom has one hydrogen more than the ion has
    bonds, so its bonding number is the number of bonds plus one (3 for two ring bonds, 5 for four), cited
    with the lambda convention. The anion is that hydride less a hydron: the suffix 'ide'
     method (2), the Blue Book, ``diphenyl-lambda3-iodanide``:41110: "the suffix 'ide' to a parent
    hydride in which the bonding number at the anionic site is higher than the standard bonding number and is
    expressed by the lambda-convention"; '1lambda3-iodinan-1-ide' and '1,1-difluoro-1lambda5-iodinan-1-ide'
    are read by OPSIN as the ring anions). The cation is that hydride less a hydride ion: the suffix 'ylium'
     "Application of the lambda-convention with the suffix 'ylium'",:41694: "Cationic heterocycle
    having a cationic center on a heteroatom that has one more skeletal bonds than it has in the corresponding
    neutral heterocycle is named by adding the suffix 'ylium' to the name of the neutral parent hydride for which
    the lambda-convention has been used to describe a nonstandard bonding state of the heteroatom, and that
    heteroatom has at least one hydrogen atom in the neutral heterocycle on which the 'ylium' suffix can
    operate"; the book prints '1lambda3-benziodol-1-ylium (PIN)',:41708; OPSIN reads '1lambda3-iodinan-1-ylium'
    as the ring cation). ``nonstandard_bonding_number`` is ``None`` for both, as it is for every ion; see
    ``ring_halogen_ion_lambda`` for the symbol the name cites."""
    atom = mol.GetAtomWithIdx(atom_idx)
    if (atom.GetSymbol() not in RING_ION_LAMBDA_ELEMENTS or not atom.IsInRing()
            or atom.GetFormalCharge() not in (-1, 1) or atom.GetTotalNumHs()
            or atom.GetNumRadicalElectrons()):
        return None
    if any(b.GetBondTypeAsDouble() != 1.0 for b in atom.GetBonds()):
        return None
    if atom.GetDegree() not in (2, 4, 6) or (atom.GetSymbol() == 'Si' and (atom.GetDegree() != 4
                                                                           or atom.GetFormalCharge() != -1)):
        return None
    return 'anion' if atom.GetFormalCharge() < 0 else 'cation'


def ring_halogen_ion_lambda(mol, atom_idx: int) -> Optional[int]:
    """The bonding number the lambda symbol cites in the name of a ring halogen ion
    (``ring_halogen_ion``): that of the hydride the ion is named from, its bonds and one hydrogen
    (3 for the two ring bonds of ``[I-]1CCCCC1`` and of ``[I+]1CCCCC1`` alike)."""
    if ring_halogen_ion(mol, atom_idx) is None:
        return None
    return mol.GetAtomWithIdx(atom_idx).GetDegree() + 1


def fused_lambda_entries(mol, ring_atoms, atom_to_locant):
    """``[(numeric locant, bonding number),...]`` ascending, for the nonstandard-bonding atoms of
    a fused ring system under the numbering ``atom_to_locant``; ``None`` when the name cannot cite
    one: a lambda on skeletal carbon, or on an atom with no numeric locant (a fusion atom, '4a').
     (the Blue Book): the symbol "follows the locant of the atom with the
    nonstandard bonding number in accordance with the numbering of the fused system"; a name
    without it would denote a different molecule, so the caller declines."""
    entries = []
    for idx in sorted(ring_atoms):
        lam = nonstandard_bonding_number(mol, idx)
        if lam is None:
            continue
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C':
            return None
        loc = atom_to_locant.get(idx)
        if loc is None or not str(loc).isdigit():
            return None
        entries.append((int(loc), lam))
    return sorted(entries)


def fused_lambda_prefix(mol, ring_atoms, atom_to_locant) -> Optional[str]:
    """The lambda tokens cited at the beginning of a fused name, ``'1λ4-'`` or ``'1λ4,3λ4-'``;
    ``''`` when no ring atom has a nonstandard bonding number; ``None`` when the name must be
    declined (see ``fused_lambda_entries``).: the tokens "are cited at the beginning
    of the name of the fused ring system", after any indicated hydrogen (``1H-1λ4-thiophene``,
    ."""
    entries = fused_lambda_entries(mol, ring_atoms, atom_to_locant)
    if entries is None:
        return None
    if not entries:
        return ''
    return ','.join(format_lambda_token(loc, lam) for loc, lam in entries) + '-'

