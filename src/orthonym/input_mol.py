"""An RDKit molecule as the input of a public naming call.

The public functions (``name_compound``, ``name_with_tree``, ``classify_limit`` and
the ``Orthonym`` methods ``name``, ``name_tiered``, ``name_with_tree``,
``name_with_confidence``) take a SMILES string or an RDKit ``Chem.Mol``. A string is
named as it always was. A Mol is named through RDKit's SMILES of it: each public
call turns a Mol into that SMILES with:func:`smiles_for` and names the string, so
a Mol gets exactly the result its SMILES gets.

:func:`smiles_for` writes the SMILES, ``Chem.MolToSmiles`` of a copy of the Mol
(the caller's object is never changed), and first proves that the SMILES holds the
molecule RDKit reads from the Mol. Where the proof fails it raises
:class:`InvalidInputError`, a ValueError, rather than let the call name another
molecule or another stereoisomer:

(a) RDKit reads the SMILES back (``Chem.MolFromSmiles``).
(b) RDKit's standard InChI of the Mol as given (without its conformers, which the
    InChI writer would read instead of the stereo tags, and before any
    sanitization, which can change what the InChI writer reads) equals the InChI of
    that reading. RDKit makes no InChI for some molecules (a dummy atom, for one);
    then the other checks decide.
(c) Atom for atom, through the order in which the writer wrote the atoms
    (``_smilesAtomOutputOrder``), the reading has the atoms and bonds of the Mol
    (sanitized on a copy) and every tetrahedral centre and double bond of the Mol
    that can be stereogenic has the same configuration in the reading, or none in
    both (a tag on an atom that cannot be a stereocentre, as RDKit's PDB reader
    sets, is dropped by the SMILES writer and not compared). This sees stereo
    that InChI does not (an amidine C=N, for one) and stereo a SMILES cannot hold
    (a double bond left open between two set ones gets a configuration in any
    SMILES). In a symmetric molecule the writer can write the configurations of a
    symmetric equivalent, so other matchings of the atoms are tried as well.
(d) Every lone-pair stereocentre of the SMILES has the same configuration in
    RDKit's reading as in the standard reading of SMILES, which the naming of a
    string follows (:func:`_lone_pair_centres_read_alike`).

A Mol with stereo this check cannot compare is declined before the SMILES is
written: an AND or OR stereo group (a SMILES holds no stereo group), an atropisomer tag (a SMILES holds no axis configuration) or a ``CHI_OTHER``
chiral tag (RDKit's SMILES writer can crash the process on one). Square-planar,
trigonal-bipyramidal and octahedral tags pass: a SMILES holds them and no name states
them, so the Mol gets the result of its SMILES.

Only a caller's own call (outside any naming) hands ``None`` or another type to
:func:`smiles_for`, where it raises; a non-str the engine itself passed to one of
its entries keeps the path it had before Mol input existed (:func:`handles`).
"""
from __future__ import annotations

from typing import Callable, Optional

from rdkit import Chem


class InvalidInputError(ValueError):
    """``None``, or a Mol whose SMILES does not hold the molecule RDKit reads from
    it, given to a public call; a ValueError, as for a SMILES RDKit cannot read."""


def handles(arg) -> bool:
    """Whether a public entry turns ``arg`` (not a ``str``) into a SMILES with
    :func:`smiles_for`: a Mol always; any other type when the call is a caller's own
    (outside any naming), where it raises."""
    if isinstance(arg, Chem.Mol):
        return True
    from .assembly.fragment_naming import _session_depth, name_scope_depth
    return name_scope_depth() == 0 and _session_depth() == 0


def smiles_for(arg) -> str:
    """The SMILES a public call names for its non-str argument ``arg``: RDKit's
    canonical SMILES of the Mol, once the proof in the module docstring holds.

    An empty Mol gives ``""``, named as the empty string is. Raises
    :class:`InvalidInputError` for ``None`` and for a Mol that the proof declines or
    that RDKit cannot sanitize or write; TypeError for any other type."""
    if arg is None:
        raise InvalidInputError("Invalid SMILES: None")
    if not isinstance(arg, Chem.Mol):
        raise TypeError("expected a SMILES string or an RDKit Chem.Mol, got "
                        f"{type(arg).__name__}")
    given = Chem.Mol(arg)
    if given.GetNumAtoms() == 0:
        return ""
    given.RemoveAllConformers()
    _decline_stereo_it_cannot_compare(given)
    try:
        return _proven_smiles(given)
    except InvalidInputError:
        raise
    except Exception as exc:  # an RDKit error the Mol causes
        raise InvalidInputError("Invalid molecule: " + " ".join(str(exc).split())) from None


_TETRAHEDRAL = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
_ST = Chem.BondStereo
_DOUBLE_BOND_TAGS = (_ST.STEREOE, _ST.STEREOZ, _ST.STEREOCIS, _ST.STEREOTRANS)
_TRANS_TAGS = (_ST.STEREOE, _ST.STEREOTRANS)   # E read as trans on the stereo atoms
_ATROPISOMER_TAGS = (_ST.STEREOATROPCW, _ST.STEREOATROPCCW)
#: Chiral tags a Mol may carry: tetrahedral ones, which the check compares, and the
#: square-planar, trigonal-bipyramidal and octahedral ones, which no name states (the
#: engine writes no configuration for them, as for the same SMILES string). Any other
#: tag (CHI_OTHER, on which RDKit's SMILES writer can crash the process) is declined.
_COMPARABLE_OR_UNNAMED = (Chem.ChiralType.CHI_UNSPECIFIED, *_TETRAHEDRAL,
                          Chem.ChiralType.CHI_SQUAREPLANAR,
                          Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
                          Chem.ChiralType.CHI_OCTAHEDRAL)


def _decline_stereo_it_cannot_compare(mol: Chem.Mol) -> None:
    for group in mol.GetStereoGroups():
        if group.GetGroupType() != Chem.StereoGroupType.STEREO_ABSOLUTE:
            raise InvalidInputError(
                "Invalid molecule: it has an AND (racemic) or OR (unknown enantiomer) "
                "stereo group, which a SMILES string does not hold; it would be named as "
                "one enantiomer. Clear the configuration of the group's atoms to name "
                "the molecule without it.")
    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag not in _COMPARABLE_OR_UNNAMED:
            raise InvalidInputError(
                f"Invalid molecule: atom {atom.GetIdx()} has a chiral tag that is not "
                f"tetrahedral ({tag}; CHI_OTHER, for one), which this check cannot compare.")
    for bond in mol.GetBonds():
        if bond.GetStereo() in _ATROPISOMER_TAGS:
            raise InvalidInputError(
                f"Invalid molecule: bond {bond.GetIdx()} has an atropisomer tag, and a "
                "SMILES string cannot hold that configuration.")


def _inchi(mol: Chem.Mol) -> str:
    """RDKit's standard InChI of a copy of ``mol`` ("" when RDKit makes none)."""
    from rdkit.Chem import inchi
    return inchi.MolToInchi(Chem.Mol(mol)) or ""


def _proven_smiles(given: Chem.Mol) -> str:
    """:func:`smiles_for` for ``given``, a conformer-free copy with atoms."""
    reference = _inchi(given)                 # before any sanitization
    sanitized = Chem.Mol(given)
    try:
        Chem.SanitizeMol(sanitized)
    except Exception as exc:  # RDKit's sanitization errors
        raise InvalidInputError(f"Invalid molecule: {exc}") from None
    written = Chem.Mol(given)
    key = Chem.MolToSmiles(written)
    parsed = Chem.MolFromSmiles(key)
    if parsed is None:                                                       # (a)
        raise InvalidInputError(
            f"Invalid molecule: RDKit cannot read back its SMILES of it ({key}).")
    if reference and _inchi(parsed) != reference:                            # (b)
        raise InvalidInputError(
            f"Invalid molecule: RDKit's SMILES of it ({key}) has another InChI than "
            "RDKit reads from the Mol, so it does not hold the same molecule or stereo.")
    params = Chem.SmilesParserParams()
    params.removeHs = False                   # the atoms the SMILES writes, one for one
    explicit = Chem.MolFromSmiles(key, params)
    order = _output_order(written)
    if (explicit is None or order is None or len(order) != given.GetNumAtoms()
            or explicit.GetNumAtoms() != given.GetNumAtoms()):
        raise InvalidInputError(
            f"Invalid molecule: RDKit's SMILES of it ({key}) cannot be matched to it "
            "atom for atom.")
    matching = [0] * len(order)
    for position, index in enumerate(order):
        matching[index] = position
    if not _same_constitution(sanitized, explicit, matching):                # (c)
        raise InvalidInputError(
            f"Invalid molecule: RDKit reads its SMILES of it ({key}) back as another "
            "molecule (other hydrogen counts, charges, isotopes, radicals or bonds on "
            "its atoms; a query Mol made by Chem.MolFromSmarts, for one).")
    if not _same_stereo_some_matching(given, sanitized, explicit, matching, key, order):
        raise InvalidInputError(
            f"Invalid molecule: its SMILES ({key}) does not hold its stereo: a "
            "stereocentre or double bond is set in one and open or set the other way "
            "in the other (a double bond left open between two set ones, for one, "
            "gets a configuration in any SMILES).")
    if not _lone_pair_centres_read_alike(key):                               # (d)
        raise InvalidInputError(
            f"Invalid molecule: its SMILES ({key}) has a lone-pair stereocentre whose "
            "configuration the check cannot confirm: RDKit reads it otherwise than the "
            "standard reading of SMILES, which the naming of a SMILES string follows, "
            "or the standard reading gives it no label (an As(III) centre, for one).")
    return key


def _lone_pair_centres_read_alike(key: str) -> bool:
    """Whether every lone-pair stereocentre of ``key`` (P(III), S(IV), an aziridine N
    ...) has the same CIP label in RDKit's reading of ``key`` as in the standard
    reading, CDK's labels of ``key`` as written. The naming of a string names the
    standard reading (``namer._lone_pair_standard_spelling``), and RDKit reads such a
    centre otherwise when it is written first or at some ring-closure positions, so
    the string would be named as another stereoisomer than the Mol. True without such
    a centre; False when a label is missing or cannot be made (fail closed)."""
    from . import namer
    if not namer._has_lone_pair_stereocentre(key):
        return True
    try:
        from rdkit.Chem import rdCIPLabeler

        from .perception.centres_bridge import centres_label_batch
        mol = namer._lp_heavy_mol(key)
        batch = centres_label_batch([key])
        if mol is None or batch is None or key not in batch:
            return False
        centres = [a.GetIdx() for a in mol.GetAtoms()
                   if namer._lone_pair_compared_atom(a) and a.GetTotalDegree() == 3]
        rdCIPLabeler.AssignCIPLabels(mol, atomsToLabel=centres, bondsToLabel=[],
                                     maxRecursiveIterations=namer._LP_CIP_MAX_ITERATIONS)
        for index in centres:
            atom = mol.GetAtomWithIdx(index)
            standard = batch[key].get(atom.GetIntProp("_lp_pos"))
            if not atom.HasProp("_CIPCode") or atom.GetProp("_CIPCode") != standard:
                return False
        return True
    except Exception:  # RDKit's CIP labeller past its bound, or any other error
        return False


def _output_order(written: Chem.Mol) -> Optional[list]:
    """The atom index at each position of RDKit's last SMILES of ``written``, None
    without one. Only that property is read: a Mol keeps the caller's own properties
    (a mol block's title, an SDF record's data fields), which need not be."""
    if not written.HasProp("_smilesAtomOutputOrder"):
        return None
    raw = written.GetProp("_smilesAtomOutputOrder")
    try:
        return [int(i) for i in raw.strip().strip("[]").split(",") if i.strip()]
    except ValueError:
        return None


#: The non-metals; on any other element (a metal ion) the radical-electron count is
#: a reader's convention (RDKit's SMILES reader gives '[Fe+3]' radical electrons, its
#: Maestro and TPL readers give the same ion none), so it is not compared there; the
#: InChI check has already compared the two.
_NON_METALS = frozenset((1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 33, 34, 35, 36,
                         52, 53, 54, 85, 86))


def _atom_invariant(atom: Chem.Atom) -> tuple:
    radicals = atom.GetNumRadicalElectrons() if atom.GetAtomicNum() in _NON_METALS else 0
    return (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(),
            atom.GetTotalNumHs(), radicals)


def _same_constitution(mol: Chem.Mol, reading: Chem.Mol, matching) -> bool:
    """Whether ``matching`` (``matching[i]``: the atom of ``reading`` for atom ``i``
    of ``mol``) maps the atoms and bonds of ``mol`` one for one onto those of
    ``reading``, each atom onto one with the same element, isotope, charge, hydrogen
    count and radical electrons, each bond onto one of the same type."""
    if len(set(matching)) != mol.GetNumAtoms() or mol.GetNumBonds() != reading.GetNumBonds():
        return False
    for atom in mol.GetAtoms():
        if _atom_invariant(atom) != _atom_invariant(
                reading.GetAtomWithIdx(matching[atom.GetIdx()])):
            return False
    for bond in mol.GetBonds():
        twin = reading.GetBondBetweenAtoms(matching[bond.GetBeginAtomIdx()],
                                           matching[bond.GetEndAtomIdx()])
        if twin is None or twin.GetBondType() != bond.GetBondType():
            return False
    return True


def _is_odd_permutation(sequence: list) -> bool:
    """Whether sorting ``sequence`` (distinct values) takes an odd number of swaps."""
    inversions = sum(1 for i, a in enumerate(sequence) for b in sequence[i + 1:] if a > b)
    return inversions % 2 == 1


def _stereo_atoms(bond: Chem.Bond) -> Optional[tuple]:
    """``bond``'s two stereo atoms, a neighbour of its begin atom and one of its end
    atom, neither the bond's other atom; None when it has no such pair (RDKit's
    editing functions can leave a tag without them, which fixes no configuration)."""
    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    atoms = list(bond.GetStereoAtoms())
    if (len(atoms) == 2
            and atoms[0] in _neighbours_but(begin, end)
            and atoms[1] in _neighbours_but(end, begin)):
        return atoms[0], atoms[1]
    return None


def _neighbours_but(atom: Chem.Atom, other: Chem.Atom) -> set:
    return {n.GetIdx() for n in atom.GetNeighbors()} - {other.GetIdx()}


#: The reading of a set double bond whose end has more than two other neighbours.
_UNREADABLE = "unreadable"


def _stereo_units(mol: Chem.Mol, to_reading: Callable[[int], int],
                  possible: Optional[tuple] = None) -> dict:
    """The set tetrahedral centres and double bonds of ``mol``, with indices mapped
    by ``to_reading``, each read so that the reading does not depend on the order of
    atoms or bonds: ``{("centre", atom): tag with the neighbours in ascending order,
    ("bond", (atom, atom)): whether the lowest neighbours of its two ends are
    trans}``. A double bond's tag reads relative to its stereo atoms, E as trans and
    Z as cis, as RDKit's SMILES and InChI writers read it. With ``possible`` (from
    :func:`_possible_stereo`), a tag on an atom or bond that cannot be stereogenic is
    left out, as RDKit's SMILES writer leaves it out."""
    units = {}
    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag not in _TETRAHEDRAL:
            continue
        if possible is not None and atom.GetIdx() not in possible[0]:
            continue
        neighbours = [to_reading(bond.GetOtherAtomIdx(atom.GetIdx()))
                      for bond in atom.GetBonds()]
        if _is_odd_permutation(neighbours):
            tag = _TETRAHEDRAL[1] if tag == _TETRAHEDRAL[0] else _TETRAHEDRAL[0]
        units[("centre", to_reading(atom.GetIdx()))] = tag
    for bond in mol.GetBonds():
        if (bond.GetBondType() != Chem.BondType.DOUBLE
                or bond.GetStereo() not in _DOUBLE_BOND_TAGS):
            continue
        if possible is not None and bond.GetIdx() not in possible[1]:
            continue
        atoms = _stereo_atoms(bond)
        if atoms is None:
            continue
        trans = bond.GetStereo() in _TRANS_TAGS
        ends = ((bond.GetBeginAtom(), bond.GetEndAtom(), atoms[0]),
                (bond.GetEndAtom(), bond.GetBeginAtom(), atoms[1]))
        for end, other, stereo_atom in ends:
            neighbours = sorted(to_reading(i) for i in _neighbours_but(end, other))
            if len(neighbours) > 2:
                trans = _UNREADABLE
                break
            if to_reading(stereo_atom) != neighbours[0]:
                trans = not trans
        pair = tuple(sorted((to_reading(bond.GetBeginAtomIdx()),
                             to_reading(bond.GetEndAtomIdx()))))
        units[("bond", pair)] = trans
    return units


def _possible_stereo(sanitized: Chem.Mol) -> tuple:
    """The atoms that can be tetrahedral stereocentres and the double bonds that can
    be stereogenic in ``sanitized`` (RDKit's ``FindPotentialStereo`` on a copy
    without stereo tags: which atoms and bonds can be stereogenic depends on the
    structure, not on the tags).
    Some readers tag atoms that cannot be (the PDB reader tags every atom with four
    neighbours from its 3D coordinates, a CH2 or an isopropyl CH as well); RDKit's
    SMILES writer drops those tags, and so does this comparison."""
    bare = Chem.Mol(sanitized)       # the structure only: RDKit's search raises on a
    for atom in bare.GetAtoms():     # double-bond tag without its stereo atoms
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in bare.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    atoms, bonds = set(), set()
    for info in Chem.FindPotentialStereo(bare, cleanIt=False, flagPossible=True):
        if info.type == Chem.StereoType.Atom_Tetrahedral:
            atoms.add(info.centeredOn)
        elif info.type == Chem.StereoType.Bond_Double:
            bonds.add(info.centeredOn)
    return atoms, bonds


def _tags_legacy_perception_drops(sanitized: Chem.Mol) -> set:
    """The atoms of ``sanitized`` whose tetrahedral tag RDKit's legacy stereo
    perception removes as no stereocentre (the CH bridgehead of a quinuclidine, for
    one, which the newer search still counts as possible). RDKit's SMILES writer
    follows the legacy perception when it is on (the default), so it drops these
    tags too. Empty when the legacy perception is off or raises (fail closed)."""
    if not Chem.GetUseLegacyStereoPerception():
        return set()
    tagged = {a.GetIdx() for a in sanitized.GetAtoms() if a.GetChiralTag() in _TETRAHEDRAL}
    if not tagged:
        return set()
    try:
        copy = Chem.Mol(sanitized)
        Chem.AssignStereochemistry(copy, cleanIt=True, force=True)
    except Exception:  # RDKit's perception on an unusual Mol
        return set()
    return {i for i in tagged if copy.GetAtomWithIdx(i).GetChiralTag() not in _TETRAHEDRAL}


#: The most atom matchings tried in a symmetric molecule (see
#::func:`_same_stereo_some_matching`).
_MAX_MATCHINGS = 10000


def _same_stereo_some_matching(given: Chem.Mol, sanitized: Chem.Mol, reading: Chem.Mol,
                               matching: list, key: str, order: list) -> bool:
    """Whether some matching of the atoms of ``given`` onto those of ``reading``
    (RDKit's reading of ``key`` with its hydrogen atoms) gives every set tetrahedral
    centre and double bond of ``given`` the configuration it has in ``reading``, and
    ``reading`` no other. Tried in turn: ``matching``, the order in which the writer
    wrote the atoms; the order in which it writes those of ``reading``; then up to
    :data:`_MAX_MATCHINGS` matchings of the whole molecule (``sanitized``, the
    sanitized ``given``) onto ``reading``. The writer's own order is not always
    enough: in a symmetric molecule it can write the configurations of a symmetric
    equivalent (cis- or trans-1,4-dimethylcyclohexane, read the other way round)."""
    target = _stereo_units(reading, lambda index: index)
    if _UNREADABLE in target.values():
        return False

    possible = _possible_stereo(sanitized)
    dropped = _tags_legacy_perception_drops(sanitized)

    def same(candidate) -> bool:
        units = _stereo_units(given, candidate.__getitem__, possible)
        for index in dropped:   # a tag no perception keeps, absent from the reading
            unit = ("centre", candidate[index])
            if unit in units and unit not in target:
                del units[unit]
        return units == target and _same_constitution(sanitized, reading, candidate)

    if same(matching):
        return True
    rewritten = Chem.Mol(reading)
    if Chem.MolToSmiles(rewritten) == key:
        reading_order = _output_order(rewritten)
        if reading_order is not None and len(reading_order) == len(order):
            candidate = [0] * len(order)
            for index, reading_index in zip(order, reading_order):
                candidate[index] = reading_index
            if same(candidate):
                return True
    for candidate in reading.GetSubstructMatches(sanitized, uniquify=False,
                                                 useChirality=False,
                                                 maxMatches=_MAX_MATCHINGS):
        if same(list(candidate)):
            return True
    return False
