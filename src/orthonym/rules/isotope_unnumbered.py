"""The decorator on a skeleton whose retained names are spelled systematically where they number no label.

 'Location of nuclides on positions not normally denoted by locants' (the Blue Book),:
"When the nuclide is located at a position in a retained name that is not numbered a systematic name that
identifies separately the relevant atom is used for the IUPAC preferred name." ('benzene(13C)carbonitrile
(PIN)',:44253; the retained 'benzonitrile' numbers no position of its nitrile carbon.)

The decorator of ``rules.isotopes`` names the isotope-STRIPPED molecule, so a label on the carbonyl carbon of a
substituted 'benzaldehyde', or on a methyl of 'L-valinate', has no place in the name it gets. Inside
``assembly.locant_omission.systematic_names_for_unnumbered_labels`` the producers of those retained names spell
the systematic name where a label sits on a position they do not number (``rules.benzene_unnumbered``,
``rules.esters.get_ring_acid_name``, ``rules.esters._name_amino_acid_ester``), and the strategies of the
decorator run again on that skeleton (:func:`decorate_with_systematic_unnumbered_parents`), the scope composer
included at the tier that runs it.
"""
from typing import Optional

from rdkit import Chem

from . import isotopes as _iso

#: an amino acid ester: the acid part of the esters whose retained names number one chain of the side chain
#: ('valinate', 'leucinate'; ``rules.esters._unnumbered_side_chain_atoms``)
_AMINO_ACID_ESTER = Chem.MolFromSmarts("[NX3][CX4][CX3](=O)O[#6]")


def label_may_sit_on_an_unnumbered_position(original: Chem.Mol) -> bool:
    """False when no producer of a retained name can find a label of ``original`` on a position its name does not
    number, so the scope has nothing to spell: a cheap superset of the places the producers ask about
    (``rules.benzene_unnumbered.label_on_unnumbered_group``, ``rules.esters._carbonyl_of_ring_acid_is_labelled`` and
    ``_unnumbered_side_chain_atoms``, ``prefix_derivation.retained_benzyl``). A labelled atom, or the heavy atom
    of a labelled hydrogen, that is (i) an acyclic carbon bonded to an aromatic ring carbon (the carbon of a group
    of a benzene, a benzylic methylene), (ii) a terminal oxygen on such a carbon, or (iii) an acyclic carbon of an
    amino acid ester."""
    labelled = set()
    for atom in original.GetAtoms():
        if atom.GetAtomicNum() == 1:
            if atom.GetIsotope() and atom.GetDegree() == 1:
                labelled.add(atom.GetNeighbors()[0].GetIdx())
        elif atom.GetIsotope():
            labelled.add(atom.GetIdx())

    def on_ring_group(atom) -> bool:
        return (atom.GetSymbol() == "C" and not atom.IsInRing()
                and any(nb.GetSymbol() == "C" and nb.GetIsAromatic() for nb in atom.GetNeighbors()))

    ester = original.HasSubstructMatch(_AMINO_ACID_ESTER)
    for idx in labelled:
        atom = original.GetAtomWithIdx(idx)
        heavy = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1]
        if (on_ring_group(atom) or (atom.GetSymbol() == "O" and len(heavy) == 1 and on_ring_group(heavy[0]))
                or (ester and atom.GetSymbol() == "C" and not atom.IsInRing())):
            return True
    return False


def _name_here(naming, kind: str, stripped_smiles: str) -> str:
    """The skeleton ``stripped_smiles`` as ``naming`` names it in the scope of that is open now, once per
    decorator pass and ``kind`` of namer ('caller': the caller's own, 'systematic': its flagged systematic twin)."""
    from ..assembly.locant_omission import scoped_skeleton_memo
    memo = scoped_skeleton_memo()
    if memo is not None and (kind, stripped_smiles) in memo:
        return memo[(kind, stripped_smiles)]
    name = _iso._name_skeleton(naming, stripped_smiles)
    if memo is not None:
        memo[(kind, stripped_smiles)] = name
    return name


def _scoped_name(naming, kind: str, stripped_smiles: str) -> str:
    """:func:`_name_here` in a scope of its own: a fresh memo scope and fragment cache (the names of the first run
    were made without the scope, as for the forced-locant run of ``gate_retry.name_with_forced_locants``)."""
    from ..assembly.fragment_naming import isolated_naming_session
    from ..assembly.locant_omission import scoped_skeleton_memo, systematic_names_for_unnumbered_labels
    from ..assembly.memo import fresh_scope, pop_sandbox
    memo = scoped_skeleton_memo()
    if memo is not None and (kind, stripped_smiles) in memo:
        return memo[(kind, stripped_smiles)]
    fresh = fresh_scope()
    try:
        with isolated_naming_session(reset_cache=True), systematic_names_for_unnumbered_labels():
            return _name_here(naming, kind, stripped_smiles)
    finally:
        pop_sandbox(fresh)


def decorate_with_systematic_unnumbered_parents(smiles, style, namer, original, stripped, label_map, sink,
                                                composer: bool = False) -> Optional[str]:
    """The decoration of ``original`` on a skeleton named with the systematic spelling of every retained name
    whose numbering does not reach a labelled position, when the skeleton with the retained names could not
    carry the labels; None when the skeleton is spelled the same way or nothing reads back.

    ``sink`` is what the strategies of the decorator left when they declined (``skeleton``, the spelling they
    ended on). It is made only after every strategy on the retained spelling declined (``rules.isotopes.
    _decorate_labelled`` calls it last), so no name built on that spelling changes, in a fresh memo scope and
    fragment cache (the names of the first run were made without the scope, as for the forced-locant run of
    ``gate_retry.name_with_forced_locants``). The skeleton is named once in the scope and compared with the
    one the strategies ended on: a molecule none of whose retained names has an unnumbered labelled position
    costs that one naming and no more. ``composer`` runs the scope composer on the new skeleton, as the caller
    does for the skeleton of the retained spelling."""
    if not label_may_sit_on_an_unnumbered_position(original):
        return None
    from ..assembly.locant_omission import systematic_names_for_unnumbered_labels
    from ..assembly.fragment_naming import isolated_naming_session
    from ..assembly.memo import fresh_scope, pop_sandbox
    stripped_smiles = Chem.MolToSmiles(stripped)
    fresh = fresh_scope()
    try:
        with isolated_naming_session(reset_cache=True), systematic_names_for_unnumbered_labels():
            systematic = (_name_here(namer, "caller", stripped_smiles) if style == "systematic"
                          else _name_here(_iso._flagged_systematic_namer(namer), "systematic", stripped_smiles))
            if not systematic or systematic == sink.get("skeleton"):
                return None
            new_sink: dict = {}
            result = _iso._decorate_isotopic_name_inner(
                smiles, style, namer, original, stripped, label_map, new_sink)
            if result is None and composer and new_sink:
                from .isotope_scopes import compose_last_resort
                result = compose_last_resort(new_sink, original, label_map)
    except Exception:  # noqa: BLE001 - an optional second spelling
        return None
    finally:
        pop_sandbox(fresh)
    return result


def skeleton_in_unnumbered_scope(namer, stripped) -> Optional[str]:
    """The isotope-stripped skeleton as ``namer`` names it inside the scope of, where a retained
    name that numbers no labelled position is spelled systematically (``rules.isotopes._scoped_parent``: the
    pre-emptive spelling of a labelled benzylic methylene or of the group of a benzene, which the retained name
    would carry as an unlocanted descriptor that reads back). Made once per decorator pass (:func:`_scoped_name`)
    and the provenance the caller had is restored: the name carries that of the skeleton it replaces."""
    from ..metrics.provenance import get_provenance, restore_provenance
    before = get_provenance()
    try:
        return _scoped_name(namer, "caller", Chem.MolToSmiles(stripped))
    except Exception:  # noqa: BLE001 - an optional spelling
        return None
    finally:
        restore_provenance(before)
