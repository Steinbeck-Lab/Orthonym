"""How the read-back of an isotope candidate compares OPSIN's parse with the input (IUPAC 2013,,.

``rules.isotopes._isotope_round_trips`` accepts a name only when OPSIN reads it as the input with its
isotopes. Two kinds of input are not read as drawn, and the comparison has to say when the reading is still
the input:

* a molecule with radical electrons: its names cite a free valence and are read with OPSIN's ``-r``
  (``RADICAL_READING``), and the reading has to carry the radical centres of the input
  (``radical_centres_agree``);
* a molecule with a guanidine or an amidine: a name that does not say where the double bond sits is read as
  the tautomer OPSIN prefers (``tautomer_of``). Such a reading is accepted only for the tautomers the Blue Book
  names without saying where the double bond is (``in_one_amidine_group``), and only in a pass of its own that
  the decorator makes when no reading as drawn was found (``tautomer_passes``), so no name it builds without
  that pass changes.
"""
from __future__ import annotations

import contextlib
import contextvars
import re
from typing import Dict, Iterator, List, Optional, Tuple

from rdkit import Chem

from ..perception.substructure import ordered_matches

#: Set by ``rules.isotopes._decorate_labelled`` while it names a molecule that has radical electrons: the
#: candidates of the decorator are then read with OPSIN's ``-r``, and not otherwise
#: (``validation/read_back_policy.py``: a name is read back with ``-r`` only when the input structure has
#: radical electrons). GENERAL METHODOLOGY (the Blue Book): "Radicals are named by modifying a
#: parent hydride name to signal the subtraction or addition of one or more hydrogen atoms"; a name with a
#: free valence ('[(2H3)methoxy]sulfinyl') does not parse without ``-r``, so a labelled radical had no
#: candidate that read back.
RADICAL_READING: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "orthonym_isotope_radical_reading", default=False)


def radical_centres_agree(original: Chem.Mol, parsed_smiles: str) -> bool:
    """True iff the reading ``parsed_smiles`` does not contradict the radical centres of ``original``. The full
    InChIKey sees neither radical electrons nor bond orders, and OPSIN's SMILES of a name with a free
    valence leaves an atom that has a valence of its own closed-shell: ``validation.radical_identity`` is the
    one definition of the comparison (the radical profile and the graph without stereo). Its verdict is "ok"
    (the same chemical radicals), "mismatch", or "n/a": neither side carries a chemical radical, so the
    checks of the caller decide. RDKit gives radical electrons to a bare metal atom ('[99Tc]' of a mixture,
    '[65Zn]'), which are not chemical radicals, ``radical_identity``): such an input has the verdict
    "n/a" and its names stand or fall on the full key and the isotope layer alone, as at the default read."""
    from ..validation.radical_identity import radical_identity_verdict
    return radical_identity_verdict(Chem.MolToSmiles(original), parsed_smiles) != "mismatch"


def _tautomer_graph(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """``mol`` with every bond single and no ordinary hydrogen counted on any atom (isotopes -- an isotopic
    hydrogen is an atom of its own --, charges and the element of every atom kept): the constitution of the
    atoms and nuclides without the bond orders and without the place of the ordinary hydrogens. Each atom
    keeps its index in ``mol`` as the property ``_t``. Two drawings with one canonical SMILES of this graph
    differ only in where the ordinary hydrogens and the double bonds sit; which of those the Blue Book names
    alike is (guanidine) and (amidine), see ``in_one_amidine_group``."""
    try:
        m = Chem.RWMol(mol)
        for atom in m.GetAtoms():
            atom.SetIntProp("_t", atom.GetIdx())
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(True)
            atom.SetIsAromatic(False)
            atom.SetNumRadicalElectrons(0)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for bond in m.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
            bond.SetStereo(Chem.BondStereo.STEREONONE)
        m.UpdatePropertyCache(strict=False)
        return Chem.RemoveHs(m.GetMol(), sanitize=False)
    except Exception:  # noqa: BLE001 - a molecule that cannot be graphed is not judged a tautomer
        return None


#: How many mappings of one graph onto another ``_moved_atoms`` compares.
_MAPPING_CAP = 200


def _hydrogen_total(mol: Chem.Mol) -> int:
    """Every hydrogen of ``mol``: the counts of its atoms and its hydrogen atoms."""
    return sum(a.GetTotalNumHs() if a.GetAtomicNum() != 1 else 1 for a in mol.GetAtoms())


def _known(original: Chem.Mol) -> tuple:
    """``(key, hydrogens, graph)`` of ``original``: what ``tautomer_of`` compares a reading with."""
    graph = _tautomer_graph(original)
    return (None if graph is None else Chem.MolToSmiles(graph)), _hydrogen_total(original), graph


def _moved_atoms(parsed: Chem.Mol, original: Chem.Mol, key: Optional[str] = None,
                 graph: Optional[Chem.Mol] = None) -> Optional[List[Tuple[Chem.Atom, bool]]]:
    """The atoms of ``parsed`` that have another number of hydrogens or another double bond than the atom of
    ``original`` they are mapped onto, each with whether its hydrogens differ, for the mapping of the graphs
    (``_tautomer_graph``) that moves the fewest; None when the graphs are not one graph. ``key`` and ``graph``
    are the key and the graph of ``original`` when the caller has them."""
    ga, gb = _tautomer_graph(parsed), graph if graph is not None else _tautomer_graph(original)
    if ga is None or gb is None or (key is not None and Chem.MolToSmiles(ga) != key):
        return None
    best = None
    for match in ordered_matches(gb, ga, uniquify=False, max_matches=_MAPPING_CAP):
        moved = []
        for i, j in enumerate(match):
            a = parsed.GetAtomWithIdx(ga.GetAtomWithIdx(i).GetIntProp("_t"))
            b = original.GetAtomWithIdx(gb.GetAtomWithIdx(j).GetIntProp("_t"))
            hydrogens = a.GetTotalNumHs() != b.GetTotalNumHs()
            if hydrogens or _has_double_bond(a) != _has_double_bond(b):
                moved.append((a, hydrogens))
        if best is None or len(moved) < len(best):
            best = moved
    return best


def _has_double_bond(atom: Chem.Atom) -> bool:
    return any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds())


def tautomer_of(parsed: Chem.Mol, original: Chem.Mol, known: Optional[tuple] = None) -> bool:
    """True iff ``parsed`` is ``original`` drawn as one of its tautomers: the same atoms, charges, nuclides
    and bonds (``_tautomer_graph``), the same number of hydrogens, and ordinary hydrogens on other atoms (so
    other bond orders). The same hydrogens on the same atoms are not this (a Kekule form or a bond-shift
    isomer, ``rules.isotopes._kekule_forms_of_one_compound`` judges those).

    A guanidine whose name does not say which nitrogen carries the double bond is read by OPSIN as the
    tautomer other than the drawn one ('N'-aryl-N-[(11C)methyl]-N-aryl-guanidine' as ``ArNC(N(Ar)[11CH3])=N``
    for the input's ``ArN=C(N)N(Ar)[11CH3]``): the same compound by the standard InChIKey, which the caller
    still requires, and by every position a nuclide is cited at, because the nuclides are part of the key
    above (a deuterium on a heteroatom is an atom of it, so one moved to another heteroatom is another key).
    The fixed-hydrogen InChI the caller makes first is never equal for two tautomers: it records where the
    hydrogens are. ``known`` is ``_known(original)``; which tautomers a name may stand for is
    ``in_one_amidine_group``."""
    key, hydrogens, graph = known if known is not None else _known(original)
    if key is None or _hydrogen_total(parsed) != hydrogens:
        return False
    moved = _moved_atoms(parsed, original, key, graph)
    return bool(moved) and any(hydrogens_differ for _a, hydrogens_differ in moved)


def in_one_amidine_group(parsed: Chem.Mol, original: Chem.Mol) -> bool:
    """True iff the two drawings of one compound (``tautomer_of``) differ only on the nitrogens of one
    guanidine or amidine group: every atom that has another number of hydrogens or another double bond is a
    non-aromatic nitrogen bonded to one non-aromatic carbon that has two or more nitrogens.

    That is the class whose name the Blue Book writes without saying where the double bond sits.
     (heading 'Guanidine and its derivatives', the Blue Book): "Hydrocarbyl derivatives
    are named as substituted guanidines. When the position of the double bond is unknown, the preferred IUPAC
    name uses a minimum number of primes." ('Substituted amidines',:34375): "Because of
    tautomerism, the locants N, N', etc. are used when only one substituent is present on each imidamide
    group." The tautomers of an amide and its imidic acid, of a ring NH of an azole, of a keto and an enol
    form, are named for the drawn tautomer (indicated hydrogen, 'one'), so a name read as another tautomer of
    those is not the name of the input and no rule makes it neutral."""
    moved = _moved_atoms(parsed, original)
    if not moved or any(a.GetSymbol() != "N" or a.GetIsAromatic() for a, _h in moved):
        return False
    centres = None
    for a, _h in moved:
        carbons = {n.GetIdx() for n in a.GetNeighbors() if n.GetSymbol() == "C" and not n.GetIsAromatic()
                   and sum(1 for m in n.GetNeighbors() if m.GetSymbol() == "N") >= 2}
        centres = carbons if centres is None else centres & carbons
    return bool(centres)


#: The state of the passes of one decorator call (``tautomer_passes``): ``possible`` is True when the input has
#: a hydrogen that moves (a mobile-hydrogen group of its standard InChI), ``accept`` is True in the pass that
#: reads a tautomer of the input as the input, ``refused`` is set when a name that reads as a tautomer was
#: refused. None outside a decorator call: a direct use of the read-back accepts no tautomer.
TAUTOMER_READING: "contextvars.ContextVar[Optional[dict]]" = contextvars.ContextVar(
    "orthonym_isotope_tautomer_reading", default=None)

_MOBILE_HYDROGEN_RE = re.compile(r"\(H\d*,")


def _has_mobile_hydrogen(mol: Chem.Mol) -> bool:
    """True when the standard InChI of ``mol`` has a mobile-hydrogen group (the hydrogens that tautomerism
    moves in the model of the InChIKey every read-back ends in)."""
    try:
        from rdkit.Chem import inchi
        text = inchi.MolToInchi(mol) or ""
    except Exception:  # noqa: BLE001 - no InChI, no key to be equal: no tautomer to accept
        return False
    layer = next((p for p in text.split("/") if p.startswith("h")), "")
    return bool(_MOBILE_HYDROGEN_RE.search(layer))


@contextlib.contextmanager
def tautomer_passes(original: Chem.Mol) -> Iterator[Dict[str, object]]:
    """The state of the passes of one decorator call over ``original``."""
    state: Dict[str, object] = {"possible": _has_mobile_hydrogen(original), "accept": False,
                                "refused": False, "known": {}}
    token = TAUTOMER_READING.set(state)
    try:
        yield state
    finally:
        TAUTOMER_READING.reset(token)


def tautomer_accepted(parsed: Chem.Mol, original: Chem.Mol) -> bool:
    """True iff ``parsed`` reads as the input because it is a tautomer of it within one guanidine or amidine
    group (``in_one_amidine_group``) and the pass accepts that. A tautomer refused is recorded
    (``refused``): the decorator makes the pass that accepts it only then. Only an input with a mobile
    hydrogen has a tautomer, so any other costs nothing here."""
    state = TAUTOMER_READING.get()
    if state is None or not state["possible"]:
        return False
    known = state["known"]
    entry = known.get(id(original))
    if entry is None:
        entry = known[id(original)] = (original, _known(original))
    if not tautomer_of(parsed, original, entry[1]) or not in_one_amidine_group(parsed, original):
        return False
    if state["accept"]:
        return True
    state["refused"] = True
    return False
