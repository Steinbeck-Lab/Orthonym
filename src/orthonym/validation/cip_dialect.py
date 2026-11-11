"""A stereo name whose OPSIN reading differs from the input only where OPSIN ranks a mancude ring without averaging.

 "Mancude rings and ring systems" (the Blue Book): "For mancude heterocycles, each duplicate atom is
given an atomic number that is the mean of what it would have if the double bonds were located at each of the possible
positions." The CIP engine of record (``centres``, ``perception/centres_bridge.py``) applies that rule; OPSIN 2.9.0
does not (the OPSIN maintainer, dan2097/opsin#349, 2026-10-31: "a lack of support for atomic number averaging in
mancude rings in OPSIN's CIP implementation"). At a stereo unit whose ranking that rule decides, OPSIN gives the
input's configuration the other letter, so a correct name reads back as the other stereoisomer at that unit, and the
read-back declines it.

The judge,:func:`cip_dialect_name`, ships such a name at the best-effort tier only (the default tier keeps the
decision of 2026-10-31 to decline it), when the disagreement is proven to be exactly that rule:

1. D = the stereo units whose label of record differs from the ``centres`` label with the averaging turned off on
   every ring atom (``_dialect_mol``: the held-atom input of ``centres_bridge._centres_input`` for every ring atom,
   a formal charge one higher in the string ``centres`` labels; is the only rule that reads it). D must not
   be empty.
2. N' is the flat name with each descriptor at the front of the part it relates to, placed by
   ``rules.prefix_stereo`` with the letters OPSIN reads right (the record's, swapped at each unit of D); N cites the
   same places and locants with the record's letters, built by construction.
3. N and N' differ in exactly |D| letters, each the swapped letter of a unit of D.
4. OPSIN reads N' to the input's full InChIKey, with the radical graph, the protonation site and the key-blind double
   bonds checked (``namer._shipped_name_round_trip``), and gives no ambiguity warning for it.
5. ``rdCIPLabeler`` gives the label of record at every unit of D: the judge removes OPSIN's cross-check exactly at D,
   so another labeller holds it there.

Then N denotes the input under the ranking of record and OPSIN's reading differs only by the rule it lacks. The name
is labelled ``systematic_verified`` with the gate outcome ``cip_dialect_verified`` (never a PIN label).
"""
import logging
from typing import Dict, Optional, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)

_SWAP = {"R": "S", "S": "R", "r": "s", "s": "r", "E": "Z", "Z": "E"}


def _labels(mol) -> Dict[Tuple[str, int], str]:
    """The ``_CIPCode`` of every atom and double bond of ``mol`` that carries an R/S/r/s/E/Z letter."""
    found = {}
    for atom in mol.GetAtoms():
        if atom.HasProp("_CIPCode") and atom.GetProp("_CIPCode") in _SWAP:
            found[("atom", atom.GetIdx())] = atom.GetProp("_CIPCode")
    for bond in mol.GetBonds():
        if bond.HasProp("_CIPCode") and bond.GetProp("_CIPCode") in ("E", "Z"):
            found[("bond", bond.GetIdx())] = bond.GetProp("_CIPCode")
    return found


def _dialect_mol(mol):
    """A copy of ``mol`` labelled by ``centres`` with the averaging of off on every ring atom, or None."""
    from ..perception.centres_bridge import _smiles_output_order, apply_centres_labels, centres_label_batch
    work = Chem.RWMol(mol)
    Chem.Kekulize(work, clearAromaticFlags=True)
    for atom in work.GetAtoms():
        if atom.IsInRing():
            hydrogens = atom.GetTotalNumHs()
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(hydrogens)
            atom.SetFormalCharge(atom.GetFormalCharge() + 1)
    work.UpdatePropertyCache(strict=False)
    smi = Chem.MolToSmiles(work)
    order = _smiles_output_order(work)
    batch = centres_label_batch([smi])
    if batch is None or order is None:
        return None
    remapped = {}
    for pos1, desc in batch.get(smi, {}).items():
        if 0 <= pos1 - 1 < len(order):
            remapped[order[pos1 - 1] + 1] = desc
    labelled = Chem.Mol(mol)
    for atom in labelled.GetAtoms():
        atom.ClearProp("_CIPCode")
    for bond in labelled.GetBonds():
        bond.ClearProp("_CIPCode")
    apply_centres_labels(labelled, remapped)
    return labelled


def _rdkit_labels(mol) -> Dict[Tuple[str, int], str]:
    from rdkit.Chem import rdCIPLabeler
    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        atom.ClearProp("_CIPCode")
    for bond in work.GetBonds():
        bond.ClearProp("_CIPCode")
    rdCIPLabeler.AssignCIPLabels(work, maxRecursiveIterations=1250000)
    return _labels(work)


def dialect_units(mol) -> Optional[Dict[Tuple[str, int], str]]:
    """``{unit: label of record}`` for the units whose label of record and dialect label are both set and differ (D),
    given ``mol`` labelled by the path of record (``perception.stereo.assign_stereochemistry``); None when the dialect
    labels cannot be computed."""
    dialect = _dialect_mol(mol)
    if dialect is None:
        return None
    record, other = _labels(mol), _labels(dialect)
    return {unit: label for unit, label in record.items()
            if unit in other and other[unit] != label and _SWAP[label] == other[unit]}


def stereo_locants_once(name: str) -> bool:
    """True when no locant stands twice in a stereodescriptor block of ``name``: '(2S,2S,2S)-' cites three units at
    one locant (substituents numbered apart), and OPSIN would give the letters by the order it reads them, so the
    reading of such a name, and of its twin, rests on that order."""
    import re
    for block in re.findall(r"\(((?:\d+[A-Za-z0-9'^]*?[RSrsEZ],?)+)\)-", name):
        locants = [item[:-1] for item in block.split(",") if item]
        if len(set(locants)) != len(locants):
            return False
    return True


def cip_dialect_name(flat_name: str, smiles: str) -> Optional[Tuple[str, str]]:
    """``(N, "<count>_units")`` with N (the name of ``smiles`` with its stereo of record composed on ``flat_name`` at OPSIN's numbering) when the
    judge of this module proves that OPSIN reads it otherwise only by the missing averaging; else None."""
    try:
        from ..perception.stereo import _CIP_ASSIGNED_PROP, assign_stereochemistry
        from .opsin_ambiguity import ambiguity_messages
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        record = Chem.Mol(mol)
        assign_stereochemistry(record)
        units = dialect_units(record)
        if not units:
            return None
        rd = _rdkit_labels(mol)
        if any(rd.get(unit) != label for unit, label in units.items()):
            return None
        swapped = Chem.Mol(record)
        for (kind, idx), label in units.items():
            target = swapped.GetAtomWithIdx(idx) if kind == "atom" else swapped.GetBondWithIdx(idx)
            target.SetProp("_CIPCode", _SWAP[label])
        swapped.SetProp(_CIP_ASSIGNED_PROP, "1")
        # each descriptor at the front of the part it relates to, the Blue Book), placed with the
        # letters OPSIN reads right (``rules.prefix_stereo``): the twin; the name cites the same places and locants
        # with the record's letters at the units of D
        from ..rules.prefix_stereo import place_pair_by_scope
        pair = place_pair_by_scope(flat_name, smiles, swapped,
                                   frozenset(("a" if kind == "atom" else "b", idx) for kind, idx in units))
        if pair is None:
            return None
        name, twin = pair
        if not name or name == flat_name or len(twin) != len(name) or not stereo_locants_once(name):
            return None
        differ = [i for i, (a, b) in enumerate(zip(name, twin)) if a != b]
        if len(differ) != len(units) or any(_SWAP.get(name[i]) != twin[i] for i in differ):
            return None
        from ..namer import _shipped_name_round_trip
        if _shipped_name_round_trip(twin, smiles) != "verified" or ambiguity_messages(twin):
            return None
        if _shipped_name_round_trip(name, smiles) == "verified":
            return None     # OPSIN reads the name itself: not a dialect case
        return name, f"{len(units)}_units"
    except Exception as exc:  # noqa: BLE001 - fail closed: no judge, no name
        logger.debug("cip dialect judge skipped (%s): %s", type(exc).__name__, exc)
        return None
