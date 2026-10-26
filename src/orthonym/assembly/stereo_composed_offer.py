"""The flat name of a molecule with the input's stereo composed on it: the last offer of the best-effort tier.

A molecule with stereo is sometimes left without a name by every path of the best-effort tier although the
same structure without its stereo is named (a ring system with stereocentres that the producers decline,
where the run on the stereo-free drawing names it). The stereo-free name is a name of the constitution; the
stereo of the input can be composed onto it in OPSIN's own numbering of that name
(``namer._try_compose_input_stereo``, the stereo-omission reclaim), and the composed name is a name of the input
when OPSIN reads it back to the input's full InChIKey.

It is an offer, not a return (a project rule): it is made only when every other path of the best-effort tier,
the other offers of ``assembly/default_tier_offer.py`` included, has left the molecule without a name, so no
name that ships changes. It runs once for the caller's own call, after the retries of that call, as the other
offers do (``default_tier_offer.default_tier_row_for`` calls:func:`stereo_composed_row`), so ``name``,
``name_compound``, ``name_with_confidence`` and ``name_tiered`` agree.

The stereo-free structure is named by an engine of the tier of the caller (``general_tier_engine``), an
outermost run of its own with the hang budgets of one compound, and the whole offer runs inside the caller's
own wall-clock limit (``--time-limit``: ``WallClockTimeout`` is a ``BaseException`` and is not caught here).
The stereo-free structure has no stereo, so it is offered nothing of this kind.

What ships is checked more than the composition checks it: OPSIN reads the composed name back to the input's
full InChIKey with the radical graph and the protonation site of the input (``namer._shipped_name_round_trip``),
the exact isomeric comparison of the canonical SMILES that the other composed names pass
(``namer.Orthonym._rt_match``: it sees the E/Z and ring-ylidene configurations the key cannot), for an input with
a lone-pair stereocentre the configuration check of the exit (``namer._lone_pair_configuration_verified``), and
the exit check of every other name (``Orthonym._exit_round_trip_check``: a name OPSIN reads as ambiguous is
withdrawn, ``validation.opsin_ambiguity``), which the offer must pass itself because ``name_with_confidence``
does not apply it to an offered name. An input with enhanced stereo groups (an AND or OR group of CXSMILES,
``|&1:1|``) is never offered: its centres are not absolute, and a descriptor would assert that they are. Any
other outcome leaves the abstention as it was, with the provenance and the abstention record of the caller.

The name is not the PIN: NAMING OF STEREOISOMERS (the Blue Book), sentence:44643, "In preferred
IUPAC names, stereodescriptors are placed immediately at the front of the part of the name to which they
relate", and "They are preceded by a numerical or letter locant to describe the position of the stereogenic
unit". The composed name cites every descriptor at the front of the complete name, in the locants of the
stereo-free name, so a descriptor of a substituent can repeat a locant of the parent. A repeated locant does
not give the position of a stereogenic unit, and the name then rests on the order in which OPSIN reads the
descriptors: ``(3S,3R,...)`` is read to the input, and ``(3R,3S,...)`` to a diastereomer. A composed name that
cites a locant more than once is therefore offered only when (1) every descriptor at that locant is the same
(``4S,4S,4S``: whichever unit takes which descriptor, the structure is one) and (2) the descriptors state every
stereo unit of the input, which has no unspecified one (:func:`repeated_locants_are_one_structure`: an unspecified
centre that shares the locant would make the equal descriptors ambiguous). Any other name with a repeated locant
is declined, and the molecule keeps its abstention. The row is labelled by the rules of
``Orthonym._tier_row_and_pin_form`` for a name of the general engine whose gate outcome is
``stereo_omission_full_key_recomposed`` (``systematic_verified``, ``is_pin`` False), and the producer is the one
that built the stereo-free name.
"""
import re
from typing import Dict, List, Optional, Tuple

#: a parenthesised group of the name with no other enclosing mark inside it, and one descriptor of such a group:
#: its locant (possibly primed or lettered: ``4a``, ``1'``; empty for ``(Z)``) and its descriptor
_GROUP = re.compile(r"\(([^()\[\]{}]+)\)")
_DESCRIPTOR = re.compile(r"^(?P<locant>.*?)(?P<descriptor>Ra|Sa|Rp|Sp|[RSEZMPrs])$")


def stereo_groups(name: str) -> List[List[Tuple[str, str]]]:
    """The groups of stereodescriptors of ``name``: each parenthesised group all of whose comma-separated items
    are ``(locant, descriptor)`` pairs, as the list of its pairs, in the order of the name. Any other group (a
    substituent name, an indicated hydrogen) is not one."""
    groups = []
    for found in _GROUP.finditer(name):
        pairs = []
        for item in found.group(1).split(","):
            parsed = _DESCRIPTOR.match(item)
            if parsed is None:
                break
            pairs.append((parsed.group("locant"), parsed.group("descriptor")))
        else:
            groups.append(pairs)
    return groups


def _defined_stereo_units(smiles: str) -> Optional[int]:
    """The number of stereo units of ``smiles`` (stereocentres and stereo double bonds, RDKit's
    ``FindPotentialStereo``) when every one of them is specified and the input has no enhanced stereo group; else
    None."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetStereoGroups():
        return None
    units = Chem.FindPotentialStereo(mol)
    if any(unit.specified != Chem.StereoSpecified.Specified for unit in units):
        return None
    return len(units)


def repeated_locants_are_one_structure(name: str, smiles: str) -> bool:
    """False when ``name`` cites a locant more than once in a group of stereodescriptors and does not name one
    structure by it: True when no locant is repeated; else only when every descriptor at each repeated locant is
    the same and the descriptors of the name are as many as the stereo units of ``smiles``, all of which are
    specified (the module docstring:, the Blue Book)."""
    groups = stereo_groups(name)
    repeated = False
    for pairs in groups:
        by_locant: Dict[str, List[str]] = {}
        for locant, descriptor in pairs:
            by_locant.setdefault(locant, []).append(descriptor)
        for descriptors in by_locant.values():
            if len(descriptors) > 1:
                repeated = True
                if len(set(descriptors)) > 1:
                    return False
    if not repeated:
        return True
    return _defined_stereo_units(smiles) == sum(len(pairs) for pairs in groups)


def has_stereo_groups(structure) -> bool:
    """True when ``structure`` (a SMILES string, CXSMILES included, or an RDKit molecule) has an enhanced
    stereo group (AND, OR): its centres are not absolute."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(structure) if isinstance(structure, str) else structure
    return mol is not None and bool(mol.GetStereoGroups())


def stereo_free_smiles(smiles: str) -> Optional[str]:
    """The canonical SMILES of ``smiles`` with every stereo element removed, or None when that is the
    SMILES of the input as it stands (a molecule with no stereo, and so the stereo-free run itself) or when
    RDKit cannot read ``smiles``. Isotopes and charges stay: only stereo is removed."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    given = Chem.MolToSmiles(mol)
    Chem.RemoveStereochemistry(mol)
    flat = Chem.MolToSmiles(mol)
    return flat if flat != given else None


def _eligible(namer) -> bool:
    """A best-effort engine of the PIN style, not one of the junior parents of the pool, with the OPSIN
    validity gate on and its jar present (a name that no OPSIN reads back is never offered), and the
    stereo-omission reclaim not switched off (``ORTHONYM_STEREO_OMISSION_RECLAIM=0``)."""
    from ..namer import _DISABLE_VALIDITY_GATE, _STEREO_OMISSION_RECLAIM, _validity_gate_jar_present
    return bool(
        getattr(namer, "style", None) == "pin" and getattr(namer, "_general_fallback_unverified", False)
        and not getattr(namer, "_forced_parent_rank", 0) and _STEREO_OMISSION_RECLAIM
        and not namer._disable_opsin_validity_gate and not _DISABLE_VALIDITY_GATE and _validity_gate_jar_present())


def stereo_composed_row(namer, smiles) -> Optional[Dict]:
    """The row ``namer`` offers a molecule it left without a name: the name of the stereo-free structure with
    the stereo of ``smiles`` composed on it, when that name reads back to the input (see the module
    docstring); else None. ``smiles`` is a SMILES string or an RDKit molecule, as for ``name_tiered``.
    Best-effort tier only. Never raises (the abstention stands)."""
    try:
        if not _eligible(namer):
            return None
        from ..input_mol import handles, smiles_for
        if has_stereo_groups(smiles):
            return None
        if not isinstance(smiles, str) and handles(smiles):
            smiles = smiles_for(smiles)
        flat = stereo_free_smiles(smiles)
        if flat is None:
            return None
        return _composed_row(namer, smiles, flat)
    except Exception:                       # the offer must never break a naming: the abstention stands
        return None


def _composed_row(namer, smiles: str, flat: str) -> Optional[Dict]:
    from ..errors import is_failure_name
    from ..metrics import provenance as pv
    from ..metrics.abstention import peek_abstention, restore_abstention
    from ..metrics.producer_trail import carried
    from .default_tier_offer import general_tier_engine
    kept_provenance, kept_abstention = pv.get_provenance(), peek_abstention()
    row = None
    try:
        engine = general_tier_engine(namer)
        flat_row = engine.name_tiered(flat)
        flat_name = flat_row.get("name")
        if not flat_name or is_failure_name(flat_name) or flat_row.get("tier") == "abstain":
            return None
        composed = _verified_composed_name(smiles, flat_name)
        if composed is None:
            return None
        pv.clear_provenance()
        pv.record_source("general_engine")
        pv.record_gate_outcome(pv.GATE_OUTCOME_STEREO_RECOMPOSED, composed)
        if carried(composed, engine._exit_round_trip_check(composed, smiles)) != composed:
            return None                     # the exit check every other name passes (ambiguity, lone pair)
        row = engine._tier_row_and_pin_form(smiles, composed, pv.get_provenance())[0]
        if row.get("name") != composed or row.get("tier") in (None, "abstain") or row.get("is_pin"):
            row = None
            return None
        row["producer"] = flat_row.get("producer")
        from ..validation.stereo_stated import name_states_fewer_stereo
        if name_states_fewer_stereo(smiles, composed) or pv.get_provenance().get("stereo_unexpressed"):
            row["stereo_unexpressed"] = True
        return row
    finally:
        # the stereo-free run was a naming of its own: what the caller reads about its abstention is
        # what it was, unless a name is offered
        restore_abstention(kept_abstention)
        if row is None:
            pv.restore_provenance(kept_provenance)


def _verified_composed_name(smiles: str, flat_name: str) -> Optional[str]:
    """``flat_name`` with the stereo of ``smiles`` composed on it, when the composed name passes every check of
    the module docstring; else None."""
    from ..errors import is_failure_name
    from ..metrics.producer_trail import carried
    from ..namer import (
        Orthonym, _has_lone_pair_stereocentre, _lone_pair_as_written, _lone_pair_configuration_verified,
        _self_consistency_full_key, _shipped_name_round_trip, _try_compose_input_stereo,
        _validity_gate_name_to_smiles)
    in_key = _self_consistency_full_key(smiles)
    if not in_key:
        return None
    composed = carried(flat_name, _try_compose_input_stereo(flat_name, smiles, in_key))
    if not composed or is_failure_name(composed) or not repeated_locants_are_one_structure(composed, smiles):
        return None
    opsin_smiles = _validity_gate_name_to_smiles(composed)
    if (opsin_smiles is None or not Orthonym._rt_match(smiles, opsin_smiles, False)
            or _shipped_name_round_trip(composed, smiles) != "verified"):
        return None
    written = _lone_pair_as_written(smiles)
    if _has_lone_pair_stereocentre(written) and not _lone_pair_configuration_verified(composed, written):
        return None
    return composed
