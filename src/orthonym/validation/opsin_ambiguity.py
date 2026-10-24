"""A name that OPSIN reads as ambiguous.

OPSIN raises the warning ``APPEARS_AMBIGUOUS`` when part of a name has more than one reading
and it had to choose ("Positioning of radical on: aziridin", "Position of isotope on benz",
"Connection of phenyl to eth"). Such a name can round-trip, because OPSIN's choice happens to
be the input, and still not say which structure it names. Citation of locants
(the Blue Book): "In preferred IUPAC names, if any locants are essential for defining the
structure of the parent structure or of a unit of structure as defined by its appropriate
enclosing marks, then all locants must be cited for the parent structure or that structural
unit." So the final validity check reads the warning and withdraws such a name, and the
callers look for a name OPSIN reads once: the same producers once more with every omission
licence switched off (``locant_omission.forced_locant_scope``, reason
:data:`FORCED_LOCANT_REASON`), then the other classes of the cascade.

Two things keep a withdrawn name from being lost.

* A class the Blue Book itself prints in a form OPSIN reads as ambiguous (a specific-method
  carbenium, a label on a delocalized group, the pseudoketone of a peroxy chain) is not
  withdrawn (:func:`licensed_ambiguity`): the class is read from the structure by the rule that
  builds it, and the name is the PIN.
* A molecule that has no other name keeps the withdrawn one (:func:`restore_withdrawn`),
  recorded as a name that is not the PIN: a name that round-trips is a better answer than none
  at the best-effort tier, where it is labelled below the PIN. The default tier ships a
  preferred name or declines (``NO_VERIFIED_PIN``: the engine built a name that is not a
  verified PIN), so the record is what makes it decline there.
"""
import logging
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

#: the OPSIN warning type that says a name has more than one reading
AMBIGUITY_WARNING = "APPEARS_AMBIGUOUS"

#: the reason of the ``forced_locant_scope`` of the retry
FORCED_LOCANT_REASON = "opsin_ambiguity"

#: (:15111) "locants for amplificant atoms are constructed by citing the actual locants of the amplificant
#: as superscripts to the locant of the superatom": in a phane name a plain number is a skeleton locant. For a plain
#: locant of 10 or more OPSIN main warns that it could also be a composite locant written without its superscript
#: ("The locant 12 could refer to an atom of the simplified skeleton or to an atom of an amplificant (superscript locants
#: should be written e.g. 1^4)"); Orthonym writes every composite locant with ``^``, so the warning describes no
#: ambiguity of the name.
_SKELETON_LOCANT = re.compile(
    r"^The locant \d+ could refer to an atom of the simplified skeleton or to an atom of an amplificant")

#: the names of the per-call side store (``assembly.memo.side_put``): the name withdrawn for a
#: structure with the provenance it had, and the name kept as the last resort
_WITHDRAWN = "opsin_ambiguity_withdrawn"
_KEPT = "opsin_ambiguity_kept"


def ambiguity_messages(name: str) -> Tuple[str, ...]:
    """The ``APPEARS_AMBIGUOUS`` messages OPSIN gives for ``name``, read with the radical
    setting of the validity gate's oracle; empty when OPSIN gives none, rejects the name or
    cannot be asked in process (the round trip of the same name fails closed on an
    unavailable OPSIN, so this check adds no verdict there)."""
    from ..jvm_bridge import OpsinUnavailable, opsin_warnings
    if not name:
        return ()
    try:
        warnings = opsin_warnings(name, allow_radicals=True)
    except OpsinUnavailable:
        return ()
    return tuple(message for kind, message in warnings
                 if kind == AMBIGUITY_WARNING and not _SKELETON_LOCANT.match(message))


def licensed_ambiguity(name: str, smiles: Optional[str]) -> Optional[str]:
    """The Blue Book rule whose class ``name`` belongs to, when that rule prints the name although
    OPSIN reads it as ambiguous, else None.

    * Specific method (the Blue Book): 'propylium (PIN)', 'methylium (PIN)'
      (:41557). The class is a terminal cation of a saturated unbranched acyclic hydrocarbon,
      read from ``smiles`` by ``rules.ions.specific_method_cation_name``, the producer's own
      class predicate; the name is the one the rule builds for it. The method names the terminal
      atom by definition, so the name has one position.
    * DELOCALIZED RADICALS AND IONS (:43540): a heavy nuclide on an atom of a delocalized
      group is one species whatever atom carries it, so its descriptor cites no locant
      (``rules.isotopes.heavy_label_on_delocalized_atoms``); only when the warnings OPSIN gives
      are about the position of that isotope.
    * Pseudoesters (:32074): "When 'E' is an element belonging to Group 16, the
      pseudoester is named as a pseudoketone", example '[1-(methylperoxy)sulfanyl]butan-1-one
      (PIN)' (:32078), the locant inside the marks of the prefix. The class is an acyl group on
      S, Se or Te with a peroxy chain that ends in an organyl group, read from ``smiles`` by
      ``rules.pseudoketones.pseudoester_book_name``, the producer's own class predicate; the
      name is the one the producer builds for it. The book spells the pseudoketones of
       (the diacyl analogue) and of with the locant in front of the marks,
      and those names are read once, so they are not in this class."""
    if not name or not smiles:
        return None
    try:
        from rdkit import Chem

        from ..rules.ions import specific_method_cation_name
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if specific_method_cation_name(mol) == name:
            return "P-73.2.2.1.1"
        from ..rules.pseudoketones import pseudoester_book_name
        if pseudoester_book_name(mol) == name:
            return "P-65.6.3.4.2"
        from ..rules.isotopes import heavy_label_on_delocalized_atoms
        messages = ambiguity_messages(name)
        if (messages and all(m.startswith("Position of isotope on") for m in messages)
                and heavy_label_on_delocalized_atoms(mol)):
            return "P-76"
    except Exception:                                           # noqa: BLE001
        return None
    return None


def reject_ambiguous(name: str, smiles: Optional[str], decided: str,
                     stats: Optional[Dict[str, int]] = None) -> str:
    """The last step of the validity gate for a name that OPSIN parsed.

    ``decided`` is what the constitution check answered for ``name``: the name itself, or the
    fallback that replaced it. A name it kept is withdrawn to the fallback when OPSIN reads it
    as ambiguous, and the gate outcome is recorded for the fallback."""
    if decided != name:
        return decided
    messages = ambiguity_messages(name)
    if not messages or licensed_ambiguity(name, smiles):
        return name
    from ..assembly.memo import side_get, side_put
    from ..metrics import provenance as pv
    from ..metrics.abstention import AbstentionCode, record_suppression
    from ..namer import _descriptive_fallback
    if side_get(_WITHDRAWN, smiles) is None:      # the first name withdrawn is the one kept
        side_put(_WITHDRAWN, smiles, (name, pv.get_provenance()))
    if stats is not None:
        stats["opsin_ambiguous_suppressed"] = stats.get("opsin_ambiguous_suppressed", 0) + 1
    logger.info("OPSIN validity gate withdrew an ambiguous name: %r (%s)", name[:80], messages[0])
    record_suppression(AbstentionCode.GATE_SUPPRESSED, detail="opsin_ambiguous", candidate=name)
    fallback = _descriptive_fallback(smiles)
    pv.record_gate_outcome(pv.GATE_OUTCOME_AMBIGUOUS_SUPPRESSED, fallback)
    return fallback


def withdrawn_name(name: str, smiles: Optional[str]) -> bool:
    """True when ``name`` is OPSIN-ambiguous and is not the name kept for ``smiles`` as the last
    resort: the check of the exit of a naming call."""
    from ..assembly.memo import side_get
    return (bool(ambiguity_messages(name)) and side_get(_KEPT, smiles) != name
            and not licensed_ambiguity(name, smiles))


def restore_withdrawn(name: str, smiles: Optional[str]) -> str:
    """The last resort of a molecule left without a name: the name that was withdrawn for it
    because OPSIN reads it as ambiguous, with the provenance it had when it was withdrawn and
    recorded as a name that is not the PIN.

    ``name`` is what the call has at its end. It is returned unchanged unless it is a failure
    name and a name was withdrawn for this structure in this call, and that name is verified:
    its gate outcome says so, or its full round trip does. The record of a name that is not
    the PIN labels it below the PIN at the best-effort tier and makes the default tier decline
    it. Never raises."""
    from ..errors import is_failure_name
    if not smiles or (name and not is_failure_name(name)):
        return name
    try:
        from ..assembly.memo import side_get, side_put
        from ..metrics import provenance as pv
        held = side_get(_WITHDRAWN, smiles)
        if held is None:
            return name
        candidate, snapshot = held
        outcome = pv.resolve_gate_outcome(
            snapshot.get("gate_outcome"), snapshot.get("gate_outcome_name"), candidate)
        verified = pv.opsin_label_for_gate_outcome(outcome) != "unverified"
        if not verified:
            from ..namer import _shipped_name_round_trip
            if _shipped_name_round_trip(candidate, smiles) != "verified":
                return name
        pv.restore_provenance(snapshot)
        if not verified:
            pv.record_gate_outcome(pv.GATE_OUTCOME_FULL_KEY_VERIFIED, candidate)
        #: a name OPSIN reads as ambiguous is not certified as the preferred name
        pv.record_non_pin_fragment(candidate)
        side_put(_KEPT, smiles, candidate)
        logger.info("kept the only name found, which OPSIN reads as ambiguous: %r", candidate[:80])
        return candidate
    except Exception:                                           # noqa: BLE001
        return name
