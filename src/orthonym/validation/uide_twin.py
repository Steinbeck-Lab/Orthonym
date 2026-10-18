"""A halogen -uide name read back through the Blue Book's method (2) spelling of the same anion.

OPSIN 2.9.0 has no grammar for a SUBSTITUTED halogen -uide token: 'diphenyliodanuide' is rejected
("unphysical valency state") although the bare 'iodanuide' reads to [IH2-]. Until this module the name
shipped with no read-back at all (the ``halogen_uide`` grammar-gap carve-out), and 111 of the 349 names
the default tier gave the small PubChem halogen anions were the name of another molecule.

 ANIONS FORMED BY ADDITION OF HYDRIDE IONS (the Blue Book) names such an anion two ways:
"(1) the suffix 'uide' describes an anion formally derived by adding a hydride ion, H-, to a parent
hydride name" (:41093); "(2) by adding the suffix 'ide' to a parent hydride in which the bonding number
at the anionic site is higher than the standard bonding number and is expressed by the lambda-convention;
the net effect of which is the addition of a hydride ion, H-, to the parent hydride with its standard
bonding number" (:41094); "Method (1) leads to preferred IUPAC names." (:41098). The book prints both
spellings of one anion: 'diphenyliodanuide (PIN) diphenyl-lambda3-iodanide' (:41110) and 'F6I-
hexafluoro-lambda5-iodanuide (preselected name) hexafluoro-lambda7-iodanide' (:41112). OPSIN reads the
method (2) spelling, to the charged structure itself.

The twin of ``<prefixes>iodanuide`` is ``<prefixes>-lambda3-iodanide``. The prefixes are the same string
and only the parent token changes, so the two spellings name one anion if the lambda number is the one
method (2) pairs with a name that carries none: the parent hydride of an unmarked -uide name has its
STANDARD bonding number s (1 for I, Br, Cl: ``rules.ions._UIDE_STD_VALENCE``), the hydride ion adds one
bond, so the anionic site holds s + 1, and method (2) reaches that anion from the parent of bonding
number s + 2 (3 for the halogens), by removing a proton. That number is read from the element of the
parent token and is not taken from the input structure: a name without a lambda describes an anion whose
centre holds s + 1 bonds, and the read-back is what decides whether the structure is one. (A number taken
from the structure would accept ``tetramethyliodanuide`` for I(CH3)4-, whose centre holds four bonds, by
adapting the twin to the structure; the unmarked name does not describe that anion.)

The twin proves the structure and never the spelling: the producer owns the spelling
(``rules.centre_ligands`` and ``mononuclear_hydrides._build_mixed_substituent_string``). A name ships
through its twin when OPSIN reads the twin to the FULL InChIKey of the input
(``namer._shipped_name_round_trip``) and gives no ``APPEARS_AMBIGUOUS`` warning for it, as the other
twin families do (``validation.twin_spellings``). The twin is never shipped.
"""
import re
from typing import Optional

from ..rules.lambda_convention import LAMBDA
from .opsin_ambiguity import ambiguity_messages

#: the rule id of the twin outcome (``twin_verified:uide_lambda``)
UIDE_TWIN_RULE = "uide_lambda"

#: a halogen -uide name: its prefixes (one string, no space) and the parent token without its 'uide'
_HALOGEN_UIDE = re.compile(r"(?P<prefixes>\S+?)(?P<stem>iodan|broman|chloran)uide")
#: the element of each parent token
_STEM_ELEMENT = {"iodan": "I", "broman": "Br", "chloran": "Cl"}


def uide_twin_outcome(outcome: Optional[str]) -> bool:
    """Whether ``outcome`` is the gate outcome of a name the best-effort tier ships although OPSIN cannot read it:
    a halogen -uide name read back through its twin (``twin_verified:uide_lambda``), a thioperoxol name read back
    through its twin (``validation.thioperoxol_twin``) -- no OPSIN grammar reads these names at any tier; the other
    twin families spell the form OPSIN reads there -- or a phane name whose structure
    ``validation.phane_read_back`` builds from the name and finds to be the input's (``carveout:phane``, recorded
    by ``namer.Orthonym._last_resort_list_name`` for such a name only: at a general tier the gate withdraws a
    phane name)."""
    from ..metrics.provenance import carveout_outcome, twin_outcome
    from .thioperoxol_twin import thioperoxol_twin_outcome
    return (outcome in (twin_outcome(UIDE_TWIN_RULE), twin_outcome("halido_o"), carveout_outcome("phane"))
            or thioperoxol_twin_outcome(outcome))


def uide_twin(name: str) -> Optional[str]:
    """The method (2) spelling of the halogen -uide name ``name``, or None when ``name`` is no such name
    (a bare 'iodanuide' has no prefixes and OPSIN reads it as it stands)."""
    found = _HALOGEN_UIDE.fullmatch(name or "")
    if found is None:
        return None
    from ..rules.ions import _UIDE_STD_VALENCE
    bonding_number = _UIDE_STD_VALENCE[_STEM_ELEMENT[found["stem"]]] + 2
    return f"{found['prefixes']}-{LAMBDA}{bonding_number}-{found['stem']}ide"


def uide_twin_round_trip(name: str, smiles: Optional[str]) -> Optional[str]:
    """``UIDE_TWIN_RULE`` when the twin of ``name`` reads to the FULL InChIKey of ``smiles`` and OPSIN reads
    it once; else None (no twin, a twin of another structure, an unavailable OPSIN, an ambiguous reading)."""
    twin = uide_twin(name)
    if twin is None or not smiles:
        return None
    from ..namer import _shipped_name_round_trip
    if _shipped_name_round_trip(twin, smiles) != "verified" or ambiguity_messages(twin):
        return None
    return UIDE_TWIN_RULE
