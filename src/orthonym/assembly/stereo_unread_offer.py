"""The best-effort offer of a name whose stereodescriptors OPSIN cannot read (user decision 2026-11-19).

The best-effort tier ships a name only when OPSIN reads it back to the input's full InChIKey, and withdraws one
OPSIN refuses for its stereo layer alone: the lowercase ``r``/``s`` of a pseudoasymmetric centre
"Achiral cyclic compounds", '(1R,3r,5S)-tropan-3-ol') or a ring system on which OPSIN cannot assign CIP labels (the
inositols). The default tier ships such a name when the strict path built it (the stereo-layer carve-out of
``namer._final_opsin_validity_gate``: OPSIN reads the stereo-stripped name to the input's constitution, and every
descriptor is checked against the input's CIP labels); a name of any other producer was withdrawn and the molecule
was left without a name.

This offer gives such a molecule one of the names the best-effort gate withdrew for it (``default_tier_offer``
holds them), when the same carve-out accepts the name and a second labeller agrees: RDKit's ``rdCIPLabeler`` gives
every cited unit the code the centres engine gives it (``validation.second_cip_labeller``). The row is labelled
``best_effort`` (the name's own string was not read back; no PIN label), as the default tier's Chapter row of
'(1R,3s,5S)-tropan-3-ol' is (``default_tier_offer._withheld_chapter_p10_row``). It is an offer, not a return: it is
made only for a molecule every other path of the tier left without a name, so no name that ships changes."""
from typing import Dict, Optional


def stereo_unread_row(namer, smiles, held) -> Optional[Dict]:
    """The row of the first name in ``held`` (names the best-effort gate withdrew for ``smiles``) that the default
    tier's stereo-layer carve-out accepts and on whose cited units both CIP labellers agree; else None."""
    if not held or not isinstance(smiles, str) or not getattr(namer, "_general_fallback_unverified", False):
        return None
    from ..metrics import provenance as pv
    from ..metrics.abstention import peek_abstention, restore_abstention
    from .default_tier_offer import _has_stereo_layer
    kept_provenance, kept_abstention = pv.get_provenance(), peek_abstention()
    row = None
    try:
        from ..metrics.producer_trail import carried
        from ..namer import _final_opsin_validity_gate
        from ..rules.stereochemistry import strip_stereo_blocks
        from ..validation.second_cip_labeller import cited_codes_agree
        for name in held:
            if not _has_stereo_layer(name):
                continue
            pv.clear_provenance()
            pv.record_source("general_engine")
            if (carried(name, _final_opsin_validity_gate(name, smiles)) != name
                    or pv.get_provenance().get("gate_outcome") != pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY
                    or not cited_codes_agree(name, smiles, strip_stereo_blocks(name)[0])):
                continue
            got = namer._tier_row_and_pin_form(smiles, name, pv.get_provenance())[0]
            if got.get("name") == name and got.get("tier") not in (None, "abstain"):
                row = dict(got, tier="best_effort", is_pin=False)
                return row
        return None
    except Exception:                       # the offer must never break a naming: the abstention stands
        return None
    finally:
        restore_abstention(kept_abstention)
        if row is None:
            pv.restore_provenance(kept_provenance)


def unread_stereo_admitted(namer, smiles, name, gate_outcome) -> bool:
    """Whether a general tier shows ``name`` for ``smiles`` without a full read-back: at the best-effort tier, an
    inositol name, exact InChIKey, ``rules.inositols``) or a name the stereo-layer carve-out accepted for this
    string (``gate_outcome``) on whose cited units both CIP labellers agree (this module's offer)."""
    if not getattr(namer, "_general_fallback_unverified", False):
        return False
    from ..rules.inositols import is_inositol_name_of
    if is_inositol_name_of(smiles, name):
        return True
    from ..metrics import provenance as pv
    if gate_outcome != pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY:
        return False
    from ..rules.stereochemistry import strip_stereo_blocks
    from ..validation.second_cip_labeller import cited_codes_agree
    return cited_codes_agree(name, smiles, strip_stereo_blocks(name)[0])
