"""The classes whose name OPSIN 2.9.0 has no grammar for at all, and the carve-out each ships by.

``namer._grammar_gap_carveout`` asks ``grammar_gap_carveout``: the carve-out slug of a construction-verified name
whose CLASS OPSIN has no grammar for, or None.

For these classes a healthy OPSIN call can only answer 'rejected' (no inositol or Table 10.1 stereoparent name, no
phane grammar, no fullerene grammar), so the name ships on its construction alone (a phane name when the structure it
states is the input's, ``validation.phane_read_back``; a fullerene name when ``validation.fullerene_read_back`` builds
the input's structure from it) and OPSIN's answer adds nothing. The validity gate therefore consults this both after a
'rejected' answer and on 'unavailable': the output of these classes does not depend on whether OPSIN could be reached
(TRIAGE j12 finding 1).

NOT here: the thioperoxol names (no OPSIN grammar either; the default tier ships each through its twin read-back,
``validation.thioperoxol_twin``; a general tier ships the name OPSIN reads, and the anion of -S-OH, which has none,
through its twin), the organometallic additive names (the '-ido' names ship through their 'chloro' twin, and a name
with an eta, kappa or mu descriptor ships at no tier: no read-back, no PIN by, the stereo-layer carve-out (its
licence is a parse of the stereo-stripped form) and the polyacid carve-out (reached only after a parse).
"""
from typing import Optional


def grammar_gap_carveout(name: str, smiles: Optional[str] = None) -> Optional[str]:
    """The carve-out slug of a construction-verified name whose class OPSIN has no grammar for, or None."""
    try:
        from ..rules.inositols import INOSITOL_NAMES
        if name in INOSITOL_NAMES:
            return "inositol"
    except Exception:
        pass
    try:
        from ..data.natural_products import NAME_EXACT_NP_PARENTS
        if name in NAME_EXACT_NP_PARENTS:
            return "np_stereoparent"
    except Exception:
        pass
    from .fullerene_read_back import fullerene_gate_ships, fullerene_name_shape
    if fullerene_name_shape(name):
        # OPSIN reads no fullerene name; the name ships when the structure it states is the input's
        return "fullerene" if fullerene_gate_ships(name, smiles) else None
    from .phane_read_back import phane_name_shape, phane_read_back
    if phane_name_shape(name):
        # OPSIN has no phane grammar; the name ships when the structure it states is the input's
        return "phane" if phane_read_back(name, smiles) else None
    return None
