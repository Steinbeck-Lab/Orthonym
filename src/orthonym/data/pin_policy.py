"""Single loader for the curated IUPAC-2013 PIN allow/deny list.

``iupac_2013_pin_list.json`` is the project's adjudicated PIN authority: each row
carries a Blue Book citation and (for a deny) a verified systematic replacement.
Before Task E the loader lived inline in ``data/__init__.py`` and therefore
governed exactly one surface -- the merged retained-names dict. The amino-acid,
natural-product and trivial-acid tables never consulted it at all, which is how
``sarcosine``, ``saccharin`` and ``camphor`` reached the DEFAULT ``--style pin``
path. This module exists so every surface reads one set, not a copy.

The deny is deliberately a *global withdrawal*, matching the behaviour every
existing deny row already has: a denied name leaves the PIN-path lookup and
moves to the general-only companion dict, reachable via ``--trivial``. It is not
a per-style switch -- ``glycerol``, ``catechol`` and ``nicotinic acid`` have
emitted their systematic names under ``--style general`` since the deny list
shipped, and Task E does not introduce a second, differently-behaving mechanism.

⚠ The OPSIN ``is_pin`` field is NOT an input here. ``scripts/import_opsin_xml.py``
hard-codes ``"is_pin": False`` (lines 268, 742, 847), so all 232 entries in
``amino_acids_opsin.py`` and all 423 in ``simple_groups.py`` carry False. The
flag is a generator default with zero per-entry information; gating on it would
withdraw ``cystine`` (Blue Book Table 10.5, P-103.1.1.2) along with the junk.
Adjudication happens per row in the JSON, with a citation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, FrozenSet, List

logger = logging.getLogger(__name__)

PIN_LIST_PATH = Path(__file__).parent / "iupac_2013_pin_list.json"


def _load() -> List[Dict]:
    try:
        with open(PIN_LIST_PATH) as fh:
            return json.load(fh)["entries"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("PIN list load failed at %s: %s", PIN_LIST_PATH, exc)
        return []


ENTRIES: List[Dict] = _load()

#: Names the curated list positively affirms as preferred IUPAC names.
PIN_ALLOW: FrozenSet[str] = frozenset(
    e["name"].lower() for e in ENTRIES if e.get("pin") is True
)

#: Names the curated list denies. Applies to the OPSIN-import surfaces.
PIN_DENY: FrozenSet[str] = frozenset(
    e["name"].lower() for e in ENTRIES if e.get("pin") is False
)

#: Deny subset that ALSO filters the hand-curated dicts. Excludes ``hc_override``
#: rows (cumene/quinuclidine/acetylene), whose deny applies only to OPSIN-side
#: promotion (Phase 150 D-11 / Phase 167 HYG-03).
PIN_DENY_HC: FrozenSet[str] = frozenset(
    e["name"].lower() for e in ENTRIES
    if e.get("pin") is False and not e.get("hc_override")
)


def is_pin_denied(name: str) -> bool:
    """True iff ``name`` is an adjudicated non-PIN name (hand-curated scope).

    Used by the amino-acid, natural-product and trivial-acid surfaces, all of
    which are hand-curated or OPSIN-imported tables with no ``hc_override``
    semantics of their own -- so they take the HC-scoped set.
    """
    return name.lower().strip() in PIN_DENY_HC
