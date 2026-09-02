""": Group-split topology-table loader.

Loads ``group_split_rules.json`` — the locked, TOPOLOGY-ONLY split-decomposition
table (CONTEXT) — into a ``Dict[str, SplitRule]`` keyed by ``fg_name``.

The split fires at the ``polyfunctional.py:get_fg_prefix_form()`` /
``substituent_no_prefix_form`` site (CONTEXT /F2 — a coarse string path,
NOT an IR tree visitor). When a
non-principal composite functional group has no clean strict-IUPAC prefix and
would otherwise be dropped, the splitter decomposes it into its
component sub-prefixes per the topology recorded here.

**Single source of truth (CONTEXT):** this table records ONLY the
decomposition TOPOLOGY — which sub-fragments (chalcogen / heteroatom linker) the
composite splits into, and a ``resolves_via`` pointer naming the existing
``fg_name`` / dispatcher key whose ``seniority.PREFIX_FORMS`` /
``assembly.substituent_prefix_forms`` entry supplies the prefix STRING at
runtime. The prefix output strings (``oxo``, the alkoxy/sulfanyl forms,...) are
NEVER stored here — duplicating them would fork the authority and invite drift.

Frozen-dataclass discipline mirrors SACRED + 's
``triviality_controller_seed.py``: ``SplitRule`` / ``SplitComponent`` are
``@dataclass(frozen=True)`` and are never mutated after construction.

Graceful degradation (PATTERNS correction): a missing JSON or schema error
degrades ``SPLIT_RULES`` to ``{}`` so the splitter sees no rules and
every ``substituent_no_prefix_form`` still drops (status quo) — never a crash. This is the
no-crash invariant.

Self-contained loader (PATTERNS NOTE): the Phase-168 seed precedent uses its OWN
module, NOT a ``data/__init__.py`` merge — this module mirrors that path and does
not touch ``data/__init__.py``.

Source: 169-CONTEXT.md,,,; 169-RESEARCH.md section "The #1
Gate" + "Code Examples"; 169-PATTERNS.md "data/group_split_rules.json + loader".
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import rdkit
from orthonym.jvm_flags import JVM_HYGIENE_FLAGS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitComponent:
    """One sub-fragment a composite FG decomposes into (TOPOLOGY ONLY).

    ``resolves_via`` names the existing ``fg_name`` / dispatcher key whose
    PREFIX_FORMS / substituent_prefix_forms entry supplies the prefix STRING at
    runtime — the string itself is never stored here (CONTEXT).
    """

    role: str                      # "chalcogen" | "linker"
    resolves_via: str              # dispatcher key (e.g. "ketone", "ether", "thioether")
    bond: Optional[str] = None     # e.g. "=O" for a chalcogen double bond
    atom: Optional[str] = None     # e.g. "O" / "S" for a heteroatom linker


@dataclass(frozen=True)
class SplitRule:
    """One locked split-decomposition entry, keyed by ``fg_name`` (CONTEXT).

    Immutable by SACRED discipline. Holds ONLY topology +
    provenance — no prefix output strings (CONTEXT).
    """

    fg_name: str
    composite_smarts: str
    components: Tuple[SplitComponent, ...]
    encoding: str
    iupac_p_section: str
    opsin_rt_verified_at: str
    notes: str


def load_split_rules(json_path: Path, *, validate: bool = False) -> Dict[str, SplitRule]:
    """Load the topology table into an ``fg_name``-keyed dict.

    Warns (does not fail) if the table's ``rdkit_version_pin`` differs from the
    running RDKit — re-run ` --rt`` to
    re-confirm round-trips (mirrors R-10 warning).

    With ``validate=True`` additionally runs the CONTEXT design-time
    OPSIN-RT re-confirmation of each entry's documented example (off by default —
    runs at CI lint time, not at every import). Raises ``ValueError`` on the
    first entry that fails to round-trip. Raises the JSON-load errors on a
    malformed file (caught by the module-load wrapper below).
    """
    with open(json_path) as f:
        data = json.load(f)

    pinned = data.get("rdkit_version_pin")
    if pinned and pinned != rdkit.__version__:
        logger.warning(
            " split table was OPSIN-RT-verified against rdkit %s but "
            "this process runs rdkit %s; re-run "
            "--rt to re-confirm round-trips.",
            pinned, rdkit.__version__,
        )

    rules: Dict[str, SplitRule] = {}
    for raw in data["entries"]:
        components = tuple(
            SplitComponent(
                role=c["role"],
                resolves_via=c["resolves_via"],
                bond=c.get("bond"),
                atom=c.get("atom"),
            )
            for c in raw["components"]
        )
        rule = SplitRule(
            fg_name=raw["fg_name"],
            composite_smarts=raw["composite_smarts"],
            components=components,
            encoding=raw["encoding"],
            iupac_p_section=raw["iupac_p_section"],
            opsin_rt_verified_at=raw["opsin_rt_verified_at"],
            notes=raw.get("notes", ""),
        )
        rules[rule.fg_name] = rule

    if validate:
        _opsin_rt_validate_examples(data["entries"])

    return rules


def _opsin_rt_validate_examples(entries) -> None:
    """CONTEXT design-time OPSIN-RT re-confirmation (opt-in; slow).

    Parses each entry's documented example name through OPSIN ``-osmi`` and
    confirms it is non-empty. The connectivity-InChI equality check the full
    audit uses lives in ``; this lightweight gate
    just asserts OPSIN accepts the documented form. Skips gracefully when java /
    the jar are unavailable (mirrors the seed loader's graceful-fallback).
    """
    import shutil

    if shutil.which("java") is None:
        logger.warning("java not on PATH; skipping OPSIN-RT re-confirmation.")
        return
    jar = Path(__file__).resolve().parents[3] / "opsin-cli-2.9.0-jar-with-dependencies.jar"
    if not jar.exists():
        logger.warning("OPSIN jar not found; skipping OPSIN-RT re-confirmation.")
        return
    for raw in entries:
        # Pull the documented example name out of the notes ("-> <name> (").
        note = raw.get("notes", "")
        if "->" not in note:
            continue
        example = note.split("->", 1)[1].split("(")[0].strip().rstrip(".")
        if not example:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
            tf.write(example + "\n")
            path = tf.name
        try:
            out = subprocess.run(["java", *JVM_HYGIENE_FLAGS, "-jar", str(jar), "-osmi", path],
                                 capture_output=True, text=True, timeout=20)
            if not out.stdout.strip():
                raise ValueError(
                    f"Split rule {raw['fg_name']!r} documented example {example!r} "
                    f"failed OPSIN-RT (no SMILES from OPSIN -osmi)."
                )
        finally:
            os.unlink(path)


_RULES_PATH = Path(__file__).parent / "group_split_rules.json"
try:
    SPLIT_RULES: Dict[str, SplitRule] = load_split_rules(_RULES_PATH, validate=False)
except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
    logger.error(
        " split-table load failed: %s; group-splitting will be a no-op", e,
    )
    SPLIT_RULES = {}
