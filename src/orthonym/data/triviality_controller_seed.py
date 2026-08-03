"""Phase 168 Plan-01: Triviality-controller seed-table loader.

Loads ``triviality_controller_seed.json`` (the locked PIN-authority seed per
CONTEXT D-03) into a ``Dict[str, SeedEntry]`` keyed by canonical SMILES, with:

* the Phase 150 ``_PIN_DENY`` gate enforced at LOAD time (CONTEXT D-11 —
  deny-list-by-data; a deny-listed name in the JSON is a hard load error),
* canonicalization-idempotence enforced per entry (CONTEXT D-13 —
  ``Chem.CanonSmiles`` is the single source of truth for the match key),
* an optional design-time OPSIN L1 round-trip pre-validator (CONTEXT D-07 T1;
  ``validate=True`` / ` --rt``).

Frozen-dataclass discipline mirrors Phase 165 D-04 SACRED: ``SeedEntry`` is
``@dataclass(frozen=True)`` and is never mutated after construction.

Graceful degradation (Phase 150 WR-06): a missing JSON or schema error degrades
``SEED_TABLE`` to ``{}`` so the downstream controller becomes a no-op, never a
crash.

Source: 168-CONTEXT.md D-03, D-07, D-11, D-13; 168-RESEARCH.md section 5.4;
168-PATTERNS.md "NEW: src/orthonym/data/triviality_controller_seed.py".
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

from rdkit import Chem
import rdkit
from orthonym.jvm_flags import JVM_HYGIENE_FLAGS

logger = logging.getLogger(__name__)


class SubstitutionType(str, Enum):
    """IUPAC 2013 P-15.1.8.1-.3 retained-name substitution Types (CONTEXT D-02).

    * ``TYPE_1``  — P-15.1.8.1: unlimited substitution (benzene, pyridine, ...).
    * ``TYPE_2A`` — P-15.1.8.2.1: substitution requires the senior group be
      expressed; principal-group-bound (phenol, aniline, benzoic acid, ...).
    * ``TYPE_2B`` — P-15.1.8.2.2: closed compulsory-prefix-only list
      (formic acid + halogen/nitro/nitroso/alkoxy).
    * ``TYPE_2C`` — P-15.1.8.2.3: per-name-specific permission (anisole,
      hydroxylamine); defaults to Type 3 unless a locus override is set.
    * ``TYPE_3``  — P-15.1.8.3: no substitution except functionalisation
      (toluene, the xylene isomers).
    """

    TYPE_1 = "type_1"
    TYPE_2A = "type_2a"
    TYPE_2B = "type_2b"
    TYPE_2C = "type_2c"
    TYPE_3 = "type_3"


@dataclass(frozen=True)
class SeedEntry:
    """One locked seed-table entry (CONTEXT D-03 + D-13).

    Immutable by Phase 165 D-04 SACRED discipline. ``canonical_smiles`` is the
    ``Chem.CanonSmiles``-stable match key; ``retained_pin_name`` is the
    substitution target the controller emits.
    """

    canonical_smiles: str
    retained_pin_name: str
    substitution_type: SubstitutionType
    iupac_p_section: str
    locant_context: Optional[Tuple[int, ...]]
    compulsory_prefix_smarts: Optional[Tuple[str, ...]]
    principal_group_required: Optional[str]
    locus_override_rule_id: Optional[str]
    stereo_in_key: bool
    notes: str
    opsin_rt_verified_at: str


def load_seed_table(json_path: Path, *, validate: bool = False) -> Dict[str, SeedEntry]:
    """Load + validate the seed JSON into a canonical-SMILES-keyed dict.

    Enforces the D-11 deny gate and the D-13 canonicalization-idempotence gate at
    load. With ``validate=True`` additionally runs the D-07 T1 OPSIN-RT
    design-time pre-validator (off by default — runs at CI lint time, not at
    every import).

    Raises ``ValueError`` on a deny-list leak, a non-idempotent canonical SMILES,
    or (when ``validate=True``) an OPSIN-RT failure. Raises the JSON-load errors
    on a malformed file (caught by the module-load wrapper below).
    """
    with open(json_path) as f:
        data = json.load(f)

    pinned = data.get("rdkit_version_pin")
    if pinned and pinned != rdkit.__version__:
        logger.warning(
            "Phase 168 seed table was OPSIN-RT-validated against rdkit %s but "
            "this process runs rdkit %s; re-run  "
            "--rt to re-confirm round-trips (R-10).",
            pinned, rdkit.__version__,
        )

    # Pattern S3 lazy-import: data/__init__.py may not be fully initialised when
    # this module is imported during package init, and Plan-02 wires the two
    # together — importing _PIN_DENY at module top would risk a circular import.
    from orthonym.data import _PIN_DENY  # noqa: PLC0415

    entries: Dict[str, SeedEntry] = {}
    for raw in data["entries"]:
        name = raw["retained_pin_name"]
        name_lower = name.lower().strip()
        if name_lower in _PIN_DENY:
            raise ValueError(
                f"Seed table contains deny-listed name {name!r}: violates "
                f"Phase 168 D-11 (deny-list-by-data). Remove from seed JSON."
            )

        stored = raw["canonical_smiles"]
        canon = Chem.CanonSmiles(stored)
        if canon != stored:
            raise ValueError(
                f"Seed entry {name!r} canonical_smiles is not Chem.CanonSmiles-stable "
                f"(D-13 idempotence): stored={stored!r} canon={canon!r}. Re-canonicalise."
            )

        loc = raw.get("locant_context")
        smarts = raw.get("compulsory_prefix_smarts")
        entry = SeedEntry(
            canonical_smiles=canon,
            retained_pin_name=name,
            substitution_type=SubstitutionType(raw["substitution_type"]),
            iupac_p_section=raw["iupac_p_section"],
            locant_context=tuple(loc) if loc is not None else None,
            compulsory_prefix_smarts=tuple(smarts) if smarts is not None else None,
            principal_group_required=raw.get("principal_group_required"),
            locus_override_rule_id=raw.get("locus_override_rule_id"),
            stereo_in_key=bool(raw.get("stereo_in_key", False)),
            notes=raw.get("notes", ""),
            opsin_rt_verified_at=raw["opsin_rt_verified_at"],
        )
        entries[canon] = entry

    if validate:
        _t1_opsin_rt_validate_all(entries)

    return entries


def _t1_opsin_rt_validate_all(entries: Dict[str, SeedEntry]) -> None:
    """CONTEXT D-07 T1 design-time OPSIN L1 round-trip pre-validator.

    Reuses ``find_opsin_jar`` + ``smiles_match_via_inchi`` from
    `` verbatim (Phase 150 oracle). Raises
    ``ValueError`` on the first entry that fails to round-trip.

    Attachment-point (P-29.6.1 substituent-prefix) entries — whose canonical
    SMILES carries a dummy atom ``*`` — are SKIPPED: OPSIN's molecule parser
    legitimately returns no SMILES for a bare substituent fragment (empirically
    confirmed at audit: ``java -jar opsin -osmi`` returns empty for phenyl /
    benzyl / methylene / benzylidene / benzylidyne / 1,4-phenylene). Those
    entries are validated by their P-29.6.1 Blue Book citation + RDKit
    canonicalization-idempotence instead (disclosed in 168-AUDIT-TRIV.md per
    honest-fail-on-data). This is not a band-aid: the OPSIN-RT oracle is defined
    over complete molecules, and a substituent prefix is not one.
    """
    helpers = _load_validator_helpers()
    if helpers is None:
        logger.warning(
            " helpers unavailable; skipping "
            "T1 OPSIN-RT pre-validation (RESEARCH section 3.8 graceful fallback)."
        )
        return
    find_opsin_jar, smiles_match_via_inchi = helpers

    if shutil.which("java") is None:
        logger.warning(
            "java not found on PATH; skipping T1 OPSIN-RT pre-validation "
            "(RESEARCH section 3.8 graceful fallback)."
        )
        return

    opsin_jar = find_opsin_jar()
    if not opsin_jar:
        logger.warning(
            "OPSIN jar not found in cwd; skipping T1 OPSIN-RT pre-validation "
            "(RESEARCH section 3.8 graceful fallback)."
        )
        return

    for entry in entries.values():
        if "*" in entry.canonical_smiles:
            # P-29.6.1 substituent prefix — OPSIN molecule round-trip N/A.
            continue
        result = subprocess.run(
            ["java", *JVM_HYGIENE_FLAGS, "-jar", opsin_jar, "-osmi"],
            input=entry.retained_pin_name + "\n",
            capture_output=True,
            text=True,
            timeout=20,
        )
        opsin_smiles = result.stdout.strip()
        if not opsin_smiles:
            raise ValueError(
                f"Seed entry {entry.retained_pin_name!r} fails OPSIN-RT "
                f"(no SMILES output from OPSIN -osmi)."
            )
        if not smiles_match_via_inchi(opsin_smiles, entry.canonical_smiles):
            raise ValueError(
                f"Seed entry {entry.retained_pin_name!r} fails OPSIN-RT: "
                f"opsin_output={opsin_smiles!r}, expected_canon={entry.canonical_smiles!r}."
            )


def _load_validator_helpers():
    """Lazily import the Phase 150 OPSIN-RT helpers from `the project tooling`.

    Returns ``(find_opsin_jar, smiles_match_via_inchi)`` or ``None`` if the
    script (or its imports) cannot be loaded — the caller then skips validation
    gracefully.
    """
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    try:
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from validate_retained_names import (  # noqa: PLC0415
            find_opsin_jar,
            smiles_match_via_inchi,
        )
    except Exception as exc:  # pragma: no cover - import-environment dependent
        logger.warning("Could not import validate_retained_names helpers: %s", exc)
        return None
    return find_opsin_jar, smiles_match_via_inchi


_SEED_PATH = Path(__file__).parent / "triviality_controller_seed.json"
try:
    SEED_TABLE: Dict[str, SeedEntry] = load_seed_table(_SEED_PATH, validate=False)
except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
    logger.error(
        "Phase 168 seed-table load failed: %s; triviality controller will be a no-op", e,
    )
    SEED_TABLE = {}
