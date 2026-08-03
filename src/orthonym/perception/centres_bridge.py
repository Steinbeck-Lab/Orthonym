"""centres CIP-labelling engine bridge (WSB-03, Phase 177).

Adopts the `centres` reference CIP implementation (SiMolecule, LGPL-3.0)
as an OPTIONAL source-of-truth for Cahn-Ingold-Prelog stereo descriptors,
invoked out-of-process via a batched single-JVM subprocess.

Design (D-12): this module mirrors two already-trusted JVM-bridge modules
verbatim in posture --
  * ``validation/opsin_roundtrip.py``  -> jar resolution + ``_java_available``
    (5 s probe) + graceful ``None`` fallback,
  * ``validation/dual_validator.py:_parse_batch_with_opsin`` -> write all
    SMILES to ONE temp file, a SINGLE ``subprocess.run`` (argv list, never a
    shell), positional ``<labels>\\t<ID>`` parse, ``os.unlink`` in a
    ``finally``.

No new JVM-bridge logic is invented; the only centres-specific work is the
both-endpoint label parsing (centres labels C=C / C=N at BOTH 1-based atom
endpoints, e.g. ``7E 8E``; tetrahedral is a single label, e.g. ``2S``) and the
mapping of those per-atom labels back onto RDKit bond ``_CIPCode`` props (D-14).

Security posture (threat model T-177-01/02/03):
  * SMILES are passed via the temp FILE, never shell-interpolated, never on
    argv -- ``subprocess.run([...])`` with an explicit argv list, no shell
    interpretation.
  * The temp file uses ``tempfile.NamedTemporaryFile`` (OS-randomised path) and
    is ``os.unlink``-ed in a ``finally`` on EVERY path (incl. timeout / error).
  * The jar is a vendored, unmodified, LGPL-3.0 artifact resolved from
    PROJECT_ROOT (not from user input); provenance recorded in NOTICE.

This engine is OPT-IN. Callers gate on ``ORTHONYM_USE_CENTRES_CIP`` and MUST
treat a ``None`` return (jar/Java absent) as "fall back to RDKit" -- a missing
JVM never hard-fails a name (D-13).
"""

import logging
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from rdkit import Chem
from orthonym.jvm_flags import JVM_HYGIENE_FLAGS

logger = logging.getLogger(__name__)


# PERF: process-level CIP cache keyed by SMILES. centres CIP labels are a pure
# function of the molecular structure, and centres_label_mol feeds centres the
# CANONICAL SMILES (Chem.MolToSmiles), so every respelling of a molecule sends
# the IDENTICAL input — 7x per stereo probe in the determinism eval. Caching the
# {1-based-pos: descriptor} map by that SMILES turns ~2000 redundant ~1.7s JVM
# boots into cache hits. SAFE for the determinism gate: the naming logic still
# runs fresh per respelling (only identical centres runs are deduped), so a
# naming order-dependence still manifests; and the per-mol _smiles_output_order
# remap is recomputed each call, so cached labels land on the right atoms. Only
# DEFINITIVE per-SMILES results are cached (an achiral empty map included); the
# 'engine unavailable' (None) outcome is never cached.
_CENTRES_LABEL_CACHE: Dict[str, Dict[int, str]] = {}


# Project root: 4 levels up from this file
# (src/orthonym/perception/centres_bridge.py -> project root)
# Same depth / parent chain as src/orthonym/validation/opsin_roundtrip.py.
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _centres_jar_version_key(path: Path):
    """Sort key for a vendored ``centres-cli-<version>.jar`` filename.

    Parses the leading dotted-numeric version so newer jars sort higher::

        centres-cli-1.2.1.jar       -> (1, 2, 1)
        centres-cli-1.5.jar         -> (1, 5)
        centres-cli-1.5-SNAPSHOT.jar-> (1, 5)

    An unparseable name sorts lowest (``(-1,)``)."""
    m = re.match(r"centres-cli-(\d+(?:\.\d+)*)", path.name)
    if not m:
        return (-1,)
    return tuple(int(p) for p in m.group(1).split("."))


def _find_centres_jar(version: Optional[str] = None) -> Optional[str]:
    """Find the vendored centres CLI jar at the project root.

    Parallel to ``opsin_roundtrip._find_opsin_jar``.

    Args:
        version: if given, resolve that exact ``centres-cli-<version>.jar``.
            If None (the default), GLOB ``centres-cli-*.jar`` and return the
            HIGHEST version present — so a freshly-vendored newer engine jar is
            picked up with no code change (the centres-engine update path).

    Returns:
        Absolute path to the jar, or None if it is not present.
    """
    if version is not None:
        jar_path = PROJECT_ROOT / f"centres-cli-{version}.jar"
        return str(jar_path) if jar_path.exists() else None
    candidates = sorted(
        PROJECT_ROOT.glob("centres-cli-*.jar"), key=_centres_jar_version_key
    )
    return str(candidates[-1]) if candidates else None


@lru_cache(maxsize=1)
def _java_available() -> bool:
    """Check if a Java runtime is available (copied from opsin_roundtrip).

    ⚠ **CACHED, and the cache is the point.** Measured 2026-07-30 by counting
    ``subprocess`` invocations whose argv contains ``java``: this probe was spawning a
    JVM **twice per molecule** — 30 of 65 total spawns across a 15-molecule batch, i.e.
    **46% of all JVM launches during naming** — purely to re-answer "is Java
    installed?". At ~130 ms of JVM startup each that was ~260 ms per molecule of pure
    process launch, against a 2.11 mol/s whole-pipeline rate.

    Caching is safe because a Java runtime cannot appear or disappear inside one
    process, so this is **provably output-neutral**: no name can change. That matters
    here — session invariant 9 records four occasions where a change that looked like a
    pure cleanup altered what got emitted, so a performance fix has to be one that
    *cannot*.

    ``maxsize=1`` matches the established idiom in ``validation/name_morphemes.py``.
    The sibling probe in ``validation/opsin_roundtrip.py`` is cached identically.
    """
    try:
        proc = subprocess.run(
            ["java", *JVM_HYGIENE_FLAGS, "-version"],
            capture_output=True, text=True, timeout=5,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def parse_centres_labels(labels_str: str) -> Dict[int, str]:
    """Parse a centres label token list into {1-based-atom-idx: descriptor}.

    centres emits a space-separated token list per molecule, e.g.::

        "2S"        -> tetrahedral stereocentre at atom 2  -> {2: 'S'}
        "7E 8E"     -> C=C labelled at BOTH endpoints       -> {7: 'E', 8: 'E'}
        "2E 3E"     -> C=N labelled at both endpoints        -> {2: 'E', 3: 'E'}
        "CT4"       -> cumulene marker (no atom index)        -> ignored here;
                       cumulene bond resolution is handled separately.

    Tokens are ``<1-based-int><descriptor>`` where descriptor is one of
    R/S/r/s (tetrahedral / pseudoasymmetric), E/Z (cis-trans), M/P/m/p
    (axial / helical). Tokens without a leading integer (e.g. bare ``CT4``)
    carry no atom index and are skipped at this layer.

    Returns an empty dict for empty / whitespace-only input.
    """
    if not labels_str or not labels_str.strip():
        return {}

    result: Dict[int, str] = {}
    for token in labels_str.strip().split():
        # <one-or-more digits><descriptor letters>
        idx_chars = ""
        i = 0
        while i < len(token) and token[i].isdigit():
            idx_chars += token[i]
            i += 1
        if not idx_chars:
            # No leading 1-based index (e.g. "CT4") -- no per-atom mapping here.
            continue
        descriptor = token[i:]
        if descriptor:
            result[int(idx_chars)] = descriptor
    return result


def centres_label_batch(
    smiles_list: List[str], timeout: float = 120.0
) -> Optional[Dict[str, Dict[int, str]]]:
    """Label a batch of SMILES with centres in a SINGLE JVM invocation.

    Mirrors ``dual_validator._parse_batch_with_opsin``: write every SMILES to
    one temp file (``<smiles>\\t<position>`` per row), one ``subprocess.run``
    (argv list -- NO shell), parse the ``<labels>\\t<ID>`` output by the
    integer position ID, ``os.unlink`` in a ``finally``.

    Args:
        smiles_list: SMILES strings to label.
        timeout: max seconds for the whole batch JVM run.

    Returns:
        ``{smiles: {1-based-atom-idx: descriptor}}`` on success, or ``None``
        when the jar or Java is absent (caller falls back to RDKit -- D-13).
        A SMILES that centres produced no labels for maps to an empty dict.
    """
    if not smiles_list:
        return {}

    # PERF: serve cached SMILES; only spawn a JVM for the uncached remainder.
    out: Dict[str, Dict[int, str]] = {}
    uncached: List[str] = []
    for smi in smiles_list:
        if smi in _CENTRES_LABEL_CACHE:
            out[smi] = _CENTRES_LABEL_CACHE[smi]
        else:
            uncached.append(smi)
    if not uncached:
        return out  # every SMILES served from cache -> no JVM, no jar probe

    jar = _find_centres_jar()
    # PERF: the in-process JVM (jvm_bridge, JPype) makes the `_java_available`
    # probe irrelevant -- there is no `java` binary to find, only a JVM already
    # running in this process -- so only pay for that probe on the subprocess
    # path. Ordering matters: asking `_java_available()` first would spawn a JVM
    # to answer a question the in-process path does not ask.
    try:
        from ..jvm_bridge import centres_available, centres_stdout
        _inproc = jar is not None and centres_available()
    except ImportError:  # pragma: no cover - jvm_bridge always present
        centres_stdout = None  # type: ignore[assignment]
        _inproc = False
    if jar is None or not (_inproc or _java_available()):
        # Engine unavailable -- signal the caller to fall back to RDKit.
        return None

    temp_input: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".smi", delete=False, encoding="utf-8"
        ) as f:
            for i, smi in enumerate(uncached):
                # ID = integer position so we can recover the exact input SMILES.
                f.write(f"{smi}\t{i}\n")
            temp_input = f.name

        # argv list, NEVER a shell; SMILES live in the temp file, not on argv.
        # This measured 30 of the 37 remaining JVM spawns (2026-07-30): a
        # function named "batch" that production calls twice per molecule with
        # ONE molecule each, so ~130 ms of process launch bought nothing. The
        # in-process path drives LabelCip.main with the IDENTICAL argv and the
        # IDENTICAL temp file (byte-identical by construction, verified over 508
        # SMILES) at 0.8 ms/call. The temp-file security posture
        # (T-177-01/02/03) is unchanged: SMILES still never touch argv.
        stdout_text: Optional[str] = None
        if _inproc and centres_stdout is not None:
            stdout_text = centres_stdout(["-i", "smi", temp_input])
        if stdout_text is None:
            # In-process unavailable or Java raised -> the original subprocess.
            if not _java_available():
                return None
            cmd = ["java", *JVM_HYGIENE_FLAGS, "-jar", jar, "-i", "smi", temp_input]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            stdout_text = proc.stdout

        for line in stdout_text.split("\n") if stdout_text else []:
            if "\t" not in line:
                continue
            labels_str, idx_str = line.rsplit("\t", 1)
            idx_str = idx_str.strip()
            if not idx_str.isdigit():
                continue
            pos = int(idx_str)
            if 0 <= pos < len(uncached):
                labels = parse_centres_labels(labels_str)
                smi = uncached[pos]
                out[smi] = labels
                # Cache ONLY a SMILES centres actually emitted a line for (a
                # definitive per-structure result; an achiral molecule yields an
                # empty-labels line, so this includes achiral {}). A SMILES with
                # NO line is a per-SMILES skip/failure — left absent (matching
                # the original .get(smi, {}) contract) and NOT cached, so it is
                # retried rather than pinned to a possibly-wrong empty map.
                _CENTRES_LABEL_CACHE[smi] = labels
        return out

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as exc:
        logger.warning("centres batch failed (%s); caller falls back to RDKit", exc)
        return None
    finally:
        if temp_input is not None:
            try:
                os.unlink(temp_input)
            except OSError:
                pass


def apply_centres_labels(mol, label_map: Dict[int, str]) -> None:
    """Set RDKit ``_CIPCode`` props on atoms/bonds from a centres label map.

    Resolution (D-14):
      * Tetrahedral / pseudoasymmetric (R/S/r/s) -- single labelled atom:
        ``mol.GetAtomWithIdx(idx-1).SetProp('_CIPCode', desc)``.
      * Cis-trans (E/Z) -- centres labels BOTH endpoints of the double bond
        with the same descriptor; convert each 1-based index to 0-based, find
        the DOUBLE bond between the two consecutive same-descriptor atoms, and
        set its ``_CIPCode``. Atoms participating in multiple double bonds
        (dienes) are handled by matching the specific labelled pair; setting
        both endpoints' descriptor onto the one bond is idempotent.

    Axial / helical (M/P/m/p) descriptors centres emits for AT/HE compounds are
    set on the atom directly (RDKit's downstream consumers read ``_CIPCode``
    generically). Labels whose target cannot be resolved on this mol are skipped
    (D-09: a missing descriptor beats a wrong one).

    The map uses 1-based atom indices (centres convention). ``mol`` is modified
    in place. This is the production analogue of the validation harness keying.
    """
    if not label_map:
        return

    n_atoms = mol.GetNumAtoms()

    # Partition into per-atom (tetrahedral/axial) and per-bond (E/Z) descriptors.
    ez_atoms: Dict[int, str] = {}     # 0-based atom idx -> E/Z descriptor
    for one_based, desc in label_map.items():
        zero_based = one_based - 1
        if not (0 <= zero_based < n_atoms):
            continue
        if desc in ("E", "Z"):
            ez_atoms[zero_based] = desc
        else:
            # Tetrahedral (R/S/r/s) or axial/helical (M/P/m/p) -> on the atom.
            mol.GetAtomWithIdx(zero_based).SetProp("_CIPCode", desc)

    if not ez_atoms:
        return

    # Map the both-endpoint E/Z atom labels onto the actual DOUBLE bonds.
    # For each DOUBLE bond whose BOTH endpoints carry the same E/Z descriptor,
    # set that descriptor on the bond. This naturally selects the specific
    # double bond for a diene (only the labelled pair's bond matches) and is
    # idempotent across the two endpoint labels of one bond.
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        da = ez_atoms.get(a)
        db = ez_atoms.get(b)
        if da is not None and db is not None and da == db:
            bond.SetProp("_CIPCode", da)


def _smiles_output_order(mol) -> Optional[List[int]]:
    """Return RDKit's atom output order for the last ``MolToSmiles(mol)`` call.

    ``Chem.MolToSmiles`` writes atoms in CANONICAL rank order (not the mol's own
    atom-index order) and records the mapping as the ``_smilesAtomOutputOrder``
    computed property: ``order[pos]`` is the ORIGINAL atom index written at
    output SMILES position ``pos``. centres parses the SMILES STRING, so its
    1-based labels are keyed to those output positions, not to ``mol``'s atom
    indices -- this list is what lets us map them back (STER-02 / Phase H).

    Returns the parsed order, or None when the property is absent/malformed
    (caller then declines the centres path -- a missing label beats a wrong one).
    """
    try:
        raw = mol.GetProp("_smilesAtomOutputOrder")
    except KeyError:
        return None
    try:
        order = [int(x) for x in raw.strip().strip("[]").split(",") if x.strip() != ""]
    except ValueError:
        return None
    return order or None


def centres_label_mol(mol) -> bool:
    """Label a single RDKit mol with centres (gated single-mol convenience).

    Canonicalises the mol to SMILES, runs the batch path over the one SMILES,
    and applies the returned labels onto ``mol``'s atoms/bonds via
    ``apply_centres_labels``.

    CRITICAL (STER-02 / Phase H): centres labels are 1-based atom indices keyed
    to the CANONICAL SMILES string emitted by ``Chem.MolToSmiles``, whose atom
    output order generally DIFFERS from ``mol``'s own atom indices. Applying the
    labels directly by ``idx-1`` (the pre-Phase-H behaviour) lands descriptors on
    the WRONG atoms whenever canonicalisation reorders (e.g. 4-hydroxyproline,
    tartaric acid) -- a wrong/malformed descriptor, which Phase H forbids. We
    remap each centres position through ``_smilesAtomOutputOrder`` back to the
    original atom index before applying. The validation harness
    (``score_suite_centres``) is unaffected: it sends the suite's ORIGINAL SMILES
    and never round-trips through canonicalisation.

    Returns:
        True if centres produced a label map and it was applied (the engine
        was available); False when the caller must fall back to RDKit (D-13).
        False has TWO causes: (1) the engine was unavailable (jar/Java absent),
        or (2) the engine ran but the `_smilesAtomOutputOrder` remap could not
        be read while there were labels to place — declining beats misplacing
        them. An AVAILABLE engine returning an empty map (an achiral molecule)
        still returns True: centres ran, there simply were no descriptors.
    """
    try:
        smi = Chem.MolToSmiles(mol)
    except (ValueError, RuntimeError):
        return False
    # Must read the output order from the SAME MolToSmiles call above.
    order = _smiles_output_order(mol)
    batch = centres_label_batch([smi])
    if batch is None:
        return False
    raw_labels = batch.get(smi, {})  # {canonical_1based_pos: descriptor}
    if order is None:
        # No output-order map available (should not happen for a real
        # MolToSmiles). If there are labels to place we cannot safely remap
        # them, so decline -> the caller falls back to RDKit (D-09: a missing
        # descriptor beats a wrong one). If there are none, centres ran cleanly.
        return not raw_labels
    # Remap canonical SMILES positions -> mol's own 1-based atom indices.
    remapped: Dict[int, str] = {}
    for pos1, desc in raw_labels.items():
        pos0 = pos1 - 1
        if 0 <= pos0 < len(order):
            remapped[order[pos0] + 1] = desc
    apply_centres_labels(mol, remapped)
    return True
