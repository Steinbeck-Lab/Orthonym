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
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from rdkit import Chem

logger = logging.getLogger(__name__)


# Project root: 4 levels up from this file
# (src/orthonym/perception/centres_bridge.py -> project root)
# Same depth / parent chain as src/orthonym/validation/opsin_roundtrip.py.
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _find_centres_jar(version: str = "1.2.1") -> Optional[str]:
    """Find the vendored centres CLI jar at the project root.

    Parallel to ``opsin_roundtrip._find_opsin_jar``.

    Args:
        version: centres version string (default "1.2.1").

    Returns:
        Absolute path to the jar, or None if it is not present.
    """
    jar_name = f"centres-cli-{version}.jar"
    jar_path = PROJECT_ROOT / jar_name
    if jar_path.exists():
        return str(jar_path)
    return None


def _java_available() -> bool:
    """Check if a Java runtime is available (copied from opsin_roundtrip)."""
    try:
        proc = subprocess.run(
            ["java", "-version"],
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

    jar = _find_centres_jar()
    if jar is None or not _java_available():
        # Engine unavailable -- signal the caller to fall back to RDKit.
        return None

    temp_input: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".smi", delete=False, encoding="utf-8"
        ) as f:
            for i, smi in enumerate(smiles_list):
                # ID = integer position so we can recover the exact input SMILES.
                f.write(f"{smi}\t{i}\n")
            temp_input = f.name

        # argv list, NEVER a shell; SMILES live in the temp file, not on argv.
        cmd = ["java", "-jar", jar, "-i", "smi", temp_input]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )

        out: Dict[str, Dict[int, str]] = {}
        for line in proc.stdout.split("\n") if proc.stdout else []:
            if "\t" not in line:
                continue
            labels_str, idx_str = line.rsplit("\t", 1)
            idx_str = idx_str.strip()
            if not idx_str.isdigit():
                continue
            pos = int(idx_str)
            if 0 <= pos < len(smiles_list):
                out[smiles_list[pos]] = parse_centres_labels(labels_str)
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
