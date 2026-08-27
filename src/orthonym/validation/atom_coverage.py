"""
Atom coverage validator for Orthonym.

Validates how completely a generated IUPAC name describes the input molecule
by parsing the name back to a structure (OPSIN) and comparing that structure
to the input.

v29 residue, Task X -- this module used to compare heavy-atom COUNTS, which
made it unsound in three separate ways, all measured:

  * ``ratio = min(parsed_heavy / total_heavy, 1.0)`` clamped away atom GAIN,
    so a name that INVENTS atoms could never score below 1.0
    (``CNCC(=O)N`` -> ``2-amino-2-(methylamino)acetamide`` scored 1.000);
  * ``is_complete = ratio >= 0.80`` passed a genuine atom DROP
    (``CS(=O)(=O)NC`` -> ``methanesulfonamide`` scored 0.833 -> complete);
  * a count cannot distinguish *the same atoms* from *the same number of
    atoms*.  ``CNCC(=O)O`` (sarcosine) named ``2-aminopropanoic acid``
    (alanine) matches on heavy count AND on molecular formula AND on the
    heavy-atom element multiset, and is a different molecule.

``is_complete`` is therefore decided by CONSTITUTION -- the InChIKey skeleton
block of the input compared with that of the parse-back -- and never by a
threshold on a count.  The counts survive only as diagnostics, and atom gain
is reported explicitly as ``extra_atoms``.

DECLARED SCOPE.  The skeleton block fixes the molecular formula, the
connectivity and the hydrogen layer.  It does NOT fix stereochemistry,
isotopes or net charge, which live in the second InChIKey block; a
stereo-blind name still covers every atom and is reported complete here.
Both full InChIKeys are exposed on the result so a caller needing the
stricter comparison can make it without re-parsing.  This module answers
"are these the same atoms, bonded the same way" and nothing more.

This is a diagnostic tool -- it reports coverage but never blocks naming
output.  The load-bearing in-process no-silent-atom-drop gate is the E1
atom->token partition certificate in ``validation/e1_certificate.py``; E1
needs a ``GeneralEngineResult`` and so cannot serve callers that hold only
(mol, name), which is what this module is for.

Two validation approaches:
  1. Parse-back (OPSIN): parse name via OPSIN JAR, compare constitution.
  2. Fallback: if OPSIN unavailable or parse fails, return 'unavailable'
     (ratio 0.0, incomplete) -- fail-closed by design.
"""

import glob as glob_mod
import os
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Set

from rdkit import Chem
from orthonym.jvm_flags import JVM_HYGIENE_FLAGS


@dataclass
class CoverageResult:
    """Result of atom coverage validation.

    Attributes:
        total_heavy_atoms: Heavy atom count of the input molecule.
        claimed_atoms: Input heavy atoms with a counterpart in the parse-back,
            counted as the size of the element-multiset intersection.
        unclaimed_atoms: Input heavy atoms with NO counterpart (atoms the name
            dropped) = total_heavy_atoms - claimed_atoms.
        coverage_ratio: claimed / total (0.0 to 1.0).  Diagnostic only -- it
            is NOT what decides ``is_complete``, and by construction it cannot
            see atom gain (see ``extra_atoms``) or a rearrangement (see
            ``constitution_match``).
        is_complete: True only when the parse-back has the SAME CONSTITUTION
            as the input, i.e. ``constitution_match``.  Never a threshold on
            a count.
        method: 'parse_back', 'trivial', 'parse_back_no_inchi', or
            'unavailable'.
        claimed_atom_indices: Indices of atoms accounted for.  Empty on the
            parse-back path, which compares whole structures rather than
            mapping atom to atom.
        unclaimed_atom_indices: Indices of atoms NOT accounted for.  Empty on
            the parse-back path, as above.
        extra_atoms: Heavy atoms present in the parse-back that the INPUT does
            not have -- atoms the name invented.  Zero for a faithful name.
        parsed_heavy_atoms: Raw heavy-atom count of the parse-back, unclamped,
            so a caller can recover the true count ratio (which may exceed 1).
        constitution_match: InChIKey skeleton block of input == that of the
            parse-back.  Fixes formula, connectivity and H layer; does not
            fix stereo, isotopes or charge (module docstring, DECLARED SCOPE).
        input_inchikey: Full InChIKey of the input molecule ('' if unavailable).
        parsed_inchikey: Full InChIKey of the parse-back ('' if unavailable).
    """

    total_heavy_atoms: int
    claimed_atoms: int
    unclaimed_atoms: int
    coverage_ratio: float
    is_complete: bool
    method: str
    claimed_atom_indices: Set[int] = field(default_factory=set)
    unclaimed_atom_indices: Set[int] = field(default_factory=set)
    extra_atoms: int = 0
    parsed_heavy_atoms: int = 0
    constitution_match: bool = False
    input_inchikey: str = ""
    parsed_inchikey: str = ""


def find_opsin_jar() -> Optional[str]:
    """Find OPSIN JAR file in project root or standard locations.

    Uses the same search patterns as ``.
    """
    # Determine project root (go up from src/orthonym/validation/)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))

    search_patterns = [
        os.path.join(project_root, "opsin-cli-*-jar-with-dependencies.jar"),
        os.path.join(project_root, "opsin-cli-*.jar"),
        os.path.join(project_root, "opsin.jar"),
        os.path.join(
            project_root,
            "opsin",
            "opsin-cli",
            "target",
            "opsin-cli-*-jar-with-dependencies.jar",
        ),
    ]

    for pattern in search_patterns:
        matches = glob_mod.glob(pattern)
        if matches:
            return matches[0]
    return None


# v33 giant-molecule hang fix: memoize OPSIN parse results. Parsing a name to a
# structure is a PURE, DETERMINISTIC function of (name, jar), so a process-global
# cache is always correct AND improves determinism (a name that times out once is
# thereafter a fast, consistent miss instead of a load-dependent flake). This is
# the lever that collapses the giant-molecule cost: the decomposition can generate
# the SAME candidate name dozens of times, but OPSIN now runs on each distinct
# name at most once instead of paying a fresh (up to 30 s) subprocess every time.
_OPSIN_PARSE_CACHE = {}
_OPSIN_PARSE_CACHE_MAX = 16384


def _parse_name_with_opsin(name: str, opsin_jar: str) -> Optional[str]:
    """Memoizing wrapper over :func:`_parse_name_with_opsin_uncached`.

    See the note above ``_OPSIN_PARSE_CACHE`` for why a process-global memo is
    sound. Bounded to ``_OPSIN_PARSE_CACHE_MAX`` distinct names; once full it stops
    inserting (never evicts) so a pathological batch cannot grow it without bound.
    """
    if not name or not opsin_jar:
        return None
    key = (name, opsin_jar)
    cached = _OPSIN_PARSE_CACHE.get(key, _CACHE_MISS)
    if cached is not _CACHE_MISS:
        return cached
    result = _parse_name_with_opsin_uncached(name, opsin_jar)
    if len(_OPSIN_PARSE_CACHE) < _OPSIN_PARSE_CACHE_MAX:
        _OPSIN_PARSE_CACHE[key] = result
    return result


_CACHE_MISS = object()


def _parse_name_with_opsin_uncached(name: str, opsin_jar: str) -> Optional[str]:
    """Parse a single IUPAC name to SMILES using OPSIN JAR.

    Mirrors the batch approach from ``
    (stdin/stdout mode with a single name written to a temp file).

    Args:
        name: IUPAC name to parse.
        opsin_jar: Path to OPSIN CLI JAR file.

    Returns:
        Canonical SMILES string if parse succeeds, else ``None``.
    """
    if not name or not opsin_jar:
        return None

    # PERF (v38 P1): prefer the ONE lazily-started in-process JVM (jvm_bridge,
    # JPype) over a fresh ~0.82 s cold `java -jar` launch per name -- that
    # cold-start dominated atom-coverage wall time. ``opsin_stdout`` returns
    # EXACTLY the bytes this ``java -jar ... -osmi`` invocation would have
    # written to stdout, so the parsing below is SHARED verbatim between the two
    # sources and cannot drift. ``served`` is False whenever the in-process path
    # cannot serve the call (jpype absent, jar unresolvable or a different OPSIN
    # version than the JVM was started with, an embedded newline, a Java-side
    # error) -- we then fall through to the subprocess exactly as before, so
    # correctness never depends on the optimization. The final structure is
    # canonicalized through RDKit either way, so a name that parses returns a
    # byte-identical canonical SMILES whether OPSIN ran warm or cold.
    stdout_text: Optional[str] = None
    served = False
    try:
        from ..jvm_bridge import opsin_stdout
        stdout_text, served = opsin_stdout(name, allow_radicals=False,
                                           jar_path=opsin_jar)
    except ImportError:  # pragma: no cover - jvm_bridge always present
        served = False

    if not served:
        stdout_text = _opsin_cli_stdout(name, opsin_jar)
        if stdout_text is None:
            return None

    stdout_lines = stdout_text.strip().split("\n") if stdout_text else []
    if stdout_lines and stdout_lines[0].strip():
        smiles = stdout_lines[0].strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    return None


def _opsin_cli_stdout(name: str, opsin_jar: str) -> Optional[str]:
    """Fresh-subprocess OPSIN parse -- exactly the previous cold path.

    Writes ``name`` to a temp file and runs ``java -jar opsin -osmi``, returning
    the raw stdout text (byte-for-byte what the CLI writes) or ``None`` when the
    subprocess could not run (temp-file failure, timeout, or any Java-side
    error). Used only as the fall-through when the in-process JVM in
    :func:`_parse_name_with_opsin_uncached` cannot serve the call, so the answer
    is identical -- only slower.
    """
    # Write single name to temp file
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(name + "\n")
            temp_input = f.name
    except OSError:
        return None

    try:
        cmd = ["java", *JVM_HYGIENE_FLAGS, "-jar", opsin_jar, "-osmi", temp_input]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        return proc.stdout
    except (subprocess.TimeoutExpired, OSError, Exception):
        return None
    finally:
        try:
            os.unlink(temp_input)
        except OSError:
            pass


def _heavy_element_multiset(mol) -> Counter:
    """Element multiset of *mol*'s heavy atoms, e.g. ``{'C': 3, 'N': 1}``."""
    return Counter(
        a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() > 1
    )


def _inchikey(mol) -> str:
    """Full InChIKey of *mol*, or ``''`` when InChI cannot be generated.

    Returning ``''`` is load-bearing: the caller treats an absent key as a
    FAILED constitution check, never as a passing one.
    """
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return ""
    return key or ""


def _skeleton(inchikey: str) -> str:
    """First (connectivity) block of an InChIKey; ``''`` if malformed.

    The skeleton block encodes formula + connectivity + H layer.  Stereo,
    isotope and charge live in the second block and are deliberately outside
    this validator's scope (see the module docstring).
    """
    if not inchikey:
        return ""
    head = inchikey.split("-")[0]
    return head if len(head) == 14 else ""


def validate_atom_coverage(
    mol,
    name: str,
    opsin_jar: Optional[str] = None,
) -> CoverageResult:
    """Validate how completely *name* describes *mol*.

    Parses *name* back to a molecule via OPSIN and compares that molecule to
    *mol* by CONSTITUTION (InChIKey skeleton block).  Heavy-atom counts are
    reported as diagnostics -- dropped atoms as ``unclaimed_atoms``, invented
    atoms as ``extra_atoms`` -- but never decide ``is_complete``, because a
    count cannot distinguish the same atoms from the same number of atoms.

    Args:
        mol: RDKit Mol object for the input molecule.
        name: Generated IUPAC name to validate.
        opsin_jar: Path to OPSIN JAR. If ``None``, attempts auto-discovery.
            If OPSIN is unavailable, returns a result with method='unavailable'.

    Returns:
        A :class:`CoverageResult` with coverage metrics.
    """
    total_heavy = mol.GetNumHeavyAtoms()

    if total_heavy == 0:
        # A molecule with no heavy atoms has an empty heavy-atom set, so that
        # set is vacuously covered.  NOTE (pre-existing, deliberately left):
        # this branch does not consult *name* at all, so it is the one exit
        # that does not prove constitution.  Reachable only for inputs such
        # as [H][H]; kept as-is rather than widened unmeasured.
        return CoverageResult(
            total_heavy_atoms=0,
            claimed_atoms=0,
            unclaimed_atoms=0,
            coverage_ratio=1.0,
            is_complete=True,
            method="trivial",
            constitution_match=True,
        )

    # Try auto-discovery if no JAR path given
    if opsin_jar is None:
        opsin_jar = find_opsin_jar()

    # Check Java availability
    if opsin_jar is not None:
        import shutil

        if not shutil.which("java"):
            opsin_jar = None

    # Parse-back approach
    if opsin_jar is not None and name:
        parsed_smiles = _parse_name_with_opsin(name, opsin_jar)
        if parsed_smiles is not None:
            parsed_mol = Chem.MolFromSmiles(parsed_smiles)
            if parsed_mol is not None:
                parsed_heavy = parsed_mol.GetNumHeavyAtoms()

                # Per-ELEMENT intersection, not a bare count: a carbon in the
                # name cannot stand in for a nitrogen in the molecule.  The
                # intersection is <= total_heavy by construction, so the old
                # `min(..., 1.0)` clamp -- which is what made atom GAIN
                # invisible -- is not merely removed but unnecessary.
                want = _heavy_element_multiset(mol)
                got = _heavy_element_multiset(parsed_mol)
                claimed = sum((want & got).values())
                unclaimed = total_heavy - claimed
                # Atoms the NAME has and the MOLECULE does not: fabrication.
                extra = parsed_heavy - claimed

                in_key = _inchikey(mol)
                out_key = _inchikey(parsed_mol)
                in_skel = _skeleton(in_key)
                out_skel = _skeleton(out_key)

                if not in_skel or not out_skel:
                    # Fail closed: with no constitution evidence we do not get
                    # to claim completeness, and we say so in `method` rather
                    # than masquerading as a successful parse_back.
                    return CoverageResult(
                        total_heavy_atoms=total_heavy,
                        claimed_atoms=claimed,
                        unclaimed_atoms=unclaimed,
                        coverage_ratio=claimed / total_heavy,
                        is_complete=False,
                        method="parse_back_no_inchi",
                        extra_atoms=extra,
                        parsed_heavy_atoms=parsed_heavy,
                        constitution_match=False,
                        input_inchikey=in_key,
                        parsed_inchikey=out_key,
                    )

                same_constitution = in_skel == out_skel

                return CoverageResult(
                    total_heavy_atoms=total_heavy,
                    claimed_atoms=claimed,
                    unclaimed_atoms=unclaimed,
                    coverage_ratio=claimed / total_heavy,
                    # NOT a threshold on the ratio.  Identical constitution is
                    # the only thing that establishes the name covers exactly
                    # these atoms -- see the module docstring for what that
                    # does and does not fix.
                    is_complete=same_constitution,
                    method="parse_back",
                    extra_atoms=extra,
                    parsed_heavy_atoms=parsed_heavy,
                    constitution_match=same_constitution,
                    input_inchikey=in_key,
                    parsed_inchikey=out_key,
                )

    # Fallback: OPSIN unavailable or parse failed
    return CoverageResult(
        total_heavy_atoms=total_heavy,
        claimed_atoms=0,
        unclaimed_atoms=total_heavy,
        coverage_ratio=0.0,
        is_complete=False,
        method="unavailable",
    )
