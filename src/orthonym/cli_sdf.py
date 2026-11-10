"""``orthonym --sdf FILE``: name every record of a MOL or SDF file.

Each record is read by RDKit (``Chem.ForwardSDMolSupplier``: sanitized, hydrogen atoms removed) and named
as an RDKit Mol is named (``input_mol.smiles_for``): through its proven SMILES, or its CXSMILES when the
record has an AND or OR stereo group (V3000 STEREO collections). The stereo is the file's own: the
tetrahedral configurations RDKit reads from the wedge bonds (or from the coordinates of a 3D record), the
double-bond configurations from the coordinates, as the MDL format defines them, and none for an 'either'
bond, a wavy bond or a crossed double bond. The coordinates are then dropped, so nothing is perceived
from them again while the structure is named.

A V2000 record's chiral flag is not read: as in the standard InChI, which reads none by default, and as
RDKit's and many other writers set it (0 also for a structure of known configuration), the tetrahedral
configurations of a record are absolute; a racemate or a relative configuration is stated by a V3000 STEREO
collection (AND, OR).

The records become the lines of ``--batch``: one output line per record, in the order of the file, the
first column the string that was named. A record RDKit cannot read, or whose SMILES does not hold it,
becomes a line no SMILES reader reads ('<record 5>'), which ``--batch`` reports as it reports an unreadable
line ('ERROR:' and exit status 1), and its number and the reason go to the error stream. A record with no
atoms gives an empty line and a message.
"""
import atexit
import os
import sys
import tempfile
from typing import List, Optional, Tuple


def record_lines(path: str) -> Tuple[List[str], List[str]]:
    """``(lines, problems)``: the string to name for each record of ``path`` ('' for one that has none) and
    one message per record that has none."""
    from rdkit import Chem

    from .input_mol import InvalidInputError, smiles_for
    from .rdkit_log import rdkit_logs_off
    lines: List[str] = []
    problems: List[str] = []
    with rdkit_logs_off("rdApp.*"), open(path, "rb") as handle:
        for number, mol in enumerate(Chem.ForwardSDMolSupplier(handle), 1):
            if mol is None:
                lines.append(f"<record {number}>")
                problems.append(f"record {number}: RDKit cannot read it")
                continue
            try:
                line = smiles_for(mol)
            except (InvalidInputError, TypeError) as exc:
                lines.append(f"<record {number}>")
                problems.append(f"record {number}: {exc}")
                continue
            if not line:
                problems.append(f"record {number}: it has no atoms")
            lines.append(line)
    return lines, problems


def batch_file(path: str) -> Optional[str]:
    """A temporary file of:func:`record_lines` for ``--batch``, removed when the process ends; the problems
    are printed on the error stream. None when ``path`` cannot be read (``--batch`` then reports it)."""
    try:
        lines, problems = record_lines(path)
    except OSError:
        return None
    for problem in problems:
        print(f"orthonym --sdf: {problem}", file=sys.stderr)
    fd, temp = tempfile.mkstemp(prefix="orthonym_sdf_", suffix=".smi")
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write("".join(line + "\n" for line in lines))
    atexit.register(lambda: os.path.exists(temp) and os.unlink(temp))
    return temp
