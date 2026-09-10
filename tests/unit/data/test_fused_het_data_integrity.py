""" — permanent OPSIN round-trip integrity guard over FUSED_HETEROCYCLE_DATA.

Every catalog entry's ``name`` must parse (via OPSIN) back to the SAME ring system as
its SMILES key. This is the LIVE counterpart to the offline ``expected_canon.json``
lint in ``test_smiles_dict_canonical_lint.py``: it needs Java/OPSIN at run time but
catches a name change the instant it lands — there is no committed snapshot to go
stale. It is the permanent promotion of the whole-table audit harness that
found 44 mislabeled + 2 OPSIN-unparseable entries (all corrected in the sweep).

Comparison is at the CONSTITUTIONAL (connectivity) level — the skeleton block of the
standard InChIKey, which normalizes the mobile ring N-H. This passes the five
intentional purine tautomers (adenine / guanine / hypoxanthine / xanthine)
automatically, while still catching every different-ring-system mislabel — the exact
"names a different molecule" class the Phase-0 self-consistency gate also targets.
"""
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
from orthonym.validation.opsin_roundtrip import _find_opsin_jar, _java_available

_OPSIN_JAR = _find_opsin_jar()
_OPSIN_OK = _java_available() and _OPSIN_JAR is not None

pytestmark = pytest.mark.skipif(
    not _OPSIN_OK, reason="OPSIN JAR / Java runtime unavailable — live round-trip skipped"
)


def _skeleton(smiles: str):
    """Constitutional skeleton = first block of the standard InChIKey (mobile-H
    normalized, so tautomers collapse but different ring systems do not)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    key = Chem.inchi.MolToInchiKey(mol)
    return key.split("-")[0] if key else None


def _opsin_batch(names):
    """Parse all names in ONE OPSIN subprocess (one JVM start). Returns SMILES per
    name aligned with ``names`` ('' on parse failure)."""
    proc = subprocess.run(
        ["java", "-jar", _OPSIN_JAR, "-osmi"],
        input="\n".join(names) + "\n",
        capture_output=True, text=True, timeout=600,
    )
    out = []
    for line in proc.stdout.split("\n"):
        s = line.strip()
        out.append("" if (not s or "could not be interpreted" in s.lower()
                          or "unsure of the meaning" in s.lower()) else s)
    return (out + [""] * len(names))[:len(names)]


@pytest.mark.roundtrip
def test_every_catalog_name_roundtrips_to_its_key():
    """No FUSED_HETEROCYCLE_DATA entry's name may parse to a different ring system."""
    items = sorted((d["name"], smi) for smi, d in FUSED_HETEROCYCLE_DATA.items())
    names = [n for n, _ in items]
    parsed = _opsin_batch(names)

    unparseable, mislabeled = [], []
    for (name, key), osmi in zip(items, parsed):
        if not osmi:
            unparseable.append(name)
            continue
        want, got = _skeleton(key), _skeleton(osmi)
        if want != got:
            mislabeled.append((name, key, osmi))

    msg = []
    if unparseable:
        msg.append(f"{len(unparseable)} OPSIN-unparseable catalog name(s): {unparseable}")
    if mislabeled:
        msg.append(f"{len(mislabeled)} mislabeled (name -> different ring system than key):")
        for name, key, osmi in mislabeled:
            msg.append(f"    {name!r}: key={key}  name->OPSIN={osmi}")
    assert not msg, "FUSED_HETEROCYCLE_DATA round-trip integrity FAILED:\n" + "\n".join(msg)
