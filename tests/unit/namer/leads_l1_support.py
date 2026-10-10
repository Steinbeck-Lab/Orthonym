"""Shared helpers of the leads-program lane L1 tests (tests/unit/namer/test_leads_l1_*.py).

OPSIN's own StdInChIKey of a name (-ostdinchikey, no SMILES and no RDKit on the name side)
is compared with RDKit's standard InChIKey of the structure the name must denote.
"""
import subprocess
from functools import lru_cache

from rdkit import Chem


@lru_cache(maxsize=None)
def opsin_key(name: str) -> str:
    """OPSIN 2.9.0's StdInChIKey of ``name``; "" when OPSIN cannot parse it."""
    from orthonym.jvm_budget import jvm_slots
    from orthonym.jvm_flags import JVM_HYGIENE_FLAGS
    from orthonym.validation.opsin_roundtrip import _find_opsin_jar
    jar = _find_opsin_jar("2.9.0")
    with jvm_slots(1, purpose="leads-L1-test"):
        proc = subprocess.run(["java", *JVM_HYGIENE_FLAGS, "-jar", jar, "-ostdinchikey"],
                              input=name + "\n", capture_output=True, text=True, timeout=120)
    return proc.stdout.strip()


def rdkit_key(smiles: str) -> str:
    """RDKit's standard InChIKey of ``smiles`` (a CXSMILES extension is read as RDKit reads it)."""
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def mol_key_without_stereo_of(smiles: str, atoms) -> str:
    """RDKit's InChIKey of ``smiles`` with the configuration of the given atom indices cleared."""
    mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
    for idx in atoms:
        mol.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    mol.SetStereoGroups([])
    out = mol.GetMol()
    Chem.AssignStereochemistry(out, cleanIt=True, force=True)
    return Chem.MolToInchiKey(out)
