"""Shared assertion helper for fusion-descriptor tests (task-123).

``assert_fusion_pin`` names a molecule, asserts the engine did not abstain,
affirmatively OPSIN round-trips the emitted name back to the input's
InChIKey (0-wrong -- never trust that a name "looks right"), and optionally
asserts an exact expected string.
"""
from typing import Optional

from rdkit import Chem

from orthonym import name_compound
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse


def assert_fusion_pin(smi: str, expected: Optional[str] = None) -> str:
    """Name ``smi``, assert it is a real (non-abstain, non-fallback,
    non-von-Baeyer-degraded) fusion name that OPSIN round-trips to the input
    structure, and (if given) assert it equals ``expected`` verbatim.

    Returns the emitted name for any further assertions the caller wants.
    """
    name = name_compound(smi)

    assert not is_failure_name(name), (
        f"{smi!r} abstained/failed: {name!r}"
    )
    assert "unknown organic compound" not in name, (
        f"{smi!r} produced the abstain sentinel: {name!r}"
    )
    assert "cyclo[" not in name, (
        f"{smi!r} degraded to a von Baeyer name instead of the fusion PIN: {name!r}"
    )

    input_mol = Chem.MolFromSmiles(smi)
    assert input_mol is not None, f"un-parseable input SMILES: {smi!r}"
    input_key = Chem.MolToInchiKey(input_mol)

    parsed_smiles = opsin_parse(name)
    assert parsed_smiles, f"OPSIN could not parse emitted name {name!r} for {smi!r}"
    parsed_mol = Chem.MolFromSmiles(parsed_smiles)
    assert parsed_mol is not None, (
        f"OPSIN parse of {name!r} produced un-parseable SMILES {parsed_smiles!r}"
    )
    parsed_key = Chem.MolToInchiKey(parsed_mol)
    assert parsed_key == input_key, (
        f"{name!r} round-trips to a DIFFERENT molecule: "
        f"input={input_key} parsed={parsed_key} (smiles {smi!r})"
    )

    if expected is not None:
        assert name == expected, f"{smi!r}: expected {expected!r}, got {name!r}"

    return name
