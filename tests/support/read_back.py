"""Plain read-back checks the test files share: the name read by OPSIN, compared with the input.

These are deliberately simpler than ``rt_assert.name_is_rt_exact`` (no radical, protonation-site,
lone-pair or dative checks), so a test that used one of them keeps the verdict it had."""
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.jvm_bridge import opsin_stdout
from orthonym.validation.opsin_roundtrip import opsin_parse
from tests.support.jars import jar_or_skip


def same_key(smiles: str, name: str) -> bool:
    """OPSIN reads ``name`` (``opsin_parse``) to the full standard InChIKey of ``smiles``."""
    if not name:
        return False
    out = opsin_parse(name)
    if not out:
        return False
    mol_in, mol_out = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(out)
    if mol_in is None or mol_out is None:
        return False
    return inchi.MolToInchiKey(mol_in) == inchi.MolToInchiKey(mol_out)


def key(smiles: str) -> str:
    """The standard InChIKey of ``smiles``."""
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def fixed_h(smiles: str) -> str:
    """The InChI of ``smiles`` with the fixed-H layer and no stereo layer: it tells the hydron sites apart."""
    return inchi.MolToInchi(Chem.MolFromSmiles(smiles), options="/FixedH /SNon")


def same_hydron_sites(name: str, smiles: str) -> bool:
    """An independent OPSIN read of ``name`` has the standard InChIKey and the fixed-H InChI of ``smiles``."""
    from tests.support.rt_assert import _independent_parse
    parsed = _independent_parse(name, smiles)
    return bool(parsed) and key(parsed) == key(smiles) and fixed_h(parsed) == fixed_h(smiles)


def same_canonical_smiles(name: str, smiles: str) -> bool:
    """OPSIN (radicals allowed) reads ``name`` to the canonical SMILES of ``smiles``; skips without the jar."""
    jar = jar_or_skip()
    txt, _ = opsin_stdout(name, allow_radicals=True, jar_path=jar)
    parsed = Chem.MolFromSmiles((txt or "").strip())
    return parsed is not None and Chem.MolToSmiles(parsed) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def spelling_rules(smiles: str, name: str) -> list:
    """The rules ``check_pin_spelling`` (strict) reports for ``name`` of ``smiles``."""
    from orthonym.validation.pin_spelling import check_pin_spelling
    return [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)]


def key_or_none(smiles: str):
    """The standard InChIKey of ``smiles``, or None when RDKit does not read it."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


def key_or_empty(smiles: str) -> str:
    """The standard InChIKey of ``smiles``, or '' for no SMILES or one RDKit does not read."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToInchiKey(mol) if mol is not None else ""


def mol(smiles: str):
    """The RDKit molecule of a fixture SMILES (an assertion error when it does not parse)."""
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"Invalid SMILES: {smiles}"
    return m
