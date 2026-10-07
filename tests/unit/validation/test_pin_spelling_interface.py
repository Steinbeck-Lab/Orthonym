"""The interface of the spelling checks of a pin_verified name (``validation/pin_spelling.py``).

A round trip proves the molecule, never the spelling: OPSIN reads 'pyrido[1,2-b]pyridazin-6-one'
back to the same structure as the PIN '6H-pyrido[1,2-b]pyridazin-6-one'. Each spelling check
decides one Blue Book rule from the RDKit structure and a lexical reading of the name, and never
calls the engine's naming code or OPSIN (so it cannot inherit a defect of the code that built the
name). Interface (binding for every lane that adds a check): ``SpellingFailure(rule, detail)``,
``register(rule_id)`` for ``fn(mol, name) -> SpellingFailure | None`` (several per rule id), and
``check_pin_spelling(mol, name) -> list[SpellingFailure]``.
"""
import ast
import dataclasses
import logging
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym.validation import pin_spelling as ps

SPELLING_DIR = Path(ps.__file__).parent / "spelling"


@pytest.fixture
def empty_registry(monkeypatch):
    """A registry with no check, so a test registers its own (the built-in ones stay unloaded)."""
    monkeypatch.setattr(ps, "_REGISTRY", {})
    monkeypatch.setattr(ps, "_loaded", True)


def test_a_failure_is_a_frozen_rule_and_detail():
    f = ps.SpellingFailure("P-14.4", "suffix locant 6 > 2")
    assert (f.rule, f.detail) == ("P-14.4", "suffix locant 6 > 2")
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.rule = "P-14.5"


def test_several_checks_share_one_rule_id(empty_registry):
    @ps.register("P-TEST")
    def a(mol, name):
        return ps.SpellingFailure("P-TEST", "a") if "a" in name else None

    @ps.register("P-TEST")
    def b(mol, name):
        return ps.SpellingFailure("P-TEST", "b") if "b" in name else None

    mol = Chem.MolFromSmiles("C")
    assert [f.detail for f in ps.check_pin_spelling(mol, "ab")] == ["a", "b"]
    assert ps.check_pin_spelling(mol, "c") == []


def test_a_check_that_raises_abstains_unless_strict(empty_registry, caplog):
    @ps.register("P-TEST")
    def broken(mol, name):
        raise RuntimeError("reader defect")

    mol = Chem.MolFromSmiles("C")
    with caplog.at_level(logging.WARNING, logger="orthonym.validation.pin_spelling"):
        assert ps.check_pin_spelling(mol, "methane") == []
    assert "reader defect" in caplog.text
    with pytest.raises(RuntimeError):
        ps.check_pin_spelling(mol, "methane", strict=True)


def test_no_structure_or_no_name_gives_no_failure():
    assert ps.check_pin_spelling(None, "ethene") == []
    assert ps.check_pin_spelling(Chem.MolFromSmiles("C=C"), "") == []
    assert ps.check_pin_spelling(Chem.MolFromSmiles("C=C"), None) == []


def test_no_structure_or_no_name_loads_no_check_module(monkeypatch):
    """The no-op call returns before the built-in modules are imported: a module path that
    cannot be imported is not reached, and the registry stays unloaded."""
    monkeypatch.setattr(ps, "_REGISTRY", {})
    monkeypatch.setattr(ps, "_loaded", False)
    monkeypatch.setattr(ps, "BUILTIN_CHECK_MODULES", ("orthonym.validation.spelling.no_such_module",))
    assert ps.check_pin_spelling(None, "ethene") == []
    assert ps.check_pin_spelling(Chem.MolFromSmiles("C=C"), "") == []
    assert ps._loaded is False
    with pytest.raises(ModuleNotFoundError):             # a call with both inputs loads them
        ps.check_pin_spelling(Chem.MolFromSmiles("C=C"), "ethene")


def test_registered_rules_are_the_rule_ids_with_a_check(empty_registry):
    assert ps.registered_rules() == ()

    @ps.register("P-TEST-A")
    def a(mol, name):
        return None

    @ps.register("P-TEST-B")
    def b(mol, name):
        return None

    @ps.register("P-TEST-A")
    def c(mol, name):
        return None

    assert ps.registered_rules() == ("P-TEST-A", "P-TEST-B")     # one id per rule, in order


def test_registered_rules_load_the_built_in_check_modules(monkeypatch):
    """``registered_rules`` imports the modules of ``BUILTIN_CHECK_MODULES`` (once) before it
    lists the rule ids, so a module's checks are listed without a check call first."""
    import types
    imported = []

    def fake_import(mod):
        imported.append(mod)
        ps.register("P-LOADED")(lambda mol, name: None)

    monkeypatch.setattr(ps, "_REGISTRY", {})
    monkeypatch.setattr(ps, "_loaded", False)
    monkeypatch.setattr(ps, "BUILTIN_CHECK_MODULES", ("orthonym.validation.spelling.checks_x",))
    monkeypatch.setattr(ps, "importlib", types.SimpleNamespace(import_module=fake_import))
    assert ps.registered_rules() == ("P-LOADED",)
    assert ps.registered_rules() == ("P-LOADED",)
    assert imported == ["orthonym.validation.spelling.checks_x"]


def test_the_built_in_check_modules_are_the_spelling_package():
    assert all(m.startswith("orthonym.validation.spelling.") for m in ps.BUILTIN_CHECK_MODULES)
    for mod in ps.BUILTIN_CHECK_MODULES:
        assert (SPELLING_DIR / (mod.rsplit(".", 1)[1] + ".py")).is_file(), mod


ALLOWED_TOP = {"__future__", "re", "dataclasses", "functools", "importlib", "logging", "typing",
               "collections", "itertools", "rdkit"}


@pytest.mark.parametrize("path", [Path(ps.__file__)] + sorted(SPELLING_DIR.glob("*.py")),
                         ids=lambda p: p.name)
def test_the_checks_import_no_naming_code_and_no_opsin(path):
    """Independence: the checks read RDKit and the name only. Every import of the module is the
    standard library, RDKit, or a module of the spelling package (relative imports)."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in ALLOWED_TOP, (path.name, alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                assert node.module.split(".")[0] in ALLOWED_TOP, (path.name, node.module)
            elif path.name == "pin_spelling.py":
                pytest.fail(f"pin_spelling.py imports {node.module!r} relatively")
            else:
                # inside the spelling package: its own modules, or..pin_spelling
                assert node.level == 1 or (node.level == 2 and node.module == "pin_spelling"), \
                    (path.name, node.level, node.module)
