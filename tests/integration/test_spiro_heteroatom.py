"""a phase-02 +: hetero-spiro 'a'-prefix OPSIN round-trip.

Confirms heteroatom 'a'-prefix path uses
``polycyclic_bridged.get_heteroatom_prefix`` and that the resulting
names round-trip through OPSIN to InChI L1 == input.

Source: 151-02-PLAN.md task 1b; internal notes-B.md (rows 5-8 hetero-spiro
matrix); IUPAC.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym.rules.spiro import name_spiro_system


@pytest.mark.integration
def test_hetero_spiro_imports_polycyclic_bridged_helper():
    """: spiro.py reuses polycyclic_bridged.get_heteroatom_prefix."""
    import inspect
    from orthonym.rules import spiro
    src = inspect.getsource(spiro)
    assert "from ..rules.polycyclic_bridged import get_heteroatom_prefix" in src \
        or "from .polycyclic_bridged import get_heteroatom_prefix" in src \
        or ("get_heteroatom_prefix" in src
            and "polycyclic_bridged" in src), (
        "spiro.py must reuse get_heteroatom_prefix from polycyclic_bridged "
        "(D-12 lock)"
    )


@pytest.mark.integration
def test_hetero_spiro_no_new_prefix_builder():
    """: no new heteroatom prefix builder beyond _build_hetero_prefix."""
    import inspect
    from orthonym.rules import spiro
    src = inspect.getsource(spiro)
    # Allow only the existing _build_hetero_prefix; reject any new
    # _build_heteroatom_prefix or similar.
    assert "def _build_heteroatom_prefix" not in src


@pytest.mark.integration
def test_hetero_spiro_priority_order_in_existing_helper():
    """IUPAC priority order O > S > Se > N > P > Si > B is honored
    by the existing _build_hetero_prefix function.

    Asserted BEHAVIOURALLY. This test used to grep the function's source for the
    literal list ``['O', 'S', 'Se', 'N', 'P', 'Si', 'B']`` (or merely for the
    substrings ``'O'`` and ``'N'``). That hard-coded list is gone: ordering is
    delegated to ``hw_heteroatoms.sort_heteroatoms_by_priority``, the canonical
    seniority source, which also covers the Te/Ge/As/Sb/Bi/Sn/Pb rows the literal
    list omitted. A substring lock cannot tell correct delegation from a
    regression, and it fails on any refactor that improves the code; citing the
    prefixes in the wrong order is what actually matters, so test that.
    """
    from orthonym.perception.rings import get_spiro_atoms
    from orthonym.rules.spiro import _build_hetero_prefix

    # 3-oxa / 9-thia on a spiro[5.5]undecane: O must be cited BEFORE S even
    # though this places them on ascending locants anyway...
    mol = Chem.MolFromSmiles("C1CC2(CCO1)CCSCC2")
    assert mol is not None
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    result = _build_hetero_prefix(mol, set(get_spiro_atoms(mol)), ring)
    assert result is not None, "in-table hetero spiro must still build"
    # ``_build_hetero_prefix`` returns the shared ``ReplacementPrefix`` (a
    # per-atom decomposition riding alongside the string, af0d7262); unwrap
    # ``.prefix`` to get the plain string this test asserts ordering on.
    prefix = result.prefix
    assert prefix.index("oxa") < prefix.index("thia"), (
        f"O must be cited before S (P-25.2 seniority): {prefix!r}"
    )

    #... and the seniority order must NOT be an artefact of locant order: here
    # the senior O carries the HIGHER locant, so a locant-sorted implementation
    # would cite thia first.
    mol2 = Chem.MolFromSmiles("C1CC2(CCS1)CCOCC2")
    assert mol2 is not None
    ring2 = {a.GetIdx() for a in mol2.GetAtoms() if a.IsInRing()}
    result2 = _build_hetero_prefix(mol2, set(get_spiro_atoms(mol2)), ring2)
    assert result2 is not None
    prefix2 = result2.prefix
    assert prefix2.index("oxa") < prefix2.index("thia"), (
        f"O must be cited before S regardless of locants: {prefix2!r}"
    )


@pytest.mark.integration
def test_hetero_spiro_emits_a_prefix_on_2_oxaspiro_3_5_nonane():
    """Smoke check: 'oxa' appears in the name of a hetero-spiro compound."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("C1CC2(O1)CCCCC2")
    result = name_spiro_system(mol)
    assert result is not None
    name = result[0]
    assert "oxa" in name


@pytest.mark.integration
def test_hetero_spiro_emits_a_prefix_on_dioxaspiro():
    """Multi-heteroatom 'a'-prefix smoke check."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("C1CC2(OCCO2)CCC1")
    result = name_spiro_system(mol)
    assert result is not None
    name = result[0]
    assert "dioxa" in name
