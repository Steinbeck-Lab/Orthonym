"""Corrected attached-component fusion prefixes for the pyran O/S/Se/Te family
(``pyrano`` / ``thiopyrano`` / ``selenopyrano`` / ``telluropyrano``) in fused and
spiro-of-fused component names.

ROOT CAUSE (fixed): ``get_fusion_prefix`` truncated ``pyran`` -> ``pyro`` (an
OPSIN-unparseable string) because it kept its OWN lookup table that lacked
``pyran`` and fell back to a general ``-an -> -o`` rule that contradicts the Blue
Book. The authoritative ``MONOCYCLIC_COMPONENTS`` registry already stored
``pyran -> pyrano`` but the function never consulted it. In ``name_mixed_spiro_fused``
this produced malformed fusion-component names such as ``pyro[3,2-d]pyrazole`` that
OPSIN cannot parse, so the whole spiro molecule abstained.

GOVERNING RULE, IUPAC 2013 **** (``the Blue Book Blue Book``):

  "The names of attached components are formed by replacing the last letter 'e' by
   'o' in the name of the component, i.e., indeno from indene (or by ADDING the
   letter 'o' when no final letter 'e' is present, i.e., **pyrano from pyran**)..."

``the Blue Book`` gives "selenopyrano (preferred prefix) (from selenopyran,
PIN)". Cross-checked against OPSIN's own ``fusionComponents`` token list
(``pyrano`` / ``thiopyrano`` / ``selenopyrano`` / ``telluropyrano``).

0-wrong: every corrected name is validated to round-trip to the INPUT constitution
(full-InChIKey) through OPSIN; a name that parses to a different molecule is a
defect this suite would catch.
"""

import pytest
from rdkit import Chem

from orthonym.rules.fusion_descriptors import get_fusion_prefix
from orthonym.data.fusion_components import get_component_prefix
from orthonym.rules.fused_rings import name_fused_heterocycle
from orthonym.rules.spiro import name_mixed_spiro_fused, is_mixed_spiro_fused
from orthonym.metrics.provenance import best_effort_ctx


def _const_key(smi):
    """Constitution key = first InChIKey block (skeleton, stereo-insensitive)."""
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m).split('-')[0] if m else None


class TestPyranFamilyFusionPrefix:
    """: no final 'e' -> ADD 'o' ("pyrano from pyran"); the '-an'
    truncation that produced 'pyro' is a defect."""

    @pytest.mark.unit
    @pytest.mark.parametrize("name,prefix", [
        ("pyran", "pyrano"),
        ("thiopyran", "thiopyrano"),
        ("selenopyran", "selenopyrano"),
        ("telluropyran", "telluropyrano"),
    ])
    def test_pyran_family_prefix(self, name, prefix):
        assert get_fusion_prefix(name) == prefix
        assert get_component_prefix(name) == prefix

    @pytest.mark.unit
    def test_no_pyro_truncation(self):
        # The specific regression: pyran must NOT become the OPSIN-unparseable 'pyro'.
        assert get_fusion_prefix("pyran") != "pyro"

    @pytest.mark.unit
    @pytest.mark.parametrize("name", ["2H-pyran", "4H-pyran"])
    def test_indicated_h_descriptor_does_not_corrupt_prefix(self, name):
        # A leading indicated-hydrogen descriptor is dropped for the attached-
        # component prefix and must not lower-case / truncate it ('2h-pyro').
        assert get_fusion_prefix(name) == "pyrano"

    @pytest.mark.unit
    @pytest.mark.parametrize("name,prefix", [
        ("benzene", "benzo"),
        ("naphthalene", "naphtho"),
        ("furan", "furo"),
        ("thiophene", "thieno"),
        ("pyridine", "pyrido"),
        ("pyrrole", "pyrrolo"),
        ("pyrimidine", "pyrimido"),
        ("cyclohexene", "cyclohexa"),
        ("cyclopentadiene", "cyclopenta"),
    ])
    def test_previously_correct_prefixes_unchanged(self, name, prefix):
        # Regression guard: the fix must not disturb any prefix that was already
        # correct (only 'pyran' and the S/Se/Te analogues change).
        assert get_fusion_prefix(name) == prefix


class TestPyranoFusedNaming:
    """The corrected prefix must reach the assembled fusion / spiro name."""

    @pytest.mark.unit
    def test_pyranopyrazole_uses_pyrano_not_pyro(self):
        # pyrano[2,3-c]pyrazole ground-truth structure (OPSIN-generated).
        m = Chem.MolFromSmiles("N=1N=CC=2C1OC=CC2")
        res = name_fused_heterocycle(m)
        assert res is not None
        name = res[0]
        assert "pyrano" in name, name
        assert "pyro[" not in name, name

    @pytest.mark.unit
    def test_spiro_fused_component_uses_pyrano(self):
        # spiro[cyclohexane-1,4'-pyrano[2,3-c]pyrazole] ring core.
        m = Chem.MolFromSmiles("N=1NC=C2C1OC=CC21CCCCC1")
        assert is_mixed_spiro_fused(m)
        tok = best_effort_ctx.set("t")
        try:
            res = name_mixed_spiro_fused(m)
        finally:
            best_effort_ctx.reset(tok)
        assert res is not None
        assert "pyrano" in res[0], res[0]
        assert "pyro[" not in res[0], res[0]


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
class TestPyranoRoundTrip:
    """Every corrected pyrano fused name must round-trip to the INPUT
    constitution through OPSIN (0-wrong)."""

    @pytest.mark.parametrize("struct,label", [
        ("N=1N=CC=2C1OC=CC2", "pyrano[2,3-c]pyrazole"),
        ("N1=C2C(C=C1)=COC=C2", "pyrano[4,3-b]pyrrole"),
        ("N1=CN=CC2=C1OCC=C2", "pyrano[2,3-d]pyrimidine"),
    ])
    def test_pyrano_fused_roundtrip(self, struct, label, opsin_to_smiles):
        m = Chem.MolFromSmiles(struct)
        res = name_fused_heterocycle(m)
        assert res is not None, f"no name for {label}"
        name = res[0]
        assert "pyro[" not in name, name  # not the old truncation
        back = opsin_to_smiles(name)
        assert back is not None, f"OPSIN could not parse {name!r} ({label})"
        assert _const_key(back) == _const_key(struct), (
            f"{name!r} round-trips to a DIFFERENT constitution than {label}")
