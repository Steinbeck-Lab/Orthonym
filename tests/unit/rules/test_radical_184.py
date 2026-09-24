"""Wave-0 RED unit suite for the non-terminal radical ``-yl`` emitter (a phase
-RADICAL, conditional sub-step -> SHIPS per RESEARCH additivity proof).

These tests are written BEFORE the implementation (the Nyquist gate). They
assert the IUPAC-2013 PINs for carbon-centred radicals and MUST FAIL (the
assertion, not collection) on the pre-implementation HEAD: today a radical name
(``pentyl``) is suppressed to ``unknown organic compound`` by the OPSIN validity
gate (which runs OPSIN WITHOUT ``--allowRadicals``) AND has no locant. Plan
184-05 (a) routes radicals through ``emit_parent_hydride_cumulative_suffix``
(suffix ``yl``) for the locant and (b) teaches ``OpsinOracle._invoke_opsin`` to
pass ``-r`` (proven strictly additive: 66 gains / 0 changes / 0 regressions over
11,668 names, RESEARCH -RADICAL). They flip to GREEN then.

Every assertion docstring cites the verbatim Blue Book P-number.

Blue Book sources (the Blue Book Blue Book):
  - Table 3.4 (line 17597): loss of H. -> suffix ``-yl``.
  - line 17608 (``ethan-2-id-1-yl``): the radical centre carries a locant.
  - /: radicals from parent hydrides; lowest locant for the
    radical centre.

RED-by-design on HEAD (verified): ``CC[CH]CC`` / ``CCC[CH]CC`` / ``[CH3]`` all
-> ``unknown organic compound`` (gate-suppressed). The ``[CH3]`` -> ``methyl``
guard is therefore ALSO RED today (the gate suppresses the retained radical too)
and is marked xfail until Plan 184-05 exempts radical names via the ``-r``
oracle; it is documented, not a collection error.
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.assembly.retained_substitution import OpsinOracle
from tests.support.jars import jar_or_skip


@pytest.mark.unit
class TestNonTerminalRadicalLocant:
    """-RADICAL: non-terminal alkyl radical -> ``...-N-yl`` with a locant
    . RED on HEAD by design (gate suppression + missing locant)."""

    def test_pentan_3_yl(self):
        """ / Table 3.4: the radical centre takes the lowest locant over the
        re-found chain. CC[CH]CC is the pentan-3-yl radical (radical carbon at
        C3 of a 5-carbon chain)."""
        assert Orthonym().name("CC[CH]CC") == "pentan-3-yl"

    def test_hexan_3_yl(self):
        """ / Table 3.4: CCC[CH]CC -> hexan-3-yl (radical carbon at C3 of a
        6-carbon chain; numbering from either end gives 3)."""
        assert Orthonym().name("CCC[CH]CC") == "hexan-3-yl"

    @pytest.mark.xfail(
        strict=False,
        reason="Retained radical methyl is gate-suppressed to 'unknown' on HEAD "
        "(OPSIN run without --allowRadicals); goes GREEN when Plan 184-05 adds "
        "-r to the OPSIN oracle. Tracked, not blocking Wave 0.",
    )
    def test_methyl_retained_no_locant(self):
        """: the retained radical name ``methyl`` carries no locant
        (single carbon). Guard that the new locant primitive does NOT inject a
        spurious locant once the gate stops suppressing radical names."""
        assert Orthonym().name("[CH3]") == "methyl"


@pytest.mark.unit
class TestOracleAdditivityBaseline:
    """The Plan-184-05 ``-r`` oracle change MUST stay strictly additive: a
    non-radical name must still round-trip through OpsinOracle unchanged. This
    baseline is GREEN today (and must remain GREEN after ``-r`` lands)."""

    def test_non_radical_name_still_parses(self):
        """RESEARCH -RADICAL-a: ``-r`` only widens acceptance of radical
        names; it never alters an existing parse. Anchor on a plain alkane:
        ``hexane`` -> a non-None SMILES via the same oracle namer builds for the
        validity gate (OpsinOracle(opsin_jar=_find_opsin_jar))."""
        jar = jar_or_skip()  # OPSIN jar not available -> additivity baseline skipped
        oracle = OpsinOracle(opsin_jar=jar)
        smi = oracle.name_to_smiles("hexane")
        assert smi is not None
        # sanity: it parses back to a 6-carbon acyclic skeleton
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        assert sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C") == 6
