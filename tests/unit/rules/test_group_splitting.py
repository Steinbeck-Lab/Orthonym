"""Phase 169 Plan-02 Wave-0 unit suite — group-splitting polyfunctional rescue.

Covers: per-rule split (ester → oxo + R-oxy, thioester → oxo + R-sulfanyl),
the deny-path (functional-class FGs stay dropped — D-03), the FAIL-CLOSED
per-split RT reject (D-05), and the Stage-A flag-OFF no-op (D-05).

Mirrors tests/unit/assembly/test_retained_substitution.py (class-per-concern,
real-perception inputs — never hand-encoded atom tuples that drift).
"""
import pytest
from rdkit import Chem, RDLogger

from orthonym import name_compound
from orthonym.assembly.group_splitting import split_composite_fg, SplitComponent

RDLogger.logger().setLevel(RDLogger.ERROR)

# OPSIN-RT-verified targets (169-RESEARCH § Code Examples).
ESTER_LOSER_SMILES = "OC(=O)CCC(=O)OCC"      # monoethyl succinate -> 4-ethoxy-4-oxobutanoic acid
THIOESTER_LOSER_SMILES = "OC(=O)CCC(=O)SCC"  # S-ethyl monothiosuccinate -> 4-(ethylsulfanyl)-4-oxobutanoic acid

_ESTER_SMARTS = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
_THIOESTER_SMARTS = Chem.MolFromSmarts("[CX3](=O)[SX2][#6]")


def _first_match(smiles, smarts):
    """Real perception of the composite FG atoms (SMARTS match order:
    carbonyl_C, =O, linker, alkyl_C) — not a hand-encoded tuple."""
    mol = Chem.MolFromSmiles(smiles)
    matches = mol.GetSubstructMatches(smarts)
    return mol, matches[0]


class _RejectOracle:
    """Stub oracle whose rt_safe always rejects (simulates RT-unsafe / FAIL-CLOSED)."""

    def rt_safe(self, target_canon, candidate_name):
        return False


@pytest.mark.unit
class TestDenyFunctionalClass:
    """D-03: genuine functional-class FGs are NOT in the split table -> stay dropped."""

    @pytest.mark.parametrize("fg", [
        "secondary_amide", "tertiary_amide", "phosphate_diester",
        "anhydride", "imide", "thioether", "carbamate",
    ])
    def test_functional_class_not_split(self, fg):
        # The SPLIT_RULES membership check short-circuits before touching mol,
        # so a non-table FG returns None even with null structural args.
        assert split_composite_fg(fg, None, None, None) is None


@pytest.mark.unit
class TestEsterSplit:
    """POLY-01: ester loser -C(=O)-O-R decomposes to oxo + R-oxy (alkoxy)."""

    def test_ester_loser_yields_oxo_plus_alkoxy(self):
        mol, match = _first_match(ESTER_LOSER_SMILES, _ESTER_SMARTS)
        comps = split_composite_fg("ester", mol, match, None)
        assert comps is not None and len(comps) == 2
        forms = {c.prefix_form for c in comps}
        assert "oxo" in forms, forms
        assert any(f.endswith("oxy") and f != "oxo" for f in forms), forms  # e.g. "ethoxy"
        # Both components share the caller's central-carbon locant (locants=None).
        assert all(c.locants is None for c in comps)


@pytest.mark.unit
class TestThioesterSplit:
    """POLY-01: thioester loser -C(=O)-S-R decomposes to oxo + R-sulfanyl."""

    def test_thioester_loser_yields_oxo_plus_sulfanyl(self):
        mol, match = _first_match(THIOESTER_LOSER_SMILES, _THIOESTER_SMARTS)
        comps = split_composite_fg("thioester", mol, match, None)
        assert comps is not None and len(comps) == 2
        forms = {c.prefix_form for c in comps}
        assert "oxo" in forms, forms
        assert any(f.endswith("sulfanyl") for f in forms), forms  # e.g. "ethylsulfanyl"


@pytest.mark.unit
class TestRtReject:
    """D-05: an RT-unsafe split (oracle.rt_safe -> False / FAIL-CLOSED) is rejected."""

    def test_rt_unsafe_split_rejected(self):
        mol, match = _first_match(ESTER_LOSER_SMILES, _ESTER_SMARTS)
        # With a rejecting oracle, the split must be refused -> None (caller behaves as DROP-23).
        assert split_composite_fg("ester", mol, match, None, oracle=_RejectOracle()) is None


@pytest.mark.unit
class TestStageAInvariants:
    """D-05 Stage A: with group-splitting OFF (the production default), the ester
    stays dropped — the split token must be absent from the flag-OFF output."""

    # A REAL DROP-23 ester case (measured): the whole ester+diacid is dropped,
    # leaving just the short acid. (monoethyl succinate is NOT a DROP-23 case —
    # it is named "ethoxycarbonyl" via get_alkoxycarbonyl_prefix; RESEARCH Pitfall 4.)
    DROP23_ESTER_SMILES = "O=C(O)CCCC(=O)OCCCO"   # -> "pentanoic acid" (ester dropped)

    def test_enabled_false_is_noop(self):
        # Default (flag OFF, production default for Stage A): the ester stays
        # DROPPED — the status-quo output is preserved (byte-identical Stage-A
        # invariant at the name_compound level; the corpus-wide proof is the
        # verify_decomp_byte_identical canary gate).
        out = name_compound(self.DROP23_ESTER_SMILES)  # enable_group_splitting defaults False
        assert out == "pentanoic acid", out
