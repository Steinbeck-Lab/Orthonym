"""a phase Plan-02 Wave-0 unit suite — group-splitting polyfunctional rescue.

Covers: per-rule split (ester → oxo + R-oxy, thioester → oxo + R-sulfanyl),
the deny-path (functional-class FGs stay dropped —), the FAIL-CLOSED
per-split RT reject , and the Stage-A flag-OFF no-op .

Mirrors tests/unit/assembly/test_retained_substitution.py (class-per-concern,
real-perception inputs — never hand-encoded atom tuples that drift).
"""
import pytest
from rdkit import Chem, RDLogger

from orthonym import name_compound
from orthonym.assembly.group_splitting import split_composite_fg, SplitComponent

RDLogger.logger().setLevel(RDLogger.ERROR)

# OPSIN-RT-verified targets (internal notes § Code Examples).
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
    """: genuine functional-class FGs are NOT in the split table -> stay dropped."""

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
    """: ester loser -C(=O)-O-R decomposes to oxo + R-oxy (alkoxy)."""

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
    """: thioester loser -C(=O)-S-R decomposes to oxo + R-sulfanyl."""

    def test_thioester_loser_yields_oxo_plus_sulfanyl(self):
        mol, match = _first_match(THIOESTER_LOSER_SMILES, _THIOESTER_SMARTS)
        comps = split_composite_fg("thioester", mol, match, None)
        assert comps is not None and len(comps) == 2
        forms = {c.prefix_form for c in comps}
        assert "oxo" in forms, forms
        assert any(f.endswith("sulfanyl") for f in forms), forms  # e.g. "ethylsulfanyl"


@pytest.mark.unit
class TestRtReject:
    """: an RT-unsafe split (oracle.rt_safe -> False / FAIL-CLOSED) is rejected."""

    def test_rt_unsafe_split_rejected(self):
        mol, match = _first_match(ESTER_LOSER_SMILES, _ESTER_SMARTS)
        # With a rejecting oracle, the split must be refused -> None (caller behaves as).
        assert split_composite_fg("ester", mol, match, None, oracle=_RejectOracle()) is None


@pytest.mark.unit
class TestStageAInvariants:
    """ Stage A: W2F-P2 retired the flag-OFF byte-identity guarantee for the
    narrow class (PG=carboxylic_acid + ester/thioester) — that class
    now splits by default, per-candidate OPSIN-RT gated. Non-class FGs stay
    dropped flag-OFF (covered by TestFunctionalClassNotSplit / TestDenyFunctionalClass)."""

    # A REAL ester case (measured): the whole ester+diacid is dropped,
    # leaving just the short acid. (monoethyl succinate is NOT a case —
    # it is named "ethoxycarbonyl" via get_alkoxycarbonyl_prefix; RESEARCH Pitfall 4.)
    DROP23_ESTER_SMILES = "O=C(O)CCCC(=O)OCCCO"   # W2F-P2: now heals by default

    # W2F-P2: Stage-A flag-OFF byte-identity is intentionally RETIRED for the
    # narrow class (PG=carboxylic_acid + ester/thioester): the
    # split now runs by default, per-candidate OPSIN-RT gated. This molecule
    # was the measured witness ("pentanoic acid"); it now names fully.
    def test_default_split_heals_measured_drop23_witness(self):
        out = name_compound(self.DROP23_ESTER_SMILES)  # no flag
        assert out == "5-(3-hydroxypropoxy)-5-oxopentanoic acid", out


@pytest.mark.unit
class TestDecomposeChainMembershipGuard:
    """W2F-P2 Task 2: the decomposition's oxo locant is only
    meaningful for a CHAIN-MEMBER carbonyl. With a chain provided, off-chain
    carbonyls decline; principal_chain=None keeps pure-decomposition mode."""

    def test_carbonyl_not_in_provided_chain_declines(self):
        mol, match = _first_match(ESTER_LOSER_SMILES, _ESTER_SMARTS)
        off_chain = [a for a in range(mol.GetNumAtoms()) if a != match[0]][:3]
        assert split_composite_fg("ester", mol, match, off_chain) is None

    def test_carbonyl_in_provided_chain_decomposes(self):
        mol, match = _first_match(ESTER_LOSER_SMILES, _ESTER_SMARTS)
        comps = split_composite_fg("ester", mol, match, [match[0]])
        assert comps is not None and len(comps) == 2

    def test_none_chain_still_decomposes(self):
        # The documented unit-mode contract (TestEsterSplit above) must hold.
        mol, match = _first_match(ESTER_LOSER_SMILES, _ESTER_SMARTS)
        assert split_composite_fg("ester", mol, match, None) is not None


@pytest.mark.unit
class TestDefaultOracle:
    """W2F-P2 Task 2: lazy module-level OpsinOracle for the narrow default-ON
    split branch (Task 3). Jar-missing -> rt_safe False -> fail-closed ."""

    def test_singleton_identity(self):
        from orthonym.assembly.group_splitting import _get_default_oracle
        a = _get_default_oracle()
        b = _get_default_oracle()
        assert a is not None and a is b

    def test_rt_safe_fail_closed_without_jar(self):
        from orthonym.assembly.retained_substitution import OpsinOracle
        oracle = OpsinOracle(opsin_jar=None)
        assert oracle.rt_safe("CCO", "ethanol") is False
