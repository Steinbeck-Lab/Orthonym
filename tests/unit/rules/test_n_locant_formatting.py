"""P-15.3.2.2.1 (N primes) / P-15.3.2.2.2 (superscript-arabic N) conformance
pins (Wave-2 P0c Task 9). BB the Blue Book. Evidence rows verified
OPSIN-RT 2026-07-09.
"""
import pytest

from orthonym.namer import name_compound


def _name(smiles):
    result = name_compound(smiles)
    return result if isinstance(result, str) else getattr(result, "name", str(result))


class TestP1532221Primes:
    def test_nn_prime_diethylurea(self):
        # One characteristic group, identical units -> primes on N.
        assert _name("CCNC(=O)NCC") == "N,N'-diethylurea"

    def test_unprimed_before_primed(self):
        # P-14.3.5: N cited before N' in the locant set (never N',N).
        name = _name("CCNC(=O)NCC")
        assert "N,N'-" in name and "N',N-" not in name


class TestP1532222SuperscriptArabic:
    def test_n1_n2_dimethylethanediamide(self):
        # Two characteristic groups -> N + parent-attachment numeral.
        assert _name("CNC(=O)C(=O)NC") == "N1,N2-dimethylethanediamide"

    def test_n1_before_n2(self):
        name = _name("CNC(=O)C(=O)NC")
        assert name.index("N1") < name.index("N2")


class TestBuilderUnitLevel:
    def test_build_n_substituted_name_primes(self):
        from orthonym.assembly.composer import _build_n_substituted_name
        assert _build_n_substituted_name(
            [("N", "ethyl"), ("N'", "ethyl")], "urea") == "N,N'-diethylurea"

    def test_build_n_substituted_single(self):
        # P-14.3.4.3 (the Blue Book): a monosubstituted urea omits the italic-N locant.
        from orthonym.assembly.composer import _build_n_substituted_name
        assert _build_n_substituted_name([("N", "methyl")], "urea") == \
            "methylurea"

    def test_build_n_substituted_single_chalcogen_keeps_locant(self):
        # The chalcogen analogue KEEPS its letter locant (P-66.1.6.1.3.1,:33451),
        # so the `== 'urea'` scoping must not fire for thiourea.
        from orthonym.assembly.composer import _build_n_substituted_name
        assert _build_n_substituted_name([("N", "methyl")], "thiourea") == \
            "N-methylthiourea"
