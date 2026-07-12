"""P-66.1.6.1.2.1 - N/N'-substituted carbamimidate citation (W2F p6 Task 5).

BB P-66.1.6.1.2.1 (BlueBookV2.md:33408): 'ethyl N'-methyl-N,N-diphenyl-
carbamimidate (PIN)'. N = amino (sp3) N; N' = imino (=N-) N; citation order
alphanumerical (methyl < phenyl). Un-nameable N-fragment -> fail closed.
"""
import orthonym
from rdkit import Chem
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.assembly.handlers.imidate import _name_carbamimidate


class TestCarbamimidateNSub:
    def test_target(self):
        # BB P-66.1.6.1.2.1 verbatim PIN (BlueBookV2.md:33408)
        assert orthonym.name_compound("CCOC(=NC)N(c1ccccc1)c1ccccc1", style="pin") == \
            "ethyl N'-methyl-N,N-diphenylcarbamimidate"

    def test_n_prime_methyl(self):
        # methyl on the imino N -> N'
        assert orthonym.name_compound("CCOC(=NC)N", style="pin") == \
            "ethyl N'-methylcarbamimidate"

    def test_n_methyl(self):
        # methyl on the amino N -> N
        assert orthonym.name_compound("CCOC(=N)NC", style="pin") == \
            "ethyl N-methylcarbamimidate"

    def test_unsubstituted_regression(self):
        assert orthonym.name_compound("COC(N)=N", style="pin") == "methyl carbamimidate"
        assert orthonym.name_compound("CCOC(N)=N", style="pin") == "ethyl carbamimidate"

    def test_unnameable_n_fragment_fails_closed(self):
        # Fail-closed boundary at the HANDLER (JAR-independent; the name_compound
        # validity gate is a subprocess/OPSIN backstop that fails OPEN in a bare
        # test runner). A phosphono N-substituent is un-nameable: name_substituent
        # returns the 'substituent' sentinel, so _name_carbamimidate must refuse
        # (return None) rather than emit a truncated/wrong name.
        # NOTE (reproduce-first re-anchor): the plan's original probe
        # -N=[Si](C)(C)C is NOT a fail-closed case -- 'trimethylsilyl' is
        # nameable and 'ethyl N'-trimethylsilylcarbamimidate' round-trips in
        # OPSIN (see test_silyl_n_prime_is_nameable_not_refused).
        mol = Chem.MolFromSmiles("CCOC(=NP(=O)(O)O)N")
        match = detect_functional_groups(mol)["carbamimidate"][0]
        assert _name_carbamimidate(mol, match) is None

    def test_silyl_n_prime_is_nameable_not_refused(self):
        # Documents the reproduce-first finding: a trimethylsilyl imino-N
        # substituent IS nameable and RT-valid (OPSIN RT verified), so it is
        # emitted, not refused.
        name = orthonym.name_compound("CCOC(=N[Si](C)(C)C)N", style="pin")
        assert name.startswith("ethyl ") and name.endswith("carbamimidate")
        assert "silyl" in name
