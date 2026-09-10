""" (BB 33196): thioamide junior to acid is expressed
"(2) by the appropriate prefixes, such as amino, in conjunction with
sulfanylidene"; BB verbatim PIN: 3-amino-3-sulfanylidenepropanoic acid.
Mechanism mirrors: the chain-terminal thioamide C
stays IN the chain (it is NOT in chains._TERMINAL_C_FGS) and its N/=S
are cited as amino + sulfanylidene at the C's locant.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestThioamideChainPrefix:
    def test_amino_sulfanylidene_propanoic(self):
        assert name_compound("S=C(N)CC(=O)O", style="pin") \
            == "3-amino-3-sulfanylidenepropanoic acid"

    def test_thioamide_suffix_protect(self):
        assert name_compound("CC(=S)N", style="pin") == "ethanethioamide"

    def test_amidine_am4_protect(self):
        # the sibling amidine-on-chain block must be untouched
        assert name_compound("N=C(N)CCC(=O)O", style="pin") \
            == "4-amino-4-iminobutanoic acid"
