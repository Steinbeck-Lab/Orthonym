""" — enclose compound N-substituents on urea / guanidine.

``_build_n_substituted_name`` built ``f"{locant}-{name}"`` with the RAW
substituent name, never enclosing a COMPOUND N-substituent (one bearing its own
locants). ``COc1ccc(NC(=O)N(C)C)cc1Cl`` therefore emitted
``N'-3-chloro-4-methoxyphenyl-N,N-dimethylurea`` (a numeral abutting the ``N,N``),
which OPSIN cannot parse -> -> ``unknown``.

The enclosed urea derivative IS a preferred IUPAC name urea is
the PIN; N-substituted derivatives are substitution products; BB
33336 ``N-[1-cyano-3-(methylsulfanyl)propyl]-N'-methylurea (PIN)`` witnesses the
bracketed compound N-substituent), so the fix belongs on the PIN-default path.

Simple N-substituents stay unenclosed; a MONOsubstituted urea additionally omits
the italic-N locant, the Blue Book `methylurea`), so `phenylurea`, while
DIsubstituted forms keep their letter locants (`N,N-dimethylurea`).
"""
from orthonym.assembly.composer import _build_n_substituted_name
from orthonym.namer import name_compound


# --- pure-function contract (authoritative; no OPSIN) ----------------------

def test_compound_n_substituent_enclosed():
    tagged = [
        ("N'", "3-chloro-4-methoxyphenyl"),
        ("N", "methyl"),
        ("N", "methyl"),
    ]
    assert _build_n_substituted_name(tagged, "urea") == \
        "N'-(3-chloro-4-methoxyphenyl)-N,N-dimethylurea"


def test_simple_n_substituents_unchanged():
    # regression: simple names never enclosed. A MONOsubstituted urea also omits
    # the italic-N locant,:2943); the DIsubstituted form keeps both.
    assert _build_n_substituted_name([("N", "phenyl")], "urea") == "phenylurea"
    assert _build_n_substituted_name(
        [("N", "methyl"), ("N", "methyl")], "urea") == "N,N-dimethylurea"
    assert _build_n_substituted_name(
        [("N", "methyl")], "guanidine") == "N-methylguanidine"


# --- full name via the gate-on namer (also RT-verifies) --------------------

def test_full_name_urea_compound_aryl():
    assert name_compound("COc1ccc(NC(=O)N(C)C)cc1Cl") == \
        "N'-(3-chloro-4-methoxyphenyl)-N,N-dimethylurea"


def test_full_name_urea_simple_regression():
    # Monosubstituted urea omits the italic-N locant,:2943).
    assert name_compound("NC(=O)Nc1ccccc1") == "phenylurea"
    assert name_compound("CN(C)C(=O)N") == "N,N-dimethylurea"
