"""W2F-P8 Task 3: substituted-phenoxy substitutive builder (O-linked, P-45.6).

A benzene parent bearing an -O-(substituted aryl) substituent yields the
`(...phenoxy)` complex prefix (braces/brackets per P-16.5.4) instead of failing
closed. The parent ring is chosen by P-45.5.1 alphanumerical order over the two
candidate names (verified against BlueBookV2.md:22236 / :6273 'bromochloro <
bromophenyl'): for the methyl-vs-1-chloroethyl ether the chloroethyl ring is the
parent because 'chloroethyl...methyl...' precedes 'chloroethyl...phenoxy...'.
"""
import orthonym


def test_asymmetric_phenoxy_methyl():
    # methyl arm vs 1-chloroethyl arm: different constitution -> substitutive.
    # Parent = chloroethyl ring (P-45.5.1: '...methylphenoxy' < '...phenoxymethyl').
    assert orthonym.name_compound(
        "Cc1ccc(Oc2ccc(C(C)Cl)cc2)cc1", style="pin"
    ) == "1-(1-chloroethyl)-4-(4-methylphenoxy)benzene"


def test_bare_phenoxy_regression():
    # unsubstituted phenoxy ring still uses retained 'phenoxy' (P-29.6.1)
    assert orthonym.name_compound(
        "C(C)(Cl)c1ccc(Oc2ccccc2)cc1", style="pin"
    ) == "1-(1-chloroethyl)-4-phenoxybenzene"


def test_ethyl_phenoxy_no_octoxy_regression():
    # a pure-alkyl-decorated aryloxy must NOT be mis-collected as an alkoxy
    # chain ('octoxy'); it routes to the substituted-aryloxy builder.
    assert orthonym.name_compound(
        "ClC(C)c1ccc(Oc2ccc(CC)cc2)cc1", style="pin"
    ) == "1-(1-chloroethyl)-4-(4-ethylphenoxy)benzene"
