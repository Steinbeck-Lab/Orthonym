""" breadth — the general-engine substituent namer must name an O-rooted alkoxy substituent
`-O-R` as `Roxy` (methoxy/ethoxy/phenoxy), NOT `hydroxy(alkyl)`.

Root cause (L1spy L5): `name_substituent` normalizes the attach atom to the ether O, then
`carbon_free_valence_prefix` declines (non-carbon root) and the cascade names the fragment as a
carbon-rooted `hydroxymethyl` — a DIFFERENT molecule (`-CH2OH` vs `-O-CH3`). The PIN path is
correct (`methoxybenzene`); only the general-engine `name_substituent` path swapped it. Wrong
connectivity → abstains → breadth loss + latent wrong-molecule. Fix: O-rooted ether ->
`get_alkoxy_prefix`. O-rooted ether routes through the alkoxy prefix builder.
"""
from rdkit import Chem
from orthonym.assembly.substituent_enumerator import name_substituent


def test_methoxy_substituent():
    m = Chem.MolFromSmiles("COc1ccccc1")   # C0-O1-c2(ring); frag {C0,O1}, root O1
    assert name_substituent(m, {0, 1}, 1, allow_mancude=True) == "methoxy"


def test_ethoxy_substituent():
    m = Chem.MolFromSmiles("CCOc1ccccc1")  # C0-C1-O2-c3; frag {C0,C1,O2}, root O2
    assert name_substituent(m, {0, 1, 2}, 2, allow_mancude=True) == "ethoxy"


def test_propoxy_substituent():
    m = Chem.MolFromSmiles("CCCOc1ccccc1")
    assert name_substituent(m, {0, 1, 2, 3}, 3, allow_mancude=True) == "propoxy"


def test_no_hydroxymethyl_for_ether():
    # regression guard: the wrong token must not come back for a true ether.
    m = Chem.MolFromSmiles("COc1ccccc1")
    assert name_substituent(m, {0, 1}, 1, allow_mancude=True) != "hydroxymethyl"


def test_real_hydroxymethyl_unchanged():
    # a genuine -CH2-OH (carbon-rooted, O is a pendant hydroxyl) stays hydroxymethyl.
    m = Chem.MolFromSmiles("OCc1ccccc1")   # O0-C1-c2(ring); frag {O0,C1}, root C1
    assert name_substituent(m, {0, 1}, 1, allow_mancude=True) == "hydroxymethyl"


def test_peroxide_not_alkoxyoxy():
    # -O-O-CH3 is a PEROXIDE (frag-side of the root O is another O, not C). The intercept
    # must NOT fire (its PIN is (methylperoxy), not the (methoxyoxy) that get_alkoxy_prefix
    # would build) -- guards the (ethylperoxy)benzene / (methylperoxy) golds.
    m = Chem.MolFromSmiles("COOc1ccccc1")  # C0-O1-O2-c3(ring); frag {C0,O1,O2}, root O2
    got = name_substituent(m, {0, 1, 2}, 2, allow_mancude=True)
    assert got != "(methoxyoxy)", got
