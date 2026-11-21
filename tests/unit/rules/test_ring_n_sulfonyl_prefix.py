"""An -SO2- group joined to a ring nitrogen, under a senior group, is the acyl prefix of the ring-N sulfonic acid.

 acyl groups derived from sulfonic acids (the Blue Book) ('-sulfonic acid' -> '-sulfonyl', 'benzenesulfonyl'); a ring nitrogen takes
the acid suffix as 'piperidine-1-carbonyl' does (the Blue Book, 'piperidine-1-carbohydrazide (PIN) [not
(piperidine-1-carbonyl)hydrazine]'). The shared writer declined a ring nitrogen, so these groups were spelled
'1-oxo-1-(pyrrolidin-1-yl)-2-oxa-1λ6-thiaeth-1-en-1-yl' (assembly.sulfur_acid_group_prefix._ring_n_sulfonyl).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("NC(=O)CS(=O)(=O)N1CCCC1", "2-(pyrrolidine-1-sulfonyl)acetamide"),
    ("OC(=O)CCS(=O)(=O)N1CCOCC1", "3-(morpholine-4-sulfonyl)propanoic acid"),
    ("CC(=O)Nc1ccc(S(=O)(=O)N2CCN(C)CC2)cc1", "N-[4-(4-methylpiperazine-1-sulfonyl)phenyl]acetamide"),
    ("OC(=O)c1ccc(S(=O)(=O)N2CCCC2)cc1", "4-(pyrrolidine-1-sulfonyl)benzoic acid"),
    ("NC(=O)c1ccc(S(=O)(=O)N2CCOCC2)cc1", "4-(morpholine-4-sulfonyl)benzamide"),
    ("OC(=O)c1ccc2cc(S(=O)(=O)N3CCCC3)ccc2n1", "6-(pyrrolidine-1-sulfonyl)quinoline-2-carboxylic acid"),
    ("OC(=O)c1ccc2cc(S(=O)(=O)N3CCCC3)ccc2c1", "6-(pyrrolidine-1-sulfonyl)naphthalene-2-carboxylic acid"),
    # nested enclosing marks
    ("CC(=O)Nc1ccc(S(=O)(=O)N2CCCCC2c2nc3ccccc3s2)cc1",
     "N-{4-[2-(1,3-benzothiazol-2-yl)piperidine-1-sulfonyl]phenyl}acetamide"),
    # (c) (the Blue Book): a substituted group takes 'bis'; a plain one 'di' ('di(benzenesulfinyl)acetic
    # acid (PIN)',:7324)
    ("OC(=O)c1cc(S(=O)(=O)N2CCN(C)CC2)cc(S(=O)(=O)N2CCN(C)CC2)c1", "3,5-bis(4-methylpiperazine-1-sulfonyl)benzoic acid"),
    ("OC(=O)c1ccc(S(=O)(=O)N2CCCC2)cc1S(=O)(=O)N1CCCC1", "2,4-di(pyrrolidine-1-sulfonyl)benzoic acid"),
    # a sulfonamide N in a ring has no suffix form (the 'hidden amide' is a pseudoketone,, the Blue Book),
    # so the senior ring system is the parent,:19408): more rings (fasudil) or a heterocycle (pyrrolidine)
    ("O=S(=O)(c1cccc2cnccc12)N1CCCNCC1", "5-(1,4-diazepane-1-sulfonyl)isoquinoline"),
    ("O=S(=O)(c1ccc2ncccc2c1)N1CCCC1", "6-(pyrrolidine-1-sulfonyl)quinoline"),
    ("O=S(=O)(c1ccc2ccccc2c1)N1CCCC1", "1-(naphthalene-2-sulfonyl)pyrrolidine"),
    # (c) applies "in the absence of nitrogen": with N in both, more skeletal atoms (e) decide
    ("O=S(=O)(c1cocn1)N1CCNCC1", "1-(1,3-oxazole-4-sulfonyl)piperazine"),
    # the sulfonamide as the only group: the ring N is acylated, unchanged
    ("Cc1ccc(S(=O)(=O)N2CCCC2)cc1", "1-(4-methylbenzene-1-sulfonyl)pyrrolidine"),
])
def test_ring_n_sulfonyl_prefix_pins(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [
    "OC(=O)c1ccc(NS(=O)(=O)N2CCCC2)cc1",          # -NH-SO2-: the amido prefix, not built here
    "OC(=O)c1ccc(S(=O)(=O)N2CCS(=O)(=O)CC2)cc1",  # a lambda-convention ring atom: not built here
])
def test_other_shapes_get_no_ring_n_sulfonyl_prefix(smiles):
    row = tier_row(smiles, "best-effort")
    assert "-1-sulfonyl" not in (row["name"] or "") and "-4-sulfonyl" not in (row["name"] or ""), row
