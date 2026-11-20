"""The acid word of an ester of a ring acid cites its stereodescriptors at the ring's own locants (lane/c4, 2026-11-19).

 "General methodology" (the Blue Book): "All preferred IUPAC names for esters are named by functional
class nomenclature." "NAMING OF STEREOISOMERS" (:44639): stereodescriptors "do not change the name or the numbering
of a compound"; "When they relate to substituent groups, they are cited at the front of the corresponding prefix."

``rules.esters._collect_ester_fragment_stereo`` numbers the acid part by counting bonds out from the C=O carbon. For a
chain acid that is the chain numbering; for a ring acid it is neither the ring numbering nor confined to the ring:
'methyl (6R)-4-[(2R)-2,6-dihydroxy-6-methylheptan-2-yl]-3-hydroxybenzoate' cited the side chain's centre again at a count
of 6, and '(3-phenoxyphenyl)methyl (2S,3R)-...-2,2-dimethylcyclopropane-1-carboxylate' counted ring C1 as 2. The gate
refused both, so the PIN tier gave no name. The ring acid now takes the locants of the oriented ring of its word."""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("COC(=O)c1ccc([C@](C)(O)CCCC(C)(C)O)c(O)c1",
     "methyl 4-[(2R)-2,6-dihydroxy-6-methylheptan-2-yl]-3-hydroxybenzoate"),
    ("CC1(C)[C@@H](C=C(Cl)Cl)[C@@H]1C(=O)OCc1cccc(Oc2ccccc2)c1",
     "(3-phenoxyphenyl)methyl (1S,3R)-3-(2,2-dichloroethen-1-yl)-2,2-dimethylcyclopropane-1-carboxylate"),
    ("COC(=O)[C@@H]1CCC[C@H](C)C1", "methyl (1R,3S)-3-methylcyclohexane-1-carboxylate"),
    ("CCOC(=O)[C@@H]1CCCC[C@H]1C", "ethyl (1R,2R)-2-methylcyclohexane-1-carboxylate"),
    # unchanged: a ring acid whose centres the old count happened to number like the ring, and heterocyclic acids
    # (named on another path, the heteroatom keeping locant 1)
    ("COC(=O)[C@@H]1CC[C@H](C)O1", "methyl (2S,5S)-5-methyloxolane-2-carboxylate"),
    ("COC(=O)[C@H]1CCC[C@@H](C)N1C(C)=O", "methyl (2R,6R)-1-acetyl-6-methylpiperidine-2-carboxylate"),
    ("COC(=O)[C@@H]1CCCNC1", "methyl (3R)-piperidine-3-carboxylate"),
    ("COC(=O)[C@@H]1C[C@H]1c1ccccc1", "methyl (1R,2R)-2-phenylcyclopropane-1-carboxylate"),
])
def test_a_ring_acid_ester_is_named_with_its_ring_stereo(smiles, name):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (name, "pin_verified"), (tier, row)
    assert_full_rt(name, smiles)


def test_a_side_chain_centre_no_prefix_cites_keeps_the_other_ester_name():
    """Review row r0106: the benzoate's prefix '2-{[(3R)-3-hydroxybutanoyl]oxy}propyl' cites no descriptor for its own
    propyl centre, so the ring locants alone would drop it; the bond count stays, the gate withdraws that word, and
    the other ester of the molecule names it, as before the ring-locant change."""
    smiles = "C[C@H](Cc1cc(O)cc(O)c1C(=O)O[C@H](C)Cc1cc2cc(O)cc(O)c2c(=O)o1)OC(=O)C[C@@H](C)O"
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (
        "(2R)-1-[2-({[(2R)-1-(6,8-dihydroxy-1-oxo-1H-2-benzopyran-3-yl)propan-2-yl]oxy}carbonyl)-3,5-dihydroxyphenyl]"
        "propan-2-yl (3R)-3-hydroxybutanoate", "pin_unverified"), row
