"""A substituted metallocene drawn as a metal(2+) and two cyclopentadienide anions is named with its ocenyl prefix
through its cyclopentyl drawing (``assembly.ocene_drawing``); OPSIN reads 'ferrocen-1-yl...' to the same drawing.
 (alphanumerical order) guards the rewrite; encloses the prefix, which has a locant."""
import pytest

from orthonym.assembly.ocene_drawing import rewrite_surrogate_prefix
from tests.support.rt_assert import assert_full_rt, name_best_effort
from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.unit


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("[CH-]1C=CC=C1.C1=C[C-](C=C1)OC(=O)C2=CC=NC=C2.[Fe+2]", "ferrocen-1-yl pyridine-4-carboxylate"),
    ("CC1=CC(=CS1)[C-]2C=CC=C2.[CH-]1C=CC=C1.[Fe+2]", "4-(ferrocen-1-yl)-2-methylthiophene"),
    ("COC1=CC=C(C=C1)C(=O)C#CC2=CC=C[CH-]2.[CH-]1C=CC=C1.[Fe+2]",
     "3-(ferrocen-1-yl)-1-(4-methoxyphenyl)prop-2-yn-1-one"),
    ("C1CCN(CC1)C[C-]2C=CC=C2.[CH-]1C=CC=C1.[Fe+2]", "1-[(ferrocen-1-yl)methyl]piperidine"),
    ("C1CN(CCN1CC2=C[CH-]C=C2)C3=NC=CC=N3.[CH-]1C=CC=C1.[Fe+2]",
     "2-{4-[(ferrocen-1-yl)methyl]piperazin-1-yl}pyrimidine"),
])
def test_ocenyl_names(smiles, name):
    row = name_best_effort(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("name,expected", [
    ("4-cyclopentyl-2-methylthiophene", "4-(ferrocen-1-yl)-2-methylthiophene"),
    ("cyclopentyl pyridine-4-carboxylate", "ferrocen-1-yl pyridine-4-carboxylate"),
    # a word that sorts between 'cyclopentyl' and 'ferrocenyl' (a compound prefix with its multiplier): at depth 0
    # the prefixes are cited again in order
    ("2-cyclopentyl-3-(2,6-dimethylanilino)pyridine", "3-(2,6-dimethylanilino)-2-(ferrocen-1-yl)pyridine"),
    # inside a compound prefix: the marks around it step up, the Blue Book)
    ("(cyclopentylmethyl)benzene", "[(ferrocen-1-yl)methyl]benzene"),
    ("2-[4-(cyclopentylmethyl)piperazin-1-yl]pyrimidine", "2-{4-[(ferrocen-1-yl)methyl]piperazin-1-yl}pyrimidine"),
    ("1-bromo-4-(cyclopentylsulfanyl)benzene", "1-bromo-4-[(ferrocen-1-yl)sulfanyl]benzene"),
    ("1-cyclopentyloxybenzene", None),                   # an unenclosed compound prefix
    ("cyclopentylidenemethanol", None),                  # another free valence
    ("{[(cyclopentylmethyl)amino]methyl}benzene", None), # the marks would need a level above { }
])
def test_the_rewrite_guards(name, expected):
    assert rewrite_surrogate_prefix(name, "ferrocen") == expected


@pytest.mark.parametrize("surrogate_name, name", [
    #: the ocenyl prefix takes its place among the depth-0 prefixes
    ("2-cyclopentyl-3-(2,6-dimethylanilino)-6-methylimidazo[1,2-a]pyridine",
     "3-(2,6-dimethylanilino)-2-(ferrocen-1-yl)-6-methylimidazo[1,2-a]pyridine"),
    ("1-cyclopentyl-4-(ethylamino)piperazine", "4-(ethylamino)-1-(ferrocen-1-yl)piperazine"),
    # hydro prefixes stay after the alphabetized ones
    ("6-cyclopentyl-4-phenyl-3,4-dihydropyrimidin-2(1H)-one", "6-(ferrocen-1-yl)-4-phenyl-3,4-dihydropyrimidin-2(1H)-one"),
    ("4-cyclopentylmorpholine", "4-(ferrocen-1-yl)morpholine"),
    ("(2E)-3-cyclopentyl-1-phenylprop-2-en-1-one", "(2E)-3-(ferrocen-1-yl)-1-phenylprop-2-en-1-one"),
    # the cyclopentane parent is the ocene: (the Blue Book), 'ferrocenecarbaldehyde' (:3963)
    ("cyclopentanecarboxylic acid", "ferrocenecarboxylic acid"),
    ("cyclopentanol", "ferrocenol"),
    # an unenclosed compound prefix is not rewritten; a locanted cyclopentane parent keeps its locants (the ocene
    # ring is numbered as cyclopentane,
    ("1-cyclopentyloxybenzene", None),
    ("1,2-dimethylcyclopentane", "1,2-dimethylferrocene"),
    # a prefix already enclosed alone keeps its marks; a multiplied prefix after it is another prefix
    ("di(adamantan-1-yl)(cyclopentyl)phosphane", "di(adamantan-1-yl)(ferrocen-1-yl)phosphane"),
    ("cyclopentyldi(phenyl)phosphane", "(ferrocen-1-yl)di(phenyl)phosphane"),
], ids=lambda v: v)
def test_rewrite_cites_the_ocene_in_its_place(surrogate_name, name):
    from orthonym.assembly.ocene_drawing import rewrite_surrogate_prefix
    assert rewrite_surrogate_prefix(surrogate_name, "ferrocen") == name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", [
    ("[CH-]1C=CC=C1.Cc1ccc2nc(C3=CC=C[CH-]3)c(Nc3c(C)cccc3C)n2c1.[Fe+2]",
     "3-(2,6-dimethylanilino)-2-(ferrocen-1-yl)-6-methylimidazo[1,2-a]pyridine"),
    ("[CH-]1C=CC=C1.C1COCCN1C1=CC=C[CH-]1.[Fe+2]", "4-(ferrocen-1-yl)morpholine"),
], ids=lambda v: v)
def test_ocenes_named_after_the_new_order(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    assert row["tier"] == "systematic_verified", row


@pytest.mark.parametrize("name,groups,expected", [
    # more groups on the ring: the substituted 'cyclopentyl' is position 1 of the ocenyl prefix, its marks exist
    ("N,N-dimethyl-1-[2-(phenylselanyl)cyclopentyl]ethan-1-amine", 2,
     "N,N-dimethyl-1-[2-(phenylselanyl)ferrocen-1-yl]ethan-1-amine"),
    ("2-methylcyclopentane-1-carboxylic acid", 2, "2-methylferrocene-1-carboxylic acid"),
    ("2-methylcyclopentyl benzoate", 2, None),          # a substituted ring needs its marks
])
def test_rewrite_of_a_ring_with_more_groups(name, groups, expected):
    assert rewrite_surrogate_prefix(name, "ferrocen", groups) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    # (the Blue Book): the positions of the second ring are primed ('1'-methylruthenocene',:40139)
    ("C[C-]1C=CC=C1.C[C-]1C=CC=C1.[Fe+2]", "1,1'-dimethylferrocene"),
    ("CC(C)(C)[C-]1C=CC(C)=C1.C[C-]1C=CC=C1.[Fe+2]", "1-tert-butyl-1',3-dimethylferrocene"),
    ("OC(=O)[C-]1C=CC=C1.OC(=O)[C-]1C=CC=C1.[Fe+2]", "ferrocene-1,1'-dicarboxylic acid"),
    ("OC(=O)[C-]1C=CC=C1.C[C-]1C=CC=C1.[Fe+2]", "1'-methylferrocene-1-carboxylic acid"),
    # the same group of another parent on both rings: the multiplicative name, the book's own PIN (:40133)
    ("CC(=O)[C-]1C=CC=C1.CC(=O)[C-]1C=CC=C1.[Fe+2]", "1,1'-(ferrocene-1,1'-diyl)di(ethan-1-one)"),
    ("OC[C-]1C=CC=C1.OC[C-]1C=CC=C1.[Fe+2]", "(ferrocene-1,1'-diyl)dimethanol"),
    ("c1ccc(cc1)P(c1ccccc1)[C-]1C=CC=C1.c1ccc(cc1)P(c1ccccc1)[C-]1C=CC=C1.[Fe+2]",
     "(ferrocene-1,1'-diyl)bis[di(phenyl)phosphane]"),
    ("C1=C[C-](C=C1)C2=NN=CC=C2.C1=C[C-](C=C1)C2=NN=CC=C2.[Fe+2]", "3,3'-(ferrocene-1,1'-diyl)dipyridazine"),
    # the other ring's groups as primed prefixes on the ocenyl prefix
    ("[CH-]1C=CC(=C1)Cc1ccccc1.[CH-]1C=CC(=C1)Cc1ccccc1.[Fe+2]", "[(1'-benzylferrocen-1-yl)methyl]benzene"),
    # two groups on one ring, a nitro group (a bonded N+/O- pair) in a group
    ("CC([C-]1C=CC=C1[Se]CC2=CC=CC=C2)N(C)C.[CH-]1C=CC=C1.[Fe+2]",
     "1-[2-(benzylselanyl)ferrocen-1-yl]-N,N-dimethylethan-1-amine"),
    ("[CH-]1C=CC=C1.C1=C[C-](C(=C1)NC2=CC=C(C=C2)[N+](=O)[O-])O.[Fe+2]", "2-(4-nitroanilino)ferrocen-1-ol"),
    # a group of one ring that has no prefix name here (an acyl group, 'ring_prefixes' is None) cannot be cited on the
    # other ring, but the ring is still the ocenyl prefix of the ketone: only its number of groups counts
    ("CC(C)(C)CC1=CC=C[CH-]1.CC(C)(C)C(=O)C1=CC=C[CH-]1.[Fe+2]",
     "1-[1'-(2,2-dimethylpropyl)ferrocen-1-yl]-2,2-dimethylpropan-1-one"),
    ("CC(C)(C)CC1=CC=C[CH-]1.CC(C)(C)C(=O)C1=CC=C[CH-]1.[Ru+2]",
     "1-[1'-(2,2-dimethylpropyl)ruthenocen-1-yl]-2,2-dimethylpropan-1-one"),
])
def test_ocenes_with_more_groups(smiles, name):
    row = name_best_effort(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [
    "C1CCC2=C(C1)[CH-]C=C2.C1CCC2=C(C1)[CH-]C=C2.[Ru+2]",           # an indenide: another ligand (eta names)
    "[CH-]1C=CC=C1.[CH-]1C=CC=C1[C-]2C=C3C=CC(=CC3=[O+]2)O.[Fe+2]",  # a C-/O+ pair is not a nitro-like pair
])
def test_out_of_scope_drawings_are_not_rewritten(smiles):
    from orthonym.assembly.ocene_drawing import _surrogate
    assert _surrogate(smiles) is None


def test_a_ring_whose_group_has_no_prefix_name_is_still_the_ocenyl_ring():
    """``two_ring_candidates`` took the number of groups of the ocenyl ring from the list of their prefix names, which
    is None for a ring with an acyl group: ``len(None)`` aborted the naming of 61 molecules. The ring's own count
    (``_surrogate``) stands in for it."""
    from orthonym import Orthonym
    from orthonym.assembly.ocene_drawing import _surrogate
    from orthonym.assembly.ocene_two_rings import ring_prefixes, two_ring_candidates
    from orthonym.cli import _emit_tier_flags
    stem, rings = _surrogate("CC(C)(C)CC1=CC=C[CH-]1.CC(C)(C)C(=O)C1=CC=C[CH-]1.[Fe+2]")
    assert sorted(r[3] for r in rings) == [1, 1]
    assert [ring_prefixes(mol, ring) is None for mol, ring, _s, _g in rings].count(True) == 1
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    assert two_ring_candidates(namer, stem, rings) == ["1-[1'-(2,2-dimethylpropyl)ferrocen-1-yl]-2,2-dimethylpropan-1-one"]


def test_a_failing_ocene_drawing_does_not_abort_the_naming(monkeypatch):
    """The drawing hook is a way to a name, never a condition of one: an exception inside it leaves the input to the
    caller, which names it as it names every other input."""
    from orthonym.assembly import ocene_drawing

    def broken(namer, smiles):
        raise TypeError("object of type 'NoneType' has no len()")

    monkeypatch.setattr(ocene_drawing, "ocene_drawing_row", broken)
    row = name_best_effort("CC(C)(C)C[C-]1C=CC=C1.CC(C)(C)C(=O)[C-]1C=CC=C1.[Fe+2]")
    assert row.get("source") != "ocene_drawing", row
    assert row.get("name"), row
