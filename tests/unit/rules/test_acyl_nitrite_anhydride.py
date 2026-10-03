"""Mixed anhydrides of carboxylic acids with nitrous and nitric acid, and the acid names of the
anhydride producer.

 Mixed anhydrides (the Blue Book): "Anhydrides derived from different monobasic
acids are named by citing in alphabetical order the names of the two acids, substituted or
unsubstituted" (:32255); "Mixed anhydrides with carbonic acid, cyanic acid, and inorganic acids
are named as anhydrides" (:32272); 'acetic cyanic anhydride (PIN)' (:32276), 'benzoic
phosphinous anhydride (PIN)' (:32280, an '-ous' inorganic acid), 'chloroacetic
4-nitrobenzene-1-sulfonic anhydride (PIN)' (:32270: the alphabetical order does not count the
locant). R-CO-O-N=O was named as an ester ('acetyl nitrite', pin_verified), the acid component of
a ring acid lost its substituents ('benzoic' for 4-nitrobenzoic, a different molecule, stopped
only by the read-back) and a nitrile carbon was counted into the acid chain ('propanoic' for
cyanoacetic, the N dropped;,:34734: the -CN group is the prefix 'cyano' under a
senior group). Every name below reads back by OPSIN 2.9.0 to the input's full InChIKey.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,pin", [
    ("CC(=O)ON=O", "acetic nitrous anhydride"),
    ("CCC(=O)ON=O", "nitrous propanoic anhydride"),
    ("O=C(ON=O)c1ccccc1", "benzoic nitrous anhydride"),
    ("O=CON=O", "formic nitrous anhydride"),
    ("O=C(ON=O)c1cccnc1", "nitrous pyridine-3-carboxylic anhydride"),
    ("O=C(O[N+](=O)[O-])c1ccncc1", "nitric pyridine-4-carboxylic anhydride"),
    ("N#CCC(=O)O[N+](=O)[O-]", "cyanoacetic nitric anhydride"),
    ("Cc1ccc(C(=O)O[N+](=O)[O-])cc1", "4-methylbenzoic nitric anhydride"),
    ("N#CCC(=O)OC(C)=O", "acetic cyanoacetic anhydride"),
    ("N#CCC(=O)OC(=O)CC#N", "bis(cyanoacetic) anhydride"),
    ("N#CCCC(=O)OC(C)=O", "acetic 3-cyanopropanoic anhydride"),
    ("CC(C)C(=O)OC(C)=O", "acetic 2-methylpropanoic anhydride"),
    ("O=C(OC(=O)CCl)c1ccc([N+](=O)[O-])cc1", "chloroacetic 4-nitrobenzoic anhydride"),
    ("O=C(OC(=O)c1ccc([N+](=O)[O-])cc1)c1ccc([N+](=O)[O-])cc1",
     "bis(4-nitrobenzoic) anhydride"),
    ("Cc1ccccc1C(=O)OC(C)=O", "acetic 2-methylbenzoic anhydride"),
])
def test_mixed_anhydride_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", [
    ("CCCCCON=O", "pentyl nitrite"),              # an alkyl ester stays an ester
    ("CC(=O)O[N+](=O)[O-]", "acetic nitric anhydride"),
    ("CC(=O)OC#N", "acetic cyanic anhydride"),
    ("ClCC(=O)OC(=O)CCl", "bis(chloroacetic) anhydride"),
    ("O=C(OC(=O)c1ccccc1)c1ccccc1", "benzoic anhydride"),
])
def test_controls_unchanged(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,name", [
    ("N#CC(=O)OC(=O)C#N", "bis(cyanomethanoic) anhydride"),
    ("CC(=O)OC(=O)C#N", "acetic cyanomethanoic anhydride"),
    ("CC(=O)OC(=O)Cl", "acetic chloromethanoic anhydride"),
    ("[O-][N+](=O)C(=O)OC(C)=O", "acetic nitromethanoic anhydride"),
    ("N#CC(=O)O[N+](=O)[O-]", "cyanomethanoic nitric anhydride"),
])
def test_a_substituted_formic_acid_component_is_below_the_pin(smiles, name):
    # (the Blue Book): -Cl, -CN, -NH2... on formic acid are named from
    # carbonic acid ('NC-CO-OH carbonocyanidic acid (PIN)',:30832; 'NC-CO-O-CO-CN dicarbonic
    # dicyanide (PIN)',:34860); keeps 'formic' for the rest ('nitroformic acid
    # (PIN)',:30696). The 'methanoic' names read back exactly but are never the PIN: the
    # best-effort tier keeps them labelled systematic_verified, the PIN tier declines.
    from tests.support.pin_tiers import name_breadth, name_default
    from tests.support.rt_assert import name_is_rt_exact
    d = name_default(smiles)
    assert d.get("tier") != "pin_verified", d
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (name, "systematic_verified"), b
    assert name_is_rt_exact(name, smiles)
