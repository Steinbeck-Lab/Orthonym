"""Acid salts of polybasic acids: the name keeps the charge on the ionized group.

 "Acid salts of polybasic organic acids are named in two ways"
(the Blue Book):
  :31593 "(1) by substitutive nomenclature in which the free acid is cited as a
          prefix to the name of the anion;"
  :31594 "(2) in the same way as the neutral salts, the remaining acid hydrogen
          atom(s) being indicated by the word 'hydrogen'..."
  :31596 "Method (1) generates preferred IUPAC names, except when the structure of
          the acid salt is unknown."
The anion is senior to the acid "SENIORITY ORDER FOR CLASSES", Table 4.1,
'4 Anions':18167 before '7 Acids':18170), and "CHOICE OF AN ANIONIC PARENT
STRUCTURE" (:41261) takes "(a) parent with the maximum number of anionic centers"
(:41265). So for the salt of the CH2 carboxylate of 2-(carboxymethyl)benzoic acid
the parent is the acetate: 'sodium (2-carboxyphenyl)acetate'. The name that used
to ship, 'sodium 2-(carboxymethyl)benzoate', makes the RING carboxylate the anion:
a different species with the same standard InChIKey.

The Blue Book draws its method (2) examples with the hydron as a bare H+ beside the
fully ionized anion (the structure of the acid salt unknown):
  :31610 "(2) sodium hydrogen 2-(carboxylatomethyl)benzoate (PIN)"
  :31612 "(2) potassium sodium hydrogen propane-1,2,3-tricarboxylate (PIN)"

Every name here is read back by a FRESH OPSIN call (tests/support/rt_assert.
_independent_parse), and for a known site its fixed-H InChI (RDKit) must be the
input's -- independent of the engine's own checks.
"""
import pytest

from tests.support.jars import jar_or_skip

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _fixed_h(smi):
    from rdkit import Chem
    from rdkit.Chem import inchi
    return inchi.MolToInchi(Chem.MolFromSmiles(smi), options="/FixedH /SNon")


def _key(smi):
    from rdkit import Chem
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smi))


def _pin(smiles):
    from orthonym import Orthonym
    return Orthonym(style="pin").name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", [
    ("O=C([O-])Cc1ccccc1C(=O)O.[Na+]", "sodium (2-carboxyphenyl)acetate"),
    ("O=C([O-])Cc1ccccc1C(=O)O", "(2-carboxyphenyl)acetate"),
    ("O=C(O)c1ccc(CC(=O)[O-])cc1.[Na+]", "sodium (4-carboxyphenyl)acetate"),
    ("CCCc1oc(CCC(=O)[O-])c(C(=O)O)c1C",
     "3-(3-carboxy-4-methyl-5-propylfuran-2-yl)propanoate"),
    # a chain parent: the ionized group ends the principal chain, every free -COOH
    # is a 'carboxy' prefix
    ("[O-]C(=O)CCC(C)C(=O)O.[Na+]", "sodium 4-carboxypentanoate"),
    ("OC(=O)CCC(C)C(=O)[O-].[Na+]", "sodium 4-carboxy-2-methylbutanoate"),
    ("OC(=O)C(C)CC(=O)[O-].[K+]", "potassium 3-carboxybutanoate"),
    # the other partial salt keeps its name: here the ring carboxylate IS the anion
    ("O=C(O)Cc1ccccc1C(=O)[O-].[Na+]", "sodium 2-(carboxymethyl)benzoate"),
    ("[K+].OC(=O)CCCCCC(=O)[O-]", "potassium 6-carboxyhexanoate"),
    # a three-carbon chain parent (not the retained acetate)
    ("OC(=O)C(C)C(=O)[O-].[Na+]", "sodium 2-carboxypropanoate"),
    # a ring parent: the ionized group is the '-carboxylate' suffix
    ("OC(=O)C1CCC(CC1)C(=O)[O-].[Na+]", "sodium 4-carboxycyclohexane-1-carboxylate"),
    ("OC(=O)C1CCCCC1C(=O)[O-].[Na+]", "sodium 2-carboxycyclohexane-1-carboxylate"),
])
def test_method_1_keeps_the_anion_site(smiles, name):
    jar_or_skip()
    from tests.support.rt_assert import _independent_parse
    r = _pin(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
    parsed = _independent_parse(name)
    assert parsed and _key(parsed) == _key(smiles) and _fixed_h(parsed) == _fixed_h(smiles)


@pytest.mark.parametrize("smiles,name", [
    ("[Na+].[H+].[O-]C(=O)Cc1ccccc1C(=O)[O-]",
     "sodium hydrogen 2-(carboxylatomethyl)benzoate"),
    ("[K+].[Na+].[H+].O=C([O-])CC(CC(=O)[O-])C(=O)[O-]",
     "potassium sodium hydrogen propane-1,2,3-tricarboxylate"),
    ("[K+].[H+].[O-]C(=O)CCCCCC(=O)[O-]", "potassium hydrogen heptanedioate"),
])
def test_method_2_for_an_open_hydron_site(smiles, name):
    """The input's H+ has no site, so the name cites it as 'hydrogen'; OPSIN reads
    the name to the input's full InChIKey."""
    jar_or_skip()
    from tests.support.rt_assert import _independent_parse
    r = _pin(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
    parsed = _independent_parse(name)
    assert parsed and _key(parsed) == _key(smiles)


def test_method_2_needs_a_remaining_anionic_site():
    """A proton for every acid site is the neutral acid, not an acid salt
    (the anion scope check of name_salt's method (2) branch)."""
    from rdkit import Chem
    from orthonym.rules.salts import _is_polybasic_oxoacid_anion
    dianion = Chem.MolFromSmiles("[O-]C(=O)Cc1ccccc1C(=O)[O-]")
    assert _is_polybasic_oxoacid_anion(dianion, 1)
    assert not _is_polybasic_oxoacid_anion(dianion, 2)
    assert not _is_polybasic_oxoacid_anion(Chem.MolFromSmiles("CC[O-]"), 1)
    assert not _is_polybasic_oxoacid_anion(Chem.MolFromSmiles("[O-]c1ccccc1[O-]"), 1)


@pytest.mark.parametrize("smiles,name", [
    # the hydrocarbyl group before 'hydrogen': (the Blue Book)
    # "the components present are cited in the order, cation, hydrocarbyl group,
    # hydrogen, anion";:35944 "sodium methyl hydrogen phosphate (PIN)"
    ("[Na+].[H+].[O-]P(=O)([O-])OC", "sodium methyl hydrogen phosphate"),
    ("[K+].[H+].[O-]P(=O)([O-])OCC", "potassium ethyl hydrogen phosphate"),
    #:31625 "CH3-P(O)(O-)2 K+ H+ potassium hydrogen methylphosphonate (PIN)"
    ("[K+].[H+].CP(=O)([O-])[O-]", "potassium hydrogen methylphosphonate"),
])
def test_bare_hydron_on_one_oxoacid_centre(smiles, name):
    jar_or_skip()
    from tests.support.rt_assert import _independent_parse
    r = _pin(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
    parsed = _independent_parse(name)
    assert parsed and _key(parsed) == _key(smiles)


def test_diphosphonate_acid_salt_keeps_a_salt_name():
    """ (the Blue Book): an organic derivative of a polybasic
    inorganic oxoacid takes method (2); the 'dihydrogen' word does not place the
    hydrons among the two equivalent phosphonic acid groups. The anion is the
    multiplicative one: the phosphonic acid is not a suffix Note
    :35457;:35461 "ethylphosphonic acid (PIN) (not ethanephosphonic acid)"), two
    of them on one linking group are multiplied (:5854 "[azanediylbis(methylene)]-
    bis(phosphonic acid) (PIN)",:7148 'bis(phosphonic acid)'), and -CH2- is
    'methylene' (:5188). The best-effort tier gives the same salt name, never the
    adduct form."""
    jar_or_skip()
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    from tests.support.rt_assert import _independent_parse
    smiles = "[Na+].[Na+].OP(=O)([O-])CP(=O)([O-])O"
    r = _pin(smiles)
    assert r["name"] == "disodium dihydrogen methylenebis(phosphonate)", r
    assert _key(_independent_parse(r["name"])) == _key(smiles)
    be = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert be["name"] == r["name"], be


@pytest.mark.parametrize("smiles,name", [
    ("OP(=O)(O)CP(=O)(O)O", "methylenebis(phosphonic acid)"),
    ("OP(=O)(O)C(Cl)(Cl)P(=O)(O)O", "(dichloromethylene)bis(phosphonic acid)"),
    ("OP(=O)(O)CCP(=O)(O)O", "(ethane-1,2-diyl)bis(phosphonic acid)"),
    ("OP(=O)(O)c1ccc(P(=O)(O)O)cc1", "(1,4-phenylene)bis(phosphonic acid)"),
    ("[Na+].OP(=O)([O-])CP(=O)(O)O", "sodium trihydrogen methylenebis(phosphonate)"),
])
def test_linked_phosphonic_acids_are_multiplied(smiles, name):
    """ (the Blue Book) with: identical phosphonic acid
    parents joined by one linking group (:35472 "(naphthalene-2,6-diyl)bis-
    (phosphonous acid) (PIN)"; 'benzene-1,4-diyl' is not recommended,:16282)."""
    jar_or_skip()
    from tests.support.rt_assert import _independent_parse
    r = _pin(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
    parsed = _independent_parse(name)
    assert parsed and _key(parsed) == _key(smiles)


@pytest.mark.parametrize("smiles,name", [
    ("OC(=O)CC(=O)[O-].[Na+]", "sodium hydrogen propanedioate"),
    ("[O-]C(=O)C(=O)O.[Na+]", "sodium hydrogen oxalate"),
    ("OC(=O)C(Cl)C(=O)[O-].[Na+]", "sodium hydrogen 2-chloropropanedioate"),
    # a benzene ring: the parent would be the retained benzoate, which the method (1)
    # producer does not build; method (2), labelled below the PIN
    ("OC(=O)c1ccccc1C(=O)[O-].[K+]", "potassium hydrogen benzene-1,2-dicarboxylate"),
])
def test_method_2_where_method_1_has_no_parent(smiles, name):
    """Default tier (the paper, Methods, "Tiers", L73: "The default configuration
    emits a name only when the pipeline can build the preferred IUPAC name (PIN);
    otherwise, it declines"; user decision 2026-09-30): these method (2) names are
    recorded as not the PIN, so the default tier declines them (NO_VERIFIED_PIN); the
    strict path's name, labelled systematic_verified ("a correct systematic name that
    is not the PIN"), is the best-effort tier's name (tests/support/default_tier.py).

    A one- or two-carbon parent is the retained formate / acetate,
    the Blue Book), which does not take its own suffix group as a prefix:
     :4969 "A suffix explicitly or implicitly present cannot be
    expressed as a prefix",:4973 "propanedioic acid (PIN) malonic acid (not
    2-carboxyacetic acid)". The decline is structural (the chain length from the
    ionized carboxy carbon), so a substituted stem ('2-carboxy-2-chloroethanoate')
    declines too. The method (2) name is right and reads back with the input's
    hydrons, but it is not the PIN for a drawn site (:31596), so it ships below."""
    jar_or_skip()
    from tests.support.default_tier import declined_pin_row
    from tests.support.rt_assert import _independent_parse
    r = declined_pin_row(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified", r
    parsed = _independent_parse(name)
    assert parsed and _fixed_h(parsed) == _fixed_h(smiles)


def test_chain_parent_length_is_structural():
    from rdkit import Chem
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.ions import _anion_chain_parent_length
    for smiles, n in (("OC(=O)CC(=O)[O-]", 2), ("OC(=O)C(Cl)C(=O)[O-]", 2),
                      ("[O-]C(=O)C(=O)O", 1), ("OC(=O)C(C)C(=O)[O-]", 3),
                      ("OC(=O)CCC(C)C(=O)[O-]", 4)):
        mol = Chem.MolFromSmiles(smiles)
        site = get_ion_sites(mol)["anions"][0]
        assert _anion_chain_parent_length(mol, site) == n, smiles


def test_anion_word_with_a_carboxylate_suffix_cites_no_acid_hydron():
    """The 'hydrogen' word is added unless the anion word cites the acid hydrons
    itself: the free-acid prefix 'carboxy' does, the '-carboxylate' suffix and the
    'carboxylato' prefix do not,:31593-:31597)."""
    from orthonym.rules.salts import _anion_word_expresses_acid_hydrons
    assert _anion_word_expresses_acid_hydrons("6-carboxyhexanoate")
    assert _anion_word_expresses_acid_hydrons("dihydrogen phosphate")
    assert not _anion_word_expresses_acid_hydrons("propane-1,2,3-tricarboxylate")
    assert not _anion_word_expresses_acid_hydrons("2-(carboxylatomethyl)benzoate")


@pytest.mark.parametrize("smiles", [
    "[O-]C(=O)CC(C(=O)O)CC(=O)[O-].[Na+].[Na+]",
    "[Na+].[Na+].CP(=O)(O)[O-].CP(=O)(O)[O-]",
])
def test_best_effort_keeps_the_salt_form(smiles):
    """The fully ionized anion word is used only where the salt's 'hydrogen' word
    will cite the hydrons (one anion fragment, a word without a free-acid
    prefix); otherwise the anion is named as drawn and the salt keeps its
    cation words, never the adduct form."""
    jar_or_skip()
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    from tests.support.rt_assert import _independent_parse
    be = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert be["name"].startswith("disodium ") and "sodium(1+)" not in be["name"], be
    parsed = _independent_parse(be["name"])
    assert parsed and _fixed_h(parsed) == _fixed_h(smiles)
