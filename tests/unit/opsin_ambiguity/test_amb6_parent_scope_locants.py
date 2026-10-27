"""A prefix, an N-substituent or a retained name that OPSIN reads as more than one structure
,.

 Citation of locants (the Blue Book): "In preferred IUPAC names, if any locants
are essential for defining the structure of the parent structure or of a unit of structure as
defined by its appropriate enclosing marks, then all locants must be cited for the parent
structure or that structural unit."
"""
import pytest

from ._helpers import ambiguity_messages, assert_clean, name_row, round_trips

# the names are the shipped ones: the OPSIN validity gate is on (tests/conftest.py)
pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_anisaldehyde_is_not_a_retained_name(tier):
    # Names of aldehydes derived from retained names of carboxylic acids,
    # (the Blue Book): "The following names are preferred IUPAC names, with
    # substitution allowed for acetaldehyde and benzaldehyde." The Blue Book retains no
    # 'anisaldehyde', and OPSIN reads the token 'anis' as more than one structure, so the
    # substituted benzaldehyde stands.
    row = assert_clean("COc1ccc(C=O)cc1", tier=tier)
    assert row["name"] == "4-methoxybenzaldehyde"


@pytest.mark.parametrize("smiles,expected", [
    ("CCN(CC)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(C)C1", "N,N-diethyllysergamide"),
    ("CCN(C)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(C)C1", "N-ethyl-N-methyllysergamide"),
    ("CN(C1CCCCC1)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(C)C1", "N-cyclohexyl-N-methyllysergamide"),
])
def test_amide_n_substituents_are_each_cited_with_its_locant(smiles, expected):
    # Substitution of primary amides, (the Blue Book): the
    # substituents R' and R'' of R-CO-NR'R'' are cited "as prefixes preceded by the locant N";
    # examples 'N,N-diethylfuran-2-carboxamide (PIN)' (:32797) and
    # 'N-methyl-N-phenylbenzamide (PIN)' (:32871).
    row = assert_clean(smiles)
    assert row["name"] == expected


@pytest.mark.parametrize("smiles,expected", [
    ("Cc1ccc(C#CI)cc1", "1-(iodoethynyl)-4-methylbenzene"),
    ("FC#Cc1ccc(C(F)(F)F)cc1", "1-(fluoroethynyl)-4-(trifluoromethyl)benzene"),
    ("IC#CC1=CCCCC1", "1-(iodoethynyl)cyclohex-1-ene"),
])
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_haloethynyl_prefix_is_enclosed(smiles, expected, tier):
    # Parentheses (the Blue Book): "Parentheses are used around compound (see
    # and complex (see prefixes", example '(chloromethyl)silane (PIN)'.
    # Omission of locants, (:2953): the locant of the halogen is omitted,
    # I-C#C- has one isomer ('oxoethenyl (preferred prefix)',:2983).
    row = assert_clean(smiles, tier=tier)
    assert row["name"] == expected


@pytest.mark.parametrize("smiles,locanted", [
    ("C=C(OC)c1cccc(SC)c1", "1-methoxyethen-1-yl"),
    ("CCOC=Cc1ccccc1C(=O)N(C)C", "2-ethoxyethen-1-yl"),
    ("C=C(N=O)C1(CCO)CCCC1", "1-nitrosoethen-1-yl"),
])
def test_ethenyl_prefix_cites_its_locants(smiles, locanted):
    # Omission of locants, (the Blue Book), the examples that need
    # locants (:3003): '2-chloroethen-1-yl (preferred prefix)'. CH2=C(X)- and XCH=CH- are two
    # isomers, so the locants of the substituent and of the free valence are cited.
    for tier in ("pin", "best-effort"):
        row = name_row(smiles, tier=tier)
        if row["tier"] in (None, "abstain"):
            assert tier == "pin", row
            continue
        assert locanted in row["name"], row
    assert_clean(smiles)


@pytest.mark.parametrize("smiles,expected", [
    ("NCC(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)O", "N-(glycylglutamyl)glycine"),
    ("NCC(=O)N[C@@H](CC(=O)O)C(=O)NCC(=O)O", "N-(glycylaspartyl)glycine"),
])
def test_peptide_acyl_group_is_enclosed(smiles, expected):
    # Definitions, (the Blue Book): 'glycylglutamyl' is a compound
    # substituent group, a simple group (glutamyl, Acyl groups,:54570) with another
    # attached; Parentheses (:7232) encloses it. Derivatives formed by
    # substitution (:54478): the retained name 'glycine' takes the locant N.
    row = assert_clean(smiles)
    assert row["name"] == expected


@pytest.mark.parametrize("smiles,expected", [
    ("CC(C)[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)CN)C(=O)O", "glycylaspartylvaline"),
    ("C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)CN)C(=O)O", "glycylaspartylalanine"),
    ("NCC(=O)N[C@@H](CC(=O)O)C(=O)N1CCC[C@@H]1C(=O)O", "glycylaspartyl-D-proline"),
    ("C[C@H](N)C(=O)N[C@@H](CCC(=O)O)C(=O)N1CCC[C@@H]1C(=O)O", "alanylglutamyl-D-proline"),
])
def test_acyl_float_cites_no_n_on_a_systematic_amino_acid_name(smiles, expected):
    # Derivatives formed by substitution (the Blue Book): "Retained names are
    # used to indicate carbon, nitrogen, oxygen and sulfur substitution." The locant N cites the
    # nitrogen of a retained name; Systematic substitutive names (:54247,:54251):
    # "When not denoted by a retained name, amino acids receive systematic substitutive names",
    # which cite that nitrogen as 'amino' or by a ring locant. A peptide whose last residue is
    # such a parent keeps its peptide name Names of peptides,:54696).
    row = assert_clean(smiles)
    assert row["name"] == expected


def test_nacyl_float_gate_withholds_n_on_a_systematic_amino_acid_where_a_peptide_name_exists(
        monkeypatch):
    # Derivatives formed by substitution (the Blue Book): the locant N on the
    # retained name 'glycine' stands, on the systematic '(2S)-2-amino-3-methylbutanoic acid' it
    # designates nothing; the peptide name cites no N.
    from orthonym.decomposition.engine import gate_nonsuffix_nacyl_float
    from orthonym.metrics import provenance as pv
    from orthonym.routing import dispatch_table
    valine = ("CC(C)[C@H](N)C(=O)O", "N-(glycylaspartyl)(2S)-2-amino-3-methylbutanoic acid",
              "(2S)-2-amino-3-methylbutanoic acid")
    valine_kw = dict(acid_name="glycylaspartic acid",
                     parent_smiles="CC(C)[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)CN)C(=O)O")
    glycine = ("NCC(=O)O", "N-(glycylglutamyl)glycine", "glycine")
    glycine_kw = dict(acid_name="glycylglutamic acid",
                      parent_smiles="NCC(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)O")
    pv.clear_provenance()
    monkeypatch.setattr(dispatch_table, "_PEPTIDE_NAME_STANDS", True)
    assert gate_nonsuffix_nacyl_float(*valine, **valine_kw) is None
    assert gate_nonsuffix_nacyl_float(*glycine, **glycine_kw) == glycine[1]
    # a molecule with no peptide name keeps the float as its demotion
    monkeypatch.setattr(dispatch_table, "_PEPTIDE_NAME_STANDS", False)
    assert gate_nonsuffix_nacyl_float(*valine, **valine_kw) == valine[1]
    pv.clear_provenance()


def test_retained_amino_acid_names():
    from orthonym.data.amino_acids import is_retained_amino_acid_name
    for name in ("glycine", "L-proline", "D-allothreonine", "L-aspartic acid"):
        assert is_retained_amino_acid_name(name), name
    for name in ("(2S)-2-amino-3-methylbutanoic acid", "(2R)-pyrrolidine-2-carboxylic acid",
                 "4-aminobutanoic acid", "ethanamine", ""):
        assert not is_retained_amino_acid_name(name), name


def test_amide_n_block_is_not_put_in_front_of_the_descriptors_of_an_amide_name():
    # NAMING OF STEREOISOMERS (the Blue Book), second paragraph (:44643): "They are
    # placed at the front of the complete name when related to the parent structure". The amide
    # name '(2S,3R,...)-...-carboxamide' begins with its descriptors, so 'N,N-dimethyl' cannot be
    # put in front of it; ALPHANUMERICAL ORDER (:3436),:3440: the prefixes are cited in
    # alphanumerical order. The name of the general engine stands.
    smiles = "COc1ccc([C@@]23Oc4cc(OC)cc(OC)c4[C@]2(O)[C@H](O)[C@H](C(=O)N(C)C)[C@H]3c2ccccc2)cc1"
    row = assert_clean(smiles)
    assert row["name"] == (
        "(2S,3R,4R,5S,6R)-2,3-dihydroxy-10,12-dimethoxy-6-(4-methoxyphenyl)-N,N-dimethyl-5-phenyl-"
        "7-oxatricyclo[6.4.0.0^2,6]dodeca-1(12),8,10-triene-4-carboxamide")


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)OOC(C)(Cl)Cl", "1-(acetylperoxy)-1,1-dichloroethane"),
    ("CC(Cl)(Cl)SC(C)(I)I", "1,1-dichloro-1-[(1,1-diiodoethyl)sulfanyl]ethane"),
    ("CC(N=C=O)SC(C)N=C=O", "1-isocyanato-1-[(1-isocyanatoethyl)sulfanyl]ethane"),
])
def test_chain_prefix_cites_its_locant_when_the_chain_carries_other_prefixes(smiles, expected):
    # Citation of locants (the Blue Book): all locants of the unit are cited when
    # any is essential. The locant 1 of a lone substituent on ethane is omitted (b),
    #:2905 'ethanol'); a second prefix on the chain makes it essential.
    row = assert_clean(smiles)
    assert row["name"] == expected


@pytest.mark.parametrize("smiles,cited", [
    ("CC(=O)NC1(CCCC1)c1nnnn1C1CCCCC1", "1-(1-cyclohexyl-1H-tetrazol-5-yl)cyclopentyl"),
    ("CCCCC(CC)C(=O)O[C@@H]1O[C@H](C(=O)[O-])[C@@H](O)[C@H](O)[C@H]1O", "1-O-(2-ethylhexanoyl)"),
])
def test_compound_prefix_after_a_locant_is_enclosed(smiles, cited):
    # Parentheses (the Blue Book): "Parentheses are used around compound (see
    # and complex (see prefixes". A prefix with its own locants and a locant
    # of its own parent in front of it is cited in marks.
    row = assert_clean(smiles)
    assert cited in row["name"], row


@pytest.mark.parametrize("smiles,expected", [
    ("[2H]C(=[N+]=[N-])C", "1-diazo(1-2H)ethane"),
    ("ClC([2H])C", "1-chloro(1-2H1)ethane"),
])
def test_prefix_locant_is_cited_while_an_isotopic_modification_cites_one(smiles, expected):
    # Omission of locants (isotopes), (the Blue Book): "if isotopic
    # modification requires a locant to specify its position, then all locants must be specified
    # and none are omitted"; example '(2-13C)ethan-1-ol [not (2-13C)ethanol]' (:44186).
    row = assert_clean(smiles)
    assert row["name"] == expected


@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)C1(N=C=S)C=CC=CC1", "1-isothiocyanatocyclohexa-2,4-diene-1-carboxylic acid"),
    ("OC(=O)C1(N=C=O)C=CCCC1", "1-isocyanatocyclohex-2-ene-1-carboxylic acid"),
    ("OS(=O)(=O)C1(N=C=S)C(C)C=CC=C1",
     "1-isothiocyanato-6-methylcyclohexa-2,4-diene-1-sulfonic acid"),
])
def test_prefix_on_the_carbon_of_a_monovalent_suffix_cites_its_locant(smiles, expected):
    # Citation of locants (the Blue Book): both groups stand on ring carbon 1, so
    # the locant of the prefix is essential and is cited.
    row = assert_clean(smiles)
    assert row["name"] == expected


def test_divalent_suffix_leaves_no_room_for_a_prefix():
    from orthonym.rules.locant_validation import suffix_leaves_no_room_for_a_prefix
    assert suffix_leaves_no_room_for_a_prefix("one")
    assert suffix_leaves_no_room_for_a_prefix("thione")
    assert not suffix_leaves_no_room_for_a_prefix("sulfonic acid")
    assert not suffix_leaves_no_room_for_a_prefix("carboxylic acid")
    assert not suffix_leaves_no_room_for_a_prefix("amine")


@pytest.mark.parametrize("smiles,expected", [
    ("NP(=O)(c1ccccc1)c1ccccc1", "P,P-diphenylphosphinic amide"),
    ("NP(=O)(c1ccc(C(F)(F)F)cc1)c1ccc(C(F)(F)F)cc1", "P,P-bis[4-(trifluoromethyl)phenyl]phosphinic amide"),
    ("CP(=O)(N)c1ccccc1", "P-methyl-P-phenylphosphinic amide"),
    ("NP(=O)(N)c1ccccc1", "P-phenylphosphonic diamide"),
    ("NP(N)c1ccccc1", "P-phenylphosphonous diamide"),
    ("NP(C)C", "P,P-dimethylphosphinous amide"),
])
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_organyl_groups_on_the_central_atom_of_an_amide_cite_the_locant_p(smiles, expected, tier):
    # Citation of locants (the Blue Book, quoted above): the amide of the acid has
    # a nitrogen that could carry the groups, so each is cited with the locant of the central
    # atom, as Amides and hydrazides writes them: '*N*,*N*,*P*,*P*-tetramethylphosphinic
    # amide (PIN)' (the Blue Book), '*N*,*N*′-dimethyl-*P*-phenylphosphonic diamide (PIN)'
    # (:35789). OPSIN reads 'diphenylphosphinic amide' as ambiguous (the phenyls on P or on N).
    row = assert_clean(smiles, tier=tier)
    assert row["name"] == expected


def test_locanted_prefixes_of_a_central_atom_are_cited_in_alphanumerical_order():
    # ALPHANUMERICAL ORDER (the Blue Book): detachable prefixes are cited in
    # alphanumerical order; a multiplied prefix is cited once with its locants repeated
    from orthonym.rules.phosphorus import _build_locanted_substituent_string
    assert _build_locanted_substituent_string(["phenyl", "methyl"], "P") == "P-methyl-P-phenyl"
    assert _build_locanted_substituent_string(["phenyl", "phenyl"], "P") == "P,P-diphenyl"
    assert _build_locanted_substituent_string(["methyl"] * 5, "P") is None


@pytest.mark.parametrize("smiles,expected", [
    ("CCCC(=O)SOOC", "[1-(methylperoxy)sulfanyl]butan-1-one"),
    ("CCCC(=O)[Se]OOC", "[1-(methylperoxy)selanyl]butan-1-one"),
    ("CCCC(=O)[Te]OOC", "[1-(methylperoxy)tellanyl]butan-1-one"),
    ("CCCC(=O)SOOCC", "[1-(ethylperoxy)sulfanyl]butan-1-one"),
    ("CCCC(=O)SOOCCCl", "{1-[(2-chloroethyl)peroxy]sulfanyl}butan-1-one"),
])
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_pseudoester_of_a_peroxy_chain_is_named_as_the_book_prints_it(smiles, expected, tier):
    # Pseudoesters (the Blue Book): "When 'E' is an element belonging to Group
    # 16, the pseudoester is named as a pseudoketone"; the example (:32078) is
    # '[1-(methylperoxy)sulfanyl]butan-1-one (PIN)', the locant inside the marks of the prefix.
    # OPSIN reads that spelling as ambiguous and gives the structure of the input, so the
    # validity gate licenses the class (tests/unit/opsin_ambiguity/test_amb7_ambiguity_gate.py).
    row = name_row(smiles, tier=tier)
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row
    assert round_trips(smiles, row["name"])


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)OOSC(C)=O", "1-[(acetylperoxy)sulfanyl]ethan-1-one"),
    ("CCC(=O)[Se]OSC", "1-{[(methylsulfanyl)oxy]selanyl}propan-1-one"),
    ("CCCC(=O)SOSC", "1-{[(methylsulfanyl)oxy]sulfanyl}butan-1-one"),
    ("CC(=O)SSOC", "1-(methoxydisulfanyl)ethan-1-one"),
])
def test_pseudoketone_with_the_book_names(smiles, expected):
    # (the Blue Book): '1-[(acetylperoxy)sulfanyl]ethan-1-one (PIN)' (:32368);
    # (:39473): '1-{[(methylsulfanyl)oxy]selanyl}propan-1-one (PIN)' (:39497);
    #: '1-(methoxydisulfanyl)ethan-1-one (PIN)' (:28277). Each cites the locant in front
    # of the marks of its prefix, and OPSIN reads each once.
    row = name_row(smiles, tier="pin")
    assert row["name"] == expected, row
    assert not ambiguity_messages(row["name"]), row


def test_pseudoketone_whose_parent_is_the_shorter_acyl_chain_is_no_pin():
    # (the Blue Book) names a diacyl chalcogen chain as a pseudoketone; each acyl group is a candidate
    # parent and the senior chain is the parent, SENIORITY OF ACYCLIC CHAINS (:20916) "(b) has the greater number of
    # skeletal atoms": '1-[(acetylsulfanyl)peroxy]propan-1-one (PIN)' (:39499, under. The engine names the
    # acetyl group, the only one on the sulfur it builds, so its name is valid and no PIN: the PIN tier declines it and
    # the best-effort tier keeps it read back (it was asserted here as a pin_verified book name before).
    smiles, name = "CCC(=O)OOSC(C)=O", "1-[(propanoylperoxy)sulfanyl]ethan-1-one"
    assert name_row(smiles, tier="pin")["tier"] != "pin_verified"
    row = name_row(smiles, tier="best-effort")
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
    assert round_trips(smiles, name)


@pytest.mark.parametrize("left,right,joined", [
    ("N-{[(2S)-oxetan-2-yl]methyl}", "6-chloro", "N-{[(2S)-oxetan-2-yl]methyl}-6-chloro"),
    ("N-[(oxetan-2-yl)methyl]", "6-chloro", "N-[(oxetan-2-yl)methyl]-6-chloro"),
    ("N-(oxetan-2-yl)", "6-chloro", "N-(oxetan-2-yl)-6-chloro"),
    ("N-methyl", "6-chloro", "N-methyl-6-chloro"),
])
def test_components_of_an_amide_are_joined_with_a_hyphen_before_a_locant(left, right, joined):
    # Multiple types of enclosing marks (the Blue Book): "the nesting order is as
    # follows: {[({})]}", so a term that ends in any of ')', ']' and '}' takes the hyphen
    # before a following locant alike.
    from orthonym.decomposition.fragment_assembly import _join_components
    assert _join_components(left, right) == joined
