"""Spelling check: an acyclic chain of four or more heterounits has a skeletal replacement ('a') PIN.

 (``## **** Skeletal replacement ('a') nomenclature in acyclic chains``,
the Blue Book; text:23348): "Skeletal replacement ('a') nomenclature rather than substitutive or
multiplicative names must be used to generate preferred IUPAC names for acyclic structures when four or more
heterounits are present in a unbranched chain containing at least one carbon atom and when none of the
heteroatoms constitute all or part of the principal characteristic group of the compound." A heterounit
(:23350) "is a set of heteroatoms having a name of its own such as, -SS-, disulfanediyl; -SiH2-O-SiH2-,
disiloxane-1,3-diyl". (:23405): "The same number of characteristic groups that would be expressed as
suffixes in substitutive names must be present in skeletal replacement ('a') names." So a substitutive or
multiplicative name of such a chain is a systematic name, not the PIN, and its label is not ``pin_verified``.

The rows of FAILS are the substitutive names the engine gave, or the Blue Book's own non-PIN spelling
(2-(2-{2-[2-(carboxymethoxy)ethoxy]ethoxy}ethoxy)acetic acid,:23367). The rows of NO_FAILURE are the
shapes the check abstains on: the 'a' name itself, three heterounits (:27756), the nitrogen atoms as the
principal amine (S2), a branch that carries heteroatoms, a ring, a charge, a second fragment.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym, name_compound
from orthonym.cli import _emit_tier_flags
from orthonym.validation import pin_spelling as ps
from orthonym.validation.spelling import checks_skeletal_chain as sc

pytestmark = pytest.mark.unit

RULE = "P-51.4.1.1"

FAILS = [  # (smiles, a substitutive name) -> SpellingFailure with rule
    ("OCCOCCOCCOCCOCCO", "2-(2-{2-[2-(2-hydroxyethoxy)ethoxy]ethoxy}ethoxy)ethan-1-ol"),
    ("OC(=O)COCCOCCOCCOCC(O)=O",
     "2-(2-{2-[2-(carboxymethoxy)ethoxy]ethoxy}ethoxy)acetic acid"),  # BB:23367
    ("NCCNCCNCCNCCNCCN",
     "N1-(2-aminoethyl)-N2-[2-({2-[(2-aminoethyl)amino]ethyl}amino)ethyl]ethane-1,2-diamine"),
    ("CCCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC(C)=O",
     "2-[2-(2-{2-[2-(2-{2-[2-(2-butoxyethoxy)ethoxy]ethoxy}ethoxy)ethoxy]ethoxy}ethoxy)ethoxy]ethyl acetate"),
    ("C[SiH2]C[SiH2]C[SiH2]C[SiH2]CCCl",
     "1-chloro-2-({[({[(methylsilyl)methyl]silyl}methyl)silyl]methyl}silyl)ethane"),
]
NO_FAILURE = [  # (smiles, name): not flagged
    ("COCCOCCOCCOC", "2,5,8,11-tetraoxadodecane"),                       # 'a' name present
    ("COCCOCCOC", "1-methoxy-2-(2-methoxyethoxy)ethane"),                 # three heterounits, BB:27756
    ("CCNCCCNCCCCNCCCNCC",
     "N1,N4-bis[3-(ethylamino)propyl]butane-1,4-diamine"),               # the N atoms are the principal amine
    ("CCCOCCOCC(COCCO)OCCOCCO",
     "1-(2-hydroxyethoxy)-2-[2-(2-hydroxyethoxy)ethoxy]-3-(2-propoxyethoxy)propane"),  # star: abstain
    ("C1CCOCCOCCOCCOCC1", "1,4,7,10-tetraoxacyclopentadecane"),            # ring
    ("[NH3+]CCOCCOCCOCCOCCOC",
     "2-{2-[2-(2-methoxyethoxy)ethoxy]ethoxy}ethan-1-aminium"),           # charged
    ("COCCOCCOCCOC.O", "2,5,8,11-tetraoxadodecane water"),                  # two fragments
]
BUTOXY_PEG_ACETATE = FAILS[3][0]


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol


@pytest.mark.parametrize("smiles,name", FAILS, ids=[s for s, _ in FAILS])
def test_failures(smiles, name):
    mol = _mol(smiles)
    assert sc.a_chain_applies(mol)
    assert not sc.names_an_a_chain(name)
    failure = sc.a_chain_name_check(mol, name)
    assert isinstance(failure, ps.SpellingFailure)
    assert failure.rule == "P-51.4.1.1"
    assert "'a' name required" in failure.detail
    assert [f.rule for f in ps.check_pin_spelling(mol, name) if f.rule == RULE] == [RULE]


@pytest.mark.parametrize("smiles,name", NO_FAILURE, ids=[s for s, _ in NO_FAILURE])
def test_no_failure(smiles, name):
    assert sc.a_chain_name_check(_mol(smiles), name) is None
    assert RULE not in [f.rule for f in ps.check_pin_spelling(_mol(smiles), name)]


def test_interface():
    assert "P-51.4.1.1" in ps.registered_rules()
    assert sc.RULE == "P-51.4.1.1"
    assert "orthonym.validation.spelling.checks_skeletal_chain" in ps.BUILTIN_CHECK_MODULES
    assert sc.a_chain_name_check in ps._REGISTRY["P-51.4.1.1"]


def test_the_check_reads_the_structure_not_the_name():
    """No structure, or one that is no chain, is never a failure; a chain with an 'a' name never is."""
    assert sc.a_chain_applies(None) is False
    assert sc.a_chain_name_check(None, "anything") is None
    assert sc.a_chain_name_check(_mol("CCO"), "ethanol") is None
    assert sc.a_chain_name_check(_mol("OCCOCCOCCOCCOCCO"), "3,6,9,12-tetraoxatetradecane-1,14-diol") is None


A_NAMES = [  # names with a block of replacement prefixes on an acyclic stem
    "2,5,8,11-tetraoxadodecane",
    "3,6,9,12-tetraoxatetradecanedioic acid",                          # BB:23367
    "dimethyl 2,4,6,8-tetrasilanonan-1-yl phosphate",                  # BB:23401: the 'a' group in a substituent
    "2,5,8,11-tetraoxatetradec-13-ene",
    "1-iodo-3,6,9,12,15-pentaoxaoctadec-17-yne",
    "7,10-dioxa-2-thia-4-silaundecane",
    "8-methyl-2,4,6,9,11-pentathiadodecane",
    "15-methyl-3,6,13-trioxa-10-azahexadecan-8-ol",
    "10-chloro-2,4,6,8-tetrasiladecane",
    "4,7,10,13-tetraoxa-1-silapentadecane",
    "2,9-dimethyl-4,7-dioxa-2,9-diboradecane",
    "2,5λ4,8,11-tetrathiadodecane",                                    # the lambda convention
    # the long polyethers the engine builds: (the Blue Book) composite multiplying prefixes and
    # the stems of forty-one carbon atoms and more
    "2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65-docosaoxahexahexacontane",
    "2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65,68,71-tetracosaoxatriheptacontan-73-oic acid",
    "2,4,6,8,10,12,14,16,18,20,22-undecaoxatricosane",
    "2,5,8,11,14,17,20,23,26,29,32-undecaoxatritriacontane",
    "2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,59,62-henicosaoxatrihexacontane",
    "2,4,6,8,10,12,14,16,18,20,22,24,26,28,30-pentadecathiahentriacontane",
    "2,5,8-trioxapentacontane",
    "2,5,8-trioxahectane",
]


@pytest.mark.parametrize("name", A_NAMES)
def test_an_a_block_on_an_acyclic_stem_is_read(name):
    assert sc.names_an_a_chain(name)


def test_the_reader_is_never_narrower_than_the_writers_pattern():
    """The check keeps its own copy of ``rules.pin_vocabulary._ACYCLIC_A_NAME`` (a check must not import the
    writer); a name the writer's pattern reads, the copy reads too."""
    from orthonym.rules.pin_vocabulary import _ACYCLIC_A_NAME
    for name in A_NAMES + [n for _s, n in FAILS + NO_FAILURE]:
        if _ACYCLIC_A_NAME.search(name):
            assert sc.names_an_a_chain(name), name


@pytest.mark.parametrize("name", [
    "ethane-1,2-diol",
    "1-methoxy-2-(2-methoxyethoxy)ethane",
    "1,4,7,10-tetraoxacyclopentadecane",                               # a ring is no acyclic stem
    "1,4,7,10-tetraoxa-13-azacyclopentadecane",
    "2-(2-hydroxyethoxy)ethan-1-ol",
    "oxolane",
])
def test_no_a_block_on_an_acyclic_stem(name):
    assert not sc.names_an_a_chain(name)


@pytest.mark.parametrize("smiles", [
    "COCCOCCOCCOC", "CCOCCOCCOCCOC", "C=CCOCCOCCOCCOC", "COC=COCCOCCOC", "C#CCOCCOCCOCCOCCOCCI",
    "C[SiH2]C[SiH2]C[SiH2]C[SiH2]CCCl", "ClCCOCCOCCOCCOCCCl", "CSCSCSCC(C)SCSC", "CC(C)OCCOCCOCCOC",
    "[SiH3]CCOCCOCCOCCOCC", "CB(C)COCCOCB(C)C", "COCCOCC[SiH2]CSC", "C[SiH2]C[SiH2]C[SiH2]CSCC",
    "OCCOCCOCCOCCOC", "OCCOCCOCCOCCOCC", "CC(O)COCCOCCOCCOC", "OCCOCCOCCOCCOCCO", "SCCOCCOCCOCCOCCS",
    "NCCOCCOCCOCCOCCN", "NCCOCCOCCOCCOCC", "NCCNCCNCCNCCNCCN", "NCCCNCCCCNCCCNCCCNCCCN",
    "NCCOCCOCCOCCOCCO", "CCOCCOCC(O)CNCCOCC(C)C", "OCCOCCOCCOCCOCCF", "CSC[SiH2]CSC[SiH2]CCN",
    "COCCOCCOCCOC(CN)C(C)(C)C", "C[C@H](O)COCCOCCOCCOC", "CCC(C(C)C)OCCOCCOCCOC",
    "C=CCOCCOCCOCCOCCO", "COCCOOCCOCCOC",
])
def test_the_chains_the_builders_name_are_chains(smiles):
    """Every chain of the expected 'a' names of the lane (S1) is a chain of the check (S2: N is a skeletal atom
    only with a senior group or the two NH2 ends)."""
    assert sc.a_chain_applies(_mol(smiles)), smiles


@pytest.mark.parametrize("smiles", [
    "CCOCCOCCOCC",                         # three heterounits
    "CCNCCCNCCCCNCCCNCC",                  # N is the principal amine
    "NCCNCCNCCNCCNCC",                     # one NH2 end: the other N atoms are the amine
    "NCC(N)CNCCNCCNCCNCCN",                # a third NH2 leaf: the N atoms are the principal amine
    "CCCOCCOCC(COCCO)OCCOCCO",             # a branch carries heteroatoms (principal group possibly off the chain)
    "CCCOCCOCC(=O)OCCOCCOC",               # a chain O in an ester is no plain heteroatom
    "COCCOCC(C(=O)O)OCCOCCOC",             # an acyl group on an interior atom: the principal group is off the chain
    "C1CCOCCOCCOCCOCC1",                   # ring
    "[NH3+]CCOCCOCCOCCOCCOC",              # charge
    "COCCOCCOCCOC.O",                      # two fragments
    "CCCCC", "CCO", "O",                   # no heterounits
])
def test_the_check_abstains_on(smiles):
    assert not sc.a_chain_applies(_mol(smiles))


def test_more_than_120_heavy_atoms_is_out():
    """The cap on the size of the molecule the check reads: 124 heavy atoms abstain, a shorter chain applies."""
    long_peg = "OCC" + "OCC" * 40 + "O"
    short_peg = "OCC" + "OCC" * 4 + "O"
    assert Chem.MolFromSmiles(long_peg).GetNumHeavyAtoms() > 120
    assert not sc.a_chain_applies(Chem.MolFromSmiles(long_peg))
    assert sc.a_chain_applies(Chem.MolFromSmiles(short_peg))


def test_heterounits_are_counted_per_run_of_consecutive_heteroatoms():
    """S1: -SS- and -SiH2-O-SiH2- are one unit each (:23350), -O-SiH2-O- is three, two different atoms are two."""
    assert sc.a_chain_applies(_mol("CSSCSSCSSCSSC"))                       # four -SS- units
    assert not sc.a_chain_applies(_mol("CSSCSSCSSC"))                      # three of them: six S atoms, three units
    four_disiloxane = _mol("C[SiH2]O[SiH2]C[SiH2]O[SiH2]C[SiH2]O[SiH2]C[SiH2]O[SiH2]C")
    three_disiloxane = _mol("C[SiH2]O[SiH2]C[SiH2]O[SiH2]C[SiH2]O[SiH2]C")
    assert sc.a_chain_applies(four_disiloxane)                              # four -SiH2-O-SiH2- units
    assert not sc.a_chain_applies(three_disiloxane)                         # three of them: nine atoms, three units
    assert sc.a_chain_applies(_mol("CO[SiH2]OC[SiH2]C"))                    # -O-SiH2-O- is three units, plus Si
    assert sc.a_chain_applies(_mol("CSOCSOC"))                              # S-O is two units, twice
    assert not sc.a_chain_applies(_mol("CSOCSSC"))                          # S-O (2) and S-S (1): three units


@pytest.mark.opsin_gate
def test_integration_label():
    """The default tier (best-effort) ships the substitutive name unchanged and lowers its label to
    ``systematic_verified``; the PIN-only tier declines it."""
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(BUTOXY_PEG_ACETATE)
    assert row["name"] == FAILS[3][1], row
    assert row["tier"] == "systematic_verified", row
    assert row["spelling_failures"][0]["rule"] == "P-51.4.1.1", row
    assert name_compound(BUTOXY_PEG_ACETATE) == FAILS[3][1]
    assert name_compound(BUTOXY_PEG_ACETATE, emit_tier="pin") == ""
