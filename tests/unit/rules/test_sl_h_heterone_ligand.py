"""SL lane H, item H1: a heterone whose ligand is bonded through O, S or N -- ``-O-P=O`` as the only
group is ``methoxyphosphanone``, not ``[(oxophosphanyl)oxy]methane``.

Governing text (the Blue Book, quoted with its section heading):

* "HETERONES" (:25977): "Compounds containing the -PO, -PO2, -AsO or -AsO2 are called
  heterones (see,. In the presence of a more senior characteristic group they are
  described by the compound prefixes oxophosphanyl, dioxo-lambda5-phosphanyl, oxoarsanyl, and
  dioxo-lambda5-arsanyl." The prefix is for a heterone that is NOT the principal group.
* example (:25985): ``[(2-methylpropyl)amino]arsanone (PIN)``, with ``N-(2-methylpropyl)-1-oxoarsanamine``
  listed beside it without the (PIN) label: the one ligand of the -AsO hub is the heteroatom group
  'amino', cited as a prefix of the heterone.
* "Acyclic heterones" (:29390): "Method (1)", the suffix 'one', "leads to preferred IUPAC
  names"; "Heterones" (:28281): 'C6H5-P=O phenylphosphanone (PIN)' (:28291).
*, under " POLYFUNCTIONAL KETONES, PSEUDOKETONES AND HETERONES" (:29585): heterones "are
  senior to hydroxy compounds and their chalcogen analogues, amines, and imines in the seniority order
  of classes".
* / (:27633,:27649): 'methoxy', 'phenoxy', 'methylsulfanyl' are the
  preferred prefixes of the R-O-, R-S- groups; (:7272) cites the one ligand of a
  mononuclear hub with enclosing marks when it is a compound prefix.

UNVERIFIED BY THE BOOK (the label rests on an analogy, and the reviewer decides): the Blue Book prints
no phosphanone with an alkoxy or aryloxy ligand. The one printed heteroatom ligand is the amino ligand
of the ARSANONE (:25985); puts -PO and -AsO in one sentence, gives the suffix 'one' to
both, and ranks a heterone above the alcohol, the amine and every other prefix-only group of
the ligand, so the heterone is the principal characteristic group and the parent. Nor does the book make
-O-P=O an acid group of its own: "Names of mononuclear noncarbon oxoacids" (:35409, the list at
:35431) holds phosphinic, phosphinous, phosphonic, phosphonous, phosphoric and phosphorous acid and no
HO-P=O, and metaphosphorous acid [HPO2]n = (-P(OH)O-)n is polymeric and "only for general nomenclature"
(:36677). The hub has bonding number 3 with ONE ligand, the shape of phenylphosphanone (:28291).
The independent check of every name below is OPSIN 2.9.0 reading it back WITHOUT the radical option to
the input's full InChIKey; what would make a row wrong is a Blue Book entry that names -O-P=O in
another way, and the book is silent.

The prefix case is unchanged: (:25977) "In the presence of a more senior characteristic group they
are described by the compound prefixes oxophosphanyl" -- methyl 2-(cyclohexylmethyl)-3-
[(oxophosphanyl)oxy]propanoate keeps it,:36299).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.rules.mononuclear_hydrides import name_heterone
from orthonym.validation.opsin_roundtrip import opsin_parse
from tests.support.pin_tiers import name_breadth, name_default

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _key(smiles):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


def _reads_back_without_radicals(name, smiles):
    """OPSIN 2.9.0 read of ``name`` WITHOUT the radical option, full InChIKey against the input."""
    back = opsin_parse(name)
    return back is not None and _key(back) == _key(smiles)


# (smiles, name, the Blue Book line the spelling follows)
HETERONES = [
    # the three names of the brief
    ("COP=O", "methoxyphosphanone", "P-63.2.2.1.1 :27633 'methoxy'; P-64.4.1 :29390 'one'"),
    ("O=POc1ccccc1", "phenoxyphosphanone", "'phenoxy (preferred prefix)' :24567"),
    ("CCCC(CC)CCCCOP=O", "[(5-ethyloctyl)oxy]phosphanone", "P-16.5.1.3.1 :7272 marks of a compound prefix"),
    # the oxy ligand: simple and compound prefixes of the shared alkoxy morphology
    ("CC(C)(C)OP=O", "tert-butoxyphosphanone", "'tert-butoxy' :27679"),
    ("CC(C)OP=O", "[(propan-2-yl)oxy]phosphanone", "'(propan-2-yl)oxy' :27683"),
    ("CC(C)COP=O", "(2-methylpropoxy)phosphanone", "'2-methylpropoxy' :27689"),
    ("O=POc1ccc(Cl)cc1", "(4-chlorophenoxy)phosphanone", "'phenoxy', full substitution :17796"),
    ("O=POCc1ccccc1", "(benzyloxy)phosphanone", "'benzyloxy (preferred prefix)' :18097"),
    ("O=POC1CCCCC1", "(cyclohexyloxy)phosphanone", "'(cyclohexyloxy)benzene (PIN)' :27768"),
    ("O=POCC(F)(F)F", "(2,2,2-trifluoroethoxy)phosphanone", "a halogen is a prefix only (P-59)"),
    ("C1=CC=C2C(=C1)C3=C2C(=CC=C3)OP=O", "[(biphenylen-1-yl)oxy]phosphanone", "a hydrocarbon ring ligand"),
    ("O=PO[C@H](C)CC", "{[(2R)-butan-2-yl]oxy}phosphanone", "the stereo descriptor of the ligand is expressed"),
    ("C/C=C/COP=O", "{[(2E)-but-2-en-1-yl]oxy}phosphanone", "the double-bond descriptor is expressed"),
    # the sulfanyl ligand
    ("CSP=O", "(methylsulfanyl)phosphanone", "'methylsulfanyl (preferred prefix)' :27651"),
    ("CC(C)(C)SP=O", "(tert-butylsulfanyl)phosphanone", "P-63.2.2.1.2 :27649"),
    # the amino ligand: the printed example (arsanone) and its phosphorus counterpart
    ("CC(C)CN[As]=O", "[(2-methylpropyl)amino]arsanone", "P-61.6 example :25985 (PIN), verbatim"),
    ("CC(C)CNP=O", "[(2-methylpropyl)amino]phosphanone", "P-61.6 :25977, by analogy with :25985"),
    ("CN(C)P=O", "(dimethylamino)phosphanone", "P-16.5.1.1 marks of a compound prefix"),
    ("CCN(C)P=O", "[ethyl(methyl)amino]phosphanone", "P-16.5.1.3.1 :7272 second prefix in marks"),
    ("O=PNc1ccccc1", "anilinophosphanone", "'anilino' (PIN)"),
    # the arsenic hub with the oxy ligand
    ("CO[As]=O", "methoxyarsanone", "P-61.6 :25977 -AsO, by analogy with :25985"),
    # the chalcogen analogues of the hub: the producer builds them for organyl ligands already
    # (ethylphosphanethione, below), so the ligand route does not narrow them: (:29585)
    # "Ketones, pseudoketones and heterones, and their chalcogen analogues in the order =O > =S > =Se > =Te"
    ("COP=S", "methoxyphosphanethione", "P-64.7.1 :29585 'their chalcogen analogues'; suffix 'thione'"),
    ("COP=[Se]", "methoxyphosphaneselone", "P-64.7.1 :29585; suffix 'selone' (P-68.3.2.3.1, :39109)"),
    ("COP=[Te]", "methoxyphosphanetellone", "P-64.7.1 :29585; suffix 'tellone' (P-68.3.2.3.1, :39109)"),
    ("CSP=S", "(methylsulfanyl)phosphanethione", "P-64.7.1 :29585; P-63.2.2.1.2 :27649"),
    ("CO[As]=S", "methoxyarsanethione", "P-64.7.1 :29585; arsanethione :39141"),
]


@pytest.mark.parametrize("smiles,name,rule", HETERONES, ids=[s for s, _, _ in HETERONES])
def test_the_heterone_with_a_heteroatom_ligand_is_the_phosphanone_at_both_tiers(smiles, name, rule):
    assert _reads_back_without_radicals(name, smiles), (name, rule)
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (name, "pin_verified", True), (rule, d)
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (name, "pin_verified"), (rule, b)


# ---- controls: the organyl ligands and the prefix case do not move ----------------------------

@pytest.mark.parametrize("smiles,name", [
    ("CCP=O", "ethylphosphanone"),
    ("O=Pc1ccccc1", "phenylphosphanone"),
    ("CCP=S", "ethylphosphanethione"),          # the chalcogen analogues of main, pin_verified there
    ("CC[As]=S", "ethylarsanethione"),
    ("CCP=[Se]", "ethylphosphaneselone"),
])
def test_the_organyl_heterones_are_unchanged(smiles, name):
    """:28291 'C6H5-P=O phenylphosphanone (PIN)'."""
    assert name_default(smiles).get("name") == name
    assert name_breadth(smiles).get("name") == name


def test_a_senior_group_keeps_the_oxophosphanyl_prefix():
    """ (:25977): 'In the presence of a more senior characteristic group they are described by
    the compound prefixes oxophosphanyl'; the ester is class 9, the heterone class 16,:18162)."""
    smiles = "COC(=O)C(CC1CCCCC1)COP=O"
    name = "methyl 2-(cyclohexylmethyl)-3-[(oxophosphanyl)oxy]propanoate"
    assert _reads_back_without_radicals(name, smiles)
    for res in (name_default(smiles), name_breadth(smiles)):
        assert (res.get("name"), res.get("tier")) == (name, "pin_verified"), res


# ---- the producer fails closed outside the class -----------------------------------------------

@pytest.mark.parametrize("smiles,why", [
    ("COP(=O)OC", "two ligands: bonding number 5, the ester of phosphonic acid (a hydrogen on P)"),
    ("COP(=O)C", "an organyl and a heteroatom ligand: an ester of phosphinic acid"),
    ("CCOP(=O)=O", "the dione with a heteroatom ligand: not built"),
    ("OP=O", "no carbon: the ligand is not an organyl-bearing group"),
    ("ClP=O", "a halogen ligand is not built"),
    ("NP=O", "an unsubstituted amino ligand is not built"),
    ("O=PN1CCCCC1", "a ring nitrogen ligand is not built"),
    ("O=PN(c1ccccc1)c1ccccc1", "a ligand the shared chokepoint cannot spell: declined, not guessed"),
    ("CC(=O)OP=O", "an acyl ligand: a carboxylic anhydride, class 7"),
    ("OCCOP=O", "a hydroxy ligand holds a second group: seniority is not compared here"),
    ("NCCOP=O", "an amino group in the ligand"),
    ("O=POOC", "a peroxy ligand: the O-O bond is not an R-O- group"),
    ("O=POP=O", "two hubs"),
    ("CO[SiH]=O", "a silanone is not in the class of P-61.6"),
])
def test_the_producer_declines_outside_the_class(smiles, why):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"fixture SMILES itself is invalid: {smiles}"
    assert name_heterone(mol) is None, why


# ---- the recursive substituent namer must not turn a heterone name into a prefix ----------------

def test_the_ligand_route_is_not_offered_to_the_recursive_fragment_namer(monkeypatch):
    """ (:25977): the prefix form of the heterone is 'oxophosphanyl'; a heterone has no '-yl' form.
    ``name_fragment_recursively`` names the H-capped fragment of a substituent as a molecule and turns
    the name into a prefix by suffix surgery, so a fragment name must not be a heterone name."""
    mol = Chem.MolFromSmiles("COP=O")
    assert name_heterone(mol) == "methoxyphosphanone"
    monkeypatch.setattr("orthonym.assembly.fragment_naming.in_fragment_recursion", lambda: True)
    assert name_heterone(mol) is None
    # the organyl ligand is not part of this route and keeps its behaviour
    assert name_heterone(Chem.MolFromSmiles("CCP=O")) == "ethylphosphanone"


@pytest.mark.parametrize("smiles,name", [
    # the enol carbon carries the group: its fragment 'C=COP=O' would be a nameable molecule
    # ('(ethenyloxy)phosphanone') and its '-yl' surgery gave '(ethenyloxy)phosphanonyl'
    ("C=C(C1CCC2(CC1)OCCO2)OP=O", "8-{1-[(oxophosphanyl)oxy]ethen-1-yl}-1,4-dioxaspiro[4.5]decane"),
    ("C=C([C@@H]1CC[C@@H](O1)CN2C=CC(=NC2=O)N)OP=O",
     "4-amino-1-{[(2R,5S)-5-{1-[(oxophosphanyl)oxy]ethen-1-yl}oxolan-2-yl]methyl}pyrimidin-2(1H)-one"),
])
def test_the_prefix_under_a_senior_group_is_not_replaced_by_the_replacement_name(smiles, name):
    """Without the guard above the best-effort tier fell to '(2-oxa-1-phosphaeth-1-en-1-yl)oxy' for
    these rows (the name still read back). The first is under a ring without a ketone, so the heterone
    is the principal group and the name is not a PIN: it keeps the best-effort label of the main tree;
    the second is a ketone on a nitrogen ring, where the prefix is the PIN spelling."""
    b = name_breadth(smiles)
    assert b.get("name") == name, b
    assert b.get("tier") == "systematic_verified", b
    assert _reads_back_without_radicals(name, smiles)
