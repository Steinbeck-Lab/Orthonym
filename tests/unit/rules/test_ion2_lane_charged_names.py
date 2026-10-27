"""Charged molecules of the PubChem left-out pool that the lane ION left without a name and now get a verified one
(lane ION2).

Every case is named at the best-effort tier and read back through OPSIN independently of the engine's own gate
(``tests.support.rt_assert``): the name must read back to the full InChIKey of the input and to the canonical SMILES
of it, which carries the charges. Each class is pinned with the rule that governs it:

* the nitrate ester on a nitrogen, "Substituent groups derived from esters of nitric acid and nitrous
  acid" (the Blue Book): '-O-NO2 nitrooxy (preselected prefix)' (:36405), the prefix of the ester oxygen whatever
  atom the oxygen is on;
* the ring nitrogen cation of a zwitterion as a prefix, "CATIONIC PREFIX NAMES" (:42290): "all prefix names are
  formed by adding the suffixes 'yl', 'ylidene', etc. to the cation name", for a fused or bridged ring the name of the
  cation made by a hydron at the nitrogen,:41366);
* the cumulative ionic suffixes of one parent, "Ionic centers in the same parent structure" (:42415): 'ium',
  'ylium', 'ide', 'uide' in that order, the centres off the parent cited in their own names;
* the ring group of an azinate, "General methodology for naming substituent groups" (:15814): the group that
  is double bonded to its parent has the stem and the hydro prefixes of the '-yl' of the same ring;
* the ion of a parent above its characteristic groups, "SENIORITY ORDER FOR CLASSES" (:18158), Table 4.1: anions,
  zwitterions and cations rank above acids, so the groups of an ionic parent are prefixes;
* the boron anion at a spiro junction, "Skeletal replacement ('a') nomenclature for anions" (:41120): the spiro
  parent and the suffix 'uide';
* the multiplied 'ylium' suffix, "General methodology" (:7029), (c) "before suffixes that are composed of two or
  more cumulative suffixes" (:7110): 'bis(ylium)' (multiple preferred suffix, describes two 'ylium' suffixes, whereas
  'diylium' could describe a cation radical suffix,:7114); 'ium', 'ide' and 'uide' keep 'di' (:41409,:41313);
* the diazonium cation on a group that is no hydrocarbon, "Diazonium ions" (:41607);
* the azanide of a group on a heteroatom, (:41069): 'acetylazanide (PIN)' (:41079).

The names of the classes the Blue Book gives no example of are not PINs: the tier of each is below the PIN's.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import _full_inchikey, _independent_parse, name_best_effort

pytestmark = [pytest.mark.unit, pytest.mark.roundtrip]


def _reads_back(smiles: str, name: str) -> bool:
    parsed = _independent_parse(name, smiles)
    if not parsed:
        return False
    read, want = Chem.MolFromSmiles(parsed), Chem.MolFromSmiles(smiles)
    return (read is not None and _full_inchikey(parsed) == _full_inchikey(smiles)
            and Chem.MolToSmiles(read) == Chem.MolToSmiles(want))


CASES = [
    # (smiles, name, rule)
    # -- the nitrate ester on a nitrogen,:36405)
    ("C1COCCN1O[N+](=O)[O-]", "4-(nitrooxy)morpholine", "P-67.1.4.3.1"),
    ("CN(C)O[N+](=O)[O-]", "[methyl(nitrooxy)amino]methane", "P-67.1.4.3.1"),
    ("C(NO[N+](=O)[O-])O[N+](=O)[O-]", "(nitrooxy)[(nitrooxy)amino]methane", "P-67.1.4.3.1"),
    ("C1(=NN(C(=N1)N)NO[N+](=O)[O-])N", "1-[(nitrooxy)amino]-1H-1,2,4-triazole-3,5-diamine", "P-67.1.4.3.1"),
    # -- the ring cation of a zwitterion as a prefix: the cation named by a hydron at its nitrogen,:42290)
    ("CS(=O)(=O)[N-][N+]1=CC2=CC=CC=C2C=C1", "(isoquinolin-2-ium-2-yl)(methanesulfonyl)azanide", "P-73.6"),
    ("CC1=CC=C(C=C1)S(=O)(=O)[N-][N+]2=CC3=CC=CC=C3C=C2C",
     "(4-methylbenzene-1-sulfonyl)(3-methylisoquinolin-2-ium-2-yl)azanide", "P-73.6"),
    ("C1C2=CC=CC=C2[N+]1=[N-]", "(7-azabicyclo[4.2.0]octa-1,3,5-trien-7-ium-7-ylidene)azanide", "P-73.6"),
    # -- the cumulative suffixes of one parent, the centres off it in their own names,:42417;
    ("[B-]1(F)(F)N2C(C)=CC(C)=C2C=C2C(C)=CC(CCC[N+](C)(C)C)=[N+]12",
     "2,2-difluoro-4,6,10-trimethyl-12-[3-(trimethylazaniumyl)propyl]-1,3-diaza-2-boratricyclo[7.3.0.0^3,7]dodeca-"
     "1(12),4,6,8,10-pentaen-1-ium-2-uide", "P-74.1.1"),
    ("[B-]1(F)(F)N2C(C)=CC(C)=C2C=C2C(C)=CC(CCCS(=O)(=O)[O-])=[N+]12",
     "2,2-difluoro-4,6,10-trimethyl-12-(3-sulfonatopropyl)-1,3-diaza-2-boratricyclo[7.3.0.0^3,7]dodeca-"
     "1(12),4,6,8,10-pentaen-1-ium-2-uide", "P-74.1.1"),
    # -- an ion of the parent ranks above every characteristic group, Table 4.1,:18158): the groups are prefixes
    ("[B-]1(F)(F)N2C(C)=CC(C)=C2C(CCC(=O)O)=C2C(C)=CC(C)=[N+]12",
     "8-(2-carboxyethyl)-2,2-difluoro-4,6,10,12-tetramethyl-1,3-diaza-2-boratricyclo[7.3.0.0^3,7]dodeca-"
     "1(12),4,6,8,10-pentaen-1-ium-2-uide", "P-41"),
    ("[B-]1(F)(F)N2C(C)=CC(C)=C2C(CCC(=O)OC)=C2C(C)=CC(C)=[N+]12",
     "2,2-difluoro-8-(3-methoxy-3-oxopropyl)-4,6,10,12-tetramethyl-1,3-diaza-2-boratricyclo[7.3.0.0^3,7]dodeca-"
     "1(12),4,6,8,10-pentaen-1-ium-2-uide", "P-41"),
    # -- the boron anion at a spiro junction: the spiro parent and 'uide',:41120)
    ("C1CC[B-]2(C1)CCCC2", "5-boraspiro[4.4]nonan-5-uide", "P-72.4"),
    ("[B-]12(OCCO1)OCCO2", "1,4,6,9-tetraoxa-5-boraspiro[4.4]nonan-5-uide", "P-72.4"),
    ("CC1(C)C[B-]2(OC1)OCCO2", "8,8-dimethyl-1,4,6-trioxa-5-boraspiro[4.4]nonan-5-uide", "P-72.4"),
    # -- the ring group of an azinate: the '-ylidene' on the stem and the hydro prefixes of the universal namer's '-yl'
    ("C1=CC=C2C(=C1)C(=[N+]([O-])[O-])C=CN2", "(1,4-dihydroquinolin-4-ylidene)azinate", "P-29.2"),
    ("C1=CC(=[N+]([O-])[O-])C2=CC=NNC2=C1", "(1,5-dihydrocinnolin-5-ylidene)azinate", "P-29.2"),
    # -- two 'ylium' centres of one parent are 'bis(ylium)', never 'diylium' (c),:7110, example:7114)
    ("C=C(C[CH-]C[CH2+])C[CH-]CCC[CH2+]", "5-methylideneundecane-1,11-bis(ylium)-3,7-diide", "P-16.3.2"),
    ("C[C-]1C(C2=C([N+]1=C)C=CC3=C2[CH+]C=C[CH+]3)(C)C",
     "3,3,4-trimethyl-5-methylidene-5-azatricyclo[7.4.0.0^2,6]trideca-1(9),2(6),7,11-tetraen-5-ium-10,13-bis(ylium)-4-ide",
     "P-16.3.2"),
    # -- the diazonium cation on a group that is no hydrocarbon,:41607)
    ("CC(C)CS(=O)(=O)[N+]#N", "(2-methylpropane-1-sulfonyl)diazonium", "P-73.2.2.3"),
    ("CCCCCCO[N+]#N", "(hexyloxy)diazonium", "P-73.2.2.3"),
    ("C1C(=O)N(C(=S)S1)[N+]#N", "(4-oxo-2-sulfanylidene-1,3-thiazolidin-3-yl)diazonium", "P-73.2.2.3"),
    ("COC1=CC=C(C=C1)C=N[N+]#N", "{[(4-methoxyphenyl)methylidene]amino}diazonium", "P-73.2.2.3"),
    # -- the azanide of a group on a heteroatom,:41079)
    ("CS(=O)(=O)[NH-]", "(methanesulfonyl)azanide", "P-72.2.2.2.4"),
    ("CS[NH-]", "(methylsulfanyl)azanide", "P-72.2.2.2.4"),
    ("CC(C)P(C(C)C)[NH-]", "[di(propan-2-yl)phosphanyl]azanide", "P-72.2.2.2.4"),
]


# The suite runs with the OPSIN validity gate off. A parent whose characteristic group has an ion beside it is offered
# a candidate that drops the charges ('3-[...-4-yl]propanoic acid') before the one of the ionic parent; the gate is what
# withdraws the first in a run of the tool, so these cases run with it on (the marker of tests/conftest.py).
_GATED = {c[0] for c in CASES if c[2] == "P-41"}


def _case(c):
    return pytest.param(*c, id=c[0], marks=pytest.mark.opsin_gate) if c[0] in _GATED else pytest.param(*c, id=c[0])


@pytest.mark.parametrize("smiles,expected,rule", [_case(c) for c in CASES])
def test_name_pinned_and_read_back(smiles, expected, rule):
    row = name_best_effort(smiles)
    assert row["name"] == expected, (rule, row["name"], row["tier"])
    assert _reads_back(smiles, row["name"])


@pytest.mark.parametrize("smiles", [c[0] for c in CASES])
def test_class_without_a_blue_book_example_is_no_pin(smiles):
    """The Blue Book gives no example of these ions: the name is valid, not preferred."""
    assert name_best_effort(smiles)["tier"] != "pin_verified"


# -- the names that were given before keep their spelling
CONTROLS = [
    ("CCO[N+](=O)[O-]", "ethyl nitrate"),
    ("NO[N+](=O)[O-]", "2-oxido-1,3-dioxa-2,4-diazabut-1-en-2-ium"),
    ("C[N+]#N", "methanediazonium"),
    ("CN(C)[N+]#N", "dimethylazanediazonium"),
    ("CN(C)[NH-]", "2,2-dimethylhydrazin-1-ide"),
    ("CC(=O)[NH-]", "acetylazanide"),
    ("C[B-]1(C)CCCCC1", "1,1-dimethylborinan-1-uide"),
    ("C1CCC[N+]1=[N-]", "(pyrrolidin-1-ium-1-ylidene)azanide"),
    ("CS(=O)(=O)[N-][N+]1=CC=CC=C1", "(methanesulfonyl)(pyridin-1-ium-1-yl)azanide"),
    ("[B-]1(F)(F)N2C(C)=CC(C)=C2C=C2C(C)=CC(C)=[N+]12",
     "2,2-difluoro-4,6,10,12-tetramethyl-1,3-diaza-2-boratricyclo[7.3.0.0^3,7]dodeca-1(12),4,6,8,10-pentaen-1-ium-2-uide"),
]


@pytest.mark.parametrize("smiles,expected", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_controls_keep_their_names(smiles, expected):
    assert name_best_effort(smiles)["name"] == expected


# -- the cumulative suffixes of a parent, without a JVM,:42417)
def _locants_of_charged_atoms(mol, locants):
    """``{atom index: locant}`` for the charged atoms of ``mol`` in the order of ``locants``: the anion first."""
    charged = sorted((a for a in mol.GetAtoms() if a.GetFormalCharge()), key=lambda a: (a.GetFormalCharge() > 0, a.GetIdx()))
    return {a.GetIdx(): loc for a, loc in zip(charged, locants)}


@pytest.mark.parametrize("smiles,locants,suffix", [
    # one cation and one anion of one parent: 'ium' before 'uide'
    ("[B-]1(F)(F)N2C=CC=C2C=C2C=CC=[N+]12", (2, 1), "-1-ium-2-uide"),
    # two cations of one ring: the multiplied suffix
    ("C[N+]1(C)CC[N+](C)(C)CC1", (1, 4), "-1,4-diium"),
])
def test_cumulative_charge_suffix_text(smiles, locants, suffix):
    from orthonym.assembly.general_engine import _cumulative_charge_suffix_text
    mol = Chem.MolFromSmiles(smiles)
    atom_to_locant = _locants_of_charged_atoms(mol, locants)
    text, ids = _cumulative_charge_suffix_text(mol, atom_to_locant)
    assert text == suffix
    assert set(ids) == set(atom_to_locant)


def test_a_cumulative_suffix_needs_a_centre_of_the_parent():
    from orthonym.assembly.general_engine import _cumulative_charge_suffix_text
    mol = Chem.MolFromSmiles("C[N+](C)(C)CC")
    carbons = {a.GetIdx(): i + 1 for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() == "C"}
    assert _cumulative_charge_suffix_text(mol, carbons) is None


# -- the nitrate ester oxygen, without a JVM,:36405)
@pytest.mark.parametrize("smiles,expected_atoms", [
    ("C1COCCN1O[N+](=O)[O-]", 4),     # on a nitrogen: the nitrate nitrogen is the one that holds the two terminal oxygens
    ("CCO[N+](=O)[O-]", 4),           # on a carbon, as before
    ("NO[N+](=O)[O-]", None),         # a carbon-free O-nitro hydroxylamine is a parent of its own (the inorganic route)
    ("C.[O-][N+](=O)O[N+](=O)[O-]", None),   # two nitro nitrogens on one oxygen: an anhydride, no nitrooxy group
])
def test_nitrooxy_atoms(smiles, expected_atoms):
    from orthonym.assembly.universal_substituent import _nitrooxy_atoms
    mol = Chem.MolFromSmiles(smiles)
    found = [len(atoms) for atoms in (_nitrooxy_atoms(mol, a.GetIdx()) for a in mol.GetAtoms()
                                      if a.GetSymbol() == "O") if atoms]
    assert (found[0] if found else None) == expected_atoms


@pytest.mark.parametrize("name,expected", [
    ("1,4-dihydroquinolin-4-yl", "1,4-dihydroquinolin-4-ylidene"),
    ("cyclohexa-2,5-dien-1-yl", "cyclohexa-2,5-dien-1-ylidene"),
    ("4-oxo-1,4-dihydronaphthalen-1-yl", "4-oxo-1,4-dihydronaphthalen-1-ylidene"),
    ("pyridin-1-ium-1-ylidene", ""),
    ("ethenylidyne", ""),
    ("cyclohexylidene", ""),
])
def test_as_ylidene(name, expected):
    from orthonym.rules.group_prefixes import _as_ylidene
    assert _as_ylidene(name) == expected


# -- the multiplied ionic suffixes, without a JVM (c), the Blue Book, example:7114)
@pytest.mark.parametrize("kind,n,expected", [
    ("ylium", 1, "ylium"), ("ylium", 2, "bis(ylium)"), ("ylium", 3, "tris(ylium)"), ("ylium", 4, "tetrakis(ylium)"),
    ("ium", 1, "ium"), ("ium", 2, "diium"), ("ium", 3, "triium"),
    ("ide", 2, "diide"), ("uide", 2, "diuide"), ("ylium", 7, None), ("ide", 11, None),
])
def test_multiplied_ionic_suffix(kind, n, expected):
    from orthonym.assembly.general_engine import _multiplied_ionic_suffix
    assert _multiplied_ionic_suffix(kind, n) == expected


@pytest.mark.parametrize("smiles,locants,suffix", [
    ("C[CH+]C[CH+]C", (2, 4), "-2,4-bis(ylium)"),            # the net-charged suffix of a parent (``_charge_suffix_text``)
    ("[CH2+]CC[CH2+]", (1, 4), "-1,4-bis(ylium)"),
    ("C[CH-]C[CH-]C", (2, 4), "-2,4-diide"),
])
def test_net_charged_suffix_of_a_parent(smiles, locants, suffix):
    from orthonym.assembly.general_engine import _charge_suffix_text
    mol = Chem.MolFromSmiles(smiles)
    charged = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()]
    atom_to_locant = {a.GetIdx(): i + 1 for i, a in enumerate(mol.GetAtoms())}
    text, ids = _charge_suffix_text(mol, atom_to_locant)
    assert text == suffix and set(ids) == set(charged)


def test_no_emitter_spells_diylium():
    """The class: no string of the package's code spells 'diylium' or 'triylium', or builds 'di'/'tri' in front of an
    'ylium' in an f-string (the docstrings that quote the rule are no code)."""
    import ast
    import pathlib
    import re
    import orthonym
    quoted = re.compile(r"\b(?:di|tri)ylium\b")
    built = re.compile(r"(?:di|tri|tetra)\{[^}]*\}ylium|(?:di|tri|tetra)\{[^}]*\}\(ylium")
    offenders = []
    for path in pathlib.Path(orthonym.__file__).parent.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                      and node.body and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                if quoted.search(node.value):
                    offenders.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.JoinedStr):
                text = "".join(v.value if isinstance(v, ast.Constant) else "{x}" for v in node.values)
                if built.search(text) or quoted.search(text):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders
