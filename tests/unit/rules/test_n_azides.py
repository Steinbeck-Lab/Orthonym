"""An azido group on nitrogen: the +1 of the group does not move, and the group is the prefix 'azido'.

 "AZIDES" (the Blue Book): "Compounds containing a -N3 (-N=N+=N-) group attached to a parent hydride,
are named using substitutive nomenclature and the prefix 'azido'. This method gives preferred IUPAC names rather
than names based on the class name 'azido' in functional class nomenclature (see also."
 "Azides may be named by three methods" (:43298): "(1) substitutively, using the compulsory prefix
azido; (2) by functional class nomenclature using the class name azide; (3) as derivatives of the
zwitterionic parent hydride 'triazadien-2-ium-1-ide.' Method (1) yields preferred IUPAC names".
"N-SUBSTITUTION OF AMINES AND IMINES BY HETEROATOMS" (:26588): "substitution on the nitrogen atom of amines and
imines was allowed for all characteristic groups cited as prefixes (see Table 5.1)".

``namer._charge_forms`` says that "a cumulated or dipolar nitrogen cation (an azide, a diazo compound, a
nitrilium) moves a -1, not a +1" and names such an input as drawn, but RDKit's enumeration of
``CN(C)N=[N+]=[N-]`` offered ``C[N+](C)=NN=[N-]`` (the +1 on the neutral amine nitrogen) and the form choice shipped
the name of THAT drawing, '4,4-dimethyltetraaz-1,3-dien-4-ium-1-ide', as pin_verified: a method (3) name whose
'-4-ium' and '-1-ide' sit on other atoms than the azide the input draws, under a label that claims the PIN of method
(1). ``charge_shift.accepted_charge_forms`` now takes no form in which an atom of an azido group (in any drawing,
``azide_atoms``) loses or gains the +1.

The N-azido amine, amide and sulfonamide have their preferred names ('N-azido-N-methylmethanamine',
'N-azidoacetamide', ``rules/n_hetero_amines.py``; ASSUMED reading, the book prints no N-azido amine: and
 above). The names asserted for the rows that remain are the best-effort names, read back by a FRESH OPSIN call
to the canonical SMILES of the input. The preferred name of the others ('3-[4-(azidoamino)phenyl]propanoic acid') is
not built: the polyfunctional assembler of ``composer`` carries the carbon-only N-substituent walk.
"""
import pytest
from rdkit import Chem, RDLogger

from tests.support.jars import jar_or_skip
from tests.support.rt_assert import _canon, _independent_parse, name_best_effort

RDLogger.DisableLog("rdApp.*")

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# the preferred names (PIN): the amine, the carboxamide and the sulfonamide are the class, the azido group a prefix of
# the N:26588,:25989,:18184 amides,:18192 amines).
# UNVERIFIED against the book: it prints no N-azido amine, amide or sulfonamide, so these names rest on the
# reading of 's general sentence ("all characteristic groups cited as prefixes (see Table 5.1)", azido is in
# Table 5.1:24990) and on the docstring above, which named 'N-azido-N-methylmethanamine' as the PIN before the
# producer existed. What would make them wrong: a Blue Book ruling that a nitrogen chain is the parent.
PIN_ROWS = [
    ("CNN=[N+]=[N-]", "N-azidomethanamine"),
    ("CN(C)N=[N+]=[N-]", "N-azido-N-methylmethanamine"),
    ("CCN(CC)N=[N+]=[N-]", "N-azido-N-ethylethanamine"),
    ("CC(=O)NN=[N+]=[N-]", "N-azidoacetamide"),
    ("CS(=O)(=O)N(C)N=[N+]=[N-]", "N-azido-N-methylmethanesulfonamide"),
]

ROWS = [
    ("C1=CC=CC=C1N(CC[N+](=O)[O-])N=[N+]=[N-]", "[azido(2-nitroethyl)amino]benzene"),
    ("OC(=O)CCc1ccc(NN=[N+]=[N-])cc1", "4-(azidoamino)-1-(3-hydroxy-3-oxopropyl)benzene"),
    ("CC(C)(C)OC(=O)N[C@@H](CCCCNN=[N+]=[N-])C(=O)O",
     "(2S)-6-(azidoamino)-2-[(tert-butoxycarbonyl)amino]-1-hydroxy-1-oxohexane"),
    ("C(CSSCCNN=[N+]=[N-])NN=[N+]=[N-]",
     "1-(azidoamino)-2-({[2-(azidoamino)ethyl]sulfanyl}sulfanyl)ethane"),
]


@pytest.mark.parametrize("smiles,name", PIN_ROWS)
def test_an_azide_on_the_nitrogen_of_an_amine_or_amide_is_a_prefix_of_its_preferred_name(smiles, name):
    from orthonym import Orthonym
    jar_or_skip()
    for res in (Orthonym(style="pin").name_tiered(smiles), name_best_effort(smiles)):
        assert (res["name"], res["tier"], res["is_pin"]) == (name, "pin_verified", True), res
    # an independent read: the azide drawing of the input, charges on the same atoms
    assert _canon(_independent_parse(name, smiles)) == _canon(smiles)


@pytest.mark.parametrize("smiles,name", ROWS)
def test_an_azide_on_nitrogen_is_named_with_the_prefix_azido(smiles, name):
    jar_or_skip()
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"]) == (name, "systematic_verified"), res
    assert res["is_pin"] is False
    # an independent read: the azide drawing of the input, charges on the same atoms
    assert _canon(_independent_parse(name, smiles)) == _canon(smiles)


@pytest.mark.parametrize("smiles", [r[0] for r in ROWS])
def test_the_default_tier_does_not_call_it_a_pin(smiles):
    """The PIN of these is the N-azido acid or arene with the prefix azido, which the engine does not build: the
    PIN tier declines instead of labelling a name that is no PIN."""
    from orthonym import Orthonym
    jar_or_skip()
    res = Orthonym(style="pin").name_tiered(smiles)
    assert res["tier"] != "pin_verified" and res.get("is_pin") is not True, res


@pytest.mark.parametrize("smiles", [
    "CN(C)N=[N+]=[N-]", "CCN(CC)N=[N+]=[N-]", "Cc1nc(N(C)N=[N+]=[N-])c(F)cc1F",
    "O=C1NCCCN1c1ccc(N=[N+]=[N-])cc1",          # an aryl azide with a donor nitrogen across the ring
])
def test_the_plus_of_an_azide_has_no_form(smiles):
    from orthonym import namer
    namer._charge_forms.cache_clear()
    assert namer._charge_forms(smiles) is None


def test_the_filter_still_takes_the_forms_of_an_amidinium_and_an_azolium():
    from orthonym import namer
    for smiles in ("CNC(=[NH2+])NC", "Cc1c[nH]c[nH+]1"):
        mol = Chem.MolFromSmiles(smiles)
        found = namer._accepted_charge_forms(mol, smiles, namer._rdkit_charge_candidates(mol))
        assert len(found) == 2, smiles


def test_the_atoms_of_the_group_in_each_drawing():
    from orthonym.perception.charge_shift import azide_atoms
    azide = Chem.MolFromSmiles("CN(C)N=[N+]=[N-]")
    assert azide_atoms(azide) == frozenset({3, 4, 5})
    twin = Chem.MolFromSmiles("CC[N-][N+]#N")
    assert azide_atoms(twin) == frozenset({2, 3, 4})
    shifted = Chem.MolFromSmiles("C[N+](C)=NN=[N-]")            # the +1 on the host: the host is in the group
    assert azide_atoms(shifted) == frozenset({1, 3, 4, 5})
    for smiles in ("C[N-][N+](C)(C)C", "CC#[N+][N-]C", "[N-]=[N+]=[N-]", "C[N+]#N", "N=[N+]=[N-]"):
        assert azide_atoms(Chem.MolFromSmiles(smiles)) == frozenset()


def test_hydrazoic_acid_keeps_its_zwitterionic_name():
    """HN3 has no host for the prefix 'azido': 'triaz-1,2-dien-2-ium-1-ide' is the only name it has
    method (3) names the group of an azide, method (1) needs a parent hydride to carry it), so it keeps its label."""
    from orthonym import Orthonym
    jar_or_skip()
    for smiles in ("N=[N+]=[N-]", "[N-]=[N+]=N"):
        res = Orthonym(style="pin").name_tiered(smiles)
        assert (res["name"], res["tier"]) == ("triaz-1,2-dien-2-ium-1-ide", "pin_verified"), res


def test_an_input_drawn_with_the_plus_on_the_amine_is_named_as_drawn():
    """'4,4-dimethyltetraaz-1,3-dien-4-ium-1-ide' reads to that drawing, so it names it; it is a method (3) name
    , so the label is below the PIN. The name never says 'azido': that reads to the other drawing."""
    jar_or_skip()
    smiles = "C[N+](C)=NN=[N-]"
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"]) == ("4,4-dimethyltetraaz-1,3-dien-4-ium-1-ide", "systematic_verified"), res
    assert _canon(_independent_parse(res["name"], smiles)) == _canon(smiles)
    assert _canon(_independent_parse(res["name"], smiles)) != _canon("CN(C)N=[N+]=[N-]")   # the other drawing
    from orthonym import Orthonym
    assert Orthonym(style="pin").name_tiered(smiles)["tier"] != "pin_verified"


def test_the_zwitterion_name_is_not_offered_for_the_azide_drawing():
    jar_or_skip()
    assert "tetraaz" not in (name_best_effort("CN(C)N=[N+]=[N-]")["name"] or "")


def test_azidobenzene_and_the_twin_of_an_azide_on_carbon_are_unchanged():
    """: 'azidobenzene (PIN)'; an azide on carbon is not an azide on nitrogen."""
    from orthonym import Orthonym
    jar_or_skip()
    for smiles, name in (("[N-]=[N+]=Nc1ccccc1", "azidobenzene"), ("CC[N-][N+]#N", "azidoethane")):
        res = Orthonym(style="pin").name_tiered(smiles)
        assert (res["name"], res["tier"]) == (name, "pin_verified")


def test_the_azido_branch_of_an_amino_prefix_is_named():
    """Pass 1b of ``name_substituent_fragment``: an azido group on the amino nitrogen is a branch of the amino prefix
    , not a hetero branch it cannot name ('azido(methyl)azanyl' was the fallback's name, another group)."""
    from orthonym.assembly.substituent_naming import name_substituent_fragment
    for smiles, name in (("CCNN=[N+]=[N-]", "(azidoamino)methyl"),
                         ("CCN(C)N=[N+]=[N-]", "[azido(methyl)amino]methyl"),
                         ("CCN(C)C", "(dimethylamino)methyl")):
        mol = Chem.MolFromSmiles(smiles)
        assert name_substituent_fragment(mol, list(range(1, mol.GetNumAtoms())), 1, [0]) == name
