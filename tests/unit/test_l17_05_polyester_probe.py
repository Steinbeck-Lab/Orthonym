"""A polyester of a ring-bearing alcohol is named as that polyester, not as a chain that holds the ring (lane/l17 item 5).

 "Polyester names formed by using substitutive nomenclature, and/or multiplicative nomenclature, and
functional class nomenclature" (the Blue Book): "Polyesters that cannot be named by functional class
multiplicative nomenclature as described above, are named by using substitutive nomenclature to generate the names of
the organyl substituents and those of the anions. When required, seniority of rings, ring systems, and chains; number
and location of substituents; and alphanumerical order is applied in forming the alcoholic component of a functional
class ester name", with '2-{2-[(acetyloxy)methyl]phenyl}ethyl acetate (PIN) (not 2-{2-[(acetyloxy)ethyl]phenyl}methyl
acetate; the ethyl chain is senior to the methyl chain)' (:31899).

The PIN tier gave no name for that molecule. The decomposition engine splits a molecule only when the name the pipeline
gives it without the split (the probe, ``assembly.ester_prefix_probe.probe_name``) is poor, and the probe held
'1,9-bis(acetyloxy)nonane' (C13H24O4, for a C13H16O4 molecule): ``rules.esters._order_backbone_chain`` took a walk that
reaches every atom of the carbon backbone for proof that the backbone is a chain, and the walk of an ortho-phenylene with a
chain on each side reaches the six ring atoms with the ring-closing bond left out. The name never shipped (the read-back
suppressed it) but it passed the engine's quality gate, so the split was never made. The producer
(``name_polyol_polyester``) also built '<alkane>ane' names for a backbone with a double bond or a substituent that the
name does not hold ('1,3-bis(acetyloxy)propane' for 2-chloropropane-1,3-diyl diacetate).

The backbone is now a path (no atom with three backbone neighbours, one bond fewer than atoms) and a bare alkane whose only
neighbours are its own atoms and the ester oxygens "Esters cited as prefixes",:31696: the acyloxy groups are
the prefixes of the bare hydride).

The polyester of one alcoholic component whose ester groups sit on identical arms of a ring (or on one substituted chain)
is named by its multivalent organyl group, "Polyesters formed from a single 'alcoholic' component" (:31815):
"Esters derived from a single 'polyhydroxylic' component are named by placing the name of the 'polyhydroxylic' component
denoted by a multivalent organyl group (alkyl, aryl, etc.) in front of the name(s) of the acid component",
(:31819) "When anions are identical functional class multiplicative nomenclature is used", with the book's own
'1,2-phenylenedi(propan-3,1-yl) diacetate [PIN; see (2) and ]' (:31829) and, for a ring that carries a third
ester group, '2-[3-(formyloxy)propyl]cyclohexane-1,1-diyldi(propane-3,1-diyl) diacetate (PIN)' (:31878). Before, the probe
named those molecules '{2-[(acetyloxy)methyl]phenyl}methyl acetate' (the acyloxy form of the substitutive rule for unequal
arms) and the PIN tier labelled it a PIN.

The group is the central multiplicative group of the alcoholic component and the rule asks for identical arms, not for a
symmetrical polyol (1)-(3),:23178;,:5208: "Substitution, expressed by prefixes or implied, is
allowed on simple multiplicative groups, symmetrical or unsymmetrical"): the central ring may carry a chloro, methyl or
hydroxy group ('(4-chloro-1,3-phenylene)bis(methylene) diacetate'), be a fused ring system ('naphthalene-1,5-diyl
diacetate'), bear two valences on one atom ('cyclopropane-1,1-diylbis(methylene)'), be an azole of one NH ('1H-pyrazole-
3,5-diyl'), or state its configuration ('[(1R,2R)-cyclohexane-1,2-diyl]bis(methylene) diacetate',.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import assert_full_rt
from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")

# the Blue Book row,:31899)
_TARGET = "CC(=O)OCCc1ccccc1COC(C)=O"
_TARGET_PIN = "2-{2-[(acetyloxy)methyl]phenyl}ethyl acetate"


def _matches(mol):
    return [tuple(m) for m in mol.GetSubstructMatches(_ESTER)]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_book_row_is_named_at_both_tiers(tier):
    """The PIN of (:31899), verified by the round trip."""
    row = tier_row(_TARGET, tier)
    assert row["name"] == _TARGET_PIN, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], _TARGET)


# an ester whose alcohol component is an ortho-disubstituted ring with an ester-bearing chain on each side: the producer
# read the ring as a chain of carbons (the walk of one end to the other covers the six ring atoms, the ring bond unread)
_RING_CLOSED = [
    "CC(=O)OCCc1ccccc1COC(C)=O",                 # 1,9-bis(acetyloxy)nonane
    "CC(=O)OCc1ccccc1COC(C)=O",                  # 1,8-bis(acetyloxy)octane
    "CCC(=O)OCCc1ccccc1COC(C)=O",                # 9-(acetyloxy)-1-(propanoyloxy)nonane
    "O=C(OCc1ccccc1COC(=O)c1ccccc1)c1ccccc1",    # 1,8-bis(benzoyloxy)octane
    "CC(=O)OCC1CCCCC1COC(C)=O",                  # a saturated ring is no chain either
]
#... and a ring that its heteroatom closes: the carbons of its backbone are a path, the oxygen beside them is not held
_RING_WITNESSES = _RING_CLOSED + ["CC(=O)OCc1ccc(COC(C)=O)o1"]


@pytest.mark.parametrize("smiles", _RING_CLOSED)
def test_a_ring_is_no_chain_for_the_polyol_backbone(smiles):
    """``_order_backbone_chain``: a backbone with a ring is not a chain, whatever walk covers its atoms."""
    from orthonym.rules.esters import _find_polyol_backbone, _order_backbone_chain
    mol = Chem.MolFromSmiles(smiles)
    matches = _matches(mol)
    exclude = {m[2] for m in matches} | {m[0] for m in matches}
    backbone = _find_polyol_backbone(mol, [m[3] for m in matches], exclude)
    assert backbone is not None and any(mol.GetAtomWithIdx(i).IsInRing() for i in backbone)
    assert _order_backbone_chain(mol, backbone, exclude) is None


@pytest.mark.parametrize("smiles", _RING_WITNESSES + [
    "CC(=O)OCC=CCOC(C)=O",           # a double bond in the backbone: 'butane' would hold two atoms fewer hydrogens
    "CC(=O)OCC(Cl)COC(C)=O",         # a chlorine the '<alkane>ane' name does not hold
    "CC(=O)OCC(O)COC(C)=O",          # a hydroxy group
])
def test_the_polyol_producer_declines_a_backbone_its_name_does_not_hold(smiles):
    from orthonym.rules.esters import name_polyol_polyester
    mol = Chem.MolFromSmiles(smiles)
    assert name_polyol_polyester(mol, _matches(mol)) is None


@pytest.mark.parametrize("smiles", _RING_WITNESSES[:4])
def test_the_probe_holds_no_name_of_another_molecule(smiles):
    """The probe of the decomposition engine gives no name, or a name of the molecule, at the PIN tier."""
    from orthonym.assembly.ester_prefix_probe import probe_name
    name = probe_name(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)), "pin")
    assert not name or ("nonane" not in name and "octane" not in name), name
    if name:
        assert_full_rt(name, smiles)


@pytest.mark.parametrize("smiles", _RING_WITNESSES)
def test_the_class_names_the_right_molecule_at_best_effort(smiles):
    """Every witness is named, and the name round-trips: the split is made and the ester is named by its class."""
    row = tier_row(smiles, "best-effort")
    assert row["name"], row
    assert_full_rt(row["name"], smiles)


# a symmetrical polyol of two parents: ring, hetero atom or chain between identical arms, named by the multivalent organyl
# group in front of the multiplied anion,:31819; (2),:5141;,:23178)
_MULTIPLICATIVE = [
    ("CC(=O)OCc1ccccc1COC(C)=O", "1,2-phenylenebis(methylene) diacetate"),
    ("O=C(OCc1ccccc1COC(=O)c1ccccc1)c1ccccc1", "1,2-phenylenebis(methylene) dibenzoate"),
    ("CC(=O)OCC1CCCCC1COC(C)=O", "cyclohexane-1,2-diylbis(methylene) diacetate"),
    ("CC(=O)OCc1ccc(COC(C)=O)o1", "furan-2,5-diylbis(methylene) diacetate"),
    ("CC(=O)OCCc1ccccc1CCOC(C)=O", "1,2-phenylenedi(ethane-2,1-diyl) diacetate"),
    ("CC(=O)OCCC1CC1CCOC(C)=O", "cyclopropane-1,2-diyldi(ethane-2,1-diyl) diacetate"),
    ("CC(=O)OCc1cccc(COC(C)=O)c1", "1,3-phenylenebis(methylene) diacetate"),
    ("CC(=O)OCc1ccc(COC(C)=O)cc1", "1,4-phenylenebis(methylene) diacetate"),
    ("O=C(OCc1ccc(COC(=O)c2ccccc2)cc1)c1ccccc1", "1,4-phenylenebis(methylene) dibenzoate"),
    ("CC(=O)OCC1CCC(COC(C)=O)CC1", "cyclohexane-1,4-diylbis(methylene) diacetate"),
]


@pytest.mark.parametrize("smiles,name", _MULTIPLICATIVE)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_a_polyol_of_identical_arms_is_named_by_its_multiplicative_organyl_group(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


# a polyol that is one substituted or unsaturated chain: the diyl group of the chain with its prefixes,:31819;
# for the group), where the bare '<alkane>ane' name of the acyloxy form held fewer atoms than the molecule
_DIYL = [
    ("CC(=O)OCC(Cl)COC(C)=O", "2-chloropropane-1,3-diyl diacetate"),
    ("CC(=O)OCC(C)(C)COC(C)=O", "2,2-dimethylpropane-1,3-diyl diacetate"),
    ("CC(=O)OCC=CCOC(C)=O", "but-2-ene-1,4-diyl diacetate"),
]


@pytest.mark.parametrize("smiles,name", _DIYL)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_a_polyol_of_one_substituted_chain_is_named_by_its_diyl_group(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


# the ester groups on one ring: the multivalent group of one parent,:31819; the ring cascade cited every
# principal ester as an acyloxy prefix, a name no PIN is)
_RING_POLYOL = [
    ("CC(=O)OC1CCCCC1OC(C)=O", "cyclohexane-1,2-diyl diacetate"),
    ("CC(=O)OC1CCC(OC(C)=O)CC1", "cyclohexane-1,4-diyl diacetate"),
    ("CC(=O)Oc1ccccc1OC(C)=O", "1,2-phenylene diacetate"),
    ("O=C(Oc1ccc(OC(=O)c2ccccc2)cc1)c1ccccc1", "1,4-phenylene dibenzoate"),
]

# the ester groups of one alcoholic component beside a group of a class below the esters, Table 4.1): the group is a
# prefix of the multivalent organyl group, or a hetero atom is its central unit
_BESIDE_A_GROUP = [
    ("CC(=O)OCCOCCOC(C)=O", "oxydi(ethane-2,1-diyl) diacetate"),
    ("CC(=O)OCCSCCOC(C)=O", "sulfanediyldi(ethane-2,1-diyl) diacetate"),
    ("CC(=O)OCCN(C)CCOC(C)=O", "(methylazanediyl)di(ethane-2,1-diyl) diacetate"),
    ("CC(=O)OCC(=O)COC(C)=O", "2-oxopropane-1,3-diyl diacetate"),
]


@pytest.mark.parametrize("smiles,name", _RING_POLYOL + _BESIDE_A_GROUP)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_polyol_of_one_ring_or_beside_a_group_is_named_by_its_organyl_group(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_ring_and_chain_book_row_is_named_by_the_functional_class(tier):
    """ (:31856): '2-[2-(acetyloxy)ethyl]phenyl acetate (PIN) (not 2-[2-(acetyloxy)phenyl]ethyl acetate;
    the ring is senior to the chain)' (:31897). The ring cascade named the molecule '1-(acetyloxy)-2-[2-(acetyloxy)ethyl]
    benzene', every ester an acyloxy prefix, so the PIN tier gave no name and the wider tiers a name below the PIN."""
    smiles = "CC(=O)OCCc1ccccc1OC(C)=O"
    row = tier_row(smiles, tier)
    assert row["name"] == "2-[2-(acetyloxy)ethyl]phenyl acetate", row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


_STEROID_DIACETATE = "CC(=O)O[C@H]1CC[C@@]2(C)C(=CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](OC(C)=O)CC[C@@H]32)C1"


def test_a_steroid_diacetate_is_no_pin_with_an_acyloxy_prefix():
    """The ring system of a steroid is one parent: '5α-cholestane-3β,6α-diyl diacetate',:31979), so the
    name '...-17-(acetyloxy)-...-3-yl acetate' of the natural-product route is no PIN at either tier."""
    assert tier_row(_STEROID_DIACETATE, "pin")["tier"] == "abstain"
    row = tier_row(_STEROID_DIACETATE, "best-effort")
    assert row["tier"] != "pin_verified" and not row["is_pin"], row
    assert "(acetyloxy)" not in row["name"], row
    assert_full_rt(row["name"], _STEROID_DIACETATE)


def test_the_symmetric_polyol_of_two_anions_is_no_pin():
    """ (:31831): the PIN is method (1), '1,2-phenylenebis(methylene) acetate propanoate', which OPSIN 2.9.0
    cannot read; the acyloxy form is method (2), acceptable in general nomenclature only."""
    smiles = "CC(=O)OCc1ccccc1COC(=O)CC"
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    row = tier_row(smiles, "best-effort")
    assert row["tier"] != "pin_verified" and not row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [
    "CC(=O)OCCc1ccccc1COC(C)=O",         # unequal arms: (:31856), the book row of the module docstring
    "CC(=O)OCCc1ccccc1OC(C)=O",          # a ring and a chain
])
def test_the_organyl_group_is_not_named_for_unequal_arms(smiles):
    """The reader of the group needs identical arms (1)-(3)): with unequal arms no group is named, and the
    polyester stays with the substitutive producers."""
    from orthonym.rules.multiplicative_general import name_ester_organyl_group
    mol = Chem.MolFromSmiles(smiles)
    assert name_ester_organyl_group(mol, _matches(mol)) is None


def test_the_organyl_group_is_not_named_for_a_second_alcohol():
    from orthonym.rules.multiplicative_general import name_ester_organyl_group
    for smiles in ("CCOC(C)=O.CC(=O)OCC",                  # two molecules
                   "CC(=O)OCCC(=O)OCCOC(C)=O",             # the second alcohol is bonded to the first acid
                   "CCOC(=O)CCC(=O)OCC"):                  # two alcoholic components
        mol = Chem.MolFromSmiles(smiles)
        assert name_ester_organyl_group(mol, _matches(mol)) is None, smiles


def test_the_mixed_anion_row_with_unequal_arms_is_named_by_substitutive_nomenclature():
    """Acetate and propanoate on the same alcohol with unequal arms: the Blue Book prints no such row. The polyester is
    named by (:31856) as the acetate row of the module docstring is, the senior anion (the propanoate)
    being the principal one."""
    smiles = "CCC(=O)OCCc1ccccc1COC(C)=O"
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert row["name"] == "2-{2-[(acetyloxy)methyl]phenyl}ethyl propanoate", row
        assert_full_rt(row["name"], smiles)


# the controls: a clean acyclic backbone is named as before (the functional class name where the anions are identical,
#:31819, the acyloxy form of the producer at the best-effort tier where they are not)
_CONTROLS = [
    ("CC(=O)OCCCCCCCCOC(C)=O", "octane-1,8-diyl diacetate", "octane-1,8-diyl diacetate"),
    ("CC(=O)OC(C)C(C)OC(C)=O", "butane-2,3-diyl diacetate", "butane-2,3-diyl diacetate"),
    ("CC(=O)OCC(OC(C)=O)COC(C)=O", "propane-1,2,3-triyl triacetate", "propane-1,2,3-triyl triacetate"),
    # different anions: the PIN is method (1) of (:31831), 'butane-1,4-diyl acetate propanoate', which
    # OPSIN 2.9.0 cannot read (the read-back is how a name ships), so the acyloxy form of method (2), which the book
    # accepts in general nomenclature, is the name and no tier labels it a PIN
    ("CCC(=O)OCCCCOC(C)=O", "", "1-(acetyloxy)-4-(propanoyloxy)butane"),
]


@pytest.mark.parametrize("smiles,pin,best_effort", _CONTROLS)
def test_the_controls_keep_their_names(smiles, pin, best_effort):
    pin_row = tier_row(smiles, "pin")
    assert (pin_row["name"] == pin) if pin else (pin_row["tier"] == "abstain"), pin_row
    row = tier_row(smiles, "best-effort")
    assert row["name"] == best_effort, row
    assert_full_rt(row["name"], smiles)


def test_the_clean_backbone_is_a_path_of_its_atoms():
    from orthonym.rules.esters import _find_polyol_backbone, _is_clean_alkane_backbone, _order_backbone_chain
    mol = Chem.MolFromSmiles("CC(=O)OCCCCOC(C)=O")
    matches = _matches(mol)
    exclude = {m[2] for m in matches} | {m[0] for m in matches}
    backbone = _find_polyol_backbone(mol, [m[3] for m in matches], exclude)
    ordered = _order_backbone_chain(mol, backbone, exclude)
    assert ordered is not None and len(ordered) == 4
    assert _is_clean_alkane_backbone(mol, ordered, {m[2] for m in matches})
    # the same chain beside an oxygen that is not one of the ester oxygens is not clean
    assert not _is_clean_alkane_backbone(mol, ordered, set())


# ---------------------------------------------------------------------------------------------------------------------
# fix a performance pass: the central group of the organyl group may be substituted,:5208), a fused ring system, a
# ring with two free valences on one atom, an azole of one NH, a group that states its configuration; the arms are
# identical, not the whole polyol symmetrical (1)-(3),:23178)
# ---------------------------------------------------------------------------------------------------------------------
# the central group carries a substituent group of its own: '(4-chloro-1,3-phenylene)bis(methylene)',
# the Blue Book: "Substitution, expressed by prefixes or implied, is allowed on simple multiplicative groups,
# symmetrical or unsymmetrical"). The old name, '{3-[(acetyloxy)methyl]-4-chlorophenyl}methyl acetate', cites one of two
# identical arms as an acyloxy prefix (1): identical linking bonds;,:31819: identical anions).
_CENTRAL_GROUP = [
    ("CC(=O)OCc1ccc(Cl)c(COC(C)=O)c1", "(4-chloro-1,3-phenylene)bis(methylene) diacetate"),
    ("CC(=O)OCc1ccc(C)c(COC(C)=O)c1", "(4-methyl-1,3-phenylene)bis(methylene) diacetate"),
    ("CC(=O)OCc1ccc(O)c(COC(C)=O)c1", "(4-hydroxy-1,3-phenylene)bis(methylene) diacetate"),
    ("CC(=O)OCc1cc(C)cc(COC(C)=O)c1", "(5-methyl-1,3-phenylene)bis(methylene) diacetate"),
    ("CC(=O)OCc1ccc(Cl)cc1COC(C)=O", "(4-chloro-1,2-phenylene)bis(methylene) diacetate"),
    ("CC(=O)OCc1ccc(COC(C)=O)c(Cl)c1", "(2-chloro-1,4-phenylene)bis(methylene) diacetate"),
    ("CC(=O)OCC1CC(C)CCC1COC(C)=O", "(4-methylcyclohexane-1,2-diyl)bis(methylene) diacetate"),
    ("CC(=O)Oc1ccc(Cl)cc1OC(C)=O", "4-chloro-1,2-phenylene diacetate"),
    ("CC(=O)OC1CCC(C)CC1OC(C)=O", "4-methylcyclohexane-1,2-diyl diacetate"),
    # an azole of one NH:, the indicated hydrogen takes the locant 1
    ("CC(=O)OCc1cc(COC(C)=O)n[nH]1", "1H-pyrazole-3,5-diylbis(methylene) diacetate"),
    ("CC(=O)OCc1ccc(COC(C)=O)[nH]1", "1H-pyrrole-2,5-diylbis(methylene) diacetate"),
]


@pytest.mark.parametrize("smiles,name", _CENTRAL_GROUP)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_central_group_of_the_organyl_group_may_carry_substituents(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


# a ring system of several rings, a ring atom with two free valences, a pyridine ring: the group of the central unit named
# with the numbering of the system, and the lowest locants of the free valences
_CENTRAL_RING_SYSTEM = [
    ("CC(=O)Oc1cccc2c(OC(C)=O)cccc12", "naphthalene-1,5-diyl diacetate"),
    ("CC(=O)OCc1ccc2cc(COC(C)=O)ccc2c1", "naphthalene-2,6-diylbis(methylene) diacetate"),
    ("CC(=O)OC1CCC2CC(OC(C)=O)CCC2C1", "decahydronaphthalene-2,6-diyl diacetate"),
    ("CC(=O)Oc1ccc2ncccc2c1OC(C)=O", "quinoline-5,6-diyl diacetate"),
    ("CC(=O)OCC1(COC(C)=O)CC1", "cyclopropane-1,1-diylbis(methylene) diacetate"),
    ("CC(=O)OCc1cccnc1COC(C)=O", "pyridine-2,3-diylbis(methylene) diacetate"),
    ("CC(=O)OCc1ccncc1COC(C)=O", "pyridine-3,4-diylbis(methylene) diacetate"),
]


@pytest.mark.parametrize("smiles,name", _CENTRAL_RING_SYSTEM)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_central_group_may_be_a_ring_system_or_have_two_valences_on_one_atom(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


# the central group states its configuration:, the descriptors of a group stand in front of its name, in one
# parenthesis in the order of the locants; with a choice of numbering the R descriptor takes the lower locant NUMBERING
# (j),:3346: "When there is a choice for lower locants related to the presence of stereogenic centers or stereoisomers, the
# lower locant is assigned to CIP stereodescriptors Z, R, M, and r [...] that are preferred to E, S, P, and s")
_CENTRAL_STEREO = [
    ("CC(=O)OC[C@@H]1CCCC[C@H]1COC(C)=O", "[(1R,2R)-cyclohexane-1,2-diyl]bis(methylene) diacetate"),
    ("CC(=O)OC[C@H]1CCCC[C@H]1COC(C)=O", "[(1R,2S)-cyclohexane-1,2-diyl]bis(methylene) diacetate"),
    ("CC(=O)OC[C@@H]1CCC[C@H]1COC(C)=O", "[(1R,2R)-cyclopentane-1,2-diyl]bis(methylene) diacetate"),
    ("CC(=O)O[C@H]1CCCC[C@@H]1OC(C)=O", "(1S,2S)-cyclohexane-1,2-diyl diacetate"),
    ("CC(=O)OC[C@@H](C)OC(C)=O", "(2R)-propane-1,2-diyl diacetate"),
    ("CC(=O)O[C@H](C)[C@@H](C)OC(C)=O", "(2R,3R)-butane-2,3-diyl diacetate"),
]


@pytest.mark.parametrize("smiles,name", _CENTRAL_STEREO)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_central_group_states_its_configuration(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_a_stereocentre_of_an_arm_is_not_cited_by_the_organyl_group():
    """A stereocentre in an arm (not the central group) has no descriptor in this writer: the group is not named."""
    from orthonym.rules.multiplicative_general import name_ester_organyl_group
    mol = Chem.MolFromSmiles("CC(=O)OC[C@H](C)c1ccccc1[C@H](C)COC(C)=O")
    assert name_ester_organyl_group(mol, _matches(mol)) is None


def test_the_ring_assembly_group_is_the_book_spelling_at_the_pin_tier_and_the_readable_one_beyond():
    """ (:5238) cites the central group first; the book writes a ring assembly group without further
    enclosing marks ('[1,1'-biphenyl]-4,4'-diylbis(oxy)',:6126). OPSIN 2.9.0 reads it only enclosed once more, so the
    PIN tier ships the book spelling through its twin (``assembly_diyl_enclosed``, the read-back of the twin to the full
    InChIKey) and a wider tier ships the spelling OPSIN reads, labelled below the PIN."""
    from orthonym.validation.twin_spellings import twin_of
    smiles = "CC(=O)OCc1ccc(-c2ccc(COC(C)=O)cc2)cc1"
    pin = tier_row(smiles, "pin")
    assert pin["name"] == "[1,1'-biphenyl]-4,4'-diylbis(methylene) diacetate", pin
    assert pin["tier"] == "pin_verified" and pin["is_pin"], pin
    twin, rules = twin_of(pin["name"])
    assert rules == ("assembly_diyl_enclosed",)
    assert twin == "([1,1'-biphenyl]-4,4'-diyl)bis(methylene) diacetate"
    assert_full_rt(twin, smiles)
    wide = tier_row(smiles, "best-effort")
    assert wide["name"] == twin, wide
    assert wide["tier"] == "systematic_verified" and not wide["is_pin"], wide
    assert_full_rt(wide["name"], smiles)


def test_the_twin_of_a_ring_assembly_group_is_no_rewrite_of_another_name():
    from orthonym.validation.twin_spellings import twin_of
    for name in ("[1,1'-biphenyl]-4-yl acetate",
                 "10,10'-[[1,1'-biphenyl]-4,4'-diylbis(oxy)]di(decanoic acid)",
                 "([1,1'-biphenyl]-4,4'-diyl)bis(methylene) diacetate"):
        assert twin_of(name) is None, name


# the mixed anions on one ring: method (2) of (:31831), the name of the split, is no PIN (method (1) is) and
# a wider tier ships it where the ring cascade used to ship the name that cites both esters as prefixes
_MIXED_ANIONS_ON_A_RING = [
    ("CC(=O)Oc1ccc(OC(=O)CC)cc1", "4-(acetyloxy)phenyl propanoate"),
    ("CC(=O)Oc1ccccc1OC(=O)CC", "2-(acetyloxy)phenyl propanoate"),
    ("CCC(=O)OC1CCCCC1OC(C)=O", "2-(acetyloxy)cyclohexyl propanoate"),
]


@pytest.mark.parametrize("smiles,name", _MIXED_ANIONS_ON_A_RING)
def test_a_wider_tier_names_the_mixed_anions_of_one_ring_by_method_two(smiles, name):
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    assert row["tier"] == "systematic_verified" and not row["is_pin"], row
    assert_full_rt(row["name"], smiles)


# the arms are unlike by a label: (:31856) names the polyester by substitutive nomenclature and
# (:21473) makes the chain with the modified atoms the senior parent, which the splice of the descriptor into the name of
# the unlabelled skeleton does not decide: the name stays and is no PIN
@pytest.mark.parametrize("smiles", [
    "CC(=O)O[13CH2]c1ccccc1COC(C)=O",
    "CC(=O)OCc1ccccc1COC(=O)[13CH3]",
])
def test_a_polyester_whose_arms_a_label_makes_unlike_keeps_its_name_below_the_pin(smiles):
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    row = tier_row(smiles, "best-effort")
    assert row["name"] and row["tier"] == "systematic_verified" and not row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)O[13CH2]c1ccccc1[13CH2]OC(C)=O", "1,2-phenylenebis[(13C)methylene] diacetate"),
    ("CC(=O)OC([2H])([2H])c1ccccc1C([2H])([2H])OC(C)=O", "1,2-phenylenebis[(2H2)methylene] diacetate"),
])
def test_a_polyester_whose_arms_a_label_keeps_alike_is_named_by_the_group(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)


# M2: the 'e' of 'ane', 'ene' and 'yne' is elided only before a vowel;: the multiplying prefix of
# 'diyl' starts with a consonant ('but-2-ene-1,4-diyl', the Blue Book; 'cyclopent-3-ene-1,2-diyl',:17278;
# '5α-cholestane-3β,6α-diyl diacetate',:31979)
_STEROID_ENE = "CC(=O)O[C@H]1CC[C@@]2(C)C(=CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](OC(C)=O)CC[C@@H]32)C1"
_STEROID_AN = "CC(=O)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](OC(C)=O)CC[C@@H]32)C1"
_ESTRADIOL = "C[C@]12CC[C@H]3[C@@H](CCc4cc(OC(C)=O)ccc34)[C@@H]1CC[C@@H]2OC(C)=O"


@pytest.mark.parametrize("smiles,name", [
    (_STEROID_ENE, "androst-5-ene-3β,17β-diyl diacetate"),
    (_STEROID_AN, "5β-androstane-3β,17β-diyl diacetate"),
    (_ESTRADIOL, "estra-1,3,5(10)-triene-3,17β-diyl diacetate"),
])
def test_the_e_of_a_steroid_parent_stays_before_a_multiplied_yl(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    assert row["tier"] != "pin_verified" and not row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_a_class_named_polyester_is_noted_by_its_producer():
    """The record is of the naming call (``assembly.memo.side_put``): read in its scope, for the molecule the producer built the
    name of, and gone with the scope."""
    from orthonym.assembly import memo
    from orthonym.rules.esters import _try_functional_class_diol_diester, is_class_named_polyester
    mol = Chem.MolFromSmiles("CC(=O)OCc1ccc(Cl)c(COC(C)=O)c1")
    token = memo.push_scope()
    try:
        name = _try_functional_class_diol_diester(mol, _matches(mol))
        assert name == "(4-chloro-1,3-phenylene)bis(methylene) diacetate"
        assert is_class_named_polyester(name, mol) and not is_class_named_polyester("pentyl acetate", mol)
        assert not is_class_named_polyester(name, Chem.MolFromSmiles("CC(=O)OCc1ccccc1COC(C)=O"))     # another structure
    finally:
        memo.pop_scope(token)
    token = memo.push_scope()
    try:
        assert not is_class_named_polyester(name, mol)           # the next naming call reads nothing of this one
    finally:
        memo.pop_scope(token)
