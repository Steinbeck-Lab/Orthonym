"""Guards of ``assembly.strict_ring_prefix`` (PB lane R): pure functions of a molecule and a fragment.

Each guard is checked on the molecule it exists for and on one it must not touch. Rule anchors are in
the module docstring and the guard docstrings (the Blue Book section headings and lines).
"""
import pytest
from rdkit import Chem

from orthonym.assembly import strict_ring_prefix as srp
from orthonym.assembly.memo import pin_promotion_var

pytestmark = pytest.mark.unit


def _frag(smiles, parent, attach):
    """(mol, atoms of the fragment that holds ``attach`` once the bond parent-attach is cut, attach)."""
    mol = Chem.MolFromSmiles(smiles)
    cut = Chem.RWMol(mol)
    cut.RemoveBond(parent, attach)
    frag = next(f for f in Chem.GetMolFrags(cut) if attach in f)
    return mol, set(frag), attach


def _core(mol, frag, attach):
    from orthonym.assembly import substituent_enumerator as se
    return se._ring_system_core_atoms(mol, attach, frag, mol.GetRingInfo())


# -- attachment, charge ---------------------------------------------------------------------------

def test_single_attachment_is_a_yl_only():
    mol, frag, a = _frag("CC(=O)C1(C)CC=C(C)CC1", 1, 3)
    assert srp.single_attachment(mol, frag, a)
    # a ring carbon joined by a double bond to the parent would be an ylidene
    mol, frag, a = _frag("CC(C)=C1CCCCC1", 1, 3)
    assert not srp.single_attachment(mol, frag, a)


def test_charged_core_is_refused():
    mol, frag, a = _frag("C[n+]1ccccc1CC(=O)O", 7, 6)
    assert srp.charged_core(mol, _core(mol, frag, a))
    mol, frag, a = _frag("Cn1cccc1CC(=O)O", 6, 5)
    assert not srp.charged_core(mol, _core(mol, frag, a))


# -- carbohydrate ring ----------------------------------------------------------------

def test_carbohydrate_core_is_refused():
    # the oxolane of an imidazole ribotide (d692): ring O + three ring carbons carrying O
    mol, frag, a = _frag("NC(=O)c1ncn([C@@H]2O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]2O)c1N", 6, 7)
    assert srp.carbohydrate_core(mol, _core(mol, frag, a))


@pytest.mark.parametrize("smiles,expected", [
    ("CC1(C)OC2(C)CCC1CC2OC1OC(CO)C(O)C(O)C1O", True),                      # d418: a sugar with no configuration
    ("OC[C@H]1O[C@@H](OC2CCCCC2)[C@H](O)[C@@H](O)[C@@H]1O", False),        # a glucoside, every centre specified
    ("CC1CCCC(CC(=O)O)O1", False),                                          # an oxane, not a sugar pattern
])
def test_unspecified_carbohydrate(smiles, expected):
    assert srp.unspecified_carbohydrate(Chem.MolFromSmiles(smiles)) is expected


def test_unspecified_carbohydrate_leaves_the_cip_labels_of_the_molecule_alone():
    """The guard reads the stereo centres of the molecule it is given and gives them back as they were: it ran at every
    wider tier with the strict producer, and RDKit's new CIP labeller (``useLegacyImplementation=False``) overwrote the
    engine's '_CIPCode' of a phosphorus centre (S to R), so a name cited the other P configuration, failed its round trip,
    and a ring-prefix route of the witness lost to a name that cites the pyrimidinedione as a prefix."""
    from orthonym.perception.stereo import assign_stereochemistry

    mol = Chem.MolFromSmiles("Cc1cn([C@H]2CC(O[P@]3O[C@](C)(c4ccccc4)[C@@H]4CCCN43)[C@@H](CO)O2)c(=O)[nH]c1=O")
    assign_stereochemistry(mol)
    before = {a.GetIdx(): a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")}
    assert before, "the engine labels the centres"
    srp.unspecified_carbohydrate(mol)
    assert {a.GetIdx(): a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")} == before


@pytest.mark.parametrize("smiles,parent,attach", [
    ("CC1=CCC(C(C)CC=O)CC1", 5, 4),              # a cyclohexene
    ("OC(=O)CC1CCCCC1=O", 3, 4),                 # a cyclohexanone
    ("CC1CCCC(CC(=O)O)O1", 6, 5),                # an oxane with no ring carbon carrying O/N
])
def test_a_ring_that_is_not_a_sugar_pattern_is_kept(smiles, parent, attach):
    mol, frag, a = _frag(smiles, parent, attach)
    assert not srp.carbohydrate_core(mol, _core(mol, frag, a))


# -- principal group in the fragment --------------------------------------------------

def test_a_principal_group_touching_the_fragment_only_at_the_attach_atom_is_fine():
    # 1-(1,4-dimethylcyclohex-3-en-1-yl)ethan-1-one (d1758): the acetyl is the parent's
    mol, frag, a = _frag("CC(=O)C1(C)CC=C(C)CC1", 1, 3)
    pg, matches = srp.principal_group(mol)
    assert pg == "ketone"
    assert not srp.principal_group_in_fragment(matches, frag, a)


def test_a_principal_group_inside_the_fragment_is_refused():
    mol, frag, a = _frag("CC(C)C1CCCCC1C(=O)O", 1, 3)
    pg, matches = srp.principal_group(mol)
    assert pg == "carboxylic_acid"
    assert srp.principal_group_in_fragment(matches, frag, a)


# -- a ring skeleton repeated outside the fragment, no principal group ---------------------------

B854 = "O=[N+]([O-])c1cnc2cc(Sc3ccc4c([N+](=O)[O-])c(Cl)cnc4c3)ccc2c1Cl"


@pytest.mark.parametrize("parent,attach", [(9, 8), (9, 10)])
def test_the_quinoline_sulfide_twin_is_refused(parent, attach):
    # BB ' General methodology' (:19420) leaves the parent to / here; the BB PIN
    # is 3-chloro-7-[(4-chloro-3-nitroquinolin-7-yl)sulfanyl]-4-nitroquinoline (:22088)
    mol, frag, a = _frag(B854, parent, attach)
    assert srp.principal_group(mol)[0] is None
    assert srp.skeleton_twin(mol, frag, a)


def test_different_ring_skeletons_are_not_twins():
    mol, frag, a = _frag("CC1=CCC(C(C)CC=O)CC1", 5, 4)
    assert not srp.skeleton_twin(mol, frag, a)


# -- the fragment must not be the parent ring, ----------------------------

@pytest.mark.parametrize("parent,attach,expected", [
    (4, 3, True),       # phenyl on 1H-benzimidazole: the benzimidazole (N, two rings) is the parent
    (3, 4, False),      # benzimidazol-2-yl on benzene: the fragment is the senior ring, so it is the parent
])
def test_parent_ring_outside_a_molecule_without_a_principal_group(parent, attach, expected):
    mol, frag, a = _frag("c1ccc(-c2nc3ccccc3[nH]2)cc1", parent, attach)
    assert srp.principal_group(mol)[0] is None
    assert srp.parent_ring_outside(mol, frag) is expected


def test_the_only_ring_system_inside_the_fragment_is_not_a_prefix_of_a_principal_group_free_molecule():
    mol, frag, a = _frag("CC(C)=CC1CCCCC1", 3, 4)
    assert srp.principal_group(mol)[0] is None
    assert not srp.parent_ring_outside(mol, frag)


@pytest.mark.parametrize("smiles,parent,attach,expected", [
    # (the Blue Book): the senior ring bonded to the amine nitrogen is the parent
    ("CN1CCC(CC1)N(C)c1ccc2[nH]cnc2c1", 7, 9, False),     # benzimidazol-5-yl beside a piperidine: parent
    ("CN1CCC(CC1)N(C)c1ccc2[nH]cnc2c1", 7, 4, True),      # piperidin-4-yl beside the benzimidazole: prefix
    ("CN(c1ccccc1)C1CCNCC1", 1, 2, True),                 # phenyl beside piperidine
    ("CN(c1ccccc1)C1CCNCC1", 1, 8, False),                # piperidin-4-yl beside benzene: parent
    ("CC(=O)Nc1ccccc1", 3, 4, True),                      # not an amine: the amide's acyl part is the parent
])
def test_amine_parent_is_the_senior_ring(smiles, parent, attach, expected):
    mol, frag, a = _frag(smiles, parent, attach)
    pg, matches = srp.principal_group(mol)
    assert srp.amine_parent_ok(mol, frag, a, pg, matches) is expected


@pytest.mark.parametrize("smiles,parent,attach,expected", [
    ("O=C(Nc1ccccc1)N1CCc2ccccc2C1", 1, 9, True),          # a urea carbon on the ring nitrogen: ring-carboxamide
    ("CC(=O)N1CCc2ccccc2C1", 1, 3, False),                 # an amide of a carbon acid: named like a ketone
])
def test_acylated_ring_nitrogen(smiles, parent, attach, expected):
    mol, frag, a = _frag(smiles, parent, attach)
    assert srp.acylated_ring_nitrogen(mol, frag, a) is expected


# -- a ring oxo needs a senior group outside the fragment ----------------------

@pytest.mark.parametrize("smiles,parent,attach,expected", [
    ("O=C(O)CC[C@@H]1NC=NC1=O", 4, 5, True),               # d713: an acid outside
    ("CCC(=O)C1CCCCPC1=O", 2, 4, False),                   # b1957: the ketone is class 16, like the ring oxo
    ("O=C1CC(C2CCC(=O)O2)Oc2ccccc21", 3, 4, False),        # a lactone beside a ring ketone,:29628)
    ("CC1=CC(=O)OC1CC(=O)[O-]", 7, 6, True),               # d1640: a carboxylate outside, not an ester
    ("CC1=CCC(C(C)CC=O)CC1", 5, 4, True),                  # no oxo on the ring: nothing to decide
])
def test_oxo_class_collision(smiles, parent, attach, expected):
    mol, frag, a = _frag(smiles, parent, attach)
    assert srp.oxo_class_ok(mol, frag, _core(mol, frag, a)) is expected


# -- a von Baeyer core is admitted only where no fusion name is preferred,:23831) ----

@pytest.mark.parametrize("smiles,parent,attach,expected", [
    ("OC(=O)CC1CC(O)C2C1C1CCC2C1", 3, 4, False),            # two five-membered rings share a bond
    ("OC(=O)CC1C(O)C2C=CC1C2", 3, 4, False),                # a ring double bond: (c) before (e)
    ("C=C1CC[C@H]2C[C@@H]1[C@@]2(C)CC/C=C(\\C)C(=O)O", 9, 7, True),   # bicyclo[3.1.1]heptane
    ("[NH3+][C@@H](C[C@@H]1CCC(=O)[C@@H]2O[C@H]12)C(=O)[O-]", 2, 3, True),   # 7-oxabicyclo[4.1.0]heptane
])
def test_cage_core_rules(smiles, parent, attach, expected):
    mol, frag, a = _frag(smiles, parent, attach)
    assert srp.cage_core_ok(mol, _core(mol, frag, a)) is expected


# -- a ring heteroatom of nonstandard bonding number, an aromatic ring atom with an exocyclic double bond --

@pytest.mark.parametrize("smiles,parent,attach,expected", [
    ("CC(=O)NC1CCS(=O)(=O)C1", 3, 4, True),       # a ring sulfone: the lambda convention,:2758)
    ("CC(=O)NC1CCSC1", 3, 4, False),               # the sulfide
])
def test_hypervalent_core(smiles, parent, attach, expected):
    mol, frag, a = _frag(smiles, parent, attach)
    assert srp.hypervalent_core(mol, _core(mol, frag, a)) is expected


@pytest.mark.parametrize("smiles,parent,attach,expected", [
    ("CC(C(=O)O)n1c(-c2cccs2)n[nH]c1=S", 1, 5, True),     # a triazole-thione: hydro prefixes needed
    ("CC(C)CC(=O)N1CCC(=O)C1", 4, 6, False),                # a saturated lactam keeps its oxo
    ("CC1=CC(=O)CC(C)(C)C1CCC(C)CC(=O)O", 10, 9, False),   # a carbocycle: ene locants carry the unsaturation
])
def test_aromatic_ring_atom_with_an_exocyclic_double_bond(smiles, parent, attach, expected):
    mol, frag, a = _frag(smiles, parent, attach)
    assert srp.aromatic_with_exocyclic_double_bond(mol, _core(mol, frag, a)) is expected


# -- (g): lowest locants for the prefix cited first (the Blue Book) -------------------

def _renumbered(smiles, parent, attach, named_by_symbol):
    """The policy's numbering of the ring-yl at ``attach`` for decorations ``{element symbol: name}``."""
    from orthonym.assembly import substituent_enumerator as se
    mol, frag, a = _frag(smiles, parent, attach)
    core = _core(mol, frag, a)
    named = [(ring_atom, named_by_symbol[nb.GetSymbol()])
             for ring_atom in sorted(core) for nb in mol.GetAtomWithIdx(ring_atom).GetNeighbors()
             if nb.GetIdx() not in core and nb.GetSymbol() in named_by_symbol]
    pos = se._monocycle_position_map(mol, core, a, [ra for ra, _ in named], mol.GetRingInfo())
    return srp.StrictRingPolicy(frag).renumber(mol, core, a, pos, named), named


@pytest.mark.parametrize("smiles,parent,attach", [
    ("OC(=O)CC1C(Br)=CCC=C1Cl", 3, 4),
    ("OC(=O)CC1C(Cl)=CCC=C1Br", 3, 4),
    ("ClC1=CCC=C(Br)C1CC(=O)O", 8, 7),
])
def test_the_prefix_cited_first_takes_the_lower_locant(smiles, parent, attach):
    # cyclohexa-2,5-dien-1-yl: the ene set {2,5} and the prefix set {2,6} tie both ways; 'bromo' is
    # cited before 'chloro', so it takes locant 2 whatever the order of the input atoms
    numbering, named = _renumbered(smiles, parent, attach, {"Br": "bromo", "Cl": "chloro"})
    assert numbering is not None
    by_name = {name: numbering[ra] for ra, name in named}
    assert by_name == {"bromo": 2, "chloro": 6}, by_name


def test_a_tie_that_would_move_the_descriptors_is_not_resolved():
    # 2,6-dimethylcyclohexyl with three stereocentres: the two numberings differ in the locants of the
    # descriptors, which (g) does not decide
    numbering, _ = _renumbered("OC(=O)C[C@H]1[C@H](C)CCC[C@@H]1C", 3, 4, {"C": "methyl"})
    assert numbering is None


# -- the hook is inert outside the promotion re-run ----------------------------------------------

def test_outside_the_rerun_the_hook_is_the_old_call_and_never_runs_the_producer(monkeypatch):
    monkeypatch.setattr(srp, "_KINDS", frozenset({"stem"}))

    def boom(*args, **kwargs):
        raise AssertionError("the strict producer ran outside the promotion re-run")
    monkeypatch.setattr(srp, "strict_ring_substituent_name", boom)
    mol, frag, a = _frag("CC1=CCC(C(C)CC=O)CC1", 5, 4)
    assert not pin_promotion_var.get()
    assert srp.pin_tier_ring_prefix(mol, sorted(frag), a, lambda: "4-methylcyclohex-3-en-1-yl") is None


def test_inside_the_rerun_the_breadth_producer_still_runs_when_the_strict_one_declines(monkeypatch):
    monkeypatch.setattr(srp, "_KINDS", frozenset({"stem"}))
    monkeypatch.setattr(srp, "strict_ring_substituent_name", lambda *a, **k: None)
    mol, frag, a = _frag("CC1=CCC(C(C)CC=O)CC1", 5, 4)
    tok = pin_promotion_var.set(True)
    try:
        assert srp.pin_tier_ring_prefix(
            mol, sorted(frag), a, lambda: "4-methylcyclohex-3-en-1-yl") == "4-methylcyclohex-3-en-1-yl"
    finally:
        pin_promotion_var.reset(tok)


# -- fix a performance pass (review): added hydrogen of an N-yl, tied amine parents, parent-naming classes -------------

@pytest.mark.parametrize("smiles,attach,tail,expected", [
    ("CC(=O)N1CCCc2ccccc21", 3, "1,2,3,4-tetrahydroquinolin-1-yl", True),               # the Blue Book
    ("CC(=O)N1CCCc2ccccc21", 3, "3,4-dihydroquinolin-1(2H)-yl", False),
    ("CC(=O)N1CC2CCCN2CC1", 3, "octahydropyrrolo[1,2-a]pyrazin-2-yl", True),            #,:16070
    ("CC(=O)N1CC2CCCN2CC1", 3, "hexahydropyrrolo[1,2-a]pyrazin-2(1H)-yl", False),
    ("CC(=O)N1Cc2ccccc2C1", 3, "2,3-dihydro-1H-isoindol-2-yl", False),                  #:17380, indicated hydrogen
    ("CC(=O)N1CCCCC1", 3, "piperidin-1-yl", False),                                      # no hydro prefix
    ("CC(=O)C1CCCc2ccccc21", 3, "1,2,3,4-tetrahydronaphthalen-1-yl", False),             # a carbon
])
def test_n_yl_without_added_hydrogen(smiles, attach, tail, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert srp.n_yl_without_added_hydrogen(mol, attach, tail) is expected


@pytest.mark.parametrize("smiles,expected", [
    # (:21698): two pyridines on one amine nitrogen tie on the ring-system score
    ("CN1C(=C(C(=N1)Br)[N+](=O)[O-])N2CCC(CC2)N(C3=CN=C(C=C3)OC(F)F)C4=C(C=CN=C4)OC", True),
    ("CN(c1ccccc1)c1ccccc1", True),
    ("CN(c1ccccc1)C1CCNCC1", False),                                  # benzene against piperidine
    ("CN1CCC(CC1)N(C)c1ccc2[nH]cnc2c1", False),
    ("CC(=O)Nc1ccccc1", False),                                       # not an amine
])
def test_tied_amine_parent(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    pg, matches = srp.principal_group(mol)
    assert srp.tied_amine_parent(mol, pg, matches) is expected


def test_the_classes_that_name_the_parent_are_not_substituent_only():
    # the Blue Book '{7-[(anthracen-2-yl)diazenyl]naphthalen-2-yl}(phenyl)diazene (PIN)'
    azo = srp.functional_group_classes(Chem.MolFromSmiles("c1ccc(N=Nc2ccc3ccc(N=Nc4ccc5cc6ccccc6cc5c4)cc3c2)cc1"))
    assert azo & srp._PARENT_NAMING and not azo <= srp._SUBSTITUENT_ONLY
    assert not srp._PARENT_NAMING & srp._SUBSTITUENT_ONLY
    ether = srp.functional_group_classes(Chem.MolFromSmiles("c1ccccc1Oc1ccccn1"))
    assert ether <= srp._SUBSTITUENT_ONLY and not ether & srp._PARENT_NAMING
