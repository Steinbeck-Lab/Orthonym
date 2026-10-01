"""The whole bridged fused name from ``rules/bridged_fused_pin.build``: the numbering
criteria in order (a), (b):14225/:14231; (b)-(g):3246-:3307), the
suffixes the slice spells, the stereodescriptors on the bridged locants, and the cases it
declines. Every name here was read back by OPSIN 2.9.0 to the input's full InChIKey
(S1 plan ledger), except the ethanoanthracene diacid (OPSIN cannot place the 9,10
descriptors)."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import build


def _name(smiles):
    res = build(Chem.MolFromSmiles(smiles))
    return res[0] if res else None


@pytest.mark.parametrize("smiles,name,rule", [
    ("CC1=CC2OC1c1ccccc12", "2-methyl-1,4-dihydro-1,4-epoxynaphthalene", "P-25.4.3.3 (a): 1,4 not 5,8"),
    ("C1CC2CC1c1c2c2ccc1o2", "5,6,7,8-tetrahydro-1,4-epoxy-5,8-methanonaphthalene", "P-25.4.3.3 (b)"),
    ("C1CCc2c3ccc(C3)c2C1", "5,6,7,8-tetrahydro-1,4-methanonaphthalene", "(a) before hydro (e)"),
    ("CC1(C)C2=CC=CC(C)(C)C23C=CC1C3",
     "1,1,5,5-tetramethyl-1,5-dihydro-2H-2,4a-methanonaphthalene", "P-14.4 (b): 2H not 5H"),
    ("OC1CC2CC1c1ccccc12", "1,2,3,4-tetrahydro-1,4-methanonaphthalen-2-ol", "P-14.4 (c)"),
    ("Cc1cc(C)c2c(c1)C1CCC2C1", "5,7-dimethyl-1,2,3,4-tetrahydro-1,4-methanonaphthalene", "P-14.4 (f)"),
    ("Cc1cc2c(cc1Cl)C1CCC2C1", "6-chloro-7-methyl-1,2,3,4-tetrahydro-1,4-methanonaphthalene", "P-14.4 (g)"),
    ("OC(=O)C1CC2c3ccccc3C1c1ccccc12", "9,10-dihydro-9,10-ethanoanthracene-11-carboxylic acid",
     "P-25.4.4: C11 next to C10"),
    ("CC1C2CCC1c1c2c2ccc1o2", "9-methyl-5,6,7,8-tetrahydro-1,4-epoxy-5,8-methanonaphthalene",
     "P-25.4.5.2: the methano on 5,8 is C9"),
    ("CC12CCC(c3ccccc31)c1ccccc12", "9-methyl-9,10-dihydro-9,10-ethanoanthracene", "P-14.4 (f): 9 not 10"),
    ("C1=CC2CCC1c1ccccc12", "1,4-dihydro-1,4-ethanonaphthalene", "P-25.4.3.4.2 (j)"),
    ("CC1CC2CCC1c1ccccc12", "2-methyl-1,2,3,4-tetrahydro-1,4-ethanonaphthalene",
     "one parent read two ways: P-14.4 (f) 2 not 9"),
    ("C1CC23CCCCC2(CC1)CC3", "octahydro-4a,8a-ethanonaphthalene", "P-31.2.3.3.2"),
    ("C=1C2C=CC3(C=CC=CC13)CC2", "2H-2,4a-ethanonaphthalene",
     "a fusion-atom bridge; P-58.2.1.2: indicated hydrogen at the lowest nonfusion atom"),
    ("C1=C2CCC3C=CC=CC13C2", "4,4a-dihydro-3H-2,8a-methanonaphthalene",
     "P-31.2.2: 3H (nonfusion, lowest) before the hydro prefixes"),
    ("C1=C2C=CC3CCC=CC13C2", "4a,5-dihydro-6H-2,8a-methanonaphthalene",
     "P-58.2.1.2 (:24685): 6H (nonfusion) over the lower fusion locant 4a"),
    ("C[C@@]12CCC[C@@]3(C)[C@@H](C1)[C@@](O)(CO)CC[C@@]23C",
     "(1R,2R,4aS,5R,8aS)-2-(hydroxymethyl)-4a,5,8a-trimethyldecahydro-1,5-methanonaphthalen-2-ol",
     "P-14.3.5 4a,5,8a; -ol on the ring, hydroxymethyl prefix"),
])
def test_numbering_rules(smiles, name, rule):
    assert _name(smiles) == name, rule


@pytest.mark.parametrize("smiles,name", [   # suffixes on the benzo ring and on a ring after a bridge suffix
    ("Oc1ccc2c(c1)C1CCC2C1", "1,2,3,4-tetrahydro-1,4-methanonaphthalen-6-ol"),
    ("OC(=O)c1ccc2c(c1)C1CCC2C1", "1,2,3,4-tetrahydro-1,4-methanonaphthalene-6-carboxylic acid"),
    ("Nc1ccc2c(c1)C1CCC2C1", "1,2,3,4-tetrahydro-1,4-methanonaphthalen-6-amine"),
    ("Oc1ccc2c(c1O)C1CCC2C1", "1,2,3,4-tetrahydro-1,4-methanonaphthalene-5,6-diol"),
    ("OC(=O)C1CC2c3ccc(Cl)cc3C1c1ccccc12", "3-chloro-9,10-dihydro-9,10-ethanoanthracene-11-carboxylic acid"),
])
def test_suffixes(smiles, name):
    assert _name(smiles) == name


@pytest.mark.parametrize("smiles,name", [   # Review Focus 4: descriptors on bridge atoms and bridgeheads
    ("O[C@@H]1C[C@@H]2C=C[C@H]1c1ccccc12", "(1R,4S,9R)-1,4-dihydro-1,4-ethanonaphthalen-9-ol"),
    ("CC1=C[C@H]2C[C@@H]1c1ccccc12", "(1S,4R)-2-methyl-1,4-dihydro-1,4-methanonaphthalene"),
    ("O[C@@H]1C[C@H]2C[C@@H]1c1ccccc12", "(1R,2R,4R)-1,2,3,4-tetrahydro-1,4-methanonaphthalen-2-ol"),
])
def test_stereodescriptors_sit_on_the_bridged_locants(smiles, name):
    assert _name(smiles) == name


def test_a_bridge_atom_descriptor_of_an_ethanoanthracene_sits_on_its_p25_4_4_locant():
    # C11 is the bridge atom next to C10:14403); the bridgeheads 9 and 10 are
    # left unspecified in the input, so only C11 carries a descriptor. The label is
    # RDKit's new CIP labeler's, at the builder's locant.
    from rdkit.Chem import rdCIPLabeler
    smiles = "C[C@@H]1CC2c3ccccc3C1c1ccccc12"
    name, _, a2l, _ = build(Chem.MolFromSmiles(smiles))
    mol = Chem.MolFromSmiles(smiles)
    rdCIPLabeler.AssignCIPLabels(mol)
    (centre,) = [a for a in mol.GetAtoms() if a.HasProp("_CIPCode")]
    assert a2l[centre.GetIdx()] == 11
    assert name == f"(11{centre.GetProp('_CIPCode')})-11-methyl-9,10-dihydro-9,10-ethanoanthracene"


@pytest.mark.parametrize("smiles", [
    "OC(=O)[C@@H]1[C@@H](C(O)=O)[C@H]2c3ccccc3[C@@H]1c1ccccc21",   # 9,10 decided by Rules 4/5
    "C[C@@H]1C[C@H]2c3ccccc3[C@@H]1c1ccccc12",
    "O=C(O)[C@@H]1C[C@H]2c3ccccc3[C@@H]1c1ccccc12",
])
def test_descriptors_decided_by_sequence_rules_4_and_5_are_declined(smiles):
    # (:45443-:45445): the two benzo ligands of C9/C10 have one constitution, so
    # only Sequence Rules 4/5 rank them; OPSIN 2.9.0 cannot place such a descriptor
    # ("Could not find atom that... 9R appeared to be referring to"). Such a name is
    # declined, never shipped as pin_unverified at the default tier.
    assert _name(smiles) is None


@pytest.mark.parametrize("smiles", [
    "O=C1CC2CC1c1ccccc12",               # ketone: needs added hydrogen (slice S3)
    "O=CC1CC2CC1c1ccccc12",              # aldehyde suffix: not spelled in S1
    "COC(=O)C1CC2CC1c1ccccc12",          # ester: functional class name
    "OCC(O)C1CC2CC1c1ccccc12",           # two -OH on the chain: the chain is the parent
    "C[C@H](Cl)C1=CC2CC1c1ccccc12",      # a stereocentre outside the ring system
    "C12C=CC(C1)c1ccc3ccccc3c12",        # a phenanthrene residual (slice S2)
    "C12=CC=C(C3=C4C=5C6=CC=C(C5C(=C13)C4)C6)C2",   # three bridges (outside S1)
])
def test_declined(smiles):
    assert _name(smiles) is None


def test_a_tie_between_different_bridges_declines(monkeypatch):
    # Review Focus 5: two readings tied on every criterion of that cite
    # different bridges -> fail closed (the von Baeyer name stays the fallback)
    from orthonym.rules import bridged_fused_pin as pkg
    real = pkg._options

    def two_readings(mol, system):
        opts = real(mol, system)
        sp, bps, state, nums = opts[0]
        other = [prefixes_like(bp) for bp in bps]
        return opts + [(sp, other, state, nums)]

    def prefixes_like(bp):
        return type(bp)("propano" if bp.name != "propano" else "butano", bp.is_pin_form)

    monkeypatch.setattr(pkg, "_options", two_readings)
    assert build(Chem.MolFromSmiles("C1CC2CC1c1ccccc12")) is None


def test_a_general_bridge_prefix_is_recorded_as_not_a_pin():
    # (:14097): the name is built, and 'epithio' is recorded as a part that
    # is never a PIN, so name_tiered labels any name carrying it below pin_verified
    from orthonym.metrics import provenance as pv
    pv.clear_provenance()
    name = _name("CC1=CC2SC1c1ccccc12")
    assert name == "2-methyl-1,4-dihydro-1,4-epithionaphthalene"
    assert pv.name_carries_non_pin_part(pv.get_provenance(), name)


def test_the_prefix_speller_puts_a_letter_locant_after_its_number():
    # (:3193): "4a... placed immediately after the corresponding numeric locant";
    # polycyclics._partial_sat_substituent_prefix sorted every int before every letter locant
    from orthonym.rules.polycyclics import _partial_sat_substituent_prefix
    mol = Chem.MolFromSmiles("CC1CCC(C)C1")
    assert _partial_sat_substituent_prefix(mol, {1: 5, 2: 6, 3: 7, 4: "4a", 6: 8}) == "4a,5-dimethyl-"


@pytest.mark.parametrize("smiles", [
    "c1ccc(nc1)C1CC2CC1c1ccccc12",           # (a): the pyridine ring system is senior
    "c1ccc2ncc(cc2c1)C1CC2CC1c1ccccc12",     # a heterocyclic ring system elsewhere
])
def test_declined_when_another_ring_system_may_be_senior(smiles):
    assert _name(smiles) is None
