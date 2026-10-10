"""Leads item N8b: the chain of a chain parent with a ring on one of two equal chains.

 'Maximum number of substituents cited as prefixes' (the Blue Book): "The
preferred IUPAC name is based on the senior parent structure that has the maximum number of
substituents cited as prefixes (other than hydro/dehydro) to the parent structure"; its example
(4),:21624, 'N,N,2-trimethyl-3-{...}propanamide (PIN)', counts the ring substituent of the
parent chain; (:21698) and (:21791) then compare locants and order of citation.

``Orthonym._classify`` handed parent selection the chain of ``find_principal_chain`` with the
ring atoms left out of the prefix count, so two chains of one length bearing one prefix each
tied and the input's atom order decided: 'OC(=O)C(C)Cc1ccccc1' was '2-methyl-3-phenylpropanoic
acid' (the PIN) for 15 random orders of 20 and '2-benzylpropanoic acid' for the other 5. The
chain is now the one the selector makes senior (``chains.p45_principal_chain``) for the
principal groups whose producer builds on the chain it is given (``chains.GIVEN_CHAIN_GROUPS``)
and for the caller's molecule; every other group keeps the chain it had.

Every name is read back by OPSIN's own StdInChIKey against RDKit's key of the input.
"""
import random
import re

import pytest
from rdkit import Chem

from orthonym import Orthonym, namer
from orthonym.perception import chains
from orthonym.perception.chains import find_principal_chain
from tests.unit.namer.leads_l1_support import opsin_key, rdkit_key

pytestmark = pytest.mark.opsin_gate

# (SMILES, PIN, the non-PIN spelling an atom order used to give) -- each PIN read back by OPSIN
CASES = [
    ("OC(=O)C(C)Cc1ccccc1", "2-methyl-3-phenylpropanoic acid", "2-benzylpropanoic acid"),
    ("NC(=O)C(C)Cc1ccccc1", "2-methyl-3-phenylpropanamide", "2-benzylpropanamide"),
    ("CNC(=O)C(C)Cc1ccccc1", "N,2-dimethyl-3-phenylpropanamide",
     "2-benzyl-N-methylpropanamide"),
    ("N#CC(C)Cc1ccccc1", "2-methyl-3-phenylpropanenitrile", "2-benzylpropanenitrile"),
    ("O=CC(C)Cc1ccccc1", "2-methyl-3-phenylpropanal", "2-benzylpropanal"),
    ("OCC(C)Cc1ccccc1", "2-methyl-3-phenylpropan-1-ol", "2-benzylpropan-1-ol"),
    ("CC(O)C(C)Cc1ccccc1", "3-methyl-4-phenylbutan-2-ol", "3-benzylbutan-2-ol"),
    ("NCC(C)Cc1ccccc1", "2-methyl-3-phenylpropan-1-amine", "2-benzylpropan-1-amine"),
    ("CC(Cc1ccccc1)C(=O)C", "3-methyl-4-phenylbutan-2-one", "3-benzylbutan-2-one"),
    ("CC(Cc1ccccc1)C(Cl)=O", "2-methyl-3-phenylpropanoyl chloride",
     "2-benzylpropanoyl chloride"),
    ("CC(Cc1ccccc1)C=NC", "N,2-dimethyl-3-phenylpropan-1-imine",
     "2-benzyl-N-methylpropan-1-imine"),
    ("OC(=O)C(C)Cc1ccccn1", "2-methyl-3-(pyridin-2-yl)propanoic acid",
     "2-[(pyridin-2-yl)methyl]propanoic acid"),
    ("OC(=O)C(C)CC1CC1", "3-cyclopropyl-2-methylpropanoic acid",
     "2-(cyclopropylmethyl)propanoic acid"),
    # the leads item: the two molecules and the nitrile of its report
    ("N(C(=O)C(C)Cc1ccc(OC)cc1)C(C)CO",
     "N-(1-hydroxypropan-2-yl)-3-(4-methoxyphenyl)-2-methylpropanamide",
     "N-(1-hydroxypropan-2-yl)-2-[(4-methoxyphenyl)methyl]propanamide"),
    ("N#CC(CO)Cc1cccc([N+](=O)[O-])c1", "2-(hydroxymethyl)-3-(3-nitrophenyl)propanenitrile",
     "3-hydroxy-2-[(3-nitrophenyl)methyl]propanenitrile"),
]


def _strict_engine():
    """The strict PIN path with the default tier's emission rule off, so the label of every
    name is read (a decline would hide it)."""
    return Orthonym(style="pin")


def _random_orders(smiles, n, seed=11):
    mol = Chem.MolFromSmiles(smiles)
    rng_state = random.getstate()
    random.seed(seed)
    seen, out, tries = set(), [], 0
    while len(out) < n and tries < 400:
        tries += 1
        s = Chem.MolToSmiles(mol, doRandom=True)
        if s not in seen:
            seen.add(s)
            out.append(s)
    random.setstate(rng_state)
    return out


@pytest.fixture
def strict():
    token = namer._DEFAULT_TIER_POLICY_OFF.set(True)
    try:
        yield _strict_engine()
    finally:
        namer._DEFAULT_TIER_POLICY_OFF.reset(token)


@pytest.mark.parametrize("smiles, pin, old", CASES)
def test_pin_is_read_back_by_opsin_and_old_spelling_is_the_same_molecule(smiles, pin, old):
    assert opsin_key(pin) == rdkit_key(smiles)
    assert opsin_key(old) == rdkit_key(smiles)          # the old spelling is valid, not the PIN


@pytest.mark.parametrize("smiles, pin, old", CASES)
def test_every_atom_order_gives_the_pin_labelled_pin_verified(smiles, pin, old, strict):
    orders = _random_orders(smiles, 12)
    assert len(orders) >= 8
    for order in orders:
        row = strict.name_tiered(order)
        assert (row["name"], row["tier"]) == (pin, "pin_verified"), order


@pytest.mark.parametrize("smiles, pin, old", CASES[:3])
def test_the_default_tier_emits_the_pin_for_the_leads_molecules(smiles, pin, old):
    for order in _random_orders(smiles, 6):
        row = Orthonym(style="pin").name_tiered(order)
        assert row["name"] == pin and row["tier"] == "pin_verified" and row["is_pin"] is True


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
@pytest.mark.parametrize("smiles, pin, old", CASES[:3] + CASES[-2:])
def test_the_wider_tiers_give_the_pin_too(smiles, pin, old, tier):
    from orthonym.cli import _emit_tier_flags
    engine = Orthonym(style="pin", **_emit_tier_flags(tier))
    for order in _random_orders(smiles, 6):
        row = engine.name_tiered(order)
        assert (row["name"], row["tier"]) == (pin, "pin_verified"), (tier, order)


def test_the_leads_ester_and_the_label_stay_honest():
    # An ester's producer walks the acid fragment itself (rules/esters.py:
    # _find_acid_principal_chain); ester is not in GIVEN_CHAIN_GROUPS, so it keeps the chain it
    # had and its labelling: a name on the chain without the senior prefixes is never
    # labelled pin_verified (given the senior chain it would be: the check would pass).
    assert "ester" not in chains.GIVEN_CHAIN_GROUPS
    token = namer._DEFAULT_TIER_POLICY_OFF.set(True)
    try:
        engine = _strict_engine()
        for smiles in ("COC(=O)C(C)Cc1ccccc1", "CC(Cc1ccccc1)C(=O)OCc1ccccc1",
                       "COC(=O)C(C)Cc1c(C)nn(-c2ccccc2Cl)c1C"):
            for order in _random_orders(smiles, 10):
                row = engine.name_tiered(order)
                if re.search(r"2-benzylpropanoate|\]methyl\}propanoate", row["name"]):
                    assert row["tier"] != "pin_verified", (order, row["name"])
    finally:
        namer._DEFAULT_TIER_POLICY_OFF.reset(token)


def test_a_fragment_named_inside_a_name_keeps_the_chain_it_had():
    # '2-hydroxyethyl 2-methyl-3-phenylpropanoate' names its acid as a nested name; the nested
    # acid must not take the senior chain while the ester keeps its own, or the nested string
    # is recorded as a non-PIN part of the ester name (7 of 10 orders lost pin_verified)
    token = namer._DEFAULT_TIER_POLICY_OFF.set(True)
    try:
        engine = _strict_engine()
        rows = [engine.name_tiered(o) for o in _random_orders("CC(Cc1ccccc1)C(=O)OCCO", 10)]
    finally:
        namer._DEFAULT_TIER_POLICY_OFF.reset(token)
    pins = [r for r in rows if r["name"] == "2-hydroxyethyl 2-methyl-3-phenylpropanoate"]
    assert pins and all(r["tier"] == "pin_verified" for r in pins)


def test_a_chain_without_the_principal_group_atoms_of_the_default_chain_is_not_taken():
    #: the chain holds the principal characteristic group before is reached. The
    # selector counts a secondary amide for any chain with a carbon that bears its nitrogen, so
    # the CH-CH2 chain of the nitrogen's other substituent ties with the acetyl chain and wins on
    # its prefixes; that chain holds no atom of the amide, and is not taken (the amide is the
    # parent, as it was: the lactone ring is not, 'Seniority of classes').
    smiles = "COc1cc([C@H](Cc2ccccc2)NC(C)=O)oc(=O)c1"
    amide = "N-[(1S)-1-(4-methoxy-2-oxo-2H-pyran-6-yl)-2-phenylethyl]acetamide"
    assert opsin_key(amide) == rdkit_key(smiles)
    from orthonym.cli import _emit_tier_flags
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert (row["name"], row["tier"]) == (amide, "systematic_verified"), row
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.seniority import get_principal_group
    mol = Chem.MolFromSmiles(smiles)
    fgs = detect_functional_groups(mol)
    pg, atoms = get_principal_group(mol, fgs)

    class F:  # the features the selector reads
        pass

    f = F()
    f.mol, f.functional_groups, f.principal_group, f.principal_group_atoms = mol, fgs, pg, atoms
    f.ring_systems = [set(r) for r in mol.GetRingInfo().AtomRings()]
    ring_atoms = {a for r in f.ring_systems for a in r}
    default = find_principal_chain(mol, fgs, pg, exclude_atoms=ring_atoms)
    assert pg == "secondary_amide"
    # the senior chain of the ring-prefix count is the other one...
    assert set(chains.principal_chain_with_ring_prefixes(f)) != set(default)
    #... and the selector keeps the chain it was given
    assert chains.p45_principal_chain(f, default) == default


def test_is_callers_molecule():
    # outside a public call every molecule is the caller's
    assert namer._is_callers_molecule(Chem.MolFromSmiles("CCO")) is True
    namer._CALLER_INPUT.smiles = "OC(=O)C(C)Cc1ccccc1"
    try:
        assert namer._is_callers_molecule(Chem.MolFromSmiles("c1ccccc1CC(C)C(=O)O")) is True
        assert namer._is_callers_molecule(Chem.MolFromSmiles("CC(C)C(=O)O")) is False
    finally:
        namer._CALLER_INPUT.smiles = None


def test_the_selector_returns_the_senior_chain_and_undecided_is_none():
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.seniority import get_principal_group
    mol = Chem.MolFromSmiles("OC(=O)C(C)Cc1ccccc1")
    fgs = detect_functional_groups(mol)
    pg, atoms = get_principal_group(mol, fgs)

    class F:  # the features the selector reads
        pass

    f = F()
    f.mol, f.functional_groups, f.principal_group = mol, fgs, pg
    f.ring_systems = [set(r) for r in mol.GetRingInfo().AtomRings()]
    senior = chains.principal_chain_with_ring_prefixes(f)
    # carboxyl carbon, the CH, then the CH2 that carries the phenyl: the chain
    assert [mol.GetAtomWithIdx(a).GetTotalNumHs() for a in senior] == [0, 1, 2]
    # kept on the features, keyed by group and ring atoms
    assert chains.principal_chain_with_ring_prefixes(f) is senior or \
        chains.principal_chain_with_ring_prefixes(f) == senior
    # a group that is not in the set keeps the chain it was given
    f.principal_group = "ester"
    assert chains.p45_principal_chain(f, [0, 1, 2]) == [0, 1, 2]
