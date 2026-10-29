"""A radical centre that is also an ionic centre: the one-atom parent of a radical ion.

``### **** RADICAL IONS DERIVED FROM PARENT HYDRIDES`` (the Blue Book), sentence:43422: "A radical ion
derived formally by the removal of one or more hydrogen atoms from a single skeleton atom or from different skeletal
atoms of an ionic or zwitterionic parent hydride is named by adding to its name the suffixes 'yl' or 'ylidene' with
appropriate multiplying prefixes before 'yl' or 'ylidene', with elision of the final letter 'e' of the name of the ionic
parent hydride." The book's examples are '(CH3)3B•- trimethylboranuidyl (PIN)' (:43430) and, under ````
(:43463), '(N,N-diethylethanaminiumyl)boranuidyl (PIN)'. The suffixes of two or three hydrogens removed from one atom are
'-ylidene' and '-ylidyne' (``### **** Specific method and retained names``,:40474).

The floor builds nothing for a charged radical centre; ``assembly/radical_ion.py`` is an offer of the radical last resort
(``assembly/radical_offer.py``), made after every name of the floor's routes, and its name ships only when OPSIN ``-r``
reads it back to the input's full InChIKey and radical graph. The suffix of the ionic centre follows from its charge and
from the bonding number of the atom beside the hydrogens of its hydride: 'uide' for an anion that holds one more bond than
its hydride, 'ide' for one that holds one less, 'ium' for a cation that holds one more, 'ylium' for one that holds one less.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.radical_ion import radical_ion_names
from orthonym.rules.radicals import ionic_parent_hydride
from orthonym.validation.radical_identity import radical_identity_verdict
from orthonym.validation.read_back_policy import radical_last_resort, reading_for
from tests.support.rt_assert import _independent_parse, name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


def _assert_named(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] in ((expected,) if isinstance(expected, str) else expected), res
    assert res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res
    parsed = _independent_parse(res["name"], smiles)
    assert parsed is not None and radical_identity_verdict(smiles, parsed) == "ok", (res, parsed)


ROWS = [
    # an anion that holds one more bond than its hydride: 'uide'; the three hydrogens removed from one atom: 'ylidyne'
    ("[B-]CC", "ethylboranuidylidyne"),
    # two hydrogens removed from the atom: 'ylidene'; the prefixes in the order of
    ("[B-](CN(C)C)F", "[(dimethylamino)methyl]fluoroboranuidylidene"),
    # one hydrogen removed: 'yl', the final 'e' of the ionic hydride elided before it. OPSIN 2.9.0 places the free valence
    # on a halogen hydride only when its bonding number is cited ('pentyl-λ1-iodanuidyl'); a later OPSIN reads the first
    ("CCCCC[I-]", ("pentyliodanuidyl", "pentyl-λ1-iodanuidyl")),
    # a cation that holds one bond fewer than its hydride: 'ylium'
    ("C[Si+]C", "dimethylsilyliumyl"),
    # the zwitterion of: the ammonium is a substituent of the boranuide
    ("[B-][NH+](C(C)C)C(C)C", "[di(propan-2-yl)azaniumyl]boranuidylidyne"),
    ("CC[N+](CC)(CC)[B-]", "(triethylazaniumyl)boranuidylidyne"),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_a_charged_radical_centre_is_a_radical_ion_of_a_one_atom_parent(smiles, expected):
    _assert_named(smiles, expected)


def test_the_diyl_and_triyl_spellings_follow_the_ylidene_and_ylidyne_ones():
    mol = Chem.MolFromSmiles("[B-]CC")
    with radical_last_resort(), reading_for(mol):
        assert list(radical_ion_names(mol)) == ["ethylboranuidylidyne", "ethylboranuidetriyl"]


@pytest.mark.parametrize("smiles,atom,expected", [
    ("[B-]CC", 0, "boranuide"),        # B holds one bond and three radical electrons: 4 = 3 + 1
    ("CCCCC[I-]", 5, "iodanuide"),     # I holds one bond and one radical electron: 2 = 1 + 1
    ("C[Si+]C", 1, "silylium"),        # Si holds two bonds and one radical electron: 3 = 4 - 1
    ("[NH-]", 0, "azanide"),           # N holds one hydrogen and one radical electron: 2 = 3 - 1
    ("[CH2+]", 0, "methylium"),        # the book's 'methyliumyl (PIN)',:43443)
    ("[NH3+]", 0, "azanium"),
])
def test_the_ionic_suffix_follows_the_charge_and_the_bonding_number(smiles, atom, expected):
    """One home for the ionic parent hydride of an atom: ``rules.radicals.ionic_parent_hydride``, which the producer of the PIN
    tiers (``_name_mononuclear_radical_ion``) names its radical ions with as well."""
    mol = Chem.MolFromSmiles(smiles)
    atom = mol.GetAtomWithIdx(atom)
    bonding = sum(round(b.GetBondTypeAsDouble()) for b in atom.GetBonds()) + atom.GetTotalNumHs() + atom.GetNumRadicalElectrons()
    assert ionic_parent_hydride(atom, bonding) == expected


def test_an_empty_orbital_cation_is_not_a_radical_ion():
    """RDKit counts two radical electrons on the empty orbital of an '-ylium' cation (C[O+], 'methoxylium'); the producer of
    the PIN tiers leaves those alone and so does the offer (user decision Q2 = A)."""
    mol = Chem.MolFromSmiles("C[O+]")
    with radical_last_resort(), reading_for(mol):
        assert list(radical_ion_names(mol)) == []


@pytest.mark.parametrize("smiles", [
    "[B-]1[P+](c2ccccc2S1)(c1ccccc1)c1ccccc1",   # the centre is in a ring: no one-atom parent
    "CC[N+](C)(C)C",                              # no radical
    "[B-]C[B-]",                                  # two centres
])
def test_a_molecule_that_is_not_a_one_atom_radical_ion_has_no_such_name(smiles):
    mol = Chem.MolFromSmiles(smiles)
    with radical_last_resort(), reading_for(mol):
        assert list(radical_ion_names(mol)) == []
