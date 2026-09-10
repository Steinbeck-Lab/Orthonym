"""Bug B (audit 2026-09-03): the last-resort branch of
``_name_n_attached_substituent_fallback`` built its fragment from ``sub_set``,
which INCLUDES the attaching nitrogen, named that fragment as an amine
("N-methylaniline", "4-methylcyclohexan-1-amine") and then wrapped the result
in "amino" again -- one nitrogen counted twice. Every candidate it produced was
a different molecule; the OPSIN gate rejected them and the molecule abstained
where an earlier path had the right name (3-(N-methylanilino)propanenitrile).
The branch never ran in production only because ``Chem`` was unbound in
composer.py and the NameError was swallowed.

Design of this function (its two earlier branches): name the CARBON part that
hangs off the nitrogen, then wrap it -- "(cyclohexylamino)", "((prefix)amino)".
The last-resort branch now does the same: nitrogen excluded, one connected
carbon branch, named with the real substituent namer from its true attach atom.
Blue Book P-62.2.1.1.1 (the Blue Book) shows the pattern for an N-attached
carbon group as a prefix: "anilino (preferred prefix) phenylamino" and
"4-chloroanilino (preferred prefix) (4-chlorophenyl)amino" -- the carbon group,
then "amino"; never an amine name wrapped in "amino".
"""
import pytest
from rdkit import Chem

from orthonym.assembly import composer


def _n_branch(smiles, chain_len):
    """chain atoms are the first chain_len SMILES atoms (the nitrile chain)."""
    mol = Chem.MolFromSmiles(smiles)
    chain_set = set(range(chain_len))
    n_idx = next(a.GetIdx() for a in mol.GetAtoms()
                 if a.GetSymbol() == "N" and a.GetIdx() >= chain_len)
    sub_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in chain_set]
    return composer._name_n_attached_substituent_fallback(
        mol, sub_atoms, set(sub_atoms), chain_set, n_idx)


def test_single_ring_branch_is_named_from_its_carbon_side():
    prefix = _n_branch("N#CCCNCC1CCCCC1", 4)          # -NH-CH2-cyclohexyl
    assert prefix is not None
    assert "cyclohexylmethyl" in prefix and prefix.endswith("amino)")
    assert "amine" not in prefix and "amino)amino" not in prefix


def test_carbon_side_the_namer_declines_is_declined_not_fabricated():
    # name_substituent_fragment returns None for a substituted cyclohexyl
    # attached through the ring; the fallback must then decline too.
    prefix = _n_branch("N#CCCNC1CCC(C)CC1", 4)
    assert prefix is None or ("amine" not in prefix and prefix.endswith("amino)"))


def test_two_branches_on_the_nitrogen_are_declined_not_double_counted():
    prefix = _n_branch("N#CCCN(C)c1ccccc1", 4)
    assert prefix is None or "amino)" not in prefix, prefix


@pytest.mark.opsin_gate   # production config: the OPSIN validity gate is ON
def test_end_to_end_ring_branch_names_and_round_trips():
    from orthonym import Orthonym
    from orthonym.errors import is_failure_name
    from orthonym.validation.opsin_roundtrip import _find_opsin_jar, opsin_parse
    if _find_opsin_jar() is None:
        pytest.skip("no OPSIN jar")
    smi = "N#CCCNCC1CCCCC1"
    name = Orthonym(style="pin").name(smi)
    assert not is_failure_name(name), name
    back = opsin_parse(name)
    key = lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)).split("-")[0]
    assert back and key(back) == key(smi), (name, back)
