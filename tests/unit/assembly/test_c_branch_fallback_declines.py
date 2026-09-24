"""Bug B, C-attached twin (audit 2026-09-03): the last-resort branch of
``_name_c_attached_ring_substituent_fallback`` turned a capped fragment named
as a molecule ("2-chloro-1-(hexyloxy)-4-methoxybenzene") into a prefix by
string surgery ("...benzenyl") with no attach locant. The OPSIN gate rejected
the candidate and the molecule ABSTAINED where the later path names it
4-[6-(2-chloro-4-methoxyphenoxy)hexyl]heptane-3,5-dione (ChEBI:177546). The
branch had been dead (NameError on ``Chem``) until the import was added.

The structure-aware namer with the real attach atom runs right above this
branch; when it declines, the fragment-SMILES route has nothing legitimate to
add (its own contract, ``parent_to_prefix``: "may only emit locants it can
justify"), so the branch must decline too.
"""
import pytest
from rdkit import Chem

from orthonym.assembly import composer

WITNESS = "CCC(=O)C(CCCCCCOc1ccc(OC)cc1Cl)C(=O)CC"   # ChEBI:177546


def test_c_attached_ring_fallback_declines_instead_of_fabricating():
    mol = Chem.MolFromSmiles(WITNESS)
    chain = {0, 1, 2, 4, 17, 19, 20}              # heptane-3,5-dione skeleton atoms
    sub = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in chain and a.GetIdx() not in (3, 18)]
    out = composer._name_c_attached_ring_substituent_fallback(mol, sub, set(sub), chain, 5)
    assert out is None or "benzenyl" not in out, out


@pytest.mark.opsin_gate   # production config: the OPSIN validity gate is ON
def test_witness_keeps_its_name_end_to_end():
    from orthonym import Orthonym
    from orthonym.errors import is_failure_name
    from orthonym.validation.opsin_roundtrip import opsin_parse
    from tests.support.jars import jar_or_skip
    jar_or_skip()
    name = Orthonym(style="pin").name(WITNESS)
    assert not is_failure_name(name), name
    back = opsin_parse(name)
    key = lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)).split("-")[0]
    assert back and key(back) == key(WITNESS), (name, back)
