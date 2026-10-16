"""SL lane G, G3: the retained word of a substituted two-carbon chain parent (pure decision).

 "Retained names as preferred IUPAC names" (the Blue Book): "All can be functionalized,
but only acetic acid, benzoic acid, and oxamic acid can be substituted"; row:29725 'CH3-COOH acetic acid
(PIN) ethanoic acid'. (:34750-:34754) 'CH3-CN acetonitrile (PIN) ethanenitrile', "with
unlimited substitution"; (:32691) 'acetamide (PIN)'. (:3031): "All locants are
omitted for parent compounds when all substitutable hydrogen atoms have the same locant";
(:7304): the second and later substituents are enclosed, 'bromo(chloro)acetic acid (PIN)' (:7312), and
"locants are required for related compounds where additional substitutable positions are available, for
example acetamide".
"""
import pytest

from rdkit import Chem

from orthonym.assembly.retained_c2_chain import RETAINED_C2_WORD, is_alkyl_attachment, retained_c2_name

pytestmark = pytest.mark.unit


def _name(suffix_core, prefix_parts, **kw):
    args = dict(hydride="ethane", chain_length=2, suffix_core=suffix_core, pg_locants=[1],
                prefix_parts=prefix_parts, charged=False, parent_stereo=False)
    args.update(kw)
    return retained_c2_name(**args)


@pytest.mark.parametrize("suffix,parts,expected", [
    ("oic acid", ["2-(quinolin-3-yl)"], ("(quinolin-3-yl)acetic acid", "acetic acid")),
    ("oic acid", ["2-chloro"], ("chloroacetic acid", "acetic acid")),
    ("oic acid", ["2,2-dichloro"], ("dichloroacetic acid", "acetic acid")),
    ("oic acid", ["2-bromo", "2-chloro"], ("bromo(chloro)acetic acid", "acetic acid")),
    ("nitrile", ["2-phenyl"], ("phenylacetonitrile", "acetonitrile")),
    ("al", ["2-phenoxy"], ("phenoxyacetaldehyde", "acetaldehyde")),
    # the amide keeps its locants: N and C-2 are both substitutable
    ("amide", ["N-methyl", "2-phenyl"], ("N-methyl-2-phenylacetamide", "acetamide")),
    ("amide", ["N,N-dimethyl"], ("N,N-dimethylacetamide", "acetamide")),
])
def test_the_retained_word_replaces_the_systematic_parent(suffix, parts, expected):
    got = _name(suffix, parts)
    assert got is not None
    assert got == expected


@pytest.mark.parametrize("kwargs", [
    dict(prefix_parts=[]),                          # nothing substituted: the exact retained lookup owns it
    dict(parent_stereo=True),                       # the descriptor is cited with its locant
    dict(charged=True),
    dict(hydride="eth-1-ene"),                      # unsaturated
    dict(chain_length=3),
    dict(pg_locants=[1, 2]),                        # two suffix groups
    dict(suffix_core="one"),                        # not a retained parent
    dict(suffix_core="al", prefix_parts=["2-chloro", "2-hydroxy"]),    # enclosing of an aldehyde follows:29852
    dict(prefix_parts=["3-chloro"]),                # a prefix that is not on C-2
    dict(alkyl_prefix=True),                        # an alkyl on C-2 needs another parent (:7304)
])
def test_the_systematic_name_stays_when_the_retained_one_is_not_certain(kwargs):
    args = dict(hydride="ethane", chain_length=2, suffix_core="oic acid", pg_locants=[1],
                prefix_parts=["2-chloro"], charged=False, parent_stereo=False)
    args.update(kwargs)
    assert retained_c2_name(**args) is None


def test_the_table_is_the_retained_names_of_the_book():
    assert RETAINED_C2_WORD == {"oic acid": "acetic acid", "nitrile": "acetonitrile",
                                "amide": "acetamide", "hydrazide": "acetohydrazide", "al": "acetaldehyde"}


@pytest.mark.parametrize("smiles,atom_symbol,expected", [
    ("CC(=O)O", "C", True),                       # a methyl group
    ("C=CC(=O)O", "C", True),                     # a vinyl group extends the chain
    ("C#CC(=O)O", "C", True),                     # an ethynyl group extends the chain
    ("CCCC(=O)O", "C", True),                     # the chain carbon of a longer alkyl
])
def test_an_alkyl_attachment_is_a_hydrocarbon_carbon_outside_the_rings(smiles, atom_symbol, expected):
    """ (the Blue Book): "In some cases, such as acetic acid, alkyl substituents are not
    allowed as the resulting compound requires a different parent structure"."""
    mol = Chem.MolFromSmiles(smiles)
    first = mol.GetAtomWithIdx(0)
    assert is_alkyl_attachment(first) is expected


@pytest.mark.parametrize("smiles,idx", [
    ("N#CCC(=O)O", 0),                            # cyano: the carbon of a triple bond to nitrogen
    ("OC(=O)CC(=O)O", 0),                         # carboxy
    ("O=CCC(=O)O", 0),                            # formyl
    ("CC(=O)CC(=O)O", 1),                         # acetyl: a carbon doubly bonded to oxygen
    ("c1ccccc1CC(=O)O", 0),                       # phenyl: in a ring
    ("C1CCCCC1CC(=O)O", 0),                       # cyclohexyl: in a ring
    ("OCC(=O)O", 0),                              # hydroxy: the first atom is no carbon
])
def test_a_group_that_is_not_an_alkyl_attachment(smiles, idx):
    assert is_alkyl_attachment(Chem.MolFromSmiles(smiles).GetAtomWithIdx(idx)) is False
