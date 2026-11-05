"""A tertiary acyl nitrogen is cited as the N-substituted amido prefix (lane/l15 task A9).

 Substituents of the types -NH-CO-R and -NH-SO2-R (the Blue Book): "When a group having preference
for citation as a principal characteristic group is present, the group R-CO-NH-, or R-SO2-NH-... of an N-substituted
amide is named in two ways: (1) substitutively, by using a prefix formed by changing the final letter 'e' in the
complete name of the amide to 'o'... (2)... 'acylamino' prefixes... Method (1) generates preferred IUPAC names."
The book's own example is '2-(N-methylpropanamido)benzene-1-sulfonic acid (PIN)' against '2-[methyl(propanoyl)amino]
benzene-1-sulfonic acid' (:33039-33040). The acyl N of -N(R')-CO-R takes the amide name of the whole group ('N-methyl-2-
chloroacetamide', the prefixes in the order of, so a decorated acyl (a ring, a halogen, a branch) is as much the
prefix as the plain 'acetamido'.

The wider tiers never cite the acyl as a bare 'N-<acyl>' float on a parent that has no nitrogen suffix: the locant N
designates nothing there, the Blue Book: a locant is placed before the part of the name it relates to; the
locant N of an N-substituted amide is that of its amide suffix, N-Substitution,:32774).
"""
import re

import pytest
from rdkit import Chem

from orthonym.assembly.substituent_naming import n_substituted_acyl_amido_prefix
from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


#: an 'N-' locant opening the name, behind a stereodescriptor block '(2S)-' or an ester word ('methyl N-...')
_FLOATS_N = re.compile(r"^(\([^)]*\)-)?(N-|[a-z]+ N-)")


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


# (smiles, the PIN) -- every name read back whole by OPSIN to the input's full InChIKey, spelled as the book spells it
_CASES = [
    # the book's own example: the acyl is the linear 'propanamido' on a benzene parent
    ("CCC(=O)N(C)c1ccccc1S(=O)(=O)O", "2-(N-methylpropanamido)benzene-1-sulfonic acid"),
    ("OC(=O)c1ccc(N(C)C(=O)C)cc1", "4-(N-methylacetamido)benzoic acid"),
    # a branched acyl: N and C-2 cite the one methyl multiplied: 'N,2-dimethyl')
    ("OC(=O)c1ccc(N(CC)C(=O)C(C)C)cc1", "4-(N-ethyl-2-methylpropanamido)benzoic acid"),
    # a decorated acyl: the enclosing marks of the prefix are square brackets
    ("COC(=O)c1ccc(N(C)C(=O)Cc2ccccc2)cc1", "methyl 4-(N-methyl-2-phenylacetamido)benzoate"),
    ("COC(=O)c1ccc(N(C)C(=O)CN2CCOCC2)cc1", "methyl 4-[N-methyl-2-(morpholin-4-yl)acetamido]benzoate"),
    ("OC(=O)c1ccc(N(C)C(=O)CN2CCCC2)cc1", "4-[N-methyl-2-(pyrrolidin-1-yl)acetamido]benzoic acid"),
    # a chain parent
    ("OC(=O)CCN(C)C(=O)CCl", "3-(2-chloro-N-methylacetamido)propanoic acid"),
    ("OC(=O)CCN(C)C(=O)c1ccccc1", "3-(N-methylbenzamido)propanoic acid"),
    ("OC(=O)CCN(C)C(=O)C(C)C", "3-(N,2-dimethylpropanamido)propanoic acid"),
    # a ring parent that is not benzene
    ("OC(=O)C1CCC(CC1)N(C)C(=O)CCl", "4-(2-chloro-N-methylacetamido)cyclohexane-1-carboxylic acid"),
    # the control that has always been named
    ("CC(=O)N(C)CCC(=O)O", "3-(N-methylacetamido)propanoic acid"),
    # an amide parent that is senior to the group: the second amide group is the suffix and the tertiary acyl nitrogen the
    # prefix; the parent chain is the longer one (b), the Blue Book), so is not reached
    ("ClCC(=O)N(C)C(C)C(N)=O", "2-(2-chloro-N-methylacetamido)propanamide"),
    # a ring amide is the parent of a chain amide (1),:19336 "Within the same class, a ring or ring system has
    # seniority over a chain"): the benzamide takes the acetamido group as a prefix
    ("NC(=O)c1ccc(N(C)C(=O)C)cc1", "4-(N-methylacetamido)benzamide"),
    # stereo: the descriptors of the acyl and of the N-substituent are cited, none lost: "In preferred
    # IUPAC names, stereodescriptors, preceded by a locant, must be cited to specify each stereogenic unit")
    ("OC(=O)c1ccc(N(C)C(=O)[C@@H](N)C)cc1", "4-[(2S)-2-amino-N-methylpropanamido]benzoic acid"),
    ("OC(=O)c1ccc(N([C@@H](C)CC)C(=O)CCl)cc1", "4-{N-[(2S)-butan-2-yl]-2-chloroacetamido}benzoic acid"),
    ("OC(=O)CCN([C@H](C)CC)C(=O)[C@H](Cl)C", "3-{(2R)-N-[(2R)-butan-2-yl]-2-chloropropanamido}propanoic acid"),
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", _CASES)
def test_tertiary_acyl_nitrogen_gets_the_n_substituted_amido_prefix(smiles, name, tier):
    """ (the Blue Book), method (1): the prefix is the complete name of the amide with the final 'e'
    changed to 'o'; '2-(N-methylpropanamido)benzene-1-sulfonic acid (PIN)' (:33040)."""
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified", row
    assert_full_rt(row["name"], smiles)


# the acyl holds an N-methylpiperazin-1-yl ring: a prefix the PIN vocabulary of the first run has no producer for, named by
# the promoted producers of the PIN tier's re-run (lane/l17 item 4: the first run's refusal of the amide of the group is
# no longer served to the re-run, ``fragment_naming``); its NH analog 'OC(=O)c1ccc(NC(=O)CN2CCN(C)CC2)cc1' is named there as well
_PIPERAZINE = ("OC(=O)c1ccc(N(C)C(=O)CN2CCN(C)CC2)cc1", "4-[N-methyl-2-(4-methylpiperazin-1-yl)acetamido]benzoic acid")


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_a_ring_bearing_acyl_with_a_ring_prefix_of_the_rerun_is_the_pin_prefix(tier):
    """The same prefix at every tier as for the NH analog ('4-[2-(4-methylpiperazin-1-yl)acetamido]benzoic acid'):
     Substituents of the types -NH-CO-R and -NH-SO2-R (the Blue Book): "Method (1) generates preferred
    IUPAC names" -- the prefix is the complete name of the amide with the final 'e' changed to 'o'; the PIN tier names it."""
    smiles, name = _PIPERAZINE
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("name", [
    "N-(phenylacetyl)-4-(methylamino)benzoic acid",
    "methyl N-(phenylacetyl)-4-(methylamino)benzoate",
    "(2S)-N-(methoxyacetyl)-2-(2,6-dimethylanilino)propanoic acid",     # behind a stereodescriptor block
])
def test_the_float_pattern_catches_a_float(name):
    assert _FLOATS_N.match(name), name


@pytest.mark.parametrize("name", ["4-(N-methyl-2-phenylacetamido)benzoic acid", "(2S)-2-[N-(2,6-dimethylphenyl)acetamido]propanoic acid"])
def test_the_float_pattern_leaves_an_n_inside_a_prefix_alone(name):
    assert not _FLOATS_N.match(name), name


@pytest.mark.parametrize("smiles,name", _CASES + [_PIPERAZINE])
def test_no_name_floats_n_on_a_parent_without_a_nitrogen_suffix(smiles, name):
    """The best-effort name cites no 'N-<acyl>' in front of an acid, an ester or a ring: (the Blue Book),
    a locant is placed before the part of the name it relates to, and the parent has no nitrogen suffix to relate N to."""
    row = _row(smiles, "best-effort")
    assert not _FLOATS_N.match(row["name"]), row


# An amide parent whose chain the group's amide ties or outranks. "Maximum number of substituent groups cited as
# prefixes" (the Blue Book): "The preferred IUPAC name is based on the senior parent structure that has the maximum
# number of substituents cited as prefixes", the example (4) counting the N-substituents ('N,N,2-trimethyl-3-{...}
# propanamide (PIN)'). With the two amide chains of one length the tertiary amide is the parent (three prefixes, against
# two for the secondary amide with the amido prefix), so '2-(N,2-dimethylpropanamido)-N-methylpropanamide' is not the PIN.
# The builder cites the group only where the amide parent is strictly senior (a ring, or the longer chain).
_AMIDE_PARENT_TIES = [
    # (smiles, the PIN, the name the amido prefix would have given)
    ("CC(C)C(=O)N(C)C(C)C(=O)NC", "N,2-dimethyl-N-[1-(methylamino)-1-oxopropan-2-yl]propanamide",
     "2-(N,2-dimethylpropanamido)-N-methylpropanamide"),
    ("CC(C)C(=O)C(=O)N(C)[C@H](C(=O)NCCc1c[nH]c2ccccc12)C(C)C",
     "N-[(2S)-1-{[2-(1H-indol-3-yl)ethyl]amino}-3-methyl-1-oxobutan-2-yl]-N,3-dimethyl-2-oxobutanamide",
     "(2S)-2-(N,3-dimethyl-2-oxobutanamido)-N-[2-(1H-indol-3-yl)ethyl]-3-methylbutanamide"),
    # the group's amide is the ring one: the benzamide is the parent (1)), not a chain amide with the amido prefix
    ("NC(=O)C(C)N(C)C(=O)c1ccccc1", "N-(1-amino-1-oxopropan-2-yl)-N-methylbenzamide",
     "2-(N-methylbenzamido)propanamide"),
    # the linear acyl, named '2-(N-methylpropanamido)propanamide' at the lane start (a false PIN of the same class)
    ("CCC(=O)N(C)C(C)C(N)=O", "N-(1-amino-1-oxopropan-2-yl)-N-methylpropanamide",
     "2-(N-methylpropanamido)propanamide"),
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,pin_name,false_name", _AMIDE_PARENT_TIES)
def test_an_amide_parent_the_group_ties_is_not_named_with_the_amido_prefix(smiles, pin_name, false_name, tier):
    """ (the Blue Book): no tier ships the amido-prefix name as a PIN; the PIN tier gives the name
    or abstains, and the name OPSIN reads back to the input is the one the book's rule gives."""
    row = _row(smiles, tier)
    assert not (row["tier"] == "pin_verified" and row["name"] == false_name), row
    if row["tier"] == "pin_verified":
        assert row["name"] == pin_name, row
    assert_full_rt(pin_name, smiles)
    if row["tier"] != "abstain":
        assert_full_rt(row["name"], smiles)


# --- the builder -------------------------------------------------------------------------------------------------------

def _frag(smiles, parent_smarts):
    mol = Chem.MolFromSmiles(smiles)
    parent = set(mol.GetSubstructMatch(Chem.MolFromSmarts(parent_smarts)))
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and not a.IsInRing()
             and any(nb.GetIdx() in parent for nb in a.GetNeighbors()))
    # the substituent is the N and every atom that hangs on it away from the parent
    seen, stack = {n}, [n]
    while stack:
        for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            i = nb.GetIdx()
            if i not in parent and i not in seen:
                seen.add(i)
                stack.append(i)
    return mol, n, seen, parent


@pytest.mark.parametrize("smiles,prefix", [
    ("OC(=O)CCN(C)C(=O)CCl", "2-chloro-N-methylacetamido"),
    ("OC(=O)CCN(C)C(=O)C(C)C", "N,2-dimethylpropanamido"),
    ("OC(=O)CCN(C)C(=O)c1ccccc1", "N-methylbenzamido"),
    ("OC(=O)CCN(C)C(=O)CN1CCN(C)CC1", "N-methyl-2-(4-methylpiperazin-1-yl)acetamido"),
    # the unsaturated acyl is as much the amide name of the group as the saturated one
    ("OC(=O)CCN(C)C(=O)C=C", "N-methylprop-2-enamido"),
])
def test_the_builder_names_a_decorated_acyl(smiles, prefix):
    """ (:32995): "a prefix formed by changing the final letter 'e' in the complete name of the amide to
    'o'": the amide of -N(CH3)-CO-CH2Cl is '2-chloro-N-methylacetamide' (the prefixes in the order of."""
    mol, n, sub, parent = _frag(smiles, "OC(=O)CC")
    assert n_substituted_acyl_amido_prefix(mol, n, sub, parent) == prefix


def test_the_builder_declines_a_group_with_a_second_amide():
    """With a second carboxamide on the N-substituent the group holds two acylated nitrogens: the complete name of its
    amide could be the name of the other amide, and its 'amido' would attach the group at the wrong nitrogen, so no
    prefix is built from it."""
    mol, n, sub, parent = _frag("OC(=O)CCN(CC(=O)NC)C(=O)CCl", "OC(=O)CC")
    assert n_substituted_acyl_amido_prefix(mol, n, sub, parent) is None


@pytest.mark.parametrize("smiles", [
    "OC(=O)CCN(C)C(=O)OC",       # a carbamate: the name is an ester name, not an amide name
    "OC(=O)CCN(C)C(=O)N(C)C",    # a urea: two acylated nitrogens
])
def test_the_builder_declines_a_hetero_acyl(smiles):
    """The complete name of the group's amide is not one amide of a carboxylic acid: 'methyl methylcarbamate',
    'N,N,N'-trimethylurea'. No 'amido' prefix is built from either."""
    mol, n, sub, parent = _frag(smiles, "OC(=O)CC")
    assert n_substituted_acyl_amido_prefix(mol, n, sub, parent) is None


def test_a_hydrazide_nitrogen_keeps_the_hydrazinyl_prefix():
    """-N(NHR)-CO-R is the hydrazin-1-yl unit,; the amido prefix is for a nitrogen whose substituent is a
    carbon group (task A9 control: the nitrogen of a hydrazide stays '1-acetyl-2-ethylhydrazin-1-yl')."""
    smiles = "CCNN(C(C)=O)c1ccc(C(=O)O)cc1"
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == ("4-(1-acetyl-2-ethylhydrazin-1-yl)benzoic acid", "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def test_a_group_with_a_second_amide_keeps_its_acylamino_name():
    """The control: the amide name of this group could be the name of the other amide, so its name stays the one the
    wider tier gave before (the acyl is cited as '2-chloro-1-oxoethyl' there) and reads back whole."""
    smiles = "OC(=O)CCN(CC(=O)NC)C(=O)CCl"
    row = _row(smiles, "best-effort")
    assert row["name"] == "3-{(2-chloro-1-oxoethyl)[2-(methylamino)-2-oxoethyl]amino}propanoic acid", row
    assert_full_rt(row["name"], smiles)


def test_the_builder_reads_the_prefix_back_against_the_fragment(monkeypatch):
    """A prefix that does not describe the fragment is not returned: the amide name is read back by OPSIN against
    the fragment capped at the nitrogen (an atom the namer dropped is not named)."""
    from orthonym.assembly import fragment_naming
    monkeypatch.setattr(fragment_naming, "name_fragment_recursively", lambda smiles, *a, **k: "N-methylacetamide")
    mol, n, sub, parent = _frag("OC(=O)CCN(C)C(=O)CCl", "OC(=O)CC")
    assert n_substituted_acyl_amido_prefix(mol, n, sub, parent) is None
