import pytest

from orthonym.rules.phane_pin import linear_phane_name
from orthonym.validation.spelling.linear_phane import linear_phane_chain, linear_phane_pin_expected

from ._helpers import gold, mol, substituted

FOUR_RINGS = "c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1"      #:23959, four benzene rings and three O


@pytest.mark.parametrize("line,name,smiles", gold("path"), ids=lambda v: str(v)[:30])
def test_book_linear_pins(line, name, smiles):
    # (:23899) and the numbering rules of: all 55 linear rows of the gold
    assert linear_phane_name(mol(smiles)) == name


def test_the_chain_is_the_chain_of_the_verdict():
    m = mol(FOUR_RINGS)
    chain = linear_phane_chain(m)
    verdict = linear_phane_pin_expected(m)
    assert sum(1 for n in chain if n[0] == "R") == verdict.ring_systems and len(chain) == verdict.nodes


@pytest.mark.parametrize("smiles", [
    "c1ccc(Oc2ccc(Oc3ccccc3)cc2)cc1",                 #:23925 3 rings, 5 nodes: a multiplicative PIN
    "c1ccccc1CCCCCCc1ccccc1",                          # two rings
    "CCCCCCCCCC",
])
def test_below_the_threshold_nothing_is_built(smiles):
    assert linear_phane_name(mol(smiles)) is None


def test_an_ion_is_not_built():
    # lane L5 decision 4: principal_class 'ion' fails closed
    assert linear_phane_name(mol("c1ccc(Oc2ccc(Oc3ccc(Oc4ccc([N+](C)(C)C)cc4)cc3)cc2)cc1")) is None


def test_a_phane_that_is_only_a_prefix_is_not_built_here():
    # (:18875): another parent cites more of the principal group; PN6b
    smiles = "OC(=O)CC(C(=O)O)c1ccc(Oc2ccc(Oc3ccc(Oc4ccc(C(=O)O)cc4)cc3)cc2)cc1"
    assert linear_phane_pin_expected(mol(smiles)).as_prefix
    assert linear_phane_name(mol(smiles)) is None


SILANE = "C[Si](C)(C)c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1"      # four benzene rings, three O, SiMe3


def test_a_senior_atom_parent_outranks_the_phane():
    # (:18917) with (:18919) "A single senior atom is sufficient to give seniority to the parent
    # hydride." and the book's row (:19325) '...heptaphan-7^4-yl]silane (PIN) (Si is senior to O)'. The reader used to
    # certify the phane here: LinearPhaneVerdict(ring_systems=4, nodes=7, principal_class=None, expressed=0,
    # occurrences=0, as_prefix=False)
    verdict = linear_phane_pin_expected(mol(SILANE))
    assert verdict.as_prefix and verdict.criterion == "P-44.1.2"
    assert linear_phane_name(mol(SILANE)) is None


# the boundary rows of (:18919-:19334) as structures of a chain of four benzene rings:
# Si, P: tetramethylsilane (Si is senior to C), (1-benzofuran-2-yl)phosphane (P is senior to O) -> the phane is no parent
# N in the phane: 1-(trimethylsilyl)-1H-imidazole (N is senior to Si) -> the phane stays
# N-N outside: 1-(2H-pyran-3-yl)-2-(silolan-2-yl)hydrazine (:18950, N > Si > O) -> the phane is no parent
# O outside: not a parent hydride that outranks the phane; the criterion is not used between rings (:18917)
# a senior atom in a ring outside the chain: "not used to choose between rings" -> the phane stays
#:19328-:19334: 'a' replacement chains (O > S, O-chain > Si-chain) are not read by -> skeleton atoms never count
@pytest.mark.parametrize("smiles,prefix", [
    (SILANE, True),
    ("Pc1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", True),                                # a phosphane
    ("C[Ge](C)(C)c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", True),                       # a germane
    ("C[Si](C)(C)c1ccc(Oc2ccc(Oc3ccc(Oc4ccncc4)cc3)cc2)cc1", False),                      # N is in the phane
    ("COc1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", False),                               # O outside heads no senior parent
    ("c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1[SiH]1CCC1", False),                       # Si in a ring outside the chain
    ("c1ccc(O[SiH2]Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", False),                          # Si in the chain is 'sila' replacement
    ("C[Si](C)(C)c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2C(=O)O)cc1", False),                # first: the phane cites the acid
    ("CN=Nc1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", True),                              # an N-N unit heads a diazene (:18950)
    ("O=[N+]([O-])c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", False),                     # a nitro group heads no parent hydride
])
def test_the_boundary_rows_of_the_senior_atom_criterion(smiles, prefix):
    verdict = linear_phane_pin_expected(mol(smiles))
    assert verdict is not None and verdict.as_prefix is prefix, verdict


def test_no_book_phane_pin_is_outranked():
    # the criterion must not decline a structure whose phane the book prints as the PIN (N and O, S of the chain
    # and the amplificants are not outside atoms)
    for line, name, smiles in gold("path"):
        assert not linear_phane_pin_expected(mol(smiles)).as_prefix, name


@pytest.mark.opsin_gate
def test_the_default_tier_still_declines_the_silane_and_best_effort_keeps_its_name():
    # the label is lowered before and after the change; only the detail of the failure names now
    from tests.support.default_tier import assert_default_tier_declines
    row = assert_default_tier_declines(SILANE)
    assert "P-52.2.5.1" in [f["rule"] for f in row["spelling_failures"]]
    assert "P-44.1.2" in " ".join(f["detail"] for f in row["spelling_failures"])


_SUBSTITUENT_SENTINEL = ("the substituent namer gives the 'substituent' sentinel for a chloro- or bromonaphthalenyl "
                         "fragment, so the prefix block is not built")


@pytest.mark.parametrize("line,name,smiles", [
    pytest.param(*row, marks=pytest.mark.xfail(strict=False, reason=_SUBSTITUENT_SENTINEL)) if row[0] != "21631" else row
    for row in substituted("compound_prefix")
], ids=lambda v: str(v)[:30])
def test_book_compound_prefix_names(line, name, smiles):
    # (:21604) the parent has the maximum number of prefixes: the naphthalene with the chloro group is the
    # chain, the other naphthalene a substituent (:21631)
    assert linear_phane_name(mol(smiles)) == name
