"""N-substituted carboxamides on fused ring systems (2026-11-20).

 (the Blue Book) and (:32774, N-Substitution): the suffix 'carboxamide' on a ring, the nitrogen
substituents cited as prefixes with the locant N ('N,N-dimethylbenzamide'). The fused producers (``rules.polycyclics``,
``rules.fused_rings``) matched only -CONH2, so these got no PIN and the default tier shipped a von Baeyer name
('N-methylbicyclo[4.4.0]deca-1(10),2,4,6,8-pentaene-3-carboxamide'). The new rule is ``rules.fused_amide_suffix``.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("O=C(NC)c1ccc2ccccc2c1", "N-methylnaphthalene-2-carboxamide"),
    ("O=C(N(C)C)c1ccc2ccccc2c1", "N,N-dimethylnaphthalene-2-carboxamide"),
    ("CC(=O)c1ccc2cc(C(=O)NC)ccc2c1", "6-acetyl-N-methylnaphthalene-2-carboxamide"),
    ("Nc1ccc2cc(C(=O)NC)ccc2c1", "6-amino-N-methylnaphthalene-2-carboxamide"),
    ("O=C(NC)c1ccc2ccccc2n1", "N-methylquinoline-2-carboxamide"),
    # was '3-(methylcarbamoyl)quinoline' below the PIN (test_oxo_prefix_senior_suffix)
    ("CNC(=O)c1cnc2ccccc2c1", "N-methylquinoline-3-carboxamide"),
    # the certified fused route now writes it, as the decomposition route did (test_s2c1_plain_fused_path)
    ("CC(C)(C)N(C(C)(C)C)C(=O)c1ccc2OCOc2c1", "N,N-di-tert-butyl-2H-1,3-benzodioxole-5-carboxamide"),
    ("Cn1nc(C(=O)NC)c2ccccc21", "N,1-dimethyl-1H-indazole-3-carboxamide"),
    ("Cn1nc(C(=O)NC2CCCCC2)c2ccccc21", "N-cyclohexyl-1-methyl-1H-indazole-3-carboxamide"),
    ("Cn1cc(C(=O)NC)c2ccccc21", "N,1-dimethyl-1H-indole-3-carboxamide"),
    ("CN(C)C(=O)c1ccc2[nH]ccc2c1", "N,N-dimethyl-1H-indole-5-carboxamide"),
    ("O=C(NC)c1cc2ccccc2o1", "N-methyl-1-benzofuran-2-carboxamide"),
    ("O=C(Nc1ccccc1)c1ccc2ccccc2c1", "N-phenylnaphthalene-2-carboxamide"),     # a ring N group on a carbocycle
    # granisetron without its stereo (CHEBI:5537)
    ("Cn1nc(C(=O)NC2CC3CCCC(C2)N3C)c2ccccc21",
     "1-methyl-N-(9-methyl-9-azabicyclo[3.3.1]nonan-3-yl)-1H-indazole-3-carboxamide"),
])
def test_fused_n_carboxamide_pins(smiles, name):
    row = tier_row(smiles, "pin")
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert_full_rt(row["name"], smiles)


def test_two_n_substituted_carboxamides_decline_here():
    """Two N-substituted carboxamides on one core would need N and N' locants; the fused producers decline, so no name
    binds the N prefixes to the wrong amide."""
    assert tier_row("O=C(NC)c1ccc2ccccc2c1C(=O)NC", "pin")["tier"] == "abstain"


@pytest.mark.parametrize("acid_smiles,second", [
    ("CNC(=O)c1cccc2cccc(C(=O)O)c12", True),             # two carboxamides on one ring system
    ("CC(=O)NC1CC2CCCC(C1)N2C(=O)O", False),             # a urea's acid part: the carboxyl on the ring N is no amide
    ("C=IC1=C(C2(CCNCC2)CN1C(=O)O)PP", False),           # the same, on a spiro ring
    ("O=C1CCc2cc(C(=O)O)ccc2N1", False),                 # a ring lactam
    ("CCCC(=O)N1CCC(C(=O)O)CC1", False),                 # an acyl group on the ring N: the ring is the amine part
    ("CNC(=O)N1CCC(C(=O)O)CC1", True),                   # a urea on the ring N is a 1-carboxamide of the ring
    ("OC(=O)c1ccc2ccccc2c1", False),
])
def test_the_amide_parent_guard_counts_only_a_second_amide(acid_smiles, second):
    """decomposition.amide_parent_guard: the acid part's own carboxyl, bonded to a ring N, also matches the amide
    pattern; counted as a second amide it declined every urea on a ring N (end check CT7 suite)."""
    from orthonym.decomposition.amide_parent_guard import second_amide_on_acid_ring
    assert second_amide_on_acid_ring(acid_smiles) is second


@pytest.mark.parametrize("smiles", ["CNC(=O)c1ccc2cc(C(N)=O)ccc2c1", "CNC(=O)c1cnc2ccc(C(N)=O)cc2c1"])
def test_a_carbamoyl_prefix_beside_the_carboxamide_suffix_is_no_pin(smiles):
    """Two carboxamides on one ring parent are both the suffix;, the Blue Book, N locants
    for diamides; 'benzene-1,4-dicarboxamide (PIN)':32718): '6-(methylcarbamoyl)naphthalene-2-carboxamide' is a
    correct structure but not the PIN spelling (rules.pin_vocabulary), so the PIN tier declines and the wider tier
    keeps a name below the PIN."""
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    row = tier_row(smiles, "best-effort")
    assert row["tier"] != "pin_verified" and row["is_pin"] is False, row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("name,flagged", [
    ("trifluoroacetic acid\u20146-(methylcarbamoyl)naphthalene-2-carboxamide (1/1)", True),   # each word on its own
    ("4'-carbamoyl[1,1'-biphenyl]-4-carboxamide", True),
    ("4-(methylcarbamoyl)benzamide", True),
    # (:32774): superscript N locants for diamides, 'N1-methylbutanediamide (PIN)'
    ("N-methylpyridine-2,4-dicarboxamide", True),
    ("N,N-dimethylbutanediamide", True),
    ("N1-methylbutanediamide", False),
    ("N2-methylnaphthalene-2,6-dicarboxamide", False),
    ("N-carbamoylbenzamide", False),
    ("N,N-diethylfuran-2-carboxamide", False),
    ("4-(carbamoylamino)benzamide", False),
])
def test_polyamide_spellings_that_are_no_pin(name, flagged):
    from orthonym.rules.pin_vocabulary import non_pin_vocabulary
    assert (non_pin_vocabulary(name) is not None) is flagged, name


def test_primary_amides_are_unchanged():
    assert tier_row("O=C(N)c1ccc2ccccc2c1", "pin")["name"] == "naphthalene-2-carboxamide"
    assert tier_row("O=C(N)c1ccc2ccccc2n1", "pin")["name"] == "quinoline-2-carboxamide"


def test_best_effort_takes_the_pin_when_the_n_group_is_left_to_the_amide_producers():
    """A ring N group with a stereocentre is left to the amide producers (benzene._detect_n_substituents); the wider
    tier's fused producer would cite a 'carbamoyl' prefix ('2-[(S)-(1-phenylethyl)carbamoyl]naphthalene'), so it asks
    the PIN tier (assembly.pin_first), whose verified PIN wins."""
    smiles = "C[C@H](NC(=O)c1ccc2ccccc2c1)c1ccccc1"
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert row["name"] == "N-[(1S)-1-phenylethyl]naphthalene-2-carboxamide", (tier, row)
        assert row["tier"] == "pin_verified", (tier, row)


@pytest.mark.parametrize("smiles", ["CNC(=O)c1cccc2cccc(C(=O)NC)c12", "CNC(=O)c1ccc(C(=O)NC)cc1"])
def test_a_second_amide_of_the_ring_is_no_prefix_of_a_pin(smiles):
    """: two carboxamides of one ring system are both the suffix ('naphthalene-1,8-dicarboxamide'). The
    decomposition named one as a prefix: 'N-methyl-8-(methylcarbamoyl)naphthalene-1-carboxamide' was pin_verified
    (decomposition.amide_parent_guard). No such name ships as a PIN."""
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert not (row["is_pin"] and "carbamoyl" in (row["name"] or "")), (tier, row)


def test_a_ring_lactam_is_no_second_amide():
    """A lactam C=O belongs to its ring, so the guard against a second amide leaves it alone (end check CT7: the sample
    row below lost its name when the guard counted the ring C=O)."""
    row = tier_row("CN1C[C@]23C=CC(O2)C(C3C1=O)C(=O)NC4=CC=CC=C4OCCCC5=CC=CC=C5", "pin")
    assert row["tier"] == "pin_verified", row

