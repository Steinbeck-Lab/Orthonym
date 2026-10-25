"""N-alkoxy amides are named 'N-(R-oxy)...amide' at the default tier, and the Weinreb amides too.

Whole-molecule names for the cases of ``tests/unit/decomposition/test_n_alkoxy_amides.py``: the
decomposition engine named ``CONC(C)=O`` 'N-(O-methylhydroxyl)acetamide' and the long benzyloxy
amide 'N-(O-benzylhydroxyl)-2-[...]acetamide' (``O-methylhydroxyl`` is no prefix of the Blue Book),
and declined the Weinreb amides (the default tier shipped nothing for ``CC(=O)N(C)OC``).

 "N-SUBSTITUTION OF AMINES AND IMINES BY HETEROATOMS" (the Blue Book): "Substitution of
amines is permitted by -OR, -SR, -SeR, and -TeR groups" (:26596);
"N,O-Disubstitution of hydroxylamine" (:38372), 'N-methoxymethanamine (PIN)' (:38380);
"Hydroxamic acids" (:32821): "named as N-hydroxy amides", 'N-hydroxypropanamide (PIN)' (:32825).
INFERRED: the Blue Book gives no N-alkoxy amide PIN; each expected name follows from these rules.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,pin", [
    ("CONC(C)=O", "N-methoxyacetamide"),
    ("CC(=O)N(C)OC", "N-methoxy-N-methylacetamide"),
    ("CON(C)C(=O)c1ccccc1", "N-methoxy-N-methylbenzamide"),
    ("CCN(C(=O)c1ccccc1)OC", "N-ethyl-N-methoxybenzamide"),
    #, verbatim: unchanged by the N-alkoxy rule
    ("CC(=O)NO", "N-hydroxyacetamide"),
])
def test_n_alkoxy_amide_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_benzyloxy_amide_default_tier():
    # the best-effort tier names this molecule with another producer
    smi = "C1CC1N2C(=NN=N2)SCC(=O)NOCC3=CC=CC=C3"
    expected = "N-(benzyloxy)-2-[(1-cyclopropyl-1H-tetrazol-5-yl)sulfanyl]acetamide"
    assert name_is_rt_exact(expected, smi)
    d = name_default(smi)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (expected, "pin_verified", True), d


@pytest.mark.parametrize("smiles,pin", [
    ("CS(=O)(=O)NOC", "N-methoxymethanesulfonamide"),
    ("CS(=O)(=O)NCC", "N-ethylmethanesulfonamide"),
])
def test_n_alkoxy_sulfonamide_pin(smiles, pin):
    # the sulfonamide of an O-substituted hydroxylamine is cited like the amide (INFERRED, as above;
    # the base shipped 'N-O-methylhydroxylmethanesulfonamide' as a PIN)
    assert_pin_at_both_tiers(smiles, pin)


SULFONAMIDE_DEFAULT_TIER = [
    # a compound R-oxy N-substituent is enclosed: (the Blue Book) "Parentheses are
    # used around compound... prefixes", (:7416) brackets where parentheses are used
    ("CS(=O)(=O)NOCc1ccccc1", "N-(benzyloxy)methanesulfonamide"),
    ("CS(=O)(=O)NOCCCCC", "N-(pentyloxy)methanesulfonamide"),
    ("CS(=O)(=O)NOC(C)C", "N-[(propan-2-yl)oxy]methanesulfonamide"),
]


@pytest.mark.parametrize("smiles,pin", SULFONAMIDE_DEFAULT_TIER)
def test_n_alkoxy_sulfonamide_default_tier(smiles, pin):
    assert name_is_rt_exact(pin, smiles)
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (pin, "pin_verified", True), d


# The best-effort tier names these molecules with another producer, at a tier below the default
# tier's (a best-effort name should not rank below the default tier's PIN for the same input).
# Recorded as a strict xfail so that the fix flips it.
@pytest.mark.xfail(strict=True, reason=(
    "best-effort returns systematic_verified (inner:tier_a_ring / inner:general_acyclic) where the "
    "default tier returns the decomposition PIN"))
@pytest.mark.parametrize("smiles,pin", [
    ("C1CC1N2C(=NN=N2)SCC(=O)NOCC3=CC=CC=C3",
     "N-(benzyloxy)-2-[(1-cyclopropyl-1H-tetrazol-5-yl)sulfanyl]acetamide"),
])
def test_best_effort_is_not_below_the_default_tier_polyfunctional(smiles, pin):
    _best_effort_is_not_below_the_default_tier(smiles, pin)


# the sulfonamides are named by ``rules/n_hetero_amines.py`` at both tiers (the xfail of these rows flipped)
@pytest.mark.parametrize("smiles,pin", SULFONAMIDE_DEFAULT_TIER)
def test_best_effort_is_not_below_the_default_tier(smiles, pin):
    _best_effort_is_not_below_the_default_tier(smiles, pin)


def _best_effort_is_not_below_the_default_tier(smiles, pin):
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (pin, "pin_verified"), b


@pytest.mark.xfail(strict=True, reason=(
    "the substitutive attempt of the decomposition engine (engine.py, 'acid_has_pg') joins the "
    "AMINE NAME of a sulfonamide split to the acid name; the branch is relied on today by "
    "test_decomposition_fragment_fallback[HA20_sulfonamide_thiazole], "
    "test_recursive_substituents_secondary::test_retained_names_appear and "
    "test_branch_review_fixes::test_m_the_ordering_sort_key_memo_is_keyed_by_the_session_depth"))
def test_sulfonamide_split_is_not_joined_to_the_acid_as_a_prefix(monkeypatch):
    """_alcohol_to_alkyl hands a name that is no alcohol back unchanged, so the substitutive attempt
    wrote 'O-benzylhydroxylamine-4-methylbenzene-1-sulfonic acid' for the amine fragment of a
    sulfonamide: a malformed name, and while it is built the split's other candidates (the sulfonamide
    assembler's 'N-(benzyloxy)-4-methylbenzene-1-sulfonamide') are not tried. Taking the branch off the
    amine side lets the decomposition go on to other bonds of the molecule, and the lane measured
    that a decomposition candidate then wins where the general engine's name did (a sulfonyl spelled
    '1-oxo-1-(piperidin-1-yl)-2-oxa-1lambda6-thiaeth-1-en-1-yl'). Recorded as a strict xfail so that the
    fix of that spelling flips it."""
    from orthonym.decomposition import fragment_assembly as fa

    acids = []
    orig = fa._join_components

    def spy(left, right):
        acids.append(right)
        return orig(left, right)

    monkeypatch.setattr(fa, "_join_components", spy)
    smi = "Cc1ccc(cc1)S(=O)(=O)NOCc1ccccc1"
    name_default(smi)
    name_breadth(smi)
    assert acids, "the spy saw no join at all"
    assert "4-methylbenzene-1-sulfonic acid" not in acids
