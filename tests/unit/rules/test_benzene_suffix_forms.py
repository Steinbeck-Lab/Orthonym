"""Missing benzene suffix forms: `-thiol` and `-carboperoxoic acid` (Phase C).

Both were absent from ``rules/benzene.py``'s ``_SUFFIX_PRIORITY`` and from its
suffix-FG detector, so each group was demoted to a prefix (or fell through to a trivial
name) and we shipped a non-PIN:

====================== ========================= =====================================
SMILES was PIN
====================== ========================= =====================================
``Sc1ccccc1`` ``sulfanylbenzene`` ``benzenethiol``
``OOC(=O)c1ccccc1`` ``perbenzoic acid`` ``benzenecarboperoxoic acid``
====================== ========================= =====================================

AUTHORITY, verified verbatim:

* ``:6656`` and ``:27292`` -- ``C6H5-SH benzenethiol (PIN) (not thiophenol)``, printed
  TWICE.
* ``:30178`` -- ``C6H5-CO-OOH benzenecarboperoxoic acid (PIN) peroxybenzoic acid
  perbenzoic acid``; ``:29797`` prints the same pair reversed. Independently ``:3009``
  states *"The prefix 'per-' is no longer recommended"*, so ``perbenzoic`` was doubly
  non-preferred.
* Existing grain proving benzene DOES take locant-free suffixes: ``:31163``
  ``benzenesulfonic acid (PIN)`` and ``:7625`` ``benzenehexol (PIN)``.

★ NEITHER WAS A LOCANT DEFECT, which is why the (c) licence could never have
produced the right name: the group was not being chosen as a suffix at all.

★ AND DENYING THE TRIVIAL NAME WAS NOT ENOUGH. Adding a ``pin: false`` row for
``perbenzoic acid`` alone produced ``unknown organic compound`` -- an abstention, not the
PIN -- because no systematic aryl peroxy-acid path existed. Session a project rule: removing
a wrong output can unmask something worse. The suffix form had to be BUILT.
"""
import pytest

from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


class TestNewSuffixForms:
    @pytest.mark.parametrize("smiles,expected,authority", [
        ("Sc1ccccc1", "benzenethiol", "BB:6656 + BB:27292, verbatim, twice"),
        ("OOC(=O)c1ccccc1", "benzenecarboperoxoic acid", "BB:30178 verbatim"),
    ])
    def test_pin_is_emitted(self, namer, smiles, expected, authority):
        assert namer.name(smiles) == expected, authority

    @pytest.mark.parametrize("smiles,expected", [
        ("Sc1ccccc1S", "benzene-1,2-dithiol"),
        ("Sc1ccc(S)cc1", "benzene-1,4-dithiol"),
    ])
    def test_multiplied_thiol_keeps_locants(self, namer, smiles, expected):
        """Two suffixes => no licence => locants cited,:2869)."""
        assert namer.name(smiles) == expected


class TestSeniorityGuard:
    """★ A REGRESSION THIS CHANGE CAUSED AND AN A/B CAUGHT BEFORE SHIPPING.

    The hydroxy->ol promotion in ``name_substituted_benzene`` was guarded on
    ``not suffix_groups`` -- "no other suffix at all". Adding the ``-thiol`` suffix form
    made that false whenever an SH was present, so the promotion was skipped and the
    JUNIOR thiol claimed the suffix: ``Sc1ccccc1O`` went from the correct
    ``2-sulfanylphenol`` to ``2-hydroxybenzenethiol``.

    The guard is now seniority-aware rather than existence-aware, so any future junior
    suffix form cannot reintroduce the inversion. ``SENIORITY_ORDER`` ranks:
    alcohols 88/89, phenol 91, thiol 94, amines 102+.
    """

    @pytest.mark.parametrize("smiles,expected,why", [
        ("Sc1ccccc1O", "2-sulfanylphenol", "-ol (91) outranks -thiol (94)"),
        ("Sc1ccc(O)cc1", "4-sulfanylphenol", "same, para"),
        ("OC(=O)c1ccc(S)cc1", "4-sulfanylbenzoic acid",
         "carboxylic acid (0) outranks both"),
        # locant cited beside the prefix, the Blue Book; the Blue Book)
        ("Nc1ccc(S)cc1", "4-aminobenzene-1-thiol", "-thiol (94) outranks -amine (102+)"),
    ])
    def test_senior_group_claims_the_suffix(self, namer, smiles, expected, why):
        assert namer.name(smiles) == expected, why

    def test_demoted_thiol_becomes_sulfanyl_not_dropped(self, namer):
        """When -thiol loses the slot it must demote to its PREFIX, not vanish --
        a project rule again: a dropped group is a wrong structure, far worse than a
        non-preferred spelling. ``_SUFFIX_TO_PREFIX['thiol'] == 'sulfanyl'``."""
        got = namer.name("Sc1ccccc1O")
        assert "sulfanyl" in got, f"the SH must survive as a prefix: {got!r}"
        assert got.endswith("phenol")

    def test_suffix_priority_places_the_new_forms_correctly(self):
        """Asserted as DATA, mirroring ``SENIORITY_ORDER``: peroxy acid at rank 1 sits
        directly after carboxylic acid at rank 0; thiol at 94 sits between -ol (91) and
        -amine (102+)."""
        from orthonym.rules.benzene import _SUFFIX_PRIORITY as P
        assert P.index("carboxylic acid") < P.index("carboperoxoic acid")
        assert P.index("carboperoxoic acid") < P.index("carboxamide")
        assert P.index("ol") < P.index("thiol") < P.index("amine")

    def test_seniority_order_agrees_with_the_priority_list(self):
        """The two orderings must not drift -- a divergence is exactly the bug that
        produced ``2-hydroxybenzenethiol``."""
        from orthonym.rules.benzene import _SUFFIX_PRIORITY as P
        from orthonym.rules.seniority import SENIORITY_ORDER
        rank = {k: i for i, k in enumerate(SENIORITY_ORDER)}
        assert rank["carboxylic_acid"] < rank["peroxy_acid"]
        assert rank["phenol"] < rank["thiol"] < rank["primary_amine"]
        assert P.index("ol") < P.index("thiol") < P.index("amine")


class TestUnchangedControls:
    @pytest.mark.parametrize("smiles,expected", [
        ("OC(=O)c1ccccc1", "benzoic acid"),
        ("OS(=O)(=O)c1ccccc1", "benzenesulfonic acid"),
        ("Oc1ccccc1", "phenol"),
        ("Nc1ccccc1", "aniline"),
        ("Oc1ccc(O)cc1", "benzene-1,4-diol"),
        ("Oc1ccccc1O", "benzene-1,2-diol"),
        ("CSSc1ccccc1", "(methyldisulfanyl)benzene"),
        ("OOC(=O)C1CCCCC1", "cyclohexanecarboperoxoic acid"),
        ("OOC(=O)CCCCC", "hexaneperoxoic acid"),
        ("Brc1ccc(Br)cc1", "1,4-dibromobenzene"),
    ])
    def test_control_unchanged(self, namer, smiles, expected):
        assert namer.name(smiles) == expected

    def test_disulfide_is_a_different_molecule(self, namer):
        """``CSSc1ccccc1`` is a DISULFIDE, not a thiol. There is an unrelated gold note
        mentioning ``sulfanylbenzene`` for it; the thiol work must not touch it."""
        assert namer.name("CSSc1ccccc1") == "(methyldisulfanyl)benzene"


class TestKnownAdjacentDefect:
    # Was a strict xfail under-citation, the Blue Book; the Blue Book
    # '4-methylbenzene-1,3-disulfonic acid (PIN)'); fixed in breadth job 3 --
    # 'sodium 4-methylbenzene-1-thiolate (PIN)' (the Blue Book).
    def test_substituted_benzenethiol_cites_its_locant(self, namer):
        assert namer.name("Sc1ccc(C)cc1") == "4-methylbenzene-1-thiol"
