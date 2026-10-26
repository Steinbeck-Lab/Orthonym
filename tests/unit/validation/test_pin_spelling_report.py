"""DT3 step B: a spelling check says PASS, FAIL or ABSTAIN, and a name gets a coverage report.

``check_pin_spelling`` keeps its contract (the list of failures, None for pass and abstain alike). A check may
return ``PASS`` (it read the name and the rule holds) or ``ABSTAIN`` (it could not read what the rule is about);
``pin_spelling_report`` keeps the verdict of every check, and the coverage of the name: the units whose parent
a reader models (``validation.spelling.coverage``), of all its units. A pass of the numbering checks on a name
no reader models is no evidence; the report is what says so.
"""
import pytest
from rdkit import Chem

from orthonym.validation import pin_spelling as ps
from orthonym.validation.spelling.coverage import NameCoverage, name_coverage

pytestmark = pytest.mark.unit

MOL = Chem.MolFromSmiles("CC")


@pytest.fixture
def empty_registry(monkeypatch):
    monkeypatch.setattr(ps, "_REGISTRY", {})
    monkeypatch.setattr(ps, "_loaded", True)


def test_the_verdicts_are_four_distinct_words():
    assert len({ps.PASS, ps.FAIL, ps.ABSTAIN, ps.UNDECIDED}) == 4


def test_the_report_keeps_each_check_as_pass_fail_abstain_or_undecided(empty_registry):
    @ps.register("P-TEST")
    def passes(mol, name):
        return ps.PASS

    @ps.register("P-TEST")
    def fails(mol, name):
        return ps.SpellingFailure("P-TEST", "broken")

    @ps.register("P-TEST")
    def abstains(mol, name):
        return ps.ABSTAIN

    @ps.register("P-TEST")
    def silent(mol, name):
        return None

    report = ps.pin_spelling_report(MOL, "hexan-2-ol")
    assert [(o.check, o.verdict) for o in report.outcomes] == [
        ("passes", ps.PASS), ("fails", ps.FAIL), ("abstains", ps.ABSTAIN), ("silent", ps.UNDECIDED)]
    assert [f.detail for f in report.failures] == ["broken"]
    assert [report.count(v) for v in (ps.PASS, ps.FAIL, ps.ABSTAIN, ps.UNDECIDED)] == [1, 1, 1, 1]


def test_a_label_reads_failures_only(empty_registry):
    """check_pin_spelling is unchanged: PASS, ABSTAIN and None are all 'no failure'."""
    ps.register("P-TEST")(lambda mol, name: ps.PASS)
    ps.register("P-TEST")(lambda mol, name: ps.ABSTAIN)
    ps.register("P-TEST")(lambda mol, name: None)
    assert ps.check_pin_spelling(MOL, "hexan-2-ol") == []
    ps.register("P-TEST")(lambda mol, name: ps.SpellingFailure("P-TEST", "x"))
    assert [f.rule for f in ps.check_pin_spelling(MOL, "hexan-2-ol")] == ["P-TEST"]


def test_a_check_that_raises_is_an_abstention_in_the_report_and_raises_when_strict(empty_registry):
    @ps.register("P-TEST")
    def broken(mol, name):
        raise RuntimeError("reader defect")

    report = ps.pin_spelling_report(MOL, "hexan-2-ol")
    assert [(o.check, o.verdict) for o in report.outcomes] == [("broken", ps.ABSTAIN)]
    with pytest.raises(RuntimeError):
        ps.pin_spelling_report(MOL, "hexan-2-ol", strict=True)


def test_no_structure_or_no_name_has_no_outcome():
    assert ps.pin_spelling_report(None, "ethene").outcomes == ()
    assert ps.pin_spelling_report(MOL, "").coverage is None


def test_the_built_in_checks_give_a_report_with_coverage():
    report = ps.pin_spelling_report(Chem.MolFromSmiles("CCCCCC(C)O"), "hexan-2-ol", strict=True)
    assert report.failures == []
    assert report.count(ps.UNDECIDED) + report.count(ps.PASS) + report.count(ps.ABSTAIN) == len(report.outcomes)
    assert report.coverage == NameCoverage(units=1, decided=1, unmodelled=(), p10=False)


@pytest.mark.parametrize("name, units, decided, full", [
    ("hexan-2-ol", 1, 1, True),
    ("cyclohexane-1,2-diol", 1, 1, True),
    ("4-(2-chlorophenyl)pyridine", 2, 2, True),
    ("2-(2,2,3-trimethylcyclopent-3-en-1-yl)ethyl acetate", 3, 2, False),         # 'acetate' is no model
    ("N-[2-(5-methoxy-1H-indol-3-yl)ethyl]acetamide", 3, 1, False),               # indole is no model
    ("methyl beta-D-glucopyranoside", 2, 1, False),                               # a name is never full
])
def test_the_coverage_counts_the_units_a_reader_models(name, units, decided, full):
    cov = name_coverage(name)
    assert (cov.units, cov.decided, cov.full) == (units, decided, full)
    assert 0.0 <= cov.fraction <= 1.0


def test_a_name_whose_marks_do_not_pair_has_no_coverage():
    assert name_coverage("N-[2-(methyl)ethyl") is None
    assert name_coverage("") is None
    assert name_coverage(None) is None
