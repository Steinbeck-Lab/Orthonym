"""``caprolactam`` is a non-PIN trivial name reaching the DEFAULT ``--style pin``
path, and it is the only break in an otherwise systematic five-member series:

    O=C1CCN1 azetidin-2-one O=C1CCCCCN1 caprolactam <-- the break
    O=C1CCCN1 pyrrolidin-2-one O=C1CCCCCCN1 azocan-2-one
    O=C1CCCCN1 piperidin-2-one

Blue Book basis (each anchor opened and quoted, not relayed):

* "PSEUDOKETONES" (the Blue Book) -> **** (:29314), verbatim:
  "Cyclic anhydrides, esters and amides are named as pseudoketones; the resulting
  names are preferred IUPAC names." Its own example list prints, at:29323:
  ``azepan-2-one (PIN) hexano-6-lactam (see ``.
* "Lactams and lactims" (:33219). Lactams "are named in two ways":
  (1) "as heterocyclic pseudoketones" (:33223); (2) the ``...o-N-lactam`` form
  (:33224). The decisive sentence is the last one,:33226 -- "Method (1)
  generates preferred IUPAC names." So even ``hexano-6-lactam`` is only the
  non-preferred alternative, and ``caprolactam`` is neither method.
* ``caprolactam`` has **0** occurrences in the Blue Book. The grep methodology
  was validated against known positives first (``succinimide`` 2, ``hexano-6-
  lactam`` 1, ``lactam`` 14, ``azepan`` 10), and all 14 ``lactam`` lines were
  enumerated by hand -- the name is in none of them. Fragment forms
  (``capro``, ``caprol``, ``prolactam``, ``aprolactam``) are 0 too, so OCR
  splitting and italic markup are excluded as false-negative sources.
* Supporting, (:53745): "The use of Greek letters to indicate
  the size of a lactone or lactam ring is not recommended." ``caprolactam`` is
  the contracted form of epsilon-caprolactam.

⚠ The pre-existing inline comment at ``retained_names.py:579`` cited "
retained lactam name". (heading:16619) is "Bi- and polycyclic von
Baeyer parent hydrides" and says nothing about lactams or retained names; that
citation was wrong and is corrected by this change.

Producer-level assertions only. Whole-molecule assertions are unsound in this
suite -- ``conftest`` disables the OPSIN gate suite-wide, so ``name_tiered`` can
select a different producer than the CLI does. User-visible CLI behaviour is
recorded in ``.the workflow tooling/sdd/-residue/TaskAB-report.md``.
"""

import json
from pathlib import Path

import pytest

CAPROLACTAM_SMILES = "O=C1CCCCCN1"


def _pin_list():
    import orthonym.data as data_pkg

    path = Path(data_pkg.__file__).parent / "iupac_2013_pin_list.json"
    with open(path) as fh:
        return json.load(fh)


def test_caprolactam_is_adjudicated_non_pin():
    """The curated list -- the project's sole PIN authority -- must deny it."""
    from orthonym.data.pin_policy import is_pin_denied

    assert is_pin_denied("caprolactam") is True


def test_caprolactam_row_carries_a_citation_and_replacement():
    rows = [e for e in _pin_list()["entries"] if e["name"].lower() == "caprolactam"]
    assert len(rows) == 1, "exactly one adjudication row expected"
    row = rows[0]
    assert row["pin"] is False
    assert row["citation"], "a deny row without a Blue Book citation is not adjudicated"
    assert "azepan-2-one" in row["note"], "the verified replacement must be recorded"


def test_caprolactam_is_withdrawn_from_the_pin_surface():
    import orthonym.data as data_pkg

    assert data_pkg.ALL_RETAINED_NAMES.get(CAPROLACTAM_SMILES) != "caprolactam"
    assert "caprolactam" not in set(data_pkg.ALL_RETAINED_NAMES.values())


def test_caprolactam_is_demoted_not_deleted():
    """the contributor guide: deny rows are DEMOTED. The name must survive on the
    general-only companion surface, exactly as glycerol and catechol do."""
    import orthonym.data as data_pkg

    assert data_pkg.GENERAL_RETAINED_NAMES.get(CAPROLACTAM_SMILES) == "caprolactam"


def test_systematic_producer_supplies_the_pin():
    """the contributor guide a project rule -- removing a wrong output must not unmask a worse
    generator. The seven-membered ring's PIN must come from the same producer
    that already serves its four siblings."""
    from orthonym.rules.lactams import name_lactam_ring

    assert name_lactam_ring(7) == "azepan-2-one"


@pytest.mark.parametrize(
    "ring_size,expected",
    [(4, "azetidin-2-one"), (5, "pyrrolidin-2-one"),
     (6, "piperidin-2-one"), (8, "azocan-2-one")],
)
def test_sibling_ring_sizes_are_untouched(ring_size, expected):
    """The regression fence: the four ring sizes that were already correct."""
    from orthonym.rules.lactams import name_lactam_ring

    assert name_lactam_ring(ring_size) == expected


def test_no_ring_size_emits_the_non_preferred_lactam_suffix_form():
    """ method (2) (``hexano-6-lactam``) is NOT preferred --:33226
    "Method (1) generates preferred IUPAC names." The producer must never build
    that form at ANY ring size.

    Asserted on OUTPUT, not on source text. ``data/opsin_imports/suffix_rules.py``
    does register 'lactam' as a suffix morpheme at:81/:205/:371, but those are
    mirrored OPSIN *parsing* tables (name -> structure) whose only consumer is
    ``validation/name_morphemes.py``; they generate nothing.
    """
    from orthonym.rules.lactams import name_lactam_ring

    emitted = {n: name_lactam_ring(n) for n in range(3, 31)}
    offenders = {n: v for n, v in emitted.items() if v and "lactam" in v}
    assert offenders == {}, f"method-(2) lactam form emitted: {offenders}"
