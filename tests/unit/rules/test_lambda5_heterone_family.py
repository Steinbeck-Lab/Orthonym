"""The λ⁵-heterone family — what is BUILT, enumerated against the Blue Book.

v29 residue Task R. The residue item recorded "λ⁵-heterones unnameable".
**The premise is stale.** Enumerating the class against every worked (PIN)
example the Blue Book prints for it found the mono-oxo heterone family complete
and byte-exact, including the λ-convention, the P-16.5.1.3 enclosing marks for
three different substituents, and the P-74.2.1.5 heterimine next door. The
producer is ``rules/mononuclear_hydrides.py::name_heterone``, dispatched from
``routing/dispatch_table.py``'s ``HETERONE`` entry.

Two members of the class ARE still unnameable, and both are recorded in
`` with their exact blocking guard:

  * the **λ⁵-heterone DIONE** (two ``=O`` on one hub) — ``phenyl-λ⁵-
    phosphanedione`` (``:25983``) and ``methyl-λ⁵-phosphanedione``
    (``:28287``), both blocked by ``name_heterone``'s ``len(doubles) != 1``;
  * the **chalcogen analogue** ``R3P=S`` — blocked by the same function's
    ``oxo.GetSymbol() != 'O'``.

They are NOT asserted here (not even as xfail): ``mononuclear_hydrides.py`` is
outside this task's file surface, so the gap is reported rather than fixed, and
a change-detector test would break whoever fixes it. This file pins the
WORKING half so that half cannot regress unnoticed.

Blue Book, ``BlueBookV2/BlueBookV2.md`` (every pointer re-opened with
``sed -n '<N>p'`` at write time):

* **P-74.2.1.4 "Phosphine oxides and chalcogen analogues"** (heading
  ``:43041``). Methods ``:43045-43047``; the decisive sentence is the one AFTER
  the list, ``:43049``: *"Method (3) leads to preferred IUPAC names."* — method
  (3) being ``:43047``: *"substitutively, as heterones, by using the suffix
  '-one' and λ⁵-phosphane as the parent hydride."*  Worked (PIN) example
  ``:43054``: ``triphenyl-λ⁵-phosphanone``. Closing sentence ``:43057``:
  *"These methods are also applied to arsine and stibine oxides, sulfides,
  etc."*
* **P-64.1.2.2 "Heterones"** (heading, above ``:28287``) — ``:28289``
  ``methylsilanone (PIN)``, ``:28291`` ``phenylphosphanone (PIN)``.
* **P-61.6 "HETERONES"** (heading, above ``:25983``).
* Definition, ``:1844``: *"**Heterone.** A compound having an oxygen atom
  doubly bonded to a heteroatom, for example methylsilanone."*
"""

import pytest

from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# (SMILES, expected PIN, Blue Book provenance)
BUILT_HETERONES = [
    (
        "O=P(c1ccccc1)(c1ccccc1)c1ccccc1",
        "triphenyl-lambda5-phosphanone",
        "P-74.2.1.4 :43054 (PIN); repeated :29410 and :39129, the latter "
        "adding '(not oxotriphenyl-lambda5-phosphane)'.",
    ),
    (
        "CP(C)(C)=O",
        "trimethyl-lambda5-phosphanone",
        "BB :7023 index entry 'trimethylphosphane oxide / trimethyl-lambda5-"
        "phosphanone (PIN, P-68.3.2.3.1, P-74.2.1.4)'.",
    ),
    (
        "O=Pc1ccccc1",
        "phenylphosphanone",
        "P-64.1.2.2 :28291 (PIN) '(not phosphorosobenzene)'; repeated :39125. "
        "BOUNDARY: lambda-3 phosphorus, so NO lambda descriptor is emitted -- "
        "the pair with triphenyl-lambda5-phosphanone proves the lambda is "
        "computed from the hub's bonding number, not pasted on.",
    ),
    (
        "C[Si](C)=O",
        "dimethylsilanone",
        "P-64.4.1 :29406 (PIN). Silicon hub, standard bonding number.",
    ),
    (
        "C[SiH]=O",
        "methylsilanone",
        "P-64.1.2.2 :28289 (PIN) 'methyl(oxo)silane' as the alternative. "
        "BOUNDARY: one organyl + one residual hub H.",
    ),
    (
        "CCC[P](C)(=O)c1ccccc1",
        "methyl(phenyl)(propyl)-lambda5-phosphanone",
        "BB :46006 '(S)-methyl(phenyl)(propyl)-lambda5-phosphanone (PIN; see "
        "P-74.2.1.4)'. BOUNDARY: three DIFFERENT substituents, so P-16.5.1.3 "
        "enclosing marks apply -- first cited unmarked, rest parenthesised.",
    ),
    (
        "C[As](C)(C)=O",
        "trimethyl-lambda5-arsanone",
        "P-74.2.1.4 :43057 'These methods are also applied to arsine and "
        "stibine oxides, sulfides, etc.'",
    ),
    (
        "C[Ge](C)=O",
        "dimethylgermanone",
        "P-21.2.2 germane series + the P-61.6/P-64.1.2.2 heterone suffix; the "
        "fourth hub element the producer supports.",
    ),
]


def _norm(name):
    """The CLI renders the lambda as 'lambda5'; normalise any unicode form."""
    return (
        name.replace("λ⁵", "lambda5")
        .replace("λ5", "lambda5")
        .replace("<sup>5</sup>", "5")
    )


@pytest.mark.parametrize(
    "smiles,expected,provenance",
    BUILT_HETERONES,
    ids=[s for s, _, _ in BUILT_HETERONES],
)
def test_heterone_family_matches_the_blue_book_pin(
    namer, smiles, expected, provenance
):
    """P-74.2.1.4 :43049 -- method (3) (substitutive, as a heterone) is the PIN."""
    assert _norm(namer.name(smiles)) == expected, provenance


def test_lambda_descriptor_tracks_the_hub_bonding_number(namer):
    """P-74.2.1.4 :43047 -- the parent hydride is λ⁵-phosphane only when the
    phosphorus actually carries bonding number 5.

    The two phosphanones differ ONLY in the hub's bonding number, so this pins
    the λ-convention itself rather than a spelling. A regression that hard-coded
    'lambda5' onto every phosphanone would pass every row above except this one.
    """
    assert "lambda5" not in _norm(namer.name("O=Pc1ccccc1"))
    assert "lambda5" in _norm(namer.name("O=P(c1ccccc1)(c1ccccc1)c1ccccc1"))


def test_p74_2_1_5_heterimine_also_emits(namer):
    """P-74.2.1.5 "Phosphine imides" (heading :43061), :43069 'Method (3) leads
    to preferred IUPAC names'; worked (PIN) example :43075.

    The =NH/=NR sibling of the heterone, included because Task R's premise
    covered the λ⁵ parent hydride generally. It works, with the P,P,P- and N-
    italicised locants.
    """
    got = _norm(namer.name("CCN=P(c1ccccc1)(c1ccccc1)c1ccccc1"))
    assert got == "N-ethyl-P,P,P-triphenyl-lambda5-phosphanimine"
