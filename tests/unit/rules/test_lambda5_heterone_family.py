"""The λ⁵-heterone family — what is BUILT, enumerated against the Blue Book.

v29 residue Task R. The residue item recorded "λ⁵-heterones unnameable".
**The premise is stale.** Enumerating the class against every worked (PIN)
example the Blue Book prints for it found the mono-oxo heterone family complete
and byte-exact, including the λ-convention, the P-16.5.1.3 enclosing marks for
three different substituents, and the P-74.2.1.5 heterimine next door. The
producer is ``rules/mononuclear_hydrides.py::name_heterone``, dispatched from
``routing/dispatch_table.py``'s ``HETERONE`` entry.

**Task R2 (2026-08-03) closed the DIONE half.** The ``-PO2``/``-AsO2`` members
were blocked by ``name_heterone``'s ``len(doubles) != 1``, verified on the
execution path by a line-level trace (line 564, the ``return None`` under that
guard, was the last line executed for both targets, against line 588 — the
success return — for the mono-oxo control). The guard is now a *window*
``1 <= len(doubles) <= 2`` plus a terminal-oxygen requirement, and the oxo count
selects the stem. Derivation and evidence:
``.

One member remains deliberately unbuilt, and it is a **spelling** gap, not a
perception gap:

  * the **chalcogen analogue** ``R3P=S`` / ``R-PS2`` — the Blue Book prints no
    worked example of a thione suffix on a phosphane stem, so its spelling
    would have to be invented. It stays fail-closed through the terminal-oxygen
    check, and that refusal is ASSERTED below so it cannot be widened by
    accident.

The refusals are asserted at **producer** level (``name_heterone(mol) is None``)
rather than through the CLI, because ``conftest`` disables the OPSIN gate
suite-wide: a whole-pipeline "must stay refused" assertion is gate-dependent and
would silently measure the gate instead of this guard.

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
* **P-61.6 "HETERONES"** body, ``:25977``: *"Compounds containing the –PO,
  –PO2, –AsO or –AsO2 are called heterones (see P-64.1.2.2, P-64.4). In the
  presence of a more senior characteristic group they are described by the
  compound prefixes oxophosphanyl, dioxo-λ⁵-phosphanyl, oxoarsanyl, and
  dioxo-λ⁵-arsanyl."* — four groups, four prefixes, 1:1. This is the sentence
  that makes the dione a member of the same class as the mono-oxo heterone, and
  that puts ``–AsO2`` in it alongside ``–PO2``.
* **P-64.1.2.2 "Heterones"** body, ``:28281``: *"Heterones are compounds having
  an oxygen atom formally doubly bonded to a heteroatom ... They are named in
  the same way as ketones except when expressed as compulsory prefixes"* — the
  clause that supplies the multiplied ``-dione`` suffix.
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.rules.mononuclear_hydrides import (
    _HETERONE_DIONE_STEMS,
    name_heterone,
)


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# (SMILES, expected PIN, Blue Book provenance)
BUILT_HETERONES = [
    (
        "O=P(c1ccccc1)(c1ccccc1)c1ccccc1",
        "triphenyl-λ5-phosphanone",
        "P-74.2.1.4 :43054 (PIN); repeated :29410 and :39129, the latter "
        "adding '(not oxotriphenyl-λ5-phosphane)'.",
    ),
    (
        "CP(C)(C)=O",
        "trimethyl-λ5-phosphanone",
        "BB :7023 index entry 'trimethylphosphane oxide / trimethyl-λ5-"
        "phosphanone (PIN, P-68.3.2.3.1, P-74.2.1.4)'.",
    ),
    (
        "O=Pc1ccccc1",
        "phenylphosphanone",
        "P-64.1.2.2 :28291 (PIN) '(not phosphorosobenzene)'; repeated :39125. "
        "BOUNDARY: lambda-3 phosphorus, so NO lambda descriptor is emitted -- "
        "the pair with triphenyl-λ5-phosphanone proves the lambda is "
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
        "methyl(phenyl)(propyl)-λ5-phosphanone",
        "BB :46006 '(S)-methyl(phenyl)(propyl)-λ5-phosphanone (PIN; see "
        "P-74.2.1.4)'. BOUNDARY: three DIFFERENT substituents, so P-16.5.1.3 "
        "enclosing marks apply -- first cited unmarked, rest parenthesised.",
    ),
    (
        "C[As](C)(C)=O",
        "trimethyl-λ5-arsanone",
        "P-74.2.1.4 :43057 'These methods are also applied to arsine and "
        "stibine oxides, sulfides, etc.'",
    ),
    (
        "C[Ge](C)=O",
        "dimethylgermanone",
        "P-21.2.2 germane series + the P-61.6/P-64.1.2.2 heterone suffix; the "
        "fourth hub element the producer supports.",
    ),
    # --- the DIONE half, shipped by Task R2 ---
    (
        "O=P(=O)c1ccccc1",
        "phenyl-λ5-phosphanedione",
        "P-61.6 :25983 VERBATIM (PIN): 'phenyl-λ5-phosphanedione (PIN) "
        "dioxo(phenyl)-λ5-phosphane (not phosphobenzene)'.",
    ),
    (
        "CP(=O)=O",
        "methyl-λ5-phosphanedione",
        "P-64.1.2.2 :28287 VERBATIM (PIN): 'CH3-PO2 methyl-λ5-"
        "phosphanedione (PIN) methyldi(oxo)-λ5-phosphane (not "
        "phosphomethane)'.",
    ),
    (
        "O=[As](=O)c1ccccc1",
        "phenyl-λ5-arsanedione",
        "DERIVED, not printed verbatim -- 'arsanedione' has 0 hits in the Blue "
        "Book. P-61.6 :25977 declares -AsO2 a heterone in the same sentence as "
        "-PO2 and prints its preselected prefix dioxo-λ5-arsanyl (also the "
        "prefix table :55895, structure O2As-, source P-61.6); P-64.1.2.2 "
        ":28281 gives the suffix ('named in the same way as ketones'); the "
        "arsane stem, its λ5 form and the mononuclear -one/-dione suffix "
        "are each printed (arsanone PIN :25985, trimethyl-λ5-arsanone "
        ":43057, phosphanedione PIN x2). Elision follows phosphane+dione. OPSIN "
        "2.9.0 parses it to the exact input structure (validity, not PIN "
        "authority).",
    ),
    (
        "C[As](=O)=O",
        "methyl-λ5-arsanedione",
        "Same derivation as the phenyl arsanedione above; the methyl member is "
        "the exact -AsO2 analogue of the printed CH3-PO2 PIN at :28287.",
    ),
]


# (SMILES, why it MUST stay refused). Asserted at producer level: the OPSIN gate
# is off under pytest, so a CLI-level refusal assertion would measure the gate.
FAIL_CLOSED = [
    (
        "S=P(c1ccccc1)(c1ccccc1)c1ccccc1",
        "P=S chalcogen heterone. P-74.2.1.4 :43043-:43049 does make method (3) "
        "the PIN for phosphine sulfides, but the Blue Book prints NO worked "
        "example of a thione suffix on a phosphane stem ('phosphanethione' has "
        "0 hits), so the spelling would have to be invented. Deliberately not "
        "built -- see ",
    ),
    (
        "O=P(=S)c1ccccc1",
        "Mixed =O/=S on one hub: the dione widening must not admit it just "
        "because the double-bond COUNT is now 2. This is the row that pins the "
        "terminal-oxygen requirement as separate from the count window.",
    ),
    (
        "C=P(=O)c1ccccc1",
        "A =C partner. Count is 2, so only the terminal-oxygen check refuses "
        "it; a widening that only counted doubles would emit a name here.",
    ),
    (
        "CP(=O)=N",
        "A =N partner (phosphanimine territory, P-74.2.1.5). Same trap as the "
        "=C row.",
    ),
    (
        "OP(=O)=O",
        "Metaphosphoric acid HO-PO2: two terminal oxo groups AND a P-OH, so it "
        "passes the oxo census and is refused only by the organyl purity loop. "
        "Inorganic (P-67 acid), never a 'hydroxy-λ5-phosphanedione'.",
    ),
    (
        "COP(=O)=O",
        "Methyl metaphosphate: the hub's third neighbour is an ESTER oxygen, "
        "not an organyl. Same purity-loop refusal as the acid.",
    ),
    (
        "O=P(=O)OP(=O)=O",
        "Two hubs -- refused by the pre-existing len(hubs) != 1, upstream of "
        "the oxo census.",
    ),
    (
        "O=P(=O)CCC(=O)O",
        "A carboxylic acid on the organyl. The COOH is senior, so the heterone "
        "must NOT capture the molecule as a dione parent; P-61.6 :25977 says "
        "that in the presence of a more senior characteristic group the group "
        "is cited by the prefix dioxo-λ5-phosphanyl instead.",
    ),
]


def _norm(name):
    """The CLI renders the lambda as 'λ5'; normalise any alternate rendering
    (superscript unicode or the retired ASCII 'lambda5' spelling) to it."""
    return (
        name.replace("λ⁵", "λ5")
        .replace("lambda5", "λ5")
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
    'λ5' onto every phosphanone would pass every row above except this one.
    """
    assert "λ5" not in _norm(namer.name("O=Pc1ccccc1"))
    assert "λ5" in _norm(namer.name("O=P(c1ccccc1)(c1ccccc1)c1ccccc1"))


@pytest.mark.parametrize(
    "smiles,reason", FAIL_CLOSED, ids=[s for s, _ in FAIL_CLOSED]
)
def test_heterone_stays_fail_closed(smiles, reason):
    """The dione widening must not admit anything but a TERMINAL oxygen.

    Four of these rows have a double-bond count of exactly 2, so a widening that
    only relaxed the count would emit a name for them.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"fixture SMILES itself is invalid: {smiles}"
    assert name_heterone(mol) is None, reason


def test_lambda_is_computed_for_the_dione_too(namer):
    """P-31.1.4.2 -- the λ⁵ on the dione comes from the hub's bonding number
    (2 oxo double bonds + 1 organyl = 5), not from the suffix.

    Paired with ``phenylphosphanone`` (λ³, no descriptor) this shows the same
    computation drives both oxo counts.
    """
    assert _norm(namer.name("CP(=O)=O")) == "methyl-λ5-phosphanedione"
    assert "λ" not in _norm(namer.name("O=Pc1ccccc1"))


def test_the_dione_element_gate_is_the_bluebook_class_not_the_hub_list():
    """P-61.6 :25977 puts exactly ``-PO``, ``-PO2``, ``-AsO``, ``-AsO2`` in the
    heterone class, so the DIONE gate is narrower than the mono-oxo hub list:
    Si and Ge take ``-one`` but never ``-dione``.

    The gate can be this narrow safely because a two-oxo Si or Ge hub is not a
    molecule at all -- RDKit rejects the valence outright, which is asserted
    here so the reason survives rather than being re-derived.
    """
    assert set(_HETERONE_DIONE_STEMS) == {"P", "As"}
    for impossible in ("C[Si](=O)=O", "O=[SiH2]=O", "O=[Ge](=O)C"):
        assert Chem.MolFromSmiles(impossible) is None, impossible


def test_a_three_oxo_hub_is_outside_the_count_window():
    """The guard is a WINDOW (1..2), not ``>= 1``. Nothing in P-61.6's class has
    three oxo groups on one mononuclear hub, so a third must fail closed.

    Built with an explicit RWMol because such a hub has no valid SMILES -- which
    is itself the point: the window's upper bound is the fail-closed edge.
    """
    rw = Chem.RWMol()
    p = rw.AddAtom(Chem.Atom(15))
    for _ in range(3):
        o = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(p, o, Chem.BondType.DOUBLE)
    c = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(p, c, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    assert name_heterone(mol) is None


def test_p74_2_1_5_heterimine_also_emits(namer):
    """P-74.2.1.5 "Phosphine imides" (heading :43061), :43069 'Method (3) leads
    to preferred IUPAC names'; worked (PIN) example :43075.

    The =NH/=NR sibling of the heterone, included because Task R's premise
    covered the λ⁵ parent hydride generally. It works, with the P,P,P- and N-
    italicised locants.
    """
    got = _norm(namer.name("CCN=P(c1ccccc1)(c1ccccc1)c1ccccc1"))
    assert got == "N-ethyl-P,P,P-triphenyl-λ5-phosphanimine"
