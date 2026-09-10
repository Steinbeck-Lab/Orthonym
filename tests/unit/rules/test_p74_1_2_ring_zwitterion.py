""" — ring zwitterions carrying an anionic characteristic group.

Task G2 (residue). `rules/ions.py::emit_zwitterion_ring_carboxylate` used to
splice a cation-substituent prefix built in ONE ring numbering onto a *substituted*
ring name carrying its own, DIFFERENT numbering. On the default path that emitted
malformed and mis-numbered names — and every one of them round-tripped cleanly
through OPSIN, so neither the round-trip metric nor could see them. The
report with the full measurement is
internal notes.

Blue Book, `the Blue Book Blue Book` (every pointer re-opened with `sed -n '<N>p'`
at write time):

* ** "Zwitterionic compounds with at least one ionic center on a
  characteristic group"** — heading `:42445`. Sentence `:42447`, quoted whole
  because the decisive clause is the last one: *"Zwitterionic compounds with at
  least one ionic center on a characteristic group may be named by adding the
  appropriate ionic suffix to the name of the ionic parent hydride. In names,
  cationic suffixes are cited before anionic suffixes. For assignment of lower
  locants, ionic centers on skeletal atoms of the parent hydride are preferred to
  the locants for positions of attachment of characteristic groups denoted by
  ionic suffixes."*
* Worked (PIN) example `:42456`: `1-methyl-4,6-diphenylpyridin-1-ium-2-carboxylate`.
* Where detachable prefixes rank: ** "NUMBERING"** heading `:3219`,
  clause (c) `:3256` *"principal characteristic groups and free valences
  (suffixes);"* vs clause (f) `:3301` *"detachable alphabetized prefixes, all
  considered together in a series of increasing numerical order;"*.

So the numbering cascade is: heteroatoms -> indicated hydrogen -> skeletal ionic
centre -> attachment of the anionic-suffix group -> detachable prefixes.
"""

import re

import pytest
from rdkit import Chem, RDLogger

from orthonym.namer import Orthonym
from orthonym.perception.rings import get_ring_systems
from orthonym.rules import ions

RDLogger.DisableLog("rdApp.*")


@pytest.fixture(scope="module")
def namer():
    return Orthonym(style="pin")


def _charge_sites(mol):
    cation = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() > 0]
    anion = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() < 0]
    return cation, anion


def _emit(smiles):
    """Call the emitter directly. Gate-independent: `tests/conftest.py` disables the
    OPSIN validity gate suite-wide, so a whole-pipeline assertion cannot prove a
    guard fired rather than the gate."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    cations, anions = _charge_sites(mol)
    assert len(cations) >= 1 and len(anions) >= 1, smiles
    return ions.emit_zwitterion_ring_carboxylate(mol, cations[0], anions[0])


# --------------------------------------------------------------------------
# 1. The Blue Book's own worked (PIN) example, and the rows that already worked
# --------------------------------------------------------------------------

BB_PIN_ROWS = [
    # BB:42456 — the worked (PIN) example. Abstained before Task G2.
    ("C[n+]1c(C(=O)[O-])cc(-c2ccccc2)cc1-c1ccccc1",
     "1-methyl-4,6-diphenylpyridin-1-ium-2-carboxylate",
     "BB :42456 (PIN)"),
]

# Rows that were already correct before Task G2 and MUST stay byte-exact. The
# first two are gold-oracle rows (benchmarks/the gold set/packs/
# characteristic_groups.json).
PRE_EXISTING_ROWS = [
    ("[O-]C(=O)c1ccc[nH+]c1", "pyridin-1-ium-3-carboxylate"),
    ("C[n+]1ccccc1C(=O)[O-]", "1-methylpyridin-1-ium-2-carboxylate"),
    ("O=C([O-])c1cccc[nH+]1", "pyridin-1-ium-2-carboxylate"),
    ("[O-]C(=O)c1cc[nH+]cc1", "pyridin-1-ium-4-carboxylate"),
    ("CC[n+]1ccccc1C(=O)[O-]", "1-ethylpyridin-1-ium-2-carboxylate"),
    ("C[n+]1cccc(c1)C(=O)[O-]", "1-methylpyridin-1-ium-3-carboxylate"),
    ("C[n+]1ccc(cc1)C(=O)[O-]", "1-methylpyridin-1-ium-4-carboxylate"),
    ("[nH+]1ccc(C)cc1C(=O)[O-]", "4-methylpyridin-1-ium-2-carboxylate"),
    ("Cc1cc[nH+]cc1C(=O)[O-]", "4-methylpyridin-1-ium-3-carboxylate"),
]

# Rows whose HEAD output was MALFORMED or mis-numbered. The comment on each row is
# the exact string HEAD emitted, so a future regression is recognisable on sight.
REPAIRED_ROWS = [
    # HEAD: 1-methyl4-methylpyridin-1-ium-2-carboxylate (no hyphen; prefixes unmerged)
    ("C[n+]1ccc(C)cc1C(=O)[O-]", "1,4-dimethylpyridin-1-ium-2-carboxylate"),
    ("C[n+]1c(C(=O)[O-])cc(C)cc1", "1,4-dimethylpyridin-1-ium-2-carboxylate"),
    # HEAD: 1-methyl3-methylpyridin-1-ium-2-carboxylate
    ("C[n+]1c(C(=O)[O-])c(C)ccc1", "1,3-dimethylpyridin-1-ium-2-carboxylate"),
    # HEAD: 1-methyl4-bromopyridin-1-ium-2-carboxylate: bromo < methyl)
    ("C[n+]1ccc(Br)cc1C(=O)[O-]", "4-bromo-1-methylpyridin-1-ium-2-carboxylate"),
    ("C[n+]1ccc(Cl)cc1C(=O)[O-]", "4-chloro-1-methylpyridin-1-ium-2-carboxylate"),
    ("C[n+]1ccc(F)cc1C(=O)[O-]", "4-fluoro-1-methylpyridin-1-ium-2-carboxylate"),
    # HEAD: 1-methyl2,4-dimethylpyridin-1-ium-6-carboxylate
    # — carboxylate numbered 6 where:42447 requires 2
    ("C[n+]1c(C(=O)[O-])cc(C)cc1C", "1,4,6-trimethylpyridin-1-ium-2-carboxylate"),
    # HEAD: 1-methyl2-ethyl-4-methylpyridin-1-ium-6-carboxylate
    ("C[n+]1c(C(=O)[O-])cc(C)cc1CC",
     "6-ethyl-1,4-dimethylpyridin-1-ium-2-carboxylate"),
    # HEAD: 1-methyl4-(propan-2-yl)pyridin-1-ium-2-carboxylate
    ("C[n+]1c(C(=O)[O-])cc(C(C)C)cc1",
     "1-methyl-4-(propan-2-yl)pyridin-1-ium-2-carboxylate"),
    ("C[n+]1ccc(C(F)(F)F)cc1C(=O)[O-]",
     "1-methyl-4-(trifluoromethyl)pyridin-1-ium-2-carboxylate"),
]

# Rows HEAD abstained on. The isopropyl row is the important one: HEAD produced
# `1-propylpyridin-1-ium-2-carboxylate` — a DIFFERENT molecule — because the old
# branch named substituents with the carbon-counting `classify_substituent`, and
# only stopped it shipping.
NEWLY_NAMED_ROWS = [
    ("C[n+]1c(C(=O)[O-])ccc(C)c1", "1,5-dimethylpyridin-1-ium-2-carboxylate"),
    ("C[n+]1ccc(O)cc1C(=O)[O-]", "4-hydroxy-1-methylpyridin-1-ium-2-carboxylate"),
    ("C[n+]1ccc(N)cc1C(=O)[O-]", "4-amino-1-methylpyridin-1-ium-2-carboxylate"),
    ("C[n+]1ccc(C#N)cc1C(=O)[O-]", "4-cyano-1-methylpyridin-1-ium-2-carboxylate"),
    ("C[n+]1ccc(-c2ccccc2)cc1C(=O)[O-]",
     "1-methyl-4-phenylpyridin-1-ium-2-carboxylate"),
    ("CC(C)[n+]1ccccc1C(=O)[O-]",
     "1-(propan-2-yl)pyridin-1-ium-2-carboxylate"),
]

ALL_NAMED_ROWS = (
    [(s, e) for s, e, _ in BB_PIN_ROWS]
    + PRE_EXISTING_ROWS + REPAIRED_ROWS + NEWLY_NAMED_ROWS
)


@pytest.mark.parametrize("smiles,expected,provenance", BB_PIN_ROWS)
def test_blue_book_worked_pin(namer, smiles, expected, provenance):
    assert namer.name(smiles) == expected, provenance


@pytest.mark.parametrize("smiles,expected", PRE_EXISTING_ROWS)
def test_rows_correct_before_task_g2_are_unchanged(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", REPAIRED_ROWS)
def test_malformed_splice_is_repaired(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", NEWLY_NAMED_ROWS)
def test_newly_named_rows(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --------------------------------------------------------------------------
# 2. Structural tripwire over EVERY emitted row of the class
# --------------------------------------------------------------------------

# The malformed-splice signature: a letter immediately followed by a digit, which
# is what `1-methyl` + `4-methyl` -> `1-methyl4-methyl` produced. Computed over ALL
# emitted rows and not just the failures (CLAUDE.md #14), so it cannot be a
# signature that also matches passing rows.
_WELDED_LOCANT = re.compile(r"[a-z]\d")


def test_no_emitted_name_welds_a_locant_onto_a_prefix(namer):
    offenders = []
    for smiles, _expected in ALL_NAMED_ROWS:
        name = namer.name(smiles)
        # an indicated hydrogen legitimately puts a digit after a letter ('1H-')
        probe = re.sub(r"\d+H-", "", name)
        if _WELDED_LOCANT.search(probe):
            offenders.append((smiles, name))
    assert not offenders, (
        "a prefix locant is welded to the preceding prefix -- the two-numbering "
        f"splice is back: {offenders}"
    )
    assert len(ALL_NAMED_ROWS) >= 26


def test_every_emitted_name_cites_cation_before_anion(namer):
    """:42447 'In names, cationic suffixes are cited before anionic
    suffixes.' Spelling-independent: the '-ium' must precede the '-carboxylate'."""
    checked = 0
    for smiles, _expected in ALL_NAMED_ROWS:
        name = namer.name(smiles)
        assert "-ium-" in name and name.endswith("-carboxylate"), name
        assert name.index("-ium-") < name.index("-carboxylate"), name
        checked += 1
    assert checked == len(ALL_NAMED_ROWS)


# --------------------------------------------------------------------------
# 3. The numbering criterion is LOAD-BEARING, not decorative
# --------------------------------------------------------------------------

def test_substituent_set_cannot_decide_the_bb_example():
    """The reason the anion-attachment criterion is needed at all.

    For BB:42456 both numbering directions put substituents at {1,2,4,6}, so
    criterion (e) is blind and, without the new criterion, an arbitrary
    canonical-rank tiebreak picks the direction. This test asserts the
    degeneracy, so if someone deletes the criterion believing (e) covers it, the
    premise is on record as false."""
    mol = Chem.MolFromSmiles("C[n+]1c(C(=O)[O-])cc(-c2ccccc2)cc1-c1ccccc1")
    cation = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() > 0][0]
    ring = next(rs for rs in get_ring_systems(mol, include_spiro=True)
                if cation in rs)
    cyclic = ions._order_ring_cycle(mol, ring)
    assert cyclic is not None
    ring_set = set(ring)

    def substituted_positions(order):
        loc = {idx: k + 1 for k, idx in enumerate(order)}
        return tuple(sorted(
            loc[i] for i in order
            if any(nb.GetIdx() not in ring_set and nb.GetSymbol() != "H"
                   for nb in mol.GetAtomWithIdx(i).GetNeighbors())))

    n = len(cyclic)
    start = cyclic.index(cation)
    forward = [cyclic[(start + i) % n] for i in range(n)]
    backward = [cyclic[(start - i) % n] for i in range(n)]
    assert substituted_positions(forward) == substituted_positions(backward), (
        "the two directions are NOT degenerate -- this test's premise is stale")


def test_dropping_the_anion_criterion_changes_the_name():
    """Mutation check: with `anion_attach_idx` withheld, the numbering is no longer
    forced by. Proves the criterion is on the execution path for the Blue
    Book example rather than merely present."""
    smiles = "C[n+]1c(C(=O)[O-])cc(C)cc1C"   # -> 1,4,6-trimethyl...-2-carboxylate
    mol = Chem.MolFromSmiles(smiles)
    cation = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() > 0][0]
    anion = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() < 0][0]
    ring = next(rs for rs in get_ring_systems(mol, include_spiro=True)
                if cation in rs)
    _carboxyl_c, ring_attach = ions._carboxyl_ring_anchor(mol, anion, ring)

    with_rule, _ = ions._charged_ring_locants(
        mol, ring, cation, anion_attach_idx=ring_attach)
    without_rule, _ = ions._charged_ring_locants(mol, ring, cation)

    assert with_rule[ring_attach] == 2, (
        "P-74.1.2 :42447 gives the anionic-suffix attachment the lower locant")
    assert without_rule[ring_attach] != with_rule[ring_attach], (
        "withholding the criterion changed nothing -- the test is vacuous and the "
        "criterion may be unreachable")


# --------------------------------------------------------------------------
# 4. Fail-closed boundaries (asserted at the emitter, so the gate cannot mask them)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,why", [
    ("C[n+]1ccc(OC)cc1C(=O)[O-]",
     "-OCH3: name_substituent spells a multi-atom heteroatom-rooted fragment "
     "'hydroxymethyl' (a different group), so it must not be trusted here"),
    ("C[n+]1ccc(SC)cc1C(=O)[O-]",
     "-SCH3: same primitive defect, spelled 'thiomethyl'"),
    ("C[n+]1ccc([N+](=O)[O-])cc1C(=O)[O-]",
     "nitro carries formal charges; a third charged atom is a further ionic "
     "centre and P-74 wants it as a suffix, not swept into a prefix"),
    ("C[n+]1ccc(N(C)C)cc1C(=O)[O-]",
     "-N(CH3)2: multi-atom heteroatom-rooted fragment"),
])
def test_uncorroborated_substituents_fail_closed(smiles, why):
    assert _emit(smiles) == "", why


def test_emitter_declines_when_the_cation_is_not_in_a_ring():
    """A quaternary-ammonium acetate is (centres on DIFFERENT parent
    structures), not this emitter's case."""
    assert _emit("C[N+](C)(C)CC(=O)[O-]") == ""
