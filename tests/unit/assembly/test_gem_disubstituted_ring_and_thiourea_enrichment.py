"""Two independent coverage gaps, both measured at ``86e69737``.

    CC(=O)NC1CCCCC1        -> N-cyclohexylacetamide      works
    CC1(CCCCC1)NC(C)=O     -> unknown organic compound   GAP A
    NC(=O)NC1CCCCC1        -> N-cyclohexylurea           works
    NC(=S)NC1CCCCC1        -> unknown organic compound   GAP B

An inherited framing said *"no 1,1-disubstituted cycloalkyl substituent is
nameable in this tree"*.  That is REFUTED by measurement --
``(1-methylcyclohexyl)methanol``, ``1-methylcyclohexan-1-amine``,
``1,1-dimethylcyclohexane`` and ``1-methylcyclohexane-1-carboxylic acid`` all
name correctly, and so does ``N-(4,4-dimethylcyclohexyl)acetamide``.  The
measured boundary is far narrower and is pinned by
``test_boundary_gem_away_from_attachment_always_worked`` below:

    **gem-disubstitution AT THE ATTACHMENT CARBON of a ring substituent.**

GAP A root cause (``rules/ring_substituents.py`` ``decorated_ring_substituent_name``)
------------------------------------------------------------------------------
The function skipped every exocyclic neighbour of the attachment atom::

    parent_nbrs = {nbr.GetIdx() for nbr in
                   mol.GetAtomWithIdx(attachment_atom).GetNeighbors()
                   if nbr.GetIdx() not in ring_set}

Exactly one of those bonds is the free valence to the parent; the rest are
genuine decorations.  On a mono-substituted attachment carbon there is only
one, so skipping "all of them" and skipping "the parent bond" coincide and the
defect is invisible.  On a **gem-disubstituted** attachment carbon the ring's
own substituent is discarded with it, ``atom_prefixes`` comes back empty, and
the ``if not any(atom_prefixes.values())`` guard returns None.  The caller
(``rules/amides.py:_name_n_substituent``) then fell through to
``name_substituent``, which returned the unnameable sentinel ``'substituent'``,
and ``_enrich_ring_n_substituent`` prepended the very decoration that had been
dropped -- emitting ``N-(1-methylsubstituent)acetamide``.  Only the OPSIN
validity gate stopped it, which is why every assertion here is made at the
PRODUCER.

The fix identifies the parent bond STRUCTURALLY -- the exocyclic neighbour that
lies outside the fragment being named -- never by a count or an index.

GAP B root cause (``assembly/composer.py`` ``_enrich_handler_name``)
--------------------------------------------------------------------
``_try_name_thiourea`` already returned the correct ``N-cyclohexylthiourea``
(measured: 4 calls, 4 correct returns).  ``_enrich_handler_name`` then
re-discovered the handler's OWN, already-spelled thiourea core as a ring
substituent and prepended it, yielding
``1-(carbamothioylamino)N-cyclohexylthiourea`` -- the same unit spelled twice,
a different molecule, suppressed by SELF-01.  The exclusion list at
``composer.py:281`` is a hand-maintained tuple that had gone stale: it carries
``'urea'`` but not ``'thiourea'`` (nor ``cyanamide`` / ``imidate`` /
``chalcogen_ester``).  The fix derives the exclusion from the handler's own
``handler_id`` so the list cannot go stale again.

Blue Book, quotations verified by opening ``BlueBookV2/BlueBookV2.md``:

* **P-66.1.6.1.3 "Chalcogen analogues of urea and isourea"** (:33437),
  **P-66.1.6.1.3.1** (:33439): *"Chalcogen analogues of urea are named by
  functional replacement nomenclature using the prefixes 'thio', 'seleno', and
  'telluro'. Preferred IUPAC names use the letter locants N, and N'. Numerical
  locants may be used for thiourea in general nomenclature."*  Restated at
  :33446 *"Numerical locants are no longer used for thiourea in the IUPAC
  preferred name."*  => the ``1,3-``/``1-`` numeral spelling is general
  nomenclature, NOT the PIN.
* The mono-N-substituted class exemplar (:33451):
  **``N-(butan-2-yl)selenourea (PIN)``** -- a singly substituted chalcogen urea
  that KEEPS its ``N`` locant.  ``N-cyclohexylthiourea`` follows it directly.
* **P-29.2** free-valence morphology and **P-14.4** (:3221) criterion (c),
  free valence low, which fixes the attachment carbon of a carbocyclic
  substituent at locant 1 -- hence ``1-methylcyclohexyl``.

OPSIN proves the structure, never the spelling; every expected name below was
checked BOTH ways -- OPSIN 2.9.0 round-trip to an InChIKey identical to the
input's, AND against the governing rule above.
"""

import pytest
from rdkit import Chem


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ring_and_frag(smiles, smarts, ring_smarts=None):
    """Return (mol, ring_tuple, attach_idx, frag_set) for the N-substituent."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol


def _n_substituent_fragment(smiles):
    """(mol, ring, attach_idx, frag_set) for the ring borne on the amide/urea N.

    Located structurally: find the N that carries the ring, walk the fragment
    on the far side of that N-C bond.  No atom indices are hard-coded.
    """
    mol = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts("[NX3][CX4;R]")
    matches = mol.GetSubstructMatches(patt)
    assert matches, f"no N-C(ring) bond in {smiles}"
    n_idx, attach_idx = matches[0]
    # BFS the substituent fragment, boundary = the nitrogen
    seen, stack = {attach_idx}, [attach_idx]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            i = nb.GetIdx()
            if i == n_idx or i in seen or nb.GetAtomicNum() <= 1:
                continue
            seen.add(i)
            stack.append(i)
    ring = next(r for r in mol.GetRingInfo().AtomRings() if attach_idx in r)
    return mol, ring, attach_idx, seen


def _features_for(smiles, fn_name):
    """Capture the real MolecularFeatures handed to a composer builder."""
    import orthonym.assembly.composer as C
    from orthonym import Orthonym as _OS
    n_atoms = Chem.MolFromSmiles(smiles).GetNumAtoms()
    grabbed = []
    orig = getattr(C, fn_name)

    def spy(f):
        if getattr(f, "mol", None) is not None and f.mol.GetNumAtoms() == n_atoms:
            grabbed.append(f)
        return orig(f)

    setattr(C, fn_name, spy)
    try:
        _OS(style="pin").name_tiered(smiles)
    finally:
        setattr(C, fn_name, orig)
    return grabbed[0] if grabbed else None


@pytest.fixture(scope="module")
def namer():
    from orthonym import Orthonym
    return Orthonym(style="pin")


def _name(namer, smiles):
    return namer.name_tiered(smiles)["name"]


# ===========================================================================
# GAP A -- at the producer
# ===========================================================================

def test_gap_a_producer_names_the_gem_disubstituted_attachment_carbon():
    """PRODUCER-level.  ``decorated_ring_substituent_name`` returned None for a
    gem-disubstituted attachment carbon because it skipped the decoration along
    with the parent bond.  Asserted here, not on the shipped name, because the
    OPSIN gate suppressed the wrong output the old path produced.
    """
    from orthonym.rules.ring_substituents import decorated_ring_substituent_name
    mol, ring, attach, frag = _n_substituent_fragment("CC(=O)NC1(C)CCCCC1")
    got = decorated_ring_substituent_name(mol, ring, attach, expected_atoms=frag)
    assert got == "1-methylcyclohexyl", got


def test_gap_a_producer_covers_exactly_the_requested_fragment():
    """The decoration must be COVERED, not dropped: a name that silently omits
    the gem-methyl would describe cyclohexyl, a different molecule.  Proven by
    asking for a fragment that does NOT include the methyl -- it must refuse.
    """
    from orthonym.rules.ring_substituents import decorated_ring_substituent_name
    mol, ring, attach, frag = _n_substituent_fragment("CC(=O)NC1(C)CCCCC1")
    ring_only = set(ring)
    assert ring_only != frag
    assert decorated_ring_substituent_name(
        mol, ring, attach, expected_atoms=ring_only) is None


def test_gap_a_no_sentinel_ever_reaches_a_name():
    """The unnameable sentinel ``'substituent'`` must never be enriched into a
    name.  ``N-(1-methylsubstituent)acetamide`` was built by
    ``rules/amides.py:_name_n_substituent`` treating the sentinel as truthy.
    """
    from orthonym.rules.amides import _name_n_substituent
    mol, ring, attach, frag = _n_substituent_fragment("CC(=O)NC1(C)CCCCC1")
    ordered = [attach] + sorted(frag - {attach})
    got = _name_n_substituent(mol, ordered, sum(
        1 for i in frag if mol.GetAtomWithIdx(i).GetSymbol() == "C"))
    assert got is None or "substituent" not in got, got


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)NC1(C)CCCCC1",        "N-(1-methylcyclohexyl)acetamide"),
    ("CC1(CCCCC1)NC(C)=O",        "N-(1-methylcyclohexyl)acetamide"),
    ("CC(=O)NC1(C)CCCC1",         "N-(1-methylcyclopentyl)acetamide"),
    ("CC(=O)NC1(C)CCC1",          "N-(1-methylcyclobutyl)acetamide"),
    ("CC(=O)NC1(C)CC1",           "N-(1-methylcyclopropyl)acetamide"),
    ("CC(=O)NC1(C)CCCCCC1",       "N-(1-methylcycloheptyl)acetamide"),
    ("CC(=O)NC1(C)CCCCCCC1",      "N-(1-methylcyclooctyl)acetamide"),
    ("CCC1(CCCCC1)NC(C)=O",       "N-(1-ethylcyclohexyl)acetamide"),
])
def test_gap_a_family_ring_sizes_3_to_8(namer, smiles, expected):
    assert _name(namer, smiles) == expected


def test_boundary_gem_away_from_attachment_always_worked(namer):
    """The measured boundary.  gem-disubstitution AWAY from the attachment
    carbon was never broken -- this is what refutes the inherited
    "no 1,1-disubstituted cycloalkyl is nameable" framing.
    """
    assert _name(namer, "CC(=O)NC1CCC(C)(C)CC1") == \
        "N-(4,4-dimethylcyclohexyl)acetamide"


# ===========================================================================
# GAP B -- at the producer
# ===========================================================================

def test_gap_b_producer_already_worked():
    """``_try_name_thiourea`` was never the defect: it returns the right name.
    Pinning this makes it impossible to "fix" Gap B in the wrong module.
    """
    from orthonym.assembly.composer import _try_name_thiourea
    features = _features_for("NC(=S)NC1CCCCC1", "_try_name_thiourea")
    assert features is not None, "the thiourea builder was never reached"
    assert _try_name_thiourea(features) == "N-cyclohexylthiourea"


def test_gap_b_enrichment_must_not_respell_the_handlers_own_core():
    """PRODUCER-level.  ``_enrich_handler_name`` re-discovered the thiourea core
    the handler had already spelled and prepended it as
    ``1-(carbamothioylamino)``.  SELF-01 suppressed the result, so asserting on
    the shipped name would have judged the GATE, not the generator.
    """
    from orthonym.assembly.composer import _enrich_handler_name
    features = _features_for("NC(=S)NC1CCCCC1", "_try_name_thiourea")
    assert features is not None
    got = _enrich_handler_name(features, "N-cyclohexylthiourea", "thiourea")
    assert got == "N-cyclohexylthiourea", got


def test_gap_b_urea_sibling_enrichment_is_unchanged():
    """The oxo sibling was already correct and must stay byte-identical."""
    from orthonym.assembly.composer import _enrich_handler_name
    features = _features_for("NC(=O)NC1CCCCC1", "_try_name_urea")
    assert features is not None
    assert _enrich_handler_name(
        features, "N-cyclohexylurea", "urea") == "N-cyclohexylurea"


@pytest.mark.parametrize("smiles,expected", [
    ("NC(=S)NC1CCCCC1",   "N-cyclohexylthiourea"),
    ("NC(=S)NC1CCCC1",    "N-cyclopentylthiourea"),
    ("NC(=S)NC1CCC1",     "N-cyclobutylthiourea"),
    ("NC(=S)NC1CC1",      "N-cyclopropylthiourea"),
    ("NC(=S)NC1CCCCCC1",  "N-cycloheptylthiourea"),
    ("NC(=S)NC1CCCCCCC1", "N-cyclooctylthiourea"),
])
def test_gap_b_family_ring_sizes_3_to_8(namer, smiles, expected):
    assert _name(namer, smiles) == expected


def test_gap_b_letter_locant_not_a_numeral(namer):
    """P-66.1.6.1.3.1 (:33439/:33446): numerals are general nomenclature only."""
    got = _name(namer, "NC(=S)NC1CCCCC1")
    assert got.startswith("N-"), got
    assert not got[0].isdigit(), got


# ===========================================================================
# Controls -- must stay byte-identical
# ===========================================================================

@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)NC1CCCCC1",   "N-cyclohexylacetamide"),
    ("NC(=O)NC1CCCCC1",   "N-cyclohexylurea"),
    ("CC(=O)NC",          "N-methylacetamide"),
    ("CC(=O)NC(C)C",      "N-(propan-2-yl)acetamide"),
    ("NC(=S)N",           "thiourea"),
    ("CC1(C)CCCCC1",      "1,1-dimethylcyclohexane"),
])
def test_controls_unchanged(namer, smiles, expected):
    assert _name(namer, smiles) == expected
