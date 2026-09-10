"""Task L3-0 (a phase, class-b fix): ``parent_to_prefix``'s locanted-ketone
and locanted-amine detectors were anchored too narrowly and silently corrupted
connectivity for any unsaturated chain (an enone/enamine) whose locant sits
behind an ``-en-``/``-yn-`` infix.

Root cause (``src/orthonym/assembly/substituent_naming.py``):

* ``m_one`` (the "Locanted ketone: -an-N-one" branch) used
  ``re.search(r'an?-(\\d+)-one$', name)`` -- it only fires when a literal
  ``a`` (+ optional ``n``) sits immediately before the locant digit
  (``butan-2-one``). For an unsaturated chain like ``hept-2-en-4-one`` the
  character immediately before ``-4-one`` is the ``n`` of ``en``, not of
  ``an``, so the regex does not match.
* ``m_amine`` (the "Locanted amine: -an-N-amine" branch) has the identical
  ``an?-(\\d+)-amine$`` anchor and the identical blind spot.

When the narrow regex misses, execution falls through to the sibling
"Unlocanted" branch, which assumes NO locant is present (a
single-position stem) and silently splices ``oxo``/``amino`` onto the stem
with **no locant at all** -- dropping a real, load-bearing locant rather than
declining. Measured (``.venv/bin/python``, HEAD ``)::

    parent_to_prefix("2-methylhept-2-en-4-one", chain_length=7,
                      attach_locant=ATTACH_LOCANT_UNKNOWN)
    -> 'oxo-2-methylhept-2-en-4-yl' # the '4' locant silently dropped

This is not a cosmetic omission: OPSIN re-parses an un-cited oxo at the
lowest available position, which is a DIFFERENT molecule. Structural proof
(the ``TestCorruptionIsAReal WrongMolecule`` class below) uses the actual
composed name captured from a a trace trace of
``name_t4_complete("CC(C)=CC(=O)CC(C)C1CCC2C3C[C@H](O)[C@H]4C[C@@H](O)CCC4(C)C3=CCC12C",...)``,
which builds this exact fragment via the ``/19`` recursive fallback at
``substituent_enumerator.py`` (the ``_name_compound_substituent`` "Fallback:
recursive naming for ring-containing compound fragments" block, which calls
``parent_to_prefix(frag_name, chain_length=carbon_count,
attach_locant=ATTACH_LOCANT_UNKNOWN)`` unconditionally -- it never computes a
real attach locant at all).

The already-correct sibling pattern lives right above these two branches:
the "Locanted alcohol: -N-ol" branch uses the BROAD ``r'-(\\d+)-ol$'``
(documented, ``substituent_naming.py:3947-3952``, to match "saturated
(-an-N-ol), unsaturated (-en-N-ol, -yn-N-ol), and bare (-N-ol) patterns").
The fix widens ``m_one``/``m_amine`` to the same broad pattern, so an
unsaturated enone/enamine locant is DETECTED and declines (fail-closed) just
like every other borrowed-numbering branch, instead of being silently
dropped.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_naming import (
    ATTACH_LOCANT_UNKNOWN,
    parent_to_prefix,
)
from orthonym.assembly import t4_coverage
from orthonym.namer import Orthonym


# ---------------------------------------------------------------------------
# Repro molecule from the a trace trace: an enone side chain
# (-CH(CH3)-CH2-C(=O)-CH=C(CH3)-CH3) hanging off a decorated cyclopentanone
# ring. The whole molecule currently abstains for unrelated fused-ring
# reasons (measured), so this file targets the fragment-conversion PRIMITIVE
# directly, which is where the corruption actually happens and where the
# brief's "class b" defect is scoped.
# ---------------------------------------------------------------------------
_ENONE_MOLECULE_SMILES = (
    "CC(C)=CC(=O)CC(C)C1CCC2C3C[C@H](O)[C@H]4C[C@@H](O)CCC4(C)C3=CCC12C"
)


def _classified(smi):
    """Build ``(mol, features)`` exactly as ``namer.py``'s dispatch does
    (mirrors ``test_t4_coverage.py::_classified``)."""
    nm = Orthonym()
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    canonical = Chem.MolToSmiles(mol, canonical=True)
    feats = nm._perceive(mol, smi, canonical)
    nm._classify(feats)
    return mol, feats


# ===========================================================================
# 1. Direct root-cause reproduction -- FAILS before the fix, PASSES after.
# ===========================================================================
class TestEnoneAttachLocantFailsClosed:
    """``parent_to_prefix`` must decline an enone whose oxo locant is hidden
    behind an ``-en-`` infix, never fabricate a locant-free prefix."""

    def test_direct_repro_declines_rather_than_fabricates(self):
        """The exact fragment name observed corrupting the a trace's repro
        molecule. Before the fix this returned
        ``'oxo-2-methylhept-2-en-4-yl'`` -- the '4' silently dropped."""
        result = parent_to_prefix(
            "2-methylhept-2-en-4-one", chain_length=7,
            attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result is None, (
            f"expected a fail-closed decline (the '4' locant cannot be "
            f"silently dropped), got {result!r}"
        )

    def test_declines_even_with_a_proven_attach_locant(self):
        """A proven int attach_locant cannot rescue this branch either (same
        reasoning as the sibling -ol/-amine branches' docstring: the two
        numbering directions disagree, so the whole numbering -- not just
        the locant -- would need recomputing from the structure)."""
        result = parent_to_prefix(
            "2-methylhept-2-en-4-one", chain_length=7, attach_locant=5)
        assert result is None, f"expected fail-closed decline, got {result!r}"

    def test_generalizes_to_a_shorter_enone(self):
        """Not overfit to the 7-carbon repro: a minimal enone also declines."""
        result = parent_to_prefix(
            "pent-3-en-2-one", chain_length=5,
            attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result is None, f"expected fail-closed decline, got {result!r}"

    def test_generalizes_to_an_ynone(self):
        """The same blind spot exists for an alkyne infix (-yn-N-one)."""
        result = parent_to_prefix(
            "hex-3-yn-2-one", chain_length=6,
            attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result is None, f"expected fail-closed decline, got {result!r}"


class TestEnamineAttachLocantFailsClosed:
    """The identical narrow-anchor defect in the sibling ``m_amine`` branch
    (same regex shape, same blind spot for an unsaturation infix)."""

    def test_direct_repro_declines_rather_than_fabricates(self):
        result = parent_to_prefix(
            "hept-2-en-4-amine", chain_length=7,
            attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result is None, (
            f"expected a fail-closed decline (the '4' locant cannot be "
            f"silently dropped), got {result!r}"
        )

    def test_declines_even_with_a_proven_attach_locant(self):
        result = parent_to_prefix(
            "hept-2-en-4-amine", chain_length=7, attach_locant=5)
        assert result is None, f"expected fail-closed decline, got {result!r}"


# ===========================================================================
# 2. Regression: previously-correct behaviour must be byte-identical.
# ===========================================================================
class TestUnaffectedFormsUnchanged:
    """The widened regex must not perturb any case that already worked."""

    @pytest.mark.parametrize("parent,n", [
        ("butan-2-one", 4),                        # already declined (m_one)
        ("propan-2-one", 3),                       # already declined (m_one)
        ("hexane-2,5-dione", 6),                   # m_dione, unaffected
        ("2,12-dimethyltetradecan-3-amine", 16),   # already declined (m_amine)
        ("ethane-1,2-diamine", 2),                 # m_diamine, unaffected
    ])
    def test_previously_declining_forms_still_decline(self, parent, n):
        assert parent_to_prefix(
            parent, chain_length=n, attach_locant=ATTACH_LOCANT_UNKNOWN
        ) is None

    @pytest.mark.parametrize("parent,n,expected", [
        ("propane", 3, "propyl"),
        ("pyridine", 0, "pyridinyl"),
        ("methanal", 1, "oxomethyl"),               #, one position
        ("acetamide", 2, "carbamoylmethyl"),
    ])
    def test_genuinely_unlocated_forms_still_emit(self, parent, n, expected):
        assert parent_to_prefix(
            parent, chain_length=n, attach_locant=ATTACH_LOCANT_UNKNOWN
        ) == expected


# ===========================================================================
# 3. Structural proof the defect is a REAL wrong-molecule risk, not merely a
# cosmetic locant omission.
# ===========================================================================
@pytest.mark.opsin_gate
@pytest.mark.slow
class TestCorruptionIsARealWrongMolecule:
    """OPSIN re-parses the pre-fix fabrication at the WRONG position."""

    def test_uncited_oxo_reparses_to_a_different_molecule(self, opsin_gate):
        """The composed name a a trace trace captured for the repro molecule
        (before this fix): the enone fragment's oxo locant is missing, and
        OPSIN places it at the lowest available position (an aldehyde
        terminus) instead of the true internal C4 ketone -- a different
        constitution, not just a different spelling."""
        from orthonym.validation.opsin_roundtrip import opsin_parse

        corrupted_name = (
            "(3S,4S,6S)-3,6-dihydroxy-9,13-dimethyl-14-(oxo-2-methylhept-2-"
            "en-4-yl)tetracyclo[8.7.0.0^4,9.0^13,17]heptadec-10-ene"
        )
        true_input_smiles = (
            "CC(C)=CC(=O)CC(C)C1CCC2C3C[C@H](O)[C@H]4C[C@@H](O)CCC4(C)"
            "C3=CCC12C"
        )
        parsed = opsin_parse(corrupted_name)
        if not parsed:
            pytest.skip("OPSIN could not parse the corrupted name at all")
        canon_parsed = Chem.CanonSmiles(parsed)
        canon_true = Chem.CanonSmiles(true_input_smiles)
        assert canon_parsed != canon_true, (
            "expected the uncited-oxo fabrication to denote a DIFFERENT "
            "molecule (structural proof for why this must fail closed); "
            "if this now matches, the corruption class may already be "
            "fixed elsewhere -- re-derive the repro before trusting this "
            "assertion"
        )


# ===========================================================================
# 4. End-to-end safety: the full pipeline on the repro molecule must never
# emit a wrong-connectivity name -- abstain or round-trip, nothing between.
# ===========================================================================
@pytest.mark.opsin_gate
@pytest.mark.slow
class TestFullPipelineNeverEmitsWrongConnectivity:
    def test_repro_molecule_either_abstains_or_round_trips(self, opsin_gate):
        from orthonym.validation.opsin_roundtrip import opsin_parse

        mol, feats = _classified(_ENONE_MOLECULE_SMILES)
        name = t4_coverage.name_t4_complete(mol, feats)
        if name is None:
            return  # honest abstain -- acceptable
        parsed = opsin_parse(name)
        assert parsed, f"T4 emitted an OPSIN-unparseable name: {name!r}"
        m1 = Chem.MolFromSmiles(parsed)
        m2 = Chem.MolFromSmiles(_ENONE_MOLECULE_SMILES)
        assert m1 is not None and m2 is not None
        Chem.RemoveStereochemistry(m1)
        Chem.RemoveStereochemistry(m2)
        assert Chem.MolToSmiles(m1) == Chem.MolToSmiles(m2), (
            f"T4 emitted a complete name {name!r} that denotes a DIFFERENT "
            f"molecule than the input -- the exact 'passes E1 but wrong "
            f"connectivity' defect this task fixes"
        )
