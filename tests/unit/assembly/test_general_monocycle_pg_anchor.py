"""P-14.4 / P-64.2.1.2 -- the principal characteristic group must win ring
numbering in ``name_general_monocycle``.

``pg_ring_atoms`` was collected from the RAW SMARTS match::

    on_ring = [i for i in match if i in ring_set]

That is not the locant-bearing atom, and it broke both ring-suffix styles:

* **inline** suffixes (``one``/``ol``/``amine``/``thiol``/``imine``) -- the
  ketone SMARTS ``[#6][CX3](=O)[#6]`` matches *both flanking ring carbons plus
  the carbonyl carbon*, so a ring monoketone contributed three ring atoms and a
  para-dione contributed all six.  ``pg`` was then identical for every candidate
  orientation inside ``_orient_carbocycle`` (key ``(pg, uns, sub, canon)``), the
  principal-group criterion was silently neutered, and the tie-break fell
  through to ring unsaturation -- ``cyclohexa-1,4-diene-3,6-dione``.
* **appended** suffixes (carboxylic acid, carbaldehyde, carbonitrile, ...) --
  the anchor atom is exocyclic by construction, so ``match & ring_set`` was
  *empty*, the criterion was absent altogether and the ring numbered
  arbitrarily: ``cyclohexane-4-carboxylic acid``.

Blue Book, **P-64.2.1.2** (``BlueBookV2/BlueBookV2.md:28307``), example at
``:28320``::

    1,4-benzoquinone   cyclohexa-2,5-diene-1,4-dione (PIN) (not benzoquinone)

The dione takes 1,4 and the diene takes 2,5 -- not the other way round.

The fix routes through the declared anchor primitive
(``seniority.PG_ATTACHMENT_INDICES`` via
``parent_selection._pg_attachment_atoms``) and then answers one question
uniformly for both styles: *which ring atom bears the principal characteristic
group?*  The anchor itself when it is on the ring, otherwise the ring
neighbour(s) it hangs from.

These tests drive the producer directly.  ``tests/conftest.py`` disables the
OPSIN validity gate by default, so whole-molecule assertions through
``Orthonym.name_tiered`` measure a *different* pipeline than the CLI does --
they are not a sound oracle for this layer.
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.assembly.general_engine import (
    name_general_monocycle, _RING_SUFFIX_STYLES,
)


def _engine(smiles):
    """Run ``name_general_monocycle`` directly -- deterministic, gate-independent."""
    nm = Orthonym(style="pin", general_fallback=True, allow_aromatic_general=True)
    mol = Chem.MolFromSmiles(smiles)
    feats = nm._perceive(mol, smiles, Chem.MolToSmiles(mol))
    nm._classify(feats)
    res = name_general_monocycle(mol, feats, allow_aromatic_general=True)
    return getattr(res, "name", None)


# ---------------------------------------------------------------------------
# Inline ring suffixes: the characteristic atom IS a ring atom
# ---------------------------------------------------------------------------

class TestInlineRingSuffixGetsLowestLocant:

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # The Blue Book PIN itself (P-64.2.1.2, :28320).
        ("O=C1C=CC(=O)C=C1", "cyclohexa-2,5-diene-1,4-dione"),
        # ortho-dione: suffix still outranks the diene.
        ("O=C1C=CC=CC1=O", "cyclohexa-3,5-diene-1,2-dione"),
        # Saturated diketone and monoketone.
        ("O=C1CCC(=O)CC1", "cyclohexane-1,4-dione"),
        ("O=C1CCCCC1", "cyclohexan-1-one"),
        ("CC1CCC(=O)CC1", "4-methylcyclohexan-1-one"),
    ])
    def test_ketone_numbering(self, smiles, expected):
        assert _engine(smiles) == expected

    @pytest.mark.unit
    def test_large_substituent_monoketone(self):
        """Measured molecule that reaches this producer in the live pipeline."""
        assert _engine("NC(C)CCCCC1CCC(=O)CC1") == \
            "4-(5-aminohexyl)cyclohexan-1-one"

    @pytest.mark.unit
    def test_large_substituent_para_dione(self):
        assert _engine("NCCCCC1=CC(=O)C(=CC1=O)CCCCN") == \
            "2,5-bis(4-aminobutyl)cyclohexa-2,5-diene-1,4-dione"

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles", [
        "NCCCCC1=CC(=O)C(=CC1=O)CCCCN",
        "CC1=CC(=O)C(=CC1=O)C(N)CCCCCCCC",
        "CC1=C(O)C(=O)C(C)=C(O)C1=O",
    ])
    def test_para_dione_skeleton_is_the_blue_book_one(self, smiles):
        """Rule-level guard independent of substituent numbering."""
        name = _engine(smiles)
        assert name is not None
        assert "cyclohexa-2,5-diene-1,4-dione" in name
        assert "diene-3,6-dione" not in name


# ---------------------------------------------------------------------------
# Appended ring suffixes: the anchor is exocyclic, the locant is its ring atom
# ---------------------------------------------------------------------------

class TestAppendedRingSuffixGetsLowestLocant:

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("OC(=O)C1CCCCC1", "cyclohexane-1-carboxylic acid"),
        ("O=CC1CCCCC1", "cyclohexane-1-carbaldehyde"),
        ("N#CC1CCCCC1", "cyclohexane-1-carbonitrile"),
        ("NC(=O)C1CCCCC1", "cyclohexane-1-carboxamide"),
    ])
    def test_attachment_ring_atom_is_c1(self, smiles, expected):
        assert _engine(smiles) == expected


# ---------------------------------------------------------------------------
# Must-not-change: the -ol/-amine family already numbered correctly
# ---------------------------------------------------------------------------

class TestAlreadyCorrectFamiliesUnchanged:

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("Oc1ccccc1", "benzen-1-ol"),
        ("Nc1ccccc1", "benzen-1-amine"),
        ("OC1=CC=CC=C1O", "benzene-1,2-diol"),
        ("OC1CCCCC1", "cyclohexan-1-ol"),
        ("NC1CCCCC1", "cyclohexan-1-amine"),
        ("SC1CCCCC1", "cyclohexane-1-thiol"),
        ("c1ccccc1", "benzene"),
        # No principal group at all -- the anchor logic must not fire.
        # (The absent '1' locant is a separate P-14.3.4 question, out of scope
        # here; recorded as the measured baseline so a change trips this test.)
        ("CC1CCCCC1", "1-methylcyclohexane"),
    ])
    def test_unchanged(self, smiles, expected):
        assert _engine(smiles) == expected


# ---------------------------------------------------------------------------
# The table-gap invariant that a previous anchor change tripped over
# ---------------------------------------------------------------------------

class TestAnchorTableCoversEveryRingPathFG:
    """``_pg_attachment_atoms`` silently falls back to SMARTS index 0 for any FG
    missing from ``PG_ATTACHMENT_INDICES``.  For three ``-ol``/``-amine`` FGs
    index 0 is the HETEROATOM, and that table gap once became a live regression.

    Every FG that can reach this ring path must therefore be *declared*: either
    it has an explicit entry, or it is on the audited no-entry allowlist whose
    SMARTS provably lead with the locant-bearing atom.  A new ring suffix, or a
    reordered SMARTS, must fail here rather than silently take index 0.
    """

    @pytest.mark.unit
    def test_no_undeclared_ring_path_fg(self):
        from orthonym.rules.seniority import (
            SENIORITY_ORDER, PG_ATTACHMENT_INDICES, get_suffix,
        )
        from orthonym.assembly.general_engine import _LEADING_ANCHOR_RING_PGS

        undeclared = []
        for fg in SENIORITY_ORDER:
            try:
                suffix = get_suffix(fg, is_ring=True)
            except Exception:
                continue
            if suffix not in _RING_SUFFIX_STYLES:
                continue
            if PG_ATTACHMENT_INDICES.get(fg) is not None:
                continue
            if fg in _LEADING_ANCHOR_RING_PGS:
                continue
            undeclared.append(fg)

        assert not undeclared, (
            "functional groups reach the monocycle ring-suffix path with no "
            "declared anchor and no allowlist entry, so they would silently "
            f"take SMARTS index 0: {undeclared}"
        )
