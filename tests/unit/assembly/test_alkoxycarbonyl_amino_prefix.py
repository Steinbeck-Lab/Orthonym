"""C4d — the chalcogen-rooted acyl-amino prefix class, ``R-X-CO-NH-``.

A carbamate (Boc, Cbz, Fmoc, methoxycarbonyl,...) is an ester of carbamic
acid, not an acyl of a carboxylic acid, so no CARBON COUNT can describe it.
Both count-based emitters proved it: one stops at the heteroatom and counts only
the carbonyl carbon, so it returned 1 for a true formyl ``H-CO-NH-`` AND for
``(CH3)3C-O-CO-NH-`` and spelled Boc 'methanoylamino' -- asserting a formyl C-H
the molecule does not have; the other WALKS ACROSS the heteroatom and counted
the organyl beyond it, spelling ``CH3-S-CO-NH-`` 'ethanoylamino'.

The Blue Book builds the prefix by CONCATENATION onto the
chalcogen-group prefix:

  BB 18116 -CO-O-CH2-C6H5 (benzyloxy)carbonyl (preferred prefix)
  BB 18128 CH3-CO-S-CO- (acetylsulfanyl)carbonyl (preferred prefix)
  BB 31698 names -CO-OR' 'alkoxycarbonyl'

Bare-vs-marked is the / simple-vs-compound split: the
retained contractions are SIMPLE (BB 27667), so *tert*-butoxy concatenates bare
(BB 54417 ``N2-(tert-butoxycarbonyl)-L-lysine``) while the concatenated
``benzyloxy`` is COMPOUND (BB 27633) and takes marks (BB 54422
``N5-acetyl-N2-[(benzyloxy)carbonyl]-L-glutamine``); both.

``amino`` is the morpheme (BB 26314), and the assembled shape is the
Blue Book's own (BB 33213 ``[(methanesulfinothioyl)amino]acetic acid (PIN)``).
Parent selection is / (BB 30676): "A carboxylic acid named by
means of a suffix is senior to a derivative of carbonic acid formed by
functional replacement", so the acid is the parent and the carbamate is the
prefix.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.assembly.substituent_enumerator import (
    ACYL_CHALCOGEN_UNNAMEABLE,
    chalcogen_rooted_acyl_amino_core,
)

# The amide/carbamate nitrogen and its carbonyl -- the two atoms the primitive
# is handed. Kept deliberately wider than the carbamate so a CARBON acyl can be
# fed in too, which is how "declines politely" gets tested.
_ACYL_N = Chem.MolFromSmarts("[NX3][CX3]=O")


def _core_for(smiles):
    """Run the primitive on the acyl-nitrogen branch of ``smiles``.

    The fragment handed over is the whole branch hanging off the parent: the
    amide N plus everything reachable through the carbonyl.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    match = mol.GetSubstructMatch(_ACYL_N)
    assert match, f"no acyl nitrogen in {smiles}"
    n_idx, c_idx = match[0], match[1]
    frag = {n_idx}
    stack = [c_idx]
    while stack:
        a = stack.pop()
        if a in frag:
            continue
        frag.add(a)
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetIdx() not in frag and nb.GetIdx() != n_idx:
                stack.append(nb.GetIdx())
    return chalcogen_rooted_acyl_amino_core(mol, c_idx, n_idx, sorted(frag))


@pytest.mark.unit
class TestPrimitiveMorphology:
    """The prefix core, straight from the primitive."""

    @pytest.mark.parametrize("smiles,expected", [
        # BB 54417: tert-butoxy is a SIMPLE retained prefix -> concatenates bare.
        ("OC(=O)CNC(=O)OC(C)(C)C", "(tert-butoxycarbonyl)amino"),
        # BB 56312 methoxycarbonyl*, BB 56003 ethoxycarbonyl* -- also simple.
        ("OC(=O)CNC(=O)OC", "(methoxycarbonyl)amino"),
        ("OC(=O)CNC(=O)OCC", "(ethoxycarbonyl)amino"),
        # BB 18116/54422: benzyloxy is COMPOUND -> keeps its own marks, and the
        # outer pair escalates  ->  per.
        ("OC(=O)CNC(=O)OCc1ccccc1", "[(benzyloxy)carbonyl]amino"),
        # A locant-bearing free valence keeps the alkyl whole inside marks
        #, BB 27683 '(propan-2-yl)oxy'), and the outer pair
        # escalates a second step  ->  -> { }.
        ("CC(C)OC(=O)NCC(=O)O", "{[(propan-2-yl)oxy]carbonyl}amino"),
        # BB 18128: the sulfur analogue takes marks round the sulfanyl.
        ("CSC(=O)NCC(=O)O", "[(methylsulfanyl)carbonyl]amino"),
    ])
    def test_core(self, smiles, expected):
        assert _core_for(smiles) == expected

    def test_tert_butoxy_is_never_spelled_butoxy(self):
        """The old count read 4 carbons and produced n-'butoxy' -- a DIFFERENT
        group. tert-Butoxy must survive as itself (BB 27679, 'not
        tert-butyloxy')."""
        core = _core_for("OC(=O)CNC(=O)OC(C)(C)C")
        assert "tert-butoxy" in core
        assert not core.startswith("(butoxy")

    def test_benzyl_is_never_spelled_heptyloxy(self):
        """Counting benzyl's 7 carbons gave 'heptyloxy' -- a straight chain for
        a ring. The ring must be perceived."""
        assert "benzyloxy" in _core_for("OC(=O)CNC(=O)OCc1ccccc1")


@pytest.mark.unit
class TestPrimitiveFailsClosed:
    """The class is CLAIMED once the acyl is chalcogen-rooted, so an
    unspellable member must fail closed -- never fall through to a count."""

    @pytest.mark.parametrize("smiles", [
        "CC(=O)NCC(=O)O",       # acetyl -- a carbon acyl
        "O=CNCC(=O)O",          # formyl -- the case the count got RIGHT
        "CCC(=O)NCC(=O)O",      # propanoyl
        "c1ccccc1C(=O)NCC(=O)O",  # benzoyl -- a ring acyl
    ])
    def test_carbon_acyl_is_declined_not_claimed(self, smiles):
        """A carbon-rooted acyl must return None -- NOT the fail-closed sentinel
        -- so the caller's own amido paths (formamido, acetamido, benzamido)
        keep working untouched."""
        assert _core_for(smiles) is None

    @pytest.mark.parametrize("smiles", [
        "O=C1OCCN1",            # oxazolidin-2-one
        "O=C1OCCCN1",           # the 6-ring homologue
        "CN1C(=O)OCC1",         # N-methylated
        "O=C1OC2CCCCC2N1",      # fused
        "OC(=O)CN1C(=O)OCC1",   # N-linked to a chain: the walk from the ester O
                                # comes back round to the nitrogen
    ])
    def test_cyclic_carbamate_fails_closed(self, smiles):
        """A cyclic carbamate's O-C(=O)-N is part of the ring; the ring handler
        owns it, so the prefix builder must refuse rather than linearize the
        ring into an 'alkoxy' chain."""
        assert _core_for(smiles) is ACYL_CHALCOGEN_UNNAMEABLE

    def test_refuses_a_fragment_it_would_not_wholly_describe(self):
        """The guard's contract with its callers: the returned prefix accounts
        for EVERY atom handed over. Given one extra atom it must refuse, because
        a name that silently omits part of the fragment reads as a complete
        description of a molecule that is not there."""
        smiles = "OC(=O)CNC(=O)OC(C)(C)C"
        mol = Chem.MolFromSmiles(smiles)
        n_idx, c_idx = mol.GetSubstructMatch(_ACYL_N)[:2]
        frag = {n_idx}
        stack = [c_idx]
        while stack:
            a = stack.pop()
            if a in frag:
                continue
            frag.add(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                if nb.GetIdx() not in frag and nb.GetIdx() != n_idx:
                    stack.append(nb.GetIdx())
        # Sanity: the honest fragment IS nameable.
        assert chalcogen_rooted_acyl_amino_core(
            mol, c_idx, n_idx, sorted(frag)) == "(tert-butoxycarbonyl)amino"
        # Now smuggle in an unrelated atom (a carboxyl oxygen, not adjacent to
        # the nitrogen, so the attachment count is untouched).
        stray = next(a.GetIdx() for a in mol.GetAtoms()
                     if a.GetSymbol() == "O" and a.GetIdx() not in frag
                     and n_idx not in [x.GetIdx() for x in a.GetNeighbors()])
        assert chalcogen_rooted_acyl_amino_core(
            mol, c_idx, n_idx, sorted(frag | {stray})
        ) is ACYL_CHALCOGEN_UNNAMEABLE


@pytest.mark.unit
class TestSecondEmitterPath:
    """``_name_amino_branch`` is the OTHER emitter. Its count walker was worse:
    it traversed every fragment atom and counted the carbons, so it walked
    ACROSS the heteroatom and spelled ``CH3-S-CO-NH-`` 'ethanoylamino'."""

    @staticmethod
    def _branch(smiles):
        from orthonym.assembly.substituent_enumerator import _name_amino_branch
        mol = Chem.MolFromSmiles(smiles)
        n_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N"][0]
        # Parent = the N-methyl carbon (the neighbour with no oxygen on it).
        parent = {nb.GetIdx() for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors()
                  if not any(x.GetSymbol() == "O" for x in nb.GetNeighbors())}
        frag = set(range(mol.GetNumAtoms())) - parent
        return _name_amino_branch(mol, sorted(frag), n_idx, parent)

    def test_chalcogen_route_reaches_this_emitter_too(self):
        assert self._branch("CSC(=O)NC") == "[(methylsulfanyl)carbonyl]amino"

    def test_carbamate_route_reaches_this_emitter(self):
        assert self._branch("CC(C)(C)OC(=O)NC") == "(tert-butoxycarbonyl)amino"

    @pytest.mark.parametrize("smiles,amido,count_spelling", [
        # isobutyryl -- a count says 4 carbons = n-butyryl
        ("CC(C)C(=O)NC", "2-methylpropanamido", "butanoylamino"),
        # acryloyl -- a count erases the double bond
        ("C=CC(=O)NC", "prop-2-enamido", "propanoylamino"),
        # pivaloyl -- a count says 5 carbons = n-pentanoyl
        ("CC(C)(C)C(=O)NC", "2,2-dimethylpropanamido", "pentanoylamino"),
    ])
    def test_count_refuses_what_it_cannot_spell(self, smiles, amido, count_spelling):
        """Never a count stem. Coercing the count to 1 here is what produced
        'methanoylamino' for groups that contain no formyl at all.

        Decision A part 2 (2026-09-27): was ``is None``. The acid-recursive method
        (1) amido prefix, the Blue Book, "Method (1) generates
        preferred IUPAC names.":32998) now runs BEFORE the count and names these
        acyls from their structure; the count still never runs on them. OPSIN 2.9.0
        full-InChIKey exact on '(2-methylpropanamido)acetic acid',
        '(prop-2-enamido)acetic acid', '(2,2-dimethylpropanamido)acetic acid'."""
        got = self._branch(smiles)
        assert got == amido
        assert count_spelling not in got

    def test_faithful_linear_acyl_still_named(self):
        """The control: an unbranched saturated acyl is unaffected."""
        assert self._branch("CCC(=O)NC") == "propanamido"


@pytest.mark.unit
class TestAlkoxyBranchIsPerceivedNotCounted:
    """``_alkoxy_name_for_branch`` feeds ``get_carbamoyloxy_prefix``, the
    PREFIX_FORMS producer for this class. It was a pure carbon COUNT into a
    1..10 table, so it spelled five of six real Boc-family branches as a
    constitutionally DIFFERENT group."""

    _CARB5 = Chem.MolFromSmarts("[NX3][CX3](=O)[OX2][#6]")

    def _alkoxy(self, smiles):
        from orthonym.assembly.substituent_prefix_forms import (
            _alkoxy_name_for_branch,
        )
        mol = Chem.MolFromSmiles(smiles)
        n, c, od, o, r = mol.GetSubstructMatch(self._CARB5)
        return _alkoxy_name_for_branch(mol, r, {n, c, od, o})

    @pytest.mark.parametrize("smiles,expected,was", [
        # BB 27679 'tert-butoxy (preferred prefix)', explicitly not
        # 'tert-butyloxy'. The count read 4 carbons and said n-'butoxy'.
        ("OC(=O)CNC(=O)OC(C)(C)C", "tert-butoxy", "butoxy"),
        # A ring is not a chain: the count read 7 carbons and said 'heptyloxy'.
        ("OC(=O)CNC(=O)OCc1ccccc1", "benzyloxy", "heptyloxy"),
        # BB 27683 '(propan-2-yl)oxy'; the count said n-'propoxy'.
        ("CC(C)OC(=O)NCC(=O)O", "(propan-2-yl)oxy", "propoxy"),
        # A double bond is not optional: the count said saturated 'propoxy'.
        ("C=CCOC(=O)NCCC(=O)O", "(prop-2-en-1-yl)oxy", "propoxy"),
        # BB 17796 / BB 24567 'phenoxy (preferred prefix)'; count said
        # 'hexyloxy', and a naive concatenation would say 'phenyloxy'.
        ("OC(=O)CNC(=O)Oc1ccccc1", "phenoxy", "hexyloxy"),
        # BB 27687 '2-methylpropoxy (preferred prefix) (not isobutoxy)'.
        ("CC(C)COC(=O)NCC(=O)O", "2-methylpropoxy", "butoxy"),
        # The one the count got right, and only by coincidence.
        ("OC(=O)CNC(=O)OC", "methoxy", "methoxy"),
    ])
    def test_branch_is_perceived(self, smiles, expected, was):
        got = self._alkoxy(smiles)
        assert got == expected
        if was != expected:
            assert got != was, f"still spelling the count's answer {was!r}"


@pytest.mark.unit
class TestWholeName:
    """End-to-end: the emitted name, which is what actually ships."""

    @pytest.mark.parametrize("smiles,expected", [
        # Boc-glycine. BB 33213 gives this exact '[(acyl)amino]acetic acid'
        # shape as a PIN, and uses 'acetic acid' as the parent for
        # N-substituted glycine.
        ("OC(=O)CNC(=O)OC(C)(C)C",
         "[(tert-butoxycarbonyl)amino]acetic acid"),
        # Boc-L-alanine: L == S, and R/S are the preferred
        # stereodescriptors.
        ("C[C@H](NC(=O)OC(C)(C)C)C(=O)O",
         "(2S)-2-[(tert-butoxycarbonyl)amino]propanoic acid"),
        # Cbz-glycine: the compound benzyloxy forces a third mark level, the
        # shape of 's '{[(cyclohexylmethyl)sulfonyl]amino}acetic
        # acid'.
        ("OC(=O)CNC(=O)OCc1ccccc1",
         "{[(benzyloxy)carbonyl]amino}acetic acid"),
        ("OC(=O)CNC(=O)OC", "[(methoxycarbonyl)amino]acetic acid"),
        ("OC(=O)CNC(=O)OCC", "[(ethoxycarbonyl)amino]acetic acid"),
        # Not just the acid parent: an alcohol parent takes the same prefix.
        ("CC(C)(C)OC(=O)NCCO",
         "2-[(tert-butoxycarbonyl)amino]ethan-1-ol"),
        # The sulfur analogue, end to end (BB 18128 '(acetylsulfanyl)carbonyl').
        ("CSC(=O)NCC(=O)O",
         "{[(methylsulfanyl)carbonyl]amino}acetic acid"),
        # An N-substituted carbamate cites both branches through the shared
        # assembler method (2), BB 33042).
        ("CCOC(=O)N(CC)CC(=O)O",
         "[(ethoxycarbonyl)(ethyl)amino]acetic acid"),
    ])
    def test_pin(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_no_name_asserts_an_absent_formyl(self):
        """The detector that sized this class: the name must not claim a formyl
        the molecule does not have."""
        formyl = Chem.MolFromSmarts("[CX3H1](=O)")
        for smiles in ("OC(=O)CNC(=O)OC(C)(C)C",
                       "CC(C)(C)OC(=O)NCCO",
                       "C=CCOC(=O)NCCC(=O)O",
                       "OC(=O)CNC(=O)OCc1ccccc1"):
            mol = Chem.MolFromSmiles(smiles)
            assert not mol.HasSubstructMatch(formyl), smiles
            low = name_compound(smiles).lower()
            for lie in ("methanoyl", "formyl", "formamido"):
                assert lie not in low, f"{smiles} -> {low}"

    def test_n_substituted_carbamate_keeps_its_branch(self):
        """An N-substituted carbamate cites both N-substituents
        method (2), BB 33042 '2-[methyl(propanoyl)amino]benzene-1-sulfonic
        acid'). Failing closed here is NOT safe: the caller then DROPS the whole
        branch and still emits, which turned this molecule into a bare
        '3-fluoropropanoic acid'."""
        got = name_compound("CC(C)(C)OC(=O)N(C)[C@@H](CF)C(=O)O")
        assert "tert-butoxycarbonyl" in got
        assert "methyl" in got
        assert "fluoro" in got

    @pytest.mark.parametrize("smiles,expected", [
        # The count path is still used where it IS faithful.
        ("O=CNCC(=O)O", "formamidoacetic acid"),
        ("CC(=O)NCC(=O)O", "acetamidoacetic acid"),
        ("CCC(=O)NCC(=O)O", "propanamidoacetic acid"),
        # Carbamate as the PARENT is unchanged (it was already right).
        ("CC(C)(C)OC(=O)NCC", "tert-butyl ethylcarbamate"),
    ])
    def test_neighbouring_classes_unchanged(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles", [
        "CC(C)C(=O)NCC(=O)O",   # isobutyryl: count says 'butanoyl' (n-butyryl)
        "C=CC(=O)NCC(=O)O",     # acryloyl: count says 'propanoyl' (saturated)
        "CC(C)(C)C(=O)NCC(=O)O",  # pivaloyl: count says 'pentanoyl'
    ])
    def test_count_path_refuses_acyls_it_cannot_describe(self, smiles):
        """A COUNT can only spell an unbranched saturated chain. For a branched
        or unsaturated acyl it named a constitutionally DIFFERENT group, so the
        soundness gate must refuse rather than guess."""
        got = name_compound(smiles).lower()
        for wrong in ("butanoylamino", "propanoylamino", "pentanoylamino"):
            assert wrong not in got, f"{smiles} -> {got}"
