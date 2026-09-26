""" — the substituted-'anilino' PREFERRED PREFIX (a phase, task P4-a).

Governing rule, verbatim from ``the Blue Book Blue Book`` under the heading
chain ``## AMINES`` / ``### Primary amines`` /
``### Retained names`` / ``****``:

    "Aniline, for C6H5-NH2, is the only name for a primary amine retained as a
     preferred IUPAC name for which full substitution is permitted on the ring and
     the nitrogen atom.... The prefix name 'anilino' is retained as the preferred
     prefix for C6H5-NH- with full substitution allowed. The name 'phenylamino' may
     be used in general nomenclature."

The Blue Book's own two-column preferred-prefix pairs:

    the Blue Book anilino (preferred prefix) phenylamino
    the Blue Book 4-chloroanilino (preferred prefix) (4-chlorophenyl)amino
    the Blue Book 4-methylanilino (preferred prefix) (4-methylphenyl)amino
                                                     (not p-toluidino)

Enclosure, from ```` (heading the Blue Book):

    the Blue Book 3-anilinobenzoic acid (PIN) 3-(phenylamino)benzoic acid
    the Blue Book 3-(N-methylanilino)phenol (PIN) 3-[methyl(phenyl)amino]phenol

i.e. a prefix carrying its own locant(s) takes enclosing marks; one carrying none
does not.

Alphanumerical-order coupling, ``the Blue Book``:

    "'hydroxyanilinomethyl' precedes 'hydroxyphenylmethylamino' in alphanumerical
     order (see "

so the switch to the preferred prefix MOVES the prefix's position in the assembled
name. That is a correctness coupling, not a cosmetic one, and every case below that
has a second prefix is asserted on the FINAL name string.

★ Scope boundary. The transformation applies ONLY where an anilino-family prefix is
already the chosen construction. It must NEVER promote an anilino prefix over a
senior name-selection criterion — see ``TestSeniorityBoundariesAreNotPromoted``,
which pins the two Blue Book boundary rows (the Blue Book multiplicative,;
the Blue Book maximum-prefix-count,.

NOTE ON THE HARNESS: ``tests/conftest.py:267`` disables the OPSIN validity gate for
every test, so ``Orthonym.name`` here returns the RAW construction rather than
the gate-filtered production output. That is deliberate — these tests assert the
PRODUCER. Each expected name in this file was additionally OPSIN-round-trip
verified against its SMILES (see ``benchmarks/the gold set/packs/p62_anilino.json``);
the round trip proves VALIDITY only, never PIN status, for which the Blue Book
citations above are the sole authority.
"""

import pytest
from rdkit import Chem

from orthonym.assembly.naming_utils import alpha_sort_key
from orthonym.namer import Orthonym
from orthonym.rules.ring_substituents import (
    anilino_prefix_from_aniline_name,
    anilino_prefix_from_n_branch,
    anilino_preferred_prefix,
)

pytestmark = pytest.mark.unit


def _name(smiles: str) -> str:
    return Orthonym().name(smiles)


# --------------------------------------------------------------------------- #
# 1. The shared string-level primitive #
# --------------------------------------------------------------------------- #
class TestAnilinoPreferredPrefixHelper:
    """``anilino_preferred_prefix`` — the head-morpheme substitution itself."""

    def test_bare_ring_gives_bare_anilino(self):
        """the Blue Book 'anilino (preferred prefix)' — no locant, so NO enclosing marks
        (the Blue Book '3-anilinobenzoic acid', cited bare)."""
        assert anilino_preferred_prefix("phenyl") == "anilino"

    def test_ring_substituted_is_enclosed(self):
        """the Blue Book '4-chloroanilino (preferred prefix) | (4-chlorophenyl)amino'.
        The prefix now carries its own locant, so the Blue Book's enclosure applies."""
        assert anilino_preferred_prefix("4-chlorophenyl") == "(4-chloroanilino)"

    def test_ring_substituted_alkyl(self):
        """the Blue Book '4-methylanilino (preferred prefix) | (4-methylphenyl)amino'."""
        assert anilino_preferred_prefix("4-methylphenyl") == "(4-methylanilino)"

    def test_multiple_ring_locants_are_carried_unchanged(self):
        """The class is OPEN, so the prefix is CONSTRUCTED, not looked up: every
        ring locant survives verbatim (they coincide with the phenyl numbering
        because both put the N-bearing carbon at 1)."""
        assert (anilino_preferred_prefix("2,3-dimethylphenyl")
                == "(2,3-dimethylanilino)")

    def test_n_alkyl_substituent_takes_the_N_locant(self):
        """the Blue Book '3-(N-methylanilino)phenol (PIN)'."""
        assert anilino_preferred_prefix("phenyl", "methyl") == "(N-methylanilino)"

    def test_n_aryl_substituent(self):
        """the Blue Book permits full substitution 'on the ring AND the nitrogen atom',
        so an N-aryl is the same construction with alkyl:= aryl."""
        assert anilino_preferred_prefix("phenyl", "phenyl") == "(N-phenylanilino)"

    def test_enclose_false_returns_the_bare_core(self):
        """Callers that apply their own enclosing marks ask for the core."""
        assert (anilino_preferred_prefix("4-chlorophenyl", enclose=False)
                == "4-chloroanilino")
        assert anilino_preferred_prefix("phenyl", enclose=False) == "anilino"

    @pytest.mark.parametrize("ring_prefix", [
        None, "", "cyclohexyl", "pyridin-2-yl", "naphthalen-2-yl", "phenyl-", "phen",
        "methyl",
    ])
    def test_fails_closed_on_anything_that_is_not_a_phenyl_ring(self, ring_prefix):
        """'anilino' is retained for C6H5-NH- ONLY. Every other ring keeps its own
        construction, so a non-phenyl input must decline rather than fabricate."""
        assert anilino_preferred_prefix(ring_prefix) is None

    def test_fails_closed_when_ring_and_nitrogen_are_both_substituted(self):
        """Both a ring locant and an N locant would have to be merged into ONE
        alphanumerical sequence. No Blue Book worked example for that
        merge was found during derivation, so the helper declines rather than invent
        an ordering. Recorded as a follow-up, not silently guessed."""
        assert anilino_preferred_prefix("4-chlorophenyl", "methyl") is None

    def test_alphanumerical_key_moves_from_p_to_a(self):
        """the Blue Book — the preferred prefix alphabetises under its own first letter,
        which is what moves it in the assembled name."""
        preferred = anilino_preferred_prefix("4-chlorophenyl", enclose=False)
        assert alpha_sort_key(preferred) == "chloroanilino"
        #... and it now sorts AFTER 'bromo', which the general form did not.
        assert alpha_sort_key("bromo") < alpha_sort_key(preferred)
        # 2026-09-25 (pre-existing-failures plan, Task 5) change-asserted-value: a prefix "is considered to begin with the first letter of its complete name" (the Blue Book) -- the general form '(4-chlorophenyl)amino' now
        # also keys at 'c' (it used to key at its inner locant '4', before every letter).
        assert alpha_sort_key("(4-chlorophenyl)amino") == "chlorophenylamino"


# --------------------------------------------------------------------------- #
# 2. The shared structural primitive (the atom-drop fix) #
# --------------------------------------------------------------------------- #
class TestAnilinoPrefixFromNBranch:
    """``anilino_prefix_from_n_branch`` — derives the ring name from the GRAPH.

    This is the entry point for the three sites that returned the bare literal
    ``"anilino"`` after checking only that the six ring atoms were inside the
    substituent, silently dropping every ring substituent.
    """

    @staticmethod
    def _branch(smiles, n_smarts="[NX3;H1,H0;!$(N=*)]"):
        """Return (mol, n_idx, branch_atoms) for the amine N and its whole branch
        away from the atom that carries the rest of the molecule."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        n_idx = mol.GetSubstructMatches(Chem.MolFromSmarts(n_smarts))[0][0]
        # walk the N-branch: everything reachable from N without crossing the
        # aliphatic-carbon neighbour that leads back to the parent chain.
        parent_side = [nb.GetIdx() for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors()
                       if not nb.GetIsAromatic()]
        assert len(parent_side) == 1, parent_side
        seen, stack, branch = {parent_side[0]}, [n_idx], []
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            branch.append(a)
            stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors())
        return mol, n_idx, branch

    def test_unsubstituted_ring_gives_bare_anilino(self):
        """The byte-identical base case: -NH-C6H5 stays 'anilino'."""
        mol, n_idx, branch = self._branch("N#CCCNc1ccccc1")
        assert anilino_prefix_from_n_branch(mol, n_idx, branch) == "anilino"

    def test_substituted_ring_carries_its_decoration(self):
        """The atom drop: the chloro must appear in the prefix."""
        mol, n_idx, branch = self._branch("N#CCCNc1ccc(Cl)cc1")
        assert anilino_prefix_from_n_branch(mol, n_idx, branch) == "(4-chloroanilino)"

    def test_two_ring_substituents(self):
        mol, n_idx, branch = self._branch("Cc1cccc(NCCC#N)c1C")
        assert (anilino_prefix_from_n_branch(mol, n_idx, branch)
                == "(2,3-dimethylanilino)")

    def test_attachment_carbon_is_locant_one(self):
        """The locant-coincidence premise: the ring is numbered with the N-bearing
        carbon at 1, so a meta substituent reads '3-', not '5-'."""
        mol, n_idx, branch = self._branch("N#CCCNc1cccc(Cl)c1")
        assert anilino_prefix_from_n_branch(mol, n_idx, branch) == "(3-chloroanilino)"

    def test_fails_closed_on_an_n_substituted_nitrogen(self):
        """-N(CH3)-C6H4-CH3: the branch carries an N-substituent as well, so ring+
        decoration cannot account for every branch atom. Declining is mandatory —
        returning 'anilino' here named a different molecule."""
        mol = Chem.MolFromSmiles("N#CCCN(C)c1ccc(C)cc1")
        n_idx = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H0](-C)(-c)"))[0][0]
        branch = [a.GetIdx() for a in mol.GetAtoms()
                  if a.GetIdx() >= n_idx]  # N + methyl + ring + ring methyl
        assert anilino_prefix_from_n_branch(mol, n_idx, branch) is None

    def test_fails_closed_on_a_fused_ring(self):
        """A benzo sub-ring of naphthalene is all-carbon and aromatic but is NOT a
        C6H5- group; 'anilino' must not claim it."""
        mol, n_idx, branch = self._branch("N#CCCNc1ccc2ccccc2c1")
        assert anilino_prefix_from_n_branch(mol, n_idx, branch) is None

    def test_fails_closed_on_a_heteroaryl_ring(self):
        mol, n_idx, branch = self._branch("N#CCCNc1ccccn1")
        assert anilino_prefix_from_n_branch(mol, n_idx, branch) is None


# --------------------------------------------------------------------------- #
# 2b. The aniline-parent-name entry point, and the 'anilinyl' unmasking #
# --------------------------------------------------------------------------- #
class TestAnilinoPrefixFromAnilineName:
    """``'<X>aniline'`` -> ``'<X>anilino'``.

    The prefix sequence is inherited from the aniline joiner, not recomputed: the
    Blue Book prints the two forms with identical decoration (the Blue Book
    '4-chloroaniline (PIN)' / the Blue Book '4-chloroanilino (preferred prefix)'), and
     orders detachable prefixes among themselves, so the head morpheme plays
    no part in the ordering.
    """

    @pytest.mark.parametrize("aniline,expected", [
        ("aniline", "anilino"),
        ("4-chloroaniline", "(4-chloroanilino)"),
        ("4-methylaniline", "(4-methylanilino)"),
        ("N-methylaniline", "(N-methylanilino)"),
        ("4-methyl-N-methylaniline", "(4-methyl-N-methylanilino)"),
        ("4-methyl-N,N-dimethylaniline", "(4-methyl-N,N-dimethylanilino)"),
        ("2,3-dimethylaniline", "(2,3-dimethylanilino)"),
    ])
    def test_head_morpheme_substitution(self, aniline, expected):
        assert anilino_prefix_from_aniline_name(aniline) == expected

    @pytest.mark.parametrize("name", [
        None, "", "pyridine", "piperidine", "benzenamine", "phenol", "aniline-",
    ])
    def test_fails_closed_on_non_aniline_names(self, name):
        assert anilino_prefix_from_aniline_name(name) is None

    def test_anilinyl_is_never_produced(self):
        """★ THE UNMASKING GUARD. ``parent_to_prefix`` reached the
        heterocyclic '-ine' -> '-inyl' rule (substituent_naming.py, the branch that
        turns 'pyridine' into 'pyridinyl') with an ANILINE parent name and emitted
        '4-methyl-N-methylanilinyl'.

        Aniline is a carbocyclic amine, not a heterocycle, and 'anilinyl' appears
        nowhere in the Blue Book — but OPSIN parses it to the correct structure, so
         could not see it and the fabricated name SHIPPED. This is the
        standing hazard in its exact form: making the composer site fail closed
        removed a wrong output and unmasked a worse generator."""
        from orthonym.assembly.substituent_naming import (
            ATTACH_LOCANT_UNKNOWN, parent_to_prefix)
        for aniline in ("aniline", "4-methylaniline", "N-methylaniline",
                        "4-methyl-N-methylaniline"):
            got = parent_to_prefix(aniline, 6, attach_locant=ATTACH_LOCANT_UNKNOWN)
            assert got is None or "anilinyl" not in got, (
                f"parent_to_prefix({aniline!r}, attach_locant=ATTACH_LOCANT_UNKNOWN) -> {got!r} fabricates 'anilinyl'"
            )

    def test_the_heterocyclic_ine_rule_still_works(self):
        """The aniline carve-out must precede, not replace,."""
        from orthonym.assembly.substituent_naming import (
            ATTACH_LOCANT_UNKNOWN, parent_to_prefix)
        assert parent_to_prefix("pyridine", 5, attach_locant=ATTACH_LOCANT_UNKNOWN) == "pyridinyl"
        assert parent_to_prefix("piperidine", 5, attach_locant=ATTACH_LOCANT_UNKNOWN) == "piperidinyl"

    def test_ring_and_nitrogen_substituted_end_to_end(self):
        """The case ``anilino_preferred_prefix`` declines (it would have to invent
        the merge) but this path serves, because the merge is inherited
        from the parent form Orthonym already ships ('4-methyl-N-methylaniline')."""
        assert (_name("N#CCCN(C)c1ccc(C)cc1")
                == "3-(4-methyl-N-methylanilino)propanenitrile")

    def test_n_alkyl_anilino_on_a_chain_parent(self):
        assert (_name("N#CCCN(C)c1ccccc1")
                == "3-(N-methylanilino)propanenitrile")


# --------------------------------------------------------------------------- #
# 3. End to end — the FINAL name string #
# --------------------------------------------------------------------------- #
class TestAnilinoEndToEnd:

    def test_bare_anilino_on_a_ring_parent_is_unchanged(self):
        """the Blue Book verbatim '3-anilinobenzoic acid (PIN)'. Bare, no marks."""
        assert _name("OC(=O)c1cccc(Nc2ccccc2)c1") == "3-anilinobenzoic acid"

    def test_ring_substituted_anilino_on_a_chain_parent(self):
        """the Blue Book. The parent (propanenitrile) was already right; the chloro was
        being dropped."""
        assert _name("N#CCCNc1ccc(Cl)cc1") == "3-(4-chloroanilino)propanenitrile"

    def test_ring_methyl_substituted_anilino(self):
        """BB:26166 — and NOT 'p-toluidino', which BB:26166 explicitly discards."""
        name = _name("N#CCCNc1ccc(C)cc1")
        assert name == "3-(4-methylanilino)propanenitrile"
        assert "toluidino" not in name

    def test_two_ring_substituents_the_roadmap_example(self):
        assert (_name("Cc1cccc(NCCC#N)c1C")
                == "3-(2,3-dimethylanilino)propanenitrile")

    def test_n_methyl_anilino_is_the_blue_books_own_pin(self):
        """the Blue Book verbatim: '3-(N-methylanilino)phenol (PIN)
        3-[methyl(phenyl)amino]phenol'. HEAD shipped a third spelling,
        '3-(N-methyl-N-phenylamino)phenol', which is non-PIN by
        (the Blue Book) because a component name is not a preferred name."""
        assert _name("Oc1cccc(N(C)c2ccccc2)c1") == "3-(N-methylanilino)phenol"

    def test_n_phenyl_anilino(self):
        assert (_name("Oc1cccc(N(c2ccccc2)c2ccccc2)c1")
                == "3-(N-phenylanilino)phenol")

    def test_anilide_prefix_on_a_chain_end(self):
        """The polyfunctional chain-end demoted-acyl site. HEAD shipped
        '5-(4-chlorophenylamino)-5-oxopentanoic acid', which is not even the
        well-formed general spelling (the Blue Book requires '(4-chlorophenyl)amino')."""
        assert (_name("O=C(O)CCCC(=O)Nc1ccc(Cl)cc1")
                == "5-(4-chloroanilino)-5-oxopentanoic acid")

    def test_bare_anilide_prefix_is_unchanged(self):
        """The already-passing gold in packs/characteristic_groups.json."""
        assert (_name("O=C(O)CCCC(=O)Nc1ccccc1")
                == "5-anilino-5-oxopentanoic acid")

    def test_substituent_enumerator_site(self):
        assert _name("OCCCCNc1ccc(Cl)cc1") == "4-(4-chloroanilino)butan-1-ol"

    def test_substituent_enumerator_bare_site_is_unchanged(self):
        """Also asserted by tests/unit/assembly/
        test_name_heteroatom_substituent_split.py::
        test_n_helper_returns_anilino_for_aniline_substituent."""
        assert _name("OCCCCNc1ccccc1") == "4-anilinobutan-1-ol"

    @pytest.mark.parametrize("smiles", [
        "N#CCCNc1ccc(Cl)cc1",
        "N#CCCNc1ccc(C)cc1",
        "Cc1cccc(NCCC#N)c1C",
        "Oc1cccc(N(C)c2ccccc2)c1",
        "Oc1cccc(N(c2ccccc2)c2ccccc2)c1",
        "O=C(O)CCCC(=O)Nc1ccc(Cl)cc1",
        "OCCCCNc1ccc(Cl)cc1",
        "OC(=O)c1ccc(Nc2ccc(Cl)cc2)c(Br)c1",
    ])
    def test_the_general_nomenclature_spelling_never_ships(self, smiles):
        """the Blue Book puts 'phenylamino' in the general-nomenclature column, so no
        emission in this class may contain it in any spelling."""
        name = _name(smiles)
        for banned in ("phenylamino", "phenyl)amino", "phenyl]amino"):
            assert banned not in name, f"{smiles} -> {name!r} contains {banned!r}"


# --------------------------------------------------------------------------- #
# 4. The alphanumerical-order coupling, asserted on the final string #
# --------------------------------------------------------------------------- #
class TestAlphanumericalOrderCoupling:
    """BB:6375 — switching to the preferred prefix MOVES the prefix in the name."""

    def test_second_prefix_now_precedes_the_anilino_prefix(self):
        """'bromo' < 'chloroanilino', so bromo is cited first. HEAD emitted
        '4-[(4-chlorophenyl)amino]-3-bromobenzoic acid' — mis-ordered even as a
        general name, because alpha_sort_key leaves the leading '(' in the key and
        '(' sorts before every letter."""
        assert (_name("OC(=O)c1ccc(Nc2ccc(Cl)cc2)c(Br)c1")
                == "3-bromo-4-(4-chloroanilino)benzoic acid")

    def test_the_unsubstituted_control_does_not_move(self):
        """'anilino' < 'bromo' already, and this is correct on HEAD: routing the
        bare site through the shared helper must not disturb it."""
        assert (_name("OC(=O)c1ccc(Nc2ccccc2)c(Br)c1")
                == "4-anilino-3-bromobenzoic acid")


# --------------------------------------------------------------------------- #
# 5. ★ Seniority boundaries — the anilino prefix must NOT be promoted #
# --------------------------------------------------------------------------- #
class TestSeniorityBoundariesAreNotPromoted:
    """Tripwires for the two Blue Book boundary rows. These guard the SCOPE of the
    transformation: it renames an anilino-family prefix that some other rule already
    chose; it never wins a parent- or name-selection contest for it."""

    def test_max_prefix_count_still_declines_an_anilino_prefix(self):
        """the Blue Book verbatim: 'N1-(4-aminophenyl)-N4-phenylbenzene-1,4-diamine (PIN)
        (maximum number of substituents cited as prefixes; see
        [not N1-(4-anilinophenyl)benzene-1,4-diamine]'.

        The bracketed alternative is 'not'-marked — tier 3, "discarded or no longer
        recommended" (the Blue Book) — so an anilino prefix must NEVER appear here.
        already declines it on HEAD; this test locks that against the fix."""
        name = _name("Nc1ccc(Nc2ccc(Nc3ccccc3)cc2)cc1")
        assert "anilino" not in name, (
            f"P-45.2.1 boundary breached: {name!r} promotes an anilino prefix that "
            "BB:26404 marks 'not'"
        )
        assert "benzene-1,4-diamine" in name

    def test_multiplicative_pin_is_still_not_reached(self):
        """the Blue Book verbatim, under '### Multiplicative nomenclature':
        '4,4'-azanediyldibenzonitrile (PIN) 4-[(4-cyanophenyl)amino]benzonitrile
        4-(4-cyanoanilino)benzonitrile'.

        Both substitutive forms are listed UNMARKED, i.e. legitimate general IUPAC
        names (tier 2, the Blue Book); neither carries a 'not'. P4-a therefore moves this
        emission WITHIN tier 2 to the form built from the preferred component, while
        the true PIN — which needs multiplicative nomenclature /
        — remains unreached. This test exists so that gap stays visible: it fails if
        the emission drifts to a spelling the Blue Book does not list at all, and it
        refuses to let the anilino spelling be mistaken for the PIN."""
        name = _name("N#Cc1ccc(Nc2ccc(C#N)cc2)cc1")
        bb_listed = {
            "4-(4-cyanoanilino)benzonitrile",            # the Blue Book, general (tier 2)
            "4-[(4-cyanophenyl)amino]benzonitrile",      # the Blue Book, general (tier 2)
        }
        assert name in bb_listed, (
            f"{name!r} is not one of the substitutive forms the Blue Book lists at "
            f"BB:26419: {sorted(bb_listed)}"
        )
        assert name != "4,4'-azanediyldibenzonitrile", (
            "the multiplicative PIN is NOT built by P4-a; if this now passes, "
            "P-45.1.1/P-62.2.5.1 landed elsewhere and the p62_anilino tripwire row "
            "must be re-categorised to 'target'"
        )


# --------------------------------------------------------------------------- #
# 6. The amine-as-parent promotion must be untouched #
# --------------------------------------------------------------------------- #
class TestAmineParentPromotionUnchanged:
    """Only the PREFIX spelling is in scope. The ``amine_candidate`` payloads that
    ride alongside these prefixes drive the aniline-as-parent promotion, and those
    names must stay byte-identical."""

    @pytest.mark.parametrize("smiles,expected", [
        ("Cc1cccc(NCC)c1C", "N-ethyl-2,3-dimethylaniline"),
        ("Cc1cccc(N)c1C", "2,3-dimethylaniline"),
        ("CN(c1ccccc1)c1ccccc1", "N-methyl-N-phenylaniline"),
        ("c1ccc(Nc2ccccc2)cc1", "N-phenylaniline"),
        ("Clc1ccc(Nc2ccccc2)cc1", "4-chloro-N-phenylaniline"),
    ])
    def test_promotion_names_are_byte_identical(self, smiles, expected):
        assert _name(smiles) == expected


# --------------------------------------------------------------------------- #
# 7. The two routed sites the first review found had no witness #
# #
# Both were raised by the P4-a adversarial review (2026-07-28): #
# - fused_rings.py's N-monoalkyl site was the one entry in the brief's site #
# census with no row in p62_anilino.json and no unit test, because every #
# end-to-end attempt is intercepted by an unrelated handler upstream. #
# - substituent_enumerator.py's enclose=False contract was documented by #
# comment only; flipping it changed no test. #
# Both are therefore pinned here by DIRECT calls, which is the only level at #
# which they are reachable. #
# --------------------------------------------------------------------------- #
class TestFusedRingNAryLSite:
    """``fused_rings._identify_fused_substituent`` — the 8th routed site.

    ``_bfs_alkyl_from`` rejects only HETEROATOMS, so an all-carbon AROMATIC ring
    passes it. When ``name_substituent_fragment`` then declined, the
    ``get_alkyl_name(carbon_count)`` fallback named the ring by its carbon COUNT:
    a quinoline bearing ``-NH-(4-methylphenyl)`` emitted ``'heptylamino'`` (7
    ring+methyl carbons) and ``-NH-(4-ethylphenyl)`` emitted ``'octylamino'``.
    Those name a DIFFERENT molecule. Pre-existing, never reached end-to-end, and
    now routed through the graph-derived primitive instead.
    """

    @staticmethod
    def _quinoline_core(mol):
        """The fused ring system carrying the aromatic (ring) nitrogen."""
        ring_info = mol.GetRingInfo()
        ar_n = [a.GetIdx() for a in mol.GetAtoms()
                if a.GetSymbol() == 'N' and a.GetIsAromatic()][0]
        core = set()
        for ring in ring_info.AtomRings():
            if ar_n in ring:
                core |= set(ring)
        for ring in ring_info.AtomRings():
            if set(ring) & core:
                core |= set(ring)
        return core

    def _identify(self, smiles: str):
        from rdkit import Chem
        from orthonym.rules.fused_rings import _identify_fused_substituent
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"unparseable test SMILES {smiles!r}"
        exo_n = [a.GetIdx() for a in mol.GetAtoms()
                 if a.GetSymbol() == 'N' and not a.GetIsAromatic()]
        assert len(exo_n) == 1, f"expected exactly one exocyclic N in {smiles!r}"
        result = _identify_fused_substituent(
            mol, exo_n[0], self._quinoline_core(mol))
        return None if result is None else result['name']

    def test_bare_phenyl_is_the_bare_retained_prefix(self):
        """BB:26151/26306 — no locant of its own, so no enclosing marks."""
        assert self._identify("c1ccc(Nc2ccc3ccccc3n2)cc1") == "anilino"

    @pytest.mark.parametrize("smiles,expected,fabricated", [
        # the Blue Book '4-methylanilino (preferred prefix)'
        ("Cc1ccc(Nc2ccc3ccccc3n2)cc1", "(4-methylanilino)", "heptylamino"),
        ("CCc1ccc(Nc2ccc3ccccc3n2)cc1", "(4-ethylanilino)", "octylamino"),
    ])
    def test_all_carbon_decorated_ring_is_not_named_by_carbon_count(
            self, smiles, expected, fabricated):
        got = self._identify(smiles)
        assert got != fabricated, (
            f"{smiles} named the aromatic ring by its carbon count as "
            f"{fabricated!r} — a different molecule"
        )
        assert got == expected

    def test_an_all_carbon_FUSED_n_aryl_keeps_its_real_ring_name(self):
        """A naphthalenyl N-substituent is all-carbon, so ``_bfs_alkyl_from``
        accepts it, and it is not a C6H5- group so the anilino primitive declines
        it — but ``name_substituent_fragment`` DOES name it, so it must keep that
        name rather than be refused.

        The carbon-count guard is deliberately narrow for exactly this reason: a
        broader "any ring atom" version refused this case, trading a structurally
        correct name for an abstention."""
        got = self._identify("c1ccc2cc(Nc3ccc4ccccc4n3)ccc2c1")
        assert got == "naphthalen-2-ylamino", got
        assert got != "decylamino", (
            "a fused naphthalenyl ring was named by its carbon count"
        )

    def test_a_genuine_alkyl_branch_still_resolves(self):
        """The carbon-count fail-closed must not swallow real alkyls."""
        assert self._identify("CC(C)Nc1ccc2ccccc2n1") == "propan-2-ylamino"


class TestSubstituentEnumeratorEnclosureContract:
    """``substituent_enumerator._name_amino_branch`` must return the BARE core.

    That producer's contract is to hand back an unenclosed prefix and let the
    caller apply the marks — its sibling returns are bare too. Passing
    the pre-enclosed form double-wrapped it, e.g.
    ``4-ethylcyclohexan-1-yl[(4-ethylanilino)]methanethioic O-acid``. The call
    site pins ``enclose=False``; this test pins that it stays pinned.
    """

    def test_returns_the_bare_prefix_without_enclosing_marks(self):
        from rdkit import Chem
        from orthonym.assembly.substituent_enumerator import _name_amino_branch
        mol = Chem.MolFromSmiles("OCCCCNc1ccc(Cl)cc1")
        assert mol is not None
        n_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'N'][0]
        parent = {a.GetIdx() for a in mol.GetAtoms()
                  if a.GetSymbol() == 'O'
                  or (a.GetSymbol() == 'C' and not a.GetIsAromatic())}
        frag, stack = {n_idx}, [n_idx]
        while stack:
            for nbr in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                idx = nbr.GetIdx()
                if idx not in frag and idx not in parent:
                    frag.add(idx)
                    stack.append(idx)
        got = _name_amino_branch(mol, sorted(frag), n_idx, sorted(parent))
        assert got == "4-chloroanilino", got
        assert not got.startswith(("(", "[")), (
            f"{got!r} is pre-enclosed; the caller applies the marks, so this "
            f"double-wraps downstream"
        )
