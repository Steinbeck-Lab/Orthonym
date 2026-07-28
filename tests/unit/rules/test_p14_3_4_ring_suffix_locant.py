"""P-14.3.4.2(c) — omission of a trivial mono-suffix locant on a ring (v29 Phase C tranche A).

Governing rule chain, verbatim from ``BlueBookV2/BlueBookV2.md``:

``P-14.3.3`` "Citation of locants" (``:2869``) is **DENY BY DEFAULT** —

    "if any locants are essential … then all locants must be cited for the parent
     structure or that structural unit"

so ``P-14.3.4`` "Omission of locants" (``:2871``) grants narrow LICENCES, and its own
preamble says *"for absolute clarity in preferred IUPAC names it is necessary to be
prescriptive about when omission of locants is permissible."*

The licence exercised here, P-14.3.4.2(c), is witnessed verbatim:

    ``:14916``  "by suffixes, such as 'cyclohexanecarboxylic acid' (PIN) and
                 'cyclohexanone' (PIN)"
    ``:26854``  "(1) cyclopentanol (PIN)"

ROOT CAUSE this file guards (found by a validated call-spy, 2026-07-28):
``_handler_shared._generate_suffix`` gates its whole P-14.3.4 elision block — including
the ``should_omit_locant_one`` call — on ``features.principal_chain`` being non-empty
(``:411``). A RING parent has no principal chain, so **rings never reached the elision
logic at all**, and the non-terminal ring branch cited the locant unconditionally. The
sibling TERMINAL ring branch already withheld it, which is why ``cyclohexanecarbaldehyde``
was correct while ``cyclohexan-1-one`` was not.

⚠ Invariant 11: removing a locant can unmask something worse. Every guard below asserts the
FULL emitted name, not merely that a locant vanished.
"""
import pytest

from orthonym.namer import Orthonym


def _name(smiles: str) -> str:
    return Orthonym().name(smiles)


# --------------------------------------------------------------------------- #
# 1. The licence fires — trivial mono-suffix locant on a saturated carbocycle  #
# --------------------------------------------------------------------------- #
class TestLicensedOmission:
    @pytest.mark.parametrize("smiles,expected,authority", [
        ("O=C1CCCCC1", "cyclohexanone", "verbatim BB:14916"),
        ("OC1CCCC1", "cyclopentanol", "verbatim BB:26854"),
        ("SC1CCCCC1", "cyclohexanethiol", "P-14.3.4.2(c) example"),
        ("OC1CCCCC1", "cyclohexanol", "P-14.3.4.2(c)"),
        ("NC1CCCCC1", "cyclohexanamine", "derived, P-14.3.4.2(c)"),
        ("OC1CCC1", "cyclobutanol", "derived, P-14.3.4.2(c)"),
    ])
    def test_trivial_ring_suffix_locant_is_omitted(self, smiles, expected, authority):
        assert _name(smiles) == expected, authority


# --------------------------------------------------------------------------- #
# 2. The deny-default holds — every one of these MUST keep its locant          #
#    (the Part-D tripwire set; each asserts the whole name)                    #
# --------------------------------------------------------------------------- #
class TestDenyByDefault:
    @pytest.mark.parametrize("smiles,expected,why", [
        # A second suffix makes the locants essential.
        ("OC1CCCCC1O", "cyclohexane-1,2-diol", "multiplied suffix"),
        # Any other cited substituent restores them.
        ("O=C1CCC(C)CC1", "4-methylcyclohexan-1-one", "ring carries a methyl"),
        ("OC1CCCCC1C", "2-methylcyclohexan-1-ol", "ring carries a methyl"),
        # ★ A substituent on an ALPHA carbon. These two were found by the GATE as
        # protect regressions, not by this file — the first version of the predicate
        # exempted every atom of `features.principal_group_atoms` from the CH2 test,
        # and a ring ketone's SMARTS match spans BOTH alpha carbons
        # (`O=C1CCCCC1CCCCC` -> match `(2, 1, 0, 6)`, atom 6 bearing the pentyl), so
        # an alpha substituent was invisible. `4-methyl...` above passed throughout
        # because position 4 is not an alpha carbon. The exemption is now the
        # suffix-BEARING ring atom only, derived from the cited locants.
        ("O=C1CCCCC1CCCCC", "2-pentylcyclohexan-1-one", "alpha-carbon substituent"),
        ("O=C1CCCC1N1CCCC1", "2-(pyrrolidin-1-yl)cyclopentan-1-one",
         "alpha-carbon substituent"),
        # Ring unsaturation makes positions distinct.
        ("OC1CC=CCC1", "cyclohex-3-en-1-ol", "ring double bond"),
        # A heterocycle keeps it (cf. piperidine-1-carbonitrile, BB:34730).
        ("OC1CCNCC1", "piperidin-4-ol", "heterocycle"),
        # ★ imine is EXCLUDED from the licence: no verbatim bare '-imine' row exists,
        # and licensing it shipped a defect the GATE caught. For an oxime the OH is
        # STRIPPED from `features.mol` before the suffix decision and re-added as an
        # `N-hydroxy` prefix, so the N-substituent is invisible at that point -- both
        # `features.mol` and `features.canonical_smiles` carry the reduced form of
        # `ON=C1CCCCC1`. The licence fired and emitted `N-hydroxycyclohexanimine`, but
        # `N` is an essential locant in the same scope and P-14.3.3 (BB:2869) restores
        # every locant in a scope once one is essential.
        ("N=C1CCCCC1", "cyclohexan-1-imine", "imine: no verbatim BB row"),
        ("ON=C1CCCCC1", "N-hydroxycyclohexan-1-imine",
         "N-substituent invisible to features.mol -- gate-found"),
        # A multi-atom suffix is out of scope for tranche A (needs whole-locant-set
        # suppression); the ring+1-heavy-atom condition denies it.
        ("OOC(=O)C1CCCCC1", "cyclohexane-1-carboperoxoic acid",
         "multi-atom suffix, tranche B"),
        # Chains are untouched by this change.
        ("OCCCl", "2-chloroethan-1-ol", "BB verbatim: NOT 2-chloroethanol"),
        ("OCC", "ethanol", "already correct, must not regress"),
        ("OCCO", "ethane-1,2-diol", "already correct"),
        ("ClCCCCC", "1-chloropentane", "already correct"),
        # Unsubstituted ring hydrocarbons / arenes are unaffected.
        ("C1=CCCCC1", "cyclohexene", "bond locant, different licence"),
        ("Oc1ccccc1", "phenol", "retained name"),
    ])
    def test_locant_is_retained(self, smiles, expected, why):
        assert _name(smiles) == expected, why

    def test_a_fused_ring_keeps_its_suffix_locant(self):
        """A fused system numbers non-equivalently, so the licence must decline.

        Asserts on the PREDICATE rather than the emitted name, because
        ``Oc1cccc2ccccc12`` resolves to the pre-existing retained name
        ``1-naphthol`` before the suffix path is reached — which would make an
        end-to-end assertion here pass for the wrong reason.
        """
        from rdkit import Chem
        from orthonym.assembly.handlers._handler_shared import (
            _ring_suffix_locant_is_trivial,
        )

        mol = Chem.MolFromSmiles("Oc1cccc2ccccc12")
        assert mol is not None
        fused_ring = next(r for r in mol.GetRingInfo().AtomRings())

        class _F:
            pass

        feats = _F()
        feats.mol = mol
        feats.principal_group_atoms = []
        feats.principal_group = "secondary_alcohol"
        feats.canonical_smiles = Chem.MolToSmiles(mol)
        assert _ring_suffix_locant_is_trivial(
            feats, list(fused_ring),
            {a: i + 1 for i, a in enumerate(fused_ring)},
        ) is False


# --------------------------------------------------------------------------- #
# 3. The predicate itself, unit-level                                         #
# --------------------------------------------------------------------------- #
class TestPredicateDeniesByDefault:
    @staticmethod
    def _probe(smiles: str, pg_smarts: str, fg_name: str = "ketone") -> bool:
        """Build the real `features`-shaped inputs the producer passes."""
        from rdkit import Chem
        from orthonym.assembly.handlers._handler_shared import (
            _ring_suffix_locant_is_trivial,
        )
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        patt = Chem.MolFromSmarts(pg_smarts)
        matches = [tuple(m) for m in mol.GetSubstructMatches(patt)]
        ring = next((r for r in mol.GetRingInfo().AtomRings()), ())

        class _F:
            pass
        feats = _F()
        feats.mol = mol
        feats.principal_group_atoms = matches
        # The predicate also reads the suffix CLASS (allowlist) and re-checks the
        # ORIGINAL input, because `features.mol` is not always the input molecule.
        feats.principal_group = fg_name
        feats.canonical_smiles = Chem.MolToSmiles(mol)
        ring_map = {a: i + 1 for i, a in enumerate(ring)}
        # The producer passes the suffix-BEARING ring atom(s), not the whole SMARTS
        # match -- here, the ring atom carrying the exocyclic characteristic
        # heteroatom. Mirror that contract exactly.
        match_atoms = {a for m in matches for a in m}
        suffix_ring_atoms = {
            a for a in ring
            if any(nbr.GetIdx() in match_atoms and nbr.GetIdx() not in set(ring)
                   for nbr in mol.GetAtomWithIdx(a).GetNeighbors())
        }
        return _ring_suffix_locant_is_trivial(
            feats, list(ring), ring_map, suffix_ring_atoms=suffix_ring_atoms)

    def test_saturated_carbocyclic_ketone_is_licensed(self):
        assert self._probe("O=C1CCCCC1", "[CX3]=[OX1]") is True

    def test_alpha_substituted_ring_ketone_is_denied(self):
        """The gate-found regression, at predicate level."""
        assert self._probe("O=C1CCCCC1CCCCC", "[CX3]=[OX1]") is False

    def test_missing_mol_denies(self):
        from orthonym.assembly.handlers._handler_shared import (
            _ring_suffix_locant_is_trivial,
        )

        class _F:
            pass
        feats = _F()
        feats.mol = None
        feats.principal_group_atoms = []
        feats.principal_group = "ketone"
        feats.canonical_smiles = "CCO"
        assert _ring_suffix_locant_is_trivial(feats, [0, 1, 2], {0: 1}) is False

    def test_empty_ring_denies(self):
        from orthonym.assembly.handlers._handler_shared import (
            _ring_suffix_locant_is_trivial,
        )
        from rdkit import Chem

        class _F:
            pass
        feats = _F()
        feats.mol = Chem.MolFromSmiles("CCO")
        feats.principal_group_atoms = []
        feats.principal_group = "ketone"
        feats.canonical_smiles = "CCO"
        assert _ring_suffix_locant_is_trivial(feats, [], {}) is False

    def test_imine_class_is_not_licensed(self):
        """The suffix-class allowlist, at predicate level."""
        assert self._probe("N=C1CCCCC1", "[CX3]=[NX2]", fg_name="imine") is False
        assert self._probe("O=C1CCCCC1", "[CX3]=[OX1]", fg_name="ketone") is True
