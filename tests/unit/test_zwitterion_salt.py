"""Zwitterion (P-74) + salt (P-65.6.2.1) naming — Phase 169.6 Plan 04.

CHOKE-01 / CHOKE-02 GUARD 4. Every test cites the governing Blue Book P-rule.

GUARD 4 (P-74.0, verbatim): "an anionic center has priority over a cationic
center in zwitterions ... anionic centers ... become the parent structure, into
which the cationic part is substituted." The cation on a DIFFERENT parent
(P-74.1.3, the betaine quaternary ammonium) is demoted to a structured
(…azaniumyl) substituent prefix; the cation INSIDE the anion's parent hydride
(P-74.1.2, a ring N+) is kept on the parent (deferred to the legacy path this
plan).

OPSIN-FORM NOTE (169.6-04 deviation): the Blue Book PIN for a quaternary
ammonium cation substituent is the `-aminiumyl` form
(`(N,N-dimethylmethanaminiumyl)acetate`). OPSIN 2.9.0 does NOT parse the
`-aminiumyl` substituent suffix but DOES round-trip the equivalent azane-based
PIN `(trimethylazaniumyl)acetate` -> `C[N+](C)(C)CC(=O)[O-]`. Accuracy is the #1
priority (the contributor guide) and the byte-identical / RT gate forbids shipping an
unparseable name when a round-tripping equivalent exists, so the producer emits
the `…azaniumyl` form (RT=1, a strict improvement over the old betaine ->
'unknown organic compound').
"""

import pytest
from rdkit import Chem

from orthonym.rules.charged_router import route_charged
from orthonym.assembly.substituent_naming import cation_to_prefix
from orthonym.perception.ions import get_ion_sites


def _rc(smiles, style="pin"):
    return route_charged(Chem.MolFromSmiles(smiles), style)


def _cation_prefix(smiles):
    """Helper: drive cation_to_prefix from a zwitterion SMILES."""
    from rdkit.Chem import rdmolops
    mol = Chem.MolFromSmiles(smiles)
    sites = get_ion_sites(mol)
    ci = sites["cations"][0]["atom_idx"]
    ai = sites["anions"][0]["atom_idx"]
    path = rdmolops.GetShortestPath(mol, ci, ai)
    return cation_to_prefix(mol, ci, path[1])


@pytest.mark.unit
class TestCationToPrefixProducer:
    """The NEW structured cation-as-substituent prefix producer (P-74.1.3).

    substituent_naming.py had ZERO cation-prefix capability before this plan."""

    def test_trimethyl_quaternary_ammonium(self):
        """(CH3)3N+- on an anion parent -> trimethylazaniumyl (P-74.1.3)."""
        assert _cation_prefix("C[N+](C)(C)CC(=O)[O-]") == "trimethylazaniumyl"

    def test_mixed_n_substituents_alphabetized_multiplied(self):
        """N-ethyl-N,N-dimethyl -> ethyldimethylazaniumyl (alphabetical + di-)."""
        assert _cation_prefix("CC[N+](C)(C)CC(=O)[O-]") == "ethyldimethylazaniumyl"

    def test_bare_protonated_nitrogen(self):
        """The producer itself yields ``azaniumyl`` for a bare [NH3+]- (no
        N-substituents). NOTE: GUARD 4 does NOT route a protonated amine through
        this producer (see TestZwitterionGuard4.test_protonated_amine_deferred);
        the producer is exercised directly here to prove the azane base form."""
        assert _cation_prefix("[NH3+]CC(=O)[O-]") == "azaniumyl"


@pytest.mark.unit
class TestZwitterionGuard4:
    """route_charged GUARD 4 = the anion-is-parent override (P-74.0)."""

    def test_betaine_p74_1_3(self):
        """THE worked target. Betaine (CH3)3N+-CH2-COO- -> the carboxylate is the
        parent (P-74.0) and the quaternary ammonium is the (trimethylazaniumyl)
        prefix (P-74.1.3). OPSIN round-trips this to C[N+](C)(C)CC(=O)[O-]."""
        assert _rc("C[N+](C)(C)CC(=O)[O-]") == "(trimethylazaniumyl)acetate"

    def test_betaine_homolog_carries_locant(self):
        """A longer-chain betaine homolog cites the attachment locant (PIN cites
        all locants — the contributor guide pitfall 3). 4-carbon -> 4-(trimethylazaniumyl)
        butanoate (RT-verified)."""
        assert _rc("C[N+](C)(C)CCCC(=O)[O-]") == "4-(trimethylazaniumyl)butanoate"

    def test_betaine_propanoate_homolog(self):
        """3-carbon betaine homolog -> 3-(trimethylazaniumyl)propanoate (RT)."""
        assert _rc("C[N+](C)(C)CCC(=O)[O-]") == "3-(trimethylazaniumyl)propanoate"

    def test_protonated_amine_deferred(self):
        """D-06: GUARD 4's azaniumyl prefix is the QUATERNARY-ammonium betaine
        class. A PROTONATED amine (NH3+, >0 H) neutralizes to a free amino
        SUBSTITUENT, so amino-acid zwitterions / zwitterionic peptides are named
        by their established neutral / retained / peptide form -> route_charged
        defers (''). (This keeps L-alanyl-L-valine, 2-amino-4-oxopentanoic acid
        etc. byte-identical — no malformed 2-(azaniumyl)4-oxo... emission.)"""
        assert _rc("[NH3+]CC(=O)[O-]") == ""                 # glycine zwitterion
        assert _rc("CC(=O)CC([NH3+])C(=O)[O-]") == ""        # 2-amino-4-oxopentanoate

    def test_p74_1_2_ring_cation_deferred(self):
        """P-74.1.2: the cation N+ is SKELETAL to the anion's parent ring
        (pyridinium-2-carboxylate) -> the cumulative ium+ate suffix is out of
        scope this plan -> route_charged defers (returns '')."""
        assert _rc("O=C([O-])c1cccc[n+]1C") == ""

    def test_ylide_amine_oxide_honest_fail(self):
        """P-74.2 dipolar / non-N onium cations are out of scope -> '' (honest-
        fail, D-06). An amine-oxide (N+-O- directly bonded) is an INTERNAL charge
        (P-59), so it is not even a zwitterion -> ''."""
        # Trimethylamine N-oxide: the N+-O- is an internal (P-59) charge, not a
        # zwitterion -> route_charged sees no ionic site -> ''.
        assert _rc("C[N+](C)(C)[O-]") == ""


@pytest.mark.unit
class TestSaltComposition:
    """Salt = cation word(s) (alphabetical) + anion as separate words
    (P-65.6.2.1). The anion is named via route_charged; the cation word comes
    from data/cation_words.py (reusing namer._METAL_NAMES + NH4->ammonium)."""

    def _salt(self, smiles, style="pin"):
        from orthonym.rules.salts import name_salt
        return name_salt(Chem.MolFromSmiles(smiles), style)

    def test_potassium_propanoate(self):
        """THE worked target. CH3CH2COO- K+ -> potassium propanoate (P-65.6.2.1:
        cation word + anion, separate words). RT-verified."""
        assert self._salt("CCC(=O)[O-].[K+]") == "potassium propanoate"

    def test_two_cations_alphabetical(self):
        """K+ -OOC-CH2CH2-COO- Na+ -> potassium sodium ... (cations ALPHABETICAL:
        potassium < sodium; P-65.6.2.1)."""
        name = self._salt("[K+].[O-]C(=O)CCC(=O)[O-].[Na+]")
        assert name.startswith("potassium sodium ")

    def test_calcium_diacetate_stoichiometry(self):
        """(CH3COO-)2 Ca2+ -> calcium diacetate (stoichiometric di- prefix)."""
        assert self._salt("[Ca+2].[O-]C(C)=O.[O-]C(C)=O") == "calcium diacetate"

    def test_ammonium_cation_word(self):
        """NH4+ -> 'ammonium' (P-73.1.1) from the CATION_WORDS table."""
        name = self._salt("[NH4+].CC(=O)[O-]")
        assert name == "ammonium acetate"

    def test_inorganic_salt_retained(self):
        """A purely inorganic salt ([Na+][Cl-]) keeps the retained-table anion
        word (chloride) — route_charged does not name a bare halide."""
        assert self._salt("[Na+].[Cl-]") == "sodium chloride"
