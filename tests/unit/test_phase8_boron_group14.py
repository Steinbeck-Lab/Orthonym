""" a phase — P-68.1/.2 boron + Group-14 depth.

Covers (all OPSIN-RT-verified):
  * 8a boron parent acids (boric/boronic/borinic) + borono ring-propagation;
  * 8b Si/Ge tetrahalides (tetrafluorosilane / tetrachlorogermane);
  * 8c multi-suffix amine spelling (silanetetramine) + the Group-14 silyl/germyl
    substituent namer (silyl / trihydroxysilyl) on ring parents;
  * 8d terminal-Group-14 skeletal-walk fail-closed (covered in
    test_skeletal_replacement.py::TestTerminalGroup14Gate).

The aliphatic-chain silyl-on-senior-carbon case is DEFERRED (the carboxylic-acid
chain handler uses its own substituent path, not the universal pipeline; the
silyl substituent IS correctly named by both substituent namers but dropped by
that handler — production stays a safe 'unknown' via SELF-01). The benzene/ring
silyl path IS fixed here.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym


@pytest.fixture(scope="module")
def st():
    return Orthonym()


class TestBoronAcids:
    @pytest.mark.parametrize("smiles,expected", [
        ("OB(O)O", "boric acid"),       # B(OH)3
        ("OBO", "boronic acid"),        # HB(OH)2 parent
        ("BO", "borinic acid"),         # H2B-OH parent
    ])
    def test_boron_parent_acids(self, st, smiles, expected):
        assert st.name(smiles) == expected

    def test_substituted_boronic_acid_unchanged(self, st):
        # R-boronic acids carry carbon -> never match the exact carbon-free table.
        assert st.name("CB(O)O") == "methylboronic acid"
        assert st.name("OB(O)c1ccccc1") == "phenylboronic acid"


class TestBoronoRingPropagation:
    @pytest.mark.parametrize("smiles,expected", [
        ("OC(=O)c1ccc(B(O)O)cc1", "4-boronobenzoic acid"),
        ("OB(O)c1ccc(C(=O)O)c(c1)[N+](=O)[O-]", "4-borono-2-nitrobenzoic acid"),
        ("OB(O)CCC(=O)O", "3-boronopropanoic acid"),  # chain (already worked)
    ])
    def test_borono_prefix(self, st, smiles, expected):
        assert st.name(smiles) == expected


class TestGroup14Halides:
    @pytest.mark.parametrize("smiles,expected", [
        ("F[Si](F)(F)F", "tetrafluorosilane"),
        ("Cl[Si](Cl)(Cl)Cl", "tetrachlorosilane"),
        ("F[Ge](F)(F)F", "tetrafluorogermane"),
        ("Cl[Ge](Cl)(Cl)Cl", "tetrachlorogermane"),
    ])
    def test_tetrahalides(self, st, smiles, expected):
        assert st.name(smiles) == expected

    def test_tetramethylsilane_unchanged(self, st):
        assert st.name("C[Si](C)(C)C") == "tetramethylsilane"


class TestMultiSuffixSpelling:
    @pytest.mark.parametrize("smiles,expected", [
        ("N[Si](N)(N)N", "silanetetramine"),     # BB P-68.2.4: not silanetetraamine
        ("N[Ge](N)(N)N", "germanetetramine"),
        ("C[Si](N)(N)N", "methylsilanetriamine"),  # tri unaffected
        ("C[Si](C)(O)O", "dimethylsilanediol"),    # di unaffected
    ])
    def test_group14_multi_suffix(self, st, smiles, expected):
        assert st.name(smiles) == expected

    # P-62.2.1.3 (BB 26221): the MONO-amine 'silanamine' is the one documented
    # exception to P-14.3.4.2(a) (BB 2891, "the locant '1' is omitted in
    # substituted mononuclear parent hydrides") -- (CH3)3Si-NH2 cites its
    # substituent locants: '1,1,1-trimethylsilanamine' (PIN, corroborated at
    # BB 37493/37495). Every sibling KEEPS the general omission: -ol/-thiol,
    # the DI-/TRI-amine (methylsilanetriamine, BB 38180), etc.
    @pytest.mark.parametrize("smiles,expected", [
        ("C[Si](C)(C)N", "1,1,1-trimethylsilanamine"),   # mono-amine: LOCANTS
        ("CC[Si](CC)(CC)N", "1,1,1-triethylsilanamine"),
    ])
    def test_mono_silanamine_cites_locants(self, st, smiles, expected):
        assert st.name(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # P-14.3.4.2(a) omission siblings that MUST NOT gain locants.
        ("C[Si](C)(C)O", "trimethylsilanol"),   # mono-ol, BB 27234
        ("C[Si](N)(N)N", "methylsilanetriamine"),  # tri-amine, BB 38180
        ("C[Si](C)(N)N", "dimethylsilanediamine"),  # di-amine
    ])
    def test_group14_locant_omission_unregressed(self, st, smiles, expected):
        assert st.name(smiles) == expected


class TestGroup14SubstituentNamer:
    """The shared _name_group14_substituent (silyl/germyl + prefixes)."""

    @pytest.mark.parametrize("smiles,attach_sym,expected", [
        ("C[SiH3]", "Si", "silyl"),
        ("C[Si](C)(C)C", "Si", "trimethylsilyl"),
        ("C[Si](O)(O)O", "Si", "trihydroxysilyl"),
        ("C[Si](C)(C)O", "Si", "hydroxydimethylsilyl"),
        ("C[GeH3]", "Ge", "germyl"),
    ])
    def test_detached_fragment(self, smiles, attach_sym, expected):
        from orthonym.assembly.substituent_naming import _name_group14_substituent
        mol = Chem.MolFromSmiles(smiles)
        attach = [a.GetIdx() for a in mol.GetAtoms()
                  if a.GetSymbol() == attach_sym][0]
        frag = set(range(mol.GetNumAtoms())) - {0}  # exclude the parent methyl C0
        assert _name_group14_substituent(mol, list(frag), attach) == expected

    def test_multivalent_fail_closed(self):
        # A silanediyl bridge (>=2 external bonds) is out of scope -> None.
        from orthonym.assembly.substituent_naming import _name_group14_substituent
        mol = Chem.MolFromSmiles("C[SiH2]C")  # dimethylsilane: Si bridges 2 C
        si = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Si"][0]
        # treat both methyls as "parent" -> Si has 2 external bonds
        assert _name_group14_substituent(mol, [si], si) is None


class TestBenzeneSilyl:
    @pytest.mark.parametrize("smiles,expected", [
        ("OC(=O)c1ccc([SiH3])cc1", "4-silylbenzoic acid"),
        ("OC(=O)c1ccc([Si](O)(O)O)cc1", "4-(trihydroxysilyl)benzoic acid"),
        ("OC(=O)c1ccc([GeH3])cc1", "4-germylbenzoic acid"),
        ("OC(=O)c1ccc([Si](C)(C)C)cc1", "4-(trimethylsilyl)benzoic acid"),
    ])
    def test_ring_silyl_prefix(self, st, smiles, expected):
        assert st.name(smiles) == expected
