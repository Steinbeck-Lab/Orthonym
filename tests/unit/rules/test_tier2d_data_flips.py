"""Wave2 — retained alkoxides, bare parents, HW indicated-H, irine stem.

Four data-level fixes / / /:

1. RETAINED ALKOXIDES, BB verbatim): methoxide/ethoxide/propoxide/
   butoxide/phenoxide/tert-butoxide are the PINs — the salt path must hit
   them before the systematic -olate chokepoint. isopropoxide is general
   nomenclature only (PIN = propan-2-olate) and is demoted.

2. BARE PARENTS: H2N-OH = 'hydroxylamine' and HN=N-CH=N-NH2 = 'formazan'
   (retained PINs; the perception SMARTS needs a C so the bare forms fell
   through to unknown / a functional-class hydrazone name).

3. MONOCYCLIC HW INDICATED HYDROGEN: a mancude monocycle
   with exactly one sp3 H-bearing eligible atom cites it (2H-1,3-dioxole,
   2H-/4H-pyran, 4H-thiopyran); the old hardcoded 'oxine'->'2H-pyran' map
   mislabelled 4H tautomers. Fail-closed for hydro forms and
   multi-indicated-H rings.

4. IRINE STEM: 3-membered mancude N-only rings use
   'irine' (1H-/2H-azirine), not 'irene'. Direction tie-break gives the
   indicated H the lowest locant (2H-azirine, not 3H-azirine).
"""

import pytest

from orthonym import name_compound


@pytest.mark.unit
class TestRetainedAlkoxides:
    """: retained alkoxide PINs win over systematic -olate in salts."""

    @pytest.mark.parametrize("smiles,expected", [
        ("[Na+].[O-]C", "sodium methoxide"),
        ("[K+].[O-]CC", "potassium ethoxide"),
        ("[Na+].[O-]CCC", "sodium propoxide"),
        ("[Na+].[O-]CCCC", "sodium butoxide"),
        ("[Na+].[O-]C(C)(C)C", "sodium tert-butoxide"),
        ("[Na+].[O-]c1ccccc1", "sodium phenoxide"),
        # BB verbatim example: "potassium propan-2-olate (PIN) potassium
        # isopropoxide" — isopropoxide is general nomenclature only
        ("[K+].[O-]C(C)C", "potassium propan-2-olate"),
    ])
    def test_alkoxide_salts(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("C[O-]", "methoxide"),
        ("CCC[O-]", "propoxide"),
        ("CCCC[O-]", "butoxide"),
        ("CC(C)[O-]", "propan-2-olate"),  # isopropoxide demoted
    ])
    def test_standalone_anions(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestBareParents:
    """: bare retained parents previously unreachable."""

    def test_hydroxylamine(self):
        assert name_compound("NO") == "hydroxylamine"

    def test_formazan(self):
        assert name_compound("NN=CN=N") == "formazan"

    @pytest.mark.parametrize("smiles,expected", [
        # neighbors of the new exact-SMILES keys must be untouched
        ("N", "ammonia"),
        ("O", "water"),
        ("NN", "hydrazine"),
        ("CCNO", "N-ethylhydroxylamine"),
    ])
    def test_neighbors_unchanged(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestIndicatedHydrogen:
    """: mancude-monocycle indicated H, fail-closed scope."""

    @pytest.mark.parametrize("smiles,expected", [
        ("O1C=COC1", "2H-1,3-dioxole"),
        ("C1=CCOC=C1", "2H-pyran"),
        ("C1C=COC=C1", "4H-pyran"),
        ("C1=CSC=CC1", "4H-thiopyran"),   # was unknown (hardcoded 2H- only)
    ])
    def test_indicated_h_emitted(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # saturated / aromatic rings must NOT gain a spurious nH-
        ("C1CCOC1", "oxolane"),
        ("C1COCCN1", "morpholine"),
        ("C1CCNCC1", "piperidine"),
        ("O1CCOCC1", "1,4-dioxane"),
        ("c1ccoc1", "furan"),
        ("c1cc[nH]c1", "1H-pyrrole"),
        ("c1ccncc1", "pyridine"),
        # pyranones own their indicated H via the pseudoketone path
        ("O=c1ccocc1", "4H-pyran-4-one"),
        ("O=c1cccco1", "2H-pyran-2-one"),
        ("Cc1cc(=O)cco1", "2-methyl-4H-pyran-4-one"),
    ])
    def test_protected_rings(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestIrineStem:
    """: N-only 3-ring mancude stem is 'irine'."""

    def test_1h_azirine(self):
        # C=C double bond, N-H sp3 -> indicated H on N1
        assert name_compound("C1=CN1") == "1H-azirine"

    def test_2h_azirine(self):
        # C=N double bond, CH2 sp3 -> lowest locant 2 (not 3H-azirine)
        assert name_compound("N1=CC1") == "2H-azirine"

    def test_aziridine_unchanged(self):
        assert name_compound("C1CN1") == "aziridine"

    def test_oxirane_unchanged(self):
        # non-N 3-ring keeps irane/irene family
        assert name_compound("C1CO1") == "oxirane"
