"""v22 Phase E1 (DD4) — numbering + locant determinism.

Rule-family coverage (guardrail A8: test the NAME OUTPUT, parameterized by rule
family, not a literal canary) for the three numbering engines routed through the
single shared ``compare_numbering`` comparator + ``ELEMENT_NUMBERING_SENIORITY``:

  * H2 skeletal-replacement element-seniority tie-break (P-15.4.1.2)
  * H1 fused-PAH / fused-heterocycle automorphism-minimization (P-25.3.3.1.2(a))
  * acridine ``iupac_locants`` data correction (re-derived via OPSIN)
  * benzene P-14.4(c) PCG-anchor tier (additive, default-off)
  * determinism: name(SMILES) == name(seeded re-spellings)
"""
import random

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.rules.locants import (
    ELEMENT_NUMBERING_SENIORITY,
    element_seniority_rank,
    compare_numbering,
)


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _name(namer, smiles):
    return namer.name(smiles)


def _respell_names(namer, smiles, k=6):
    """Names produced from k seeded atom-permutation re-spellings (+ input)."""
    mol = Chem.MolFromSmiles(smiles)
    names = {namer.name(smiles)}
    for seed in range(k):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        names.add(namer.name(Chem.MolToSmiles(Chem.RenumberAtoms(mol, order))))
    return names


# --------------------------------------------------------------------------- #
# ELEMENT_NUMBERING_SENIORITY (single source of truth) + derived table         #
# --------------------------------------------------------------------------- #
class TestElementSeniorityConstant:
    def test_canonical_order_prefix(self):
        order = "F Cl Br I At O S Se Te Po N P As Sb Bi C Si Ge Sn Pb B".split()
        ranks = [ELEMENT_NUMBERING_SENIORITY[s] for s in order]
        assert ranks == sorted(ranks), "ELEMENT_NUMBERING_SENIORITY must be ascending"

    @pytest.mark.parametrize("senior,junior", [
        ("F", "O"), ("O", "S"), ("S", "N"), ("O", "N"), ("N", "C"),
        ("C", "Si"), ("Si", "B"), ("Br", "I"),
    ])
    def test_pairwise_seniority(self, senior, junior):
        assert element_seniority_rank(senior) < element_seniority_rank(junior)

    def test_unknown_element_sorts_last(self):
        assert element_seniority_rank("Xx") > element_seniority_rank("Tl")

    def test_ring_substituent_table_is_derived_and_byte_identical(self):
        # The numbering consumer must reproduce the historic dense ranks exactly.
        from orthonym.rules.ring_substituents import _HETEROATOM_SENIORITY
        expected = {
            'F': 0, 'Cl': 1, 'Br': 2, 'I': 3, 'O': 4, 'S': 5, 'Se': 6, 'Te': 7,
            'N': 8, 'P': 9, 'As': 10, 'Sb': 11, 'Bi': 12, 'Si': 13, 'Ge': 14,
            'Sn': 15, 'Pb': 16, 'B': 17,
        }
        assert _HETEROATOM_SENIORITY == expected


# --------------------------------------------------------------------------- #
# compare_numbering tier behaviour                                             #
# --------------------------------------------------------------------------- #
class TestCompareNumbering:
    def test_pcg_tier_dominates(self):
        a = {"pcg": [2], "substituents": [1]}
        b = {"pcg": [3], "substituents": [1]}
        assert compare_numbering(a, b) == -1

    def test_heteroatom_positional_first(self):
        a = {"heteroatoms": [(2, "S"), (4, "O")]}
        b = {"heteroatoms": [(3, "O"), (5, "S")]}
        assert compare_numbering(a, b) == -1  # {2,4} < {3,5} regardless of element

    def test_heteroatom_element_seniority_on_positional_tie(self):
        # COCSC: positions {2,4} either way -> senior O wins locant 2.
        fwd = {"heteroatoms": [(2, "O"), (4, "S")]}
        rev = {"heteroatoms": [(2, "S"), (4, "O")]}
        assert compare_numbering(fwd, rev) == -1

    def test_trisilanonane_worked_example(self):
        # 2-oxa-4,6,8-trisilanonane: sets {2,4,6,8} tie -> O over Si at locant 2.
        a = {"heteroatoms": [(2, "O"), (4, "Si"), (6, "Si"), (8, "Si")]}
        b = {"heteroatoms": [(2, "Si"), (4, "Si"), (6, "Si"), (8, "O")]}
        assert compare_numbering(a, b) == -1

    def test_substituent_tier(self):
        assert compare_numbering({"substituents": [2, 4]}, {"substituents": [3, 5]}) == -1

    def test_genuine_symmetry_ties_to_zero(self):
        a = {"heteroatoms": [(2, "O"), (4, "O")]}
        b = {"heteroatoms": [(2, "O"), (4, "O")]}
        assert compare_numbering(a, b) == 0

    def test_alpha_tier(self):
        # P-14.4(g) lowest locant to the alphabetically-first prefix: the tier
        # compares the supplied sortable keys; None (no prefix at position 1)
        # sorts last.
        assert compare_numbering({"alpha": (("bromo", 1),)},
                                 {"alpha": (("chloro", 1),)}) == -1
        assert compare_numbering({"alpha": (("chloro", 1),)},
                                 {"alpha": (("bromo", 1),)}) == 1
        assert compare_numbering({"alpha": (("a", 1),)}, {"alpha": None}) == -1
        assert compare_numbering({"alpha": None}, {"alpha": (("a", 1),)}) == 1

    def test_pcg_outranks_substituents(self):
        # WR-01 PAH PCG-anchor: pcg locant set is compared before substituents.
        assert compare_numbering({"pcg": [2], "substituents": [6]},
                                 {"pcg": [6], "substituents": [2]}) == -1


# --------------------------------------------------------------------------- #
# H2 — skeletal-replacement element-seniority determinism                      #
# --------------------------------------------------------------------------- #
class TestSkeletalReplacementSeniority:
    @pytest.mark.parametrize("smiles", ["COCSC", "CSCOC"])
    def test_oxa_thia_carbon_over_ether_pin(self, namer, smiles):
        # v22 Phase E2 (SEN-02): COCSC now routes to the carbon-parent PIN
        # methoxy(methylsulfanyl)methane (P-41 cls 40 > 41/42), not the skeletal
        # 2-oxa-4-thiapentane. E1 made it DETERMINISTIC (both SMILES orders give one
        # name — still true); E2 made it the correct PIN. The O-senior-to-S element
        # numbering tiebreak is still covered directly at the comparator level by
        # test_heteroatom_element_seniority_on_positional_tie + the aza case below.
        assert _name(namer, smiles) == "methoxy(methylsulfanyl)methane"

    @pytest.mark.parametrize("smiles", ["COCNC", "CNCOC"])
    def test_oxa_aza_senior_oxygen_low_locant(self, namer, smiles):
        assert _name(namer, smiles) == "2-oxa-4-azapentane"

    def test_distinct_positional_set_protect(self, namer):
        # Distinct positional set -> the element tier never fires (byte-identical).
        assert _name(namer, "OCCOCCOCCOC") == "3,6,9-trioxadecan-1-ol"

    @pytest.mark.parametrize("smiles", ["COCSC", "CSCOC", "COCNC", "CNCOC"])
    def test_deterministic_across_respellings(self, namer, smiles):
        assert len(_respell_names(namer, smiles)) == 1


# --------------------------------------------------------------------------- #
# H1 — fused PAH automorphism-minimization                                     #
# --------------------------------------------------------------------------- #
class TestPolycyclicAutomorphismMin:
    def test_methylanthracene_pin(self, namer):
        assert _name(namer, "Cc1ccc2cc3ccccc3cc2c1") == "2-methylanthracene"

    def test_anthracene_amine_locant(self, namer):
        # Locant fixed to 2 (the DD4/H1 defect). Prefix-vs-suffix style (PIN
        # anthracen-2-amine) is a separate P-66.1 expression concern out of E1.
        assert _name(namer, "Nc1ccc2cc3ccccc3cc2c1") == "2-aminoanthracene"

    def test_phenanthrene_amine_locant(self, namer):
        assert _name(namer, "Nc1ccc2ccc3ccccc3c2c1") == "3-aminophenanthrene"

    @pytest.mark.parametrize("smiles,expected", [
        ("Cc1ccc2ccccc2c1", "2-methylnaphthalene"),
        ("Cc1cccc2ccccc12", "1-methylnaphthalene"),
    ])
    def test_naphthalene_protect(self, namer, smiles, expected):
        assert _name(namer, smiles) == expected

    def test_pah_pcg_outranks_prefix_substituent(self, namer):
        # WR-01: the principal characteristic group (carboxylic acid suffix) must
        # take the lowest locant before the detachable methyl prefix (P-14.4(c)).
        assert _name(namer, "Cc1ccc2cc(C(=O)O)ccc2c1") == "6-methylnaphthalene-2-carboxylic acid"

    def test_pah_single_suffix_unchanged(self, namer):
        # PCG-anchor is a no-op for a single suffix (byte-identical).
        assert _name(namer, "OC(=O)c1ccc2ccccc2c1") == "naphthalene-2-carboxylic acid"

    @pytest.mark.parametrize("smiles", [
        "Cc1ccc2cc3ccccc3cc2c1", "Nc1ccc2cc3ccccc3cc2c1",
        "Nc1ccc2ccc3ccccc3c2c1",
    ])
    def test_deterministic_across_respellings(self, namer, smiles):
        assert len(_respell_names(namer, smiles)) == 1

    def test_populated_pahs_never_use_naphthalene_heuristic(self, monkeypatch):
        # DD4 regression guard: cataloged PAHs with populated iupac_numbering must
        # resolve substituent locants via the stored map, NOT the naphthalene-only
        # alpha/beta heuristic _map_pah_atoms_to_iupac.
        import orthonym.rules.polycyclics as poly

        calls = []
        orig = poly._map_pah_atoms_to_iupac

        def _spy(mol, pah_name, match_atoms):
            calls.append(pah_name)
            return orig(mol, pah_name, match_atoms)

        monkeypatch.setattr(poly, "_map_pah_atoms_to_iupac", _spy)
        n = Orthonym()
        for smi in ("Cc1ccc2cc3ccccc3cc2c1", "Nc1ccc2ccc3ccccc3c2c1",
                    "Cc1ccc2ccccc2c1"):
            n.name(smi)
        assert calls == [], f"heuristic reached for populated PAHs: {calls}"


# --------------------------------------------------------------------------- #
# Azulene substituent path (now wired via iupac_numbering)                     #
# --------------------------------------------------------------------------- #
class TestAzuleneSubstituentPath:
    def test_bare_azulene(self, namer):
        assert _name(namer, "C1=CC=C2C=CC=CC=C12") == "azulene"

    def test_substituted_azulene_deterministic_and_symmetry_consistent(self, namer):
        # Methylate each peripheral carbon; azulene's 5 unique positions are
        # 1,2,4,5,6 (orbits 1=3, 4=8, 5=7; 2 and 6 unique). Each must be a single
        # deterministic methylazulene locant in that set.
        azulene = Chem.MolFromSmiles("C1=CC=C2C=CC=CC=C12")
        ri = azulene.GetRingInfo()
        from collections import defaultdict
        cnt = defaultdict(int)
        for ring in ri.AtomRings():
            for a in ring:
                cnt[a] += 1
        produced = set()
        for atom in azulene.GetAtoms():
            if atom.GetSymbol() != "C" or cnt[atom.GetIdx()] > 1:
                continue
            rw = Chem.RWMol(azulene)
            c = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(atom.GetIdx(), c, Chem.BondType.SINGLE)
            m = rw.GetMol()
            Chem.SanitizeMol(m)
            smi = Chem.MolToSmiles(m)
            assert len(_respell_names(namer, smi, k=4)) == 1
            produced.add(namer.name(smi))
        assert produced == {
            "1-methylazulene", "2-methylazulene", "4-methylazulene",
            "5-methylazulene", "6-methylazulene",
        }


# --------------------------------------------------------------------------- #
# Acridine data fix + fused-heterocycle automorphism-min                       #
# --------------------------------------------------------------------------- #
class TestAcridineAndFusedHeterocycles:
    def test_acridin_2_amine_pin(self, namer):
        assert _name(namer, "Nc1ccc2nc3ccccc3cc2c1") == "acridin-2-amine"

    def test_bare_acridine(self, namer):
        assert _name(namer, "c1ccc2nc3ccccc3cc2c1") == "acridine"

    def test_acridine_amino_orbit_is_2_7(self):
        # The corrected iupac_locants map must put the two C2v-symmetric images of
        # the substituted carbon at the orbit {2,7} (min -> 2), NOT the broken
        # cross-orbit {3,7}.
        from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
        smi, rec = next((s, r) for s, r in FUSED_HETEROCYCLE_DATA.items()
                        if r.get("name") == "acridine")
        pat = Chem.MolFromSmiles(smi)
        mol = Chem.MolFromSmiles("Nc1ccc2nc3ccccc3cc2c1")
        ci = next(nb.GetIdx()
                  for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetDegree() == 1
                  for nb in a.GetNeighbors())
        locs = set()
        for m in mol.GetSubstructMatches(pat, uniquify=False):
            molidx_to_loc = {m[pi]: L for pi, L in rec["iupac_locants"].items() if pi < len(m)}
            if ci in molidx_to_loc:
                locs.add(molidx_to_loc[ci])
        assert locs == {2, 7}

    @pytest.mark.parametrize("smiles,expected", [
        ("c1ccc2[nH]ccc2c1", "1H-indole"),          # asymmetric core (single match)
        ("Cc1ccc2[nH]ccc2c1", "5-methyl-1H-indole"),
        ("c1ccc2c(c1)[nH]c1ccccc12", "9H-carbazole"),  # symmetric, bare -> no locants
    ])
    def test_asymmetric_and_bare_cores_byte_identical(self, namer, smiles, expected):
        assert _name(namer, smiles) == expected

    def test_acridine_amine_deterministic(self, namer):
        assert len(_respell_names(namer, "Nc1ccc2nc3ccccc3cc2c1")) == 1


# --------------------------------------------------------------------------- #
# Benzene P-14.4(c) PCG tier (additive, default-off)                           #
# --------------------------------------------------------------------------- #
class TestBenzenePcgTier:
    @pytest.mark.parametrize("smiles,expected", [
        ("Oc1ccccc1", "phenol"),
        ("Oc1ccccc1Br", "2-bromophenol"),
        ("Cc1cc(C)cc(O)c1", "3,5-dimethylphenol"),
        ("Cc1ccccc1O", "2-methylphenol"),
        ("Oc1ccccc1C", "2-methylphenol"),
        ("OC(=O)c1ccccc1Cl", "2-chlorobenzoic acid"),
        ("Cc1ccc(C)cc1", "1,4-dimethylbenzene"),
    ])
    def test_protect_rows_byte_identical(self, namer, smiles, expected):
        assert _name(namer, smiles) == expected

    def test_orient_benzene_default_is_noop(self):
        # Without principal_group_positions, orient_benzene is byte-identical.
        from orthonym.rules.benzene import orient_benzene, get_benzene_substituents
        mol = Chem.MolFromSmiles("Oc1ccccc1Br")
        ring = next(r for r in mol.GetRingInfo().AtomRings() if len(r) == 6)
        subs = get_benzene_substituents(mol, tuple(ring))
        base = orient_benzene(mol, tuple(ring), subs)
        again = orient_benzene(mol, tuple(ring), subs, principal_group_positions=None)
        assert base == again

    def test_pcg_tier_anchors_principal_group(self):
        # With a PCG set, the orientation must give the PCG atom the lowest locant.
        from orthonym.rules.benzene import orient_benzene, get_benzene_substituents
        mol = Chem.MolFromSmiles("OC(=O)c1ccccc1Cl")  # 2-chlorobenzoic acid
        ring = next(r for r in mol.GetRingInfo().AtomRings() if len(r) == 6)
        subs = get_benzene_substituents(mol, tuple(ring))
        pcg = {idx for idx, sl in subs.items() if any(s.get("is_suffix") for s in sl)}
        oriented = orient_benzene(mol, tuple(ring), subs, principal_group_positions=pcg)
        pcg_locants = [oriented.index(a) + 1 for a in pcg]
        assert min(pcg_locants) == 1
