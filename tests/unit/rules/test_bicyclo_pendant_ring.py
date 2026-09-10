"""
Regression test for the bicyclo pendant-ring scope bug (a phase, beta-lactam layer 1).

`is_bicyclo_system`'s zero-bridge ring-size guard (bicyclo.py, the
`if lengths and min(lengths) == 0:` block) used to read `ri.AtomRings`
(whole-molecule SSSR) instead of the connected-component ring scope already
computed a few lines earlier (`ring_atoms_set`, via
`_get_largest_connected_ring_component`). That meant ANY pendant substituent
ring of size >= 5 elsewhere in the molecule (a phenyl, e.g. the acylamino side
chain of penicillin G) could flip the guard to False and block bicyclo
detection of a genuinely fused bicyclic core (bicyclo[x.y.0]).

Note: the guard only fires for a ZERO-length bridge (a genuinely FUSED
bicyclic, e.g. the penam/cephem beta-lactam cores). A BRIDGED bicyclic like
norbornane (bridge lengths [2,2,1], never 0) never reaches this code path
at all, pendant ring or not -- so it is not a suitable probe for this
specific bug (a norbornane with a phenyl attached directly to a ring atom
instead exercises a separate, out-of-scope defect in
`perception.rings.find_ring_bridgeheads`, which also defaults to an
unscoped whole-molecule ring-atom set; that is not touched by this fix).

This test asserts:
1. A fused (zero-bridge) bicyclic core with a pendant >=5-ring is still
   detected as bicyclo (the verdict must match the same core WITHOUT the
   pendant ring).
2. Existing von-Baeyer/bicyclo names are byte-identical (no pendant ring
   present, so the scope fix must not change their verdict or output).
3. The bicyclo composer is now actually REACHED (invoked) for a
   phenyl-bearing fused bicyclic, even though a full correct name still
   needs a separate, not-yet-fixed layer (the acylamino substituent
   spelling) -- so a full name is intentionally NOT asserted here.
"""

from rdkit import Chem

from orthonym.rules.bicyclo import is_bicyclo_system
from orthonym import name_compound


# ============================================================================
# 1. Pendant-ring bicyclo detection
# ============================================================================

class TestPendantRingDoesNotBlockBicyclo:
    def test_penam_core_with_pendant_phenyl_is_bicyclo(self):
        """
        Penicillin-G-like penam core (fused beta-lactam + thiazolidine,
        zero-length bridge) with a pendant phenyl on the acylamino side
        chain. Before the fix, ri.AtomRings over the WHOLE molecule
        included the phenyl's 6-ring, inflating sorted ring sizes to
        [4, 5, 6] so ring_sizes[-2] (5) >= 5 tripped the "route to fused"
        branch and is_bicyclo_system returned False -- even though the
        actual bicyclic core (rings of size 4 and 5 only) has no known
        fused retained parent and should stay bicyclo[x.y.0].
        """
        core_only = Chem.MolFromSmiles("OC(=O)C1N2C(=O)CC2SC1(C)C")
        penicillin_g = Chem.MolFromSmiles(
            "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O"
        )

        assert is_bicyclo_system(core_only) is True
        # The pendant phenyl (reached via the acylamino side chain, not
        # directly bonded to the ring) must not change the verdict.
        assert is_bicyclo_system(penicillin_g) is True

    def test_cephem_core_with_pendant_ring_is_bicyclo(self):
        """
        Same scope bug, second fused (zero-bridge) core: a cephem
        (beta-lactam + dihydrothiazine) with a pendant phenyl substituent
        reachable via whole-molecule SSSR.
        """
        core_only = Chem.MolFromSmiles("OC(=O)C1=CCSC2CC(=O)N12")
        with_phenyl_sidechain = Chem.MolFromSmiles(
            "OC(=O)C1=C(COC(C)=O)CSC2CC(=O)N12"
        )  # cefalotin-like acetoxymethyl side chain -- no extra >=5 ring,
        # so also probe a phenylacetamido variant of the cephem core:
        with_phenylacetamido = Chem.MolFromSmiles(
            "OC(=O)C1=CCSC2C(NC(=O)Cc3ccccc3)C(=O)N12"
        )

        assert is_bicyclo_system(core_only) is True
        assert is_bicyclo_system(with_phenyl_sidechain) is True
        assert is_bicyclo_system(with_phenylacetamido) is True

    def test_composer_reached_for_phenyl_bearing_fused_bicyclic(self):
        """
        Integration-level, via monkeypatch trace on the composer entry point
        (the same technique the coordinator's a trace used): confirm
        `_assemble_complete_bicyclo_name` is actually CALLED (the gate is
        reached) when naming penicillin G, whose pendant phenyl used to
        block `is_bicyclo_system` outright (0 calls before the fix). We do
        not assert a full/correct final name -- the substituent (acylamino)
        spelling is a separate, not-yet-fixed layer, and the top-level
         gate is expected to suppress a wrong candidate for it,
        which is correct 0-wrong behaviour, not a test failure.
        """
        import orthonym.assembly.composer as composer_module

        calls = []
        original = composer_module._assemble_complete_bicyclo_name

        def spy(mol, features):
            calls.append(1)
            return original(mol, features)

        composer_module._assemble_complete_bicyclo_name = spy
        try:
            penicillin_g = "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O"
            name_compound(penicillin_g)  # return value intentionally unchecked
        finally:
            composer_module._assemble_complete_bicyclo_name = original

        assert len(calls) > 0, (
            "Bicyclo composer was never invoked for a phenyl-bearing fused "
            "bicyclic -- is_bicyclo_system is still blocking it."
        )


# ============================================================================
# 2. Byte-identical von-Baeyer / bicyclo regression sweep
# ============================================================================

class TestVonBaeyerByteIdenticalRegression:
    """
    None of these molecules has a pendant ring outside the bicyclic/fused
    core, so the connected-component ring scope already equals the
    whole-molecule ring scope for them. The scope fix must therefore leave
    every one of these names byte-identical.
    """

    CASES = [
        ("C1CC2CCC1C2", "norbornane"),
        (
            "OC(=O)C1N2C(=O)CC2SC1(C)C",
            "3,3-dimethyl-7-oxo-4-thia-1-azabicyclo[3.2.0]heptane-2-carboxylic acid",
        ),
        (
            "OC(=O)C1=CCSC2CC(=O)N12",
            "8-oxo-5-thia-1-azabicyclo[4.2.0]oct-2-ene-2-carboxylic acid",
        ),
        ("C1CCC2CCCCC2C1", "decahydronaphthalene"),
        ("C1=CC2CC1CC2", "bicyclo[2.2.1]hept-2-ene"),
        ("C1CC2CCC3(CCCC3)CC12", "spiro[bicyclo[4.2.0]octane-3,1'-cyclopentane]"),
        ("C1CC2(CCC1)CC2", "spiro[2.5]octane"),
        ("C1CC2CCC1(C)C2", "1-methylbicyclo[2.2.1]heptane"),
        ("OC1CC2CCC1CC2", "bicyclo[2.2.2]octan-2-ol"),
        ("C1CC2CC1C(=O)C2", "bicyclo[2.2.1]heptan-2-one"),
        # M4#1: main bridge maximized (2) before symmetric division,
        # so tricyclo[5.2.2.0^2,6] (main bridge 2) is preferred over the older
        # tricyclo[4.3.0.2^2,5] (main bridge 0). Both OPSIN-round-trip to the same
        # C11 cage; the new form is the more -conformant decomposition.
        ("C1CC2CCC1C1CCCC21", "tricyclo[5.2.2.0^2,6]undecane"),
    ]

    def test_byte_identical_names(self):
        for smiles, expected in self.CASES:
            name = name_compound(smiles)
            assert name == expected, (
                f"Regression: {smiles!r} expected {expected!r} but got {name!r}"
            )
