"""Regression locks for the "hardcoded SMILES does not match its name" bug class.

These are the OPSIN-free distillate of the full sweep in
``smiles_name_inchikey_sweep.py`` (run that for the whole surface; it needs a JVM).
Everything here uses RDKit only, so it is safe in the fast unit tier.

METHOD, and why it is not circular
----------------------------------
A test of the form ``assert lookup(smiles) == name`` cannot catch this bug class,
because it merely restates the table. Every reference structure below is therefore
written out INDEPENDENTLY in this file, from the compound's structural definition,
and compared by InChIKey. If a data table is re-keyed to the wrong isomer again,
these fail.

Two instances of the bug class are on record in this repo, both wrong-isomer:
``bicyclo_systems.py`` mapped isoquinuclidine's SMILES to "quinuclidine", and
earlier stored a non-cubane (CH)8 cage under "cubane".
"""

import pytest
from rdkit import Chem

from orthonym.data.bicyclo_systems import (
    BICYCLO_RETAINED_NAMES,
    get_retained_bicyclo_name,
)
from orthonym.data.ion_retained_names import INORGANIC_ANIONS, RETAINED_CATIONS
from orthonym.data.natural_products import NATURAL_PRODUCT_DERIVATIVES
from orthonym.data.retained_names import RETAINED_NAMES


def ik(smiles):
    """InChIKey of a SMILES; None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


def skeleton(smiles):
    """Constitution-only (first) block of the InChIKey."""
    key = ik(smiles)
    return key.split("-")[0] if key else None


# --- Reference structures, each built here from the compound's definition -----
# NOTE these are deliberately NOT imported from orthonym.data.
REF = {
    # quinuclidine = 1-azabicyclo[2.2.2]octane: N at a BRIDGEHEAD.
    "quinuclidine": "C1CN2CCC1CC2",
    # isoquinuclidine = 2-azabicyclo[2.2.2]octane: N one atom OFF the bridgehead.
    "isoquinuclidine": "C1CC2CCC1NC2",
    # decahydroisoquinoline: N at ring position 2, NOT at a fusion atom.
    "decahydroisoquinoline": "C1CCC2CNCCC2C1",
    # quinolizidine: N AT the ring-fusion atom.
    "quinolizidine": "C1CCN2CCCCC2C1",
    # piperazine-2,5-dione (glycine anhydride): the two C=O are PARA-related,
    # one adjacent to each N.
    "piperazine-2,5-dione": "O=C1CNC(=O)CN1",
    # piperazine-2,6-dione: both C=O adjacent to the SAME N.
    "piperazine-2,6-dione": "O=C1CNCC(=O)N1",
    # pinacolone = 3,3-dimethylbutan-2-one, C6H12O.
    "pinacolone": "CC(=O)C(C)(C)C",
    # 4,4-dimethylpentan-2-one, C7H14O - one CH2 longer than pinacolone.
    "4,4-dimethylpentan-2-one": "CC(=O)CC(C)(C)C",
    # true cubane, SSSR 6.
    "cubane": "C12C3C4C1C1C2C3C41",
    # the non-cubane (CH)8 cage that was once stored under "cubane", SSSR 4.
    "not-cubane-cage": "C12C3C4C1C5C3C4C25",
}


@pytest.mark.unit
class TestWrongIsomerRegressionLocks:
    """Each row was a real wrong-structure mapping; none may come back."""

    def test_reference_isomer_pairs_are_genuinely_distinct(self):
        """Guard the guard: the pairs below must be different constitutions.

        If RDKit/InChI ever normalised these together, every assertion in this
        class would pass vacuously.
        """
        pairs = [
            ("quinuclidine", "isoquinuclidine"),
            ("decahydroisoquinoline", "quinolizidine"),
            ("piperazine-2,5-dione", "piperazine-2,6-dione"),
            ("pinacolone", "4,4-dimethylpentan-2-one"),
            ("cubane", "not-cubane-cage"),
        ]
        for a, b in pairs:
            assert skeleton(REF[a]) is not None, f"{a} reference unparseable"
            assert skeleton(REF[a]) != skeleton(REF[b]), (
                f"reference pair {a}/{b} collapsed to one constitution; "
                "these locks would be vacuous"
            )

    def test_isoquinuclidine_is_not_named_quinuclidine(self):
        """PA1 R6: bicyclo_systems.py mapped isoquinuclidine to 'quinuclidine'."""
        assert get_retained_bicyclo_name(REF["isoquinuclidine"]) is None
        for key, name in BICYCLO_RETAINED_NAMES.items():
            if "quinuclidin" in name.lower():
                assert skeleton(key) == skeleton(REF["quinuclidine"]), (
                    f"{name!r} is keyed on {key!r}, which is not quinuclidine"
                )

    def test_quinuclidine_is_not_a_pin_headline(self):
        """BB:9881 / BB:9893 (P-23.7): general nomenclature only."""
        from orthonym.data import ALL_RETAINED_NAMES, GENERAL_RETAINED_NAMES

        canon = Chem.CanonSmiles(REF["quinuclidine"])
        assert ALL_RETAINED_NAMES.get(canon) != "quinuclidine", (
            "quinuclidine is retained for GENERAL nomenclature only (BB:9881); "
            "the PIN is 1-azabicyclo[2.2.2]octane (BB:9893)"
        )
        assert GENERAL_RETAINED_NAMES.get(canon) == "quinuclidine", (
            "the general-only name must stay reachable via --trivial"
        )

    @pytest.mark.parametrize("wrong_key,wrong_name,table", [
        ("quinolizidine", "decahydroisoquinoline", "RETAINED_NAMES"),
        ("piperazine-2,6-dione", "piperazine-2,5-dione", "RETAINED_NAMES"),
        ("4,4-dimethylpentan-2-one", "pinacolone", "RETAINED_NAMES"),
    ])
    def test_retained_names_has_no_wrong_isomer_rows(self, wrong_key, wrong_name,
                                                     table):
        """The structure of `wrong_key` must not carry the name `wrong_name`."""
        canon = Chem.CanonSmiles(REF[wrong_key])
        assert RETAINED_NAMES.get(canon) != wrong_name, (
            f"{table}[{canon!r}] == {wrong_name!r}, but that SMILES is "
            f"{wrong_key}, a different compound"
        )

    def test_isopropylium_is_not_the_tert_butyl_cation(self):
        """C[C+](C)C is C4H9+ (tert-butyl); isopropylium is C3H7+."""
        assert RETAINED_CATIONS.get("C[C+](C)C") != "isopropylium"
        for key, name in RETAINED_CATIONS.items():
            if name == "isopropylium":
                assert skeleton(key) == skeleton("C[CH+]C"), (
                    f"'isopropylium' keyed on {key!r}, which is not the "
                    "propan-2-yl cation"
                )

    def test_sulfate_is_not_keyed_on_bisulfite(self):
        """O=[SH](=O)[O-] is HO3S- (hydrogensulfite), not sulfate O4S2-."""
        assert INORGANIC_ANIONS.get("O=[SH](=O)[O-]") != "sulfate"
        sulfate_keys = [k for k, v in INORGANIC_ANIONS.items() if v == "sulfate"]
        assert sulfate_keys, "the correct sulfate key must still be present"
        for key in sulfate_keys:
            assert skeleton(key) == skeleton("[O-]S(=O)(=O)[O-]"), (
                f"'sulfate' keyed on {key!r}, which is not the SO4 dianion"
            )

    @pytest.mark.parametrize("name,ref_smiles", [
        # beta-pinene is C10H16 with an EXOCYCLIC methylidene. The wrong key that
        # was removed here was an endocyclic-alkene C9H14, and there is no
        # correct key in its place -- so this asserts the INVARIANT (if the table
        # claims this name, the structure must match), not coverage. Adding a
        # correct beta-pinene row is a separate decision about whether that
        # terpene trivial name should be emitted at all.
        ("beta-pinene", "C=C1CCC2CC1C2(C)C"),
        # gamma-terpineol is a monocyclic C10H18O, not an acyclic chain. Here a
        # correct key DOES exist, so presence is asserted too (below).
        ("gamma-terpineol", "CC(C)=C1CCC(C)(O)CC1"),
        ("alpha-pinene", "CC1=CCC2CC1C2(C)C"),
    ])
    def test_natural_product_keys_match_their_names(self, name, ref_smiles):
        """If the table attaches this name to a SMILES, it must be that compound."""
        want = skeleton(ref_smiles)
        assert want is not None, f"reference for {name} is unparseable"
        for key, value in NATURAL_PRODUCT_DERIVATIVES.items():
            if value == name:
                assert skeleton(key) == want, (
                    f"NATURAL_PRODUCT_DERIVATIVES[{key!r}] == {name!r}, but that "
                    f"SMILES is a different compound"
                )

    def test_correct_gamma_terpineol_and_alpha_pinene_keys_survive(self):
        """The two rows that HAD a correct key must not have been lost."""
        for name, ref in (("gamma-terpineol", "CC(C)=C1CCC(C)(O)CC1"),
                          ("alpha-pinene", "CC1=CCC2CC1C2(C)C")):
            keys = [k for k, v in NATURAL_PRODUCT_DERIVATIVES.items()
                    if v == name and skeleton(k) == skeleton(ref)]
            assert keys, f"the correct {name} key was lost"


@pytest.mark.unit
class TestBicycloTableInvariants:
    """General invariants over the table that carried two of these bugs."""

    def test_all_keys_are_rdkit_canonical(self):
        """Lookup is by canonical SMILES, so a non-canonical key is dead."""
        dead = [k for k in BICYCLO_RETAINED_NAMES
                if Chem.MolToSmiles(Chem.MolFromSmiles(k)) != k]
        assert dead == [], f"non-canonical (unreachable) keys: {dead}"

    def test_no_two_names_share_one_structure(self):
        seen = {}
        for key, name in BICYCLO_RETAINED_NAMES.items():
            k = ik(key)
            assert k is not None, f"unparseable key {key!r}"
            if k in seen and seen[k] != name:
                pytest.fail(f"{seen[k]!r} and {name!r} are the same structure")
            seen[k] = name

    def test_cubane_entry_is_real_cubane(self):
        """BB:9881/9889 (P-23.7): cubane is retained AND a PIN."""
        keys = [k for k, v in BICYCLO_RETAINED_NAMES.items() if v == "cubane"]
        assert len(keys) == 1, f"expected exactly one cubane key, got {keys}"
        assert skeleton(keys[0]) == skeleton(REF["cubane"])
        assert skeleton(keys[0]) != skeleton(REF["not-cubane-cage"])
        assert len(Chem.GetSymmSSSR(Chem.MolFromSmiles(keys[0]))) == 6, (
            "cubane has 6 SSSR rings; the stale cage isomer has 4"
        )


@pytest.mark.unit
def test_sweep_discovery_still_sees_the_data_tables():
    """The sweep must not silently go blind if a module is renamed or moved.

    Without this, ``discover_pairs`` could quietly return nothing and the whole
    sweep would "pass" while checking zero mappings.
    """
    from tests.unit.data.smiles_name_inchikey_sweep import discover_pairs

    pairs, stats = discover_pairs()
    assert stats["swept_pairs"] > 1500, (
        f"discovery collapsed to {stats['swept_pairs']} pairs"
    )
    origins = {o for origins in pairs.values() for o in origins}
    for required in (
        "orthonym.data.retained_names.RETAINED_NAMES",
        "orthonym.data.bicyclo_systems.BICYCLO_RETAINED_NAMES",
        "orthonym.data.ion_retained_names.INORGANIC_ANIONS",
        "orthonym.data.natural_products.NATURAL_PRODUCT_DERIVATIVES",
        "orthonym.data.polycyclic_data._SMILES_TO_NAME",
    ):
        assert required in origins, f"sweep no longer discovers {required}"
