"""P7 (Wave-8) sub-plan 7b — carbohydrate semisystematic tail (P-102).

7b.2 — open-chain uronic acid PINs (P-102.5.6.6.4.1).

Blue Book P-102.5.6.6.4.1 (BB:53779-53783): "Names of individual uronic acids
are formed by changing the ending 'ose' in the retained or systematic name of
the corresponding aldose to 'uronic acid'. The numbering of the aldose is kept
intact; the locant '1' is still assigned to the (potential) aldehydic group."
Example given: D-glucuronic acid. The RING form (alpha-D-glucopyranuronic acid)
already names correctly; the OPEN-CHAIN aldehydo form emitted the systematic
...-6-oxohexanoic acid. All target names are OPSIN-parseable (normal gate).

Structures are OPSIN name->structure authoritative (reproduce-first).
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym

RAW = Orthonym(_disable_opsin_validity_gate=True)
GATED = Orthonym()


@pytest.mark.parametrize("smiles,expected", [
    ("O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)C(=O)O", "D-glucuronic acid"),
    ("O=C[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)C(=O)O", "D-galacturonic acid"),
    ("O=C[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)C(=O)O", "D-mannuronic acid"),
    ("O=C[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)C(=O)O", "L-iduronic acid"),
    ("O=C[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)C(=O)O", "L-guluronic acid"),
])
def test_open_chain_uronic_acid_pin(smiles, expected):
    can = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert GATED.name(can) == expected
    assert RAW.name(can) == expected


def test_uronic_ring_form_unchanged():
    # Regression: the pyranuronic ring form must keep its existing PIN.
    can = Chem.MolToSmiles(Chem.MolFromSmiles(
        "O=C(O)[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O"))
    assert GATED.name(can) == "alpha-D-glucopyranuronic acid"


# 7b.3 — aldarate mono-ester (P-102.5.6.6.5.3). The C1 (or C6) carboxyl of an
# aldaric acid esterified: `<n>-<alkyl> hydrogen <config>-<stem>arate`. The 1-vs-6
# locant identifies distinct isomers (BB gives both 1-methyl and 6-methyl hydrogen
# L-altrarate), disambiguated by a HARD OPSIN round-trip (fail-closed w/o Java).
def test_aldarate_mono_methyl_ester_pin():
    can = Chem.MolToSmiles(Chem.MolFromSmiles(
        "COC(=O)[C@H](O)[C@@H](O)[C@@H](O)[C@@H](O)C(=O)O"))
    assert GATED.name(can) == "1-methyl hydrogen L-altrarate"


def test_meso_aldarate_ester_fails_closed():
    # meso galactarate mono-methyl ester: galactaric acid is not a cataloged
    # (D/L) aldaric -> the aldarate-ester path must fail closed (valid systematic),
    # never a wrong retained name.
    can = Chem.MolToSmiles(Chem.MolFromSmiles(
        "COC(=O)[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)C(=O)O"))
    out = GATED.name(can)
    assert "arate" not in out  # no retained aldarate name; systematic instead


@pytest.mark.parametrize("smi,expected", [
    # 7b.4 (P-102.5.6.3.2): C-substitution replacing a non-terminal OH (deoxy-C).
    ("OC[C@H]1O[C@H](O)[C@H](c2ccccc2)[C@@H](O)[C@@H]1O",
     "2-deoxy-2-phenyl-alpha-D-glucopyranose"),
    # 7b.4 (P-102.5.6.3.1): C-substitution ADDED at a non-terminal C (n-C-R).
    ("C[C@]1(O)[C@@H](O)O[C@H](CO)[C@@H](O)[C@@H]1O",
     "2-C-methyl-alpha-D-glucopyranose"),
    ("C[C@@]1(O)[C@@H](CO)O[C@@H](O)[C@@H]1O",
     "3-C-methyl-beta-D-ribofuranose"),
])
def test_c_substituted_sugar_pin(smi, expected):
    can = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    assert GATED.name(can) == expected


def test_glycosyloxy_n_o_yl_pin():
    # 7b.5 (P-102.6.2): sugar attached via a NON-anomeric O to acetic acid ->
    # (beta-D-glucopyranos-2-O-yl)acetic acid.
    can = Chem.MolToSmiles(Chem.MolFromSmiles(
        "O=C(O)CO[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]1O"))
    assert GATED.name(can) == "(beta-D-glucopyranos-2-O-yl)acetic acid"


def test_aldonate_ester_unchanged():
    # Regression: single-carboxyl aldonate ester keeps its existing PIN.
    can = Chem.MolToSmiles(Chem.MolFromSmiles(
        "CC(C)OC(=O)[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)CO"))
    assert GATED.name(can) == "propan-2-yl L-gulonate"
