"""The ester of a carboxylic acid on a ring nitrogen ('tert-butyl piperidine-1-carboxylate').

"Carboxy groups attached to cyclic parent hydrides or heteroacyclic parent hydrides are always named by using the suffix
'carboxylic acid'" "Carboxylic acids", the Blue Book: 'pyrrolidine-1-carboxylic acid (PIN)'), and
"All preferred IUPAC names for esters are named by functional class nomenclature". The carbamate namer
read the two ring bonds of the nitrogen as two N-substituents ('ethyl dipentylcarbamate' for ethyl
piperidine-1-carboxylate, the ring counted twice), and the decomposition engine dropped an amine it could not cite
('tert-butyl carbamate' for a Boc-piperidinol); only the round trip withdrew them. Now the carbamate builder
(``composer._name_carbamate``) declines a ring nitrogen and ``assembly.ring_n_carboxylate`` writes the ester of the ring-nitrogen acid.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import name_is_rt_exact


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("CCOC(=O)N1CCCCC1", "ethyl piperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCCCC1", "tert-butyl piperidine-1-carboxylate"),
    ("COC(=O)N1CCCC1", "methyl pyrrolidine-1-carboxylate"),
    ("CCOC(=O)N1CCOCC1", "ethyl morpholine-4-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCN(CC1)c1ccccc1", "tert-butyl 4-phenylpiperazine-1-carboxylate"),
    ("O=C(OCc1ccccc1)N1CCCC1", "benzyl pyrrolidine-1-carboxylate"),
    # beside a junior group the ester is still the parent: esters, class 9, above amides, nitriles,
    # ketones, hydroxy compounds and amines; '2-hydroxypropyl (2-aminoethyl)carbamate (PIN)',
    ("CC(C)(C)OC(=O)N1CCC(O)CC1", "tert-butyl 4-hydroxypiperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(CO)CC1", "tert-butyl 4-(hydroxymethyl)piperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(N)CC1", "tert-butyl 4-aminopiperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(=O)CC1", "tert-butyl 4-oxopiperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCN(c2ccc(C#N)cc2)CC1", "tert-butyl 4-(4-cyanophenyl)piperazine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC[C@H]1C(N)=O", "tert-butyl (2S)-2-carbamoylpyrrolidine-1-carboxylate"),
    # the acyclic carbamates keep their names
    ("CCOC(=O)NC", "ethyl methylcarbamate"),
    ("CC(C)(C)OC(=O)NCCO", "tert-butyl (2-hydroxyethyl)carbamate"),
])
def test_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row
    assert name_is_rt_exact(expected, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "CC(C)(C)OC(=O)N1CCC(O)CC1",      # was 'tert-butyl carbamate' before the gate; the amine is never dropped
    "CC(C)(C)OC(=O)N1CCC(CC1)C(=O)O",  # the carboxylic acid is senior: not this producer's name
    "CC(C)(C)OC(=O)N1CCC(C(=O)OC)CC1",  # two esters: the PIN cites both ('1-tert-butyl 4-methyl...-1,4-dicarboxylate')
])
def test_no_wrong_pin(smiles):
    row = tier_row(smiles, "pin")
    assert row["tier"] in ("abstain", "pin_verified"), row
    if row["tier"] == "pin_verified":
        assert name_is_rt_exact(row["name"], smiles), row
    assert "dipentyl" not in (row["name"] or "") and row["name"] != "tert-butyl carbamate"
    assert not (row["name"] or "").endswith("4-(methoxycarbonyl)piperidine-1-carboxylate")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["NC(=N)N1CCCCC1", "CNC(=N)N1CCCCC1"])
def test_the_ring_nitrogen_guard_is_the_carbamate_builders_only(smiles):
    # the guanidine builder keeps its own behaviour: no prefix-only name ('1-carbamimidoylpiperidine') becomes a PIN
    assert tier_row(smiles, "pin")["tier"] != "pin_verified"
