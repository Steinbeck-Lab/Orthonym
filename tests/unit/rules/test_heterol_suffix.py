""" (Heterols) / (Hydroperoxides): -OH/-OOH on a ring heteroatom
takes the -ol / -peroxol suffix, not a hydroxy/hydroperoxy prefix.

BB verbatim: piperidin-1-ol (PIN), pyrrolidine-1,2-diol (PIN), pyrrolidine-1-peroxol
(PIN), 3,4,5,6-tetrahydro-1λ4,2-thiazin-1-ol. On a ring heteroatom the hydroxylamine /
sulfinimidic-acid functional parent cannot exist (its valences are ring bonds), so the
OH is reclassified into the ordinary alcohol/hydroperoxide suffix machinery and the
 demotion cascade handles it (demoting to hydroxy when a senior group is present).

Each naming case uses a fresh engine and does NOT interleave a module-level opsin
import (see test_composite_oxy_enclosure for the JVM-init-order hazard). Round-trips
are checked separately.
"""
import pytest
from rdkit import Chem

_FLAGS = dict(general_fallback=True, general_fallback_unverified=True,
              allow_aromatic_general=True)


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


@pytest.mark.parametrize("smiles,expected", [
    ("ON1CCCCC1", "piperidin-1-ol"),                          #
    ("OC1CCCN1O", "pyrrolidine-1,2-diol"),                    #
    ("OON1CCCC1", "pyrrolidine-1-peroxol"),                   #
    ("OS1=NCCCC1", "3,4,5,6-tetrahydro-1λ4,2-thiazin-1-ol"),  # (λ4-S)
])
def test_ring_heteroatom_oh_takes_suffix(smiles, expected):
    from orthonym import Orthonym
    got = Orthonym(**_FLAGS).name_tiered(smiles)["name"]
    assert got == expected, f"{smiles}: got {got!r}"


@pytest.mark.parametrize("smiles,expected", [
    ("ON1CCCCC1", "piperidin-1-ol"),
    ("OC1CCCN1O", "pyrrolidine-1,2-diol"),
    ("OON1CCCC1", "pyrrolidine-1-peroxol"),
    ("OS1=NCCCC1", "3,4,5,6-tetrahydro-1λ4,2-thiazin-1-ol"),
])
def test_heterol_pins_round_trip(smiles, expected):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    rt = opsin_parse(expected)
    got = Chem.MolFromSmiles(rt) if rt else None
    assert got is not None and Chem.MolToInchiKey(got) == _inchikey(smiles), \
        f"RT failed for {expected!r}"


@pytest.mark.parametrize("smiles,expected", [
    # senior group present -> OH stays a hydroxy prefix demotion)
    ("N#CC1CCCN(O)C1", "1-hydroxypiperidine-3-carbonitrile"),  # nitrile senior
    ("ONc1ccc(O)cc1", "4-(hydroxyamino)phenol"),               # phenol senior, N not in ring
    # acyclic hydroxylamine / sulfinimidic acid -> untouched (not ring heteroatoms)
    ("CS(=N)O", "methanesulfinimidic acid"),
    ("CNO", "N-methylhydroxylamine"),
    ("CCNO", "N-ethylhydroxylamine"),
    # ordinary ring-carbon alcohols / peroxols -> unchanged
    ("OC1CCCCC1", "cyclohexanol"),
    ("OOC1CCCCC1", "cyclohexane-1-peroxol"),
])
def test_heterol_canaries_unchanged(smiles, expected):
    from orthonym import Orthonym
    got = Orthonym(**_FLAGS).name_tiered(smiles)["name"]
    assert got == expected, f"{smiles}: got {got!r}"
