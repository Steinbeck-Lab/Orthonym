"""v33 charged-lever completion tests (Slice B).

Task B1: audit RETAINED_AMINO_ACID_ZWITTERIONS for stereo/constitution
mislabels (every entry's name must OPSIN-parse to the SAME full InChIKey
as its keyed SMILES).

Task B2: a net-zero amino-acid zwitterion whose retained-name path declines
(non-standard AA) must fall through to a systematic neutral name that still
OPSIN full-InChIKey round-trips, rather than aborting to an abstention.
"""

import pytest
from rdkit import Chem


def test_retained_aa_zwitterion_table_is_stereo_correct():
    from orthonym.rules.salts import RETAINED_AMINO_ACID_ZWITTERIONS
    from orthonym.validation.opsin_roundtrip import opsin_parse
    bad = []
    for smi, name in RETAINED_AMINO_ACID_ZWITTERIONS.items():
        g = opsin_parse(name)
        ok = g and Chem.MolToInchiKey(Chem.MolFromSmiles(g)) == Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
        if not ok:
            bad.append((smi, name, g))
    assert not bad, f"mislabeled table entries: {bad}"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", [
    "[NH3+][C@@H](CC[SeH])C(=O)[O-]",   # selenohomocysteine zwitterion
    "CC(C)C[C@@H]([NH3+])C(=O)[O-]",     # leucine zwitterion (retained path OR systematic)
])
def test_netzero_aa_zwitterion_names_and_roundtrips(smi):
    from orthonym.namer import Orthonym
    from orthonym.validation.opsin_roundtrip import opsin_parse
    n = Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True).name(smi)
    assert n and n != "unknown organic compound"
    g = opsin_parse(n)
    assert g and Chem.MolToInchiKey(Chem.MolFromSmiles(g)) == Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
