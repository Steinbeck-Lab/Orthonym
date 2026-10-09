""" breadth — peptides with PROLINE (or any cyclic imino acid) as the AMINE
component of a peptide bond (X-Pro, X-Pro-Y).

Root cause: the peptide-bond SMARTS `[CX3](=O)[NX3;H1][CX4]` required 1 H on the
amide N, but when proline is the amine acceptor the bond terminates at proline's
RING nitrogen — a TERTIARY amide (H0) — so every X-Pro bond went undetected and the
whole peptide abstained. Fix: admit H0 (`[NX3;H0,H1]`). Because proline's ring N has
TWO ring-carbon `[CX4]` neighbours, the SMARTS then matches the SAME C(=O)-N bond
twice; the duplicate bond index crashed RDKit's FragmentOnBonds (segfault), so the
pairs are deduplicated before cleaving. The existing `_is_alpha_carboxyl_bond` guard
+ standard-amino-acid residue identification keep non-peptide tertiary amides failing
closed. 0-wrong holds (gates every emission).
"""
import pytest

from orthonym import Orthonym
from rdkit import Chem
from rdkit.Chem import inchi

pytestmark = pytest.mark.unit


def _pin():
    return Orthonym(style="pin")


def _skeleton(smi):
    m = Chem.MolFromSmiles(smi)
    return inchi.MolToInchiKey(m).split("-")[0] if m else None


@pytest.mark.parametrize("smi,expected", [
    # tripeptide with proline as the INTERNAL (amine-acceptor) residue — RT-exact
    # j7 (TRIAGE g3 C05): the substitutive PIN method (1)), OPSIN exact
    ("N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)NCC(=O)O",
     "{(2S)-1-[(2S)-2-aminopropanoyl]pyrrolidine-2-carboxamido}acetic acid"),
    ("N[C@@H](CCC(=O)O)C(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
     "glutamylprolylphenylalanine"),
])
def test_proline_acceptor_tripeptide_names(smi, expected):
    assert _pin().name(smi) == expected


def test_x_pro_does_not_crash():
    """The proline ring-N matches the peptide-bond SMARTS twice; without the
    dedup, FragmentOnBonds gets a duplicate bond and RDKit SEGFAULTS. Every X-Pro
    input must name-or-abstain cleanly, never crash."""
    p = _pin()
    for smi in ["N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)O",          # Ala-Pro (C-term Pro)
                "NCC(=O)N1CCC[C@H]1C(=O)O",                   # Gly-Pro
                "N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)NCC(=O)O"]:   # Ala-Pro-Gly
        out = p.name(smi)               # must not raise / segfault
        assert isinstance(out, str)


def test_proline_acceptor_0_wrong_constitution():
    """Every X-Pro peptide that EMITS a name must denote the right CONSTITUTION
    (InChIKey skeleton), even where a histidine imidazole tautomer makes it not
    full-InChIKey-exact. Abstentions (str sentinel) are allowed; a wrong
    constitution is not."""
    import sys
    sys.path.insert(0, "scripts")
    from diagnose import diagnose
    smis = [
        "N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)NCC(=O)O",                       # Ala-Pro-Gly
        "N[C@@H](CCC(=O)O)C(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc1ccccc1)C(=O)O", # Glu-Pro-Phe
        "N[C@@H](CCC(=O)O)C(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O", # Glu-Pro-His
    ]
    rows = diagnose(smis, style="pin", use_opsin=True)
    for r in rows:
        op = r.get("opsin_smiles")
        if op:  # emitted + parsed → constitution must match the input skeleton
            assert _skeleton(r["smiles"]) == _skeleton(op), (
                r["smiles"], r.get("name"), op)


# The two dipeptide rows below take the substitutive PIN, as the tripeptide row already did
# (j7, 4f2284331, 2026-09-27: "a tripeptide takes the substitutive PIN through a nested amido
# prefix"; controller ruling there: peptide names are not PINs,:50943).
# "Substituents of the types -NH-CO-R and -NH-SO2-R" (the Blue Book): the R-CO-NH-
# group is named "(1) substitutively, by using a prefix formed by changing the final letter
# 'e' in the complete name of the amide to 'o'... 'carboxamide' into 'carboxamido'" and
# "Method (1) generates preferred IUPAC names." The old retained peptide names read back to
# the same molecule with OPSIN 2.9.0 (full an InChIKey and
# an InChIKey, identical for old and new names), so the change is the
# PIN-conformant spelling, not a different structure. The parametrize ids are the OLD
# strings on purpose: the node id embeds the expected value and is kept stable.
@pytest.mark.parametrize("smi,expected", [
    # non-proline peptides + proline-as-acyl (N-terminal) take the substitutive PIN
    pytest.param("N[C@@H](Cc1ccccc1)C(=O)N[C@@H](C)C(=O)O",
                 "(2S)-2-[(2S)-2-amino-3-phenylpropanamido]propanoic acid",
                 id="N[C@@H](Cc1ccccc1)C(=O)N[C@@H](C)C(=O)O-phenylalanylalanine"),
    # j7 (TRIAGE g3 C05): the substitutive PIN method (1)), OPSIN exact
    ("N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)O",
     "(2S)-2-{(2S)-2-[(2S)-2-aminopropanamido]propanamido}propanoic acid"),
    pytest.param("OC(=O)[C@@H](C)NC(=O)[C@@H]1CCCN1",
                 "(2R)-2-[(2S)-pyrrolidine-2-carboxamido]propanoic acid",
                 id="OC(=O)[C@@H](C)NC(=O)[C@@H]1CCCN1-prolyl-D-alanine"),
])
def test_non_proline_acceptor_unchanged(smi, expected):
    assert _pin().name(smi) == expected
