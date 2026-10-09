""" PF: a retained name must not assert double-bond geometry its key leaves open.

`crotonic acid`, `sorbic acid` and `crotonaldehyde` were keyed by geometry-FREE
SMILES (`CC=CC(=O)O`, `CC=CC=CC(=O)O`, `CC=CC=O`) while each name denotes one
specific isomer -- crotonic acid IS (E)-but-2-enoic acid (the Z isomer has its own
name, isocrotonic acid) and sorbic acid IS (2E,4E). So the emitted name named a
stereoisomer the input never claimed.

 cannot catch this: it compares the InChIKey **skeleton** block and is
stereo-insensitive by design, which is exactly why these shipped with every gate
green. The assertions below are structural, not string comparisons, so they keep
their teeth if the spelling of the systematic name ever changes.

Evidence for the change (`change-asserted-value`, three artifacts):

  1. PRIMARY SOURCE -- all three names occur **0** times in the Blue Book under a
     markup/OCR-tolerant search whose known-positive control passed in the same run
     (acetic acid 178, benzoic acid 185, but-2-enoic acid 4), while the replacement
     is marked PIN verbatim: "but-2-enoic acid (PIN)".
  2. INDEPENDENT CHECK -- OPSIN (a separate implementation, not code under test)
     parses the old names to a DIFFERENT InChIKey than the input, i.e. the old value
     violated the necessary condition "the name denotes the input molecule". That is
     what `test_old_names_denote_a_different_molecule_than_the_key` asserts.
  3. MUTATION -- restoring the three dict entries makes
     `test_geometry_free_keys_are_not_in_the_retained_table` fail.

⚠ The unqualified amino-acid entries (alanine, leucine, isoleucine,...) were first
left alone here, and were then removed from the flat retained table on purpose by
9292c013d (a phase T5, "stop fabricating implicit-L on stereo-undefined
AAs/esters"): "The stereodescriptors 'D' and 'L'" (the Blue Book)
says "The stereodescriptor 'ξ' (Greek letter xi) indicates unknown configuration",
and OPSIN reads a bare `alanine` as the defined L form, so the bare name cannot
describe a stereo-free input (same skeleton, stereo layer only on the OPSIN side).
`test_amino_acids_are_kept` now pins the current policy: the entry is absent from the
flat table, the bare name is still the amino-acid table's name for a defined-
configuration input, and the stereo-free input gets the systematic name.
"""

import pytest

from orthonym.data import ALL_RETAINED_NAMES

#: (geometry-free key, the removed name, the geometry it silently asserted)
_REMOVED = [
    ("CC=CC(=O)O", "crotonic acid", "(2E)"),
    ("CC=CC=CC(=O)O", "sorbic acid", "(2E,4E)"),
    ("CC=CC=O", "crotonaldehyde", "(2E)"),
]

#: Bare amino-acid names that belong to the amino-acid table (a defined-configuration
#: input), not to the flat retained table.
_KEPT_AMINO_ACIDS = [
    ("CC(N)C(=O)O", "alanine"),
    ("CC(C)CC(N)C(=O)O", "leucine"),
    ("CCC(C)C(N)C(=O)O", "isoleucine"),
    ("CC(O)C(N)C(=O)O", "threonine"),
]

#: What the stereo-free key is named instead (OPSIN 2.9.0 parses each to the key's full
#: InChIKey, which has no stereo layer: QNAYBMKLOCPYGJ / ROHFNLRQFUQHCH / AGPKZVBTJJNPAG /
#: AYFVYJQAPQTCCC, all -UHFFFAOYSA-N).
_STEREO_FREE_SYSTEMATIC = {
    "CC(N)C(=O)O": "2-aminopropanoic acid",
    "CC(C)CC(N)C(=O)O": "2-amino-4-methylpentanoic acid",
    "CCC(C)C(N)C(=O)O": "2-amino-3-methylpentanoic acid",
    "CC(O)C(N)C(=O)O": "2-amino-3-hydroxybutanoic acid",
}


def _defined_stereo(smiles):
    """Count of DEFINED stereo elements (chiral tags + bond stereo)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unreadable SMILES {smiles!r}"
    return (sum(1 for a in mol.GetAtoms()
                if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
            + sum(1 for b in mol.GetBonds()
                  if b.GetStereo() != Chem.BondStereo.STEREONONE))


@pytest.mark.parametrize("key,name,geometry", _REMOVED)
def test_geometry_free_keys_are_not_in_the_retained_table(key, name, geometry):
    """The mutation target: restoring any of these three entries fails here."""
    assert _defined_stereo(key) == 0, (
        f"premise broken -- {key!r} was supposed to leave geometry undefined"
    )
    assert ALL_RETAINED_NAMES.get(key) != name, (
        f"{name!r} is back for {key!r}. That key defines no geometry, but the name "
        f"denotes the {geometry} isomer specifically, so it asserts stereochemistry "
        f"the input never claimed -- and SELF-01 cannot see it."
    )


@pytest.mark.parametrize("key,name", _KEPT_AMINO_ACIDS)
def test_amino_acids_are_kept(key, name):
    """The bare amino-acid name stays with the amino-acid table, not the flat table.

    Policy of 9292c013d: a bare retained name asserts the DEFINED (L) configuration
     "The stereodescriptors 'D' and 'L'", the Blue Book: 'The
    stereodescriptor "xi" indicates unknown configuration'), so a structure drawn
    without stereochemistry must not be keyed to it in the stereo-blind flat table.
    The name is NOT lost: `STANDARD_AMINO_ACIDS` still carries it for a defined-
    configuration input, and the stereo-free input gets the systematic name.
    """
    from orthonym import Orthonym
    from orthonym.data.amino_acids import STANDARD_AMINO_ACIDS

    assert ALL_RETAINED_NAMES.get(key) != name, (
        f"{name!r} is back in the flat retained table for the stereo-free key {key!r}. "
        f"A bare amino-acid name asserts the L configuration (P-103.1.3.1); 9292c013d "
        f"removed these entries on purpose."
    )
    assert STANDARD_AMINO_ACIDS.get(key) == name, (
        f"{name!r} was lost from the amino-acid table, where defined-stereo inputs "
        f"resolve it."
    )
    assert Orthonym().name(key) == _STEREO_FREE_SYSTEMATIC[key]


@pytest.mark.opsin_gate
@pytest.mark.slow
@pytest.mark.parametrize("key,name,geometry", _REMOVED)
def test_old_names_denote_a_different_molecule_than_the_key(key, name, geometry,
                                                            opsin_gate):
    """ARTIFACT 2 -- independent check, using OPSIN rather than our own namer.

    The old value violated a necessary condition: a name must denote the molecule
    it was emitted for. Asserting that directly is what makes this a defect rather
    than a preference -- it does not depend on any judgement of ours.
    """
    from rdkit import Chem
    from rdkit.Chem import inchi

    from orthonym.validation.opsin_roundtrip import opsin_parse

    parsed = opsin_parse(name)
    if not parsed:
        pytest.skip(f"OPSIN could not parse {name!r}; the check needs its parse")
    got = inchi.MolToInchiKey(Chem.MolFromSmiles(parsed))
    want = inchi.MolToInchiKey(Chem.MolFromSmiles(key))
    assert got != want, (
        f"premise broken: {name!r} DOES denote {key!r}, so removing it was wrong "
        f"and this whole change should be reverted."
    )


@pytest.mark.opsin_gate
@pytest.mark.slow
def test_the_replacements_round_trip_exactly(opsin_gate):
    """The other half of a project rule: verify what is emitted AFTER the removal.

    Removing a wrong output can unmask a worse generator -- four times in. So
    this asserts the replacement is not merely different but correct.
    """
    from rdkit import Chem
    from rdkit.Chem import inchi

    from orthonym import Orthonym

    for key, old, _ in _REMOVED:
        emitted = Orthonym().name(key)
        from orthonym.validation.opsin_roundtrip import opsin_parse
        parsed = opsin_parse(emitted)
        assert parsed, f"{key!r} now emits {emitted!r}, which OPSIN cannot parse"
        assert (inchi.MolToInchiKey(Chem.MolFromSmiles(parsed))
                == inchi.MolToInchiKey(Chem.MolFromSmiles(key))), (
            f"{key!r} now emits {emitted!r} (was {old!r}) and it does NOT round-trip "
            f"-- the removal unmasked a worse generator"
        )
