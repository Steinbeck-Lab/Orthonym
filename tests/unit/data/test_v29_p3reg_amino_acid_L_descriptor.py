"""-REGRESSION I12 — a free amino acid must carry its L configuration.

THE DEFECT
----------
`data/amino_acids.py` discarded the `L-` descriptor on free amino acids while
keeping `D-`, so L-alanine and the (achiral) glycine were spelled the same way a
configuration-free name is spelled: `alanine`. The CIP verdict was computed
correctly and then thrown away —

    return "D-" if _get_stereo_prefix(mol, name) == "D-" else ""

— and the allo quartet hard-coded the same suppression (`L` -> `""`,
`L-allo` -> `"allo-"`). That is a stereo LOSS on one of the most common classes in
chemistry, and neither the gate nor OPSIN could see it: OPSIN resolves bare
`alanine` to the L structure, so the round-trip compares EQUAL and reports OK. The
gold rows pinned the bare names, so the gate agreed too. Orthonym's own stereo
backstop was the only thing complaining ("`alanine` has 1 R/S but name lacks
descriptors").

THE AUTHORITY — and the honest limit of it
------------------------------------------
* `## **** The stereodescriptors 'D' and 'L'` (the Blue Book), at
  :54293: "*The absolute configuration at the α-carbon atom of the α-amino
  carboxylic acids is designated by the stereodescriptor 'D' or 'L' to indicate a
  formal relationship to 'D- or L-glyceraldehyde'.*" NOTE: this sentence is
  INDICATIVE — it establishes that D/L are what designate the α-carbon, not an
  imperative "must be cited". The requirement is carried by the next two items.
* `### **** Indication of configuration in peptides` (:54715), at:54717:
  "*The stereodescriptor 'L' is not indicated in the names nor in the symbolic
  representation of peptides composed of amino acids listed in Table 10.4. In
  contrast, the stereodescriptor 'D' is indicated at the front of the acyl group or
  name of each component having that configuration.*" The omission licence is
  scoped to PEPTIDES (and, within them, to Table-10.4 members). A free amino acid
  is outside it — which is exactly what `rules/peptides.py` already says in its own
  docstring: "*the L-omission is a display rule applied here, so a standalone amino
  acid still shows L*". This module violated that stated invariant.
* `## **** Use of the prefix 'allo'` (:54320), the four worked
  examples at:54324-54330 — every one carries the descriptor, and each is offered
  in exactly two forms, neither of them bare:
      L-isoleucine... (2S,3S)-2-amino-3-methylpentanoic acid
      L-alloisoleucine... (2S,3R)-2-amino-3-methylpentanoic acid
      L-threonine... (2S,3R)-2-amino-3-hydroxybutanoic acid
      L-allothreonine... (2S,3S)-2-amino-3-hydroxybutanoic acid
  The code cited THIS section for emitting `allo-` with no `L-`; the section
  refutes it.

⚠ NOT A PIN CLAIM. `### ** INTRODUCTION**` (:50939), at:50943: "*Preferred
IUPAC names (PINs) are not identified for the compounds in this Chapter.*" So
`L-alanine` is the Blue Book's prescribed retained name for the free acid, NOT its
PIN — Chapter identifies no PINs at all. Nothing here should be described as
a PIN correction.

The peptide path must be untouched: `D-alanylglycine` (gold W5-A4) is a live
tripwire that the D descriptor is still cited at the front of its acyl group.
"""

import pytest

from orthonym.data.amino_acids import _aa_config_descriptor, get_amino_acid_name

pytestmark = pytest.mark.unit


# SMILES taken verbatim from the gold rows that pin these molecules, so the test
# and the gate are talking about the same structures.
# gold_pins.json #58/#59, stereo_config.json #13/#14/#15,
# amino_acid_derivatives.json #0..#7
FREE_AMINO_ACIDS = [
    # (SMILES, expected name, gold row it comes from)
    ("C[C@H](N)C(=O)O", "L-alanine", "gold_pins #59"),
    ("C[C@@H](N)C(=O)O", "D-alanine", "gold_pins #47"),
    ("C[C@@H](C(=O)O)N", "L-alanine", "stereo_config #13"),
    ("C([C@@H](C(=O)O)N)O", "L-serine", "gold_pins #58"),
    ("N[C@@H](CS)C(=O)O", "L-cysteine", "stereo_config #14"),
    ("N[C@H](CS)C(=O)O", "D-cysteine", "stereo_config #15"),
    ("NCC(=O)O", "glycine", "achiral — no descriptor"),
]

# The threonine/isoleucine quartet: two stereocentres, four retained names.
ALLO_QUARTET = [
    ("C[C@@H](O)[C@H](N)C(=O)O", "L-threonine", "amino_acid_derivatives #4"),
    ("C[C@H](O)[C@@H](N)C(=O)O", "D-threonine", "amino_acid_derivatives #5"),
    ("C[C@H](O)[C@H](N)C(=O)O", "L-allothreonine", "amino_acid_derivatives #0"),
    ("C[C@@H](O)[C@@H](N)C(=O)O", "D-allothreonine", "amino_acid_derivatives #1"),
    ("CC[C@H](C)[C@H](N)C(=O)O", "L-isoleucine", "amino_acid_derivatives #6"),
    ("CC[C@@H](C)[C@@H](N)C(=O)O", "D-isoleucine", "amino_acid_derivatives #7"),
    ("CC[C@@H](C)[C@H](N)C(=O)O", "L-alloisoleucine", "amino_acid_derivatives #2"),
    ("CC[C@H](C)[C@@H](N)C(=O)O", "D-alloisoleucine", "amino_acid_derivatives #3"),
]


def _canon(smiles):
    from rdkit import Chem
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)


def _lacks_configuration(name) -> bool:
    """Does this emitted amino-acid name carry NO configurational prefix?

    Extracted so the class-level sweep below has a predicate that is ITSELF
    pinned. Mutation-testing the first version of that sweep showed the inline
    predicate could be neutered — adding `""` to the `startswith` tuple makes it
    match everything, the sweep finds nothing, and it passes vacuously with no
    other test objecting. `TestTheDetector` closes that hole.
    """
    return bool(name) and not name.startswith(("L-", "D-"))


class TestTheDetector:
    """The sweep's predicate, pinned in both directions."""

    @pytest.mark.parametrize("name,expected", [
        ("alanine", True),
        ("threonine", True),
        ("allothreonine", True),          # the pre-fix spelling: no configuration
        ("cysteine", True),
        ("L-alanine", False),
        ("D-alanine", False),
        ("L-allothreonine", False),
        ("D-alloisoleucine", False),
        ("", False),                        # nothing emitted is not "bare"
        (None, False),
    ])
    def test_lacks_configuration(self, name, expected):
        assert _lacks_configuration(name) is expected


class TestFreeAminoAcidCarriesItsConfiguration:
    @pytest.mark.parametrize("smiles,expected,source", FREE_AMINO_ACIDS)
    def test_free_amino_acid_name(self, smiles, expected, source):
        assert get_amino_acid_name(_canon(smiles), with_descriptor=True) == expected

    @pytest.mark.parametrize("smiles,expected,source", ALLO_QUARTET)
    def test_allo_quartet(self, smiles, expected, source):
        assert get_amino_acid_name(_canon(smiles), with_descriptor=True) == expected

    def test_L_is_not_silently_dropped_by_the_descriptor_helper(self):
        """The helper itself must return 'L-', not ''.

        Pinned separately from the name so a future 'fix' that re-suppresses L one
        layer down cannot pass by having the caller re-add it.
        """
        from rdkit import Chem
        mol = Chem.MolFromSmiles("C[C@H](N)C(=O)O")
        assert _aa_config_descriptor(mol, "alanine") == "L-"

    def test_glycine_gets_no_descriptor(self):
        from rdkit import Chem
        assert _aa_config_descriptor(Chem.MolFromSmiles("NCC(=O)O"), "glycine") == ""

    def test_every_chiral_free_amino_acid_carries_a_descriptor(self):
        """Class-level assertion, not a name list: no chiral standard amino acid may
        come back bare. Catches a member the parametrized rows do not name."""
        bare = []
        for smiles, expected, _src in FREE_AMINO_ACIDS + ALLO_QUARTET:
            if expected == "glycine":
                continue
            got = get_amino_acid_name(_canon(smiles), with_descriptor=True)
            if _lacks_configuration(got):
                bare.append((smiles, got))
        assert bare == [], f"free amino acids emitted with no configuration: {bare}"


class TestPeptideSuppressionIsUntouched:
    """ is a PEPTIDE display rule and must keep working — the fix narrows
    where the rule applies, it does not delete the rule."""

    @pytest.mark.parametrize("smiles,expected", [
        # gold_pins #204 / #205 — L suppressed inside a peptide.
        ("C[C@@H](C(=O)N[C@@H](C)C(=O)O)N", "alanylalanine"),
        ("NCC(=O)N[C@@H](C)C(=O)O", "glycylalanine"),
    ])
    def test_peptide_L_still_suppressed(self, smiles, expected):
        from orthonym.rules.peptides import name_peptide
        from rdkit import Chem
        assert name_peptide(Chem.MolFromSmiles(smiles)) == expected

    def test_peptide_D_still_cited(self):
        """gold_pins #206 W5-A4 tripwire: a D residue IS cited at the front of its
        acyl group, and L-suppression must not also drop it.

        SMILES copied from the gold row. An earlier draft of this test used a
        hand-written `C[C@@H](C(=O)NCC(=O)O)N`, which is the L diastereomer and
        therefore correctly named `alanylglycine` — the test was wrong, not the code.
        """
        from orthonym.rules.peptides import name_peptide
        from rdkit import Chem
        mol = Chem.MolFromSmiles("C[C@H](C(=O)NCC(=O)O)N")
        assert name_peptide(mol) == "D-alanylglycine"

    def test_bare_name_contract_for_the_peptide_caller_is_preserved(self):
        """`with_descriptor=False` is the contract `name_peptide` relies on.

        Two halves, both documented at `get_amino_acid_name`: the stereo-FREE key
        returns the BARE retained name, and a stereo-TAGGED key returns ``None``
        because the strip fallback is deliberately gated to the descriptor path
        ("the DEFAULT path stays byte-identical... so name_peptide... is
        completely unaffected"). `name_peptide` relies on that ``None``.
        """
        from rdkit import Chem
        stereo_free = Chem.CanonSmiles(
            Chem.MolToSmiles(Chem.MolFromSmiles("C[C@H](N)C(=O)O"),
                             isomericSmiles=False))
        assert get_amino_acid_name(stereo_free) == "alanine"
        assert get_amino_acid_name(_canon("C[C@H](N)C(=O)O")) is None
