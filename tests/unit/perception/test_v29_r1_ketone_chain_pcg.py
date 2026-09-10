""" R1: the principal chain must contain the ketone's CARBONYL CARBON.

The defect (a SILENT ATOM DROP -- a wrong-molecule emission, the one outcome
this project treats as unacceptable):

    CCCCCCCCC(CCCC)C(C)(CC(C)C)C(=O)C C21H42O
      emitted 5-butyl-2,4-dimethyltridecan-4-one C19H38O <-- 2 carbons gone
      correct 4-butyl-3-methyl-3-(2-methylpropyl)dodecan-2-one

``perception.chains.find_principal_chain`` built ``fg_atoms`` /
``fg_bearing_carbons`` from the WHOLE SMARTS match. The ketone pattern
``[#6][CX3](=O)[#6]`` (``perception/functional_groups.py``) carries BOTH
flanking carbons, so a chain running through a mere NEIGHBOUR of the carbonyl
scored ``contains_fg=1`` / ``fg_count=1``. Criterion 3 (chain length) then
handed the win to a longer carbonyl-FREE chain; the acyl carbons were dropped as
an unnameable substituent (````) and the ``=O`` was re-expressed as a
``-one`` suffix on the attachment atom.

Blue Book authority (heading + deciding sentence, both opened and read):

   "Acyclic ketones" (the Blue Book Blue Book) -- "Unsubstituted
  acyclic ketones are systematically named in two ways: (1) substitutively, using
  the suffix 'one'... Method (1) generates preferred IUPAC names." Its own PIN
  examples -- ``butan-2-one (PIN)``, ``heptan-3-one (PIN)``,
  ``5-methylhexan-2-one (PIN)`` -- all number the CARBONYL CARBON as a skeletal
  atom of the parent chain. A flanking carbon never bears the 'one' locant.

   (:18875) -- "The senior parent structure has the maximum number of
  substituents corresponding to the principal characteristic group (suffix) or
  senior parent hydride in accord with the seniority of classes and the
  seniority of suffixes."

   (:18873) -- these criteria "must always be applied before those
  applicable to rings and ring systems (see and to chains (see ".

Together: a chain omitting the carbonyl carbon bears ZERO ketones, so it loses at
 before chain length is ever consulted.

The fix reuses ``SKELETAL_SUFFIX_PGS`` + ``_pg_attachment_atoms``, the primitives
that already encode exactly this for RINGS (``parent_selection.py:265`` -- "the
bonded-to-ring relaxation is invalid and mis-parented every aryl ketone").
"""

import os
import subprocess
import tempfile

import pytest
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from orthonym import Orthonym
from orthonym.perception.chains import find_principal_chain
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.parent_selection import SKELETAL_SUFFIX_PGS

R1_SMILES = "CCCCCCCCC(CCCC)C(C)(CC(C)C)C(=O)C"
R1_NAME = "4-butyl-3-methyl-3-(2-methylpropyl)dodecan-2-one"
R1_FORMULA = "C21H42O"

# The wrong name the defect produced. Pinned so it can never come back.
R1_DEFECT_NAME = "5-butyl-2,4-dimethyltridecan-4-one"

# Molecules where the longest carbon chain EXCLUDES the carbonyl carbon, so the
# whole-match contains-check picked a carbonyl-free parent.
DROPPING_POSITIVES = [
    (R1_SMILES, R1_NAME, R1_FORMULA),
    ("CCCCC(C)(CC(C)C)C(=O)C", "3-methyl-3-(2-methylpropyl)heptan-2-one", "C12H24O"),
    ("CCCCCCC(C)(CCC)C(=O)C", "3-methyl-3-propylnonan-2-one", "C13H26O"),
]

# Straight ketones whose longest chain already contains the carbonyl. These must
# be byte-identical; the last is the Blue Book's own PIN example.
UNCHANGED_CONTROLS = [
    ("CCC(C)=O", "butan-2-one"),
    ("CCCCCCC(C)=O", "octan-2-one"),
    ("CC(C)CCC(C)=O", "5-methylhexan-2-one"),
]


def _carbonyl_carbon(mol) -> int:
    """The single ketone carbonyl carbon, found independently of the fix."""
    hits = mol.GetSubstructMatches(Chem.MolFromSmarts("[#6][CX3](=O)[#6]"))
    carbons = {m[1] for m in hits}
    assert len(carbons) == 1, f"expected exactly one ketone C, got {carbons}"
    return carbons.pop()


def _opsin_formula(name: str):
    """Parse ``name`` with real OPSIN and return the molecular formula.

    Invoked with a temp FILE, never the name as a CLI argument -- OPSIN reads a
    bare argument as a filename and silently yields nothing.
    """
    from orthonym.validation.opsin_roundtrip import _find_opsin_jar

    jar = _find_opsin_jar()
    if jar is None:
        pytest.skip("OPSIN jar not available")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(name + "\n")
        path = fh.name
    try:
        out = subprocess.run(
            ["java", "-jar", jar, "-osmi", path],
            capture_output=True, text=True, timeout=300,
        ).stdout.strip()
    finally:
        os.unlink(path)
    if not out:
        return None
    mol = Chem.MolFromSmiles(out.splitlines()[0].strip())
    return CalcMolFormula(mol) if mol is not None else None


class TestPrincipalChainKeepsTheCarbonyl:
    """The unit-level invariant, no OPSIN, no naming."""

    @pytest.mark.parametrize("smiles", [p[0] for p in DROPPING_POSITIVES])
    def test_chain_contains_the_carbonyl_carbon(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, principal_group="ketone")
        cc = _carbonyl_carbon(mol)
        assert cc in chain, (
            f"P-44.1.1: the parent chain must bear the ketone, but the carbonyl "
            f"carbon (atom {cc}) is absent from chain {chain}. A chain without "
            f"it bears ZERO principal characteristic groups."
        )

    @pytest.mark.parametrize("smiles", [c[0] for c in UNCHANGED_CONTROLS])
    def test_controls_also_contain_the_carbonyl_carbon(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, principal_group="ketone")
        assert _carbonyl_carbon(mol) in chain

    def test_ketone_is_a_skeletal_suffix_pg(self):
        # The fix is scoped by this membership set; if ketone ever left it the
        # whole-match semantics would silently return.
        assert "ketone" in SKELETAL_SUFFIX_PGS


class TestEmittedName:
    @pytest.mark.parametrize("smiles,expected,_formula", DROPPING_POSITIVES)
    def test_name_is_the_carbonyl_bearing_parent(self, smiles, expected, _formula):
        assert Orthonym().name(smiles) == expected

    def test_the_atom_dropping_name_is_never_emitted_again(self):
        assert Orthonym().name(R1_SMILES) != R1_DEFECT_NAME

    @pytest.mark.parametrize("smiles,expected", UNCHANGED_CONTROLS)
    def test_plain_ketones_are_unchanged(self, smiles, expected):
        assert Orthonym().name(smiles) == expected

    def test_gate_on_and_gate_off_agree_on_the_constitution(self):
        # Before the fix these DIFFERED: gate off emitted the 2-carbon-short
        # name, gate on suppressed it and abstained. The gate must not be what
        # decides the constitution.
        off = Orthonym(_disable_opsin_validity_gate=True).name(R1_SMILES)
        on = Orthonym().name(R1_SMILES)
        assert off == on == R1_NAME


class TestFormulaEquality:
    """The specific invariant the defect violated: no atom may be dropped."""

    @pytest.mark.parametrize("smiles,expected,formula", DROPPING_POSITIVES)
    def test_emitted_name_has_the_input_molecular_formula(
        self, smiles, expected, formula
    ):
        name = Orthonym().name(smiles)
        assert CalcMolFormula(Chem.MolFromSmiles(smiles)) == formula
        out = _opsin_formula(name)
        assert out is not None, f"OPSIN could not parse the emitted name {name!r}"
        assert out == formula, (
            f"SILENT ATOM DROP: input {formula} but the emitted name {name!r} "
            f"denotes {out}"
        )

    def test_the_old_defect_name_really_did_lose_two_carbons(self):
        # Guards the premise itself: if this ever stops holding, the regression
        # tests above are pinning something other than the reported defect.
        # (The defect name is not OPSIN-parseable as emitted -- locant 4 is
        # quaternary -- so the comparison uses the valence-legal repair the
        # C2 investigation used, which isolates the constitution error.)
        legal_repair = "5-butyl-2,4-dimethyltridecan-7-one"
        out = _opsin_formula(legal_repair)
        assert out == "C19H38O"
        assert out != R1_FORMULA
