""" a phase (E3) Tasks 4-6: producer ASSEMBLY/enumeration fixes.

All three witnesses share one shape: the substituent FRAGMENT names correctly
(verified directly against `name_substituent_fragment`/`classify_and_name_fragment`
in internal notes), but the PARENT producer either
never enumerated the branch, mis-assembled it, or double-counted an atom it
already spoke for. Fixed sites (all re-anchored by direct a trace, not by the
plan doc's guessed location -- `name_polyfunctional` was OFF PATH for two of
the three witnesses, confirmed via producer tracing):

- Task 4 (saccharopine): `assembly/substituent_enumerator.py::_name_amino_branch`
  declined a single N-branch whenever it carried its OWN heteroatom
  decoration (has_hetero gate), so a backbone-N substituent never got
  enumerated. Fixed by routing the has_hetero case through
  `composed_prefix_organyl_name` (already proven correct for exactly this
  shape). A SEPARATE double-count bug then surfaced in
  `assembly/handlers/_handler_shared.py` (a non-principal secondary-amine FG
  match re-emitted the same nitrogen as a bare "amino" alongside the correctly
  named branch) and is fixed there too.
- Task 5 (anandamide analogue): `rules/polyfunctional.py::name_polyfunctional`
  had no exclusion for a PRINCIPAL AMIDE's own N-branch (only a PRINCIPAL
  AMINE had one), so the amide nitrogen -- bonded directly to the chain's
  acyl carbon -- was walked as an ordinary substituent and mis-named a plain
  "(R)amino" prefix ON C1. Fixed by mirroring the existing amine exclusion +
  N-prefix block for a stereo-bearing on-chain secondary/tertiary amide (the
  case the earlier stereo-free amide delegate declines). A companion fix
  removes a DESTRUCTIVE side effect of `Chem.FindMolChiralCenters` (called,
  on the SHARED mol, inside the amide guard a few lines earlier) that wiped
  the molecule's already-assigned E/Z `_CIPCode` labels for every later
  consumer in the same naming attempt -- without it, the corrected assembly
  still shipped stereo-incomplete and was safely suppressed.
- Task 6 (CHEBI:168479 shape): a non-principal ALDEHYDE fully contained in an
  already-correctly-named chain substituent branch was ALSO re-emitted as a
  second, unlocated "oxo" prefix on the parent
  (`assembly/handlers/_handler_shared.py`'s FG-prefix loop), grafting a
  phantom carbonyl onto the parent chain. Fixed with a narrow, verified guard
  (NOT a `BRANCH_HANDLED_FGS` addition -- that blanket-trust set deliberately
  excludes 'aldehyde'/'ketone'/... per
  `test_bugb_guard.py::test_no_dangerous_entries`, and blindly trusting it
  unmasks a SEPARATE pre-existing bug where a `-CH2-CHO` branch is mis-named
  "(2-hydroxyethyl)" by the compound-substituent namer). The guard instead
  requires concrete evidence: the branch containing the aldehyde must already
  have been rendered as one prefix whose TEXT literally mentions "oxo".
  The harder witness (168479 itself) still ABSTAINS: its arm's own "(Z)-"
  double-bond descriptor is dropped by a DIFFERENT, deeper namer-capability
  gap in `assembly/substituent_naming.py::name_substituent_fragment` (a
  second, PARALLEL substituent chokepoint that has no stereo-injection path
  at all) -- out of scope for an assembly-only fix; correctly
  suppresses the resulting name rather than shipping it, so 0-wrong holds
  and the molecule FALLS THROUGH to abstain, per this batch's explicit
  license for an unresolved residual.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi
from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _full_rt(smiles: str, name: str) -> bool:
    o = opsin_parse(name)
    if not o:
        return False
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(o))


# ---------------------------------------------------------------------------
# Task 4 -- saccharopine (CHEBI:16927): N-substituent off a backbone N
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_saccharopine_names_and_full_rt(namer):
    # CHEBI:16927 saccharopine. ChEBI's own name uses the retained
    # "L-glutamic acid" parent with an "N-[...]" locant, a convention this
    # tree does not build ANYWHERE for a non-standard N-substituted retained
    # amino acid (verified: sarcosine, CNCC(=O)O, already emits the
    # systematic "(methylamino)acetic acid", not "N-methylglycine"). The
    # systematic equivalent this fix produces --
    # "(2S)-2-{[(5-amino-5-carboxypentyl]amino}pentanedioic acid" spelled
    # with a bare "(S)" stereo descriptor on the substituent -- denotes the
    # IDENTICAL molecule (RT-verified below, full InChIKey incl. the stereo
    # layer). Before this fix the whole molecule abstained: the N-branch was
    # never enumerated at all (suppressed the atom-incomplete
    # '(2S)-aminopentanedioic acid' candidate).
    smi = "N[C@@H](CCCCN[C@@H](CCC(=O)O)C(=O)O)C(=O)O"
    name = namer.name(smi)
    assert name == "(2S)-2-{[(S)-5-amino-5-carboxypentyl]amino}pentanedioic acid", name
    assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
def test_saccharopine_integration_via_name_tiered(namer):
    # CHOKE-POINT-OFF-PATH guard: assert through the SAME integration surface
    # a caller actually uses, not only namer.name, since a direct-helper
    # test can pass while the real dispatch path never reaches the fix.
    smi = "N[C@@H](CCCCN[C@@H](CCC(=O)O)C(=O)O)C(=O)O"
    result = namer.name_tiered(smi)
    name = result.get("name")
    assert name == "(2S)-2-{[(S)-5-amino-5-carboxypentyl]amino}pentanedioic acid", result
    assert _full_rt(smi, name), name


# ---------------------------------------------------------------------------
# Task 5 -- anandamide analogue (5,8,11-icosatrienoyl ethanolamide):
# N-substituent assembled as N-(...)amide, not a C1-amino prefix
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_anandamide_analogue_names_and_full_rt(namer):
    # NOTE the coordinator-supplied witness in the task brief
    # ('CCCCC/C=C\\C/C=C\\C/C=C\\CCCC(=O)NCCO', 17 carbons) does NOT match
    # its own quoted target name/InChIKey (icosa- = 20 carbons, trien- = 3
    # double bonds = 5,8,11-icosatrienoyl / "Mead acid" ethanolamide, C22
    # overall). Verified the CORRECT witness by reconstructing the target
    # name's structure and confirming its InChIKey14 is YKGQBEGMUSSPFY,
    # matching the brief exactly; the given SMILES's InChIKey14 does not
    # match anything in the brief. Do not requote the given SMILES elsewhere.
    smi = "CCCCCCCC/C=C\\C/C=C\\C/C=C\\CCCC(=O)NCCO"
    name = namer.name(smi)
    assert name == "(5Z,8Z,11Z)-N-(2-hydroxyethyl)icosa-5,8,11-trienamide", name
    assert _full_rt(smi, name), name
    assert inchi.MolToInchiKey(Chem.MolFromSmiles(smi)).startswith("YKGQBEGMUSSPFY")


@pytest.mark.opsin_gate
def test_anandamide_analogue_integration_via_name_tiered(namer):
    smi = "CCCCCCCC/C=C\\C/C=C\\C/C=C\\CCCC(=O)NCCO"
    result = namer.name_tiered(smi)
    name = result.get("name")
    assert name == "(5Z,8Z,11Z)-N-(2-hydroxyethyl)icosa-5,8,11-trienamide", result
    assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
def test_short_chain_amide_control_unchanged(namer):
    # REGRESSION guard (task brief's explicit control): the short, saturated
    # acyl case must stay byte-identical -- it already succeeded via the
    # EARLIER, stereo-free amide delegate (rules/amides.py::name_amide),
    # which `return`s before ever reaching the new N-substituent /
    # FindMolChiralCenters-copy code added for the stereo-bearing case.
    smi = "CCCCC(=O)NCCO"
    name = namer.name(smi)
    assert name == "N-(2-hydroxyethyl)pentanamide", name
    assert _full_rt(smi, name), name
    # And through the integration surface too.
    assert namer.name_tiered(smi).get("name") == "N-(2-hydroxyethyl)pentanamide"


# ---------------------------------------------------------------------------
# Task 6 -- CHEBI:168479 shape: parent aldehyde mis-expressed as an
# unlocanted oxo (duplicate carbonyl on the parent chain)
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_168479_no_longer_ships_a_duplicate_carbonyl(namer):
    # CHEBI:168479, 'C/C=C(\\C=O)C(CC(=O)O)CC(=O)O'. Before this fix the
    # emitted name was 'oxo-3-(1-oxobut-2-en-2-yl)pentanedioic acid' --
    # OPSIN-parseable, but to a 2-OXOpentanedioic acid the input SMILES does
    # not contain (a wrong, different molecule). After the fix the duplicate
    # unlocated "oxo" is gone and the name is atom-correct
    # ('3-(1-oxobut-2-en-2-yl)pentanedioic acid', the exact target from
    # CHEBI:168479's own reference name minus its "(Z)-" descriptor).
    #
    # The arm's own "(Z)-" stereo descriptor is still dropped by a SEPARATE,
    # deeper namer-capability gap (`substituent_naming.py::
    # name_substituent_fragment` has no stereo-injection path at all, unlike
    # the sibling `substituent_enumerator.py::name_substituent` cascade) --
    # out of scope for an assembly fix. correctly refuses to ship the
    # now-stereo-incomplete name, so the molecule FALLS THROUGH to abstain
    # (0-wrong holds; this is the explicitly-licensed residual for Task 6).
    smi = "C/C=C(\\C=O)C(CC(=O)O)CC(=O)O"
    name = namer.name(smi)
    assert name == "unknown organic compound", (
        f"expected the documented fall-through abstain, got: {name}"
    )
    assert inchi.MolToInchiKey(Chem.MolFromSmiles(smi)).startswith("IVZLDVMNNVAWDA")


@pytest.mark.opsin_gate
def test_aldehyde_branch_duplicate_oxo_fixed_same_shape_no_stereo(namer):
    # Same defect class as 168479 but with NO stereo double bond in the
    # branch, so the fix reaches a full, RT-exact win end to end (proves the
    # duplicate-oxo assembly fix itself, isolated from Task 6's separate
    # residual stereo gap). Before the fix: 'oxo-2-(2-oxoethyl)pentanoic
    # acid' (duplicate carbonyl, -suppressed). After: clean and
    # RT-exact.
    smi = "OC(=O)C(CC=O)CCC"
    name = namer.name(smi)
    assert name == "2-(2-oxoethyl)pentanoic acid", name
    assert _full_rt(smi, name), name
    assert namer.name_tiered(smi).get("name") == "2-(2-oxoethyl)pentanoic acid"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # REGRESSION guard (task brief): ordinary polyfunctional acids must be
    # unaffected -- one amino acid, one hydroxy acid, both unrelated to any
    # of the three defect classes above.
    ("CC(N)C(=O)O", "2-aminopropanoic acid"),
    ("CC(O)CC(=O)O", "3-hydroxybutanoic acid"),
])
def test_ordinary_polyfunctional_acids_unchanged(namer, smi, expected):
    name = namer.name(smi)
    assert name == expected, name
    assert _full_rt(smi, name), name


def test_bugb_guard_contract_untouched():
    # Task 6's fix was deliberately NOT a BRANCH_HANDLED_FGS addition -- guard
    # against a future regression re-adding 'aldehyde' (or a sibling
    # high-seniority FG) to that blanket-trust set, which
    # test_bugb_guard.py::test_no_dangerous_entries already protects
    # structurally. Re-asserted here as a cross-file tripwire since the two
    # files are easy to touch independently.
    from orthonym.assembly.naming_utils import BRANCH_HANDLED_FGS
    dangerous = {
        'carboxylic_acid', 'aldehyde', 'ketone', 'nitrile',
        'primary_amide', 'ester', 'anhydride', 'acid_chloride',
    }
    assert not (dangerous & BRANCH_HANDLED_FGS), (
        f"BRANCH_HANDLED_FGS must not contain: {dangerous & BRANCH_HANDLED_FGS}"
    )
