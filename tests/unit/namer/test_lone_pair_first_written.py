"""A lone-pair centre written first (TRIAGE.md 'Lone-pair centre written first -- naming and
the gate probe').

RDKit reads a lone-pair stereocentre written first in a SMILES string unlike an implicit
hydrogen in that position; OpenSMILES, CDK (the ``centres`` labeller) and OPSIN treat the lone
pair like an implicit hydrogen (Daylight SMILES theory: "If the central carbon is first in the
SMILES, the implicit hydrogen is taken to be the 'from' atom"; the RDKit Book: missing ligands
"in SMILES... are treated the same as implicit hydrogens"). So '[S@](=O)(C)CC' is (S) by the
standard reading and (R) for RDKit. The exit check of 96c536927 withdrew RDKit's (R) name, so
such a string was not named. The engine now names the standard reading
(``namer._lone_pair_standard_spelling``): '(S)-(methanesulfinyl)ethane', the form
('(S)-(methanesulfinyl)ethane (PIN)', the Blue Book). An input RDKit reads the standard
way is named unchanged.

The 12 spellings are the RDKit random spellings of 'CC[S@](=O)C' with which the gate's
determinism probe reported 'NEW NONDET: CC[S@](=O)C' at adee5416e. The expected name of each
spelling is derived from CDK's reading of it, and each expected name is read back by OPSIN's
own StdInChIKey (no RDKit on the name side).
"""
import subprocess
from functools import lru_cache

import pytest

from orthonym import Orthonym, namer
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.opsin_gate

R_NAME = "(R)-(methanesulfinyl)ethane"
S_NAME = "(S)-(methanesulfinyl)ethane"
# Spellings on which RDKit and CDK agree, one per enantiomer.
REFERENCE = {"R": "CC[S@@](C)=O", "S": "CC[S@](C)=O"}
# spelling -> CDK's label of its sulfur. The four that write the sulfur first are (S).
SPELLINGS = {
    "CC[S@](=O)C": "R",
    "[S@](=O)(C)CC": "S",
    "CC[S@@](C)=O": "R",
    "C(C)[S@](=O)C": "R",
    "O=[S@](C)CC": "R",
    "[S@](CC)(=O)C": "S",
    "C([S@@](C)=O)C": "R",
    "[S@@](=O)(CC)C": "S",
    "[S@](C)(CC)=O": "S",
    "C(C)[S@@](C)=O": "R",
    "C[S@](CC)=O": "R",
    "C([S@](=O)C)C": "R",
}
EXPECTED = {s: (R_NAME if lab == "R" else S_NAME) for s, lab in SPELLINGS.items()}
SULFUR_FIRST = sorted(s for s in SPELLINGS if s.startswith("[S"))


@lru_cache(maxsize=None)
def _opsin_own_stdinchikey(name: str) -> str:
    """OPSIN 2.9.0's own StdInChIKey of ``name`` (-ostdinchikey): no SMILES, no RDKit."""
    from orthonym.jvm_flags import JVM_HYGIENE_FLAGS
    from orthonym.validation.opsin_roundtrip import _find_opsin_jar
    jar = _find_opsin_jar("2.9.0")
    proc = subprocess.run(["java", *JVM_HYGIENE_FLAGS, "-jar", jar, "-ostdinchikey"],
                          input=name + "\n", capture_output=True, text=True, timeout=120)
    return proc.stdout.strip()


def _rdkit_key(smiles: str) -> str:
    from rdkit import Chem
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def _centres(smiles: str) -> dict:
    from orthonym.perception.centres_bridge import centres_label_batch
    return centres_label_batch([smiles])[smiles]


def _rdkit_cip(smiles: str) -> dict:
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler
    mol = Chem.MolFromSmiles(smiles)
    rdCIPLabeler.AssignCIPLabels(mol)
    return {a.GetIdx() + 1: a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")}


def test_references_are_read_alike_and_read_back_by_opsin():
    for label, ref in REFERENCE.items():
        assert _centres(ref) == {3: label}
        assert _rdkit_cip(ref) == {3: label}
    assert _opsin_own_stdinchikey(R_NAME) == _rdkit_key(REFERENCE["R"])
    assert _opsin_own_stdinchikey(S_NAME) == _rdkit_key(REFERENCE["S"])
    assert _rdkit_key(REFERENCE["R"]) != _rdkit_key(REFERENCE["S"])


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_expected_name_is_cdks_reading(spelling):
    # CDK's one label of the input as written is the descriptor of the expected name, and
    # RDKit's reading differs exactly for the spellings that write the sulfur first.
    (label,) = _centres(spelling).values()
    assert label == SPELLINGS[spelling]
    assert EXPECTED[spelling].startswith(f"({label})-")
    (rdkit_label,) = _rdkit_cip(spelling).values()
    assert (rdkit_label != label) == (spelling in SULFUR_FIRST)
    assert _opsin_own_stdinchikey(EXPECTED[spelling]) == _rdkit_key(REFERENCE[label])


@pytest.mark.parametrize("tier", ["best-effort", "pin"])
@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_spelling_is_named_as_its_standard_reading(spelling, tier):
    namer_ = (Orthonym(style="pin") if tier == "pin"
              else Orthonym(style="pin", **_emit_tier_flags(tier)))
    res = namer_.name_tiered(spelling)
    assert res.get("name") == EXPECTED[spelling]
    assert res.get("tier") == "pin_verified"
    assert res.get("is_pin") is True
    # The exit check of 96c536927, on its own terms: CDK's reading of the caller's string
    # as written against OPSIN's own SMILES of the name.
    assert namer._lone_pair_configuration_verified(res.get("name"), spelling) is True


def test_other_entry_points_name_the_standard_reading():
    o = Orthonym()
    for spelling in SULFUR_FIRST:
        assert o.name(spelling) == S_NAME
        assert o.name_with_confidence(spelling)["name"] == S_NAME
        assert o.name_with_tree(spelling).name == S_NAME


def test_an_input_rdkit_reads_the_standard_way_is_unchanged():
    for spelling in SPELLINGS:
        spelled = namer._lone_pair_standard_spelling(spelling)
        if spelling in SULFUR_FIRST:
            assert spelled == REFERENCE["S"]
        else:
            assert spelled is spelling
    # RDKit's CIP labels of its reading equal CDK's labels of the string as written --
    # RDKit reads these the standard way (the P rows of test_prefix_order_fallback.py
    # and the N-first aziridine of test_lone_pair_configuration.py, whose names are
    # limited by RDKit's canonical SMILES, not by its reading): unchanged.
    for smiles in ("C[C@H](O)CC", "CCO", "C[S@@](=O)c1ccccc1",
                   "C1(=CC=CC=C1)N1[P@@](N2CCC[C@H]2C1)OC(C)C",
                   "C1(=CC=CC=C1)N1[P@](N2CCC[C@H]2C1)OC(C)C",
                   "[N@@]1(Cc2cc(-c3cccc4c3OC(F)(F)O4)ccc2F)[C@@H](C(OC)=O)C1"):
        if "@" in smiles:
            assert _rdkit_cip(smiles) == _centres(smiles), smiles
        assert namer._lone_pair_standard_spelling(smiles) is smiles
    # RDKit misreads the ring-closure P / N of these rows, but its canonical SMILES of
    # the inverted reading repeats the misreading, so CDK does not read it as the input
    # and the input is kept; the engine's CIP labels (CDK's, of RDKit's canonical SMILES)
    # are the standard reading's there already.
    for smiles in ("CO[P@@]1OC[C@H](C)O1", "CC(C)O[P@]1N(c2ccccc2)C[C@@H]2CCCN21",
                   "COC(=O)[C@H]1C[N@]1Cc1cc(-c2cccc3c2OC(F)(F)O3)ccc1F"):
        assert _rdkit_cip(smiles) != _centres(smiles), smiles
        assert namer._lone_pair_standard_spelling(smiles) is smiles


def test_the_exit_check_reads_the_callers_string():
    spelling = "[S@](=O)(C)CC"
    assert namer._lone_pair_configuration_verified(R_NAME, spelling) is False
    assert namer._lone_pair_configuration_verified(S_NAME, spelling) is True
    spelled, opened = namer._lone_pair_input_enter(spelling)
    try:
        assert opened and spelled == REFERENCE["S"]
        assert namer._lone_pair_as_written(spelled) == spelling
        # A nested public call names the string it is given.
        assert namer._lone_pair_input_enter("[S@](C)(CC)=O") == ("[S@](C)(CC)=O", False)
        # An atom map of the named string is re-keyed to the caller's atoms.
        from rdkit import Chem
        named, given = Chem.MolFromSmiles(spelled), Chem.MolFromSmiles(spelling)
        rekeyed = namer._lone_pair_caller_atoms({k: k for k in range(named.GetNumAtoms())})
        assert sorted(rekeyed) == list(range(given.GetNumAtoms()))
        for caller_idx, named_idx in rekeyed.items():
            a, b = given.GetAtomWithIdx(caller_idx), named.GetAtomWithIdx(named_idx)
            assert (a.GetSymbol(), a.GetDegree()) == (b.GetSymbol(), b.GetDegree())
    finally:
        namer._lone_pair_input_exit(opened)
    assert namer._lone_pair_as_written(REFERENCE["S"]) == REFERENCE["S"]
    assert namer._lone_pair_input_enter(spelling)[1] is True
    namer._lone_pair_input_exit(True)


def test_without_centres_the_input_is_unchanged(monkeypatch):
    import orthonym.perception.centres_bridge as cb
    monkeypatch.setattr(cb, "centres_label_batch", lambda smiles, timeout=120.0: None)
    for spelling in SULFUR_FIRST:
        assert namer._lone_pair_standard_spelling(spelling) is spelling
