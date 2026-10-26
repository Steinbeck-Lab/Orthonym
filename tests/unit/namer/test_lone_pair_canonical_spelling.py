"""A lone-pair stereocentre that RDKit reads like the standard reading but writes, or whose OPSIN
SMILES it reads, the other way (PubChem 1M, lane S2: five phosphorus rows lost their names).

OpenSMILES, CDK (the ``centres`` labeller) and OPSIN treat a lone pair like an implicit hydrogen;
RDKit does not in two positions of a string, the atom written first and a ring-closure digit at
P(III) or an aziridine N (the module docstring of ``orthonym.perception.lone_pair_reading``). Two
things followed:

* the descriptor. The engine labels RDKit's canonical SMILES of the Mol with ``centres``. For
  'C1C[C@H]([P@@](C1)c1ccccc1)c1ccccc1', which RDKit reads the standard way, RDKit's canonical
  SMILES puts a ring-closure digit at the P and CDK reads that string as the other configuration,
  so the name was (1S,2S) for the (1R,2S) input and the exit check withdrew it
  (``namer._lone_pair_standard_spelling``, now compared by the labels the engine gives).
* the read-back. The full-key round trip compared RDKit's key of the input with RDKit's key of OPSIN's
  SMILES, so it rejected right names and accepted the mirror image whenever the two strings put the
  centre in positions RDKit reads differently (``perception.lone_pair_reading``).

The expectation of every name below rests on OPSIN 2.9.0's own StdInChIKey of the name (no SMILES, no
RDKit: ``-ostdinchikey``), against RDKit's key of the input where RDKit reads the input the standard
way, and against CDK's reading of the input as written (``namer._lone_pair_configuration_verified``,
the exit check) everywhere.
"""
import subprocess
from functools import lru_cache

import pytest
from rdkit import Chem

from orthonym import Orthonym, namer
from orthonym.cli import _emit_tier_flags
from orthonym.perception.lone_pair_reading import standard_reading_inchikey, standard_reading_mol

pytestmark = pytest.mark.opsin_gate

# PubChem 1M lane S2, the five rows (benchmarks/pubchem_1m, singles r102 r358 r665 r322 r633)
R102 = "C1C[C@H]([P@@](C1)C2=CC=CC=C2)C3=CC=CC=C3"
R358 = "C[C@H](C[C@H]1CCC[P@]1C2=CC=CC=C2)P(C3=CC=CC=C3)C4=CC=CC=C4"
R665 = "COC1C([C@@H]2[C@H](O1)[C@H]3CO[P@](O3)O2)C(=O)OC"
R322 = ("C[C@]1([C@@H]2CCCN2[P@](O1)O[C@@H]3[C@H](O[C@H]([C@@H]3F)N4C=NC5=C(N=CN=C54)NC(=O)C6=CC=CC=C6)"
        "COC(C7=CC=CC=C7)(C8=CC=C(C=C8)OC)C9=CC=C(C=C9)OC)C1CCCC1")
R633 = ("C[C@]12CCCC[C@@H]1N([P@@](O2)O[C@@H]3[C@H](O[C@H]([C@@H]3F)N4C=NC5=C(N=CN=C54)NC(=O)C6=CC=CC=C6)"
        "COC(C7=CC=CC=C7)(C8=CC=C(C=C8)OC)C9=CC=C(C=C9)OC)C")
# P with a ring-closure digit in the input (RDKit misreads it) and RDKit's own canonical SMILES
P_RING_ROW = "CO[P@@]1OC[C@H](C)O1"
FIRST_ATOM_SULFOXIDE = "[S@@](C)(=O)c1ccccc1"
# P(III) not at a ring-closure position, N-ring: RDKit reads it as CDK does and writes it otherwise
N_RING_P = "C1(=CC=CC=C1)N1[P@@](N2CCC[C@H]2C1)OC(C)C"
N_RING_P_OTHER = "C1(=CC=CC=C1)N1[P@](N2CCC[C@H]2C1)OC(C)C"
AZIRIDINE_FIRST = "[N@@]1(Cc2cc(-c3cccc4c3OC(F)(F)O4)ccc2F)[C@@H](C(OC)=O)C1"

R102_NAME = "(1R,2S)-1,2-diphenylphospholane"
R358_NAME = "(1R,2R)-2-[(2R)-2-(diphenylphosphanyl)propyl]-1-phenylphospholane"
R665_NAME = ("methyl (1R,2S,6R,8S)-4-methoxy-3,7,9,11-tetraoxa-8-phosphatricyclo[6.2.1.0^2,6]"
             "undecane-5-carboxylate")
R322_NAME = ("6-benzamido-9-[(2R,3R,4R,5R)-5-{[bis(4-methoxyphenyl)(phenyl)methoxy]methyl}-4-"
             "{[(2S,4R,5S)-4-cyclopentyl-4-methyl-3-oxa-1-aza-2-phosphabicyclo[3.3.0]octan-2-yl]oxy}-"
             "3-fluorooxolan-2-yl]-9H-purine")
R633_NAME = ("6-benzamido-9-[(2R,3R,4R,5R)-5-{[bis(4-methoxyphenyl)(phenyl)methoxy]methyl}-4-"
             "{[(1S,6S,8R)-6,9-dimethyl-7-oxa-9-aza-8-phosphabicyclo[4.3.0]nonan-8-yl]oxy}-"
             "3-fluorooxolan-2-yl]-9H-purine")


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
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def _opsin_smiles(name: str, smiles: str) -> str:
    from tests.support.rt_assert import _independent_parse
    return _independent_parse(name, smiles)


def _row(smiles: str, tier: str) -> dict:
    namer_ = (Orthonym(style="pin") if tier == "pin"
              else Orthonym(style="pin", **_emit_tier_flags(tier)))
    return namer_.name_tiered(smiles)


# ---------------------------------------------------------------------------------------------
# The reading: OPSIN's own key is the one the standard reading gives
# ---------------------------------------------------------------------------------------------

READ_ROWS = [
    # (input, right name, the name of the other configuration at the lone-pair centre)
    (R102, R102_NAME, "(1S,2S)-1,2-diphenylphospholane"),
    (P_RING_ROW, "(2R,4S)-2-methoxy-4-methyl-1,3,2-dioxaphospholane",
     "(2S,4S)-2-methoxy-4-methyl-1,3,2-dioxaphospholane"),
    (FIRST_ATOM_SULFOXIDE, "(S)-(methanesulfinyl)benzene", "(R)-(methanesulfinyl)benzene"),
    (R358, R358_NAME, "(1S,2R)-2-[(2R)-2-(diphenylphosphanyl)propyl]-1-phenylphospholane"),
]


@pytest.mark.parametrize("smiles,right,other", READ_ROWS)
def test_the_standard_reading_key_is_opsins_own_key(smiles, right, other):
    # OPSIN's own key of each name, and the standard-reading key of OPSIN's SMILES of it: the same.
    for name in (right, other):
        assert standard_reading_inchikey(_opsin_smiles(name, smiles)) == _opsin_own_stdinchikey(name)
    assert _opsin_own_stdinchikey(right) != _opsin_own_stdinchikey(other)
    # and the key of the input as written is that of the right name, not of the other
    assert standard_reading_inchikey(smiles) == _opsin_own_stdinchikey(right)
    assert namer._lone_pair_configuration_verified(right, smiles) is True
    assert namer._lone_pair_configuration_verified(other, smiles) is False


def test_rdkits_key_of_opsins_smiles_is_the_mirror_image_for_the_right_name():
    # The cause, witnessed: the input of R102 is read the standard way (RDKit's key is OPSIN's own key
    # of the right name), OPSIN's SMILES of that name carries a ring-closure digit at the P and RDKit
    # reads it as the other configuration, so the old comparison rejected the right name.
    assert _rdkit_key(R102) == _opsin_own_stdinchikey(R102_NAME)
    opsin = _opsin_smiles(R102_NAME, R102)
    assert _rdkit_key(opsin) != _opsin_own_stdinchikey(R102_NAME)
    assert standard_reading_inchikey(opsin) == _opsin_own_stdinchikey(R102_NAME)


def test_the_read_back_oracle_reads_the_lone_pair_centre_the_standard_way():
    # tests.support.rt_assert.name_is_rt_exact shared the defect: it said False for the right name of
    # R102 and True for the name of the other configuration
    from tests.support.rt_assert import name_is_rt_exact
    assert name_is_rt_exact(R102_NAME, R102)
    assert not name_is_rt_exact("(1S,2S)-1,2-diphenylphospholane", R102)
    assert name_is_rt_exact("(2R,4S)-2-methoxy-4-methyl-1,3,2-dioxaphospholane", P_RING_ROW)
    assert not name_is_rt_exact("(2S,4S)-2-methoxy-4-methyl-1,3,2-dioxaphospholane", P_RING_ROW)


def test_a_string_without_a_lone_pair_stereocentre_keeps_rdkits_reading():
    for smiles in ("CCO", "C[C@H](O)CC", "C[C@@H](O)c1ccccc1", "c1ccccc1", "[2H]C([2H])O"):
        assert Chem.MolToSmiles(standard_reading_mol(smiles)) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        assert standard_reading_inchikey(smiles) == _rdkit_key(smiles)
    assert standard_reading_mol("") is None and standard_reading_mol(None) is None
    assert standard_reading_mol("not a smiles") is None and standard_reading_inchikey("not a smiles") is None


# ---------------------------------------------------------------------------------------------
# The names
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["best-effort", "pin"])
def test_r102_is_named_with_the_descriptor_of_the_input(tier):
    res = _row(R102, tier)
    assert res.get("name") == R102_NAME, res.get("name")
    assert res.get("tier") == "pin_verified"
    assert _opsin_own_stdinchikey(res["name"]) == _rdkit_key(R102)
    assert namer._lone_pair_configuration_verified(res["name"], R102) is True


NAMED_AT_BEST_EFFORT = [
    pytest.param(R358, R358_NAME, id="r358"),
    pytest.param(R665, R665_NAME, id="r665"),
    pytest.param(R322, R322_NAME, id="r322"),
    pytest.param(R633, R633_NAME, id="r633"),
]


@pytest.mark.parametrize("smiles,name", NAMED_AT_BEST_EFFORT)
def test_the_other_phosphorus_rows_are_named_at_best_effort(smiles, name):
    res = _row(smiles, "best-effort")
    assert res.get("name") == name, res.get("name")
    assert res.get("tier") != "abstain"
    assert _opsin_own_stdinchikey(name) == standard_reading_inchikey(smiles)
    assert namer._lone_pair_configuration_verified(name, smiles) is True


@pytest.mark.parametrize("smiles,wrong", [
    # the names the earlier code, and the paper's run, gave: the other configuration at the P
    (R102, "(1S,2S)-1,2-diphenylphospholane"),
    (R322, R322_NAME.replace("[(2S,4R,5S)-4-cyclopentyl", "[(2R,4R,5S)-4-cyclopentyl")),
    (R633, R633_NAME.replace("[(1S,6S,8R)-6,9-dimethyl", "[(1S,6S,8S)-6,9-dimethyl")),
])
def test_the_name_of_the_other_configuration_is_not_the_name_of_the_input(smiles, wrong):
    assert _opsin_own_stdinchikey(wrong) != standard_reading_inchikey(smiles)
    assert namer._lone_pair_configuration_verified(wrong, smiles) is False


# ---------------------------------------------------------------------------------------------
# The string the engine names
# ---------------------------------------------------------------------------------------------


def _labels_by_input_atom(smiles: str):
    """CDK's labels of ``smiles`` as written, keyed by the atoms of RDKit's reading of it."""
    from orthonym.perception.centres_bridge import centres_label_batch
    mol = namer._lp_heavy_mol(smiles)
    written = centres_label_batch([smiles])[smiles]
    return {a.GetIdx(): written[a.GetIntProp("_lp_pos")] for a in mol.GetAtoms()
            if a.GetIntProp("_lp_pos") in written}


@pytest.mark.parametrize("smiles", [R102, N_RING_P, N_RING_P_OTHER, AZIRIDINE_FIRST, R665])
def test_a_string_rdkit_reads_alike_but_writes_otherwise_is_named_by_its_standard_spelling(smiles):
    from orthonym.perception.centres_bridge import centres_label_batch
    mol = namer._lp_heavy_mol(smiles)
    # RDKit's canonical SMILES of the input, read by CDK, gives another label at a lone-pair centre
    canonical = Chem.MolToSmiles(mol)
    order = [int(i) for i in mol.GetProp("_smilesAtomOutputOrder").strip("[]").split(",") if i.strip()]
    read = {order[pos - 1]: lab for pos, lab in centres_label_batch([canonical])[canonical].items()}
    as_written = _labels_by_input_atom(smiles)
    assert read != as_written
    spelled, spelled_order = namer._lone_pair_standard_spelling_and_order(smiles)
    assert spelled != smiles and spelled_order is not None
    # the string the engine names: CDK reads RDKit's canonical SMILES of it as it reads the input
    again = {spelled_order[pos - 1]: lab
             for pos, lab in centres_label_batch([spelled])[spelled].items()}
    assert again == as_written


@pytest.mark.parametrize("smiles", [P_RING_ROW, "C[C@@H](O)CC", "CCO", "C[S@@](=O)c1ccccc1"])
def test_a_string_the_engine_already_labels_as_written_is_kept(smiles):
    assert namer._lone_pair_standard_spelling(smiles) is smiles


@pytest.mark.parametrize("smiles,name", [
    # '[(propan-2-yl)oxy]' for '(1-methylethoxy)': "Retained names" (:27665), '(CH3)2CH-O- isopropoxy
    # 1-methylethoxy (propan-2-yl)oxy (preferred prefix)' (:27683). 'phenyl' precedes 'propan-2-yl' in the order
    # (:3436), but OPSIN's SMILES of that spelling puts a ring-closure digit on the P, which RDKit reads the other way
    # (``universal_substituent._prefix_order_fallback``): the name that reads back cites the P-substituent first. It is
    # a valid name, not the PIN; the mechanical '2-methyl-1-oxapropan-1-yl' it replaces is the worse spelling.
    # change-asserted-value: the quote; the read-back below; HEAD spells '1-methylethoxy'.
    (N_RING_P, "(2R,5S)-2-[(propan-2-yl)oxy]-3-phenyl-1,3-diaza-2-phosphabicyclo[3.3.0]octane"),
    (N_RING_P_OTHER, "(2S,5S)-2-[(propan-2-yl)oxy]-3-phenyl-1,3-diaza-2-phosphabicyclo[3.3.0]octane"),
    (AZIRIDINE_FIRST,
     "methyl (1S,2R)-1-{[5-(8,8-difluoro-7,9-dioxabicyclo[4.3.0]nona-1,3,5-trien-2-yl)-2-fluorophenyl]"
     "methyl}aziridine-2-carboxylate"),
])
def test_rows_that_were_limited_by_rdkits_canonical_smiles_are_named(smiles, name):
    # RDKit reads each of these as CDK does (the keys are OPSIN's own), so the check needs no code of
    # the engine; the earlier code named none of them.
    res = _row(smiles, "best-effort")
    assert res.get("name") == name, res.get("name")
    assert _opsin_own_stdinchikey(name) == _rdkit_key(smiles)
    assert namer._lone_pair_configuration_verified(name, smiles) is True
