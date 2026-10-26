"""The hydrogens the standard InChIKey cannot see (``validation/key_blind_hydrogens.py``).

The standard InChI drops the hydrogens of a hypervalent halogen: ``C[IH2]`` and ``CI`` are one InChI string,
``C[IH2+]C`` and ``C[I+]C`` another. A gate that accepts a name on full-key equality alone then accepts
'2-(4-iodocyclohexyl)-1-methylpyrrolidine' for the lambda3-iodane ``CN1CCCC1C1CCC(CC1)[IH2]`` (it shipped as a
preferred name) and '3lambda3-iodaspiro[5.5]undecan-3-ium' for the spiro iodonium. The rule is one more answer of the
verdict every full-key gate already asks (``radical_identity_verdict``): the same number of hydrogen atoms on both
sides, asked when either structure holds a halogen of bonding number two or more.
"""
import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

from orthonym.validation.key_blind_hydrogens import hydrogen_atom_count, key_blind_hydrogen_verdict
from orthonym.validation.radical_identity import radical_identity_verdict

RDLogger.DisableLog("rdApp.*")

L = "λ"


def _key(smiles):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


# ---------------------------------------------------------------------------------------------
# 1. The class is measured: wherever the key is blind to a hydrogen, the verdict says mismatch.
# ---------------------------------------------------------------------------------------------

def _blind_pairs():
    """Every pair of structures RDKit reads that differ in the number of hydrogens and share a standard InChIKey,
    over the elements He to Rn, zero to seven hydrogens on the atom, the charges 0, +1 and -1 and five drawings."""
    pt = Chem.GetPeriodicTable()
    templates = ["C[{x}]C", "C[{x}]", "[{x}]", "C1CC[{x}]CC1", "C[{x}](C)C"]
    pairs = []
    for z in range(2, 87):
        sym = pt.GetElementSymbol(z)
        for template in templates:
            for charge in ("", "+", "-"):
                read = {}
                for n in range(8):
                    h = "" if n == 0 else ("H" if n == 1 else f"H{n}")
                    smiles = template.replace("{x}", f"{sym}{h}{charge}")
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    try:
                        key = inchi.MolToInchiKey(mol)
                    except Exception:  # noqa: BLE001 - no key, nothing to be blind with
                        key = None
                    if key:
                        read[n] = (smiles, key, hydrogen_atom_count(mol))
                ns = sorted(read)
                for i in range(len(ns)):
                    for j in range(i + 1, len(ns)):
                        a, b = read[ns[i]], read[ns[j]]
                        if a[1] == b[1] and a[2] != b[2]:
                            pairs.append((sym, a[0], b[0]))
    return pairs


def test_every_hydrogen_the_key_cannot_see_is_a_mismatch():
    """The measurement behind the rule, run again: an element the InChI library comes to treat the way it treats
    chlorine, bromine and iodine fails here until ``key_blind_hydrogens`` covers it."""
    pairs = _blind_pairs()
    assert {sym for sym, _a, _b in pairs} <= {"Cl", "Br", "I"}, sorted({s for s, _a, _b in pairs})
    assert pairs, "the measurement found no blind pair: it would hold for any rule"
    for sym, a, b in pairs:
        assert key_blind_hydrogen_verdict(a, b) == "mismatch", (sym, a, b)
        assert key_blind_hydrogen_verdict(b, a) == "mismatch", (sym, b, a)
        assert radical_identity_verdict(a, b) == "mismatch", (sym, a, b)


# ---------------------------------------------------------------------------------------------
# 2. The verdict.
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("input_smiles,parsed", [
    ("C[IH2]", "CI"),                                   # methyl-lambda3-iodane vs iodomethane
    ("CN1CCCC1C1CCC(CC1)[IH2]", "CN1CCCC1C1CCC(CC1)I"),  # the shipped wrong name's structure
    ("C1CCC2(CC1)CC[I+]CC2", "C1C[IH2+]CCC12CCCCC2"),   # the spiro iodonium and its hydride-plus-ium
    ("C[I+]C", "C[IH2+]C"),
])
def test_a_name_that_adds_or_loses_a_hydrogen_of_a_hypervalent_halogen_is_a_mismatch(input_smiles, parsed):
    assert _key(input_smiles) == _key(parsed), "the premise: the standard key is the same"
    assert key_blind_hydrogen_verdict(input_smiles, parsed) == "mismatch"
    assert key_blind_hydrogen_verdict(parsed, input_smiles) == "mismatch"
    # every full-key gate asks the radical verdict: the answer reaches them through it
    assert radical_identity_verdict(input_smiles, parsed) == "mismatch"


@pytest.mark.parametrize("input_smiles,parsed", [
    ("C[IH2]", "C[IH2]"),
    ("C1CC[IH]CC1", "C1CC[IH]CC1"),
    ("C1=NI=C1", "I1=NC=C1"),                            # two Kekule forms of the mancude ring
    ("OI(O)c1ccccc1", "OI(O)c1ccccc1"),
    ("CI(Cl)Cl", "CI(Cl)Cl"),
    ("C1CC[I-]CC1", "C1CC[I-]CC1"),
    ("[H+].[I-]", "I"),                                  # a halide drawn with its proton or without
    ("C[N+](C)(C)C.[I-]", "C[N+](C)(C)C.[I-]"),
])
def test_same_hydrogens_are_ok(input_smiles, parsed):
    assert key_blind_hydrogen_verdict(input_smiles, parsed) in ("ok", "n/a")
    assert radical_identity_verdict(input_smiles, parsed) != "mismatch"


@pytest.mark.parametrize("input_smiles,parsed", [
    ("CCO", "CCO"), ("CC(=O)O", "CC(=O)[O-]"), ("CI", "CI"), ("c1ccccc1Cl", "c1ccccc1Cl"),
    ("C1CC[SH2]CC1", "C1CCSCC1"),                       # sulfur: the key sees its hydrogens, so not this rule's
    ("CCO", "CC"),                                      # a different molecule is the key's job
])
def test_no_hypervalent_halogen_is_not_this_rules_question(input_smiles, parsed):
    assert key_blind_hydrogen_verdict(input_smiles, parsed) == "n/a"


def test_an_unreadable_parse_of_a_hypervalent_halogen_input_fails_closed():
    assert key_blind_hydrogen_verdict("C[IH2]", "not a smiles") == "mismatch"
    assert key_blind_hydrogen_verdict("CCO", "not a smiles") == "n/a"
    assert key_blind_hydrogen_verdict("not a smiles", "CCO") == "n/a"


# ---------------------------------------------------------------------------------------------
# 3. The gates: the acceptance of a name that parses to the other structure.
# ---------------------------------------------------------------------------------------------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,wrong_name", [
    ("CN1CCCC1C1CCC(CC1)[IH2]", "2-(4-iodocyclohexyl)-1-methylpyrrolidine"),
    ("C1CCC2(CC1)CC[I+]CC2", f"3{L}3-iodaspiro[5.5]undecan-3-ium"),
])
def test_the_read_back_does_not_accept_the_name_of_the_other_structure(smiles, wrong_name):
    """OPSIN reads each name, the key of what it reads is the input's, and the name is wrong."""
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    from orthonym.validation.reconstruct import verify_or_none
    check = opsin_roundtrip_check(smiles, wrong_name)
    assert check["inchi_match"] is True and check["passed"] is False, check
    assert verify_or_none(wrong_name, smiles) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("emit_tier", ["pin", "best-effort"])
def test_the_defect_row_is_not_named_with_the_hydrogens_dropped(emit_tier):
    """``CN1CCCC1C1CCC(CC1)[IH2]`` was named '2-(4-iodocyclohexyl)-1-methylpyrrolidine' at both tiers, at the PIN
    tier as pin_verified. Whatever it is named now reads back as the same structure, hydrogens and all."""
    from orthonym import Orthonym
    from orthonym.tiers import emit_tier_flags
    from tests.support.rt_assert import _independent_parse
    smiles = "CN1CCCC1C1CCC(CC1)[IH2]"
    res = Orthonym(style="pin", **emit_tier_flags(emit_tier)).name_tiered(smiles)
    assert res.get("name") != "2-(4-iodocyclohexyl)-1-methylpyrrolidine", res
    if res.get("name") and res.get("tier") != "abstain":
        back = _independent_parse(res["name"], smiles)
        assert hydrogen_atom_count(Chem.MolFromSmiles(back)) == hydrogen_atom_count(Chem.MolFromSmiles(smiles)), res


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("C[IH2]", f"methyl-{L}3-iodane"),
    ("CI", "iodomethane"),
    ("C1CC[IH]CC1", f"1{L}3-iodinane"),
    ("OI(O)c1ccccc1", f"dihydroxy(phenyl)-{L}3-iodane"),
    ("C1=NI=C1", f"1{L}3,2-iodazete"),                  # the other Kekule form of OPSIN's reading: same structure
])
def test_controls_that_name_still_name(smiles, name):
    from tests.support.rt_assert import name_best_effort
    res = name_best_effort(smiles)
    assert res.get("name") == name, res
