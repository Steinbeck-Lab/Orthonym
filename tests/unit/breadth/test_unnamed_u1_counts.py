"""Unnamed-breadth lane U1 (plan 2026-10-15-unnamed-breadth): PubChem rows that main left unnamed
because a name spelled a multiplying count as DIGITS (``hexahexaconta-1,3,...,65-33yne``,
``...-21oxatetrahexacont-63-ene``), which OPSIN cannot read. Best-effort must now name each row whose
count OPSIN 2.9.0 can read, with a full read-back (full InChIKey + radical identity + protonation
site, a fresh OPSIN call); the others keep no name and never a digit multiplier.

Governing rules (the Blue Book Blue Book):
* 'Basic multiplicative prefixes' (:2790, Table 1.4 at:2794): "Basic multiplicative prefixes
  denote simple features and, in general, are the first choice among such prefixes to specify
  multiplicity". The terms above 20 are composed by 'Derivation of basic numerical terms'
  (:2811, text:2813): "composite numerical terms are formed systematically by citing the basic
  terms in the order opposite to that of the constituent digits in the arabic numbers. The composite
  terms are formed by direct joining of the basic terms, without hyphen(s). The letter 'i' in
  'icosa' is elided after a vowel." Examples (:2820-:2822): 21 henicosa, 22 docosa, 23 tricosa, 24
  tetracosa, 41 hentetraconta, 52 dopentaconta, 111 undecahecta.
* 'Numerical terms for compound or complex features' (:2824, text:2826): "Multiplicative
  prefixes for compound or complex entities... are formed by adding the ending 'kis' to the basic
  multiplicative prefix ending in 'a'". Example (:2832): 231 hentriacontadictakis.
* 'General methodology', (:16497): "The multiplying prefixes 'di', 'tri', etc.,
  are placed before endings denoting unsaturation... when the endings 'ene' and 'yne' are preceded
  by a multiplying prefix and a locant the letter 'a' is inserted. There is no elision of the final
  letter 'a' of a multiplying prefix before 'ene' or 'yne', for example, 'tetraene' and 'pentayne'."
* 'Elision of vowels', (c) (:7619): the terminal 'a' of a numerical multiplicative
  prefix is elided before a suffix beginning with 'a' or 'o' ('benzenehexol').
* 'Hydrocarbons' (:7984): the chain stem is a numerical term plus 'ane' (tricosane (PIN),
  heptacontane (PIN)).

PIN track: the four rows the PIN path names (``PIN_CASES``) carry the same chain, 'ene'/'yne' and
multiplier rules as the shorter chains the Blue Book prints with (PIN), and the book prints two
(PIN) names whose multiplying prefix is past 20: ':25799' 'pentacosafluoro' (25) and ':17098'
'tetracosahydro' (24). The other rows: none (best-effort only); the default tier must not claim a PIN
for them.

OPSIN 2.9.0 reads a simple multiplying prefix up to 50 ('pentaconta') and a 'kis' prefix up to 30
('triacontakis') (opsin-core ``multipliers.xml``); a name whose multiplier is larger is spelled right
and still cannot be read back, so such a row stays unnamed (``LIMIT_ROWS``).
"""
import ast
import pathlib
import re

import pytest

from orthonym.data.chain_names import multiplying_prefix, numerical_term
from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort


def _locants(*args):
    return ",".join(str(i) for i in range(*args))


# ---------------------------------------------------------------------------------------------
# 1. The one function: every count has a word, in the simple and the complex form.
# ---------------------------------------------------------------------------------------------

# (n, simple, complex, where the book prints it)
WORDS = [
    (2, "di", "bis", "Table 1.4 :2794; P-14.2.2 :2826 'bis' for 2"),
    (3, "tri", "tris", "Table 1.4; P-14.2.2 'tris' for 3"),
    (4, "tetra", "tetrakis", "P-14.2.2 'tetrakis'"),
    (20, "icosa", "icosakis", "Table 1.4 :2794 'icosa'"),
    (21, "henicosa", "henicosakis", "P-14.2.1.2 example :2820 'henicosa'"),
    (22, "docosa", "docosakis", "P-14.2.1.2 example :2820 'docosa'"),
    (24, "tetracosa", "tetracosakis", "P-14.2.1.2 :2820; '(PIN)' tetracosahydro :17098"),
    (25, "pentacosa", "pentacosakis", "'(PIN)' pentacosafluoro :25799"),
    (33, "tritriaconta", "tritriacontakis", "composed: 'tri' + 'triaconta' (Table 1.4 :2796)"),
    (43, "tritetraconta", "tritetracontakis", "composed: 'tri' + 'tetraconta' (Table 1.4 :2797)"),
    (50, "pentaconta", "pentacontakis", "Table 1.4 :2798 'pentaconta'"),
    (66, "hexahexaconta", "hexahexacontakis", "composed: 'hexa' + 'hexaconta'"),
    (87, "heptaoctaconta", "heptaoctacontakis", "composed: 'hepta' + 'octaconta'"),
    (100, "hecta", "hectakis", "Table 1.4 :2794 'hecta'"),
    (101, "henhecta", "henhectakis", "Table 1.4 :2794 '101 henhecta'"),
    (111, "undecahecta", "undecahectakis", "P-14.2.1.2 example :2822 'undecahecta'"),
    (120, "icosahecta", "icosahectakis", "composed: 'icosa' + 'hecta' ('i' kept: no vowel before)"),
    (231, "hentriacontadicta", "hentriacontadictakis", "P-14.2.2 example :2832"),
    (1001, "henkilia", "henkiliakis", "Table 1.4 :2794 '1001 henkilia'"),
]


@pytest.mark.parametrize("n,simple,complex_,source", WORDS, ids=[str(w[0]) for w in WORDS])
def test_multiplying_prefix_spells_the_count(n, simple, complex_, source):
    assert multiplying_prefix(n) == simple, source
    assert multiplying_prefix(n, "simple") == simple, source
    assert multiplying_prefix(n, "complex") == complex_, source


def test_no_count_is_ever_spelled_as_digits():
    """Every count from 2 to 9999 gives a word of letters only, the complex form is the simple form
    plus 'kis' above 20, and above 20 the simple form is ``numerical_term``."""
    for n in range(2, 10000):
        simple, complex_ = multiplying_prefix(n), multiplying_prefix(n, "complex")
        assert simple.isalpha() and complex_.isalpha(), n
        if n > 20:
            assert simple == numerical_term(n) and complex_ == simple + "kis", n


@pytest.mark.parametrize("n", [0, 1, -3, 10000])
def test_a_count_with_no_word_raises_and_never_returns_digits(n):
    with pytest.raises(ValueError):
        multiplying_prefix(n)
    with pytest.raises(ValueError):
        multiplying_prefix(n, "complex")


def test_a_producer_table_wins_for_the_counts_it_has_and_the_rest_is_composed():
    table = {1: "", 2: "di", 3: "tri"}
    assert multiplying_prefix(1, table=table) == ""
    assert multiplying_prefix(2, table=table) == "di"
    assert multiplying_prefix(21, table=table) == "henicosa"
    assert multiplying_prefix(30, "complex", table={30: "own"}) == "own"
    with pytest.raises(ValueError):
        multiplying_prefix(1)
    with pytest.raises(ValueError):
        multiplying_prefix(2, "neither")


def test_the_old_entry_points_agree_with_the_one_function():
    from orthonym.assembly.composer import get_multiplier
    from orthonym.assembly.naming_utils import (
        get_multiplier_prefix, get_suffix_multiplier_prefix, simple_multiplier_word)
    for n in (2, 20, 21, 22, 33, 50, 66, 111, 231):
        assert simple_multiplier_word(n) == multiplying_prefix(n)
        assert get_multiplier(n) == multiplying_prefix(n)
        assert get_multiplier(n, True) == multiplying_prefix(n, "complex")
        # (a): a substituted prefix takes the derived ('kis') multiplier, a simple one 'di'...
        assert get_multiplier_prefix(n, "chloromethyl") == multiplying_prefix(n, "complex")
        assert get_multiplier_prefix(n, "methyl") == multiplying_prefix(n)
        # (b): only 'dithioic acid' and its kind take the derived multiplier
        assert get_suffix_multiplier_prefix(n, "dithioic acid") == multiplying_prefix(n, "complex")
        assert get_suffix_multiplier_prefix(n, "ol") == multiplying_prefix(n)
    assert get_multiplier(1) == "" and simple_multiplier_word(1) is None
    assert simple_multiplier_word(10000) is None


# ---------------------------------------------------------------------------------------------
# 2. The producers that used to spell digits: each form, at a count past 20.
# ---------------------------------------------------------------------------------------------

def test_hydrocarbon_endings_take_the_multiplier_word():
    """ (:16497): 'a' + locants + multiplier + 'ene'/'yne', the multiplier's 'a' kept."""
    from orthonym.assembly.composition_primitives import _build_hydrocarbon_name
    assert (_build_hydrocarbon_name("hexahexacont", [], list(range(1, 66, 2)))
            == f"hexahexaconta-{_locants(1, 66, 2)}-tritriacontayne")
    assert (_build_hydrocarbon_name("nonatetracont", list(range(1, 49)), [])
            == f"nonatetraconta-{_locants(1, 49)}-octatetracontaene")
    assert (_build_hydrocarbon_name("docos", list(range(1, 22)), [])
            == f"docosa-{_locants(1, 22)}-henicosaene")


def test_skeletal_replacement_prefixes_take_the_multiplier_word():
    """ (:23348) skeletal replacement ('a') names: '3,6,9,12-tetraoxatetradecane', and past
    20 the term is composed."""
    from orthonym.rules.skeletal_replacement import _build_replacement_name
    oxa = [(3 * k, "O") for k in range(1, 22)]                      # 21 O at 3,6,...,63
    assert (_build_replacement_name(64, oxa)
            == f"{','.join(str(p) for p, _ in oxa)}-henicosaoxatetrahexacontane")
    assert (_build_replacement_name(64, oxa, ene_locants=[1])
            == f"{','.join(str(p) for p, _ in oxa)}-henicosaoxatetrahexacont-1-ene")
    aza = [(2 * k, "N") for k in range(1, 23)]                      # 22 N at 2,4,...,44
    assert (_build_replacement_name(45, aza)
            == f"{','.join(str(p) for p, _ in aza)}-docosaazapentatetracontane")


def test_von_baeyer_replacement_prefix_past_twenty_is_a_word():
    """A 22-Si bicyclo[10.9.1] (MAX_CAGE_ATOMS is 40): the prefix is 'docosasila', never '22sila'."""
    from rdkit import Chem
    from orthonym.rules.ring_replacement import build_replacement_prefix
    mol = Chem.MolFromSmiles("[SiH2]1" + "[SiH2]" * 9 + "[SiH]2" + "[SiH2]" * 9 + "[SiH]1[SiH2]2")
    ring = [a.GetIdx() for a in mol.GetAtoms()]
    res = build_replacement_prefix(mol, {i: i + 1 for i in ring}, ring)
    assert res.prefix == f"{_locants(1, 23)}-docosasila" and res.unexpressed == ()


# An empty-string default on a multiplier table that is right: the reason is the entry's value.
EMPTY_DEFAULT_ALLOWED = {
    ('rules/inorganic_acids.py', 'multiplier_table(4).get(n_count, "")'):
        "boron has at most four neighbours, so n_count is 2..4 and the table is complete",
    ('rules/phosphorus.py', "_PHOSPHOXANE_MULT.get(n, '')"):
        "the next statement refuses (returns None) when the table has no entry: the empty "
        "string is never emitted",
}


def test_no_multiplier_lookup_falls_back_to_digits_or_to_nothing():
    """Static guard over the package: no ``<multiplier table>.get(count, <digits>)`` and no
    ``<multiplier table>.get(count, '')`` remains, except the two sites in ``EMPTY_DEFAULT_ALLOWED``
    (each with its reason). A digit multiplier is unreadable by OPSIN; an empty one drops the count
    from the name. Every such site goes through ``multiplying_prefix``."""
    import orthonym
    root = pathlib.Path(orthonym.__file__).parent
    receiver = re.compile(r"mult|MULT|STOICHIOMETRIC_PREFIXES")
    found, allowed_seen = [], set()
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(ast.parse(text)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and len(node.args) == 2):
                continue
            recv = ast.get_source_segment(text, node.func.value) or ""
            if not receiver.search(recv):
                continue
            default = node.args[1]
            digits = (isinstance(default, ast.Call) and getattr(default.func, "id", "") == "str") or (
                isinstance(default, ast.JoinedStr)
                and any(isinstance(v, ast.FormattedValue) for v in default.values))
            empty = isinstance(default, ast.Constant) and default.value == ""
            if not (digits or empty):
                continue
            segment = ast.get_source_segment(text, node)
            if empty and (rel, segment) in EMPTY_DEFAULT_ALLOWED:
                allowed_seen.add((rel, segment))
                continue
            found.append(f"{rel}:{node.lineno} {segment}")
    assert not found, "digit or empty fallback in a multiplier lookup:\n" + "\n".join(found)
    assert allowed_seen == set(EMPTY_DEFAULT_ALLOWED), "a stale allow-list entry"


# ---------------------------------------------------------------------------------------------
# 3. Whole molecules: full read-back at best-effort.
# ---------------------------------------------------------------------------------------------

# Synthetic rows: a count past 20 in each producer family. Each reads back to the full InChIKey.
SYNTHETIC = [
    ("polyether, 22 oxa (P-51.4.1.1, 'docosaoxa')", "CO" + "CCO" * 21 + "C"),
    ("polyol, 21 hydroxy ('henicosol', P-16.7.1(c))", "OC" + "C(O)" * 19 + "CO"),
    ("cumulene, 21 double bonds ('henicosaene')", "C" + "=C" * 21),
    ("polyyne, 21 triple bonds on 42 C ('henicosayne')", "C#C" * 21),
    ("perfluorodocosane, 46 fluoro ('hexatetracontafluoro')", "FC(F)(F)" + "C(F)(F)" * 20 + "C(F)(F)F"),
    ("polyketone, 21 oxo ('henicosaoxo')", "C(=O)" + "CCCC(=O)" * 21 + "C"),
]


@pytest.mark.opsin_gate  # what SHIPS: the engine's own validity gate on, as in production
@pytest.mark.parametrize("what,smiles", SYNTHETIC, ids=[s[0].split(",")[0] for s in SYNTHETIC])
def test_synthetic_long_chain_has_a_word_multiplier_and_reads_back(what, smiles):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles, what=f"{what}: ")
    assert not re.search(DIGIT_MULTIPLIER, name), name
    if what.startswith("polyether"):
        # "Skeletal replacement ('a') nomenclature in acyclic chains" (:23346),
        # (:23348): "Skeletal replacement ('a') nomenclature rather than substitutive or multiplicative
        # names must be used to generate preferred IUPAC names for acyclic structures when four or more
        # heterounits are present in a unbranched chain containing at least one carbon atom and when
        # none of the heteroatoms constitute all or part of the principal characteristic group of the
        # compound." So the 'a' name, not the substitutive 'methoxy...ethoxy' one that was the fallback
        # while the 'a' name spelled its count as digits.
        assert "docosaoxa" in name, name


# The unnamed-panel rows of the strategy study (CIDs of the snapshot; main gives none a name).
ROWS = [
    ("21043395", "C#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#C"),
    ("57774013", "C=CCOOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C=C"),
    ("57882001", "C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C"),
    ("59741388", "C#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#N"),
    ("124094021", "CCOS(=S)(=S)SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS"),
    ("60028779", "C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C"),
]

# Rows whose multiplier is past what OPSIN 2.9.0 reads (simple > 50): the name is spelled right
# ('heptaheptaconta' 77, 'tetrapentaconta' 54, 'nonapentaconta' 59, 'heptahexaconta' 67,
# 'tripentaconta' 53) and cannot be read back, so no name ships (0-wrong: no unverified name).
LIMIT_ROWS = [
    ("123257338", "CC(C)C(=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C)C(C)C"),
    ("165366016", "C(CCCCCCC(CC(C(C(C(C(C(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)I)CCCCCC(CC(C(C(C(C(C(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)I"),
    ("17970089", "C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F"),
    ("89829763", "CCOS(=S)(=S)SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS"),
    ("177268397", "CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#CC#C[CH2-]"),
]

# A digit directly before an 'ene'/'yne'/replacement/'kis'/substituent word is a digit multiplier
# ('33yne', '21oxa', '22kis', '54fluoro'); locants are always followed by a hyphen or a comma.
DIGIT_MULTIPLIER = r"\d(?:ene|yne|en\b|yn\b|oxa|aza|thia|sila|fluoro|chloro|oxo|hydroxy|kis)"

PIN_ROWS = {"21043395", "57882001", "59741388", "60028779"}  # each with its cited rule: PIN_CASES


@pytest.mark.opsin_gate
@pytest.mark.parametrize("cid,smiles", ROWS, ids=[c for c, _ in ROWS])
def test_best_effort_names_row_with_full_read_back(cid, smiles):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles, what=f"CID {cid}: ")
    assert not re.search(DIGIT_MULTIPLIER, name), name


def _pairs(*args):
    return ",".join(f"{i},{i}" for i in range(*args))


# Names OPSIN 2.9.0 cannot read (no count word from 51 up) that ship because the reconstructor reads their
# structure from the name text. Each is built here from the rules, not copied from the engine: count words
# (:2794; 'heptaheptaconta' 77, 'tetrapentaconta' 54, 'nonapentaconta' 59), chain stems
# (:7984, a numerical term followed by 'ane'), 'a' kept before 'ene' (:16497).
RECONSTRUCTED = {
    # 80 C chain from the CH2= end, 77 cumulated double bonds; C78 carries the second propan-2-yl
    "123257338": f"79-methyl-78-(propan-2-yl)octaconta-{_locants(1, 78)}-heptaheptacontaene",
    # 42 C, iodine at 15 and 28, fluorine on every other hydrogen position: 3 + 2*12 + 2*12 + 3 = 54
    "165366016": f"15,28-diiodo-1,1,1,{_pairs(2, 14)},{_pairs(30, 42)},42,42,42-tetrapentacontafluorodotetracontane",
    # 29 C, one hydrogen left on C29: 3 + 2*28 = 59 fluorine
    "17970089": f"1,1,1,{_pairs(2, 30)}-nonapentacontafluorononacosane",
}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("cid,smiles", LIMIT_ROWS, ids=[c for c, _ in LIMIT_ROWS])
def test_row_past_the_opsin_multiplier_range_ships_no_unreadable_name(cid, smiles):
    """Either no name, or a name that reads back exactly (a newer OPSIN may read it), or the rule-built name
    that the reconstructor reads back; never a name that spells the count as digits."""
    from orthonym.errors import is_failure_name
    res = name_best_effort(smiles)
    name = res.get("name")
    if name and not is_failure_name(name) and res.get("source") != "abstain":
        assert not re.search(DIGIT_MULTIPLIER, name), name
        if res.get("verified") == "reconstructor":
            assert name == RECONSTRUCTED.get(cid), f"CID {cid}: {name}"
            assert res.get("tier") == "systematic_verified", res.get("tier")
        else:
            assert_full_rt(name, smiles, what=f"CID {cid}: ")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("cid,smiles", ROWS + LIMIT_ROWS, ids=[c for c, _ in ROWS + LIMIT_ROWS])
def test_default_tier_never_claims_a_pin_without_a_citation(cid, smiles):
    res = default_tier_row(smiles)
    assert not res.get("is_pin") or cid in PIN_ROWS, f"CID {cid}: PIN claimed without a cited rule"


# PIN track. Each name is built here from the rules, not copied from the engine: chain stem
# (:7984, 'a numerical term (see Table 1.4) followed by the ending ane'), multiplier
# (:2811), 'a' + locants + multiplier + 'yne'/'ene' with no elision (:16497), the (PIN)
# spellings past 20 ':25799' pentacosafluoro and ':17098' tetracosahydro.
PIN_CASES = [
    # 66 C stem 'hexa'+'hexaconta'; 33 triple bonds at 1,3,...,65: 'tri'+'triaconta' + 'yne'
    ("21043395", ROWS[0][1],
     f"hexahexaconta-{_locants(1, 66, 2)}-tritriacontayne"),
    # 49 C ('nona'+'tetraconta'); 48 cumulated double bonds: 'octa'+'tetraconta' + 'ene'
    ("57882001", ROWS[2][1],
     f"nonatetraconta-{_locants(1, 49)}-octatetracontaene"),
    # 87 C ('hepta'+'octaconta'); 43 triple bonds at 2,4,...,86 ('tri'+'tetraconta'), nitrile suffix
    ("59741388", ROWS[3][1],
     f"heptaoctaconta-{_locants(2, 87, 2)}-tritetracontaynenitrile"),
    # 50 C ('pentaconta'); 24 double bonds at 2,4,...,48 ('tetracosa' + 'ene'), all E
    ("60028779", ROWS[5][1],
     f"({','.join(f'{i}E' for i in range(2, 49, 2))})-pentaconta-{_locants(2, 49, 2)}-tetracosaene"),
]


# Acyclic 'a'-chains with more than 20 oxygens. "Skeletal replacement ('a') nomenclature in
# acyclic chains" (:23346), (:23348): "Skeletal replacement ('a') nomenclature rather than
# substitutive or multiplicative names must be used to generate preferred IUPAC names for acyclic
# structures when four or more heterounits are present in a unbranched chain containing at least one
# carbon atom and when none of the heteroatoms constitute all or part of the principal characteristic
# group of the compound." Printed (PIN) examples of the pattern: ':23367' 3,6,9,12-tetraoxatetradecane-
# dioic acid [PIN, a skeletal replacement ('a') name] and ':25317' 3,6,9,12-tetraoxapentadecan-15-oic acid
# (PIN). Before the multiplier past 20 had a spelled word these names were not built and the default
# tier gave the substitutive '1-methoxy-2-[2-(...' name (pin_verified), which that rule excludes. The
# names are built here from the rule: the 'a' term 'docosa' (22) / 'tetracosa' (24) from
# (:2811), oxygen at 2,5,...,3k-1 of the CH3-O-[CH2CH2-O]n chain, chain stem 'hexahexaconta' (66) and
# 'triheptaconta' (73),:7984), suffix 'an-73-oic acid' as in ':25317'.
GLYCOL_CASES = [
    ("22 O, 66-atom dimethyl ether", "CO" + "CCO" * 21 + "C",
     f"{_locants(2, 66, 3)}-docosaoxahexahexacontane"),
    ("24 O, 73-atom carboxylic acid", "CO" + "CCO" * 23 + "CC(=O)O",
     f"{_locants(2, 72, 3)}-tetracosaoxatriheptacontan-73-oic acid"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("cid,smiles,pin", PIN_CASES, ids=[c[0] for c in PIN_CASES])
def test_pin_tier_gives_the_cited_pin(cid, smiles, pin):
    res = default_tier_row(smiles)
    assert res.get("name") == pin and res.get("is_pin"), f"CID {cid}: {res.get('name')!r}"
    assert_full_rt(pin, smiles, what=f"CID {cid}: ")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("what,smiles,pin", GLYCOL_CASES, ids=[c[0].split(",")[0] for c in GLYCOL_CASES])
def test_default_tier_gives_the_acyclic_a_chain_pin(what, smiles, pin):
    res = default_tier_row(smiles)
    assert res.get("name") == pin and res.get("is_pin"), f"{what}: {res.get('name')!r}"
    assert_full_rt(pin, smiles, what=f"{what}: ")
