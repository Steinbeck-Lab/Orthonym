"""F-spell-oxy — alkoxy substituent-prefix morphology.

The naive `name[:-2] + 'oxy'` idiom over-contracted a COMPLEX / ring / C5+ R'
group ('oxan-2-yl' -> 'oxan-2-oxy', 'cyclohexyl' -> 'cyclohexoxy', 'pentyl' ->
'pentoxy') and dropped the inner enclosing marks on a locant-bearing heteroaryl
('pyridin-2-yl' -> 'pyridin-2-yloxy'). Every PIN-default emitter now routes
through the BB-verbatim `composed_alkoxy_prefix`:

  * retained set contracts (methoxy/ethoxy/propoxy/butoxy, and their
    substituted primary members: chloromethoxy, methoxymethoxy, 2-methoxyethoxy,
    2-methylpropoxy) — the Blue Book "fully substitutable", the Blue Book chloromethoxy PIN;
  * a locant-bearing / ring free valence keeps the whole '-yl' inside marks
    ('(oxan-2-yl)oxy', '(pyridin-2-yl)oxy') — the Blue Book "(pyridin-2-yl)oxy
    (preferred prefix)", the Blue Book "(butan-2-yl)oxy";
  * C5+ concatenates ('pentyloxy'); a ring with no '-N-yl' locant concatenates
    without inner marks ('cyclohexyloxy', the Blue Book).

Every expected name is OPSIN RT-EXACT (verified 2026-08-08).
"""
import pytest
from orthonym.namer import Orthonym
from orthonym.assembly.substituent_enumerator import (
    composed_alkoxy_prefix,
    alkoxy_prefix_from_substituent,
)

pytestmark = pytest.mark.unit

# Gate-off proves the raw producer path (not the OPSIN suppression gate).
RAW = Orthonym(_disable_opsin_validity_gate=True)


# --- The boundary primitive, exercised directly (fast, no JVM) --------------
@pytest.mark.parametrize("token,expected", [
    # retained contracted set + substituted primary members (PRESERVE)
    ("methyl", "methoxy"), ("ethyl", "ethoxy"),
    ("propyl", "propoxy"), ("butyl", "butoxy"),
    ("tert-butyl", "tert-butoxy"), ("phenyl", "phenoxy"),
    ("chloromethyl", "chloromethoxy"),
    ("methoxymethyl", "methoxymethoxy"),
    ("2-methoxyethyl", "2-methoxyethoxy"),
    ("2-methylpropyl", "2-methylpropoxy"),
    # secondary / ring / locant-bearing -> WRAP
    ("propan-2-yl", "(propan-2-yl)oxy"),
    ("butan-2-yl", "(butan-2-yl)oxy"),
    ("oxan-2-yl", "(oxan-2-yl)oxy"),
    ("oxolan-2-yl", "(oxolan-2-yl)oxy"),
    ("pyridin-2-yl", "(pyridin-2-yl)oxy"),
    ("naphthalen-1-yl", "(naphthalen-1-yl)oxy"),
    # C5+ / bare ring -> concatenate (no inner marks)
    ("pentyl", "pentyloxy"),
    ("cyclohexyl", "cyclohexyloxy"),
    # cycloalkyls whose name ENDS in a retained stem must NOT contract
    # (a review MINOR-1: 'cyclopropoxy'/'cyclobutoxy' occur 0x in the BB; concatenate
    # like 'cyclohexyloxy' the Blue Book). A genuine substituent prefix before the stem
    # still contracts ('cyclopropylmethyl' -> 'cyclopropylmethoxy').
    ("cyclopropyl", "cyclopropyloxy"),
    ("cyclobutyl", "cyclobutyloxy"),
    ("cyclopentyl", "cyclopentyloxy"),
    ("cyclopropylmethyl", "cyclopropylmethoxy"),
])
def test_composed_alkoxy_prefix_boundary(token, expected):
    assert composed_alkoxy_prefix(token) == expected


# The emitter-facing wrapper ALSO contracts a DECORATED phenyl to the retained,
# fully-substitutable 'phenoxy' / BB 17796) — composed_alkoxy_prefix
# declines '...phenyl' (biphenyl conservatism), so this must be preserved when
# routing general substituent names through the alkoxy emitter (F-spell-oxy).
@pytest.mark.parametrize("token,expected", [
    ("4-methylphenyl", "4-methylphenoxy"),
    ("4-chlorophenyl", "4-chlorophenoxy"),
    ("4-methoxyphenyl", "4-methoxyphenoxy"),
    ("phenyl", "phenoxy"),
    # a locant-bearing biphenyl is NOT a bare '...phenyl' -> stays bracketed
    ("[1,1'-biphenyl]-4-yl", "([1,1'-biphenyl]-4-yl)oxy"),
    ("oxan-2-yl", "(oxan-2-yl)oxy"),
    ("pentyl", "pentyloxy"),
    ("methoxymethyl", "methoxymethoxy"),
])
def test_alkoxy_prefix_from_substituent_phenyl_contraction(token, expected):
    assert alkoxy_prefix_from_substituent(token) == expected


# --- Class A: over-contraction fixed (whole-name) ---------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)CCOC1CCCCO1", "3-[(oxan-2-yl)oxy]propanoic acid"),
    ("OCCOC1CCCCO1",       "2-[(oxan-2-yl)oxy]ethan-1-ol"),
    # compound (concatenated) prefix takes enclosing marks on benzene
    #; the Blue Book "(cyclohexyloxy)benzene (PIN)").
    ("c1ccccc1OC1CCCCC1",  "(cyclohexyloxy)benzene"),
    ("c1ccccc1OCCCCC",     "(pentyloxy)benzene"),
])
def test_class_a_overcontraction_fixed(smiles, expected):
    assert RAW.name(smiles) == expected


# --- Class B: inner enclosing marks on locant-bearing heteroaryl/fused -------
@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)COc1ccccn1",       "[(pyridin-2-yl)oxy]acetic acid"),
    ("OC(=O)COc1cccc2ccccc12", "[(naphthalen-1-yl)oxy]acetic acid"),
    ("OC(=O)COc1ccc2ccccc2c1", "[(naphthalen-2-yl)oxy]acetic acid"),
    ("OC(=O)COc1ccc2ccccc2n1", "[(quinolin-2-yl)oxy]acetic acid"),
    ("OC(=O)CCOc1ccccn1",      "3-[(pyridin-2-yl)oxy]propanoic acid"),
])
def test_class_b_heteroaryloxy_enclosed(smiles, expected):
    # A locant-bearing citation escalates the outer marks to [...] because the
    # inner '(pyridin-2-yl)' already carries '(...)'. All RT-EXACT.
    assert RAW.name(smiles) == expected


# --- Preserve: retained contractions must NOT flip to wrapped -----------------
@pytest.mark.parametrize("smiles,must_contain", [
    ("OC(=O)CCOCOC",  "methoxymethoxy"),      # NOT (methoxymethyl)oxy
    ("OC(=O)CCOCCOC", "2-methoxyethoxy"),
    ("OC(=O)CCOCCl",  "chloromethoxy"),
    ("OC(=O)CCOCC(C)C", "2-methylpropoxy"),
])
def test_retained_contractions_preserved(smiles, must_contain):
    assert must_contain in RAW.name(smiles)


# --- Protect: phenoxy contraction unchanged (w8p2 gold protect rows) ---------
@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)COc1ccccc1",      "phenoxyacetic acid"),
    ("OC(=O)COc1ccc(Cl)cc1",  "(4-chlorophenoxy)acetic acid"),
])
def test_phenoxy_contraction_protected(smiles, expected):
    assert RAW.name(smiles) == expected
