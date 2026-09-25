"""No emission path may splice a failure sentinel ('unknown...') or a bare,
space-separated list of fragment names into a name (Review Focus 3; plan
`docs/the workflow tooling/plans/2026-09-24-preexisting-test-failures.md` Task 3).

A failure is only ever shipped WHOLE -- as one of the descriptive-fallback
strings (`errors._DESCRIPTIVE_FALLBACK_NAMES`) -- never welded into a real name.
The two sentinel molecules below used to build ``(2E)-2-methyl-5-unknownpent-2-
enoic acid`` and ``(3R)-3,7-dimethyl-9-unknownnona-1,6-dien-3-ol`` at the PIN
tier: `composer._generate_ring_substituent_prefixes` wrote the literal
``'unknown'`` as the prefix of a decorated saturated ring it could not name (trace
evidence and the site: internal notes, " producer
sites"). With the OPSIN validity gate off (this suite's default, and the
``ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE`` hatch) `Orthonym.name` and
`name_tiered` shipped that string.

Breadth never drops (plan Global Constraints): every molecule here must get an
OPSIN full-InChIKey-exact name at the best-effort tier.
"""
import pytest
from rdkit import Chem

from orthonym.cli import _emit_tier_flags
from orthonym.errors import _DESCRIPTIVE_FALLBACK_NAMES, is_refusal_sentinel
from orthonym.namer import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse
from tests.support.jars import jar_or_none

pytestmark = pytest.mark.skipif(
    jar_or_none() is None,
    reason="the round-trip assertions need the OPSIN jar",
)

# CHEBI:131506 (tests/integration/test_handler_stereo_roundtrip.py): used to
# emit '(2E)-2-methyl-5-unknownpent-2-enoic acid'.
CHEBI_131506 = "C=C1CC[C@H]2C[C@@H]1[C@@]2(C)CC/C=C(\\C)C(=O)O"
# Canary call 171 (tests/unit/rules/test_opsin_format_compliance.py): used to
# emit '(3R)-3,7-dimethyl-9-unknownnona-1,6-dien-3-ol'.
CANARY_171 = "C=C[C@](C)(O)CCC=C(C)CCC1OC(C)(C)OC1(C)C"
# GPI mannoside (tests/unit/decomposition/test_no_partial_assembly.py): its
# mixed decomposition used to space-join 4 fragment names into one string.
GPI_MANNOSIDE = (
    "NCCOP(=O)(O)OC[C@H]1O[C@H](O[C@@H]2[C@@H](OC[C@H]3O[C@H](O[C@H]4[C@H](O)"
    "[C@@H](N)[C@H](OCCCCCCS)O[C@@H]4CO)[C@@H](O)[C@@H](O)[C@@H]3O)O[C@H](CO)"
    "[C@@H](O)[C@@H]2O)[C@@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O)"
    "[C@@H](O)[C@@H]1O"
)
# The GlcN-thioether fragment cut from the GPI mannoside (same test file).
GPI_FRAGMENT = (
    "N[C@H]1[C@H](OCCCCCCS)O[C@H](CO)[C@@H](O[C@H]2O[C@H](CO)[C@@H](O)"
    "[C@H](O)[C@@H]2O)[C@@H]1O"
)

SENTINEL_CASES = [CHEBI_131506, CANARY_171]
SENTINEL_IDS = ["chebi_131506", "canary_171"]
# Molecules whose bad candidate was a GLUE (fragment names joined by blanks).
# Only these carry the word-count check: a real name may contain a blank
# ('pent-2-enoic acid'), so the check is scoped to the known glue producer's
# input, as the plan's Task 3 Step 5 specifies.
GLUE_CASES = {GPI_MANNOSIDE}
CASES = SENTINEL_CASES + [GPI_MANNOSIDE]
CASE_IDS = SENTINEL_IDS + ["gpi_mannoside"]
TIERS = ["pin", "valid", "complete", "best-effort"]


def _namer(tier: str) -> Orthonym:
    return Orthonym(style="pin", **_emit_tier_flags(tier))


def _is_whole_or_clean(name) -> bool:
    """A name is acceptable iff it is empty, a WHOLE descriptive fallback, or
    carries no refusal sentinel at all ('unknown...', '(not supported)', the
    cascade placeholder)."""
    return (not name) or name in _DESCRIPTIVE_FALLBACK_NAMES or not is_refusal_sentinel(name)


def _rt_exact(smiles: str, name: str) -> bool:
    """OPSIN full-InChIKey round trip (stereo and charge included)."""
    back = opsin_parse(name) if name else None
    if not back:
        return False
    m_in, m_out = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(back)
    if m_in is None or m_out is None:
        return False
    return Chem.MolToInchiKey(m_in) == Chem.MolToInchiKey(m_out)


def _assert_no_splice_or_glue(smi: str, tier: str) -> None:
    n_components = len(smi.split("."))
    for label, name in (
        ("name", _namer(tier).name(smi)),
        ("name_tiered", _namer(tier).name_tiered(smi).get("name")),
    ):
        assert _is_whole_or_clean(name), (tier, label, name)
        if smi in GLUE_CASES and name and name not in _DESCRIPTIVE_FALLBACK_NAMES:
            # No bare space-separated fragment list: one connected input is
            # one name (a word count above the component count is a glue).
            assert len(name.split(" ")) <= n_components, (tier, label, name)


@pytest.mark.parametrize("smi", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("tier", TIERS)
def test_no_spliced_sentinel_raw_generator(smi, tier):
    """Validity gate OFF (the suite default): the raw generator's output."""
    _assert_no_splice_or_glue(smi, tier)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("tier", TIERS)
def test_no_spliced_sentinel_production(smi, tier):
    """Validity gate ON (production)."""
    _assert_no_splice_or_glue(smi, tier)


@pytest.mark.parametrize("smi", SENTINEL_CASES, ids=SENTINEL_IDS)
def test_no_candidate_splices_a_sentinel(smi, monkeypatch):
    """Producer level: no CANDIDATE built by `_name_impl` carries a welded
    sentinel. The exit of `name` also refuses a welded sentinel, so this trace
    is what shows the producer itself voids the candidate instead of
    splicing the string."""
    seen = []
    orig = Orthonym._name_impl

    def spy(self, smiles, *a, **k):
        out = orig(self, smiles, *a, **k)
        seen.append(out)
        return out

    monkeypatch.setattr(Orthonym, "_name_impl", spy)
    _namer("pin").name(smi)
    assert seen, "the spy never fired -- it proves nothing"
    bad = [s for s in seen if isinstance(s, str) and not _is_whole_or_clean(s)]
    assert not bad, bad


@pytest.mark.parametrize("welded", [
    "(2E)-2-methyl-5-unknownpent-2-enoic acid",       # the CHEBI:131506 shape
    "zinc compound (not supported)ylethane",           # errors.is_refusal_sentinel's witness
    "N-substituentformamide",                          # the cascade-placeholder family
])
def test_name_exit_never_ships_a_welded_sentinel(welded, monkeypatch):
    """Exit side of the same invariant, for every sentinel family: whatever a
    producer hands back, `name` / `name_tiered` return either a real name or
    a WHOLE fallback. Validity gate OFF (suite default), PIN tier, so no rescue
    can replace the injected candidate."""
    monkeypatch.setattr(Orthonym, "_name_impl", lambda self, smiles, *a, **k: welded)
    smi = "CCCCC(=O)O"
    for label, name in (
        ("name", _namer("pin").name(smi)),
        ("name_tiered", _namer("pin").name_tiered(smi).get("name")),
    ):
        assert name in _DESCRIPTIVE_FALLBACK_NAMES, (label, name)


@pytest.mark.opsin_gate
@pytest.mark.parametrize(
    "smi",
    [CHEBI_131506, CANARY_171, GPI_MANNOSIDE, GPI_FRAGMENT],
    ids=["chebi_131506", "canary_171", "gpi_mannoside", "gpi_fragment"],
)
def test_best_effort_names_every_case_rt_exact(smi):
    """Breadth never drops: voiding the bad candidate must leave the molecule
    named, OPSIN full-InChIKey exact, at the best-effort tier."""
    name = _namer("best-effort").name_tiered(smi).get("name")
    assert name and name not in _DESCRIPTIVE_FALLBACK_NAMES, name
    assert _rt_exact(smi, name), name


def test_bond_type_assembler_never_space_joins_components():
    """`fragment_assembly._assemble_by_bond_type` groups fragments by the bond
    type that was cut, not by how they connect, so it cannot know whether two
    independently named parts form one molecule. A blank-joined list of
    component names is not a name of the input (OPSIN parses the GPI glue to
    C38H75N2O31PS, the input is C38H71N2O28PS; TRIAGE.csv row
    test_iterative_mixed_decompose_achieves_atom_complete_assembly). It must
    decline, never glue. The four (fragment, name) pairs are the ones the GPI
    mannoside's mixed decomposition passes in (traced 2026-09-25)."""
    from orthonym.decomposition.fragment_assembly import _assemble_by_bond_type

    named = [
        ({"smiles": "NCCOP(=O)(O)O", "parent_bond_type": "phosphodiester"},
         "2-(phosphonooxy)ethan-1-amine"),
        ({"smiles": GPI_FRAGMENT, "parent_bond_type": "glycosidic"},
         "(2R,3R,4R,5S,6R)-3-amino-6-(hydroxymethyl)-5-(α-D-mannopyranosyloxy)"
         "-2-(6-sulfanylhexyloxy)oxan-4-ol"),
        ({"smiles": "OC[C@H]1O[C@H](O)[C@@H](O)[C@@H](O)[C@@H]1O",
          "parent_bond_type": "ether"},
         "α-D-mannopyranosyloxy"),
        ({"smiles": "OC[C@H]1O[C@H](O[C@H]2[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]2O)"
                    "[C@@H](O)[C@@H](O)[C@@H]1O",
          "parent_bond_type": "ether"},
         "α-D-mannopyranosyl-(1->2)-α-D-mannopyranose"),
    ]
    out = _assemble_by_bond_type(named, {"phosphodiester", "glycosidic", "ether"}, "pin")
    assert out is None, out


@pytest.mark.parametrize("acid, alkyl, expected, smiles", [
    # "Esters of mononuclear noncarbon oxoacids" (BB:35918): partial
    # esters cite the groups as separate words in alphanumeric order, then
    # 'hydrogen', then the anion; BB:35944 'sodium methyl hydrogen phosphate
    # (PIN)'. OPSIN: 'ethyl methyl hydrogen phosphate' -> P(=O)(OCC)(OC)O.
    ("methyl dihydrogen phosphate", "ethanol", "ethyl methyl hydrogen phosphate",
     "CCOP(=O)(O)OC"),
    # Identical groups take the multiplying prefix: (BB:4857)
    # 'di' for simple groups, 'bis' for compound ones; BB:35946 'dimethyl
    # phosphonate (PIN)'. The old body returned 'methyl methyl dihydrogen
    # phosphate', which OPSIN cannot parse.
    ("methyl dihydrogen phosphate", "methanol", "dimethyl hydrogen phosphate",
     "COP(=O)(O)OC"),
    ("2-chloroethyl dihydrogen phosphate", "2-chloroethan-1-ol",
     "bis(2-chloroethyl) hydrogen phosphate", "ClCCOP(=O)(O)OCCCl"),
    # A substitutively named acid fragment has no ester word to extend: the old
    # body glued 'ethyl 2-(phosphonooxy)ethan-1-amine', which OPSIN parses to a
    # DIFFERENT molecule (C(C)C(COP(=O)(O)O)N). Declined.
    ("2-(phosphonooxy)ethan-1-amine", "ethanol", None, None),
    # An alcohol-side name the alkyl converter cannot turn into a group word.
    ("methyl dihydrogen phosphate", "ethan-1-amine", None, None),
], ids=["mixed_diester", "identical_simple", "identical_compound",
        "substitutive_acid", "non_group_alcohol_side"])
def test_phosphodiester_assembler_builds_the_p67_form_or_declines(acid, alkyl, expected, smiles):
    from orthonym.decomposition.fragment_assembly import _assemble_phosphodiester

    out = _assemble_phosphodiester({"acid": acid, "alkyl": alkyl}, "pin")
    assert out == expected
    if expected is not None:
        assert _rt_exact(smiles, out), out
