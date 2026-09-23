"""v52 a phase Task 5 — mononuclear / multi ``-ide`` & ``-uide`` ANION
completeness (IUPAC.x).

Three root-cause fixes, each cited to its governing rule:

* **N-aminide**, the Blue Book *"Amines... having one
  negative charge on each nitrogen atom are named by using the suffix
  'aminide'"*): ``[NH-]c1ccccc1`` -> ``benzenaminide``. The default ionize seam
  converts ``amine``->``aminide`` on the re-entered neutral, but an aromatic amine
  re-enters as the RETAINED ``aniline`` (no ``-amine`` token). The new emitter
  builds the SYSTEMATIC ``benzenamine`` parent and converts it, RT-gated.

* **Multiply-bonded skeletal Si -ide**, the Blue Book *"Anions
  derived from parent hydrides and their derivatives"*): ``C#[Si-]`` ->
  ``methylidynesilanide``. ``classify_anion``'s valence gate used ``GetDegree``
  (a neighbour count) which undercounts a triple bond, so the silanide fell to
  'unknown'; and the neutralize->re-enter path names the neutral ``silylmethane``
  (carbon wins parent selection), so the anion is now named DIRECTLY on the
  silane parent with a ``methylidyne`` substituent.

* **Polycyano methanide**: ``N#C[C-](C#N)C#N`` ->
  ``tricyanomethanide``. ``classify_substituent`` counted only the carbon of a
  ``-C#N`` group and mis-named it 'methyl' (a WRONG MOLECULE,
  ``1,1,1-trimethylmethanide``); it now returns 'cyano', and the single-carbon
  (methane) parent omits its substituent locants.

All three targets reproduce with the PLAIN engine (``Orthonym.name(...)`` and
``.name_tiered(...)`` — verified 2026-09-22); the general/best-effort flags used by
the fixture are for parity with the bb harness and do not change these outputs.
"""
import pytest

from rdkit import Chem

from orthonym import Orthonym

FLAGS = dict(general_fallback=True, general_fallback_unverified=True,
             allow_aromatic_general=True)


@pytest.fixture(scope="module")
def namer():
    return Orthonym(**FLAGS)


# ------------------------------------------------------------------ targets ---
TARGETS = [
    # (smiles, expected_PIN, rule)
    ("[NH-]c1ccccc1", "benzenaminide", "P-72.2.2.2.3"),
    ("C#[Si-]", "methylidynesilanide", "P-72.2.2.1"),
    ("N#C[C-](C#N)C#N", "tricyanomethanide", "P-72.2.2.1"),
]


@pytest.mark.parametrize("smiles,expected,rule", TARGETS)
def test_anion_ide_uide_targets(namer, smiles, expected, rule):
    assert namer.name_tiered(smiles)["name"] == expected


# --------------------------------------------------------------- regressions --
# Anion controls that already worked and must NOT regress (the same emitters /
# classification the targets touch).
CONTROLS = [
    # single heteroatom-hydride -ide (neutralize->re-enter path)
    ("C[P-]C", "dimethylphosphanide"),
    ("C[Si-](C)C", "trimethylsilanide"),
    ("[SiH3-]", "silanide"),
    # chain aminide (default ionize seam; NOT the new aryl emitter)
    ("[NH-]C", "methanaminide"),
    ("[NH-]CC", "ethanaminide"),
    ("[NH-]CCC", "propan-1-aminide"),
    ("[NH-]C1CCCCC1", "cyclohexanaminide"),
    # carbanion / multi / imine / bis(aminide)
    ("[C-]#[C-]", "ethynediide"),
    ("CCCC=[N-]", "butaniminide"),
    ("[NH-]CC[NH-]", "ethane-1,2-bis(aminide)"),
    ("CCCC=[N-]", "butaniminide"),
    ("CCC(=O)[S-]", "propanethioate"),
]


@pytest.mark.parametrize("smiles,expected", CONTROLS)
def test_anion_controls_unchanged(namer, smiles, expected):
    assert namer.name_tiered(smiles)["name"] == expected


# The multiply-bonded-skeletal generalization is not Si-only covers
# the whole Group-14/15 -ide family): Ge behaves identically.
def test_methylidynegermanide(namer):
    assert namer.name_tiered("C#[Ge-]")["name"] == "methylidynegermanide"


# ------------------------------------------------------------------- 0-wrong --
# The new aryl-amine-anide emitter is FULL-InChIKey RT-gated: a substituted /
# heteroatom parent whose bare (locant-omitted) amine spelling does not round-trip
# must ABSTAIN (None) or emit an RT-valid name — NEVER a wrong molecule.
@pytest.mark.parametrize("smiles", [
    "[NH-]c1ccc(C)cc1",   # 'toluenamine' would not round-trip -> must not ship it
    "[NH-]c1ccncc1",      # 'pyridinamine' likewise
])
def test_aryl_aminide_never_wrong_molecule(namer, smiles):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    name = namer.name_tiered(smiles)["name"]
    if name is None:
        return  # abstain is acceptable (0-wrong)
    parsed = opsin_parse(name)
    assert parsed, f"emitted an OPSIN-unparseable name {name!r}"
    want = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    got = Chem.MolToInchiKey(Chem.MolFromSmiles(parsed))
    assert got == want, f"WRONG MOLECULE: {name!r} -> {parsed!r}"


# ------------------------------------------------------ unit-level fixtures ----
def test_classify_substituent_nitrile_is_cyano():
    """A ``-C#N`` substituent is 'cyano', never the carbon-count
    'methyl' that silently drops the nitrogen."""
    from orthonym.perception.chains import classify_substituent
    m = Chem.MolFromSmiles("N#CC")   # acetonitrile; treat -C#N as a substituent on the CH3
    # atoms: 0=N, 1=C(nitrile), 2=C(methyl)
    info = classify_substituent(m, [0, 1], {2})
    assert info["name"] == "cyano"


def test_classify_anion_silanide_multibond():
    """The -ide valence gate counts BOND ORDERS, so a triple-bonded skeletal Si
    anion classifies as a heteroatom-hydride anion (not 'unknown')."""
    from orthonym.rules.ions import classify_anion
    from orthonym.perception.ions import get_ion_sites
    m = Chem.MolFromSmiles("C#[Si-]")
    site = get_ion_sites(m)["anions"][0]
    assert classify_anion(m, site) == "heteroatom_hydride_anion"


def test_occupied_valence_triple_bond():
    from orthonym.rules.ions import _occupied_valence
    m = Chem.MolFromSmiles("C#[Si-]")
    si = [a for a in m.GetAtoms() if a.GetSymbol() == "Si"][0]
    assert _occupied_valence(si) == 3   # triple bond (3) + 0 H


# --------------------------------------------------- shared-consumer 0-wrong --
# The cyano widening of classify_substituent also reaches emit_chalcogen_ylium
# chalcogen ylium). That return is now full-InChIKey RT-gated
# (charged_router), so the emitted cyano name is PROVEN, not lucky. Lock it.
def test_chalcogen_ylium_cyano_is_gated_and_correct(namer):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    name = namer.name_tiered("[S+]C#N")["name"]
    assert name == "cyanosulfanylium"
    parsed = opsin_parse(name)
    assert parsed
    want = Chem.MolToInchiKey(Chem.MolFromSmiles("[S+]C#N"))
    got = Chem.MolToInchiKey(Chem.MolFromSmiles(parsed))
    assert got == want, f"WRONG MOLECULE: {name!r} -> {parsed!r}"


# The polyvalent-radical consumer (emit_parent_hydride_polyvalent_suffixes) is now
# RT-gated too; every in-scope diyl/triyl name still emits (no regression from the gate).
@pytest.mark.parametrize("smiles,expected", [
    ("[CH2][CH2]", "ethane-1,2-diyl"),
    ("[CH2][CH][CH2]", "propane-1,2,3-triyl"),
    ("CC([CH2])[CH2]", "2-methylpropane-1,3-diyl"),
])
def test_polyvalent_radical_still_emits_under_gate(smiles, expected):
    from orthonym.rules.charged_router import route_charged
    assert route_charged(Chem.MolFromSmiles(smiles), "pin") == expected
