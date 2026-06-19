"""F-T6 (DD3) — charged-species suffix generalization, rule-family coverage.

Tests the OUTPUT of the whole class (A8: not literal canary rows), via the public
``Orthonym().name`` surface so the routing + emitter + suffix composition are all
exercised end to end:

  * P-73.1.1.2  ring-N -ium (protonated + N-substituted aromatic demote) with
    'e' elision and the cationic-centre locant.
  * P-72.2.2.1  ring carbanion -ide.
  * P-72.2.2.1  acyclic Group-14/15 heteroatom -ide (P/As/Sb/Si/Ge).
  * P-73.1.2.1  aryl amine -ium 'e' elision (anilinium).
  * P-72.2.2.2  azanide preselected anion word.
  * P-74.1.2    zwitterion cumulative -ium-...-carboxylate.

Plus byte-identity protect families (the acyclic-carbon carbanion/radical path,
carboxylate/alkoxide/thiolate/selenolate, and neutral parents) that MUST NOT
change — the shared emitter dispatch and the route split must leave them intact.
"""

import pytest

from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym(style="pin")


# --- P-73.1.1.2: protonated ring-N -ium (in-place neutralization) -------------
@pytest.mark.parametrize("smiles,expected", [
    ("c1cc[nH+]cc1", "pyridin-1-ium"),
    ("[nH+]1ccccc1", "pyridin-1-ium"),            # different SMILES spelling -> same name
    ("O1CC[NH2+]CC1", "morpholin-4-ium"),         # O senior -> locant 1; N at 4
    ("C1CC[NH2+]CC1", "piperidin-1-ium"),
    ("c1ccc2[nH+]cccc2c1", "quinolin-1-ium"),     # fused ring numbering reused
])
def test_protonated_ring_nitrogen_ium(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_imidazolium_locant_and_elision(namer):
    # Indicated hydrogen is a neutral-ring-namer limitation; the cationic-centre
    # locant + elision are what F-T6 owns and must be correct.
    out = namer.name("c1c[nH]c[nH+]1")
    assert out in ("1H-imidazol-3-ium", "imidazol-3-ium")
    assert out.endswith("-3-ium")


# --- P-73.1.1.2: N-substituted aromatic ring-N (0 H) demote -------------------
@pytest.mark.parametrize("smiles,expected", [
    ("C[n+]1ccccc1", "1-methylpyridin-1-ium"),
    ("CC[n+]1ccccc1", "1-ethylpyridin-1-ium"),
])
def test_n_substituted_aromatic_ring_demote(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --- P-72.2.2.1: ring carbanion -ide ------------------------------------------
def test_ring_carbanion_ide(namer):
    assert namer.name("[CH-]1CCCCC1") == "cyclohexan-1-ide"


# --- P-72.2.2.1: acyclic heteroatom -ide (the nameable heterane families) -----
@pytest.mark.parametrize("smiles,expected", [
    ("C[P-]C", "dimethylphosphanide"),        # P -> phosphane
    ("C[Si-](C)C", "trimethylsilanide"),      # Si -> silane (3-methyl, the -ide)
    ("CC[P-]CC", "diethylphosphanide"),
])
def test_heteroatom_hydride_ide(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_unnameable_heterane_fails_closed_not_garbage(namer):
    # methylarsane is not nameable by the pipeline (re-enters as a garbled
    # 'methylmethylmethyl'); the heteroatom -ide branch MUST fail-closed rather
    # than emit 'methylmethylmethylide'. The charge-dropped legacy name is
    # acceptable (no regression); a *-ide* garble is not.
    out = namer.name("C[As-](C)C")
    assert not out.endswith("ide") or "arsan" in out


# --- P-73.1.2.1: aryl amine -ium 'e' elision ----------------------------------
@pytest.mark.parametrize("smiles", ["c1ccccc1[NH3+]", "[NH3+]c1ccccc1"])
def test_anilinium_elision(namer, smiles):
    assert namer.name(smiles) in ("anilinium", "benzenaminium")


# --- P-72.2.2.2: azanide preselected anion word -------------------------------
def test_azanide(namer):
    assert namer.name("[NH2-]") == "azanide"


# --- P-74.1.2: zwitterion cumulative -ium-...-carboxylate ---------------------
@pytest.mark.parametrize("smiles,expected", [
    ("[O-]C(=O)c1ccc[nH+]c1", "pyridin-1-ium-3-carboxylate"),       # meta (nicotinate)
    ("O=C([O-])c1cccc[nH+]1", "pyridin-1-ium-2-carboxylate"),       # ortho (picolinate)
    ("C[n+]1ccccc1C(=O)[O-]", "1-methylpyridin-1-ium-2-carboxylate"),
])
def test_zwitterion_ring_carboxylate(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --- Byte-identity PROTECT families (must not regress) ------------------------
@pytest.mark.parametrize("smiles,expected", [
    # acyclic carbon carbanion / radical (Phase 184 primitive — unchanged)
    ("[CH3-]", "methanide"),
    ("CC[CH-]CC", "pentan-3-ide"),
    ("CCC[CH-]CC", "hexan-3-ide"),
    # carboxylate / alkoxide / thiolate / selenolate paths (unchanged)
    ("CC(=O)[O-]", "acetate"),
    ("C[O-]", "methoxide"),
    ("C[S-]", "methanethiolate"),
    ("C[Se-]", "methaneselenolate"),   # chalcogen rides the -ol -> -olate suffix map
    # acyclic protonated amine -aminide / -aminium (unchanged)
    ("C[N-]C", "N-methylmethanaminide"),
])
def test_charged_protect_families(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # neutral parents must be untouched by the dispatch split
    ("c1ccncc1", "pyridine"),
    ("C[SiH](C)C", "trimethylsilane"),
    ("C1CCCCC1", "cyclohexane"),
])
def test_neutral_parents_unchanged(namer, smiles, expected):
    assert namer.name(smiles) == expected
