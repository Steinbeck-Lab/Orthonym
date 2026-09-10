"""F- (DD3) — charged-species suffix generalization, rule-family coverage.

Tests the OUTPUT of the whole class (A8: not literal canary rows), via the public
``Orthonym.name`` surface so the routing + emitter + suffix composition are all
exercised end to end:

  * ring-N -ium (protonated + N-substituted aromatic demote) with
    'e' elision and the cationic-centre locant.
  * ring carbanion -ide.
  * acyclic Group-14/15 heteroatom -ide (P/As/Sb/Si/Ge).
  * aryl amine -ium 'e' elision (anilinium).
  * azanide preselected anion word.
  * zwitterion cumulative -ium-...-carboxylate.

Plus byte-identity protect families (the acyclic-carbon carbanion/radical path,
carboxylate/alkoxide/thiolate/selenolate, and neutral parents) that MUST NOT
change — the shared emitter dispatch and the route split must leave them intact.
"""

import pytest

from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym(style="pin")


# ---: protonated ring-N -ium (in-place neutralization) -------------
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
    # locant + elision are what F- owns and must be correct.
    out = namer.name("c1c[nH]c[nH+]1")
    assert out in ("1H-imidazol-3-ium", "imidazol-3-ium")
    assert out.endswith("-3-ium")


# ---: N-substituted aromatic ring-N (0 H) demote -------------------
@pytest.mark.parametrize("smiles,expected", [
    ("C[n+]1ccccc1", "1-methylpyridin-1-ium"),
    ("CC[n+]1ccccc1", "1-ethylpyridin-1-ium"),
])
def test_n_substituted_aromatic_ring_demote(namer, smiles, expected):
    assert namer.name(smiles) == expected


# ---: ring carbanion -ide ------------------------------------------
def test_ring_carbanion_ide(namer):
    assert namer.name("[CH-]1CCCCC1") == "cyclohexanide"  # locant omitted, (c) the Blue Book


# ---: imine anion -iminide, WITH locant omission ---------
@pytest.mark.parametrize("smiles,expected", [
    # the Blue Book butaniminide (PIN) -- the imine-position locant is OMITTED
    # (not butan-1-iminide); the =N- anion has no substitutable N-H.
    ("CCCC=[N-]", "butaniminide"),
    ("CCC=[N-]", "propaniminide"),
    ("CC=[N-]", "ethaniminide"),
    # the Blue Book trimethyl-λ5-phosphaniminide (PIN) -- the P,P,P substituent
    # locants are OMITTED, only one kind of substitutable H on the
    # sole heteroatom centre); contrast the NEUTRAL As,As,As-... which keeps them.
    ("CP(C)(C)=[N-]", "trimethyl-λ5-phosphaniminide"),
])
def test_iminide_pin_locant_omission(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_iminide_internal_keeps_locant(namer):
    # 0-wrong safety: an INTERNAL imine anion (butan-2-iminide) must KEEP its
    # locant -- dropping it would collapse to the position-1 default (a different
    # molecule), which the full-InChIKey RT audit rejects.
    assert namer.name("CC(=[N-])CC") == "butan-2-iminide"


# ---: acyclic heteroatom -ide (the nameable heterane families) -----
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


# ---: aryl amine -ium 'e' elision ----------------------------------
@pytest.mark.parametrize("smiles", ["c1ccccc1[NH3+]", "[NH3+]c1ccccc1"])
def test_anilinium_elision(namer, smiles):
    assert namer.name(smiles) in ("anilinium", "benzenaminium")


# ---: azanide preselected anion word -------------------------------
def test_azanide(namer):
    assert namer.name("[NH2-]") == "azanide"


# ---: zwitterion cumulative -ium-...-carboxylate ---------------------
@pytest.mark.parametrize("smiles,expected", [
    ("[O-]C(=O)c1ccc[nH+]c1", "pyridin-1-ium-3-carboxylate"),       # meta (nicotinate)
    ("O=C([O-])c1cccc[nH+]1", "pyridin-1-ium-2-carboxylate"),       # ortho (picolinate)
    ("C[n+]1ccccc1C(=O)[O-]", "1-methylpyridin-1-ium-2-carboxylate"),
])
def test_zwitterion_ring_carboxylate(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --- Byte-identity PROTECT families (must not regress) ------------------------
@pytest.mark.parametrize("smiles,expected", [
    # acyclic carbon carbanion / radical (a phase primitive — unchanged)
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


# --- Code-review hardening (valence gate, / fail-closed,) -
@pytest.mark.parametrize("smiles,expected", [
    # 4-coordinate (over-coordinated) Group-14 metalloid anion = the -uide
    # family, NOT a parent-hydride -ide. The valence gate (degree+H+1 == valence)
    # must EXCLUDE it so it stays on the neutral organometallic path and keeps its
    # structure-preserving name — never charge-dropped to '' (the regression).
    ("C[Si-](C)(C)C", "tetramethylsilane"),
    ("C[Sn-](C)(C)C", "tetramethylstannane"),
])
def test_overcoordinated_heteroatom_anion_not_regressed(namer, smiles, expected):
    out = namer.name(smiles)
    assert out == expected
    assert not out.endswith("ide")          # not a spurious -ide
    assert out and "unknown" not in out      # not charge-dropped to ''/unknown


def test_valence_gate_excludes_invalid_arities():
    """classify_anion must return 'heteroatom_hydride_anion' ONLY for a valid
    parent-hydride -ide arity (degree + H + 1 == standard valence). A 3-coordinate
    P anion (no P-H was lost) and a 4-coordinate Si anion (the -uide family) must
    NOT be classified as the -ide class."""
    from rdkit import Chem
    from orthonym.rules.ions import classify_anion
    from orthonym.perception.ions import get_ion_sites

    def _acls(smi):
        m = Chem.MolFromSmiles(smi)
        return classify_anion(m, get_ion_sites(m)["anions"][0])

    assert _acls("C[P-]C") == "heteroatom_hydride_anion"        # 2-coord P -> phosphanide
    assert _acls("C[Si-](C)C") == "heteroatom_hydride_anion"    # 3-coord Si -> silanide
    assert _acls("C[P-](C)C") != "heteroatom_hydride_anion"     # 3-coord P (invalid -ide arity)
    assert _acls("C[Si-](C)(C)C") != "heteroatom_hydride_anion"  # 4-coord Si (-uide)


def test_heteroatom_hydride_anide_is_the_pin():
    """ (BB 39765 'dimethylarsanido'): a Group-15 heteroatom-hydride anion
    (loss of H+ from dimethylarsane/dimethylstibane) IS named with the '-anide'
    parent-hydride suffix — dimethylarsanide / dimethylstibanide are the PINs, NOT
    garbage. The heterane-stem guard only fails closed when the neutral hydride
    genuinely cannot be named."""
    assert Orthonym().name("C[As-]C") == "dimethylarsanide"
    assert Orthonym().name("C[Sb-]C") == "dimethylstibanide"


def test_unnameable_heteroatom_anion_never_emits_fake_ide():
    """The guard still fails closed for a genuinely un-nameable heteroatom anion:
    (CH3)3As- (no As-H to lose -> not a clean '-anide') must NOT emit a fake
    '*anide' (it fails closed to a descriptive 'not supported' form)."""
    out = Orthonym().name("C[As-](C)C")
    assert not out.endswith("anide"), f"C[As-](C)C -> {out} (spurious heteroatom -ide)"


# --- Follow-on fixes: adjacent-N azolium (diazonium mis-class), indicated H,
# >=3-heteroatom rings, and the -uide / borate family --------------------
@pytest.mark.parametrize("smiles,expected", [
    ("c1cc[nH+][nH]1", "1H-pyrazol-2-ium"),     # was 'pyrazolediazonium'
    ("c1cc[nH+]nc1", "pyridazin-1-ium"),         # was 'pyridazinediazonium'; cation lowest locant
    ("c1cc[nH+]cn1", "pyrimidin-1-ium"),
    ("c1c[nH]c[nH+]1", "1H-imidazol-3-ium"),     # indicated H injected (was 'imidazol-3-ium')
])
def test_adjacent_n_azolium_not_diazonium(namer, smiles, expected):
    """The diazonium check now requires a DOUBLE/TRIPLE N-N bond (P-73.2.2.3), so an
    adjacent-N AROMATIC ring cation routes to the ring -ium emitter with the correct
    indicated hydrogen and cation-lowest locant — NOT the bogus '…diazonium'."""
    assert namer.name(smiles) == expected


def test_real_diazonium_preserved(namer):
    """A genuine terminal -N2+ diazonium (triple N#N) is unaffected by the gate."""
    from rdkit import Chem
    from orthonym.rules.ions import classify_cation
    from orthonym.perception.ions import get_ion_sites
    m = Chem.MolFromSmiles("c1ccccc1[N+]#N")
    assert classify_cation(m, get_ion_sites(m)["cations"][0]) == "diazonium"


@pytest.mark.parametrize("smiles", [
    "c1c[nH+][nH]n1",   # 1,2,3-triazolium
    "c1nc[nH+][nH]1",   # 1,2,4-triazolium
])
def test_triazolium_deterministic(namer, smiles):
    """>=3 equal-priority heteroatom rings: the single-charged-ring enumerator does a
    GLOBAL all-starts numbering, so the name is deterministic across SMILES spellings."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    outs = {namer.name(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
            for _ in range(8)}
    assert len(outs) == 1, f"nondeterministic: {outs}"
    assert "diazonium" not in next(iter(outs))


@pytest.mark.parametrize("smiles,expected", [
    ("C[B-](C)(C)C", "tetramethylboranuide"),          # -uide / borate
    ("CC[B-](CC)(CC)CC", "tetraethylboranuide"),
    ("[B-](F)(F)(F)F", "tetrafluoroboranuide"),
])
def test_group13_uide_borate(namer, smiles, expected):
    """ (DD3 Fix 5): a Group-13 centre one bond above its standard valence
    bearing the -1 charge is the ate-complex / -uide (tetramethylboranuide,
    tetrafluoroboranuide), named via the substituted-'-uide'-parent emitter."""
    assert namer.name(smiles) == expected


def test_overcoordinated_silicon_has_no_uide(namer):
    """`C[Si-](C)(C)C` (4-coordinate Si, 0 H) is NOT silanuide (that name adds an H:
    C[SiH-](C)(C)C). It has no clean PIN, so it stays the organometallic neutral name
    rather than a wrong -uide."""
    assert namer.name("C[Si-](C)(C)C") == "tetramethylsilane"


def test_fg_substituted_ring_cation_fails_closed():
    """: the ring -ium emitter must NOT compose a malformed
    `<ring>-<locant>-ol-<locant>-ium` for a ring cation that also bears a
    principal-group FG (-ol/-al). It fails closed (the FG case is /
    territory). Direct emitter check: returns '' rather than a malformed string."""
    from rdkit import Chem
    from orthonym.rules.ions import _emit_ring_cumulative_suffix
    from orthonym.perception.ions import get_ion_sites
    for smi in ["Oc1ccc[nH+]c1", "O=Cc1ccc[nH+]c1"]:
        m = Chem.MolFromSmiles(smi)
        ci = get_ion_sites(m)["cations"][0]["atom_idx"]
        assert _emit_ring_cumulative_suffix(m, ci, "ium") == ""
