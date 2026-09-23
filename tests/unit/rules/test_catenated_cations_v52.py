"""v52 a phase Task 3 —.x catenated / substituted parent-hydride cations.

A cation formed by adding a hydron to a catenated homonuclear parent hydride
(hydrazine / trisulfane / diphosphane / dioxidane) or to a substituted mononuclear
pnictogen hydride (phosphane) is named on the parent hydride + the '-ium' suffix at
the cationic-centre locant, NOT the '-a'/'-onia'-replacement form and NOT the
linearized ``name_quaternary_aminium`` spelling.

Governing rule (VERIFIED, verbatim): ** "General rule for systematically
naming cationic centers in parent hydrides"** (``the Blue Book``): "A cation
derived formally by adding one or more hydrons to any position of a neutral parent
hydride... is named by replacing the final letter 'e' of the parent hydride name,
if any, by the suffix 'ium'... These names for mononuclear cations derived from the
mononuclear parent hydrides of the Group 15, 16, and 17 elements are the preferred
IUPAC names and not those given in Table 7.3." All five expected names appear
verbatim as ``(PIN)`` at ``the Blue Book-41390`` (and the dioxidane family at
``:41429/:41474``); every ``expected`` string round-trips through OPSIN 2.9.0 to the
input InChIKey (the 0-wrong RT gate the router applies before shipping).

The bb-conformance harness names with
``Orthonym(general_fallback=True, general_fallback_unverified=True,
allow_aromatic_general=True).name_tiered(smiles)``; the plain CLI abstains for these
(no ``general_fallback_unverified``), so every assertion here uses those flags.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.perception.ions import get_ion_sites
from orthonym.rules import ions

FLAGS = dict(general_fallback=True, general_fallback_unverified=True,
             allow_aromatic_general=True)


@pytest.fixture(scope="module")
def namer():
    return Orthonym(**FLAGS)


# (smiles, expected_PIN, bb_def_id) — verbatim BB PINs (bb_measure_rows.jsonl).
CATENATED_CATIONS = [
    # catenated homonuclear parent-hydride cations
    ("CN(C)[N+](C)(C)C", "pentamethylhydrazinium", "73.1.1.2"),
    ("CS[S+](C)SC", "1,2,3-trimethyltrisulfan-2-ium", "73.1.1.2"),
    ("C[P+](C)(C)P(Cl)Cl", "2,2-dichloro-1,1,1-trimethyldiphosphan-1-ium", "73.1.1.2"),
    # substituted mononuclear pnictogen cation
    ("C[P+](C)(C)Cl", "chlorotri(methyl)phosphanium", "73.1.1.1"),
    # catenated chalcogen (dioxidane) cation — deferred from Task 2
    ("O=C(O[OH2+])c1ccccc1", "2-benzoyldioxidan-1-ium", "73.1.2.1"),
]


@pytest.mark.parametrize("smiles,expected,bb", CATENATED_CATIONS)
def test_catenated_cation_direct(smiles, expected, bb):
    """The dedicated direct emitter builds the exact PIN (BB rule <bb>)."""
    mol = Chem.MolFromSmiles(smiles)
    cat_idx = get_ion_sites(mol)["cations"][0]["atom_idx"]
    assert ions.emit_catenated_hydride_cation(mol, cat_idx) == expected


@pytest.mark.parametrize("smiles,expected,bb", CATENATED_CATIONS)
def test_catenated_cation_end_to_end(namer, smiles, expected, bb):
    """The router, RT-gated) ships the PIN via name_tiered."""
    assert namer.name_tiered(smiles)["name"] == expected


@pytest.mark.parametrize("smiles,expected,bb", CATENATED_CATIONS)
def test_catenated_cation_deterministic(smiles, expected, bb):
    """Determinism (a project rule): the same input always yields the same name."""
    mol = Chem.MolFromSmiles(smiles)
    cat_idx = get_ion_sites(mol)["cations"][0]["atom_idx"]
    first = ions.emit_catenated_hydride_cation(mol, cat_idx)
    second = ions.emit_catenated_hydride_cation(
        Chem.MolFromSmiles(smiles),
        get_ion_sites(Chem.MolFromSmiles(smiles))["cations"][0]["atom_idx"])
    assert first == second == expected


# orientation tie-break: when both chain numberings tie on the cationic
# locant AND the substituent-locant set, the lowest locants go to the substituent
# cited FIRST alphanumerically (chloro < methyl), NOT to whichever end RDKit's atom
# index (SMILES input order) happens to favour. The SAME molecule written two ways
# must give ONE PIN. (The identical-string re-parse in the deterministic test above
# cannot catch this class — the two SMILES here differ in atom order.)
TIE_BREAK_PAIRS = [
    # 1-chloro-2,3-dimethyltrisulfan-2-ium, entered from each terminus
    ("ClS[S+](C)SC", "CS[S+](C)SCl", "1-chloro-2,3-dimethyltrisulfan-2-ium"),
    # the diphosphan target, chain written from either P
    ("C[P+](C)(C)P(Cl)Cl", "ClP(Cl)[P+](C)(C)C",
     "2,2-dichloro-1,1,1-trimethyldiphosphan-1-ium"),
]


@pytest.mark.parametrize("smi_a,smi_b,expected", TIE_BREAK_PAIRS)
def test_catenated_cation_orientation_tiebreak_p1452(smi_a, smi_b, expected):
    """: atom-order-reversed SMILES of one molecule -> one PIN."""
    def nm(s):
        mol = Chem.MolFromSmiles(s)
        return ions.emit_catenated_hydride_cation(
            mol, get_ion_sites(mol)["cations"][0]["atom_idx"])
    # both SMILES really are the same molecule
    from rdkit.Chem.inchi import MolToInchiKey
    assert (MolToInchiKey(Chem.MolFromSmiles(smi_a))
            == MolToInchiKey(Chem.MolFromSmiles(smi_b)))
    assert nm(smi_a) == nm(smi_b) == expected


# The direct emitter must DECLINE (fail closed -> '') for every centre the existing
# structured handlers own, so no working name is ever displaced (0-regression).
DECLINE = [
    "C[S+](C)C",          # trimethylsulfanium (chalcogen mononuclear -> Task 2)
    "[SH3+]", "[PH4+]",   # bare retained onium parents (retained table)
    "[NH4+]",             # azanium (mononuclear N -> aminium path)
    "c1ccc([I+]c2ccccc2)cc1",  # diphenyliodanium (halogen onium)
    "C[n+]1ccccc1",       # 1-methylpyridin-1-ium (ring)
    "CC#[O+]", "C[F+]Cl", "C[Cl+]C(C)=O",  # other-task cations
    "C[N+]12CCN(CC1)C2",  # ring quaternary N
]


@pytest.mark.parametrize("smiles", DECLINE)
def test_catenated_cation_declines_out_of_scope(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for c in get_ion_sites(mol).get("cations", []):
        assert ions.emit_catenated_hydride_cation(mol, c["atom_idx"]) == ""


# ---------------------------------------------------------------------------
# / enclosing marks on a COMPLEX substituent cited ONCE.
#
# v52 a phase whole-branch (a review) review, Task 8. ``emit_catenated_hydride_cation``
# enclosed a complex substituent ONLY when multiplied (``is_complex_substituent(nm)
# and len(locs_sorted) > 1``); a count-1 complex prefix fell to the ``else`` and lost
# its marks, shipping ``…-1-propan-2-yldisulfan-1-ium``. A detachable ``propan-2-yl``
# prefix is ALWAYS parenthesized — BB ``2-methyl-5-(propan-2-yl)phenol`` (PIN,
# the Blue Book), ``1,4-di(propan-2-yl)cyclohexane`` (PIN,:25719). RT is blind
# to this (all forms round-trip), so it is asserted as a string PIN AND an RT pass.
COMPLEX_SUB_CATIONS = [
    ("CC(C)[SH+]SC", "2-methyl-1-(propan-2-yl)disulfan-1-ium"),
    ("CC(C)S[S+](C)SC", "1,2-dimethyl-3-(propan-2-yl)trisulfan-2-ium"),
]


@pytest.mark.parametrize("smiles,expected", COMPLEX_SUB_CATIONS)
def test_count1_complex_substituent_is_enclosed(smiles, expected):
    """A complex substituent cited ONCE keeps its enclosing marks."""
    mol = Chem.MolFromSmiles(smiles)
    cat_idx = get_ion_sites(mol)["cations"][0]["atom_idx"]
    assert ions.emit_catenated_hydride_cation(mol, cat_idx) == expected


@pytest.mark.parametrize("smiles,expected", COMPLEX_SUB_CATIONS)
def test_count1_complex_substituent_roundtrips(smiles, expected):
    """The enclosed PIN OPSIN-round-trips to the input structure (0-wrong)."""
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    assert opsin_roundtrip_check(smiles, expected).get("passed") is True


def test_multiplied_complex_substituent_still_di_and_enclosed():
    """Non-regression: a MULTIPLIED complex substituent keeps ``di(propan-2-yl)``
     — a bare-locant simple-prefix parent multiplier, NOT ``bis``, NOT the
    bare ``dipropan-2-yl``). The fix must not disturb the already-correct case."""
    smi = "CC(C)[SH+]SC(C)C"
    mol = Chem.MolFromSmiles(smi)
    cat_idx = get_ion_sites(mol)["cations"][0]["atom_idx"]
    name = ions.emit_catenated_hydride_cation(mol, cat_idx)
    assert name == "1,2-di(propan-2-yl)disulfan-1-ium"
    assert "di(propan-2-yl)" in name and "bis(" not in name
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    assert opsin_roundtrip_check(smi, name).get("passed") is True


def test_simple_substituent_stays_unenclosed():
    """Non-regression: a SIMPLE prefix (methyl/chloro) is NEVER enclosed —
    ``1,2,3-trimethyltrisulfan-2-ium`` / ``1-chloro-2,3-dimethyltrisulfan-2-ium``."""
    for smi, expected in (("CS[S+](C)SC", "1,2,3-trimethyltrisulfan-2-ium"),
                          ("ClS[S+](C)SC", "1-chloro-2,3-dimethyltrisulfan-2-ium")):
        mol = Chem.MolFromSmiles(smi)
        cat_idx = get_ion_sites(mol)["cations"][0]["atom_idx"]
        name = ions.emit_catenated_hydride_cation(mol, cat_idx)
        assert name == expected
        assert "(" not in name  # no enclosing marks on simple prefixes


# ---------------------------------------------------------------------------
# Same count-1 defect in the sibling polyvalent radical emitter
# (``emit_parent_hydride_polyvalent_suffixes``, ions.py:~2324). Confirmed LIVE: it
# shipped ``2-4-methylcyclohexylpropane-1,3-diyl`` as ``pin_verified`` via the plain
# CLI. OPSIN 2.9.0 cannot parse a bare bivalent (``…-diyl``) name, so the RT gate is
# structurally blind to this family — the only guard is the spelling itself. Fixed by
# routing through ``enclose_if_compound`` (same primitive as the cation fix).
def _radical_centers(mol):
    return [(a.GetIdx(), a.GetNumRadicalElectrons())
            for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0]


POLYVALENT_COMPLEX = [
    ("[CH2]C([CH2])C1CCC(C)CC1", "2-(4-methylcyclohexyl)propane-1,3-diyl"),
    ("[CH2]C([CH2])c1ccccc1Cl", "2-(2-chlorophenyl)propane-1,3-diyl"),
    ("[CH2]C([CH2])c1ccc(C)cc1", "2-(4-methylphenyl)propane-1,3-diyl"),
]


@pytest.mark.parametrize("smiles,expected", POLYVALENT_COMPLEX)
def test_polyvalent_count1_complex_substituent_is_enclosed(smiles, expected):
    """A count-1 complex substituent on a polyvalent-radical parent is enclosed."""
    mol = Chem.MolFromSmiles(smiles)
    assert ions.emit_parent_hydride_polyvalent_suffixes(
        mol, _radical_centers(mol)) == expected


def test_polyvalent_simple_substituent_stays_unenclosed():
    """Non-regression: a SIMPLE ring/alkyl prefix on the radical parent stays bare."""
    for smi, expected in (
            ("[CH2]C([CH2])C1CCCCC1", "2-cyclohexylpropane-1,3-diyl"),
            ("[CH2]CC[CH2]", "butane-1,4-diyl")):
        mol = Chem.MolFromSmiles(smi)
        name = ions.emit_parent_hydride_polyvalent_suffixes(
            mol, _radical_centers(mol))
        assert name == expected
        assert "(" not in name
