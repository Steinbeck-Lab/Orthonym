"""Unit tests for the class-agnostic conjugate classifier (a phase, -03).

The conjugate controller (`rules/conjugate_controller.py`,) is a NEW standalone,
class-agnostic primitive: it classifies a sulfate / phosphate / glycosyl(uronyl)
fragment reached through a scaffold heteroatom linker and emits the functional-class
word/head, deriving the charge→word in place (, never neutralize-then-rename).

WAVE 0 CONTRACT: imports of the not-yet-built `conjugate_controller` symbols go
INSIDE each test body (NOT at module level) so `pytest --collect-only` succeeds while
the module does not yet exist; the finder/word/uronic tests are RED at run time until
Task 2 lands the module. `uronic_glycoside_head` is added to `data/sugar_names.py` in
Task 2, so `test_uronic_head` / `test_glucuronide_fragment_caps_to_catalog` flip GREEN
after Task 2.

Root-cause-only (CLAUDE.md): the asserted behaviour is structural — charge counting,
atom-level glycosidic-bond capping (no string surgery), explicit uronic head map.
"""

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Cohort SMILES (verified live from benchmark_multi_corpus_results.csv)
# ---------------------------------------------------------------------------
CHEBI_136579 = (
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H]"
    "(OS(=O)(=O)[O-])CC[C@]4(C)[C@H]3CC[C@]12C"
)  # cholest-5-en-3β-yl sulfate (anion -OSO2[O-])
CHEBI_133103 = (
    "C[C@]12CC[C@@H](O)C[C@@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)"
    "[C@@H](OS(=O)(=O)O)CC[C@@H]12"
)  # 3α-hydroxy-5α-androstan-17β-yl hydrogen sulfate (neutral -OSO2OH)
CHEBI_133504 = (
    "C[C@]12CC[C@H](O[C@@H]3O[C@H](C(=O)O)[C@@H](O)[C@H](O)[C@H]3O)"
    "C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12"
)  # 17-oxo-5β-androstan-3β-yl β-D-glucopyranosiduronic acid


def _build(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol


# ---------------------------------------------------------------------------
# Finders
# ---------------------------------------------------------------------------
def test_find_sulfate():
    """classify_conjugate detects the -OSO2[O-] fragment on CHEBI:136579.

    Live-verified substituent shape: attach=17, first=18(O),
    atoms={18(O),19(S),20(O),21(O),22(O-)} — the anion sulfate.
    """
    from orthonym.rules.conjugate_controller import classify_conjugate

    mol = _build(CHEBI_136579)
    from orthonym.perception.natural_products import (
        detect_natural_product,
        get_scaffold_substituents,
    )

    info = detect_natural_product(mol)
    assert info and info.get("scaffold_class") == "steroid"
    scaffold_atoms = set(info["matched_atoms"])
    sub = None
    for s in get_scaffold_substituents(mol, info["matched_atoms"]):
        if s["attachment_atom"] == 17:
            sub = s
            break
    assert sub is not None
    out = classify_conjugate(mol, sub["attachment_atom"], sub["first_atom"], scaffold_atoms)
    assert out is not None
    assert out["kind"] == "sulfate"
    assert out["word"] == "sulfate"
    assert out["all_atoms"] == {18, 19, 20, 21, 22}
    assert out["linker_kind"] == "O"


# ---------------------------------------------------------------------------
# Charge -> word (the load-bearing detail,)
# ---------------------------------------------------------------------------
def test_sulfate_word():
    """Anion -OSO2[O-] -> 'sulfate'; neutral -OSO2OH -> 'hydrogen sulfate'.

    Derived in place from the protonation state — never neutralize-then-rename.
    """
    from orthonym.rules.conjugate_controller import classify_conjugate

    # Minimal anion: methyl sulfate anion CO-S(=O)(=O)[O-]
    anion = _build("COS(=O)(=O)[O-]")
    # attach = the methyl C (idx 0), first = the linker O (idx 1)
    a_out = classify_conjugate(anion, 0, 1, {0})
    assert a_out is not None and a_out["kind"] == "sulfate"
    assert a_out["word"] == "sulfate"

    # Minimal neutral free acid: methyl hydrogen sulfate CO-S(=O)(=O)O
    neutral = _build("COS(=O)(=O)O")
    n_out = classify_conjugate(neutral, 0, 1, {0})
    assert n_out is not None and n_out["kind"] == "sulfate"
    assert n_out["word"] == "hydrogen sulfate"


def test_phosphate_word():
    """Phosphate three ionisation states :
    -OPO(OH)2 -> 'dihydrogen phosphate'; mono-anion -> 'hydrogen phosphate';
    di-anion -> 'phosphate'.
    """
    from orthonym.rules.conjugate_controller import classify_conjugate

    # neutral di-acid: methyl dihydrogen phosphate CO-P(=O)(O)(O)
    di_acid = _build("COP(=O)(O)O")
    out = classify_conjugate(di_acid, 0, 1, {0})
    assert out is not None and out["kind"] == "phosphate"
    assert out["word"] == "dihydrogen phosphate"

    # mono-anion: CO-P(=O)(O)([O-])
    mono = _build("COP(=O)(O)[O-]")
    out = classify_conjugate(mono, 0, 1, {0})
    assert out is not None and out["kind"] == "phosphate"
    assert out["word"] == "hydrogen phosphate"

    # di-anion: CO-P(=O)([O-])([O-])
    di = _build("COP(=O)([O-])[O-]")
    out = classify_conjugate(di, 0, 1, {0})
    assert out is not None and out["kind"] == "phosphate"
    assert out["word"] == "phosphate"


# ---------------------------------------------------------------------------
# Uronic glycoside head (the real gap,)
# ---------------------------------------------------------------------------
def test_uronic_head():
    """uronic_glycoside_head('β','D','glucuronopyranose')
    == 'β-D-glucopyranosiduronic acid' (NOT '...glucuronopyranoside').
    Unknown base -> None.
    """
    from orthonym.data.sugar_names import uronic_glycoside_head

    assert (
        uronic_glycoside_head("β", "D", "glucuronopyranose")
        == "β-D-glucopyranosiduronic acid"
    )
    assert uronic_glycoside_head("α", "D", "xyz") is None


# ---------------------------------------------------------------------------
# Fail-closed / completeness contract (full exercise in 182-02)
# ---------------------------------------------------------------------------
def test_completeness_honest_fail():
    """classify_conjugate is importable and fails closed for a non-conjugate
    fragment (a plain hydroxyl / methyl substituent returns None).

    The full no-silent-drop completeness invariant  lands in 182-02; here we
    assert the thin contract: the classifier returns None rather than guessing.
    """
    from orthonym.rules.conjugate_controller import classify_conjugate

    # A plain -OH on ethanol: attach=0(C), first=1(O), O carries an H (not a linker).
    mol = _build("CCO")
    assert classify_conjugate(mol, 1, 2, {0, 1}) is None

    # A plain methyl substituent: attach=0(C), first=1(C), no heteroatom linker.
    mol2 = _build("CC")
    assert classify_conjugate(mol2, 0, 1, {0}) is None


def test_rt_fallback():
    """The Phase-181 OPSIN RT gate `_alpha_beta_rt_ok` is the reused RT-check
    fallback . Contract only here; full behaviour wired in 182-02.
    """
    from orthonym.rules.natural_products import _alpha_beta_rt_ok  # noqa: F401

    assert callable(_alpha_beta_rt_ok)


# ---------------------------------------------------------------------------
# Blocker-1 acceptance: atom-level glycosidic-bond capping (RESEARCH Open Q2)
# ---------------------------------------------------------------------------
def test_glucuronide_fragment_caps_to_catalog():
    """The capped CHEBI:133504 sugar fragment canonicalizes to a URONIC_ACID_NAMES
    key (no string surgery) and resolves to the uronic head.

    Proves the root-cause capping resolution end-to-end:
      Chem.FragmentOnBonds + restore anomeric -OH + Chem.CanonSmiles
        -> URONIC_ACID_NAMES key
        -> lookup_sugar == ('β','D','glucuronopyranose')
        -> uronic_glycoside_head == 'β-D-glucopyranosiduronic acid'
    """
    from orthonym.rules.conjugate_controller import classify_conjugate  # noqa: F401
    from orthonym.data.sugar_names import (
        URONIC_ACID_NAMES,
        lookup_sugar,
        uronic_glycoside_head,
    )
    from orthonym.perception.natural_products import (
        detect_natural_product,
        get_scaffold_substituents,
    )

    mol = _build(CHEBI_133504)
    info = detect_natural_product(mol)
    assert info and info.get("scaffold_class") == "steroid"

    # Locate the glycosidic substituent: linker-O bonded to an anomeric ring-C.
    anomeric = linker_o = None
    for s in get_scaffold_substituents(mol, info["matched_atoms"]):
        first = s["first_atom"]
        fa = mol.GetAtomWithIdx(first)
        if fa.GetAtomicNum() != 8:
            continue
        for nbr in fa.GetNeighbors():
            if nbr.GetIdx() == s["attachment_atom"]:
                continue
            if nbr.GetAtomicNum() == 6 and nbr.IsInRing():
                # anomeric C also bonded to a ring-O
                if any(
                    rn.GetAtomicNum() == 8 and rn.IsInRing()
                    for rn in nbr.GetNeighbors()
                ):
                    anomeric, linker_o = nbr.GetIdx(), first
                    break
        if anomeric is not None:
            break
    assert anomeric is not None and linker_o is not None

    bond = mol.GetBondBetweenAtoms(anomeric, linker_o)
    frag = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(0, 0)])
    mapping = []
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping)
    sugar = None
    for fm, mp in zip(frags, mapping):
        if anomeric in mp:
            sugar = fm
            break
    assert sugar is not None
    rw = Chem.RWMol(sugar)
    for at in rw.GetAtoms():
        if at.GetAtomicNum() == 0:
            at.SetAtomicNum(8)
            at.SetNoImplicit(False)
            at.SetNumExplicitHs(1)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        Chem.SanitizeMol(
            rw, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
    clean = Chem.RemoveHs(rw)
    canon = Chem.MolToSmiles(clean)

    assert canon in URONIC_ACID_NAMES, canon
    tup = lookup_sugar(canon)
    assert tup == ("β", "D", "glucuronopyranose"), tup
    assert uronic_glycoside_head(*tup) == "β-D-glucopyranosiduronic acid"
