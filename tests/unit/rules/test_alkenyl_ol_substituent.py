""" B2 — unsaturated hydroxy-alkenyl substituent prefix (lignin monomers).

Two coupled fixes let a phenol-parent ring carry an unsaturated hydroxy-alkenyl
substituent at PIN:

  bug 1 ``substituent_naming._name_polyfunctional_acyclic_substituent`` now builds
         an unsaturated chain substituent (``(1E)-3-hydroxyprop-1-en-1-yl``),
         numbered from the free valence, ene locant + E/Z from structure.
  bug 2 ``benzene._identify_alkyl_group`` no longer mislabels a vinyl-STARTED
         hetero-bearing fragment (``-CH=CH-CH2OH``) as ``ethenyl`` (which dropped
         the CH2OH tail -> ``4-ethenylphenol``); it declines so the recursive
         ``name_substituent`` fallback names it correctly.

Targets VERIFIED RT-exact in OPSIN (p-coumaryl / sinapyl alcohol, lignin monomers).
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.unit


def _pin():
    return Orthonym(style="pin")


@pytest.mark.parametrize("smi,expected", [
    # p-coumaryl alcohol
    ("OC/C=C/c1ccc(O)cc1", "4-[(1E)-3-hydroxyprop-1-en-1-yl]phenol"),
    # sinapyl alcohol (2,6-dimethoxy)
    ("COc1cc(/C=C/CO)cc(OC)c1O",
     "4-[(1E)-3-hydroxyprop-1-en-1-yl]-2,6-dimethoxyphenol"),
])
def test_lignin_monomer_alkenyl_ol_substituent(smi, expected):
    assert _pin().name(smi) == expected


def test_alkenyl_ol_substituent_unit():
    """The recursive substituent namer builds the unsaturated hydroxy-alkenyl
    prefix from structure, numbered from the free valence."""
    from rdkit import Chem
    from orthonym.assembly.substituent_enumerator import name_substituent
    m = Chem.MolFromSmiles("OC/C=C/c1ccccc1")  # -CH=CH-CH2OH on benzene
    attach = next(a.GetIdx() for a in m.GetAtoms()
                  if a.GetSymbol() == "C" and not a.GetIsAromatic()
                  and any(n.GetIsAromatic() for n in a.GetNeighbors()))
    frag = set()
    stack = [attach]
    while stack:
        x = stack.pop()
        if x in frag:
            continue
        at = m.GetAtomWithIdx(x)
        if at.GetIsAromatic():
            continue
        frag.add(x)
        for n in at.GetNeighbors():
            if not n.GetIsAromatic():
                stack.append(n.GetIdx())
    assert name_substituent(m, sorted(frag), attach) == "(1E)-3-hydroxyprop-1-en-1-yl"


@pytest.mark.parametrize("smi,expected", [
    # PURE vinyl on a ring must stay byte-identical (named via the fallback, NOT the
    # removed ethenyl special-case) -- the atom-drop guard must not regress these.
    ("C=Cc1ccccc1", "ethenylbenzene"),
    ("C=Cc1ccc(O)cc1", "4-ethenylphenol"),
    # chain-parent (single -OH on the chain) is unaffected by the ring-path fix.
    ("OC/C=C/c1ccccc1", "(2E)-3-phenylprop-2-en-1-ol"),
    # plain alkyl / retained arene substituents unchanged.
    ("Cc1ccccc1", "toluene"),
    ("CCc1ccccc1", "ethylbenzene"),
    ("COc1cc(/C=C/C)ccc1O", "2-methoxy-4-[(1E)-prop-1-en-1-yl]phenol"),  # isoeugenol
])
def test_no_regression_vinyl_and_alkyl(smi, expected):
    assert _pin().name(smi) == expected


# ---- a review-b2 review findings (all resolved) ----

@pytest.mark.parametrize("smi", [
    "CO/C=C/c1ccc(O)cc1",             # enol ether -CH=CH-OMe
    "O=[N+]([O-])/C=C/c1ccc(O)cc1",   # nitrovinyl
    "CS/C=C/c1ccc(O)cc1",             # vinyl thioether
])
def test_blocker1_stereo_completeness_never_drops_ez(smi):
    """fable-b2 BLOCKER 1 / invariant 9: a substituent with a DEFINED-stereo C=C
    must NEVER ship a name that DROPS the E/Z descriptor (the legacy `Xethenyl`
    tier — a wrong-molecule name). The invariant is 0-wrong, and it is now met two
    ways: an abstention (`unknown organic compound`) OR a stereo-COMPLETE name that
    round-trips with full stereo. v33 Phase 6 (E3): the FindMolChiralCenters
    shared-mol `_CIPCode`-wipe fix lets the nitrovinyl now name CORRECTLY WITH its
    (1E) descriptor (`4-[(1E)-2-nitroeth-1-en-1-yl]phenol`, RT-full) instead of
    abstaining; the enol-ether / vinyl-thioether still abstain (their substituent
    prefix isn't built). Either outcome is 0-wrong; a descriptor-less `ethenylphenol`
    or a bare `…enyl…phenol` with no E/Z is the forbidden case."""
    from rdkit import Chem
    from rdkit.Chem import inchi
    from orthonym.validation.opsin_roundtrip import opsin_parse
    n = _pin().name(smi) or ""
    # the forbidden stereo-dropped forms
    assert "ethenylphenol" not in n
    named = n and "unknown" not in n
    if named:
        # if it ships a name for a defined-stereo C=C, it MUST carry the descriptor
        # AND round-trip with full stereo (0-wrong)
        assert any(d in n for d in ("(E)", "(Z)", "1E", "1Z", "2E", "2Z")), n
        o = opsin_parse(n)
        assert o and inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) == \
            inchi.MolToInchiKey(Chem.MolFromSmiles(o)), n


@pytest.mark.opsin_gate
def test_blocker1_nitrovinyl_now_names_with_stereo():
    """The nitrovinyl witness specifically now names to the exact stereo-complete
    PIN (a phase E3 stereo-wipe fix)."""
    from rdkit import Chem
    from rdkit.Chem import inchi
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smi = "O=[N+]([O-])/C=C/c1ccc(O)cc1"
    n = _pin().name(smi)
    assert n == "4-[(1E)-2-nitroeth-1-en-1-yl]phenol", n
    o = opsin_parse(n)
    assert o and inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(o))


def test_blocker2_benzonitrile_brackets_complex_substituent():
    """a review-b2 BLOCKER 2: a monosubstituted benzonitrile must bracket a complex
    substituent, not ship the markless `4-(1E)-...ylbenzonitrile`."""
    assert _pin().name("N#Cc1ccc(/C=C/CO)cc1") == "4-[(1E)-3-hydroxyprop-1-en-1-yl]benzonitrile"
    # simple substituents unchanged
    assert _pin().name("N#Cc1ccc(Cl)cc1") == "4-chlorobenzonitrile"
    assert _pin().name("N#Cc1ccccc1") == "benzonitrile"


@pytest.mark.parametrize("smi,expected", [
    # a review-b2 RISK 3: a chain R/S centre with an UNDEFINED-geometry core C=C names
    # (the descriptor-less name lets _stereo_route add the R/S).
    ("C[C@H](O)C=Cc1ccc(O)cc1", "4-[(S)-3-hydroxybut-1-en-1-yl]phenol"),
])
def test_risk3_geometryless_rs_names(smi, expected):
    assert _pin().name(smi) == expected


def test_risk3_ez_plus_rs_still_abstains():
    """When BOTH an E/Z descriptor and an R/S centre are present they can't be
    merged (the _stereo_route double-apply guard drops the R/S) -> fail closed."""
    n = _pin().name("C[C@H](O)/C=C/c1ccc(O)cc1") or ""
    assert "but-1-en" not in n, n  # abstains rather than ship stereo-incomplete


# ---- polyene substituent extension (B2 follow-on) ----

@pytest.mark.parametrize("smi,expected", [
    # dienyl-ol arm on phenol: euphonic-'a' multiplied ene stem + merged (nE,nE) block
    ("OC/C=C/C=C/c1ccc(O)cc1", "4-[(1E,3E)-5-hydroxypenta-1,3-dien-1-yl]phenol"),
    ("OC/C=C\\C=C/c1ccc(O)cc1", "4-[(1Z,3Z)-5-hydroxypenta-1,3-dien-1-yl]phenol"),
])
def test_polyene_hydroxy_alkenyl_substituent(smi, expected):
    assert _pin().name(smi) == expected


def test_polyene_rt_exact():
    """Both diene geometries round-trip (0-wrong)."""
    import sys
    sys.path.insert(0, "scripts")
    from diagnose import diagnose
    rows = diagnose(["OC/C=C/C=C/c1ccc(O)cc1", "OC/C=C\\C=C/c1ccc(O)cc1"],
                    style="pin", use_opsin=True)
    assert all(r.get("verdict") == "OK" for r in rows), [
        (r["smiles"], r.get("verdict"), r.get("name")) for r in rows]


# ---- a review-polyene review BLOCKERs, resolved ----

def test_polyene_blocker2_two_stereocentres_fail_closed():
    """a review-polyene BLOCKER 2 / a project rule: >=2 chain R/S centres + undefined C=C
    geometry would ship a stereo-DROPPED wrong molecule (_stereo_route drops multi-
    centre R/S). Must fail closed (abstain), not emit the stereo-bare name."""
    import sys
    sys.path.insert(0, "scripts")
    from diagnose import diagnose
    smis = ["OC(=O)c1ccc(C=CC=C[C@@H](O)[C@@H](O)CO)cc1",
            "OC(=O)c1ccc(C=CC=C[C@@H](N)[C@@H](O)CO)cc1"]
    rows = diagnose(smis, style="pin", use_opsin=True)
    for r in rows:
        assert r.get("verdict") != "RT-MISMATCH", (r["smiles"], r.get("name"))
    # single centre + undefined C=C still names (RT-OK)
    assert _pin().name("OC(=O)c1ccc(C=CC=C[C@@H](O)CCO)cc1") == \
        "4-[(S)-5,7-dihydroxyhepta-1,3-dien-1-yl]benzoic acid"


@pytest.mark.parametrize("name,count,want", [
    # a review-polyene BLOCKER 1 / (a): a SUBSTITUTED polyene prefix takes bis
    ("5-hydroxypenta-1,3-dien-1-yl", 2, "bis"),
    ("6-hydroxyhexa-2,4-dien-1-yl", 2, "bis"),
    # (b): the UNSUBSTITUTED polyene keeps di (the trap — must not flip)
    ("penta-1,3-dien-1-yl", 2, "di"),
    ("hexa-2,4-dien-1-yl", 2, "di"),
    # controls unchanged
    ("methyl", 2, "di"), ("naphthalen-2-yl", 2, "di"),
    ("bromomethyl", 2, "bis"), ("propan-2-yl", 2, "di"),
])
def test_polyene_blocker1_di_vs_bis(name, count, want):
    from orthonym.assembly.naming_utils import get_multiplier_prefix
    assert get_multiplier_prefix(count, name) == want


def test_polyene_blocker1_full_molecule_bis():
    """The symmetric bis-dienol arene names with bis (a)); the bare-polyene
    analogue keeps di (b))."""
    assert _pin().name("OC(=O)c1cc(C=CC=CCO)cc(C=CC=CCO)c1") == \
        "3,5-bis(5-hydroxypenta-1,3-dien-1-yl)benzoic acid"
    assert _pin().name("OC(=O)c1cc(/C=C/C=C/C)cc(/C=C/C=C/C)c1") == \
        "3,5-di[(1E,3E)-penta-1,3-dien-1-yl]benzoic acid"
