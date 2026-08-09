"""v30 B2 — unsaturated hydroxy-alkenyl substituent prefix (lignin monomers).

Two coupled fixes let a phenol-parent ring carry an unsaturated hydroxy-alkenyl
substituent at PIN:

  bug 1  ``substituent_naming._name_polyfunctional_acyclic_substituent`` now builds
         an unsaturated chain substituent (``(1E)-3-hydroxyprop-1-en-1-yl``),
         numbered from the free valence (P-29.2), ene locant + E/Z from structure.
  bug 2  ``benzene._identify_alkyl_group`` no longer mislabels a vinyl-STARTED
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
    prefix from structure, numbered from the free valence (P-29.2)."""
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


# ---- fable-b2 review findings (all resolved) ----

@pytest.mark.parametrize("smi", [
    "CO/C=C/c1ccc(O)cc1",             # enol ether -CH=CH-OMe
    "O=[N+]([O-])/C=C/c1ccc(O)cc1",   # nitrovinyl
    "CS/C=C/c1ccc(O)cc1",             # vinyl thioether
])
def test_blocker1_stereo_completeness_abstains(smi):
    """fable-b2 BLOCKER 1 / invariant 9: a substituent with a DEFINED-stereo C=C
    whose name would drop the E/Z descriptor (legacy `Xethenyl` tier) must NOT ship
    (the gate's stereo carve-out would pass a wrong-molecule name). The stereo-
    completeness guard declines -> abstain (0-wrong), never a `...ethenylphenol`."""
    n = _pin().name(smi) or ""
    assert "ethenylphenol" not in n and n != "", None  # abstains, not the wrong name
    assert n == "unknown organic compound", n


def test_blocker2_benzonitrile_brackets_complex_substituent():
    """fable-b2 BLOCKER 2: a monosubstituted benzonitrile must bracket a complex
    substituent (P-16.5.2.4), not ship the markless `4-(1E)-...ylbenzonitrile`."""
    assert _pin().name("N#Cc1ccc(/C=C/CO)cc1") == "4-[(1E)-3-hydroxyprop-1-en-1-yl]benzonitrile"
    # simple substituents unchanged
    assert _pin().name("N#Cc1ccc(Cl)cc1") == "4-chlorobenzonitrile"
    assert _pin().name("N#Cc1ccccc1") == "benzonitrile"


@pytest.mark.parametrize("smi,expected", [
    # fable-b2 RISK 3: a chain R/S centre with an UNDEFINED-geometry core C=C names
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
