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
