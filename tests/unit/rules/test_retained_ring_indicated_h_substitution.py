"""A retained heteromonocycle keeps its retained name when its ring N is substituted.

THE DEFECT
----------
``Cn1cccc1`` was named ``1-methylazole`` and ``Cn1ccnc1`` ``1-methyl-1,3-diazole``.
``azole`` is the Hantzsch-Widman systematic form; ``1,3-diazole`` likewise.  Both rings
have *retained* names that are PINs, and substituting the ring nitrogen does not demote
them.

THE RULE
--------
* **P-15.1.8.1** "Substitution rules for Type 1 retained names"
  (``BlueBookV2/BlueBookV2.md:4916``), sentence ``:4918``: "Type 1 retained names of parent
  hydrides described in Chapters P-2 and P-3 have **unlimited substitution** by substituent
  groups cited either as suffixes or prefixes."  Substitution never demotes a retained
  parent hydride to its systematic form.
* **P-22.2.1** "Retained names of heteromonocycles" (``:8109``) prints, verbatim:
  ``pyrrole (1H-isomer shown; the PIN is 1H-pyrrole)`` (``:8163``),
  ``imidazole (1H-isomer shown; the PIN is 1H-imidazole)`` (``:8131``) and
  ``pyrazole (1H-isomer shown; the PIN is 1H-pyrazole)`` (``:8154``).
* **P-14.3.4.2** "The locant '1' is omitted:" (``:2891``) prints
  ``1H-tetrazole (PIN) (not 1H-1,2,3,4-tetrazole)`` (``:2989``) -- the HW form this code
  used to emit for ``Cn1cnnn1`` is named there as the form to avoid.
* **P-14.7.1** "Indicated hydrogen" (``:3721``): "in a preferred IUPAC name a locant and
  the symbol '*H*' must be cited" -- so the indicated hydrogen survives into the
  substituted name.
* The Blue Book prints an N-substituted azole PIN outright, under **P-44.1** "SENIORITY
  ORDER FOR PARENT STRUCTURES" (``:18871``), at ``:18940``::

      1-(trimethylsilyl)-1H-imidazole (PIN) (N is senior to Si)

  Retained name kept, indicated hydrogen kept, substituent at N1.

THE ROOT CAUSE
--------------
``get_ring_canonical_smiles`` extracts the ring with ``Chem.MolFragmentToSmiles`` and looks
the string up in an exact-SMILES table.  An N-substituted aromatic azole's ring nitrogen
carries **zero** hydrogens, so the fragment comes back with a bare aromatic ``n``
(``c1ccnc1``), which is not a kekulisable molecule at all.  It can never equal the table key
``c1cc[nH]c1``, and the lookup falls through to ``build_hw_name``.

The substituent had displaced the ring's *indicated hydrogen*; the extraction has to put it
back.  Two properties of the repair are load-bearing and are asserted below:

1. **Strictly additive.**  It only runs when the plain fragment is not a parseable molecule
   -- i.e. only for keys that match nothing today.  Every ring whose fragment already
   parses keeps its exact current key (``test_additive_*``).
2. **Never crosses a fusion bond.**  Indole contains a pyrrole ring but is its own retained
   name; ``1-methyl-1H-indole`` must not become a substituted pyrrole
   (``test_fused_*``).
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.heterocycles import get_ring_canonical_smiles, name_heterocycle


def _mol_ring(smiles: str, size: int | None = None):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    rings = [r for r in mol.GetRingInfo().AtomRings()
             if size is None or len(r) == size]
    assert rings, f"no ring of size {size} in {smiles}"
    return mol, rings[0]


# --------------------------------------------------------------------------
# 1. The ring key -- the displaced indicated hydrogen is restored
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected_key", [
    ("Cn1cccc1", "c1cc[nH]c1"),          # 1-methylpyrrole
    ("CCn1cccc1", "c1cc[nH]c1"),         # 1-ethylpyrrole
    ("c1ccc(-n2cccc2)cc1", "c1cc[nH]c1"),  # 1-phenylpyrrole
    ("Cn1ccnc1", "c1c[nH]cn1"),          # 1-methylimidazole
    ("Cn1cccn1", "c1cn[nH]c1"),          # 1-methylpyrazole
    ("Cn1cnnn1", "c1nnn[nH]1"),          # 1-methyltetrazole
])
def test_n_substituted_azole_ring_key_is_its_parent_hydride(smiles, expected_key):
    mol, ring = _mol_ring(smiles, size=5)
    assert get_ring_canonical_smiles(mol, ring) == expected_key


@pytest.mark.parametrize("smiles,expected", [
    ("Cn1cccc1", "1H-pyrrole"),
    ("Cn1ccnc1", "1H-imidazole"),
    ("Cn1cccn1", "1H-pyrazole"),
    ("Cn1cnnn1", "1H-tetrazole"),
])
def test_n_substituted_azole_ring_resolves_to_the_retained_pin(smiles, expected):
    mol, ring = _mol_ring(smiles, size=5)
    assert name_heterocycle(mol, ring) == expected


# --------------------------------------------------------------------------
# 2. End to end
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("Cn1cccc1", "1-methyl-1H-pyrrole"),
    ("Cn1ccnc1", "1-methyl-1H-imidazole"),
    ("Cn1cccn1", "1-methyl-1H-pyrazole"),
    ("CCn1cccc1", "1-ethyl-1H-pyrrole"),
])
def test_end_to_end_name(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,historical_wrong_name", [
    # Measured at fa9e4687, the commit this fix is based on.  ``azole`` is not a
    # name at all; ``1,2,3,4-tetrazole`` is the form P-14.3.4.2 (``:2989``)
    # prints as ``(not ...)``; the diazoles are the HW forms of retained PINs.
    ("Cn1cccc1", "1-methylazole"),
    ("Cn1ccnc1", "1-methyl-1,3-diazole"),
    ("Cn1cccn1", "1-methyl-1,2-diazole"),
    ("Cn1cnnn1", "1-methyl-1,2,3,4-tetrazole"),
    ("CCn1cccc1", "1-ethylazole"),
    ("c1ccc(-n2cccc2)cc1", "1-phenylazole"),
    ("OC(=O)c1cccn1C", "1-methylazole-2-carboxylic acid"),
])
def test_the_hw_systematic_form_is_gone(smiles, historical_wrong_name):
    assert name_compound(smiles) != historical_wrong_name


# --------------------------------------------------------------------------
# 2b. The SAME root cause one layer up: the numbering.
#
# ``orient_heterocycle_with_substituents`` decided which ring atom carries the
# indicated hydrogen with ``GetTotalNumHs() >= 1`` -- an atom count the
# substituent has already consumed.  An N-substituted azole nitrogen therefore
# read as pyridine-type, lost its claim on locant 1, and the suffix took it
# instead: ``Cn1nccc1C(=O)O`` came out ``2-methyl-1H-pyrazole-3-carboxylic
# acid``, whose ``1H`` and whose ``2-methyl`` contradict each other.
#
# P-14.4 "NUMBERING" (``BlueBookV2.md:3219``) assigns low locants "in the
# following decreasing order of seniority", and prints
# **(b) indicated hydrogen** ahead of **(c) principal characteristic groups and
# free valences (suffixes)**.  (b)'s own caveat -- "a higher locant may be
# needed at another position to accommodate a substituent suffix in accordance
# with structural feature (d)" -- points at (d) *added* indicated hydrogen
# (``3,4-dihydronaphthalen-1(2H)-one``), not at ordinary substitution.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("Cn1nccc1C(=O)O", "1-methyl-1H-pyrazole-5-carboxylic acid"),
    ("Cn1ncc(C(=O)O)c1", "1-methyl-1H-pyrazole-4-carboxylic acid"),
    ("Cn1cccc1C(=O)O", "1-methyl-1H-pyrrole-2-carboxylic acid"),
    ("Cn1ccnc1C(=O)O", "1-methyl-1H-imidazole-2-carboxylic acid"),
    ("OC(=O)c1cccn1C", "1-methyl-1H-pyrrole-2-carboxylic acid"),
])
def test_indicated_hydrogen_outranks_the_suffix_in_numbering(smiles, expected):
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------
# 3. Fused controls -- a ring that merely CONTAINS the pattern is a different
#    ring system.  The repair must not cross a fusion bond.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    "Cn1ccc2ccccc21",             # 1-methyl-1H-indole
    "Cn1cnc2ccccc21",             # 1-methyl-1H-benzimidazole
    "Cn1c2ccccc2c2ccccc21",       # 9-methyl-9H-carbazole
])
def test_fused_five_ring_never_acquires_a_monocycle_retained_name(smiles):
    mol, ring = _mol_ring(smiles, size=5)
    key = get_ring_canonical_smiles(mol, ring)
    assert key not in ("c1cc[nH]c1", "c1c[nH]cn1", "c1cn[nH]c1", "C1CCNC1"), \
        f"{smiles}: fused 5-ring keyed as a monocycle ({key})"
    assert name_heterocycle(mol, ring) not in ("1H-pyrrole", "1H-imidazole",
                                               "pyrrolidine")


@pytest.mark.parametrize("smiles,expected", [
    ("c1ccc2[nH]ccc2c1", "1H-indole"),
    ("Cn1ccc2ccccc21", "1-methyl-1H-indole"),
    ("c1ccc2occc2c1", "benzofuran"),
    ("c1ccc2[nH]c3ccccc3c2c1", "9H-carbazole"),
    ("c1ccc2[nH]cnc2c1", "1H-benzimidazole"),
    ("Cn1cnc2ccccc21", "1-methyl-1H-benzimidazole"),
    ("c1ccc2ncccc2c1", "quinoline"),
])
def test_fused_names_unchanged(smiles, expected):
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------
# 4. Additivity -- every ring whose plain fragment already parses keeps its
#    exact current key, and every ring the repair cannot prove stays unmatched.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,size,expected_key", [
    ("c1cc[nH]c1", 5, "c1cc[nH]c1"),      # bare pyrrole
    ("Cc1cc[nH]c1", 5, "c1cc[nH]c1"),     # C-substituted pyrrole
    ("Cc1ccoc1", 5, "c1ccoc1"),           # C-substituted furan
    ("Cc1ccsc1", 5, "c1ccsc1"),
    ("c1ccncc1", 6, "c1ccncc1"),          # pyridine
    ("CN1CCCC1", 5, "C1CCNC1"),           # N-substituted SATURATED ring
    ("CN1CCOCC1", 6, "C1COCCN1"),
    ("O=C1CCCN1", 5, "C1CCNC1"),          # exocyclic C=O on a saturated ring
    ("C[C@H]1CCCO1", 5, "C1CCOC1"),       # ring stereocentre lost with its substituent
])
def test_additive_parseable_fragments_keep_their_exact_key(smiles, size, expected_key):
    mol, ring = _mol_ring(smiles, size=size)
    assert get_ring_canonical_smiles(mol, ring) == expected_key


@pytest.mark.parametrize("smiles,size,expected_key", [
    # An exocyclic C=O -- not a displaced indicated hydrogen -- so the repair
    # declines and the raw (unmatchable) fragment is returned exactly as today.
    ("O=c1cccc[nH]1", 6, "c1cc[nH]cc1"),     # pyridin-2(1H)-one
    ("Nc1cc[nH]c(=O)n1", 6, "c1cnc[nH]c1"),  # cytosine's ring
])
def test_additive_unrepairable_fragments_are_returned_unchanged(smiles, size,
                                                                expected_key):
    mol, ring = _mol_ring(smiles, size=size)
    assert get_ring_canonical_smiles(mol, ring) == expected_key


@pytest.mark.parametrize("smiles,expected", [
    ("Cc1cc[nH]c1", "3-methyl-1H-pyrrole"),
    ("Cc1c[nH]cn1", "4-methyl-1H-imidazole"),
    ("Cc1ccoc1", "3-methylfuran"),
    ("Cc1ccsc1", "3-methylthiophene"),
    ("CN1CCCC1", "1-methylpyrrolidine"),
    ("CN1CCOCC1", "4-methylmorpholine"),
    ("c1cc[nH]c1", "1H-pyrrole"),
    ("c1ccoc1", "furan"),
    ("c1ccncc1", "pyridine"),
])
def test_additive_names_unchanged(smiles, expected):
    assert name_compound(smiles) == expected
