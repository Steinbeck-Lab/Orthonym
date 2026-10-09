"""Roadmap N5d: rings of ten or fewer members by their book names, at the writers.

* (the Blue Book): "Benzene is the retained name for C6H6";
  (:16290): 'phenyl' -- never 'cyclohexa-1,3,5-trien-1-yl'.
* (:8482): "Mancude and saturated heteromonocyclic compounds with up to and
  including ten ring members are named by the extended Hantzsch-Widman system";
   (:23682); 'azetidine (PIN)' (:8402); never '1-azacyclobutan-1-yl'.
* (:8284) and (:8318): numbering from the senior heteroatom,
  indicated hydrogen ('1H-imidazol-1-yl'; 'di(1H-imidazol-1-yl)methanethione (PIN)',
  :29544); hydro prefixes for a partly hydrogenated ring.
* (1) (:15813), (c) (:2913): 'cyclopropyl', 'bromobenzene'.

``rules.monocycle_forms`` builds the name and its numbering and audits the name's
heteroatom and indicated-hydrogen locants against the numbering; the writers
(``terminal_ring.terminal_ring_name``, the floor's ring spine, the enumerator's
monocycle tail) use it before the 'a'-replacement spelling.
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.book_prefixes import mechanical_forms
from orthonym.assembly.universal_substituent import (
    name_universal_substituent_prefix, name_universal_substitutive)
from orthonym.cli import _emit_tier_flags
from orthonym.rules.monocycle_forms import monocycle_form
from orthonym.rules.terminal_fragment import terminal_fragment_name
from orthonym.rules.terminal_ring import terminal_ring_name

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from name_quality import forms as F  # noqa: E402
from tests.support.rt_assert import name_is_rt_exact  # noqa: E402


def _ring_and_fv(smiles):
    """The ring of ``smiles`` and the ring atom bonded to its first atom (the probe
    group is always written first, e.g. 'OC' + ring)."""
    mol = Chem.MolFromSmiles(smiles)
    ring = next(r for r in mol.GetRingInfo().AtomRings()
                if any(mol.GetBondBetweenAtoms(1, a) for a in r))
    fv = next(a for a in ring if mol.GetBondBetweenAtoms(1, a))
    return mol, list(ring), fv


@pytest.mark.parametrize("smiles,parent,prefix", [
    ("OCc1ccccc1", "benzene", "phenyl"),
    ("OCC1CC1", "cyclopropane", "cyclopropyl"),
    ("OCN1CCC1", "azetidine", "azetidin-1-yl"),
    ("OCN1CCOCC1", "morpholine", "morpholin-4-yl"),
    ("OCn1ccnc1", "1H-imidazole", "1H-imidazol-1-yl"),
    ("OCc1ncc[nH]1", "1H-imidazole", "1H-imidazol-2-yl"),
    ("OCc1nncn1C", "4H-1,2,4-triazole", "4H-1,2,4-triazol-3-yl"),
    ("OCc1ncco1", "1,3-oxazole", "1,3-oxazol-2-yl"),
    ("OCC1=CC=COC1", "2H-pyran", "2H-pyran-3-yl"),
    ("OCC1CC=CCO1", "3,6-dihydro-2H-pyran", "3,6-dihydro-2H-pyran-2-yl"),
    ("OCN1C(=O)C=CC1=O", "2,5-dihydro-1H-pyrrole", "2,5-dihydro-1H-pyrrol-1-yl"),
    ("OCC1=CCCO1", "4,5-dihydrofuran", "4,5-dihydrofuran-2-yl"),     # (c) before (e)
])
def test_monocycle_form_names_and_prefixes(smiles, parent, prefix):
    mol, ring, fv = _ring_and_fv(smiles)
    branches = [a for a in ring for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                if nb.GetIdx() not in ring and nb.GetIdx() != 1]
    form = monocycle_form(mol, ring, fv, branches)
    assert form is not None
    assert (form.parent, form.prefix(fv)) == (parent, prefix)


@pytest.mark.parametrize("smiles", [
    "OCC1CCS(=O)(=O)C1",          # a lambda-6 ring atom
    "OCC1=CCCCC1",                # partly unsaturated carbocycle: 'cyclohex-1-en-1-yl'
    "OC[n+]1ccccc1",              # a ring cation
    "OCC1CCCCCCCCCC1",            # eleven members: 'a' names are the PINs (:8482)
])
def test_monocycle_form_declines_what_it_does_not_spell(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ring = max(mol.GetRingInfo().AtomRings(), key=len)
    fv = next(a for a in ring if mol.GetBondBetweenAtoms(1, a))
    assert monocycle_form(mol, list(ring), fv, []) is None


def test_terminal_ring_name_uses_the_book_name_and_its_numbering():
    mol = Chem.MolFromSmiles("Clc1ccc(cc1)C")
    ring = [1, 2, 3, 4, 5, 6]
    tr = terminal_ring_name(mol, ring, 4)
    assert tr.name == "phenyl" and tr.numbering[4] == 1 and tr.numbering[1] == 4
    with mechanical_forms():
        assert terminal_ring_name(mol, ring, 4).name == "cyclohexa-1,3,5-trien-1-yl"


@pytest.mark.parametrize("smiles,frag,attach,expected", [
    ("Clc1ccc(cc1)C", [0, 1, 2, 3, 4, 5, 6], 4, "4-chlorophenyl"),
    ("Fc1c(F)c(F)c(c(F)c1F)C", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 6, "pentafluorophenyl"),
    ("CN1CCC1C", [0, 1, 2, 3, 4], 4, "1-methylazetidin-2-yl"),
])
def test_terminal_fragment_ring_core(smiles, frag, attach, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert terminal_fragment_name(mol, set(frag), attach).name == expected


@pytest.mark.parametrize("smiles,expected", [
    ("c1ccccc1C(F)(F)F", "(trifluoromethyl)benzene"),
    ("Brc1ccccc1", "bromobenzene"),
    ("Clc1ccc(Cl)cc1", "1,4-dichlorobenzene"),
    ("C1=CCOC(C1)Cc1ccccc1", "[(3,6-dihydro-2H-pyran-2-yl)methyl]benzene"),
])
def test_floor_ring_parents(smiles, expected):
    assert name_universal_substitutive(Chem.MolFromSmiles(smiles)).name == expected


def test_floor_ring_branch():
    mol = Chem.MolFromSmiles("CCc1ccncc1")
    assert name_universal_substituent_prefix(mol, [2, 3, 4, 5, 6, 7], 2) == "pyridin-4-yl"


def _row(smiles, tier):
    if tier == "pin":
        return Orthonym().name_tiered(smiles)
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


#: rows whose name carried a BENZ / MONO / ANYL form at the base (shared baseline)
E2E = [
    ("CC1=CC(=O)OC1CC(=O)[O-]", "valid", "2,5-dihydrofuran-2-yl"),
    ("C1CCN(CC1)C2=CCCNC2=O", "valid", "tetrahydropyridine"),
    ("C=C[C@@H](/C=C\\c1ccc(O)cc1)c1ccc(O)cc1", "valid", "phenyl"),
    ("Cc1cc(O)cc(O)c1C(=O)OC1=COC(C)C(O)C1=O", "best-effort", "3,4-dihydro-2H-pyran-5-yl"),
    ("C1=CC(C(C(=C1)C(=O)O)C(=O)O)(N2C(=O)C=CC2=O)N3C(=O)C=CC3=O", "best-effort",
     "2,5-dihydro-1H-pyrrol-1-yl"),
    ("O=C(NCc1ccccn1)c1ccc(Oc2ccccc2)cc1", "best-effort", "phen"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,tier,ring_text", E2E)
def test_rings_by_book_names_end_to_end(smiles, tier, ring_text):
    row = _row(smiles, tier)
    name = row.get("name")
    assert name and name_is_rt_exact(name, smiles), row
    forms = F.detect(name)
    assert not {"BENZ", "MONO", "ANYL"} & set(forms), (name, forms)
    assert ring_text in name, name


# --- the writer's reach does not grow (Review Focus 6) -------------------------------------
# ``name_substituent`` calls the terminal-fragment writer before the chain composer, which
# names a decorated chain with the recursive prefixes and numbering. A ring the
# writer's replacement spelling declined (an aromatic 4H-1,2,4-triazol-4-yl) but the book
# names must not let the writer take the fragment: the composer's name is the better one
# ('3-methyl-5-(propan-2-yl)-4H-1,2,4-triazol-4-yl'; (g), the Blue Book, gives
# the methyl group, cited first, the lower locant).

def test_a_book_spelling_never_widens_the_terminal_fragment_writer():
    mol = Chem.MolFromSmiles("Cn1c(C)nnc1C(C)C")
    frag = set(range(mol.GetNumAtoms()))
    with mechanical_forms():
        assert terminal_fragment_name(mol, frag, 0) is None
    assert terminal_fragment_name(mol, frag, 0) is None


def test_a_fragment_the_writer_names_takes_the_book_spelling():
    mol = Chem.MolFromSmiles("Cn1ccnc1")
    frag = set(range(mol.GetNumAtoms()))
    with mechanical_forms():
        assert terminal_fragment_name(mol, frag, 0).name == (
            "1-(1,3-diazacyclopenta-2,4-dien-1-yl)methyl")
    assert terminal_fragment_name(mol, frag, 0).name == "(1H-imidazol-1-yl)methyl"


# (the Blue Book): "Locants are omitted when no isomer can be generated";
# '1H-tetrazole (PIN) (not 1H-1,2,3,4-tetrazole)' (:2989).
@pytest.mark.parametrize("smiles,parent,prefix", [
    ("Cn1nnnc1", "1H-tetrazole", "1H-tetrazol-1-yl"),
    ("Cc1nnn[nH]1", "1H-tetrazole", "1H-tetrazol-5-yl"),
    ("Cc1nn[nH]n1", "2H-tetrazole", "2H-tetrazol-5-yl"),
    ("Cn1nncn1", "2H-tetrazole", "2H-tetrazol-2-yl"),
])
def test_tetrazole_cites_no_heteroatom_locants(smiles, parent, prefix):
    mol = Chem.MolFromSmiles(smiles)
    ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    form = monocycle_form(mol, ring, 1)
    assert form is not None
    assert (form.parent, form.prefix(1)) == (parent, prefix)


# The same rule in the shared heteromonocycle namer: its retained table held the 1H
# tautomer only, so the 2H tautomer (and a ring substituted on N2) came out
# '2H-1,2,3,4-tetrazole' -- at the PIN tier too, labelled pin_verified on main.
@pytest.mark.parametrize("smiles,name", [
    ("c1nn[nH]n1", "2H-tetrazole"),
    ("c1nnn[nH]1", "1H-tetrazole"),
])
def test_name_heterocycle_cites_no_tetrazole_heteroatom_locants(smiles, name):
    from orthonym.rules.heterocycles import name_heterocycle
    assert name_heterocycle(Chem.MolFromSmiles(smiles), tuple(range(5))) == name


@pytest.mark.opsin_gate
def test_the_pin_tier_names_2h_tetrazole_without_heteroatom_locants():
    row = _row("c1nn[nH]n1", "pin")
    assert (row.get("name"), row["tier"]) == ("2H-tetrazole", "pin_verified"), row
    assert name_is_rt_exact(row["name"], "c1nn[nH]n1")


# (the Blue Book): "When there are an equal number of indicated
# hydrogen atoms and... free valences to be accommodated, the indicated hydrogen atoms
# are placed at peripheral atoms that will accommodate these... free valences"; a free
# valence on an atom with no hydrogen in the mancude parent (a ring N with two ring
# bonds) takes the indicated hydrogen. A ring carbon keeps one hydrogen in the mancude
# parent, so the lowest locant takes it: '3,4-dihydro-2H-pyran-3-yl' (:17317).
@pytest.mark.parametrize("smiles,prefix", [
    ("OCN1CNN=C1", "1,5-dihydro-4H-1,2,4-triazol-4-yl"),
    ("OCN1CC=NN1", "2,5-dihydro-1H-1,2,3-triazol-1-yl"),
    ("OCC1COC=CC1", "3,4-dihydro-2H-pyran-3-yl"),
])
def test_the_free_valence_atom_takes_the_indicated_hydrogen(smiles, prefix):
    mol, ring, fv = _ring_and_fv(smiles)
    form = monocycle_form(mol, ring, fv)
    assert form is not None and form.prefix(fv) == prefix, form


# (the Blue Book): a ring whose own name cites locants keeps the locants of
# its prefixes even when every substitutable position carries the same prefix; a ring
# without them omits all,:3007).
@pytest.mark.parametrize("smiles,name", [
    ("Cc1nnc(C)o1", "2,5-dimethyl-1,3,4-oxadiazole"),
    ("Cc1nnc(C)n1C", "3,4,5-trimethyl-4H-1,2,4-triazole"),
    ("Fc1c(F)c(F)c(F)c(F)c1F", "hexafluorobenzene"),
    ("Clc1nc(Cl)c(Cl)c(Cl)c1Cl", "pentachloropyridine"),
])
def test_floor_cites_the_locants_a_ring_name_needs(smiles, name):
    res = name_universal_substitutive(Chem.MolFromSmiles(smiles))
    assert res is not None and res.name == name, res
