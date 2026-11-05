"""A pseudoester polyacid has one name in every atom order (lane/l17 item 9, principle 4: same molecule, same name).

The row of the lead's finding, ``bis(tributyltin) 2-methylidenebutanedioate``, was named
``tributylstannyl 3-{oxo[(tributylstannyl)oxy]methyl}but-3-enoate`` in one SMILES and
``2-methylidene-1,4-dioxo-1,4-bis[(tributylstannyl)oxy]butane`` in another. Both names were read back and labelled
``systematic_verified``, so no gate saw the pair. Two causes met in ``rules.esters.name_noncarbon_ester``: the producer cut
ONE acid group, the first match of the perception, so the atom order decided which, and a bracket atom of a random SMILES
(``[O]``, ``[CH2]``) kept no hydrogen on the cut, so the free acid was unnameable and the general engine named the
spelling. lane/l16 b3 cut every group of one polyacid, in the order of the canonical ranks; this file pins
the class on more rows and more spellings than that commit's test (other seeds, other acids, other metals).

 "Polyesters derived from a single acid component", (the Blue Book): "Fully esterified
acids derived from a single acid are systematically named by placing the name(s) of the hydroxylic component denoted by an
organyl group(s)... in front of the name of the acid component... Multiplicative prefixes are used to denote a
multiplicity of identical organyl groups", 'dimethyl butanedioate (PIN)' (:31775). "Pseudoesters",
(:32049): when E of R-CO-O-E is one of B, Al, In, Ga, Tl, Si, Ge, Sn, Pb,..., "the pseudoester is named as a traditional
ester, unless other names must be selected in accordance with the seniority order of classes, in decreasing order: salts >
acids > anhydrides > esters", so the ester of a Group 14 organyl is named as the ester of the whole acid: 'bis(trimethylsilyl)
butanedioate'. SENIORITY ORDER FOR CLASSES, Table 4.1 (:18158 ff.): class 9 esters rank above class 16 ketones and class 41
ethers, so the name that cites the second ester in an 'oxo[(...)oxy]methyl' prefix or as '1,4-dioxo-...butane' is a name of a
lower class.
"""
import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# rows that were order-dependent at the parent of 1529f6889b (measured with the seeded spellings below), and their name
_POLYACID_ROWS = [
    ("CCCC[Sn](CCCC)(CCCC)OC(=O)CC(=C)C(=O)O[Sn](CCCC)(CCCC)CCCC",
     "bis(tributylstannyl) methylidenebutanedioate"),
    ("C[Sn](C)(C)OC(=O)c1ccc(C(=O)O[Sn](C)(C)C)cc1",
     "bis(trimethylstannyl) benzene-1,4-dicarboxylate"),
    ("C[Ge](C)(C)OC(=O)CCC(=O)O[Ge](C)(C)C",
     "bis(trimethylgermyl) butanedioate"),
    ("C[Si](C)(C)OC(=O)CC(CC(=O)O[Si](C)(C)C)(C(=O)O[Si](C)(C)C)O",
     "tris(trimethylsilyl) 2-hydroxypropane-1,2,3-tricarboxylate"),
]

_SEEDS = (23, 37)


def _spellings(smiles, count=8):
    """The written SMILES, RDKit's canonical one and ``count`` seeded random ones for each seed of ``_SEEDS``."""
    mol = Chem.MolFromSmiles(smiles)
    out = [smiles, Chem.MolToSmiles(mol)]
    for seed in _SEEDS:
        out += list(Chem.MolToRandomSmilesVect(mol, count, randomSeed=seed))
    return list(dict.fromkeys(out))


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles, expected", _POLYACID_ROWS)
def test_a_polyacid_of_identical_pseudoester_groups_has_one_name_in_every_atom_order(tier, smiles, expected):
    """Every spelling gives the ester of the whole acid, at the PIN tier and at best-effort."""
    namer = _tiered_namer("pin", tier)
    rows = {s: namer.name_tiered(s) for s in _spellings(smiles)}
    assert len(rows) > 8, "the random spellings of the row must differ from one another"
    assert {r["name"] for r in rows.values()} == {expected}, {s: r["name"] for s, r in rows.items()}
    assert {r["tier"] for r in rows.values()} == {"pin_verified"}, {s: r["tier"] for s, r in rows.items()}
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles", [
    "C[Si](C)(C)OC(=O)CCC(=O)O[Sn](C)(C)C",
    "C[Sn](C)(C)OC(=O)CCC(=O)O[Si](C)(C)C",
])
def test_a_polyacid_of_two_different_pseudoester_organyls_has_one_name_in_every_atom_order(smiles):
    """The group that is cut is the one of the lower canonical rank, not the first match of the atom order: the molecule
    has one name, whatever the spelling and whichever of the two drawings of it is the written one."""
    namer = _tiered_namer("pin", "best-effort")
    names = {}
    for written in (smiles, "C[Si](C)(C)OC(=O)CCC(=O)O[Sn](C)(C)C"):
        for s in _spellings(written):
            names[s] = namer.name_tiered(s)["name"]
    assert len(set(names.values())) == 1 and next(iter(names.values())), names
