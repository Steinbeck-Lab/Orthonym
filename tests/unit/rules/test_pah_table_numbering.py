"""The numbering rows of the PAH table (``data/polycyclic_data.py``) and what they name.

A ring system of the table with no ``iupac_numbering`` row is numbered by the fusion-numbering
engine or, for a peri-fused one, by a peripheral guess; a substituted perylene or benzo[pqr]tetraphene
had neither and fell to a von Baeyer name (``5-chloropentacyclo[10.6.2.0^2,7.0^9,19.0^16,20]icosa-...``).
The ``perylene`` row also carried the SMILES of another C20 skeleton, so a substituted perylene never
matched it.

Governing rules (``the Blue Book``):

- NUMBERING (:3219), criterion (a) "fixed numbering in chains, rings, or ring systems, i.e.,
  when the numbering of a system is fixed, for example in purine, anthracene, and phenanthrene, this
  numbering must be used, both in PINs and in general nomenclature" (:3227); criterion (f) "detachable
  alphabetized prefixes, all considered together in a series of increasing numerical order" (:3301);
- Lowest set of locants (:3189): "the set that, when compared term by term with other locant
  sets, each cited in order of increasing value, has the lowest term at the first point of difference";
- Numbering of peripheral skeletal atoms, (:12501): numbering "starts from the
  uppermost ring... Each fusion carbon atom is given the same number as the immediately preceding
  nonfusion skeletal atom, modified by a Roman letter 'a', 'b', 'c', 'd', etc.";
- "Retained names for hydrocarbon parent ring components in descending order of
  seniority": '(5) perylene (PIN)' (:11354); 'benzo[pqr]tetraphene (PIN)' (:12390).

The independent oracle is OPSIN 2.9.0: the rows are its ``<name> -o extendedsmi`` labels, and the
expected name of a structure is the lowest locant set among the names OPSIN reads back to that same
structure, brute-forced over every locant of the parent (no numbering of this package is used).
"""
import itertools

import pytest
from rdkit import Chem

from orthonym.data.polycyclic_data import POLYCYCLIC_DATA

pytestmark = pytest.mark.unit

# name -> (OPSIN 2.9.0 SMILES, OPSIN locants in the atom order of that SMILES): ``<name> -o extendedsmi``
OPSIN_ROWS = {
    'acenaphthylene': ('C1=CC2=CC=CC3=CC=CC1=C23',
                       '1;2;2a;3;4;5;5a;6;7;8;8a;8b'),
    'chrysene': ('C1=CC=CC=2C3=CC=C4C=CC=CC4=C3C=CC12',
                 '1;2;3;4;4a;4b;5;6;6a;7;8;9;10;10a;10b;11;12;12a'),
    'tetracene': ('C1=CC=CC2=CC3=CC4=CC=CC=C4C=C3C=C12',
                  '1;2;3;4;4a;5;5a;6;6a;7;8;9;10;10a;11;11a;12;12a'),
    'triphenylene': ('C1=CC=CC=2C3=CC=CC=C3C3=CC=CC=C3C12',
                     '1;2;3;4;4a;4b;5;6;7;8;8a;8b;9;10;11;12;12a;12b'),
    'tetraphene': ('C1=CC=CC2=CC=C3C=C4C=CC=CC4=CC3=C12',
                   '1;2;3;4;4a;5;6;6a;7;7a;8;9;10;11;11a;12;12a;12b'),
    'benzo[c]phenanthrene': ('C1=CC=CC=2C=CC=3C=CC=4C=CC=CC4C3C21',
                             '1;2;3;4;4a;5;6;6a;7;8;8a;9;10;11;12;12a;12b;12c'),
    'pentacene': ('C1=CC=CC2=CC3=CC4=CC5=CC=CC=C5C=C4C=C3C=C12',
                  '1;2;3;4;4a;5;5a;6;6a;7;7a;8;9;10;11;11a;12;12a;13;13a;14;14a'),
    'picene': ('C1=CC=CC2=CC=C3C4=CC=C5C=CC=CC5=C4C=CC3=C21',
               '1;2;3;4;4a;5;6;6a;6b;7;8;8a;9;10;11;12;12a;12b;13;14;14a;14b'),
    'perylene': ('C1=CC=C2C=CC=C3C4=CC=CC5=CC=CC(C1=C23)=C45',
                 '1;2;3;3a;4;5;6;6a;6b;7;8;9;9a;10;11;12;12a;12b;12c;12d'),
    'benzo[pqr]tetraphene': ('C1=C2C3=C4C(=C5C=CC=CC5=CC4=CC=C3C=C1)C=C2',
                             '1;12a;12c;12b;10b;10a;10;9;8;7;6a;6;5a;5;4;3a;3;2;11;12'),
}

# the four rows of the table that are a hydro or indicated-hydrogen FORM of another parent keep an
# empty numbering (see the comment on each row); the other ten above are filled
HYDRO_FORM_ROWS = ('fluorene', '9,10-dihydroanthracene', '1,2-dihydronaphthalene')

# the name an entry is read under: the retained 'fluorene' needs its indicated hydrogen, 'acenaphthene'
# is named 1,2-dihydroacenaphthylene, the rest are their own name
READ_AS = {'fluorene': '9H-fluorene'}


def _same_numbering(opsin_smiles, opsin_labels, entry):
    """True iff the entry's numbering, read on its own molecule, is OPSIN's labelled graph up to the
    symmetry of the ring system."""
    labels = opsin_labels.split(';')
    theirs = Chem.MolFromSmiles(opsin_smiles)
    mine = Chem.MolFromSmiles(entry['canonical_smiles'])
    assert theirs.GetNumAtoms() == mine.GetNumAtoms() == len(labels)
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    query = Chem.AdjustQueryProperties(theirs, params)
    numbering = {i: str(v) for i, v in entry['iupac_numbering'].items()}
    return any(all(numbering[match[i]] == labels[i] for i in range(len(labels)))
               for match in mine.GetSubstructMatches(query, uniquify=False, maxMatches=100000))


def _key(smiles):
    mol = Chem.MolFromSmiles(smiles.split(' |')[0])
    return Chem.MolToInchiKey(mol) if mol is not None else None


def _join(prefix, parent):
    return f"{prefix}{'-' if parent[0].isdigit() else ''}{parent}"


# --------------------------------------------------------------------------- the rows

@pytest.mark.parametrize('name', sorted(OPSIN_ROWS))
def test_the_row_is_opsins_numbering(name):
    smiles, labels = OPSIN_ROWS[name]
    assert _same_numbering(smiles, labels, POLYCYCLIC_DATA[name])


def test_the_comparison_can_fail():
    """Mutation guard of the check above: a row with two labels swapped is not OPSIN's."""
    smiles, labels = OPSIN_ROWS['perylene']
    entry = dict(POLYCYCLIC_DATA['perylene'])
    swapped = dict(entry['iupac_numbering'])
    a, b = sorted(swapped)[:2]
    swapped[a], swapped[b] = swapped[b], swapped[a]
    entry['iupac_numbering'] = swapped
    assert not _same_numbering(smiles, labels, entry)


@pytest.mark.parametrize('name', sorted(OPSIN_ROWS))
def test_the_row_numbers_every_atom_once(name):
    entry = POLYCYCLIC_DATA[name]
    numbering = entry['iupac_numbering']
    mol = Chem.MolFromSmiles(entry['canonical_smiles'])
    assert len(numbering) == mol.GetNumAtoms() == entry['num_atoms']
    assert len(set(numbering.values())) == len(numbering)
    assert sorted(v for v in numbering.values() if isinstance(v, int)) == sorted(entry['substituent_positions'])


def test_the_perylene_row_is_perylene():
    """The row carried another C20 skeleton (six rings, 12 CH) that no perylene matched."""
    mol = Chem.MolFromSmiles(POLYCYCLIC_DATA['perylene']['canonical_smiles'])
    assert mol.GetRingInfo().NumRings() == POLYCYCLIC_DATA['perylene']['num_rings'] == 5
    perylene = Chem.MolFromSmiles('c1cc2cccc3c4cccc5cccc(c(c1)c23)c54')
    assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(perylene)


@pytest.mark.parametrize('name', sorted(POLYCYCLIC_DATA))
@pytest.mark.roundtrip
@pytest.mark.opsin_gate
def test_the_row_smiles_is_the_named_compound(name):
    """Every entry's SMILES is the structure OPSIN reads from the entry's own name."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smiles = opsin_parse(READ_AS.get(name, name))
    assert smiles, name
    theirs = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    mine = Chem.MolToSmiles(Chem.MolFromSmiles(POLYCYCLIC_DATA[name]['canonical_smiles']))
    assert theirs == mine


# --------------------------------------------------------------------------- the names

@pytest.fixture(scope='module')
def namer():
    from orthonym import Orthonym
    from orthonym.tiers import emit_tier_flags
    return Orthonym(**emit_tier_flags('pin'))


PERYLENE = 'c1cc2cccc3c4cccc5cccc(c(c1)c23)c54'
BENZO_PQR_TETRAPHENE = 'c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34'

PINNED = {
    # the three monochloroperylenes (CH positions 1, 2, 3 of the 12; the others are equivalent)
    'Clc1ccc2cccc3c4cccc5cccc(c1c23)c54': '1-chloroperylene',
    'Clc1cc2cccc3c4cccc5cccc(c(c1)c23)c54': '2-chloroperylene',
    'Clc1ccc2c3cccc4cccc(c5cccc1c52)c43': '3-chloroperylene',
    # dichloro: the lowest of the equivalent locant sets,
    'Clc1ccc2cccc3c4cccc5ccc(Cl)c(c1c23)c54': '1,12-dichloroperylene',
    'Clc1cc(Cl)c2c3cccc4cccc(c5cccc1c52)c43': '1,3-dichloroperylene',
    'Clc1ccc2c3cccc4ccc(Cl)c(c5cccc1c25)c43': '1,9-dichloroperylene',
    # benzo[pqr]tetraphene (benzo[a]pyrene): all twelve monochloro isomers
    'Clc1ccc2ccc3cc4ccccc4c4ccc1c2c34': '1-chlorobenzo[pqr]tetraphene',
    'Clc1cc2ccc3cc4ccccc4c4ccc(c1)c2c34': '2-chlorobenzo[pqr]tetraphene',
    'Clc1ccc2ccc3c4ccccc4cc4ccc1c2c43': '3-chlorobenzo[pqr]tetraphene',
    'Clc1cc2cc3ccccc3c3ccc4cccc1c4c23': '4-chlorobenzo[pqr]tetraphene',
    'Clc1cc2cccc3ccc4c5ccccc5cc1c4c32': '5-chlorobenzo[pqr]tetraphene',
    'Clc1c2ccccc2c2ccc3cccc4ccc1c2c43': '6-chlorobenzo[pqr]tetraphene',
    'Clc1cccc2c1cc1ccc3cccc4ccc2c1c34': '7-chlorobenzo[pqr]tetraphene',
    'Clc1ccc2c(c1)cc1ccc3cccc4ccc2c1c34': '8-chlorobenzo[pqr]tetraphene',
    'Clc1ccc2cc3ccc4cccc5ccc(c2c1)c3c45': '9-chlorobenzo[pqr]tetraphene',
    'Clc1cccc2cc3ccc4cccc5ccc(c12)c3c45': '10-chlorobenzo[pqr]tetraphene',
    'Clc1cc2cccc3ccc4cc5ccccc5c1c4c32': '11-chlorobenzo[pqr]tetraphene',
    'Clc1cc2c3ccccc3cc3ccc4cccc1c4c32': '12-chlorobenzo[pqr]tetraphene',
    'Clc1ccc2ccc3c(Cl)c4ccccc4c4ccc1c2c34': '1,6-dichlorobenzo[pqr]tetraphene',
    'Clc1ccc2c(Cl)cc3c4ccccc4cc4ccc1c2c43': '3,12-dichlorobenzo[pqr]tetraphene',
    'Clc1c(Cl)c2c3ccccc3cc3ccc4cccc1c4c32': '11,12-dichlorobenzo[pqr]tetraphene',
}


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize('smiles,name', sorted(PINNED.items(), key=lambda kv: kv[1]))
def test_a_substituted_perylene_or_benzo_pqr_tetraphene_is_named_by_its_fusion_name(namer, smiles, name):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    row = namer.name_tiered(smiles)
    assert (row['name'], row['tier']) == (name, 'pin_verified'), row
    assert _key(opsin_parse(row['name'])) == _key(smiles)


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize('parent', [PERYLENE, BENZO_PQR_TETRAPHENE])
def test_the_parents_alone_stay_named(namer, parent):
    row = namer.name_tiered(parent)
    assert (row['name'], row['tier']) == (
        'perylene' if parent == PERYLENE else 'benzo[pqr]tetraphene', 'pin_verified'), row


def _isomers(parent_smiles):
    """Every distinct mono- and dichloro derivative (a chlorine on each of two CH atoms)."""
    parent = Chem.MolFromSmiles(parent_smiles)
    ch = [a.GetIdx() for a in parent.GetAtoms() if a.GetTotalNumHs() >= 1 and a.IsInRing()]

    def derive(sites):
        rw = Chem.RWMol(parent)
        for site in sites:
            rw.AddBond(site, rw.AddAtom(Chem.Atom(17)), Chem.BondType.SINGLE)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)

    mono = sorted({derive([i]) for i in ch})
    di = sorted({derive(p) for p in itertools.combinations(ch, 2)})
    return mono, di


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize('name', [
    'fluorene', 'acenaphthylene', 'chrysene', 'tetracene', 'triphenylene', 'tetraphene',
    'benzo[c]phenanthrene', 'pentacene', 'picene', 'perylene', 'benzo[pqr]tetraphene',
    '9,10-dihydroanthracene', '1,2-dihydronaphthalene'])
def test_every_chloro_isomer_gets_the_lowest_locants(namer, name):
    """The 14 rows that had no numbering: every mono- and dichloro isomer is named by its parent with
    the lowest locant set (a) fixed numbering, then (f) with, against the lowest
    among all the names OPSIN reads to the same structure, and the name reads back to the input."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    entry = POLYCYCLIC_DATA[name]
    parent = READ_AS.get(name, name)
    mono, di = _isomers(entry['canonical_smiles'])
    # every candidate locant of the parent; which candidate names are THIS structure is OPSIN's call
    positions = sorted(entry['substituent_positions'])
    best = {}
    for locants in [(p,) for p in positions] + list(itertools.combinations(positions, 2)):
        prefix = f"{','.join(map(str, locants))}-{'chloro' if len(locants) == 1 else 'dichloro'}"
        read = opsin_parse(_join(prefix, parent))
        if not read:
            continue
        k = _key(read)
        if k not in best or locants < best[k][0]:
            best[k] = (locants, _join(prefix, parent))
    assert best, name
    wrong = []
    for smiles in mono + di:
        expected = best.get(_key(smiles))
        got = namer.name_tiered(smiles)['name']
        if expected is None or got != expected[1]:
            wrong.append((smiles, got, expected and expected[1]))
    assert not wrong, wrong[:5]


# --------------------------------------------------------------------------- the hydro forms

HYDRO = {
    # a partially saturated ring system of the ten filled rows is a fusion name, not a von Baeyer one
    'C1=Cc2c(ccc3ccc4ccccc4c23)CC1': '3,4-dihydrobenzo[c]phenanthrene',
    'Clc1cc2c(c3c1ccc1ccccc13)C=CCC2': '6-chloro-3,4-dihydrobenzo[c]phenanthrene',
    'C1=c2cccc3c2c(c2cccc4cccc3c42)CC1': '1,2-dihydroperylene',
    'C1=C2CCC=c3c2c(c2cccc4cccc3c42)CC1': '1,2,4,5-tetrahydroperylene',
    # the four rows that stay without a numbering: with one, partial_saturation took them for a mancude
    # parent and these came out as 1,2-dihydrofluorene, 4,5-dihydroacenaphthene,...
    'C1=CC2=C(CC1)Cc1ccccc12': '1,2-dihydro-9H-fluorene',
    'C1=C2CCc3cccc(c32)CC1': '1,2,4,5-tetrahydroacenaphthylene',
    'C1=CC2=C(CC1)Cc1ccccc1C2': '1,2,9,10-tetrahydroanthracene',
    'C1=CC2=C(CC1)CCC=C2': '1,2,7,8-tetrahydronaphthalene',
}


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize('smiles,name', sorted(HYDRO.items(), key=lambda kv: kv[1]))
def test_a_hydro_form_is_named_by_its_mancude_parent(namer, smiles, name):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    row = namer.name_tiered(smiles)
    assert row['name'] == name, row
    assert _key(opsin_parse(row['name'])) == _key(smiles)


@pytest.mark.parametrize('name', HYDRO_FORM_ROWS)
def test_the_hydro_form_rows_have_no_numbering(name):
    """partial_saturation reads every row with a numbering as a mancude parent (see the comment on
    each row); the names pinned in HYDRO are what a numbering on one of these four rows breaks."""
    assert POLYCYCLIC_DATA[name]['iupac_numbering'] == {}
