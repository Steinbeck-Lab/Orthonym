"""The open hydrocarbon series of: polyacenes, polyaphenes, polyalenes,
polyphenylenes, polynaphthylenes and polyhelicenes (``rules/hydrocarbon_series.py``).

 "Systematically named hydrocarbon parent components" (the Blue Book) defines a
CLASS of parent components for each series, one member for every count of rings, so these tests
read members the tables never listed (octacene, a pentadecaphenylene) as well as the printed
ones. Rule lines:

- Polyacenes,:11438 -- "four or more ortho-fused benzene rings in a straight linear
  arrangement... a numerical prefix ('tetra', 'penta', etc.)... followed by the ending 'acene'
  ... with elision of a letter 'a'"; 'tetracene (PIN) pentacene (PIN)':11442;
- Polyaphenes,:11444 -- "(n + 1)/2 rings (if n is odd) or n/2 and (n/2) + 1 rings (if
  n is even) with a common benzene ring and that make a formal angle of 120 degrees";
  'pentaphene (n = 5) (PIN)':11449, 'hexaphene (n = 6) (PIN)':11457;
- Polyalenes,:11455 -- "two identical ortho-fused monocyclic hydrocarbon rings...
  'penta', 'hepta', etc.... 'alene'... with elision of a letter 'a'. The name naphthalene is
  retained"; 'pentalene (PIN) octalene (PIN)':11459;
- Polyphenylenes,:11461 -- "a monocyclic hydrocarbon with an even number of carbon
  atoms and benzene rings ortho-fused to alternate sides... 'phenylene'. The traditional name
  biphenylene is retained"; 'biphenylene (PIN) (not diphenylene)':11465, 'triphenylene (PIN)'
  :11467;
- Polynaphthylenes,:11469 -- "ortho-fused on alternate sides to the 2,3-positions of
  naphthalene rings... The series begins with three naphthalene rings, trinaphthylene";
  'trinaphthylene (PIN)':11475;
- Polyhelicenes,:11477 -- "of six or more rings... a benzene ring ortho-fused to the
  3,4-position of phenanthrene and further benzene rings fused in a similar way";
  'hexahelicene (PIN)':11483.

The numbering is (:12501); the locant vectors below are OPSIN 2.9.0's own
(``<name> -o extendedsmi``, the ``$_AV`` labels), the independent reading of the same name.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.rules import hydrocarbon_series as hs

pytestmark = pytest.mark.unit

# name -> (OPSIN 2.9.0 SMILES, OPSIN locants in the atom order of that SMILES)
OPSIN_NUMBERING = {
    'tetracene': ('C1=CC=CC2=CC3=CC4=CC=CC=C4C=C3C=C12',
                  '1;2;3;4;4a;5;5a;6;6a;7;8;9;10;10a;11;11a;12;12a'),
    'octacene': ('C1=CC=CC2=CC3=CC4=CC5=CC6=CC7=CC8=CC=CC=C8C=C7C=C6C=C5C=C4C=C3C=C12',
                 '1;2;3;4;4a;5;5a;6;6a;7;7a;8;8a;9;9a;10;10a;11;12;13;14;14a;15;15a;16;16a;17;17a;'
                 '18;18a;19;19a;20;20a'),
    'octaphene': ('C1=CC=CC2=CC3=CC4=CC=C5C=C6C=C7C=C8C=CC=CC8=CC7=CC6=CC5=C4C=C3C=C12',
                  '1;2;3;4;4a;5;5a;6;6a;7;8;8a;9;9a;10;10a;11;11a;12;13;14;15;15a;16;16a;17;17a;18;'
                  '18a;18b;19;19a;20;20a'),
    'nonaphene': ('C1=CC=CC2=CC3=CC4=CC5=CC=C6C=C7C=C8C=C9C=CC=CC9=CC8=CC7=CC6=C5C=C4C=C3C=C12',
                  '1;2;3;4;4a;5;5a;6;6a;7;7a;8;9;9a;10;10a;11;11a;12;12a;13;14;15;16;16a;17;17a;18;'
                  '18a;19;19a;19b;20;20a;21;21a;22;22a'),
    'octalene': ('C1=CC=CC=CC2=CC=CC=CC=C12', '1;2;3;4;5;6;6a;7;8;9;10;11;12;12a'),
    'nonalene': ('C1=CC=CC=CC=C2C=CC=CC=CC=C12', '1;2;3;4;5;6;7;7a;8;9;10;11;12;13;14;14a'),
    'tetraphenylene': ('C1=CC=CC=2C3=CC=CC=C3C3=CC=CC=C3C3=CC=CC=C3C12',
                       '1;2;3;4;4a;4b;5;6;7;8;8a;8b;9;10;11;12;12a;12b;13;14;15;16;16a;16b'),
    'tetranaphthylene': ('C1=CC=CC2=CC=3C4=CC5=CC=CC=C5C=C4C4=CC5=CC=CC=C5C=C4C4=CC5=CC=CC=C5C=C4C3C=C12',
                         '1;2;3;4;4a;5;5a;5b;6;6a;7;8;9;10;10a;11;11a;11b;12;12a;13;14;15;16;16a;17;17a;'
                         '17b;18;18a;19;20;21;22;22a;23;23a;23b;24;24a'),
    'hexahelicene': ('C1=CC=CC2=CC=C3C=CC4=CC=C5C=CC6=CC=CC=C6C5=C4C3=C12',
                     '1;2;3;4;4a;5;6;6a;7;8;8a;9;10;10a;11;12;12a;13;14;15;16;16a;16b;16c;16d;16e'),
    'nonahelicene': ('C1=CC=CC2=CC=C3C=CC4=CC=C5C=CC6=CC=C7C=CC8=CC=C9C=CC=CC9=C8C7=C6C5=C4C3=C12',
                     '1;2;3;4;4a;5;6;6a;7;8;8a;9;10;10a;11;12;12a;13;14;14a;15;16;16a;17;18;18a;19;20;'
                     '21;22;22a;22b;22c;22d;22e;22f;22g;22h'),
}


def _same_numbering(opsin_smiles, opsin_labels, entry):
    """True iff the series entry's numbering, read on its own molecule, is OPSIN's labelled graph
    up to the symmetry of the ring system (bond orders ignored: a localized drawing has two)."""
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


# --------------------------------------------------------------------------- names

@pytest.mark.parametrize('series,n,name', [
    ('acene', 4, 'tetracene'), ('acene', 5, 'pentacene'), ('acene', 8, 'octacene'),
    ('acene', 15, 'pentadecacene'), ('acene', 17, 'heptadecacene'), ('acene', 20, 'icosacene'),
    ('acene', 21, 'henicosacene'),
    ('aphene', 4, 'tetraphene'), ('aphene', 5, 'pentaphene'), ('aphene', 6, 'hexaphene'),
    ('aphene', 9, 'nonaphene'),
    ('alene', 5, 'pentalene'), ('alene', 7, 'heptalene'), ('alene', 8, 'octalene'),
    ('alene', 10, 'decalene'),
    ('phenylene', 2, 'biphenylene'), ('phenylene', 3, 'triphenylene'),
    ('phenylene', 4, 'tetraphenylene'), ('phenylene', 11, 'undecaphenylene'),
    ('phenylene', 15, 'pentadecaphenylene'),
    ('naphthylene', 3, 'trinaphthylene'), ('naphthylene', 4, 'tetranaphthylene'),
    ('helicene', 6, 'hexahelicene'), ('helicene', 9, 'nonahelicene'),
])
def test_series_name_and_its_inverse(series, n, name):
    """The numerical prefix, the ending, and the elision of a letter 'a' of 'acene' and 'alene'
     :11438,:11455); the prefix of biphenylene is 'bi' (:11465)."""
    assert hs.series_name(series, n) == name
    assert hs.parse_series_name(name) == hs.SeriesSpec(series, n)


@pytest.mark.parametrize('series,n', [
    ('acene', 3),          # anthracene is retained, the series begins at tetra (:11438)
    ('aphene', 3),         # phenanthrene; the rule says n > 3 (:11444)
    ('alene', 6),          # six carbon atoms in each ring is naphthalene, retained (:11455)
    ('phenylene', 1),
    ('naphthylene', 2),    # 'dinaphthylene' is dibenzo[b,h]biphenylene, not a parent (:11469)
    ('helicene', 5),       # the series begins with six rings (:11477 note)
])
def test_not_a_member(series, n):
    assert hs.series_name(series, n) is None


@pytest.mark.parametrize('name', ['anthracene', 'naphthalene', 'phenanthrene', 'tetralin', 'octane',
                                  'acenaphthylene', 'diphenylene', 'dinaphthylene', 'pentahelicene',
                                  'hexalene', ''])
def test_parse_rejects_other_names(name):
    assert hs.parse_series_name(name) is None


# --------------------------------------------------------------------------- numbering

@pytest.mark.parametrize('name', sorted(OPSIN_NUMBERING))
def test_numbering_is_OPSINs(name):
    """ (:12501): the entry's numbering is OPSIN 2.9.0's, on a member the tables
    do not list (octacene, nonaphene, tetranaphthylene, nonahelicene) as on a printed one."""
    smiles, labels = OPSIN_NUMBERING[name]
    assert _same_numbering(smiles, labels, hs.series_entry(name)), name


def test_the_comparison_can_fail():
    """Mutation guard of the check above: a numbering with two labels swapped is not OPSIN's."""
    smiles, labels = OPSIN_NUMBERING['octacene']
    entry = dict(hs.series_entry('octacene'))
    swapped = dict(entry['iupac_numbering'])
    a, b = sorted(swapped)[:2]
    swapped[a], swapped[b] = swapped[b], swapped[a]
    entry['iupac_numbering'] = swapped
    assert not _same_numbering(smiles, labels, entry)


@pytest.mark.parametrize('name', ['octacene', 'octaphene', 'tetraphenylene', 'tetranaphthylene',
                                  'hexahelicene', 'octalene'])
def test_entry_numbers_every_atom_once(name):
    entry = hs.series_entry(name)
    numbering = entry['iupac_numbering']
    mol = Chem.MolFromSmiles(entry['canonical_smiles'])
    assert len(numbering) == mol.GetNumAtoms() == entry['num_atoms']
    assert len(set(numbering.values())) == len(numbering)
    assert entry['substituent_positions'] == sorted(v for v in numbering.values() if isinstance(v, int))


def _labelled_equal(smiles_a, numbering_a, smiles_b, numbering_b):
    """Two numberings of one skeleton agree up to the symmetry of the skeleton."""
    return _labelled_equal_mols(Chem.MolFromSmiles(smiles_a), numbering_a,
                                Chem.MolFromSmiles(smiles_b), numbering_b)


def _labelled_equal_mols(a, numbering_a, b, numbering_b):
    if a.GetNumAtoms() != b.GetNumAtoms():
        return False
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    query = Chem.AdjustQueryProperties(a, params)
    la = {i: str(v) for i, v in numbering_a.items()}
    lb = {i: str(v) for i, v in numbering_b.items()}
    return any(all(la[i] == lb[m[i]] for i in la)
               for m in b.GetSubstructMatches(query, uniquify=False, maxMatches=100000))


def _tabulated_members():
    from orthonym.data.polycyclic_data import POLYCYCLIC_DATA
    return sorted(n for n in POLYCYCLIC_DATA if hs.parse_series_name(n))


@pytest.mark.parametrize('name', _tabulated_members())
def test_the_class_reproduces_every_tabulated_member(name):
    """One rule for the whole series: where ``POLYCYCLIC_DATA`` lists a member (a hand-verified
    row; the table holds no helicene past hexahelicene, the class serves them) its numbering is
    the class's; a row with no numbering of its own is numbered by the fusion engine, which the
    class also takes for the polyacenes and polyaphenes."""
    from orthonym.data.polycyclic_data import POLYCYCLIC_DATA
    from orthonym.rules.fusion_numbering import compute_fused_numbering
    row = POLYCYCLIC_DATA[name]
    mine = hs.series_entry(name)
    theirs = row['iupac_numbering']
    if not theirs:
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        theirs = {i: (f'{v[0]}{v[1]}' if isinstance(v, tuple) else v)
                  for i, v in compute_fused_numbering(mol, set(range(mol.GetNumAtoms()))).items()}
    assert _labelled_equal(row['canonical_smiles'], theirs,
                           mine['canonical_smiles'], mine['iupac_numbering']), name


# --------------------------------------------------------------------------- recognition

def _hit(smiles):
    hit = hs.identify_series(Chem.MolFromSmiles(smiles))
    return hit.name if hit else None


# the user's rows 10-13 (.scratch/user14.smi), as written
HEPTADECACENE = ('C1=CC=CC2=CC3=CC4=CC5=CC6=CC7=CC8=CC9=CC%10=CC%11=CC%12=CC%13=CC%14=CC%15=CC%16=CC%17='
                 'CC=CC=C%17C=C%16C=C%15C=C%14C=C%13C=C%12C=C%11C=C%10C=C9C=C8C=C7C=C6C=C5C=C4C=C3C=C12')
PENTADECACENE = ('C1=CC=CC2=CC3=CC4=CC5=CC6=CC7=CC8=CC9=CC%10=CC%11=CC%12=CC%13=CC%14=CC%15=CC=CC=C%15'
                 'C=C%14C=C%13C=C%12C=C%11C=C%10C=C9C=C8C=C7C=C6C=C5C=C4C=C3C=C12')
UNDECAPHENYLENE = 'C1=CC=CC=2' + 'C3=CC=CC=C3' * 10 + 'C12'
PENTADECAPHENYLENE = 'C1=CC=CC=2' + 'C3=CC=CC=C3' * 14 + 'C12'


@pytest.mark.parametrize('smiles,name', [
    (HEPTADECACENE, 'heptadecacene'), (PENTADECACENE, 'pentadecacene'),
    (UNDECAPHENYLENE, 'undecaphenylene'), (PENTADECAPHENYLENE, 'pentadecaphenylene'),
])
def test_users_rows_are_read_from_the_ring_graph(smiles, name):
    assert _hit(smiles) == name


@pytest.mark.parametrize('smiles', [
    'c1ccc2ccccc2c1',                                 # naphthalene
    'c1ccc2cc3ccccc3cc2c1',                           # anthracene
    'c1ccc2c(c1)ccc1ccccc12',                         # phenanthrene
    'c1cc2ccc3cccc4ccc(c1)c2c34',                     # pyrene: an atom in three rings
    'c1ccc2c(c1)ccc1c3ccccc3ccc21',                   # chrysene
    'c1ccc2c(c1)ccc1c2ccc2c3ccccc3ccc21',             # picene: all inner rings angular, alternating
    'c1ccc2c(c1)ccc1ccc3ccccc3c12',                   # benzo[c]phenanthrene, [4]helicene
    'C1Cc2cc3cc4cc5ccccc5cc4cc3cc2CC1',               # a saturated ring: not mancude
    'O=C1C=Cc2cc3cc4cc5ccccc5cc4cc3cc2C1',            # a double bond to a substituent
    'c1ccc2cc3cc4cc5ccccc5nc4cc3cc2c1',               # a ring nitrogen
    'c1ccc2c(c1)Cc1ccccc1-2',                         # fluorene
    'c1ccccc1-c1ccccc1',                              # biphenyl
    'C1=CC=C2C=CC=CC=C2C=C1',                         # heptalene: kept on its own producer
])
def test_other_ring_systems_are_not_members(smiles):
    assert _hit(smiles) is None


def test_a_substituent_is_free_to_be_anything():
    assert _hit('Cc1ccc2cc3cc4cc5cc6cc7cc8ccccc8cc7cc6cc5cc4cc3cc2c1') == 'octacene'
    assert _hit('OC(=O)c1ccc2c(c1)-c1ccccc1-c1ccccc1-c1ccccc1-2') == 'tetraphenylene'


def test_the_larger_ring_system_is_the_parent():
    """ 'greater number of skeletal ring atoms': pentacene and octacene joined by a
    single bond are two ring systems, and the octacene is the parent."""
    joined = Chem.MolFromSmiles(
        'c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1-c1ccc2cc3cc4cc5cc6cc7cc8ccccc8cc7cc6cc5cc4cc3cc2c1')
    assert hs.identify_series(joined).name == 'octacene'


def test_a_chain_with_its_turn_in_the_wrong_place_is_no_aphene():
    """ (:11444): the two straight arms hold (n + 1)/2 rings, or n/2 and n/2 + 1; six
    rings with the 120-degree turn after the second ring make arms of 2 and 5."""
    assert _hit('c1ccc2cc3cc4cc5c(ccc6ccccc65)cc4cc3cc2c1') is None


def test_a_zigzag_chain_is_no_helicene():
    """ (:11477): the rings are fused 'in a similar way', every turn the same way round;
    six rings whose turns alternate are not hexahelicene."""
    assert _hit('c1ccc2c(c1)ccc1c2ccc2c3ccc4ccccc4c3ccc12') is None


@pytest.mark.parametrize('name', ['nonahelicene', 'tetranaphthylene', 'decaphenylene', 'octalene'])
@pytest.mark.parametrize('seed', range(3))
def test_the_numbering_does_not_depend_on_the_atom_order(name, seed):
    """The walk of starts from the structure, never from the order the atoms are
    listed in: a shuffled drawing of the skeleton is numbered as the entry is (up to symmetry)."""
    import random
    entry = hs.series_entry(name)
    mol = Chem.MolFromSmiles(entry['canonical_smiles'])
    order = list(range(mol.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    shuffled = Chem.RenumberAtoms(mol, order)
    walked = hs._walk_numbering(shuffled, hs.parse_series_name(name)) if name != 'octalene' else None
    if walked is None:       # the engine numbers an alene; compare through the entry builder instead
        from orthonym.rules.fusion_numbering import compute_fused_numbering
        walked = compute_fused_numbering(shuffled, set(range(shuffled.GetNumAtoms())))
    got = {i: (f'{v[0]}{v[1]}' if isinstance(v, tuple) else v) for i, v in walked.items()}
    assert _labelled_equal_mols(mol, entry['iupac_numbering'], shuffled, got), (name, seed)


def test_the_parent_ring_system_of_two_series_members_is_the_one_of_more_rings():
    """ (d) before (e) (the Blue Book): hexaphenylene has 36 atoms in 7 rings, octacene
    34 atoms in 8 rings; joined by a single bond, the octacene is the parent."""
    hexaphenylene = Chem.MolFromSmiles('C1=CC=CC=2' + 'C3=CC=CC=C3' * 5 + 'C12')
    octacene = Chem.MolFromSmiles('c1ccc2cc3cc4cc5cc6cc7cc8ccccc8cc7cc6cc5cc4cc3cc2c1')
    rw = Chem.RWMol(Chem.CombineMols(hexaphenylene, octacene))
    rw.AddBond(0, hexaphenylene.GetNumAtoms(), Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    assert hs.identify_series(mol).name == 'octacene'


@pytest.mark.parametrize('seed', range(4))
def test_recognition_does_not_depend_on_the_atom_order(seed):
    import random
    rng = random.Random(seed)
    for smiles, name in ((HEPTADECACENE, 'heptadecacene'), (UNDECAPHENYLENE, 'undecaphenylene')):
        mol = Chem.MolFromSmiles(smiles)
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        shuffled = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
        assert _hit(shuffled) == name


def test_either_drawing_of_a_localized_system_is_the_same_member():
    """ 'Localized double bonds' (:14595): octalene has two drawings; both are octalene."""
    one = Chem.MolFromSmiles('C1=CC=CC=CC2=CC=CC=CC=C12')
    other = Chem.MolFromSmiles('c1cccc2ccccccc=2cc1')
    assert Chem.MolToInchiKey(one) == Chem.MolToInchiKey(other)
    assert hs.identify_series(one).name == hs.identify_series(other).name == 'octalene'


# --------------------------------------------------------------------------- naming (read back by OPSIN)

def _full_key(smiles):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


@pytest.fixture(scope='module')
def namer():
    from orthonym import Orthonym
    from orthonym.tiers import emit_tier_flags
    return Orthonym(**emit_tier_flags('pin'))


@pytest.fixture(scope='module')
def wide_namer():
    from orthonym import Orthonym
    from orthonym.tiers import emit_tier_flags
    return Orthonym(**emit_tier_flags('best-effort'))


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize('smiles,name', [
    (HEPTADECACENE, 'heptadecacene'),
    (PENTADECACENE, 'pentadecacene'),
    (UNDECAPHENYLENE, 'undecaphenylene'),
    (PENTADECAPHENYLENE, 'pentadecaphenylene'),
])
def test_users_rows_are_named_at_both_tiers(namer, wide_namer, smiles, name):
    """Each name is PIN-verified: the series names are the parent components of the
    PINs) and OPSIN reads it back to the full InChIKey of the input."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    for who in (namer, wide_namer):
        row = who.name_tiered(smiles)
        assert (row['name'], row['tier']) == (name, 'pin_verified')
    assert _full_key(opsin_parse(name)) == _full_key(smiles)


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize('series,n', [
    ('acene', 8), ('acene', 20), ('aphene', 7), ('aphene', 12), ('alene', 8), ('alene', 12),
    ('phenylene', 5), ('phenylene', 18), ('naphthylene', 4), ('naphthylene', 9),
    ('helicene', 13), ('helicene', 20),
])
def test_members_past_the_tables_are_named(namer, series, n):
    """A bare member is named by whatever its size: 'icosacene' (82 atoms) is shorter
    than the character-count floor of the candidate pool allows, and a macrocycle of five or more
    benzene rings is a fusion system, not a cyclophane (1),:23828)."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    name = hs.series_name(series, n)
    smiles = opsin_parse(name)          # the structure by OPSIN's reading of the name, not ours
    row = namer.name_tiered(smiles)
    assert (row['name'], row['tier']) == (name, 'pin_verified'), row


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize('name', ['octacene', 'octaphene', 'tetraphenylene', 'tetranaphthylene',
                                  'nonahelicene'])
def test_substituted_members_are_named(namer, name):
    """2-methyl and the '-yl' prefix of the same ring system lowest free-valence locant)."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    # a ring CH on a terminal ring: the position of locant 2 (or 3 where the numbering starts higher)
    parent = Chem.MolFromSmiles(opsin_parse(name))
    entry = hs.series_entry(name)
    mine = Chem.MolFromSmiles(entry['canonical_smiles'])
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    match = parent.GetSubstructMatch(Chem.AdjustQueryProperties(mine, params))
    locant_atom = {entry['iupac_numbering'][i]: match[i] for i in range(len(match))}
    for locant in (1, 2):
        atom = locant_atom[locant]
        for sub, expect in (('C', f'{locant}-methyl{name}'),
                            ('C(C)=O', f'1-({name[:-1]}-{locant}-yl)ethan-1-one')):
            rw = Chem.RWMol(Chem.CombineMols(parent, Chem.MolFromSmiles(sub)))
            rw.AddBond(atom, parent.GetNumAtoms(), Chem.BondType.SINGLE)
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            row = namer.name_tiered(smiles)
            assert row['tier'] == 'pin_verified', (smiles, row)
            assert row['name'] == expect, (smiles, row['name'], expect)


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize('smiles,expected', [
    #: 'tetraphenylene' is one parent name, not a multiplied 'phenylene' prefix cited before 'carboxy'
    ('OC(=O)c1ccc2c(c1)c1ccccc1c1ccccc21', 'triphenylene-2-carboxylic acid'),
    ('O=C(O)c1ccc2c(c1)-c1ccccc1-c1ccccc1-c1ccccc1-2', 'tetraphenylene-2-carboxylic acid'),
])
def test_polyphenylene_parent_is_not_read_as_a_multiplied_prefix(namer, smiles, expected):
    row = namer.name_tiered(smiles)
    assert (row['name'], row['tier']) == (expected, 'pin_verified'), row


def test_ortho_fused_macrocycle_is_not_a_cyclophane():
    """ (1) (:23828): a cyclophane has a mancude ring attached at NONADJACENT positions;
     (:23833): rings attached to adjacent atoms of a large ring form a fusion system,
    'A cyclophane name is not allowed'."""
    from orthonym.rules.phane import is_cyclophane
    assert not is_cyclophane(Chem.MolFromSmiles(PENTADECAPHENYLENE))
    assert not is_cyclophane(Chem.MolFromSmiles('C1=CC=CC=2' + 'C3=CC=CC=C3' * 4 + 'C12'))
    assert is_cyclophane(Chem.MolFromSmiles('c1cc2ccc1CCc1ccc(cc1)CC2'))      # [2.2]paracyclophane


def test_a_substituted_polyalene_is_left_to_the_other_producers():
    """ 'Localized double bonds' (:14595): the bond-shift drawings of a polyalene
    are told apart by a Delta descriptor, so a locant for a substituent depends on the drawing.
    The class names the bare ring system, in either drawing, and nothing that carries a substituent."""
    assert _hit('C1=CC=CC=CC2=CC=CC=CC=C12') == 'octalene'
    assert _hit('c1cccc2ccccccc=2cc1') == 'octalene'
    assert _hit('CC1=CC=CC=CC2=CC=CC=CC=C12') is None


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
def test_both_drawings_of_bare_octalene_get_one_name(namer):
    """The bare name has no locant to depend on the drawing 'octalene (PIN)',:11459)."""
    for drawing in ('C1=CC=CC=CC2=CC=CC=CC=C12', 'c1cccc2ccccccc=2cc1'):
        row = namer.name_tiered(drawing)
        assert (row['name'], row['tier']) == ('octalene', 'pin_verified'), row


# --------------------------------------------------------------------------- (g)

# name -> the same structure under the other numbering, which ties on every locant set: the
# prefix cited first ('chloro' before 'methyl', takes the lower locant, (g),
# the Blue Book "lowest locants for the substituent cited first as a prefix in the name"
G_CASES = [
    '1-chloro-9-methyltetraphenylene',
    '2-chloro-3-methylpentaphenylene',
    '1-chloro-8-methylhexaphenylene',
    '2-chloro-15-methylhexaphenylene',
    '1-chloro-10-methylheptacene',
    '1-chloro-14-methyloctacene',
    '1-chloro-7-methylcoronene',
    '1-chloro-8-methylpentacene',
]


def _swapped(name):
    import re
    cl, me, tail = re.match(r'^(\d+)-chloro-(\d+)-methyl(.+)$', name).groups()
    return f'{me}-chloro-{cl}-methyl{tail}'


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize('name', G_CASES)
def test_the_prefix_cited_first_takes_the_lower_locant(namer, name):
    """The structure is read from the other name, whose numbering is the tie's other end, so a
    producer that took the first numbering would give '9-chloro-1-methyl...'; the name must not
    depend on which numbering the atoms were listed in either."""
    import random
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smiles = opsin_parse(_swapped(name))
    assert _full_key(smiles) == _full_key(opsin_parse(name)), 'not a tie'
    mol = Chem.MolFromSmiles(smiles)
    for seed in range(3):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        row = namer.name_tiered(Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False))
        assert (row['name'], row['tier']) == (name, 'pin_verified'), (seed, row)


@pytest.mark.parametrize('name', ['tetraphenylene', 'octacene', 'hexaphene', 'coronene', 'anthracene'])
def test_every_chloro_methyl_isomer_gets_the_lower_locant_to_chloro(name):
    """No fixed structure list: every Cl, CH3 pair of a ring system, against a brute-force best of
    all numberings (lowest locant set first, then chloro lowest, (f) and (g))."""
    import itertools
    from orthonym.rules.polycyclics import _pah_entry, get_polycyclic_iupac_locants
    entry = _pah_entry(name)
    parent = Chem.MolFromSmiles(entry['canonical_smiles'])
    ch = [a.GetIdx() for a in parent.GetAtoms() if a.GetTotalNumHs() == 1]
    seen = {}
    for a, b in itertools.permutations(ch, 2):
        rw = Chem.RWMol(Chem.CombineMols(parent, Chem.MolFromSmiles('Cl.C')))
        n0 = parent.GetNumAtoms()
        rw.AddBond(a, n0, Chem.BondType.SINGLE)
        rw.AddBond(b, n0 + 1, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        seen.setdefault(Chem.MolToSmiles(mol), (mol, a, b))
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    query = Chem.AdjustQueryProperties(parent, params)
    numbering = entry['iupac_numbering'] or {i: i + 1 for i in range(parent.GetNumAtoms())}
    for smiles, (mol, a, b) in seen.items():
        if not entry['iupac_numbering']:
            continue
        pairs = set()
        for m in mol.GetSubstructMatches(query, uniquify=False, maxMatches=100000):
            if set(m) != set(range(parent.GetNumAtoms())):
                continue
            num = {m[i]: v for i, v in numbering.items()}
            pairs.add((num[a], num[b]))
        key = lambda l: (l, '') if isinstance(l, int) else (int(l[:-1]), l[-1])
        best = min(pairs, key=lambda t: (sorted(map(key, t)), key(t[0])))
        got = get_polycyclic_iupac_locants(mol, name, substituent_atoms={a, b})
        # the entry's own atoms are 0..n-1 here: Cl sits on ``a``, CH3 on ``b``
        loc = lambda x: (got[x] if isinstance(got[x], int) else f'{got[x][0]}{got[x][1]}')
        assert (loc(a), loc(b)) == best, (name, smiles, (loc(a), loc(b)), best)


def test_coronene_has_its_own_numbering():
    """Coronene had no numbering, so a substituted one was numbered by the order its atoms were
    listed in; the row is OPSIN 2.9.0's ``coronene -o extendedsmi``."""
    smiles = 'C1=CC2=CC=C3C=CC4=CC=C5C=CC6=CC=C1C1=C6C5=C4C3=C21'
    labels = '1;2;2a;3;4;4a;5;6;6a;7;8;8a;9;10;10a;11;12;12a;12g;12f;12e;12d;12c;12b'
    from orthonym.data.polycyclic_data import POLYCYCLIC_DATA
    assert _same_numbering(smiles, labels, POLYCYCLIC_DATA['coronene'])


def test_the_parent_ring_system_is_chosen_by_rings_before_atoms(monkeypatch):
    """ (d) 'has the greater number of rings' before (e) 'the greater number of
    skeletal atoms' (the Blue Book): a series member with more rings but fewer atoms than
    the table's PAH is the parent."""
    from orthonym.data import polycyclic_data
    from orthonym.rules import polycyclics
    # three benzene rings in a chain: anthracene (3 rings, 14 atoms) in a molecule with a
    # biphenylene (3 rings, 12 atoms) -- and a table row made to have fewer rings but more atoms
    monkeypatch.setitem(polycyclic_data.POLYCYCLIC_DATA, 'anthracene',
                        dict(polycyclic_data.POLYCYCLIC_DATA['anthracene'], num_rings=2, num_atoms=40))
    mol = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1-c1ccc2c(c1)-c1ccccc1-2')
    polycyclics._POLY_CACHE.clear()
    assert polycyclics._identify_polycyclic_impl(mol) == 'biphenylene'


# --------------------------------------------------------------------------- the producer's measure

def test_the_polycyclic_producer_measures_the_partition_of_a_bare_ring_system(monkeypatch):
    """The producer, not the pool, says that a name of the open series accounts for every atom:
    'icosacene' (82 atoms, ratio 0.073) is bare, so its candidate is added as a measured
    partition; a methylated one is not a measured partition."""
    from orthonym.assembly import candidate_pool
    from orthonym.assembly.handlers import _handler_shared
    seen = []
    real = candidate_pool.CandidatePool.add

    def spy(self, name, handler_id, features, *args, **kwargs):
        if handler_id == 'polycyclic':
            seen.append((name, kwargs.get('coverage_measured')))
        return real(self, name, handler_id, features, *args, **kwargs)

    monkeypatch.setattr(candidate_pool.CandidatePool, 'add', spy)
    from orthonym import Orthonym
    from orthonym.tiers import emit_tier_flags
    from orthonym.validation.opsin_roundtrip import opsin_parse
    bare = opsin_parse('icosacene')
    namer = Orthonym(**emit_tier_flags('pin'))
    assert namer.name_tiered(bare)['name'] == 'icosacene'
    assert ('icosacene', True) in seen, seen
    seen.clear()
    rw = Chem.RWMol(Chem.CombineMols(Chem.MolFromSmiles(bare), Chem.MolFromSmiles('C')))
    atom = next(a.GetIdx() for a in rw.GetAtoms() if a.GetTotalNumHs() == 1)
    rw.AddBond(atom, Chem.MolFromSmiles(bare).GetNumAtoms(), Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    assert namer.name_tiered(Chem.MolToSmiles(mol))['name'].endswith('methylicosacene')
    assert seen and all(measured is False for _name, measured in seen), seen
    assert _handler_shared.pool_polycyclic_candidate


# --------------------------------------------------------------------------- phane: ring systems

@pytest.mark.parametrize('smiles', [
    'C1Cc2cccc3cccc(c23)CCc2cccc3cccc1c23',        # two naphthalenes joined at peri 1,8
    'C1Cc2ccccc2CCc2cccc3cccc1c23',                # one ortho benzene, one peri naphthalene
    'C1CCc2ccccc2CCCc2cccc3cccc1c23',
    'c1cc2ccc1CCc1ccc(cc1)CC2',                    # [2.2]paracyclophane
])
def test_a_ring_system_joined_at_nonadjacent_positions_makes_a_cyclophane(smiles):
    """ (1) (the Blue Book): 'at least one ring or ring system of which must be a
    mancude system attached... at nonadjacent ring positions'. A naphthalene joined at 1 and 8
    meets the macrocycle in two atoms of each ring, and three of the system."""
    from orthonym.rules.phane import is_cyclophane
    assert is_cyclophane(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize('smiles', [
    'C1CCCc2ccccc2CCCCc2ccccc21',                  # every ring joined at adjacent atoms
    'C1=CC=CC=2' + 'C3=CC=CC=C3' * 4 + 'C12',     # pentaphenylene
])
def test_rings_joined_at_adjacent_atoms_are_a_fusion_system(smiles):
    """ (:23833): 'A cyclophane name is not allowed'."""
    from orthonym.rules.phane import is_cyclophane
    assert not is_cyclophane(Chem.MolFromSmiles(smiles))


# --------------------------------------------------------------------------- the spelling reader

def test_a_series_parent_is_not_split_into_a_multiplied_prefix():
    from orthonym.validation.spelling.units import split_simple
    for word in ('tetraphenylene', 'triphenylene', 'pentadecaphenylene'):
        assert split_simple(word) == ([], word)
    assert split_simple('diphenylene') == ([('di', 'phenylene')], '')      # not a parent name
    assert split_simple('tetraphenyl') == ([('tetra', 'phenyl')], '')


def test_the_readers_copy_of_the_series_rule_agrees_with_the_rule():
    """The spelling reader may not import the writer, so it holds its own copy of 'a numerical
    prefix and phenylene is a parent component' (``units._SERIES_PARENT_RE``); every simple
    multiplier it knows, with every ending the copy covers, is a parent exactly when the rule says so."""
    from orthonym.validation.spelling.units import MULT, _is_series_parent
    for word in MULT:
        for ending in ('phenylene', 'naphthylene', 'helicene', 'acene', 'phene', 'lene'):
            candidate = word + ending
            if ending in ('phenylene', 'naphthylene'):
                assert _is_series_parent(candidate) == (hs.parse_series_name(candidate) is not None), candidate
            else:
                assert not _is_series_parent(candidate), candidate     # no word of its lexicon starts them

