"""v30 P3-T1b: the terminal ring namer and its reconstruction audit.

One test per primitive the task specifies (descriptor, replacement prefix + λ,
unsaturation locants, the audit rejecting a wrong descriptor), plus the two
guards that make the wiring safe: the PIN default stays byte-identical and a
fused ring keeps its FUSION name rather than a von Baeyer one.

Every test here was proven able to fail by a source mutation -- see the
``MUTATION:`` note on each.
"""
import pytest
from rdkit import Chem

from orthonym.rules.ring_substituents import get_ring_substituent_name
from orthonym.rules.terminal_ring import (
    audit_monocycle_replacement_name,
    build_monocycle_replacement_name,
    monocycle_numbering,
    parse_monocycle_replacement_name,
    terminal_ring_name,
)


def _ring(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol, sorted(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())


def _first_h_ring_atom(mol, ring):
    for i in ring:
        if mol.GetAtomWithIdx(i).GetTotalNumHs() > 0:
            return i
    return ring[0]


# ---------------------------------------------------------------- descriptor

def test_descriptor_primitive_names_a_cage_and_a_spiro_system():
    """The rank>=2 branch delegates to the audited universal analyzers.

    MUTATION: in ``terminal_ring._polycyclic_terminal_name``, drop
    ``analyze_cage_universal`` from the tuple -> adamantane comes back None.
    """
    mol, ring = _ring('C1C2CC3CC1CC(C2)C3')          # adamantane
    res = terminal_ring_name(mol, ring, _first_h_ring_atom(mol, ring))
    assert res is not None
    assert res.basis == 'von_baeyer'
    assert res.name.startswith('tricyclo[3.3.1.1^3,7]decan-')
    assert res.name.endswith('-yl')

    mol, ring = _ring('C1CCC2(CC1)CCCCC2')            # spiro[5.5]undecane
    res = terminal_ring_name(mol, ring, _first_h_ring_atom(mol, ring))
    assert res is not None
    assert res.basis == 'spiro'
    assert res.name.startswith('spiro[5.5]undecan-')


def test_monocycle_branch_is_total_where_von_baeyer_cannot_reach():
    """Von Baeyer starts at two rings (``vonbaeyer_universal.py:424``), so the
    monocycle branch is the new capability. It must answer, not refuse.

    MUTATION: make ``build_monocycle_replacement_name`` return None for
    ``n < 7`` -> both assertions below fail.
    """
    for smiles, expected in (
        ('C1CCOCC1', '1-oxacyclohexane'),
        ('C1CNCCN1', '1,4-diazacyclohexane'),
        ('C1COCCOCCOCCOCCO1', '1,4,7,10,13-pentaoxacyclopentadecane'),
    ):
        mol, ring = _ring(smiles)
        res = terminal_ring_name(mol, ring, None)
        assert res is not None, smiles
        assert res.basis == 'monocycle'
        assert res.name == expected


# -------------------------------------------- replacement prefix, with lambda

def test_replacement_prefix_cites_every_heteroatom_with_a_locant_and_lambda():
    """A locant per non-carbon skeletal atom, and λ for hypervalence
    (P-15.4.1.3 places it immediately after the locant, no hyphen).

    MUTATION: in ``ring_replacement.build_replacement_prefix``, pass ``None``
    instead of ``vb_lambda_for_atom(...)`` -> the λ4 disappears and the name
    becomes ``1-thiacyclopenta-2,4-diene``, i.e. thiophene: a DIFFERENT
    molecule, which is exactly what the λ is there to prevent.
    """
    mol, ring = _ring('[SH2]1C=CC=C1')               # λ4-thiophene
    res = terminal_ring_name(mol, ring, None)
    assert res is not None
    assert res.name == '1lambda4-thiacyclopenta-2,4-diene'

    # two heteroatoms -> two locants, multiplied, in P-23.3.1 element order
    mol, ring = _ring('C1COCCS1')
    res = terminal_ring_name(mol, ring, None)
    assert res is not None
    assert res.name == '1-oxa-4-thiacyclohexane'


def test_off_table_skeletal_element_refuses_rather_than_dropping_the_atom():
    """Table 1.5 is CLOSED (P-15.4.1.1). Hg has no ring 'a' prefix, so the ring
    stem would count an atom no morpheme spells -- the documented
    ``bicyclo[3.3.0]octane``-for-a-mercury-ring failure.

    MUTATION: delete the ``if repl.unexpressed: return None`` block in
    ``build_monocycle_replacement_name`` -> the SPELLER emits ``cyclohexane``, a
    hydrocarbon name for a mercury ring, and the second assertion fails. (The
    entry point stays None either way because the reconstruction audit
    independently rejects the wrong element -- defence in depth -- which is why
    the speller is asserted directly here.)
    """
    mol, ring = _ring('C1CC[Hg]CC1')
    assert terminal_ring_name(mol, ring, None) is None
    numbering = monocycle_numbering(mol, ring)
    assert numbering is not None
    assert build_monocycle_replacement_name(mol, ring, numbering, None) is None


def test_charged_skeletal_ring_atom_is_out_of_scope_and_refuses():
    """A ring cation is P-73, not replacement nomenclature; a neutral
    'a'-prefix name would denote a DIFFERENT species.

    MUTATION: delete the formal-charge loop in ``terminal_ring_name`` ->
    quinuclidinium comes back as ``1-azabicyclo[2.2.2]octane``, the NEUTRAL
    amine. The POLYCYCLIC case is the one that isolates this guard:
    ``audit_von_baeyer_descriptor`` has no charge clause (only the monocycle
    audit does), so on the rank>=2 branch this loop is the only thing between a
    ring cation and a neutral name.
    """
    mol, ring = _ring('C1C[NH+]2CCC1CC2')            # quinuclidinium, rank 2
    assert terminal_ring_name(mol, ring, None) is None
    mol, ring = _ring('c1cc[n+](C)cc1')
    ring = [i for i in ring if mol.GetAtomWithIdx(i).IsInRing()]
    assert terminal_ring_name(mol, ring, None) is None


# ------------------------------------------------------- unsaturation locants

def test_every_ring_multiple_bond_gets_a_locant_over_a_kekulized_graph():
    """RDKit reports an aromatic bond order as 1.5, so the ring must be
    kekulized before the bonds are read or every double bond is silently lost.

    MUTATION: in ``terminal_ring._kekulized_copy``, return ``(mol, ring_atoms)``
    without kekulizing -> benzene names as ``cyclohexane`` (all six double bonds
    dropped) and the audit then refuses, so the call returns None.
    """
    mol, ring = _ring('c1ccccc1')
    res = terminal_ring_name(mol, ring, None)
    assert res is not None
    assert res.name == 'cyclohexa-1,3,5-triene'

    mol, ring = _ring('c1ccsc1')                     # thiophene
    res = terminal_ring_name(mol, ring, None)
    assert res is not None
    assert res.name == '1-thiacyclopenta-2,4-diene'

    # a single ene elides the stem's 'a' (P-16.3.3); a multiplied one keeps it
    mol, ring = _ring('C1=CCOC1')                    # 2,5-dihydrofuran
    res = terminal_ring_name(mol, ring, None)
    assert res is not None
    assert res.name == '1-oxacyclopent-3-ene'


def test_the_audit_reads_an_aromatic_bond_as_1_5_and_never_as_single():
    """The 1.5 hazard, proven at the audit rather than argued: an un-kekulized
    aromatic ring must FAIL the audit for its own saturated name.

    MUTATION: in ``audit_monocycle_replacement_name`` change the bond-order
    branch to ``elif order in (1.0, 1.5):`` -> benzene passes the audit for
    ``cyclohexane`` and the assertion below fails.
    """
    mol = Chem.MolFromSmiles('c1ccccc1')
    ring = sorted(a.GetIdx() for a in mol.GetAtoms())
    numbering = monocycle_numbering(mol, ring)
    assert numbering is not None
    assert audit_monocycle_replacement_name(
        mol, ring, numbering, 'cyclohexane') is False


# -------------------------------------------------------- reconstruction audit

def test_audit_rejects_a_wrong_descriptor():
    """The gate that makes "always emit" safe: the emitted STRING is parsed back
    into a skeleton and set-compared to the graph. Every one of these mutations
    of a CORRECT name must be rejected.

    MUTATION: make ``audit_monocycle_replacement_name`` ``return True`` on entry
    -> all six assertions below fail.
    """
    mol, ring = _ring('C1CCOCC1')                    # oxane
    numbering = monocycle_numbering(mol, ring)
    assert numbering is not None
    good = build_monocycle_replacement_name(mol, ring, numbering, None)
    assert good == '1-oxacyclohexane'
    assert audit_monocycle_replacement_name(mol, ring, numbering, good)

    wrong = [
        'cyclohexane',            # heteroatom silently dropped
        '1-thiacyclohexane',      # wrong element at locant 1
        '2-oxacyclohexane',       # right element, WRONG locant
        '1-oxacyclopentane',      # wrong ring size
        '1-oxacyclohexa-2,4-diene',   # unsaturation the molecule does not have
        '1,4-dioxacyclohexane',   # a heteroatom the molecule does not have
    ]
    for name in wrong:
        assert audit_monocycle_replacement_name(
            mol, ring, numbering, name) is False, name


def test_audit_rejects_a_free_valence_locant_that_drifts_onto_another_atom():
    """The measured failure mode that kept the heteromonocycle rescue narrow --
    "the independently computed attachment locant lands on a ring OXYGEN".

    MUTATION: delete the free-valence block at the end of
    ``audit_monocycle_replacement_name`` -> the wrong-locant name passes.
    """
    mol, ring = _ring('C1CCOCC1')
    attach = _first_h_ring_atom(mol, ring)
    numbering = monocycle_numbering(mol, ring, attach)
    good = build_monocycle_replacement_name(
        mol, ring, numbering, numbering[attach])
    assert audit_monocycle_replacement_name(
        mol, ring, numbering, good, attach)
    # cite the ring oxygen (locant 1) as the free valence instead
    drifted = good.replace(f'-{numbering[attach]}-yl', '-1-yl')
    assert drifted != good
    assert audit_monocycle_replacement_name(
        mol, ring, numbering, drifted, attach) is False


def test_parser_refuses_text_it_cannot_fully_account_for():
    """A morpheme the grammar silently ignored would be a dropped heteroatom, so
    the parser must tile the string exactly.

    MUTATION: drop the ``$`` anchor from ``_MONO_RE`` -> '1-oxacyclohexane-junk'
    parses and the first loop assertion fails.

    ⚠ Honest note: the ``if consumed != len(unsat)`` check inside the function is
    NOT covered by this test and could not be made to fail. ``_MONO_RE``'s own
    unsaturation group uses the same grammar as ``_UNSAT_TERM_RE``, so once the
    anchored match succeeds the terms always tile. The check is defence in depth
    against a future divergence between the two patterns, not live logic.
    """
    assert parse_monocycle_replacement_name('1-oxacyclohexane') is not None
    for bad in ('1-oxacyclohexane-junk', 'oxacyclohexane', '1-zzzacyclohexane',
                '1,4-trioxacyclohexane', 'cyclohexa-1,3-triene'):
        assert parse_monocycle_replacement_name(bad) is None, bad


# ------------------------------------------------- the two safety-net controls

@pytest.mark.parametrize('smiles,expected', [
    ('c1ccc2ccccc2c1', 'naphthalen-2-yl'),
    ('c1ccc2[nH]ccc2c1', '1H-indol-5-yl'),
    ('c1ccc2ncccc2c1', 'quinolin-6-yl'),
])
def test_fused_ring_control_keeps_its_fusion_name_at_both_tiers(smiles, expected):
    """A von Baeyer name substituted where a FUSION name exists is the specific
    failure this wiring must not introduce. The terminal namer sits strictly
    after every retained/fusion producer, so these are untouched.

    MUTATION: move the ``terminal_ring_name`` block in
    ``ring_substituents.get_ring_substituent_name`` above the retained-name
    lookup -> naphthalene comes back as a bicyclo/von-Baeyer polyene and every
    parametrisation fails.
    """
    mol, ring = _ring(smiles)
    attach = _first_h_ring_atom(mol, ring)
    for allow in (False, True):
        got = get_ring_substituent_name(
            mol, tuple(ring), attach, allow_mancude=allow)
        assert got == expected, (smiles, allow, got)
        assert 'cyclo[' not in (got or '')


@pytest.mark.parametrize('smiles', [
    'C1CCOCC1', 'C1CNCCN1', 'c1ccccc1', 'C1CCCCC1', 'C1CSCCC1',
    'C1=NC=NC1', 'C1COCCOCCOCCOCCO1', 'C1=CCOC1', 'C1CC[Hg]CC1',
    'C1C2CC3CC1CC(C2)C3',
])
def test_pin_default_is_untouched(smiles):
    """The whole wiring is gated on ``allow_mancude``; the PIN default must be
    byte-identical, including its refusals.

    MUTATION: drop the ``allow_mancude and`` conjunct from either
    ``terminal_ring_name`` guard in ``ring_substituents.py`` -> the four rows
    that newly name under the general tier also change on the PIN path and this
    test fails on them.
    """
    expected = {
        'C1CCOCC1': 'oxanyl', 'C1CNCCN1': 'piperazinyl', 'c1ccccc1': 'phenyl',
        'C1CCCCC1': 'cyclohexyl', 'C1CSCCC1': 'thianyl', 'C1=NC=NC1': None,
        'C1COCCOCCOCCOCCO1': None, 'C1=CCOC1': None, 'C1CC[Hg]CC1': None,
        'C1C2CC3CC1CC(C2)C3': None,
    }
    mol, ring = _ring(smiles)
    attach = _first_h_ring_atom(mol, ring)
    assert get_ring_substituent_name(
        mol, tuple(ring), attach, allow_mancude=False) == expected[smiles]


def test_general_ring_prefix_emission_is_not_labelled_a_pin():
    """v30 P3-T1c: the composer stamps every emission ``source='pin_path'`` ->
    ``T1``, ``is_pin=True`` (``namer.py:2728``). A ring substituent prefix only
    the GENERAL tier could build is a valid but NOT preferred form -- the ring PIN
    here is the retained name *adamantane*, not ``tricyclo[3.3.1.1^3,7]decane`` --
    so it must be demoted rather than shipped as a PIN.

    MUTATION: delete the ``elif prov.get("general_ring_prefix")`` branch in
    ``namer.py`` -> the demoted molecule comes back ``T1`` / ``is_pin=True`` and
    the first two assertions fail.

    The assertion is on the DEMOTION, not on the shape of the name: this test
    originally required ``tricyclo[3.3.1.1^3,7]`` in the name, and the P-23.7
    retained-stem fix in the same session turned that molecule into the PIN
    ``(3-hydroxyadamantan-1-yl)acetic acid``. A name-shape assertion here was
    testing the wrong thing -- the flag records *which tier produced the prefix*,
    not what the prefix looks like. The contrast against a PIN-route molecule is
    what makes it meaningful.
    """
    from orthonym import Orthonym
    namer = Orthonym(general_fallback=True, general_fallback_unverified=True,
                      allow_aromatic_general=True)
    demoted = namer.name_tiered('OC(=O)CC12CC3CC(O)(CC(C3)C1)C2')
    assert demoted['name'], demoted
    assert demoted['is_pin'] is False, demoted
    assert demoted['tier'] in ('T3', 'T4'), demoted
    # CONTRAST: a composer emission whose ring prefix the PIN route produced is
    # untouched -- still T1 / is_pin. Without this the test would also pass if the
    # demotion fired for every composer emission.
    pin_route = namer.name_tiered('OCc1ccc2ccccc2c1')
    assert pin_route['name'] == '(naphthalen-2-yl)methanol', pin_route
    assert pin_route['is_pin'] is True, pin_route
    assert pin_route['tier'] == 'T1', pin_route


@pytest.mark.parametrize('smiles,expected', [
    # adamantane cage as a substituent
    ('OC(=O)CC12CC3CC(CC(C3)C1)C2', 'adamantan-1-yl'),
    # cubane cage as a substituent
    ('OC(=O)CC12C3C4C1C1C4C3C21', 'cuban-1-yl'),
])
def test_retained_pin_stem_beats_the_von_baeyer_descriptor(smiles, expected):
    """P-23.7: a cage with a RETAINED PIN name must use it, not the descriptor.

    **P-23.7 "RETAINED NAMES FOR VON BAEYER PARENT HYDRIDES"**
    (``BlueBookV2/BlueBookV2.md:9879``): *"The retained names adamantane and
    cubane are used in general nomenclature and as preferred IUPAC names."*
    Table 2.6 (``:9885``) prints *"adamantane (PIN) tricyclo[3.3.1.1^3,7]decane"*
    and *"cubane (PIN) pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]octane"* -- the retained
    name is the PIN and the descriptor is the ALTERNATIVE. Emitting the descriptor
    where a retained name exists is the one thing this project claims over , and the substituent side was missing it while the parent
    side (``tricyclo.get_retained_tricyclo_name``) already had it.

    MUTATION: make ``_retained_pin_cage_stem`` return None -> both cases come back
    as ``tricyclo[…]decan-N-yl`` / ``pentacyclo[…]octan-N-yl`` and fail.
    """
    from orthonym.assembly.substituent_enumerator import name_substituent
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    attach = next(i for i in ring
                  if any(not mol.GetAtomWithIdx(n.GetIdx()).IsInRing()
                         for n in mol.GetAtomWithIdx(i).GetNeighbors()))
    got = name_substituent(mol, ring, attach, allow_mancude=True)
    assert got == expected, got
    # PIN default unchanged: still the fail-closed sentinel.
    assert name_substituent(mol, ring, attach,
                            allow_mancude=False) == 'substituent'


def test_retained_stem_is_refused_for_a_hetero_or_unsaturated_cage():
    """Table 2.6 asserts nothing about a cage carrying a heteroatom or a ring
    multiple bond, so the retained stem must NOT be borrowed for one -- an
    ``adamantan-`` stem on an aza cage would drop the nitrogen.

    MUTATION: delete the ``if cage.hetero_prefix: return None`` guard in
    ``_retained_pin_cage_stem`` -> the aza cage below names as ``adamantan-…-yl``
    and the assertion fails.
    """
    from orthonym.rules.ring_substituents import _retained_pin_cage_stem

    class _Cage:
        descriptor = 'tricyclo[3.3.1.1^3,7]'
        hetero_prefix = '2-aza'
        unsaturation = {'double_bonds': [], 'triple_bonds': []}

    assert _retained_pin_cage_stem(_Cage(), 'decan') is None
    _Cage.hetero_prefix = ''
    assert _retained_pin_cage_stem(_Cage(), 'decan') == 'adamantan'
    _Cage.unsaturation = {'double_bonds': ['2'], 'triple_bonds': []}
    assert _retained_pin_cage_stem(_Cage(), 'decan') is None


def test_pin_tier_never_sees_the_ambient_general_flag():
    """The ambient-tier read is what makes the reachability fix work; this is the
    other half -- on the pin tier the contextvar is False, so the composer's ring
    path behaves exactly as before.

    MUTATION: make ``_tier_general`` unconditionally True in
    ``composer._generate_ring_substituent_prefixes`` -> the pin-tier row below
    gains a name (or changes one) and the equality against the general tier's
    own PIN-route answer breaks.
    """
    from orthonym import Orthonym
    pin = Orthonym(general_fallback=False, general_fallback_unverified=False,
                    allow_aromatic_general=False)
    row = pin.name_tiered('OC(=O)CC12CC3CC(O)(CC(C3)C1)C2')
    # the PIN tier declines this ring rather than reaching the general-tier route
    name = row.get('name') or ''
    assert 'tricyclo' not in name, row
    assert 'adamantan' not in name, row


def test_bare_ring_guard_refuses_a_decorated_fragment():
    """The atom-conservation guard: the generator names a RING, so calling it on
    a decorated fragment would silently DROP the decorations.

    MUTATION: invert the guard to ``if True: return None`` -> the second
    assertion fails, proving the test reaches and depends on this function.

    ⚠ Honest note: the NEGATIVE half could NOT be made to fail by deleting
    guards. Removing all three of ``any(ri.NumAtomRings(a) == 0 …)``,
    ``seen != frag`` and ``_cycle_order``'s ``len(nb) != 2`` still refuses,
    because ``_cycle_order``'s walk cannot close a cycle over a set holding a
    degree-1 atom (two further ``return None`` sites). Atom conservation here is
    over-determined rather than under-tested; the inverting mutation is what
    proves the test is not vacuous.
    """
    from orthonym.assembly.substituent_enumerator import (
        _terminal_bare_ring_substituent,
    )
    mol = Chem.MolFromSmiles('CC1CCOCC1')            # 4-methyloxane fragment
    frag = list(range(mol.GetNumAtoms()))
    ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    attach = ring[0]
    assert _terminal_bare_ring_substituent(mol, frag, attach) is None
    # the BARE ring system is accepted
    assert _terminal_bare_ring_substituent(mol, ring, attach) is not None
