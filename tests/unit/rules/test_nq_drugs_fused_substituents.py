"""Substituents of a fused heterocycle parent (``rules.fused_rings``): a ring nitrogen bonded to
the core is the free-valence atom of a ring-system substituent, and a carbon-count name is given
only to the chain shape it describes.

- (the Blue Book): the free valence of a substituent group is locant 1 of the chain;
  '3-hydroxypropyl' says HO-CH2-CH2-CH2-, nothing else. A branched '2-hydroxypropan-2-yl' or a
  '1-hydroxyethyl' read as '3-hydroxypropyl' / '2-hydroxyethyl' is another molecule.
- 'acetyl' is CH3-CO-,:17959 "for example 'acetyl', CH3-CO-"); -CH2-CHO is
  '2-oxoethyl'.
- A ring-system substituent bonded through a ring nitrogen takes the ring-yl name with its
  free valence ('piperazin-1-yl', '1H-imidazol-1-yl'); walked as a chain, a pyrrolidine was two
  butyl groups ('butan-1-yl(butan-4-ylamino)'). It has the PIN spelling at every public tier
  ('4-carbamoylpiperidin-1-yl', not '4-(amino-oxomethyl)piperidin-1-yl'), and a molecule that
  joins two identical ring systems by a bond (a ring assembly,,:15542, the parent of
  its PIN) gets no ring-yl from this route.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.rules import fused_rings
from tests.support.pin_tiers import assert_pin_at_both_tiers
from tests.support.rt_assert import assert_full_rt

pytestmark = pytest.mark.opsin_gate


def _core_and_start(smiles, core_size):
    """The ring system of ``core_size`` atoms and the first non-core atom bonded to it."""
    mol = Chem.MolFromSmiles(smiles)
    systems = []
    for ring in mol.GetRingInfo().AtomRings():
        r = set(ring)
        for s in [s for s in systems if s & r]:
            r |= s
            systems.remove(s)
        systems.append(r)
    core = next(s for s in systems if len(s) == core_size)
    start = next(n.GetIdx() for a in sorted(core) for n in mol.GetAtomWithIdx(a).GetNeighbors()
                 if n.GetIdx() not in core)
    return mol, core, start


def _name(result):
    return result.get("name") if isinstance(result, dict) else result


#: (SMILES on quinolin-3-yl, the substituent name; main gave the name in the comment)
QUINOLINE_SUBSTITUENTS = [
    ("CC(C)(O)c1cnc2ccccc2c1", "2-hydroxypropan-2-yl"),   # main: '3-hydroxypropyl'
    ("CC(O)c1cnc2ccccc2c1", "1-hydroxyethyl"),            # main: '2-hydroxyethyl'
    ("O=CCc1cnc2ccccc2c1", "2-oxoethyl"),                 # main: 'acetyl'
    ("OCC(C)c1cnc2ccccc2c1", "1-hydroxypropan-2-yl"),     # main: '3-hydroxypropyl'
    ("OCCCc1cnc2ccccc2c1", "3-hydroxypropyl"),            # unchanged
    ("N#CCc1cnc2ccccc2c1", "cyanomethyl"),                # unchanged
    ("OC(=O)CCc1cnc2ccccc2c1", "2-carboxyethyl"),         # unchanged
    ("CC(=O)c1cnc2ccccc2c1", "acetyl"),                   # unchanged
    ("CCC(=O)c1cnc2ccccc2c1", "propanoyl"),               # unchanged
    ("c1cnc2ccc(N3CCCC3)cc2c1", "pyrrolidin-1-yl"),       # main: 'butan-1-yl(butan-4-ylamino)'
    ("c1cnc2ccc(N3CCNCC3)cc2c1", "piperazin-1-yl"),       # main: None
    ("c1cnc2ccc(N3CCOCC3)cc2c1", "morpholin-4-yl"),       # main: None
    ("c1cnc2ccc(-n3ccnc3)cc2c1", "1H-imidazol-1-yl"),     # main: None
    ("Cc1cn(cn1)-c1ccc2ncccc2c1", "4-methyl-1H-imidazol-1-yl"),
]


@pytest.mark.parametrize("smiles,expected", QUINOLINE_SUBSTITUENTS)
def test_a_fused_core_substituent_is_named_by_its_shape(smiles, expected):
    mol, core, start = _core_and_start(smiles, 10)
    assert _name(fused_rings._identify_fused_substituent(mol, start, set(core))) == expected


@pytest.mark.parametrize("smiles", [
    "CC(C)(O)c1cnc2ccccc2c1", "CC(O)c1cnc2ccccc2c1", "O=CCc1cnc2ccccc2c1", "OCC(C)c1cnc2ccccc2c1"])
def test_the_carbon_count_namer_declines_other_chain_shapes(smiles):
    mol, core, start = _core_and_start(smiles, 10)
    assert fused_rings._identify_functionalized_substituent(mol, start, set(core)) is None


def test_a_ring_bearing_fragment_is_never_a_carbon_count_chain():
    # the 1H-indol-3-yl core of this molecule carries a tricyclic acid; main named the tricycle
    # '12-carboxydodecyl'
    mol, core, start = _core_and_start(
        "O=C1N[C@H](C(=O)O)Cc2c([nH]c3ccccc23)C1c1c[nH]c2ccccc12", 9)
    assert fused_rings._identify_functionalized_substituent(mol, start, set(core)) is None


@pytest.mark.parametrize("smiles", ["CC(C)(O)c1cnc2ccccc2c1", "OCCCc1cnc2ccccc2c1"])
def test_the_terminal_group_chain_shape(smiles):
    mol, core, start = _core_and_start(smiles, 10)
    chain = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in core]
    expected = smiles.startswith("OCCC")
    assert fused_rings._terminal_group_chain(mol, chain, start) is expected


@pytest.mark.parametrize("smiles,pin", [
    # ciprofloxacin; added hydrogen with the 4-oxo prefix
    ("OC(=O)C1=CN(C2CC2)c2cc(N3CCNCC3)c(F)cc2C1=O",
     "1-cyclopropyl-6-fluoro-4-oxo-7-(piperazin-1-yl)-1,4-dihydroquinoline-3-carboxylic acid"),
    ("OC(=O)c1cnc2ccc(N3CCCC3)cc2c1", "6-(pyrrolidin-1-yl)quinoline-3-carboxylic acid"),
    ("c1cnc2ccc(N3CCCC3)cc2c1", "6-(pyrrolidin-1-yl)quinoline"),
    # a stereocentre of the ring-yl is cited inside its prefix,:44643)
    ("C[C@@H]1CCCN1c1ccc2ncccc2c1", "6-[(2R)-2-methylpyrrolidin-1-yl]quinoline"),
])
def test_ring_nitrogen_substituents_of_a_fused_parent_reach_the_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def _tier_row(tier, smiles):
    if tier == "pin":
        return Orthonym(style="pin").name_tiered(smiles)
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.parametrize("tier", ["pin", "valid", "complete", "best-effort"])
def test_a_decorated_ring_nitrogen_substituent_has_its_pin_at_every_public_tier(tier):
    # the best-effort context used to name the group '4-(amino-oxomethyl)piperidin-1-yl'
    smiles = "OC(=O)c1cnc2ccc(N3CCC(C(N)=O)CC3)cc2c1"
    pin = "6-(4-carbamoylpiperidin-1-yl)quinoline-3-carboxylic acid"
    row = _tier_row(tier, smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def _ring_n_start(smiles):
    """The ring nitrogen bonded to another ring system, and the ring system it is bonded to."""
    mol = Chem.MolFromSmiles(smiles)
    systems = []
    for ring in mol.GetRingInfo().AtomRings():
        r = set(ring)
        for s in [s for s in systems if s & r]:
            r |= s
            systems.remove(s)
        systems.append(r)
    owner = {a: k for k, s in enumerate(systems) for a in s}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in owner and j in owner and owner[i] != owner[j]:
            n, c = (i, j) if mol.GetAtomWithIdx(i).GetSymbol() == "N" else (j, i)
            return mol, n, systems[owner[c]]
    raise AssertionError(smiles)


def test_a_ring_assembly_gets_no_ring_nitrogen_substituent(monkeypatch):
    # two indoles joined by a bond are the ring assembly 1H,1'H-1,5'-biindole,:15542);
    # the best-effort delegate names the N-linked indole, so the screen is what keeps a
    # substitutive name with an indole parent out
    from orthonym.metrics.provenance import best_effort_ctx
    from orthonym.rules import ring_assembly_screen
    mol, n, core = _ring_n_start("c1ccc2c(c1)ccn2-c1ccc2[nH]ccc2c1")
    tok = best_effort_ctx.set(True)
    try:
        assert fused_rings._ring_rooted_substituent(mol, n, set(core)) is None
        # known positive: without the screen the same call names the group
        monkeypatch.setattr(ring_assembly_screen, "joins_identical_ring_systems", lambda m: False)
        assert _name(fused_rings._ring_rooted_substituent(mol, n, set(core)))
    finally:
        best_effort_ctx.reset(tok)


def test_the_best_effort_call_runs_only_after_the_strict_call_declines(monkeypatch):
    # 6-methoxy-1H-indol-1-yl on a quinoline: the strict call (allow_mancude=False, the
    # best-effort context cleared) declines the group, the best-effort call names it, and the
    # best-effort context is set again after the call
    from orthonym.metrics.provenance import best_effort_ctx
    from orthonym.rules import ring_substituents
    smiles = "c1cnc2ccc(cc2c1)-n1ccc2ccc(OC)cc21"
    mol, n, core = _ring_n_start(smiles)
    real = ring_substituents.name_ring_system_substituent
    calls, depth = [], [0]

    def spy(*args, **kwargs):
        depth[0] += 1
        try:
            out = real(*args, **kwargs)
        finally:
            depth[0] -= 1
        if depth[0] == 0:   # the calls of _ring_rooted_substituent, not the delegate's own
            calls.append((kwargs.get("allow_mancude"), best_effort_ctx.get(), out))
        return out

    monkeypatch.setattr(ring_substituents, "name_ring_system_substituent", spy)
    tok = best_effort_ctx.set(True)
    try:
        res = fused_rings._ring_rooted_substituent(mol, n, set(core))
        assert best_effort_ctx.get() is True
    finally:
        best_effort_ctx.reset(tok)
    assert [c[:2] for c in calls] == [(False, False), (True, True)], calls
    assert calls[0][2] is None and calls[1][2], calls
    ring_yl = calls[1][2]
    assert _name(res) == ring_yl
    # at the PIN tier only the strict call runs, and the route declines
    calls.clear()
    assert fused_rings._ring_rooted_substituent(mol, n, set(core)) is None
    assert [c[:2] for c in calls] == [(False, False)], calls
    # the best-effort name of the molecule carries that ring-yl and reads back to the input
    row = _tier_row("best-effort", smiles)
    assert row["name"] and ring_yl in row["name"], row
    assert_full_rt(row["name"], smiles)
