"""Breadth restore -- the universal floor's '<acyl>oxy' leaf never costs a name.

76fb18964 (TRIAGE g8 C4) taught the best-effort universal floor to spell an
O-attached ester branch through the shared acid engine ('acetyloxy') instead of
a replacement chain ending on O. For an acyl side that carries its own free
-COOH the engine named the isolated POLYACID ('butanedioic acid') and the
'-oyl' conversion gave the divalent acyl 'butanedioyloxy',
the Blue Book: '-oyl' removes the -OH of EACH carboxy group). The floor
name failed the full-InChIKey round trip and, the floor being the last rung,
steroid and spiro hydrogen-succinate / -glutarate esters that best-effort named
round-trip exact before that commit abstained (dev2000: cholesteryl hydrogen
succinate).

Fixed at the leaf and behind it:
  * the leaf names the monovalent acyl of the ester site with the other carboxy
    groups as 'carboxy' prefixes, '(3-carboxypropanoyl)oxy'
    method (1), 'ammonium 3-carboxypropanoate (PIN)', the Blue Book), or
    declines (stereo in the acyl, an engine name that puts the '-oate' on
    another carboxy group, no name) so the generic spelling stays;
  * a '-carboxylic acid' acyl is '-carbonyl',:30624), not the
    OPSIN-unparseable '-carboxylyl';
  * t4_coverage keeps a leaf-bearing floor name only when it round-trips and
    otherwise offers the floor rebuilt without the leaf, so any other leaf
    defect degrades to the pre-leaf spelling instead of an abstention.

Witnesses: the dev2000 row and hand-made analogues (no a holdout split row is a
fixture). Every name is checked by an independent OPSIN 2.9.0 full-InChIKey
round trip (tests/support/rt_assert.py) and must not carry the PIN label.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

# Every test runs with the OPSIN validity gate ON, the production state: the
# helper names fragments through the engine, whose output depends on the gate
# (gate off, the citrate anion is the trianion '...tricarboxylate').
pytestmark = pytest.mark.opsin_gate

# Class A: a steroid / triterpenoid skeleton (von Baeyer floor parent) with a
# hydrogen dicarboxylate ester side group.
CHOLESTERYL_H_SUCCINATE = (  # dev2000 row
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](OC(=O)CCC(=O)O)"
    "CC[C@]4(C)[C@H]3CC[C@]12C")
CHOLESTANYL_H_GLUTARATE = (
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@@H](OC(=O)CCCC(=O)O)"
    "CC[C@]4(C)[C@H]3CC[C@]12C")
LANOSTERYL_H_SUCCINATE = (
    "CC(C)=CCC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC"
    "[C@H](OC(=O)CCC(=O)O)C(C)(C)[C@@H]1CC3")
CHOLENIC_ACID_H_SUCCINATE = (
    "C[C@H](CCC(=O)O)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](OC(=O)CCC(=O)O)"
    "CC[C@]4(C)[C@H]3CC[C@]12C")
#... with a 3-hydroxy-3-methylglutarate whose C-3 is a stereocentre only in
# the ester (the leaf declines; the floor's own stereo layer cites (4S)).
CHOLESTERYL_HMG = (
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](OC(=O)C[C@@](C)(O)CC(=O)O)"
    "CC[C@]4(C)[C@H]3CC[C@]12C")
LANOSTERYL_HMG = (
    "CC(C)=CCC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC"
    "[C@H](OC(=O)C[C@@](C)(O)CC(=O)O)C(C)(C)[C@@H]1CC3")
# Class B: a spiro compound with the hydrogen succinate ester side group.
SPIRO_H_SUCCINATE = (
    "O=C(O)CCC(=O)O[C@@H]1CCO[C@]2(C1)CC[C@@H](OC(C)=O)C1=CC(=O)CCC12")

# The tetracyclo locants follow the lowest secondary-bridge superscripts,
# 'tetracyclo[8.7.0.0^2,7.0^11,15]', the Blue Book), so the
# ester oxygen of the steroid 3-position is locant 5.
CARBOXY_ACYL_WITNESSES = [
    (CHOLESTERYL_H_SUCCINATE, "5-[(3-carboxypropanoyl)oxy]"),
    (CHOLESTANYL_H_GLUTARATE, "5-[(4-carboxybutanoyl)oxy]"),
    (LANOSTERYL_H_SUCCINATE, "5-[(3-carboxypropanoyl)oxy]"),
    (CHOLENIC_ACID_H_SUCCINATE, "5-[(3-carboxypropanoyl)oxy]"),
    (SPIRO_H_SUCCINATE, "4'-[(3-carboxypropanoyl)oxy]"),
]


def _shipped(smiles):
    res = name_best_effort(smiles)
    name = res.get("name")
    assert name and res.get("source") != "abstain", (smiles, res)
    assert name_is_rt_exact(name, smiles), (
        f"best-effort name is not OPSIN full-InChIKey exact for {smiles}: {name!r}")
    assert res.get("tier") != "pin_verified", (name, res.get("tier"))
    return name


@pytest.mark.parametrize("smiles,prefix", CARBOXY_ACYL_WITNESSES)
def test_hydrogen_dicarboxylate_ester_is_a_carboxy_acyloxy_prefix(smiles, prefix):
    """The ester side group is '(3-carboxypropanoyl)oxy' (-glutarate:
    '(4-carboxybutanoyl)oxy'), never the divalent 'butanedioyloxy'."""
    name = _shipped(smiles)
    assert prefix in name, name
    assert "dioyloxy" not in name, name


@pytest.mark.parametrize("smiles", [CHOLESTERYL_HMG, LANOSTERYL_HMG])
def test_stereo_bearing_acyl_keeps_the_floor_spelling(smiles):
    """A carboxy acyl with a stereocentre is not named from its isolated anion
    (the site O(-) and the ester O-R rank differently under CIP): the leaf
    declines and the floor's generic, stereo-complete spelling ships."""
    name = _shipped(smiles)
    assert "(4S)-4,6-dihydroxy-4-methyl-2-oxo-1,7-dioxahept-6-en-1-yl" in name, name
    assert "pentanedioyl" not in name, name


# --- the shared acyloxy helper, opt-in used by the floor --------------------

def _site(smiles):
    """(mol, carbonyl_idx, ester_o_idx) from atom maps:2 (carbonyl C),:1 (ester O)."""
    mol = Chem.MolFromSmiles(smiles)
    o = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 1)
    c = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 2)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return mol, c, o


@pytest.mark.parametrize("mapped,token", [
    ("c1ccccc1[O:1][C:2](=O)CCC(=O)O", "(3-carboxypropanoyl)oxy"),
    ("c1ccccc1[O:1][C:2](=O)CCCCCC(=O)O", "(6-carboxyhexanoyl)oxy"),
    # the ring carboxy group is the site: its '-oate' is the benzoate
    ("c1ccccc1[O:1][C:2](=O)c1ccc(CC(=O)O)cc1", "[4-(carboxymethyl)benzoyl]oxy"),
    # 'oxalooxy (preferred prefix)' (the Blue Book)
    ("c1ccccc1[O:1][C:2](=O)C(=O)O", "oxalooxy"),
    # 'cyclohexanecarbonyl (preferred prefix)' (:30628)
    ("c1ccccc1[O:1][C:2](=O)C1CCCCC1", "(cyclohexanecarbonyl)oxy"),
    # these two declined while the engine named the anion of the site with the charge
    # on another carboxy group ('4-(carboxymethyl)benzoate') or by a retained anion
    # name ('dihydrocitrate'); the protonation branch names the anion of the mapped
    # site itself, so the acyl of that site is derived, acyl groups from
    # the acid of the site; the ester reads back below)
    ("c1ccccc1[O:1][C:2](=O)Cc1ccc(C(=O)O)cc1", "[(4-carboxyphenyl)acetyl]oxy"),
    ("c1ccccc1[O:1][C:2](=O)CC(O)(CC(=O)O)C(=O)O", "(3,4-dicarboxy-3-hydroxybutanoyl)oxy"),
])
def test_acyloxy_token_is_the_monovalent_site_acyl(mapped, token):
    from orthonym.rules.lipids import _acyloxy_for_site
    mol, c, o = _site(mapped)
    got = _acyloxy_for_site(mol, ("acyl", c, o), carboxy_prefixed_polyacid=True)
    assert got == token
    # the token denotes the ester: '[<token>]benzene' is the phenyl ester itself
    probe = f"[{got}]benzene" if got.startswith("(") else f"({got})benzene"
    assert name_is_rt_exact(probe, Chem.MolToSmiles(mol)), probe


@pytest.mark.parametrize("mapped", [
    # a stereocentre that exists only in the ester (CIP of O(-) vs O-R)
    "c1ccccc1[O:1][C:2](=O)C[C@@H](O)CC(=O)O",
])
def test_acyloxy_token_declines_when_the_site_acyl_is_not_proven(mapped):
    from orthonym.rules.lipids import _acyloxy_for_site
    mol, c, o = _site(mapped)
    assert _acyloxy_for_site(mol, ("acyl", c, o), carboxy_prefixed_polyacid=True) is None


# --- the leaf-free retry behind the leaf -------------------------------------

CHOLESTANYL_H_SUCCINATE = (
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@@H](OC(=O)CCC(=O)O)"
    "CC[C@]4(C)[C@H]3CC[C@]12C")


def test_a_wrong_leaf_token_degrades_to_the_leaf_free_floor_name(monkeypatch):
    """Any leaf token that makes the floor name fail the round trip (here the
    pre-fix 'butanedioyloxy', forced by ignoring the opt-in) must cost only the
    spelling: t4_coverage rebuilds the floor without the leaf and best-effort
    still ships a full-InChIKey exact name, never an abstention."""
    from orthonym.assembly import composer

    real = composer._acyloxy_prefix_for_frag

    def pre_fix_leaf(mol, frag_atoms, attach_idx, carboxy_prefixed_polyacid=False):
        return real(mol, frag_atoms, attach_idx)

    monkeypatch.setattr(composer, "_acyloxy_prefix_for_frag", pre_fix_leaf)
    name = _shipped(CHOLESTANYL_H_SUCCINATE)
    assert "butanedioyloxy" not in name, name


CHOLESTANYL_H_ADIPATE = (
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@@H](OC(=O)CCCCC(=O)O)"
    "CC[C@]4(C)[C@H]3CC[C@]12C")


def test_verify_before_commit_rung_retries_the_floor_without_the_leaf(monkeypatch):
    """The same guarantee in ``_prefer_verified_floor`` (the rung that swaps an
    unverifiable engine candidate for the floor): with the pre-fix leaf forced,
    the leaf-bearing floor names fail, and the leaf-free floor name replaces the
    failing candidate instead of the candidate being kept."""
    from orthonym.assembly import composer
    from orthonym.assembly.t4_coverage import _Candidate, _prefer_verified_floor

    real = composer._acyloxy_prefix_for_frag

    def pre_fix_leaf(mol, frag_atoms, attach_idx, carboxy_prefixed_polyacid=False):
        return real(mol, frag_atoms, attach_idx)

    monkeypatch.setattr(composer, "_acyloxy_prefix_for_frag", pre_fix_leaf)
    mol = Chem.MolFromSmiles(CHOLESTANYL_H_ADIPATE)
    got = _prefer_verified_floor(mol, _Candidate(name="methane", result_obj=None))
    assert got.name != "methane"
    assert "hexanedioyloxy" not in got.name, got.name
    assert name_is_rt_exact(got.name, CHOLESTANYL_H_ADIPATE), got.name
