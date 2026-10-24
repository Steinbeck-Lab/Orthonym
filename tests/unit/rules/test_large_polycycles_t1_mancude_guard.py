"""The von Baeyer PIN path refuses an aromatic cage before the main-ring search (large-polycycle
plan, Task 1).

``polycyclic.name_polycyclic_complete`` refuses a cage that carries an aromatic atom (G0: von
Baeyer nomenclature cannot express aromaticity) and delegates to the bridged fused and fused
producers first. None of that reads the von Baeyer descriptor, so at the PIN tier it runs
before the main-ring search, the costly step: "Five-membered ring requirement"
(the Blue Book) -- "Fusion nomenclature gives preferred IUPAC names only to compounds
having at least two rings of at least five or more members." -- says such a cage has a fusion
or bridged fused PIN, so a von Baeyer name can never be its PIN and the search result is never
used. The best-effort tier keeps the original order and keeps naming such a cage.

A mancude cage drawn as a Kekule structure (no RDKit aromaticity) is NOT refused early: its von
Baeyer name is recorded as a non-PIN fragment and the default tier declines it as
NO_VERIFIED_PIN, the code that tells the caller a wider tier returns a name. An early refusal
there would report UNNAMEABLE for molecules the best-effort tier names.
"""
import pytest

from tests.support.large_polycycles import be, full_molecule_smiles, ladder, pin
from tests.support.rt_assert import rt_exact_name

pytestmark = pytest.mark.opsin_gate
L = ladder()


def _spy_analyze_inside_npc(monkeypatch, tier_call, smiles):
    """Names ``smiles`` and returns (npc entered, analyze calls made while inside npc)."""
    import orthonym.assembly.composer as composer
    from orthonym.rules import polycyclic
    inside, calls, entered = [], [], []
    orig_npc = polycyclic.name_polycyclic_complete
    orig_an = polycyclic.VonBaeyerAnalyzer.analyze

    def npc(*a, **k):
        entered.append(1)
        inside.append(1)
        try:
            return orig_npc(*a, **k)
        finally:
            inside.pop()

    def an(self, *a, **k):
        if inside:
            calls.append(1)
        return orig_an(self, *a, **k)

    monkeypatch.setattr(polycyclic, "name_polycyclic_complete", npc)
    monkeypatch.setattr(composer, "name_polycyclic_complete", npc)
    monkeypatch.setattr(polycyclic.VonBaeyerAnalyzer, "analyze", an)
    tier_call(smiles)
    return entered, calls


@pytest.mark.unit
@pytest.mark.parametrize("label", ["B01_trimethanoanthracene", "B02_1458dimethanoanthracene"])
def test_bridged_mancude_cage_keeps_its_bridged_fused_pin_and_best_effort_name(label):
    # PROTECT (main names these through the bridged fused builder): the guard sits behind it.
    r = pin(L[label])
    assert r["is_pin"] and "cyclo[" not in (r["name"] or ""), r
    assert rt_exact_name(L[label]) is not None


# Class members that reach name_polycyclic_complete (R00_corannulene and W01 do not: other
# producers take them first).
_AROMATIC_CAGES = ["R01_corannulene+1unit_BH", "R05_core_21rings", "G01m_core+1methyl"]
# Kekule drawings no RDKit aromaticity flags, whose fused core is mancude (fusion_pin_class True)
_KEKULE_CAGES = ["C1=CC2=C3C1=CC=C3C=C2", "N1=NC2=C3C1=NN=C3N=N2", "C1=CC=P23C=CC=CC2=CC=CC3=C1"]


@pytest.mark.unit
@pytest.mark.parametrize("key", _AROMATIC_CAGES)
def test_aromatic_cage_pin_path_never_runs_the_main_ring_search(key, monkeypatch):
    """The von Baeyer PIN path refuses an aromatic cage before ``analyze``. Other callers (the
    general engine, prefix builders) may analyse the cage; they never label a PIN."""
    smiles = L.get(key, key)
    entered, calls = _spy_analyze_inside_npc(monkeypatch, pin, smiles)
    assert entered, "the spy never saw name_polycyclic_complete -- the routing changed, re-spy"
    assert calls == [], "the PIN path must refuse before analyze (G0 hoisted)"
    assert not pin(smiles)["is_pin"]


@pytest.mark.unit
@pytest.mark.parametrize("key", _KEKULE_CAGES)
def test_kekule_mancude_cage_is_declined_as_a_name_that_exists(key):
    """A cage no aromaticity flag marks keeps its analysis and its code: the PIN tier declines
    the von Baeyer name it built (NO_VERIFIED_PIN, 'a wider tier returns it'), and the
    best-effort tier names it."""
    smiles = L.get(key, key)
    r = pin(smiles)
    assert (r["tier"], r["limit_code"]) == ("abstain", "NO_VERIFIED_PIN"), r
    assert be(smiles)["name"]


@pytest.mark.unit
def test_issue_molecule_pin_path_never_runs_the_main_ring_search(monkeypatch):
    entered, calls = _spy_analyze_inside_npc(monkeypatch, pin, full_molecule_smiles())
    assert entered, "the spy never saw name_polycyclic_complete -- the routing changed, re-spy"
    assert calls == []


@pytest.mark.unit
@pytest.mark.parametrize("key", ["R01_corannulene+1unit_BH", "C1=CC2=C3C1=CC=C3C=C2"])
def test_best_effort_tier_still_enters_the_analysis(key, monkeypatch):
    """The guard is the PIN tier's: at best-effort the von Baeyer name of a mancude fusion-class
    cage is a legitimate (non-PIN) name and its producer keeps running (breadth never drops)."""
    smiles = L.get(key, key)
    entered, calls = _spy_analyze_inside_npc(monkeypatch, be, smiles)
    assert entered and calls, (entered, calls)
    r = be(smiles)
    assert r["name"] and not r["is_pin"], r


@pytest.mark.unit
def test_pendant_aromatic_ring_on_a_cage_is_not_the_cage(monkeypatch):
    """ control: a naphthyl joined to norbornane by a single bond is a substituent, never
    part of the cage, so the aromatic refusal does not fire on it (the hoisted test reads the
    ring system the cage analysis would number)."""
    r = pin("c1ccc2ccccc2c1C1CC2CCC1C2")
    assert r["name"] and "bicyclo[2.2.1]heptan" in r["name"], r


@pytest.mark.unit
@pytest.mark.parametrize("label,expected", [
    ("B04_14methanonaphthalene_ctrl", "1,4-methanonaphthalene"),          # BB:19479 "(PIN)"
    ("B03_1458dimethanonaphthalene_ctrl", "1,4:5,8-dimethanonaphthalene"),
    ("A01_adamantane_ctrl", "adamantane"),                                 # BB:9885
])
def test_negative_controls_keep_their_pin(label, expected):
    r = pin(L[label])
    assert (r["name"], r["tier"]) == (expected, "pin_verified")


@pytest.mark.unit
def test_unfused_mancude_cage_output_is_unchanged():
    # S3 (ASSUMED, a review #13): 'bicyclo[4.4.1]undeca-1,3,5,7,9-pentaene' appears nowhere in the Blue
    # Book, so this asserts only that the guard does not fire (no ortho-fused pair) -- the PIN-tier
    # output recorded while writing this plan is unchanged. It is NOT a claim that it is the PIN.
    r = pin(L["B05_16methano10annulene_ctrl"])
    assert r["name"] == "bicyclo[4.4.1]undeca-1,3,5,7,9-pentaene"
