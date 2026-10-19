"""A ring cation with an N+=N bond is not a diazonium ion: ``classify_cation`` says 'diazonium'
only for the terminal -N2+ group.

 "Diazonium ions" (the Blue Book): "Cations containing an -N2+ group attached
to a parent hydride are traditionally named according to the principles of substitutive
nomenclature by using the suffix 'diazonium'... The use of the suffix 'diazonium' results in
preferred IUPAC names." The examples are methanediazonium, 2,4-dioxopentane-3-diazonium and
benzene-1,4-bis(diazonium): the group is the terminal N2 unit hanging off one parent atom.

In the Kekule drawing the engine works on, the ring cation of a sydnone (1,2,3-oxadiazol-3-ium-5-
olate), an oxatriazolium, a tetrazolium, a triazolium or a pyrazolium has an N+=N bond inside the
ring. ``classify_cation`` used to say 'diazonium' for any N+ with a double or triple bond to
another N, so these cations were handed to the diazonium emitter, which declines a ring, and
to the zwitterion planner and the charge suffix, which have no 'diazonium' base: the molecule
had no name at best-effort. A ring N+ is a skeletal heteroatom of the heterocycle:
"General rule for systematically naming cationic centers in parent hydrides" (:41366), "A
cation derived formally by adding one or more hydrons to any position of a neutral parent
hydride... is named by replacing the final letter 'e' of the parent hydride name, if any, by
the suffix 'ium'" (example:41396, '1H-imidazol-3-ium (PIN)'); with an anion on a characteristic
group, "Zwitterionic compounds with at least one ionic center on a characteristic
group" (:42445): "Zwitterionic compounds... may be named by adding the appropriate ionic suffix
to the name of the ionic parent hydride. In names, cationic suffixes are cited before anionic
suffixes."

The names asserted below are the engine's best-effort names. Each is read back by a FRESH OPSIN
call to the input's full InChIKey. The 5-oxido spelling of the sydnone anion is the floor's
(``5-oxido-1,2,3-oxadiazol-3-ium``); the spelling is ``-3-ium-5-olate``, a separate
spelling lead that these tests do not pin.
"""
import pytest
from rdkit import Chem, RDLogger

from tests.support.jars import jar_or_skip

RDLogger.DisableLog("rdApp.*")

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _cation_class(smiles):
    """The classes of the cation sites of the KEKULE form, the drawing the engine classifies:
    RDKit perceives a sydnone or a tetrazolium as aromatic, whose N-N bond is not a double
    bond, and only the kekulized ring has the N+=N bond."""
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.ions import classify_cation
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    sites = get_ion_sites(mol)
    return [classify_cation(mol, c) for c in sites["cations"]]


@pytest.mark.parametrize("smiles", [
    "c1ccccc1[N+]#N",              # benzenediazonium
    "C[N+]#N",                     # methanediazonium example)
    "CC(=O)C([N+]#N)C(C)=O",       # 2,4-dioxopentane-3-diazonium example)
    "c1ccccc1N=[N+]",              # the charge-shifted twin: terminal N+ of R-N=N+ (no H)
])
def test_terminal_n2_group_is_still_diazonium(smiles):
    assert _cation_class(smiles) == ["diazonium"]


@pytest.mark.parametrize("smiles", [
    "c1ccccc1[NH+]=N",      # R-NH+=NH, protonated phenyldiazene
    "CC[NH+]=N",
    "c1ccccc1N=[NH2+]",     # R-N=NH2+
    "c1ccccc1[N+]=N",       # open-shell: RDKit puts the H on the terminal N (one radical electron on N+)
    "[NH+]#N",              # HN2+, the parent cation, no R on it
])
@pytest.mark.parametrize("explicit_h", [False, True])
def test_n2_unit_carrying_hydrogen_is_not_diazonium(smiles, explicit_h):
    """ (the Blue Book): the group is "-N2+", two nitrogens and nothing else
    on them. A hydrogen on either nitrogen makes a protonated diazene, whether the H is an
    implicit count or an atom in the graph."""
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.ions import classify_cation
    mol = Chem.MolFromSmiles(smiles)
    if explicit_h:
        mol = Chem.AddHs(mol)
    assert [classify_cation(mol, c) for c in get_ion_sites(mol)["cations"]] == ["aminium"]


def test_the_radical_carve_out_keeps_its_behaviour():
    """``route_charged`` clears the two spurious radical electrons that RDKit puts on the
    terminal N+ of the twin ``R-N=N+`` (a valence-model artifact), keyed on the 'diazonium'
    class, so the twin is named benzenediazonium. The open-shell ``R-N+=NH`` (one radical
    electron, an H on the terminal N) never reached that carve-out, and does not now: it is
    declined here exactly as before. Both results are the base tree's."""
    from orthonym.rules import charged_router as cr
    assert cr.route_charged(Chem.MolFromSmiles("c1ccccc1N=[N+]"), "pin") == "benzenediazonium"
    assert cr.route_charged(Chem.MolFromSmiles("c1ccccc1[N+]#N"), "pin") == "benzenediazonium"
    assert cr.route_charged(Chem.MolFromSmiles("c1ccccc1[N+]=N"), "pin") == ""


def test_both_sites_of_a_bis_diazonium_are_diazonium():
    assert _cation_class("N#[N+]c1ccc([N+]#N)cc1") == ["diazonium", "diazonium"]


@pytest.mark.parametrize("smiles", [
    "C[N+]1=NOC(=C1C)[O-]",                      # 3,4-dimethyl-1,2,3-oxadiazol-3-ium-5-olate (sydnone)
    "c1ccccc1[N+]2=NOC(=N2)[O-]",                # 1,2,3,4-oxatriazol-3-ium-5-olate
    "c1ccccc1N2N=C(C#N)N=[N+]2c3ccccc3",         # 5-cyano-2,3-diphenyl-2H-tetrazol-3-ium
    "CC(C)C1=NC[N+](=N1)C(C)C",                  # 3H-1,2,4-triazol-2-ium
    "CC1C(=C([N+](=N1)C)Cl)C(C)C",               # 3H-1,2-diazol-1-ium (pyrazolium)
])
def test_ring_n_plus_double_bonded_to_n_is_not_diazonium(smiles):
    """No N of the ring N+=N bond is terminal; the cascade gives the same class the
    aromatic drawing of the cation always had, 'aminium' (a skeletal ring -ium)."""
    assert _cation_class(smiles) == ["aminium"]


@pytest.mark.parametrize("smiles", [
    "C[N+](C)=N",        # 1,1-dimethyldiazenium: R2N+=NH, the N+ carries two substituents
    "CN=[N+](C)C",       # 1,1,2-trimethyldiazenium: no terminal N at all
])
def test_acyclic_n_plus_double_bonded_to_n_is_not_diazonium_unless_terminal(smiles):
    """ names the -N2+ group attached to ONE parent atom, R-N+#N. An N+ with two
    substituents besides the N partner is a substituted diazenium, not that group."""
    assert _cation_class(smiles) == ["aminium"]


# (smiles, best-effort name): the name is read back by a fresh OPSIN call to the input's InChIKey
@pytest.mark.parametrize("smiles,name", [
    # sydnone: N+=N and the O- on the ring carbon zwitterion, floor spelling 5-oxido)
    ("CCCC(=O)N(C)CC1=C(ON=[N+]1C)[O-]",
     "4-{[butanoyl(methyl)amino]methyl}-3-methyl-5-oxido-1,2,3-oxadiazol-3-ium"),
    ("C[N+]1=NOC(=C1CNCCCN(C)C)[O-]",
     "4-({[3-(dimethylamino)propyl]amino}methyl)-3-methyl-5-oxido-1,2,3-oxadiazol-3-ium"),
    # oxatriazolium with the exocyclic O-
    ("C1=CC(=CC(=C1)[N+]2=NOC(=N2)[O-])C(F)(F)F",
     "1-(5-oxido-1,2,3,4-oxatriazol-3-ium-3-yl)-3-(trifluoromethyl)benzene"),
    # tetrazolium, no anion: a single-sign ring cation
    ("COC1=CC=C(C=C1)N2N=C(N=[N+]2C3=CC=C(C=C3)OC)C#N",
     "4-[5-cyano-3-(4-methoxyphenyl)-2H-tetrazol-3-ium-2-yl]-1-methoxybenzene"),
    # 1,2,3-oxadiazol-3-ium as a substituent parent, no anion
    ("CC(C1=C(ON=[N+]1C2CCCCCC2)NC(=O)C)O",
     "[5-acetamido-4-(1-hydroxyethyl)-1,2,3-oxadiazol-3-ium-3-yl]cycloheptane"),
    # 1,2,4-triazol-2-ium and 1,2-diazol-1-ium. The second is the engine's current label
    # (pin_verified), not a PIN claim: "Retained names of heteromonocycles"
    # (:8109), "pyrazole (1H-isomer shown; the PIN is 1H-pyrazole)" (:8154), so the
    # Hantzsch-Widman spelling '1,2-diazol-1-ium' is not the preferred name.
    ("CC(C)C1=NC[N+](=N1)C(C)C",
     "2,5-bis(propan-2-yl)-3H-1,2,4-triazol-2-ium"),
    ("CC1C(=C([N+](=N1)C)Cl)C(C)C",
     "5-chloro-1,3-dimethyl-4-(propan-2-yl)-3H-1,2-diazol-1-ium"),
])
def test_ring_n2_cation_is_named_and_reads_back_exact(smiles, name):
    jar_or_skip()
    from tests.support.rt_assert import name_best_effort, rt_exact_name
    got = name_best_effort(smiles)
    assert got["name"] == name, got
    assert got["tier"] != "abstain", got
    assert rt_exact_name(smiles) == name


@pytest.mark.parametrize("smiles,name", [
    # R-NH+=NH and R-N=NH2+ were 'diazonium' by bond order, which the emitter declined (a
    # hydrogen on the terminal N is not the -N2+ group); they are protonated diazenes
    #: 'ium' on the parent hydride) and are named on the diazene skeleton
    ("c1ccccc1[NH+]=N", "(1,2-diazaeth-1-en-1-ium-1-yl)benzene"),
    ("CC[NH+]=N", "1,2-diazabut-1-en-2-ium"),
    ("c1ccccc1N=[NH2+]", "(1,2-diazaeth-1-en-2-ium-1-yl)benzene"),
])
def test_protonated_diazene_is_named_and_reads_back_exact(smiles, name):
    jar_or_skip()
    from tests.support.rt_assert import name_best_effort, rt_exact_name
    got = name_best_effort(smiles)
    assert got["name"] == name, got
    assert got["tier"] != "abstain", got
    assert rt_exact_name(smiles) == name


@pytest.mark.parametrize("smiles,name", [
    ("c1ccccc1[N+]#N", "benzenediazonium"),
    ("C[N+]#N", "methanediazonium"),
    ("CC(=O)C([N+]#N)C(C)=O", "2,4-dioxopentane-3-diazonium"),   # example
    ("N#[N+]c1ccc([N+]#N)cc1", "benzene-1,4-bis(diazonium)"),     # example
])
def test_real_diazonium_names_are_unchanged(smiles, name):
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
