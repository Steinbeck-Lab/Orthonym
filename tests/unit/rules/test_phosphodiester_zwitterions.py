"""The zwitterion of a phosphoric diester with a quaternary ammonium group in one of its groups
(``rules.ester_zwitterions``).

 "Acid esters of organic acids" (the Blue Book): acid esters of inorganic acids are named by the
method of 'hydrogen salts'; the anion of a diester is the name of the diester without 'hydrogen' ('diethyl phosphate').
 "Anionic and cationic centers on different parent structures" (:42462): the cationic part is cited as a prefix to
the anionic parent, here inside the name of the group that holds it. The groups are cited in alphanumerical order
: '(2S,3R)-2-amino-3-hydroxyoctadecyl' before '2-(N,N-dimethylmethanaminiumyl)ethyl'.

The nitrogen with three methyl groups is the '-aminiumyl' PIN prefix of the book's own example,:42296;
'(N,N-dimethylmethanaminiumyl)acetate (PIN)',:42473): the default tier spells it so and verifies the name through its twin
'2-(trimethylazaniumyl)ethyl...', the spelling OPSIN reads (``validation.twin_spellings``, ``aminiumyl_azaniumyl``; the
maintainers' ruling of 2026-11-01), so the name is a PIN. The best-effort tier spells the form OPSIN reads, 'azaniumyl', a
systematic name.

Every name is read back by an independent OPSIN call (the full standard InChIKey; the twin literal for the PIN spelling).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.perception.ions import get_ion_sites
from orthonym.rules import charged_router
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

ROWS = [
    # (smiles, the PIN of the default tier, the general name of the best-effort tier)
    ("CCOP(=O)([O-])OCC[N+](C)(C)C", "2-(N,N-dimethylmethanaminiumyl)ethyl ethyl phosphate",
     "ethyl 2-(trimethylazaniumyl)ethyl phosphate"),
    # the phosphatidylcholines and sphingomyelins of the dev2000 and druglike2000 populations
    ("CCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COP(=O)([O-])OCC[N+](C)(C)C",
     "2-(N,N-dimethylmethanaminiumyl)ethyl (2R)-2-hydroxy-3-(icosanoyloxy)propyl phosphate",
     "(2R)-2-hydroxy-3-(icosanoyloxy)propyl 2-(trimethylazaniumyl)ethyl phosphate"),
    ("CCCCCCCCCCCCCCC[C@@H](O)[C@@H](N)COP(=O)([O-])OCC[N+](C)(C)C",
     "(2S,3R)-2-amino-3-hydroxyoctadecyl 2-(N,N-dimethylmethanaminiumyl)ethyl phosphate",
     "(2S,3R)-2-amino-3-hydroxyoctadecyl 2-(trimethylazaniumyl)ethyl phosphate"),
    ("CCCCCCCCCC(=O)OC[C@H](CO)OP(=O)([O-])OCC[N+](C)(C)C",
     "(2S)-1-(decanoyloxy)-3-hydroxypropan-2-yl 2-(N,N-dimethylmethanaminiumyl)ethyl phosphate",
     "(2S)-1-(decanoyloxy)-3-hydroxypropan-2-yl 2-(trimethylazaniumyl)ethyl phosphate"),
]
#: the sphingomyelin with an amide group in its group: the PIN tier has no name for the group, the best-effort tier names it
BEST_EFFORT_ONLY = [
    ("CCCCCCCCCCCCCCCC(=O)N[C@@H](COP(=O)([O-])OCC[N+](C)(C)C)[C@H](O)CCCCCCCCCCCCCCC",
     "(2S,3R)-2-hexadecanamido-3-hydroxyoctadecyl 2-(trimethylazaniumyl)ethyl phosphate"),
]


def _row(smiles, tier):
    return Orthonym(style="pin", **(_emit_tier_flags(tier) if tier != "pin" else {})).name_tiered(smiles)


@pytest.mark.parametrize("smiles, pin, general", ROWS)
def test_the_default_tier_names_the_diester_with_the_aminiumyl_pin(smiles, pin, general):
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"], row["is_pin"], row["opsin"]) == (pin, "pin_verified", True, "verified_twin"), row
    assert name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("smiles, pin, general", ROWS)
def test_the_best_effort_tier_names_it_with_the_azaniumyl_prefix_below_the_pin(smiles, pin, general):
    row = _row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (general, "systematic_verified"), row
    assert name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("smiles, general", BEST_EFFORT_ONLY)
def test_a_group_the_pin_tier_cannot_name_is_named_at_best_effort(smiles, general):
    assert _row(smiles, "pin")["tier"] == "abstain"
    row = _row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (general, "systematic_verified"), row
    assert name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("smiles, pin, general", ROWS)
def test_the_name_is_the_one_the_route_builds(smiles, pin, general):
    assert charged_router.route_charged(Chem.MolFromSmiles(smiles), "pin") == pin


@pytest.mark.parametrize("smiles", [
    "C[N+](C)(C)CCOP(=O)(O)OCC",               # a free hydroxy group instead of the anion
    "[NH3+]CCOP(=O)([O-])OCC",                 # a protonated amine, not a quaternary ammonium group
    "C[N+](C)(C)CCOP(=O)([O-])OCC.[Na+]",      # (not one fragment; the group of the cation is cut off)
])
def test_the_builder_declines_other_shapes(smiles):
    from orthonym.rules.ester_zwitterions import phosphodiester_zwitterion_name
    mol = Chem.MolFromSmiles(smiles)
    p = next(a for a in mol.GetAtoms() if a.GetSymbol() == "P")
    ester = [n.GetIdx() for n in p.GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() == 2]
    cations = get_ion_sites(mol)["cations"]
    cation = cations[0]["atom_idx"] if cations else 0
    assert phosphodiester_zwitterion_name(mol, cation, p.GetIdx(), ester) == ""


def test_the_monoester_zwitterion_takes_the_aminiumyl_pin_spelling_too():
    smiles = "C[N+](C)(C)CCOP(=O)(O)[O-]"
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == ("2-(N,N-dimethylmethanaminiumyl)ethyl hydrogen phosphate", "pin_verified"), row
    row = _row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == ("2-(trimethylazaniumyl)ethyl hydrogen phosphate", "systematic_verified"), row
