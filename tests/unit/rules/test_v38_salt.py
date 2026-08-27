"""v38 SALT increment — two sited salt-naming sub-gaps.

Sub-gap 1 — ROUTE_MISSES: `name_salt` produces a correct, RT-verified salt
name for a STEREO-bearing ionic salt (drug.[H+].[X-]), but `name_compound`
discarded it. Root cause: the SELF-01 self-consistency verdict
(`namer._self_consistency_verdict`) reached its RegistrationHash-tautomer
fallback for stereo inputs (na>0, so the `na==0 -> ok` short-circuit did not
apply) and there the ionic input form (`...[Cl-].[H+]`) and the neutral
molecular form OPSIN re-perceives (`...Cl`) hash DIFFERENTLY even though they
share ONE full InChIKey — so a correct `... hydrochloride` name was suppressed.
Fix: an equal FULL standard InChIKey (the headline round-trip identity) is the
strongest identity signal and short-circuits the verdict to "ok" (0-wrong-safe:
it only turns a spurious mismatch into ok, never the reverse). Non-stereo salts
(pyridine.HCl) already worked via the `na==0` path — this closes the stereo case.

Sub-gap 2 — SALT_ASSEMBLY_GAP ([H+]-orphaned organic polyacid): an organic
oxoanion written ionically with bare protons and NO neutral base
(`dicarboxylate.[H+].[H+]`) hit the `if h_plus_frags: return ''` 0-wrong guard
(the amine-protonation branch needs a neutral base, absent here). Fix: when all
cations are `[H+]`, there is no neutral base, and every anion is an organic
oxoanion (a deprotonated -O(-) acid site), REATTACH the protons to the acid
sites (reconstruct the neutral free acid) and name THAT via the ordinary namer;
the top-level SELF-01 gate RT-verifies it (identical full InChIKey), so a wrong
reconstruction fails closed. The halide `[H+].[Cl-]` guard is untouched.

Determinism on `CanonicalRankAtoms`. 0-wrong ABSOLUTE (RT-verify or abstain).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import name_compound
from orthonym.rules.salts import name_salt, _reattach_protons_to_acids


def _ik(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    return inchi.MolToInchiKey(m) if m is not None else None


def _rt_ik(name: str):
    """Full InChIKey of the OPSIN re-perception of ``name`` (None if it does
    not parse). Mirrors the project's headline round-trip metric."""
    from orthonym.namer import _validity_gate_name_to_smiles
    smi = _validity_gate_name_to_smiles(name)
    return _ik(smi) if smi else None


def _rt_matches(name: str, input_smiles: str) -> bool:
    return bool(name) and _rt_ik(name) == _ik(input_smiles)


# ---------------------------------------------------------------------------
# Sub-gap 1 — ROUTE_MISSES (stereo-bearing ionic salt names were discarded)
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
class TestSubgap1RouteMisses:
    def test_adrenaline_hydrochloride_end_to_end(self):
        # adrenaline.HCl written ionically. name_salt already RT-verifies this;
        # name_compound used to return 'unknown organic compound'.
        smi = "CNC[C@H](O)c1ccc(O)c(O)c1.[Cl-].[H+]"
        name = name_compound(smi, style="pin")
        assert name == (
            "4-[(R)-1-hydroxy-2-(methylamino)ethyl]benzene-1,2-diol "
            "hydrochloride"
        )
        assert _rt_matches(name, smi)

    def test_s_amphetamine_hydrochloride(self):
        smi = "C[C@H](N)Cc1ccccc1.[Cl-].[H+]"
        name = name_compound(smi, style="pin")
        assert name == "(2S)-1-phenylpropan-2-amine hydrochloride"
        assert _rt_matches(name, smi)

    def test_r_amphetamine_hydrochloride(self):
        smi = "C[C@@H](N)Cc1ccccc1.[Cl-].[H+]"
        name = name_compound(smi, style="pin")
        assert name == "(2R)-1-phenylpropan-2-amine hydrochloride"
        assert _rt_matches(name, smi)

    def test_stereo_hydrobromide_route_miss(self):
        # (R)-configured secondary alcohol amine as a hydrobromide.
        smi = "CNC[C@H](O)c1ccc(O)c(O)c1.[Br-].[H+]"
        name = name_compound(smi, style="pin")
        assert name and "unknown" not in name
        assert name.endswith("hydrobromide")
        assert _rt_matches(name, smi)

    def test_nonstereo_route_miss_still_works(self):
        # Control: the achiral case always worked (na==0 short-circuit); it must
        # keep working after the full-InChIKey short-circuit is added.
        smi = "CCN.[Cl-].[H+]"
        assert name_compound(smi, style="pin") == "ethanamine hydrochloride"


# ---------------------------------------------------------------------------
# Sub-gap 2 — SALT_ASSEMBLY_GAP ([H+]-orphaned organic polyacid)
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
class TestSubgap2ProtonOrphanedPolyacid:
    def test_tetrahydroxy_dicarboxylate_reconstructs_to_free_acid(self):
        smi = ("O=C([O-])[C@@H](O)[C@H](O)[C@H](O)[C@H](O)C(=O)[O-]"
               ".[H+].[H+]")
        # name_salt itself now reconstructs + names the neutral free acid.
        assert name_salt(Chem.MolFromSmiles(smi)) == "D-altraric acid"
        name = name_compound(smi, style="pin")
        assert name == "D-altraric acid"
        assert _rt_matches(name, smi)

    def test_tartrate_dianion_plus_two_protons(self):
        smi = "[O-]C(=O)[C@H](O)[C@@H](O)C(=O)[O-].[H+].[H+]"
        name = name_compound(smi, style="pin")
        assert name == "(2R,3R)-2,3-dihydroxybutanedioic acid"
        assert _rt_matches(name, smi)

    def test_oxalate_dianion_plus_two_protons(self):
        # Reconstructs to the systematic PIN (ethanedioic acid), not the retained
        # 'oxalic acid' -- the ordinary namer's parent decision, RT-verified.
        smi = "[O-]C(=O)C(=O)[O-].[H+].[H+]"
        name = name_compound(smi, style="pin")
        assert name == "ethanedioic acid"
        assert _rt_matches(name, smi)

    def test_adipate_dianion_plus_two_protons(self):
        smi = "[O-]C(=O)CCCCC(=O)[O-].[H+].[H+]"
        name = name_compound(smi, style="pin")
        assert name == "hexanedioic acid"
        assert _rt_matches(name, smi)

    def test_single_charge_carboxylate_plus_one_proton(self):
        # A monoanion + one [H+], no base -> reconstruct the neutral acid.
        smi = "CC(=O)[O-].[H+]"
        name = name_compound(smi, style="pin")
        assert name == "acetic acid"
        assert _rt_matches(name, smi)


# ---------------------------------------------------------------------------
# 0-wrong scope guards — the reconstruction must NOT fire on the halide /
# unrecognized-coformer cases the existing `if h_plus_frags: return ''` guard
# protects.
# ---------------------------------------------------------------------------

class TestScopeGuards:
    def test_bare_halide_proton_still_abstains(self):
        # [H+].[Cl-] with no organic base and no oxoanion: no acid O to
        # protonate -> the reattach branch does NOT fire; the halide 0-wrong
        # guard keeps it abstaining (never 'chloride', never a wrong species).
        assert name_salt(Chem.MolFromSmiles("[H+].[Cl-]")) == ""

    def test_reattach_helper_declines_non_oxygen_anion(self):
        # The acid-reattach reconstruction is scoped to all-OXYGEN anions, so a
        # halide (or any non-O anion) returns None -> the halide 0-wrong guard
        # keeps its behaviour. A carboxylate anion DOES reconstruct.
        assert _reattach_protons_to_acids(Chem.MolFromSmiles("[Cl-].[H+]")) is None
        acetic = _reattach_protons_to_acids(Chem.MolFromSmiles("CC(=O)[O-].[H+]"))
        assert acetic is not None
        assert Chem.MolToSmiles(acetic) == "CC(=O)O"

    def test_reattach_helper_declines_multi_fragment_mixture(self):
        # Two separate mono-acid anions + 2 [H+] is a MIXTURE of free acids, not a
        # single parent acid -> decline (never name a mixture as one compound).
        mix = "CC(=O)[O-].CCC(=O)[O-].[H+].[H+]"
        assert _reattach_protons_to_acids(Chem.MolFromSmiles(mix)) is None


# ---------------------------------------------------------------------------
# Determinism — identical output across SMILES orderings.
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
class TestDeterminism:
    @pytest.mark.parametrize("orderings", [
        ("CNC[C@H](O)c1ccc(O)c(O)c1.[Cl-].[H+]",
         "[Cl-].[H+].CNC[C@H](O)c1ccc(O)c(O)c1",
         "[H+].CNC[C@H](O)c1ccc(O)c(O)c1.[Cl-]"),
        ("O=C([O-])[C@@H](O)[C@H](O)[C@H](O)[C@H](O)C(=O)[O-].[H+].[H+]",
         "[H+].[H+].O=C([O-])[C@@H](O)[C@H](O)[C@H](O)[C@H](O)C(=O)[O-]",
         "[H+].O=C([O-])[C@@H](O)[C@H](O)[C@H](O)[C@H](O)C(=O)[O-].[H+]"),
    ])
    def test_same_name_across_orderings(self, orderings):
        names = {name_compound(s, style="pin") for s in orderings}
        assert len(names) == 1, names
        (only,) = names
        assert only and "unknown" not in only


# ---------------------------------------------------------------------------
# Byte-identity controls — currently-naming salts must not regress.
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
class TestByteIdentityControls:
    @pytest.mark.parametrize("smi,expected", [
        ("CC(=O)[O-].[Na+]", "sodium acetate"),
        ("[NH4+].[Cl-]", "ammonium chloride"),
        ("CC(=O)[O-].CC(=O)[O-].[Ca+2]", "calcium diacetate"),
        ("c1cc[nH+]cc1.[Cl-]", "pyridin-1-ium chloride"),
        ("CC(=O)[O-].[Na+].O", "sodium acetate monohydrate"),
        ("CCO", "ethanol"),
    ])
    def test_control(self, smi, expected):
        assert name_compound(smi, style="pin") == expected
