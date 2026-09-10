"""a phase: aromatic N-oxide locant fix.

``assembly/composer.py::_name_aromatic_n_oxide`` used to emit
``f"{base_name} 1-oxide"`` with the locant HARDCODED to "1" and zero locant
computation. This is harmless for a monoazine (pyridine: the ring's single N
is always locant 1 by the ring's own definition -- no other heteroatom can
compete) and for a SYMMETRIC unsubstituted diazine/triazine (every oxidisable
N is symmetry-equivalent, so "1" is unambiguous by construction). It is NOT
harmless for a substituted, symmetry-broken diazine: the reduced parent's own
independently-chosen numbering does not always place the oxidised nitrogen
at position 1, so the hardcoded "1" produced a name with a real locant
mismatch -- caught downstream by SELF-01 (different molecule) and demoted to
a total abstention, rather than degrading to a correct-but-uglier name.

a trace: internal notes
Plan: docs/superpowers/plans/2026-08-12-phase3-correctness-tail-hardening.md sec 3C

Every "was WRONG / now correct" case below was hand round-trip-verified
(name -> OPSIN -> InChIKey == input) before being asserted here; see the
inline comments recording that verification.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi
from orthonym import name_compound
from orthonym.validation.opsin_roundtrip import opsin_parse

# This module is directly ABOUT the OPSIN validity/SELF-01 gate's interaction
# with the N-oxide handler (the old hardcoded-locant bug was rescued from
# 0-wrong only by SELF-01 catching the wrong molecule downstream -- see the
# a trace doc). Per tests/conftest.py's documented convention, the gate is
# suite-wide OFF by default and must be explicitly re-enabled here, or the
# "abstains honestly" / "was rescued to abstain" claims below would be
# green-but-blind.
pytestmark = pytest.mark.opsin_gate


def _full_rt(smiles: str, name: str) -> bool:
    """OPSIN parse `name`, compare full InChIKey against `smiles`."""
    if not name:
        return False
    o = opsin_parse(name)
    if not o:
        return False
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(o))


# ---------------------------------------------------------------------------
# Regression: symmetric / structurally-unambiguous cases stay byte-identical.
# (No ring position competes with the oxidised N for locant 1, so the fix
# must not change these -- they are exactly the cases the old hardcoded "1"
# already got right.)
# ---------------------------------------------------------------------------

class TestRegressionByteIdentical:
    def test_pyridine_n_oxide_unchanged(self):
        smi = "[O-][n+]1ccccc1"
        name = name_compound(smi)
        assert name == "pyridine 1-oxide", name
        assert _full_rt(smi, name), name

    def test_4_methylpyridine_n_oxide_unchanged(self):
        smi = "Cc1cc[n+]([O-])cc1"
        name = name_compound(smi)
        assert name == "4-methylpyridine 1-oxide", name
        assert _full_rt(smi, name), name

    def test_2_4_dimethylpyridine_n_oxide_unchanged(self):
        smi = "Cc1cc[n+]([O-])c(C)c1"
        name = name_compound(smi)
        assert name == "2,4-dimethylpyridine 1-oxide", name
        assert _full_rt(smi, name), name

    def test_trimethylamine_n_oxide_unchanged(self):
        # Aliphatic path (_name_aliphatic_n_oxide) is untouched by this fix
        # (no numeric ring locant to hardcode) -- included as a sibling
        # regression per the task's "check any sibling" instruction.
        smi = "C[N+](C)([O-])C"
        name = name_compound(smi)
        assert name == "N,N-dimethylmethanamine N-oxide", name
        assert _full_rt(smi, name), name

    def test_pyrazine_symmetric_n_oxide_unchanged(self):
        smi = "[O-][n+]1ccncc1"
        name = name_compound(smi)
        assert name == "pyrazine 1-oxide", name
        assert _full_rt(smi, name), name

    def test_pyrimidine_symmetric_n_oxide_unchanged(self):
        smi = "[O-][n+]1cccnc1"
        name = name_compound(smi)
        assert name == "pyrimidine 1-oxide", name
        assert _full_rt(smi, name), name

    def test_2_methylpyrazine_n_oxide_regiochem_b_unchanged(self):
        # methyl adjacent to the OXIDISED N: base numbering (lowest locant to
        # methyl) and oxide-priority numbering coincide here -> still "1".
        smi = "Cc1cncc[n+]1[O-]"
        name = name_compound(smi)
        assert name == "2-methylpyrazine 1-oxide", name
        assert _full_rt(smi, name), name

    def test_4_methylpyrimidine_n_oxide_regiochem_a_unchanged(self):
        smi = "Cc1nc[n+]([O-])cc1"
        name = name_compound(smi)
        assert name == "4-methylpyrimidine 1-oxide", name
        assert _full_rt(smi, name), name


# ---------------------------------------------------------------------------
# The fix: asymmetric substituted ring N-oxides now NAME with the CORRECT
# (non-"1") locant instead of abstaining (RED before the fix: both emitted
# the sentinel "unknown organic compound", per the a trace table).
# ---------------------------------------------------------------------------

class TestAsymmetricRingNOxideNowNamesCorrectly:
    def test_2_methylpyrazine_n_oxide_regiochem_a(self):
        # Methyl is adjacent to the PLAIN (non-oxidised) ring N, two bonds
        # from the oxidised N. Hand-verified: opsin_parse("2-methylpyrazine
        # 1-oxide") reparses to the OTHER isomer (Cc1cncc[n+]1[O-]) --
        # confirming "1" is genuinely wrong here -- while opsin_parse(
        # "2-methylpyrazine 4-oxide") reparses back to THIS exact SMILES
        # (full InChIKey match). Before the fix this abstained entirely
        # (SELF-01 caught the wrong-locant "2-methylpyrazine 1-oxide" and
        # suppressed it to the sentinel).
        smi = "Cc1c[n+]([O-])ccn1"
        name = name_compound(smi)
        assert name is not None and "unknown" not in name, name
        assert name == "2-methylpyrazine 4-oxide", name
        assert _full_rt(smi, name), name

    def test_4_methylpyrimidine_n_oxide_regiochem_b(self):
        # Methyl is directly adjacent to the OXIDISED ring N here (the other
        # pyrimidine regiochemistry vs the "A" case above). Hand-verified:
        # opsin_parse("4-methylpyrimidine 1-oxide") reparses to the OTHER
        # isomer (Cc1nc[n+]([O-])cc1); opsin_parse("4-methylpyrimidine
        # 3-oxide") reparses back to THIS exact SMILES (full InChIKey
        # match). Before the fix this abstained entirely.
        smi = "Cc1ccnc[n+]1[O-]"
        name = name_compound(smi)
        assert name is not None and "unknown" not in name, name
        assert name == "4-methylpyrimidine 3-oxide", name
        assert _full_rt(smi, name), name
        # the wrong hardcoded locant must not appear
        assert "1-oxide" not in name


# ---------------------------------------------------------------------------
# Void guard (0-wrong): a bis-N-oxide (more than one simultaneously-oxidised
# ring N) has no "di-oxide" construction in this handler. It must decline
# outright -- never silently name only one of the two oxidised nitrogens
# with a bare "oxide" or a guessed single locant.
# ---------------------------------------------------------------------------

class TestVoidGuardAmbiguousMultiOxide:
    def test_pyrazine_1_4_dioxide_abstains_honestly(self):
        smi = "[O-][n+]1cc[n+]([O-])cc1"
        name = name_compound(smi)
        # Never emit a mono-oxide name that silently drops the second oxide,
        # and never a bare "pyrazine oxide" (ambiguous which N).
        assert name is None or "unknown" in name.lower(), (
            f"bis-N-oxide must abstain honestly, not guess a name: {name!r}"
        )
        if name is not None:
            assert "pyrazine oxide" not in name
            assert "pyrazine 1-oxide" not in name
