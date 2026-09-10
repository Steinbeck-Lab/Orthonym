""" a phase enabler (P-74.1.3 / P-73): a pendant ONIUM cation branch off a
chain carbon must be nameable as a locanted substituent prefix.

Root cause: ``_name_polyfunctional_acyclic_substituent``
(assembly/substituent_naming.py) is the builder behind ``2-hydroxyethyl`` /
``2-chloroethyl`` -- but until this fix it declined (returned None) as soon as
its Pass-2 loop reached ANY charged atom, even one confined to a pendant
branch cation_to_prefix could already name. This test suite proves:

  (1) the new capability fires for the branch shape (direct test),
  (2) it composes end-to-end through the public namer (integration, OPSIN
      gate disabled by default per this repo's conftest),
  (3) neutral compound substituents are BYTE-IDENTICAL (regression) -- the
      hard constraint for a change to this core module.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.substituent_enumerator import name_substituent
from orthonym.assembly.substituent_naming import name_substituent_fragment


def _terminal_attach(m):
    """The lone terminal carbon whose only heavy neighbour is carbon."""
    candidates = [
        a.GetIdx() for a in m.GetAtoms()
        if a.GetSymbol() == 'C' and a.GetDegree() == 1
        and all(n.GetSymbol() == 'C' for n in a.GetNeighbors())
    ]
    return candidates[-1]


def test_cationic_branch_substituent():
    """C[N+](C)(C)CC rooted at the terminal ethyl carbon ->
    2-(trimethylazaniumyl)ethyl -- the exact RT-verified target in the spec
    (docs/superpowers/specs/2026-08-17--phase3-cationic-substituent-capability.md)."""
    m = Chem.MolFromSmiles("C[N+](C)(C)CC")
    attach = _terminal_attach(m)
    frag = frozenset(a.GetIdx() for a in m.GetAtoms())
    assert name_substituent(m, frag, attach) == "2-(trimethylazaniumyl)ethyl"


def test_cationic_branch_substituent_three_carbon_chain():
    """One more carbon between the attachment and the onium branch pushes
    the locant to 3 -- proves the locant is read from chain POSITION, not
    hardcoded."""
    m = Chem.MolFromSmiles("C[N+](C)(C)CCC")
    attach = _terminal_attach(m)
    frag = frozenset(a.GetIdx() for a in m.GetAtoms())
    assert name_substituent(m, frag, attach) == "3-(trimethylazaniumyl)propyl"


def test_cationic_branch_one_carbon_locant_elided():
    """Boundary case: 1-carbon backbone with cation directly attached.
    Locant is elided for a 1-carbon substituent (CH2 becomes methyl, not
    1-methyl). This tests the direct-attach boundary where host == attach_idx
    within Pass 1d's component search."""
    m = Chem.MolFromSmiles("C[N+](C)(C)C")
    attach = 0  # the first carbon (the methyl that is the sole backbone)
    frag = frozenset(a.GetIdx() for a in m.GetAtoms())
    assert name_substituent_fragment(m, frag, attach, set()) == "(trimethylazaniumyl)methyl"


def test_direct_attach_shape_unaffected_by_generic_namer():
    """The DIRECT-ATTACHMENT shape (cation itself IS attach_idx) is NOT
    reached through this generic ``name_substituent`` entry point at all --
    measured on HEAD (pre-fix): ``name_substituent`` calls
    ``name_substituent_fragment(..., parent_chain=[])`` (Tier 4, empty
    parent list), which blanks the ``parent_set`` Step 2e's direct-attach
    branch depends on, so it always falls to the DROP-26 decline regardless
    of structure. The REAL working direct-attach mechanism
    (`trimethylazaniumyl`, `4-(trimethylazaniumyl)butanoate`) lives entirely
    in ``rules/charged_router.py`` / ``rules/ions.py``, calling
    ``cation_to_prefix`` directly -- never through this cascade -- and is
    covered by ``tests/unit/test_zwitterion_salt.py``. This test proves the
    new BRANCH pass added here does not change that (already-declining)
    behaviour for the direct-attach shape: my new pass only fires when it
    can find a HOST chain atom distinct from the cation itself, which is
    impossible when the cation IS ``attach_idx`` (the component search blocks
    on it), so this stays a clean no-op, byte-identical before and after."""
    m = Chem.MolFromSmiles("C[N+](C)(C)CC")
    cat_idx = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'N')
    # Just the cation + its 3 methyl branches (the ethyl chain plays the role
    # of the external "parent" here, mirroring how a real substituent
    # fragment excludes the parent atoms).
    frag = frozenset([0, 1, 2, 3])
    assert name_substituent(m, frag, cat_idx) == "substituent"


# ---------------------------------------------------------------------------
# Integration: through the public namer. OPSIN gate is OFF by default in this
# repo's conftest (autouse fixture), so no JVM is required here; @opsin_gate
# turns it ON to also prove round-trip validity.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.mark.opsin_gate
def test_choline_sulfate_via_capability(namer):
    out = namer.name("C[N+](C)(C)CCOS(=O)(=O)[O-]")
    # B1 (charged_router wiring) is a separate follow-on task (spec section 6);
    # this enabler alone may still leave the zwitterion abstaining. Either the
    # RT-valid target or an honest abstention is acceptable here -- never a
    # wrong-molecule name.
    assert out in ("2-(trimethylazaniumyl)ethyl sulfate", "unknown organic compound")


# ---------------------------------------------------------------------------
# Regression: neutral compound substituents must be BYTE-IDENTICAL. This is
# the hard constraint (CLAUDE.md a project rule / the task's #1 requirement) --
# _name_polyfunctional_acyclic_substituent is a CORE namer and the new pass
# must be a pure no-op whenever no atom in the fragment carries a positive
# formal charge.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi,expected", [
    ("OCC", "2-hydroxyethyl"),
    ("ClCC", "2-chloroethyl"),
    ("NCC", "2-aminoethyl"),
    ("OCCC", "3-hydroxypropyl"),
    ("OC(=O)CC", "2-carboxyethyl"),
])
def test_neutral_compound_substituent_unchanged(smi, expected):
    m = Chem.MolFromSmiles(smi)
    attach = _terminal_attach(m)
    frag = frozenset(a.GetIdx() for a in m.GetAtoms())
    assert name_substituent(m, frag, attach) == expected


def test_neutral_branched_fragment_unaffected():
    """A neutral, charge-free branched fragment never enters the new pass's
    loop at all (it only looks at atoms with formal charge > 0) -- sanity
    check that it still resolves to a stable, non-sentinel name."""
    m = Chem.MolFromSmiles("CC(C)(C)CC")  # neutral acyclic control, no charge
    attach = _terminal_attach(m)
    frag = frozenset(a.GetIdx() for a in m.GetAtoms())
    result = name_substituent(m, frag, attach)
    assert result and result != "substituent"


def test_phosphatidylcholine_full_pipeline_unchanged():
    """The ONE place `2-(trimethylazaniumyl)ethyl` already appears in this
    codebase pre-fix is a hardcoded lipid head-group lookup
    (`rules/lipids.py:274`, keyed to a detected "choline" head group), not
    the general recursive substituent namer this task extends. Prove the two
    mechanisms coexist without collision: the lipid pipeline's full name is
    unchanged after adding the new general pass."""
    from orthonym import name_compound
    smi = ("CCCCCCCCCCCCCCCC(=O)OCC(COP([O-])(=O)OCC[N+](C)(C)C)"
           "OC(=O)CCCCCCCCCCCCCCC")
    assert name_compound(smi) == (
        "[2,3-bis(hexadecanoyloxy)propyl] 2-(trimethylazaniumyl)ethyl "
        "phosphate"
    )
