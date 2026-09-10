"""Task -- a ``_name_r_group`` refusal must never be silently dropped.

``composer._name_r_group`` returns ``None`` when it cannot PROVE a fragment is
describable. Since (``) removed the carbon-count fabrication that
used to paper over those cases, that ``None`` is honest and load-bearing.

Eight of the seventeen call sites consumed it as::

    sub_name = _name_r_group(mol, nidx, exclude_atoms=core)
    if sub_name: # <-- the refusal vanishes here
        n_subs.append(sub_name)

which is indistinguishable, downstream, from a nitrogen that never carried a
substituent. The result is a **silent atom drop**: a name for a molecule the
input is not.

Every test below asserts at the PRODUCER, not at the pipeline output. That is
deliberate and is the same reasoning as
``test_thiourea_bridging_and_parent.py::test_try_name_thiourea_refuses_r3...``:
on the default path the OPSIN/ validity gate suppresses all of these to
``unknown organic compound``, so a whole-molecule assertion would pass even with
the defect fully present. The gate must not be what saves us.

Measured dispositions of all 17 call sites, and the before/after emissions, are
recorded in `internal notes`.
"""

import pytest
from rdkit import Chem

from orthonym.assembly.composer import (
    _name_carbamate,
    _name_carbamic_acid,
    _named_n_substituents,
    _try_name_guanidine,
    _try_name_urea,
)


# --------------------------------------------------------------------------
# Reproducers. Each has an N-substituent that _name_r_group genuinely refuses.
# --------------------------------------------------------------------------

# The oxo analogue of the thiourea residue R3. Its N'-substituent is the
# 1-(tert-butyldiazenyl)cyclohexyl group, which _name_r_group declines.
# 20 heavy atoms; the defect emitted 'N-tert-butylurea' (8 heavy atoms).
UREA_R3 = "CC(C)(C)NC(=O)NC1(CCCCC1)N=NC(C)(C)C"
GUANIDINE_R3 = "CC(C)(C)NC(=N)NC1(CCCCC1)N=NC(C)(C)C"
CARBAMATE_R3 = "COC(=O)NC1(CCCCC1)N=NC(C)(C)C"

# 11 heavy atoms; the defect emitted the bare retained name 'carbamic acid'
# (4 heavy atoms), dropping the whole phosphate arm.
CARBAMIC_PHOSPHATE = "OC(=O)NCCOP(=O)(O)O"

# The phospholipid from TaskT-count-suspects.md. Before its phosphate arm
# was FABRICATED as 'N-tritetracontyl' (a C43 chain the molecule does not
# contain); after the refusal was honest and this caller deleted the arm,
# emitting '...-N-methyl-2-(stearoyloxy)propanamine' -- no phosphorus at all.
PHOSPHOLIPID = (
    "CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)OCC(COP(=O)(O)OCCNC)"
    "OC(=O)CCCCCCCCCCCCCCCCC"
)


def _features_for(smiles, handler_name):
    """The MolecularFeatures the dispatcher hands ``handler_name``.

    Captured through the real pipeline so the object under test is the one the
    handler actually receives, and filtered to the WHOLE molecule (the pipeline
    also names sub-fragments through the same builders).
    """
    import orthonym.assembly.composer as C
    from orthonym import Orthonym as _OS

    n_atoms = Chem.MolFromSmiles(smiles).GetNumAtoms()
    grabbed = []
    orig = getattr(C, handler_name)

    def spy(f):
        if getattr(f, "mol", None) is not None and f.mol.GetNumAtoms() == n_atoms:
            grabbed.append(f)
        return orig(f)

    setattr(C, handler_name, spy)
    try:
        _OS(style="pin").name_tiered(smiles)
    finally:
        setattr(C, handler_name, orig)
    return grabbed[0] if grabbed else None


def _heavy(smiles):
    return Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()


# --------------------------------------------------------------------------
# The shared primitive
# --------------------------------------------------------------------------

def test_helper_refuses_rather_than_shortening_the_list():
    """A refused substituent yields None -- NOT a shorter list.

    This is the whole point of the helper: ```` must keep meaning "this
    nitrogen is genuinely unsubstituted", so the refusal needs its own value.
    """
    mol = Chem.MolFromSmiles(UREA_R3)
    # SMARTS-independent: locate the urea core directly.
    core_match = mol.GetSubstructMatch(Chem.MolFromSmarts("[NX3][CX3](=O)[NX3]"))
    assert core_match, "the urea core must be present for this fixture"
    n1, c, o, n2 = core_match
    core = {n1, c, o, n2}

    subs_n1 = _named_n_substituents(mol, n1, c, core)
    subs_n2 = _named_n_substituents(mol, n2, c, core)
    # One nitrogen carries tert-butyl (nameable), the other the cyclohexyl /
    # diazenyl half (refused). Exactly one side must come back None.
    assert None in (subs_n1, subs_n2), (subs_n1, subs_n2)


def test_helper_returns_empty_list_for_a_genuinely_unsubstituted_nitrogen():
    """ and None must not be conflated in the other direction either."""
    mol = Chem.MolFromSmiles("NC(=O)N")           # urea itself
    core_match = mol.GetSubstructMatch(Chem.MolFromSmarts("[NX3][CX3](=O)[NX3]"))
    n1, c, o, n2 = core_match
    core = {n1, c, o, n2}
    assert _named_n_substituents(mol, n1, c, core) == []
    assert _named_n_substituents(mol, n2, c, core) == []


# --------------------------------------------------------------------------
# The five silently-omitting producers
# --------------------------------------------------------------------------

def test_urea_builder_refuses_instead_of_dropping_half_the_molecule():
    """composer.py urea N1/N2 sites.

    The defect returned 'N-tert-butylurea' for a 20-heavy-atom molecule,
    deleting the entire cyclohexyl/diazenyl half -- precisely the drop the
    thiourea sibling's test docstring predicted this builder would make.
    """
    features = _features_for(UREA_R3, "_try_name_urea")
    assert features is not None, "the urea builder was never reached"
    out = _try_name_urea(features)
    assert out is None, out
    assert _heavy(UREA_R3) == 20


def test_guanidine_builder_refuses_instead_of_dropping_half_the_molecule():
    """composer.py guanidine N / N' / N'' sites (defect: 'N-tert-butylguanidine')."""
    features = _features_for(GUANIDINE_R3, "_try_name_guanidine")
    assert features is not None, "the guanidine builder was never reached"
    out = _try_name_guanidine(features)
    assert out is None, out


def test_carbamic_acid_builder_refuses_instead_of_dropping_the_phosphate_arm():
    """composer.py carbamic-acid N site.

    The defect emitted the bare retained name 'carbamic acid' -- 4 heavy atoms
    for an 11-heavy-atom input.
    """
    features = _features_for(CARBAMIC_PHOSPHATE, "_name_carbamic_acid")
    assert features is not None, "the carbamic acid builder was never reached"
    out = _name_carbamic_acid(features)
    assert out is None, out
    assert _heavy(CARBAMIC_PHOSPHATE) == 11
    assert _heavy("NC(=O)O") == 4          # what 'carbamic acid' denotes


def test_carbamate_builder_refuses_instead_of_dropping_the_n_substituent():
    """composer.py carbamate N site (defect fell through to 'methyl carbamate')."""
    features = _features_for(CARBAMATE_R3, "_name_carbamate")
    assert features is not None, "the carbamate builder was never reached"
    out = _name_carbamate(features)
    assert out is None, out


def test_principal_amine_refuses_instead_of_deleting_the_phosphate_arm():
    """rules/polyfunctional.py principal-amine N-substituent site.

    This block is the ONLY producer of those prefixes -- the branch is
    deliberately excluded from the substituent walk -- so skipping an
    un-nameable substituent deletes atoms rather than degrading the name.
    """
    from orthonym.rules.polyfunctional import name_polyfunctional

    import orthonym.rules.polyfunctional as P

    n_atoms = Chem.MolFromSmiles(PHOSPHOLIPID).GetNumAtoms()
    grabbed = []
    orig = P.name_polyfunctional

    def spy(f):
        if getattr(f, "mol", None) is not None and f.mol.GetNumAtoms() == n_atoms:
            grabbed.append(f)
        return orig(f)

    P.name_polyfunctional = spy
    try:
        from orthonym import Orthonym as _OS

        _OS(style="pin").name_tiered(PHOSPHOLIPID)
    finally:
        P.name_polyfunctional = orig

    assert grabbed, "name_polyfunctional was never reached for the phospholipid"
    out = name_polyfunctional(grabbed[0])
    # The pre-fix emission was
    # '1-((11Z,14Z)-icosa-11,14-dienoyloxy)-N-methyl-2-(stearoyloxy)propanamine'
    # -- a name containing no phosphorus for a phosphate-bearing molecule.
    assert out is None or "propanamine" not in out, out


# --------------------------------------------------------------------------
# Non-regression: a NAMEABLE substituent must be untouched by all of the above
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CNC(=O)NC", "N,N'-dimethylurea"),
        ("NC(=O)N", "urea"),
        ("CNC(=S)NC", "N,N'-dimethylthiourea"),
        ("NC(=N)N", "guanidine"),
        ("CNC(=N)N", "N-methylguanidine"),
        ("COC(=O)NC", "methyl N-methylcarbamate"),
        ("OC(=O)NC", "N-methylcarbamic acid"),
        ("OC(=O)NCCOC", "N-(2-methoxyethyl)carbamic acid"),
        ("CCCNC(=O)NCCOC", "N-(2-methoxyethyl)-N'-propylurea"),
        ("CN(C)CCO", "2-(dimethylamino)ethan-1-ol"),
        ("OC(=O)CCN(C)C", "3-(dimethylamino)propanoic acid"),
    ],
)
def test_nameable_substituents_are_byte_identical(smiles, expected):
    """The helper returns the same list the old inline loop did whenever no
    substituent is refused, so every one of these must be unchanged."""
    from orthonym import name_compound

    res = name_compound(smiles)
    assert getattr(res, "name", res) == expected
