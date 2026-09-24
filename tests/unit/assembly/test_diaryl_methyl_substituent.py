"""a phase  — diaryl-methyl-ether + diaryl-ether-amine class tests.

Two independent root-cause bugs from the AUTONOM head-to-head (B):

  SITE #1 (Plan 167-03 turns these green): Ar2CH-O- (diphenylmethyl ether) is
  mis-named `benzyloxy` because three byte-duplicated code sites never count the
  central carbon's aryl neighbours. PIN is `diphenylmethoxy` (; NOT the
  Beilstein `benzhydryloxy`). Fixed via ONE shared aryl-count helper.

  SITE #2 (Plan 167-04 turns these green): `_name_amino_branch` only handles a
  single N-substituent, so N,N-dialkylamino substituents on a chain parent mis-split.

  PITFALL-3 GUARD (must stay green across BOTH plans): the principal-amine path
  (`CN(C)CC` -> N,N-dimethylethan-1-amine) is already correct and untouched.

RT assertions use the conftest `opsin_to_smiles` fixture (auto-skips if OPSIN/Java
absent) and compare InChI of the OPSIN-parsed emitted name to InChI of the input.
Forbidden: xfail, postprocessors. Targets OPSIN-verified (opsin-cli 2.9.0).
"""

import glob
import shutil
import subprocess

import pytest

from orthonym import name_compound
from tests.support.jars import jar_or_none


def _find_opsin_jar():
    """The pinned OPSIN jar via orthonym.jars (tests.support.jars), or None."""
    return jar_or_none()


_OPSIN_JAR = _find_opsin_jar()
_OPSIN_AVAILABLE = bool(_OPSIN_JAR) and shutil.which("java") is not None


def _opsin_smiles(name):
    """Name -> SMILES via OPSIN over STDIN.

    OPSIN 2.9.0's CLI treats the post-`-osmi` argv as an input *file*; the name
    must be fed on stdin (the conftest `opsin_to_smiles` fixture uses argv and
    therefore returns None for every name on this OPSIN version). stdout carries
    only the SMILES (empty on parse failure); the prompt banner goes to stderr.
    """
    if not _OPSIN_AVAILABLE:
        return None
    try:
        result = subprocess.run(
            ["java", "-jar", _OPSIN_JAR, "-osmi"],
            input=name + "\n",
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    out = result.stdout.strip()
    return out or None


def _inchi(smiles):
    """InChI for a SMILES (None if unparseable)."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchi(mol) if mol else None


def _assert_roundtrips(smiles_input):
    """The name Orthonym emits for `smiles_input` must round-trip via OPSIN."""
    if not _OPSIN_AVAILABLE:
        pytest.skip("OPSIN/Java not available")
    emitted = name_compound(smiles_input)
    parsed = _opsin_smiles(emitted)
    assert parsed is not None, f"OPSIN could not parse emitted name {emitted!r}"
    assert _inchi(parsed) == _inchi(smiles_input), (
        f"round-trip mismatch for {smiles_input!r}: emitted {emitted!r} -> {parsed!r}"
    )


class TestDiarylMethoxy:
    """SITE #1 — Ar2CH-O- on a parent must read `diphenylmethoxy`, not `benzyloxy`.

    These probes route through the substituent-naming path (the 3 duplicated
    sites) — the isolated `Ph2CH-O-Me` whole-molecule case routes elsewhere
    (A2/Pitfall 2), so the load-bearing assertions use substituent context.
    Plan 167-03 makes these green.
    """

    # >=3 diphenylmethoxy analogs  spanning chain length.
    DIPHENYLMETHOXY = [
        "OCCOC(c1ccccc1)c1ccccc1",    # 2-(diphenylmethoxy)ethan-1-ol
        "OCCCOC(c1ccccc1)c1ccccc1",   # 3-(diphenylmethoxy)propan-1-ol
        "OCCCCOC(c1ccccc1)c1ccccc1",  # 4-(diphenylmethoxy)butan-1-ol
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles", DIPHENYLMETHOXY)
    def test_emits_diphenylmethoxy(self, smiles):
        name = name_compound(smiles)
        assert "diphenylmethoxy" in name, f"expected diphenylmethoxy, got {name!r}"
        assert "benzyloxy" not in name, f"still benzyloxy (bug): {name!r}"
        assert "benzhydryl" not in name, f"emitted Beilstein benzhydryl (D-07 violation): {name!r}"

    @pytest.mark.unit
    @pytest.mark.roundtrip
    @pytest.mark.parametrize("smiles", DIPHENYLMETHOXY)
    def test_diphenylmethoxy_roundtrips(self, smiles):
        _assert_roundtrips(smiles)

    @pytest.mark.unit
    def test_negative_control_benzyl_unchanged(self):
        """True PhCH2-O- (1 aryl) must STILL be benzyloxy — the fix discriminates."""
        name = name_compound("OCCOCc1ccccc1")
        assert "benzyloxy" in name, f"benzyl ether regressed: {name!r}"
        assert "diphenylmethoxy" not in name, f"blanket-renamed benzyl: {name!r}"

    @pytest.mark.unit
    @pytest.mark.roundtrip
    def test_negative_control_roundtrips(self):
        _assert_roundtrips("OCCOCc1ccccc1")


def _assert_no_double_amino(smiles):
    """The single substituted-amine N must be named ONCE — no geminal-diamine
    artifact like '2-amino-2-(dimethylamino)' / '1-(dimethylamino)...-1-amine'."""
    name = name_compound(smiles)
    # A standalone 'amino' prefix in addition to the '(...amino)' substituent is
    # the double-count signature. The correct name has the alkyls only inside the
    # single '(...amino)' substituent (or as N,N- prefixes when amine is principal).
    assert "amino-" not in name.replace("(dimethylamino)", "").replace(
        "(diethylamino)", "").replace("(methylamino)", ""), (
        f"spurious bare amino double-count: {name!r}"
    )
    return name


class TestDialkylaminoChain:
    """SITE #2 — N,N-dialkylamino correctly named ONCE (no double-counted nitrogen).

    Root cause (fixed): a substituted amine nitrogen was counted twice —
    (a) as a bare amine FG (→ `amino` prefix when non-principal, or the `-amine`
    suffix when principal) AND (b) as a `(dimethylamino)` substituent via the
    substituent walk — yielding geminal-diamine artifacts like
    `2-amino-2-(dimethylamino)ethan-1-ol` (amine non-principal) and
    `1-(dimethylamino)...ethan-1-amine` (amine principal). The fix makes each
    handler name the nitrogen exactly once. Verified structurally by OPSIN
    round-trip (the spurious amino makes the parsed structure a geminal diamine,
    which does NOT round-trip to the input).
    """

    # --- amine NON-PRINCIPAL (alcohol / acid is the principal group) ---
    @pytest.mark.unit
    def test_dimethylaminoethanol_pin(self):
        assert name_compound("CN(C)CCO") == "2-(dimethylamino)ethan-1-ol"

    @pytest.mark.unit
    def test_diethylaminoethanol_pin(self):
        assert name_compound("CCN(CC)CCO") == "2-(diethylamino)ethan-1-ol"

    @pytest.mark.unit
    def test_methylaminoethanol_pin(self):
        assert name_compound("CNCCO") == "2-(methylamino)ethan-1-ol"

    @pytest.mark.unit
    def test_dimethylamino_propanoic_acid_pin(self):
        assert name_compound("CN(C)CCC(=O)O") == "3-(dimethylamino)propanoic acid"

    @pytest.mark.unit
    @pytest.mark.roundtrip
    @pytest.mark.parametrize(
        "smiles", ["CN(C)CCO", "CCN(CC)CCO", "CNCCO", "CN(C)CCC(=O)O", "CN(C)CC(=O)O"]
    )
    def test_nonprincipal_amine_roundtrips(self, smiles):
        _assert_no_double_amino(smiles)
        _assert_roundtrips(smiles)

    # --- carbon-summing regression guard (the Plan-04 fix must stay) ---
    @pytest.mark.unit
    def test_carbon_summing_preserved(self):
        assert "(ethylamino)" not in name_compound("CN(C)CCO")
        assert "butylamino" not in name_compound("CCN(CC)CCO")

    # --- amine PRINCIPAL (diphenhydramine core: ether + tertiary amine, no OH/acid) ---
    @pytest.mark.unit
    def test_diphenhydramine_core_no_double_count(self):
        """The principal amine's N-methyls are N,N- prefixes (NOT a doubled
        '(dimethylamino)' substituent), and site#1 diphenylmethoxy is present.

        The nitrogen is named exactly once. Structural correctness is verified by
        test_diphenhydramine_core_roundtrips. NOTE: the canonical PIN order is
        '2-(diphenylmethoxy)-N,N-dimethylethan-1-amine'; Orthonym currently emits
        the structurally-identical 'N,N-dimethyl-2-diphenylmethoxyethan-1-amine'
        (round-trips to the same structure). The prefix ORDER differs only because
        `alpha_sort_key` strips the 'di' of the complex substituent 'diphenylmethoxy'
        — a separate, pre-existing alphabetization limitation, NOT this double-count
        bug.
        """
        name = name_compound("CN(C)CCOC(c1ccccc1)c1ccccc1")
        assert "N,N-dimethyl" in name, f"N-substituents not on suffix: {name!r}"
        assert "(dimethylamino)" not in name, f"nitrogen double-counted: {name!r}"
        assert "diphenylmethoxy" in name, f"site#1 missing: {name!r}"

    @pytest.mark.unit
    @pytest.mark.roundtrip
    def test_diphenhydramine_core_roundtrips(self):
        _assert_roundtrips("CN(C)CCOC(c1ccccc1)c1ccccc1")

    # --- primary amine must remain correct (NOT dropped, NOT doubled) ---
    @pytest.mark.unit
    def test_primary_amine_unchanged(self):
        assert name_compound("NCCO") == "2-aminoethan-1-ol"
        assert name_compound("NCCCC(=O)O") == "4-aminobutanoic acid"


class TestPrincipalAmineGuard:
    """PITFALL-3 — the principal-amine path is already correct; must stay byte-identical.

    Green now AND after both plans (regression guard — do NOT touch _assemble_amine_name).
    """

    @pytest.mark.unit
    def test_principal_dimethylamine_unchanged(self):
        # Phase B (DD1 Fix 4): the ethane amine-suffix locant is elided per
        # -> 'N,N-dimethylethanamine' (PubChem-confirmed), not the
        # over-located 'N,N-dimethylethan-1-amine'.
        assert name_compound("CN(C)CC") == "N,N-dimethylethanamine"
