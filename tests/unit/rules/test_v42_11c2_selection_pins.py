""" a phase (batch 11C2) — selection / retained-parent / demotion PIN spelling.

Three shipped fixes (the batch's other six items were deferred as separate defect
classes / a likely gold-integrity issue — see task-11C2-report.md):

#6 demote non-PIN trivial names to the systematic PIN (deny in
    ``iupac_2013_pin_list.json``):
      * 2,4,6-trinitrotoluene -> 2-methyl-1,3,5-trinitrobenzene
         (the Blue Book) "toluene (PIN) (no substitution...)": toluene is not
        substitutable in a PIN, so a ring-substituted toluene is named on benzene.
        BOTH the hand-curated '2,4,6-trinitrotoluene' AND the OPSIN alias
        'trinitrotoluene' are denied (denying only the first UNMASKS the second,
        which promotes) — a project rule.
      * anthra-1,2-quinone -> anthracene-1,2-dione
        the Blue Book "No retained quinone names are used as preferred IUPAC names";
        the Blue Book "anthracene-1,2-dione (PIN) 1,2-anthraquinone".

#31 retained ``piperidine``, not the Hantzsch-Widman systematic
    ``azinane``, for a saturated heteromonocyclic SPIRO component names
    each component by its own PIN). Real site: ``spiro.py::
    _name_hw_monocycle_component`` saturated branch (the brief's ``name_heterocycle``
    at spiro.py:2262 was OFF-PATH — it already returns 'piperidine'). Numbering is
    shared with the HW stem, so only the ring word swaps; extends to
    pyrrolidine/morpholine/piperazine.

#17 retained ``hydrazinyl`` prefix for a bare terminal -NH-NH2
    substituent leaf, not the compositional ``aminoamino``. Real site:
    ``substituent_enumerator.py::name_substituent`` N-rooted amino branch. Only the
    clean single-token-swap row ships; the nested ``hydrazinylmethylidene`` row
    (66.4.2.3.3) is a DIFFERENT producer and was deferred.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.data import pin_policy

_OPSIN_JAR = Path(__file__).resolve().parents[2].parent / "opsin-cli-2.9.0-jar-with-dependencies.jar"
_OPSIN_OK = shutil.which("java") is not None and _OPSIN_JAR.is_file()


def _ikey(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m) if m is not None else None


def _name_general_fallback(smiles: str) -> str:
    """Name via the general_fallback config in a FRESH interpreter.

    The general_fallback tier is measured by bb_conformance with one namer per
    worker process (bb_measure.py). A fresh subprocess reproduces that isolated
    state: mixing the DEFAULT-config ``name_compound`` calls above with a
    general_fallback namer in the SAME pytest process pollutes global selection
    state and can let an unverified wrong candidate win (an artifact of the
    unverified tier's relaxed self-consistency gate, unrelated to this fix; the
    fix is verified 0-wrong in a fresh process)."""
    code = (
        "from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')\n"
        "from orthonym import Orthonym\n"
        "n=Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)\n"
        f"print(n.name_tiered({smiles!r}).get('name'))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""


# ---------------------------------------------------------------------------
# #6 — demotion of non-PIN trivial names (default PIN path)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    # (the Blue Book): toluene not substitutable in a PIN.
    ("Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
     "2-methyl-1,3,5-trinitrobenzene"),
    # the Blue Book / the Blue Book: no retained quinone name is a PIN.
    ("O=C1C=Cc2cc3ccccc3cc2C1=O", "anthracene-1,2-dione"),
])
def test_demote_trivial_to_systematic_pin(smiles, expected):
    assert name_compound(smiles) == expected


def test_tnt_deny_covers_both_surfaces():
    """Both the hand-curated headline and the OPSIN alias must be denied
    (a project rule: denying only the hand-curated form unmasks the promotable
    OPSIN 'trinitrotoluene')."""
    assert "2,4,6-trinitrotoluene" in pin_policy.PIN_DENY
    assert "trinitrotoluene" in pin_policy.PIN_DENY
    assert "anthra-1,2-quinone" in pin_policy.PIN_DENY


def test_trinitrobenzene_component_unaffected():
    """Regression: the 1,3,5-trinitrobenzene component of the addition compounds
    (bb rows 14.8.1, MATCH) still builds correctly."""
    assert name_compound("O=[N+]([O-])c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1") == \
        "1,3,5-trinitrobenzene"


# ---------------------------------------------------------------------------
# #31 — retained piperidine (not azinane) in a spiro component (default PIN path)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    # / — the target row (bb 24.5.1).
    ("c1ccc2c(c1)Oc1ccccc1C21CCNCC1", "spiro[piperidine-4,9'-xanthene]"),
    # class extension: pyrrolidine (retained) not azolidine.
    ("c1ccc2c(c1)Oc1ccccc1C21CCNC1", "spiro[pyrrolidine-3,9'-xanthene]"),
])
def test_spiro_retained_heterocycle_component(smiles, expected):
    assert name_compound(smiles) == expected


def test_spiro_nonretained_component_stays_hw():
    """Regression: oxane's HW word equals its retained-table value, so the swap is
    a no-op (spiro[oxane-...]); a genuinely non-retained ring is untouched."""
    assert name_compound("c1ccc2c(c1)Oc1ccccc1C21CCOCC1") == \
        "spiro[oxane-4,9'-xanthene]"


@pytest.mark.skipif(not _OPSIN_OK, reason="OPSIN JVM required")
def test_spiro_piperidine_roundtrips():
    from orthonym.validation.opsin_roundtrip import opsin_parse
    name = name_compound("c1ccc2c(c1)Oc1ccccc1C21CCNCC1")
    osmi = opsin_parse(name)
    assert osmi and _ikey(osmi) == _ikey("c1ccc2c(c1)Oc1ccccc1C21CCNCC1")


# ---------------------------------------------------------------------------
# #17 — hydrazinyl prefix (not aminoamino) for a bare -NH-NH2 leaf.
# Emits at systematic_verified only under general_fallback (invariant-16 gap).
# ---------------------------------------------------------------------------
def test_hydrazinyl_prefix_leaf():
    assert _name_general_fallback("NNC(=O)CC(=O)O") == "3-hydrazinyl-3-oxopropanoic acid"


@pytest.mark.parametrize("smiles,expected", [
    # Regression: acyl hydrazide keeps its senior -hydrazide suffix (the -NH-NH2
    # must NOT be pulled out as a hydrazinyl prefix).
    ("CC(=N)NN", "ethanimidohydrazide"),
    ("N=CNN", "methanimidohydrazide"),
    ("CC(=O)NN", "acetohydrazide"),
    ("O=C(NN)c1ccccc1", "benzohydrazide"),
])
def test_acyl_hydrazide_suffix_unaffected(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.skipif(not _OPSIN_OK, reason="OPSIN JVM required")
def test_hydrazinyl_roundtrips():
    name = _name_general_fallback("NNC(=O)CC(=O)O")
    from orthonym.validation.opsin_roundtrip import opsin_parse
    osmi = opsin_parse(name)
    assert osmi and _ikey(osmi) == _ikey("NNC(=O)CC(=O)O")
