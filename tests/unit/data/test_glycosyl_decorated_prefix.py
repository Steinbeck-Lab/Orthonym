import sys; sys.path.insert(0, "src")
import pytest
from rdkit import Chem
from orthonym import Orthonym

# The whole-molecule OPSIN validity gate is disabled test-suite-wide by an
# autouse fixture (tests/conftest.py:369); these tests round-trip the emitted
# name themselves and need the namer's own gate ON so it matches production
# (CLI/library) behaviour -- otherwise an internally-rejected candidate name
# leaks through instead of the namer falling back to the next tier.
pytestmark = pytest.mark.opsin_gate

_BE = Orthonym(style="pin", general_fallback=True,
                general_fallback_unverified=True, allow_aromatic_general=True)


def _rt(smi):
    from rdkit.Chem import inchi
    import subprocess, tempfile, os
    name = _BE.name(smi)
    if not name:
        return name, False
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(name + "\n"); p = fh.name
    try:
        out = subprocess.run(["java", "-jar", "opsin-cli-2.9.0-jar-with-dependencies.jar", "-osmi"],
                             stdin=open(p), capture_output=True, text=True, timeout=120).stdout.strip()
    finally:
        os.unlink(p)
    ok = bool(out) and inchi.MolToInchiKey(Chem.MolFromSmiles(out)) == inchi.MolToInchiKey(Chem.MolFromSmiles(smi))
    return name, ok


def test_n_acetylglucosaminyl_arm_on_senior_aglycone_rt():
    # beta-D-GlcNAc glycosidically O-linked to 4'-hydroxyacetophenone: the
    # aglycone's ketone is SENIOR to hydroxy, so P-102.6.1.2 requires the
    # decorated-glycosyloxy substituent-PREFIX form on the ketone parent
    # (name_glycosyloxy_aglycone, sugar_names.py:3251) rather than the
    # sugar-as-oxane-parent fallback. SMILES built atom-for-atom off the
    # catalog's verified beta-D-2-acetamido-2-deoxy-glucopyranose entry
    # (sugar_names.py:134-136) with the free anomeric -OH's H replaced by the
    # aglycone, so the stereochemistry is guaranteed correct D-GlcNAc (a
    # hand-written diastereomer here would silently miss the sugar catalog
    # lookup entirely -- verified while developing this test).
    smi = "CC(=O)N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]1Oc2ccc(C(C)=O)cc2"
    name, ok = _rt(smi)
    assert name and "oxan" not in name, name   # decoration not dropped to a bare/general oxane
    assert "acetamido" in name, name           # the N-acetyl decoration survives into the name
    assert ok, f"did not round-trip: {name}"


def test_n_acetylglucosaminyl_arm_on_non_senior_aglycone_unaffected():
    # Same decorated sugar, but O-linked to a plain phenol: hydroxy is NOT
    # senior to hydroxy, so P-102.6.1.2's substituent-prefix form does not
    # apply (BB requires a SENIOR aglycone group) -- name_glycosyloxy_aglycone
    # correctly declines and an unrelated (unaffected-by-this-fix) general
    # construction wins. This just guards that Lever 2b's fix does not leak
    # into the non-senior-aglycone case.
    smi = "CC(=O)N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]1Oc2ccc(O)cc2"
    name, ok = _rt(smi)
    assert name and ok, name


def test_arbutin_unchanged_regression():
    # clean glucoside must still name (sugar-as-oxane-parent path), unchanged
    smi = "Oc1ccc(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc1"
    name, ok = _rt(smi)
    assert name and ok, name
