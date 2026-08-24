"""v36 core-namer: ring-bearing compound-substituent silent-drop witnesses.

At HEAD (before the v36 core-namer fix) each witness ABSTAINS at the best-effort
tier: a ring-parent substituent enumerator names the parent but silently drops a
ring-bearing compound substituent it cannot render, and SELF-01 then suppresses
the atom-incomplete partial name (final output ``unknown organic compound``).

Two root causes were found (full spy trail:
`` + the fix
commits):

1. The fused-heterocycle substituent enumerator
   (``rules/fused_rings.py::_identify_fused_substituent``) never delegated a
   ring-bearing compound substituent rooted at a HETEROATOM linker.  The
   pantoprazole C2 ``-S(=O)-CH2-[pyridine]`` half is rooted at the sulfinyl S, so
   it never reached the carbon branch's ring delegate at ``:1724`` -- it hit the
   sulfur branch, which has no ring delegate, returned ``None``, and the caller
   silently ``continue``d.
2. The chalcogen prefix primitive
   (``assembly/substituent_prefix_forms.py::get_sulfanyl_prefix``) named a
   sulfinyl ``-S(=O)-`` as ``sulfanyl``, silently dropping the =O -> a wrong
   (reduced-thioether) molecule that SELF-01 suppressed to an abstention.

The fix threads the best-effort context into the un-wired ring-parent substituent
enumerators and makes ``get_sulfanyl_prefix`` fail closed on a non-divalent
chalcogen (so the fragment degrades to the oxo-preserving replacement name).
Each witness must then emit a name that round-trips through OPSIN to the full
input InChIKey (0-wrong).

Every witness below was A/B-verified with `` to ABSTAIN at
HEAD and to NAME + round-trip with the fix.  Fresh process per witness
(warm-cache hazard -- the contributor guide).
"""
import json
import subprocess
import sys

import pytest

# --- fused-heterocycle parent, ring-bearing compound substituent (Tasks 2/4) --
# The pantoprazole class: a benzimidazole parent carrying an S(=O)-rooted
# ring-bearing compound substituent at C2.  All three ABSTAIN at HEAD.
FUSED_HETEROCYCLE_WITNESSES = [
    # pantoprazole (trifluoromethoxy variant) -- the charter anchor.
    "COc1ccnc(CS(=O)c2nc3ccc(OC(F)(F)F)cc3[nH]2)c1OC",
    # pantoprazole (difluoromethoxy variant, the C4-spy shape).
    "COc1ccnc(CS(=O)c2nc3ccc(OC(F)F)cc3[nH]2)c1OC",
    # a benzimidazole-sulfinyl-benzyl (trifluoromethyl on the benzo ring).
    "O=S(Cc1ccccc1)c1nc2ccc(C(F)(F)F)cc2[nH]1",
]

WITNESSES = list(FUSED_HETEROCYCLE_WITNESSES)


def _cn_rt(smiles):
    """Best-effort name + OPSIN full-InChIKey round-trip, in a FRESH process.

    Returns ``{"name": str|None, "tier": str, "rt": bool|None}``.  A subprocess
    per call is deliberate: the fragment memo cache is per-process and warm-cache
    state has repeatedly produced false greens in this project (the contributor guide).
    """
    out = subprocess.run(
        [sys.executable, __file__, smiles],
        capture_output=True, text=True, timeout=180,
    )
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and '"name"' in line:
            return json.loads(line)
    raise AssertionError(
        f"no result JSON for {smiles!r}\n"
        f"stdout:\n{out.stdout[-2000:]}\nstderr:\n{out.stderr[-2000:]}"
    )


@pytest.mark.parametrize("smiles", FUSED_HETEROCYCLE_WITNESSES)
def test_fused_heterocycle_ring_bearing_substituent_names_and_rt(smiles):
    """A ring-bearing compound substituent on a fused-heterocycle parent must be
    named (not dropped) and the whole name must round-trip (0-wrong)."""
    res = _cn_rt(smiles)
    name = res.get("name")
    assert name and "unknown" not in name, (
        f"abstained (ring-bearing compound substituent silently dropped): "
        f"{smiles} -> {res}")
    assert res.get("rt") is True, (
        f"emitted a name that does NOT round-trip to the input InChIKey: "
        f"{smiles} -> {name!r} (rt={res.get('rt')})")


def _name_one(smiles):
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="v36-cn-test"):
        from orthonym.namer import Orthonym
        from orthonym.validation import opsin_roundtrip_check
        namer = Orthonym(
            style="pin", general_fallback=True,
            general_fallback_unverified=True, allow_aromatic_general=True)
        res = namer.name_tiered(smiles)
        name = res.get("name")
        rt = None
        if name and "unknown" not in name:
            rt = bool(opsin_roundtrip_check(smiles, name)["passed"])
        return {"name": name, "tier": res.get("tier"), "rt": rt}


if __name__ == "__main__":
    print(json.dumps(_name_one(sys.argv[1])))
