"""SL lane G: the moved asserted values come back when the lever that moved them is switched off.

Three expected values changed in lane G (``orthonym-skills:change-asserted-value``): the gold row
P62-ANILINO-RING-AND-N (G1) and the two COMPOUND rows of ``test_nq_oxoacid_group_routing`` (G3). A moved
value is credible when the test that holds it fails without the change that moved it. Each case runs in
a fresh interpreter (the namer memoises names within a process, so a patch made after a first call is not
seen), against the tree under test.

* G1, (the Blue Book) "Multiplicative prefixes... are the principal method for describing
  a multiplicity of identical features of a structure in chemical names"; the italic N before the numerals
  ,:3195). With the grouping off (``identical_prefixes_cited_apart``) AND the spelling
  check off, the old value '3-(4-methyl-N-methylanilino)propanenitrile' comes back as pin_verified: the
  grouping is the lever. With the grouping off and the check on, the check lowers the old spelling and the
  default tier declines, and the breadth tier names the molecule with the general '[methyl(4-methylphenyl)
  amino]' prefix: the check is the second lever. The row's value is UNVERIFIED by a verbatim example
  (the book prints no anilino prefix with the same substituent on the ring and on the nitrogen); it is
  rule-backed, no verbatim BB example.
* G3, (:29715) "only acetic acid, benzoic acid, and oxamic acid can be substituted", row:29725
  'acetic acid (PIN) ethanoic acid'. With ``retained_c2_name`` patched out of the general engine the two
  rows come back as '2-{[bis(...amino)phosphoryl]oxy}ethanoic acid'.
"""
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.opsin_gate

_PRELUDE = """
import json, sys
from orthonym import Orthonym
BEST = dict(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)
"""


def _fresh(code: str) -> dict:
    proc = subprocess.run([sys.executable, "-c", _PRELUDE + code], capture_output=True, text=True,
                          env=dict(os.environ), timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


ROW = "N#CCCN(C)c1ccc(C)cc1"
NEW = "3-(N,4-dimethylanilino)propanenitrile"
OLD = "3-(4-methyl-N-methylanilino)propanenitrile"


def test_the_gold_row_is_the_grouped_spelling_with_the_levers_on():
    out = _fresh(f"print(json.dumps(Orthonym().name_tiered({ROW!r})))")
    assert (out["name"], out["tier"]) == (NEW, "pin_verified"), out


def test_the_old_spelling_comes_back_with_the_grouping_and_the_spelling_check_off():
    out = _fresh(f"""
from orthonym.assembly.composition_primitives import identical_prefixes_cited_apart
from orthonym.validation import pin_spelling as ps
ps._ensure_loaded()
ps._REGISTRY.pop('P-16.3.1')
with identical_prefixes_cited_apart():
    print(json.dumps(Orthonym().name_tiered({ROW!r})))
""")
    assert (out["name"], out["tier"]) == (OLD, "pin_verified"), out


def test_the_old_spelling_is_lowered_by_the_check_when_only_the_grouping_is_off():
    out = _fresh(f"""
from orthonym.assembly.composition_primitives import identical_prefixes_cited_apart
with identical_prefixes_cited_apart():
    d = Orthonym().name_tiered({ROW!r})
    b = Orthonym(**BEST).name_tiered({ROW!r})
print(json.dumps([d, b]))
""")
    default, breadth = out
    assert default["tier"] == "abstain", default
    assert breadth["name"] == "3-[methyl(4-methylphenyl)amino]propanenitrile", breadth
    assert breadth["tier"] == "systematic_verified", breadth


@pytest.mark.parametrize("smiles,new,old", [
    ("CNP(=O)(NC)OCC(=O)O", "{[bis(methylamino)phosphoryl]oxy}acetic acid",
     "2-{[bis(methylamino)phosphoryl]oxy}ethanoic acid"),
    ("CN(C)P(=O)(N(C)C)OCC(=O)O", "{[bis(dimethylamino)phosphoryl]oxy}acetic acid",
     "2-{[bis(dimethylamino)phosphoryl]oxy}ethanoic acid"),
])
def test_the_two_carbon_acid_rows_come_back_as_ethanoic_acid_without_the_retained_word(smiles, new, old):
    with_lever = _fresh(f"print(json.dumps(Orthonym(**BEST).name_tiered({smiles!r})))")
    assert with_lever["name"] == new, with_lever
    # the engine imports the decision inside ``_assemble``, so the module attribute is what it reads
    mutated = _fresh(f"""
from orthonym.assembly import retained_c2_chain
retained_c2_chain.retained_c2_name = lambda **kw: None
print(json.dumps(Orthonym(**BEST).name_tiered({smiles!r})))
""")
    assert mutated["name"] == old, mutated
