""" — hydrazinyl retained substituent prefix on an N-heterocycle.

``H2N-NH-`` is the retained PREFERRED substituent prefix **hydrazinyl** (P-62.4).
The ring-substituent collector (``rules/heterocycles.py::_identify_hetero_substituent``)
used to flag a ring-borne ``-NH-NH2`` as *unnameable*, so the whole heterocycle
candidate declined and the molecule abstained. BB 19376 ``2-hydrazinylpyridine``
(PIN) is the target.

## Seniority scope (P-44.1) — the load-bearing guard

Hydrazine's senior skeletal element is **N** (top of the P-44.1.2 order
``N >... > O > S >... > C``). A ring is the senior parent over the 2-N hydrazine
chain ONLY when it also holds N -- then P-44.1.2.2's ring-senior-to-chain tie-break
fires (pyridine / pyrimidine / pyrrole / dihydroimidazole -> ``2-hydrazinyl<ring>``).
A ring whose senior element is JUNIOR to N is junior to the hydrazine chain, so
hydrazine stays the PARENT and the PIN is ``<ring-yl>hydrazine``:

  * all-carbon benzene -> ``phenylhydrazine`` (retained PIN, P-68.3.1.2), and
  * BB 18950 ``1-(2H-pyran-3-yl)-2-(silolan-2-yl)hydrazine`` (PIN) keeps hydrazine
    the parent even against two O/Si heterocyclic ring substituents.

So the fix is gated on a nitrogen in the parent ring system; an S/O heterocycle
(thiophene / furan) is NOT named ``2-hydrazinyl<ring>`` (rules/polyazane owns the
hydrazine-parent spelling for those). This gate is the invariant-9 unmask guard:
without it a naive hydrazinyl producer would ship the WRONG PIN on those rings
(clean round-trip, SELF-01-invisible).
"""

import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestHydrazinylOnNitrogenHeterocycle:
    """The fix: an N-heterocycle bearing ``-NH-NH2`` names it as a hydrazinyl
    prefix at the ring locant (BB 19376 `2-hydrazinylpyridine` (PIN))."""

    def test_2_hydrazinylpyridine(self):
        # The BB 19376 (PIN) target row. Was: abstain (unknown organic compound).
        assert name_compound("NNc1ccccn1") == "2-hydrazinylpyridine"

    def test_2_hydrazinyl_dihydroimidazole(self):
        # The second target row. Was: abstain.
        assert name_compound("NNC1=NCCN1") == "2-hydrazinyl-4,5-dihydro-1H-imidazole"

    def test_4_hydrazinylpyridine(self):
        # whole-class: the 4-locant isomer of the pyridine target.
        assert name_compound("NNc1ccncc1") == "4-hydrazinylpyridine"

    def test_2_hydrazinylpyrimidine(self):
        # whole-class: a diazine ring (2 N) still resolves to the ring parent.
        assert name_compound("NNc1ncccn1") == "2-hydrazinylpyrimidine"

    def test_3_hydrazinyl_pyrrole(self):
        # whole-class: an azole (1 N) ring parent.
        assert name_compound("NNc1cc[nH]c1") == "3-hydrazinyl-1H-pyrrole"


@pytest.mark.unit
class TestHydrazineParentUnchanged:
    """Invariant-9 unmask canaries: the fix must NOT touch any case where
    hydrazine (or a hydrazide) is the correct parent. These are the BEFORE names,
    re-asserted AFTER the fix."""

    def test_phenylhydrazine_unchanged(self):
        # All-carbon ring -> hydrazine parent (retained PIN, P-68.3.1.2). The
        # risk the brief flagged: a naive producer would ship `hydrazinylbenzene`.
        assert name_compound("NNc1ccccc1") == "phenylhydrazine"

    def test_methylhydrazine_unchanged(self):
        assert name_compound("CNN") == "methylhydrazine"

    def test_acetohydrazide_unchanged(self):
        # -C(=O)-NH-NH2 is a hydrazide (different FG), never a hydrazinyl prefix.
        assert name_compound("CC(=O)NN") == "acetohydrazide"


@pytest.mark.unit
class TestSeniorityScopeGuard:
    """P-44.1 seniority: an S/O heterocycle is JUNIOR to the hydrazine chain, so it
    must NOT be spelled `2-hydrazinyl<ring>` (hydrazine is the parent). The gate
    keys on a nitrogen in the ring; without it these ship the wrong PIN."""

    def test_thiophene_not_hydrazinyl_prefix(self):
        # Thiophene's senior element is S (< N); hydrazine is the parent. The fix
        # must not fabricate the wrong ring-parent PIN.
        assert name_compound("NNc1cccs1") != "2-hydrazinylthiophene"

    def test_furan_not_hydrazinyl_prefix(self):
        assert name_compound("NNc1ccco1") != "2-hydrazinylfuran"


@pytest.mark.unit
class TestChainHydrazinylUnchanged:
    """Regression pins: the chain path (a senior chain carrying a principal group)
    already spelled hydrazinyl and is untouched by the ring-collector fix."""

    def test_4_hydrazinylbutanoic_acid(self):
        assert name_compound("NNCCCC(=O)O") == "4-hydrazinylbutanoic acid"

    def test_2_hydrazinylethanol(self):
        assert name_compound("NNCCO") == "2-hydrazinylethan-1-ol"
