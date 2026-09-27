"""task-123: 2-component hetero-fusion abstains recovered.

Root cause (measured, HEAD): ``_identify_ring_name`` (fusion_descriptors.py)
had no registry entry for either triazole isomer (1,2,3- or 1,2,4-), so a
triazole-fused pair returned child_name='' and
generate_systematic_name_for_fused_pair declined outright, and the engine
abstained to the 'unknown organic compound' sentinel instead of trying the
correct descriptor.

Fix: 1,2,3-triazole / 1,2,4-triazole added to MONOCYCLIC_COMPONENTS
(data/fusion_components.py) with the (the Blue Book)
bracketed-locant citation ('cite_locants') needed because both isomers
contract to the identical prefix 'triazolo'.

Governing rule: "General principles" (the Blue Book).
"""
from tests.unit.rules.fusion_engine_helpers import assert_fusion_pin


def test_triazolopyrimidine_recovered():
    assert_fusion_pin("c1cnc2ncnn2c1", "[1,2,4]triazolo[1,5-a]pyrimidine")


def test_triazolopyridine_recovered():
    # Suite fix j4 (TRIAGE g3 C07): (the Blue Book) "In
    # preferred IUPAC names, all indicated hydrogen atoms must be cited when
    # the names are constructed in accordance with the principles of fusion
    # nomenclature". The input is the 3H tautomer (N-H next to C-3a); the bare
    # name parses to the 1H tautomer (same standard InChIKey, different
    # canonical SMILES), so the '3H-' is what makes the name denote the input.
    assert_fusion_pin("c1cnc2[nH]nnc2c1", "3H-[1,2,3]triazolo[4,5-b]pyridine")
