""" fusion-descriptor hyphen conformance (Wave-2 P0c Task 10).
BB the Blue Book: 'Hyphens separate the two parts of a fusion
descriptor, i.e., numbers and italicized letters.'
"""
import re

from orthonym.rules.fusion_descriptors import generate_fusion_descriptor
from orthonym.namer import name_compound

_DESCRIPTOR_SHAPE = re.compile(r"^\[\d+(?:,\d+)*-[a-z]+\]$")


class TestP16243HyphenPlacement:
    def test_descriptor_shape_numbers_hyphen_letter(self):
        desc = generate_fusion_descriptor(
            [0, 1, 2, 3, 4, 5], [0, 1, 6, 7, 8], {0, 1})
        assert desc == "[1,2-a]"
        assert _DESCRIPTOR_SHAPE.match(desc)

    def test_benzene_child_letter_only_no_hyphen(self):
        # Benzene child omits locants -> bare letter, no dangling hyphen.
        desc = generate_fusion_descriptor(
            [0, 1, 2, 3, 4, 5], [0, 1, 6, 7, 8], {0, 1}, child_is_benzene=True)
        assert desc == "[a]"

    def test_end_to_end_furofuran(self):
        result = name_compound("C1=CC2=C(O1)C=CO2")
        name = result if isinstance(result, str) else getattr(result, "name", str(result))
        assert name == "furo[3,2-b]furan"
