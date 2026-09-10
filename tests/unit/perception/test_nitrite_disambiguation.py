"""-05  — nitroso (C-N=O) vs nitrite ester (R-O-N=O) disambiguation.

Wave-0 scaffold (a phase). The TARGET assertions are xfail until the -05
code plan (175-05) lands the `[#6]` C-attachment guard on the `nitroso` SMARTS
+ the collision-suppression rule + the nitrite-ester emitter. The C-nitroso
CONTROL is a non-xfail regression guard (must stay nitroso, never a nitrite).

Blue Book: R-O-N=O is an ester of nitrous acid ("<alkyl> nitrite"); the
`nitroso` prefix applies only to C-N=O.
"""

import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestNitriteDisambiguation:
    def test_ethyl_nitrite(self):
        # R-O-N=O is a nitrous-acid ester, not a C-nitroso compound.
        assert name_compound("CCON=O").strip().lower() == "ethyl nitrite"

    def test_methyl_nitrite(self):
        assert name_compound("CON=O").strip().lower() == "methyl nitrite"

    def test_c_nitroso_control_stays_nitroso(self):
        # CONTROL (regression guard, NOT xfail): CN=O is genuine C-nitroso.
        # The [#6] guard must not perturb a truly C-attached N=O.
        name = name_compound("CN=O").strip().lower()
        assert "nitrite" not in name, f"C-nitroso must not become a nitrite: {name!r}"
        assert name == "nitrosomethane", f"C-nitroso regression: {name!r}"
