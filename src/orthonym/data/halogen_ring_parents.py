"""Benzoheterocycle parents with a ring halogen in the lambda state that the Blue Book prints as a PIN.

``rules/halogen_ring_fusion.py`` builds the benzo name of any benzene ring fused to a five- to
seven-membered heteromonocycle with a ring iodine, the Blue Book). A name is
labelled a preferred IUPAC name only on a parent listed here: the book prints a (PIN) example on
that ring system, with its lambda token, its hydrogen and its locants, so the construction of the
name is attested and not only derived. Every other parent (the six-membered 1,2-benziodoxine, the
isomeric 2,1-benziodoxole, a benziodazole, the lambda5 states) is a valid systematic name that the
module records as not the PIN, so the default tier declines it and the best-effort tier ships it.

Key: the bare mancude parent exactly as ``halogen_ring_fusion.benzo_hw_name`` spells it.
Value: (rule heading, the Blue Book line, the example as the book prints it).
OCR of the benziodole line was cross-checked against the QMUL copy of
(https://iupac.qmul.ac.uk/BlueBook/P6a.html), which prints '1H-1λ3-benziodole (PIN)' and
'1-methoxy-1λ3,2-benziodoxol-3(1H)-one (PIN)'.
"""
from typing import Dict, Tuple

PRINTED_PIN_PARENTS: Dict[str, Tuple[str, int, str]] = {
    "1λ3,2-benziodoxole": (
        "P-68.5.1 'Nomenclature based on halogen parent hydrides'", 39674,
        "1-methoxy-1λ3,2-benziodoxol-3(1H)-one (PIN)"),
    "1λ3-benziodole": (
        "P-68.5.1 'Nomenclature based on halogen parent hydrides'", 39692,
        "1H-1λ3-benziodole (PIN)"),
    "3λ3,2,4-benziodadioxepine": (
        "P-25.6 'Fused ring systems with skeletal atoms with nonstandard bonding numbers'", 14557,
        "3H-3λ3,2,4-benziodadioxepine (PIN)"),
}
