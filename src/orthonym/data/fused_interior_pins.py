"""Fused parents the Blue Book prints with an interior-atom locant.

 Numbering of interior carbon atoms (the Blue Book), (:12658): "Interior
atoms are numbered by identifying the minimum number of bonds linking them to a peripheral atom. The
locant for the interior atom is that of the peripheral atom with a superscript number corresponding to
the number of bonds between the two atoms." The interior atom of phenalene is '3a^1', not '9b'.

 LIMITATIONS OF FUSION NOMENCLATURE: THREE COMPONENTS ORTHO- AND PERI-FUSED TOGETHER (:14456),
 (:14470): "If the corresponding hydrocarbon fused ring system can be named by fusion
principles or has a retained name, then heteroatoms are identified by skeletal replacement ('a')
nomenclature using the appropriate 'a' prefixes (see. The numbering of the fused hydrocarbon
system is not altered by the 'a' prefixes." Its example: '1,3a^1,4,9-tetraazaphenalene (PIN)' (:14491).

The names are in their own table: OPSIN 2.9.0 has no grammar for the superscript locant, so a name from
this table is checked through its twin spelling with the old locant (``validation.twin_spellings``,
rule ``interior_phenalene``), and a shared entry of ``FUSED_HETEROCYCLE_DATA`` would reach every reader
of that table at every tier. Only the default tier reads this table (``rules.fused_interior``).

Each entry is the parent hydride as a SMILES whose atoms are written in the order of ``locants``: atom i
carries locant i of the tuple (peripheral locants as in 1H-phenalene, the interior atom last).
"""
from typing import Dict, Tuple

#: the locants of the 13 atoms of phenalene, in the order the SMILES below write them
_PHENALENE_LOCANTS: Tuple = (1, 2, 3, "3a", 4, 5, 6, "6a", 7, 8, 9, "9a", "3a^1")

#: skeleton SMILES (atom order = numbering) -> the printed parent
INTERIOR_PIN_PARENTS: Dict[str, Dict] = {
    # (:14470), example:14491; under Seniority of heteroatoms over
    # suffixes (:25297,:25319): the analysis (:25337-:25339) gives the heterocyclic parent hydride and
    # the result is '1,3a^1,4,9-tetraazaphenalene-3-carboxylic acid (PIN)' (:25350)
    "N1=CC=C2N=CC=C3C=CN=C1N23": {
        "name": "1,3a^1,4,9-tetraazaphenalene",
        "locants": _PHENALENE_LOCANTS,
        "bluebook": "P-25.5.1.1 :14470, :14491",
    },
    # (:23474): "Skeletal replacement ('a') nomenclature is used to derive preferred IUPAC
    # names for heteropolycyclic ring systems for which fusion nomenclature based on heterocyclic parent
    # rings is not applicable (see ", example '1,3a^1,4-triazaphenalene (PIN) [not
    # 1,4,9b-triazaphenalene; see ]' (:23486; the text of the Blue Book reads 'phenylene'
    # there, the page of the recommendations reads 'phenalene')
    "N1=CC=C2N=CC=C3C=CC=C1N23": {
        "name": "1,3a^1,4-triazaphenalene",
        "locants": _PHENALENE_LOCANTS,
        "bluebook": "P-51.4.2.4 :23474, :23486",
    },
}
