"""Whether the name of an amine fragment cites its nitrogen as the suffix 'amine' or as the prefix 'amino'.

The acyl-prefix branch of ``fragment_assembly.assemble_amide`` writes ``N-<acyl>-<amine name>``: the locant N
names the nitrogen of the amine, which is the suffix of the name ('N-acetyl-N-methylcyclohexanamine'). When a ring
heterone stands in the amine fragment, the heterone is the suffix, the Blue Book; the ketone class is
senior to the amines,,:18158) and the amine is cited as the prefix 'amino': 'N-(acetylcarbamoyl)-3-amino-1λ6-
thiolane-1,1-dione' cites an N that belongs to no parent. The caller leaves such an amine to the producers that cite
the acylated amine as a prefix of the heterone's name ('3-[(acetylcarbamoyl)amino]-1λ6-thiolane-1,1-dione').

Only the ring heterone is refused. The same float on the amine of an acid, an ester, a sulfonate or a ring ketone is the
demotion of decision A (``engine.gate_nonsuffix_nacyl_float``), and the names the producers after it give those
molecules are no better: with the float refused for every 'amino' amine, 12 tests that pass on main name a
2-aminoethanesulfonate amide '1-oxo-1-[(2-sulfonatoethyl)amino]icosa-...' for '2-(...amido)ethane-1-sulfonate', or
shorten a name to 'heptanamide'.
"""

#: the endings of a name whose suffix is the amine ('cyclohexanamine', 'ethane-1,2-diamine', 'aniline')
_AMINE_ENDINGS = ("amine", "aniline")


def amine_is_on_ring_heterone(amine_name: str, amine_smiles: str) -> bool:
    """True when the amine fragment holds a ring heterone and its name cites an 'amino' prefix, no amine suffix."""
    if not amine_name or not amine_smiles or "amino" not in amine_name or amine_name.endswith(_AMINE_ENDINGS):
        return False
    from rdkit import Chem
    from ..rules.ring_chalcogen_oxide import has_ring_heterone
    return has_ring_heterone(Chem.MolFromSmiles(amine_smiles))
