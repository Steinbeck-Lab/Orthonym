"""Wave2 T1c — acyl-nitrogen prefix subsystem (IUPAC P-66.1.1.4.3).

For R-CO-NH- on a parent with a senior characteristic group, method (1)
generates the PIN: the amide name with its final 'e' changed to 'o'
(amide -> amido). Blue Book examples: 4-formamidobenzoic acid (PIN),
(4-acetamido-3-methylphenyl)arsonic acid (PIN), 4-benzamidobenzene-1-
sulfonic acid (PIN). Method (2) '{acyl}amino' does NOT generate PINs.

Covers the shared builders in assembly/substituent_naming.py, both chain
emitters (composer._check_for_acylamino, substituent_enumerator.
_name_amino_branch), the benzene ring-parent recognizer, and the
alpha_sort_key chain-stem guard.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.assembly.naming_utils import alpha_sort_key
from orthonym.assembly.substituent_naming import (
    acid_name_to_amido_prefix,
    acyl_amido_prefix_from_branch,
    acyl_carbons_to_amido_prefix,
    linear_acyl_amido_prefix,
)


@pytest.mark.unit
class TestAmidoBuilders:
    """Unit tests for the shared amido-prefix builders."""

    @pytest.mark.parametrize("carbons,expected", [
        (1, "formamido"),      # formamide is a retained amide PIN
        (2, "acetamido"),      # acetamide is a retained amide PIN
        (3, "propanamido"),
        (4, "butanamido"),
        (5, "pentanamido"),
        (8, "octanamido"),
        (18, "octadecanamido"),
    ])
    def test_acyl_carbons_to_amido(self, carbons, expected):
        assert acyl_carbons_to_amido_prefix(carbons) == expected

    @pytest.mark.parametrize("acid,expected", [
        ("formic acid", "formamido"),
        ("acetic acid", "acetamido"),
        ("benzoic acid", "benzamido"),
        ("pentanoic acid", "pentanamido"),
        ("4-methylbenzoic acid", "4-methylbenzamido"),
        ("naphthalene-1-carboxylic acid", "naphthalene-1-carboxamido"),
        ("cyclohexanecarboxylic acid", "cyclohexanecarboxamido"),
    ])
    def test_acid_name_to_amido(self, acid, expected):
        assert acid_name_to_amido_prefix(acid) == expected

    @pytest.mark.parametrize("acid", [
        "pentanedioic acid",                  # two acid groups
        "cyclohexane-1,2-dicarboxylic acid",  # dicarboxylic
        "pentanethioic acid",                 # functional replacement
        "",                                   # empty
        "not an acid",
    ])
    def test_acid_name_to_amido_fails_closed(self, acid):
        assert acid_name_to_amido_prefix(acid) is None

    def test_linear_builder_rejects_branched_acyl(self):
        """Isobutyryl (branched) must NOT get a linear amido name."""
        mol = Chem.MolFromSmiles("CC(C)C(=O)NCC(=O)O")
        # atoms: 0=C,1=C,2=C,3=C(=O),4=O,5=N ... acyl branch = {0,1,2,3,4}
        sub = [5, 3, 4, 0, 1, 2]
        assert linear_acyl_amido_prefix(mol, 3, 5, sub) is None

    def test_linear_builder_rejects_unsaturated_acyl(self):
        """Acryloyl (unsaturated) must NOT get a saturated amido name."""
        mol = Chem.MolFromSmiles("C=CC(=O)NCC(=O)O")
        sub = [4, 2, 3, 0, 1]  # N, C(=O), O, CH2=, =CH-
        assert linear_acyl_amido_prefix(mol, 2, 4, sub) is None

    def test_linear_builder_rejects_dropped_atoms(self):
        """Coverage guard: extra substituent atoms (N-methyl) -> None."""
        mol = Chem.MolFromSmiles("CN(C(C)=O)CCC(=O)O")
        # sub includes the N-methyl (atom 0): {0,1,2,3,4}; acyl C=2
        assert linear_acyl_amido_prefix(mol, 2, 1, [0, 1, 2, 3, 4]) is None

    def test_branch_builder_ring_acyl(self):
        """Benzoyl branch -> benzamido via the acid-name path."""
        mol = Chem.MolFromSmiles("O=C(NCC(=O)O)c1ccccc1")
        # locate the amide N and its benzoyl carbonyl C by substructure
        match = mol.GetSubstructMatch(
            Chem.MolFromSmarts("[NX3;H1]([CX3](=O)c1ccccc1)")
        )
        n_idx = match[0]
        carbonyl_c = match[1]
        carbonyl_o = match[2]
        phenyl = list(match[3:])
        sub = [n_idx, carbonyl_c, carbonyl_o] + phenyl
        assert acyl_amido_prefix_from_branch(
            mol, n_idx, carbonyl_c, sub
        ) == "benzamido"


@pytest.mark.unit
class TestAmidoEndToEnd:
    """End-to-end: both chain emitters and the ring-parent recognizer."""

    @pytest.mark.parametrize("smiles,expected", [
        # chain parent (composer._check_for_acylamino)
        ("O=CNCCC(=O)O", "3-formamidopropanoic acid"),
        ("CC(=O)NCCC(=O)O", "3-acetamidopropanoic acid"),
        ("CCC(=O)NCCCC(=O)O", "4-propanamidobutanoic acid"),
        ("CCCCC(=O)NC(CCC(=O)O)C(=O)O", "2-pentanamidopentanedioic acid"),
        ("CC(=O)NC(CC(=O)O)C(=O)O", "2-acetamidobutanedioic acid"),
        # ring acyl on a chain parent (fragment path)
        ("OC(=O)CNC(=O)c1ccccc1", "2-benzamidoethanoic acid"),
        ("OC(=O)CNC(=O)c1ccc(C)cc1", "2-(4-methylbenzamido)ethanoic acid"),
        ("OC(=O)CNC(=O)c1cccc2ccccc12",
         "2-(naphthalene-1-carboxamido)ethanoic acid"),
        # benzene ring parent (P-66.1.1.4.3 Blue Book example verbatim)
        ("O=CNc1ccc(C(=O)O)cc1", "4-formamidobenzoic acid"),
        ("CC(=O)Nc1ccc(C(=O)O)cc1", "4-acetamidobenzoic acid"),
        ("CCC(=O)Nc1ccc(C(=O)O)cc1", "4-propanamidobenzoic acid"),
    ])
    def test_amido_pin_emission(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_alpha_order_hydroxy_before_octadecanamido(self):
        """P-14.5.2: hydroxy ('h') cites before octadecanamido ('o')."""
        name = name_compound("CCCCCCCCCCCCCCCCCC(=O)N[C@@H](CO)C(=O)O")
        assert name == "(2S)-3-hydroxy-2-octadecanamidopropanoic acid", (
            f"Got '{name}'"
        )

    @pytest.mark.parametrize("smiles,expected", [
        # ureido is a SEPARATE preferred prefix (P-66.1.1.4.5.1) — untouched
        ("NC(=O)NCCC(=O)O", "3-(carbamoylamino)propanoic acid"),
        ("NC(=O)NCCCC(N)C(=O)O", "2-amino-5-(carbamoylamino)pentanoic acid"),
        # amide as the principal group stays a suffix parent (P-66.1.1.4.3
        # closing note: never a ring substituent when it is principal)
        ("CC(=O)Nc1ccccc1", "N-phenylacetamide"),
    ])
    def test_protected_neighbours_unchanged(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestAlphaSortChainStemGuard:
    """alpha_sort_key must not strip a chain stem as a multiplier."""

    @pytest.mark.parametrize("name,key", [
        ("octadecanamido", "octadecanamido"),   # not 'decanamido'
        ("pentanamido", "pentanamido"),         # not 'namido'
        ("decanamido", "decanamido"),
        ("tridecyl", "tridecyl"),               # C13 alkyl, not 'decyl'
        ("octadecyl", "octadecyl"),
        ("pentacosyl", "pentacosyl"),           # C25
        ("triacontyl", "triacontyl"),           # C30
        ("decanoyloxy", "decanoyloxy"),
        # genuine multipliers still strip
        ("dimethyl", "methyl"),
        ("triethyl", "ethyl"),
        ("tetrachloro", "chloro"),
        # Expectation corrected. `di` + an acylamido prefix is a form the Blue
        # Book REJECTS: BlueBookV2.md:33107 prints "*N*-acetylacetamido
        # (preferred prefix) diacetylamino (not diacetylazanyl) (not
        # diacetamido)", and :55811 indexes "diacetamido: see
        # N-acetylacetamido". An acylamido prefix is compound, so P-16.3.5(a)
        # multiplies it with `bis(...)`, and this engine emits exactly that --
        # `get_multiplier_prefix(2, 'octadecanamido')` is 'bis'. The string
        # `dioctadecanamido` is therefore never produced; when it IS handed to
        # the key, P-14.5.2's complete-name rule applies and it keys at 'd'.
        ("dioctadecanamido", "dioctadecanamido"),
    ])
    def test_sort_key(self, name, key):
        assert alpha_sort_key(name) == key
