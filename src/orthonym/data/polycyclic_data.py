"""
Polycyclic Aromatic Hydrocarbon (PAH) data for IUPAC naming.

Contains lookup tables for common polycyclic aromatic hydrocarbons including:
- Canonical SMILES for exact matching
- SMARTS patterns for substructure matching
- Number of atoms in the PAH core
- IUPAC standard numbering (atom index to IUPAC locant mapping)
- Allowed substituent positions

IUPAC Naming Rules for PAHs:
- PAH numbering is FIXED by IUPAC, not reoriented based on substituents
- Use retained names (naphthalene, anthracene, etc.) as parent
- Substituents are named with their IUPAC locant position

Reference: IUPAC Blue Book 2013, Section P-25 (Fused and Bridged Fused Ring Systems)

PAH Classification:
- Bicyclic: naphthalene
- Tricyclic: anthracene, phenanthrene, fluorene, acenaphthene, acenaphthylene
- Tetracyclic: pyrene, chrysene, tetracene, triphenylene, benz[a]anthracene, benzo[c]phenanthrene
- Pentacyclic: pentacene, perylene, benzo[a]pyrene
- Hexacyclic+: coronene

Partially saturated PAHs:
- 9,10-dihydroanthracene
- 1,2-dihydronaphthalene
"""

from typing import Any, Dict, List, Optional, Set, Tuple

# Polycyclic aromatic hydrocarbon data
# Key: retained name
# Value: dict with canonical_smiles, smarts, num_atoms, iupac_numbering, substituent_positions
POLYCYCLIC_DATA: Dict[str, Dict[str, Any]] = {
    # Wave2 T4 (P-25.1.2): higher fused-hydrocarbon series members (acene/aphene/
    # helicene/pleiadene) that were fail-closed 'unknown' at HEAD. Bare-name
    # catalog entries; OPSIN-RT + numbering verified (extendedsmi mapped onto
    # RDKit-canonical, confirmed via methyl-isomer attachment).
    'pentaphene': {
        'canonical_smiles': 'c1ccc2cc3c(ccc4cc5ccccc5cc43)cc2c1',
        'smarts': 'c1ccc2cc3c(ccc4cc5ccccc5cc43)cc2c1',
        'num_atoms': 22,
        'iupac_numbering': {0: 3, 1: 2, 2: 1, 3: '14a', 4: 14, 5: '13b', 6: '5a', 7: 6, 8: 7, 9: '7a', 10: 8, 11: '8a', 12: 9, 13: 10, 14: 11, 15: 12, 16: '12a', 17: 13, 18: '13a', 19: 5, 20: '4a', 21: 4},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'num_rings': 5,
    },
    'hexaphene': {
        'canonical_smiles': 'c1ccc2cc3cc4c(ccc5cc6ccccc6cc54)cc3cc2c1',
        'smarts': 'c1ccc2cc3cc4c(ccc5cc6ccccc6cc54)cc3cc2c1',
        'num_atoms': 26,
        'iupac_numbering': {0: 11, 1: 12, 2: 13, 3: '13a', 4: 14, 5: '14a', 6: 15, 7: '15a', 8: '7a', 9: 7, 10: 6, 11: '5a', 12: 5, 13: '4a', 14: 4, 15: 3, 16: 2, 17: 1, 18: '16a', 19: 16, 20: '15b', 21: 8, 22: '8a', 23: 9, 24: '9a', 25: 10},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'num_rings': 6,
    },
    'hexacene': {
        'canonical_smiles': 'c1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1',
        'smarts': 'c1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1',
        'num_atoms': 26,
        'iupac_numbering': {0: 3, 1: 2, 2: 1, 3: '16a', 4: 16, 5: '15a', 6: 15, 7: '14a', 8: 14, 9: '13a', 10: 13, 11: '12a', 12: 12, 13: 11, 14: 10, 15: 9, 16: '8a', 17: 8, 18: '7a', 19: 7, 20: '6a', 21: 6, 22: '5a', 23: 5, 24: '4a', 25: 4},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'num_rings': 6,
    },
    'heptacene': {
        # Linear 7-fused-ring acene (P-25.1.2, next member of the acene series
        # after hexacene). AUTHORITATIVE numbering derived 2026-08-18 from
        # OPSIN 2.9.0 `heptacene -o extendedsmi` ($_AV: locants) —
        # C1=CC=CC2=CC3=CC4=CC5=CC6=CC7=CC=CC=C7C=C6C=C5C=C4C=C3C=C12
        # |$_AV:1;2;3;4;4a;5;5a;6;6a;7;7a;8;8a;9;9a;10;11;12;13;13a;14;14a;15;
        # 15a;16;16a;17;17a;18;18a$| — mapped onto this RDKit-canonical
        # SMILES via GetSubstructMatch (same molecule confirmed by identical
        # InChI). Per-position round-trip VERIFIED: all 18 peripheral integer
        # locants (1-18) reconstructed as mono-methylheptacene, named via
        # Orthonym, and confirmed both to carry the expected locant in the
        # emitted name AND to round-trip through OPSIN to the identical
        # InChIKey as the substituted input (see
        # tests/unit/rules/test_heptacene.py).
        'canonical_smiles': 'c1ccc2cc3cc4cc5cc6cc7ccccc7cc6cc5cc4cc3cc2c1',
        'smarts': 'c1ccc2cc3cc4cc5cc6cc7ccccc7cc6cc5cc4cc3cc2c1',
        'num_atoms': 30,
        'iupac_numbering': {0: 3, 1: 2, 2: 1, 3: '18a', 4: 18, 5: '17a', 6: 17, 7: '16a', 8: 16, 9: '15a', 10: 15, 11: '14a', 12: 14, 13: '13a', 14: 13, 15: 12, 16: 11, 17: 10, 18: '9a', 19: 9, 20: '8a', 21: 8, 22: '7a', 23: 7, 24: '6a', 25: 6, 26: '5a', 27: 5, 28: '4a', 29: 4},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        'num_rings': 7,
    },
    # (pentahelicene DEFERRED — [5]helicene's PIN is the fusion name
    # dibenzo[c,g]phenanthrene, NOT 'pentahelicene': BB P-25.1.2.6 starts the
    # helicene series at SIX rings. The fusion name needs the polycomponent
    # (3+-component) fusion engine; a pre-existing tripwire (DD7-fusion-1)
    # already defers this SMILES. Stays fail-closed.)
    'hexahelicene': {
        'canonical_smiles': 'c1ccc2c(c1)ccc1ccc3ccc4ccc5ccccc5c4c3c12',
        'smarts': 'c1ccc2c(c1)ccc1ccc3ccc4ccc5ccccc5c4c3c12',
        'num_atoms': 26,
        'iupac_numbering': {0: 3, 1: 2, 2: 1, 3: '16e', 4: '4a', 5: 4, 6: 5, 7: 6, 8: '6a', 9: 7, 10: 8, 11: '8a', 12: 9, 13: 10, 14: '10a', 15: 11, 16: 12, 17: '12a', 18: 13, 19: 14, 20: 15, 21: 16, 22: '16a', 23: '16b', 24: '16c', 25: '16d'},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'num_rings': 6,
    },
    # (pleiadene DEFERRED — non-benzenoid o-quinoid PAH with 2 non-aromatic ring
    # carbons; identify_polycyclic matches it but the aromatic-PAH dispatcher
    # does not route there, so it stays fail-closed 'unknown'. Needs o-quinoid
    # routing — out of this batch's scope.)
    'naphthalene': {
        'canonical_smiles': 'c1ccc2ccccc2c1',
        'smarts': 'c1ccc2ccccc2c1',  # For substructure matching
        'num_atoms': 10,
        # IUPAC numbering for naphthalene:
        #     8  1
        #    /  \ /
        #   7    2
        #   |    |
        #   6    3
        #    \  / \
        #     5  4
        # Maps canonical atom index -> IUPAC position (1-indexed)
        # Note: This mapping is determined empirically based on RDKit's canonical ordering
        # Empirically derived from RDKit canonical SMILES 'c1ccc2ccccc2c1':
        # Rings: [0,9,8,3,2,1] and [4,5,6,7,8,3]. Fusion: idx 3, idx 8.
        # Peripheral path: 9->0->1->2->[3]->4->5->6->7->[8]
        # IUPAC: 1->2->3->4->[4a]->5->6->7->8->[8a]
        'iupac_numbering': {
            9: 1, 0: 2, 1: 3, 2: 4, 3: '4a', 4: 5, 5: 6, 6: 7, 7: 8, 8: '8a'
        },
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],  # Allowed positions
        'num_rings': 2,
    },
    'azulene': {
        # Retained fused-ring hydrocarbon (IUPAC 2013 P-25.1.1, Table 28.1):
        # a 5-membered ring ortho-fused to a 7-membered ring, fully mancude
        # (aromatic). Carbocyclic, so it is NOT caught by name_fused_heterocycle,
        # and being aromatic it is NOT caught by _name_saturated_fused_carbocyclic
        # either -- without this entry identify_polycyclic() returns None and the
        # molecule falls through to the acyclic chain catch-all, which emits the
        # malformed empty stem 'ane' (V-1 / theme T10).
        'canonical_smiles': 'c1ccc2cccc-2cc1',
        'smarts': 'c1ccc2cccc-2cc1',  # fusion bond is formally single (c-c)
        'num_atoms': 10,
        # IUPAC numbering for azulene: 1,2,3 on the 5-membered ring; 4,5,6,7,8 on
        # the 7-membered ring; 3a / 8a the two fusion atoms. Maps canonical atom
        # index (of 'c1ccc2cccc-2cc1') -> IUPAC position. Derived from the
        # canonical topology: fusion atoms idx 3 (=3a) and idx 7 (=8a); the
        # 5-ring non-fusion arc 3-[4-5-6]-7 carries C3,C2,C1; the 7-ring
        # non-fusion arc 7-[8-9-0-1-2]-3 carries C8,C7,C6,C5,C4. (Azulene has a
        # mirror plane through C2/C6, so this orientation's locant set is the
        # unique lowest set regardless of the C3a/C8a labelling direction.)
        # NOTE: this dict is consumed by get_polycyclic_iupac_locants (correct
        # for bare azulene). The *substituent*-naming path
        # (rules/polycyclics._map_pah_atoms_to_iupac) is hardcoded to
        # naphthalene's symmetric 6-6 alpha/beta pattern and has no azulene
        # branch, so SUBSTITUTED azulene locants are unproven/likely wrong here
        # -- that is fused-ring numbering, owned by Phase E1/DD4, out of C-T10
        # scope. Bare azulene (the V-1 gold) is correct.
        'iupac_numbering': {
            6: 1, 5: 2, 4: 3, 3: '3a', 2: 4, 1: 5, 0: 6, 9: 7, 8: 8, 7: '8a'
        },
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],
        'num_rings': 2,
    },
    'anthracene': {
        'canonical_smiles': 'c1ccc2cc3ccccc3cc2c1',
        'smarts': 'c1ccc2cc3ccccc3cc2c1',
        'num_atoms': 14,
        # IUPAC numbering for anthracene (linear tricyclic). AUTHORITATIVE:
        # re-derived 2026-06-22 from OPSIN `anthracene -o extendedsmi`
        # ($_AV: locants) mapped onto this canonical SMILES. The prior numbering
        # was INVALID — it placed a *meso* carbon (central-ring atoms 4 & 11) at
        # locant 5 instead of the correct 9/10, so 9-substituted/9,10-dihydro
        # anthracenes were mis-numbered (e.g. 9-methyl -> wrong "5-methyl").
        # Meso (central-ring CH) atoms 4 -> 9, 11 -> 10.
        'iupac_numbering': {
            2: 1, 1: 2, 0: 3, 13: 4, 9: 5, 8: 6, 7: 7, 6: 8, 4: 9, 11: 10,
            3: '9a', 5: '8a', 10: '10a', 12: '4a',
        },
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num_rings': 3,
    },
    'phenanthrene': {
        'canonical_smiles': 'c1ccc2c(c1)ccc1ccccc12',
        'smarts': 'c1ccc2c(c1)ccc1ccccc12',
        'num_atoms': 14,
        # IUPAC numbering for phenanthrene (angular tricyclic)
        # Empirically derived from RDKit canonical SMILES 'c1ccc2c(c1)ccc1ccccc12':
        # Rings: [0,5,4,3,2,1], [6,7,8,13,3,4], [9,10,11,12,13,8]
        # Fusion: idx 3(=4a), 4(=10a), 8(=8a), 13(=4b)
        # Ring A(R0): 5->0->1->2 peripheral, Ring C(R1): 6->7 peripheral,
        # Ring B(R2): 12->11->10->9 peripheral
        # IUPAC: 1->2->3->4->[4a]->[10a]->10->9->[8a]->[4b]->5->6->7->8
        'iupac_numbering': {
            5: 1, 0: 2, 1: 3, 2: 4, 3: '4a', 4: '10a',
            6: 10, 7: 9, 8: '8a', 13: '4b',
            12: 5, 11: 6, 10: 7, 9: 8
        },
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num_rings': 3,
    },
    '1H-cyclopenta[a]naphthalene': {
        # Ortho-fused cyclopenta + naphthalene (3-ring cata-fused carbocycle,
        # P-25.3.1.3). The mancude parent CARRIES an intrinsic indicated hydrogen
        # (the >CH2 at position 1), so the parent-name KEY itself is
        # '1H-cyclopenta[a]naphthalene' (as '1H-indene' is stored in
        # FUSED_HETEROCYCLE_DATA) — the '1H-' is a detachable added-IH cited
        # before the stem (P-31.1.4). Emitting the bare 'cyclopenta[a]naphthalene'
        # would drop the mandatory indicated hydrogen (a non-conformant PIN).
        # AUTHORITATIVE numbering: OPSIN `1H-cyclopenta[a]naphthalene
        # -o extendedsmi` ($_AV) = C1C=CC=2C1=C1C=CC=CC1=CC2
        # |$1;2;3;3a;9b;9a;9;8;7;6;5a;5;4$|, mapped onto this canonical SMILES
        # (RDKit parses in written order so index i -> the i-th $_AV locant).
        # 5-ring: 1,2,3,3a,9b ; 6-rings share the 3a/5a/9a/9b fusion carbons.
        'canonical_smiles': 'C1C=CC=2C1=C1C=CC=CC1=CC2',
        'smarts': 'C1C=CC=2C1=C1C=CC=CC1=CC2',
        'num_atoms': 13,
        'iupac_numbering': {
            0: 1, 1: 2, 2: 3, 3: '3a', 4: '9b', 5: '9a',
            6: 9, 7: 8, 8: 7, 9: 6, 10: '5a', 11: 5, 12: 4,
        },
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'num_rings': 3,
    },
    'pyrene': {
        'canonical_smiles': 'c1cc2ccc3cccc4ccc(c1)c2c34',
        'smarts': 'c1cc2ccc3cccc4ccc(c1)c2c34',
        'num_atoms': 16,
        # IUPAC numbering for pyrene (peri-condensed tetracyclic). AUTHORITATIVE:
        # re-derived 2026-06-22 from OPSIN `pyrene -o extendedsmi`
        # ($_AV: locants) mapped onto this canonical SMILES. The prior numbering
        # was INVALID (interior/fusion locants '3b'/'10a' instead of the correct
        # peri carbons 10a/10b/10c), so substituted pyrenes were mis-numbered.
        # Interior carbons: 14 -> 10b, 15 -> 10c.
        'iupac_numbering': {
            1: 1, 0: 2, 13: 3, 11: 4, 10: 5, 8: 6, 7: 7, 6: 8, 4: 9, 3: 10,
            12: '3a', 9: '5a', 5: '8a', 2: '10a', 14: '10b', 15: '10c',
        },
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num_rings': 4,
    },
    'fluorene': {
        'indicated_h': '9H',  # P-31.1.4.3.4 (Wave-2 completion): leads substituted names
        'canonical_smiles': 'c1ccc2c(c1)Cc1ccccc1-2',  # Has sp3 carbon (position 9)
        'smarts': 'c1ccc2c(c1)Cc1ccccc1-2',
        'num_atoms': 13,
        # IUPAC numbering for fluorene (with methylene bridge at position 9)
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9],  # 9 is the sp3 carbon
        'num_rings': 3,
    },
    'acenaphthene': {
        'canonical_smiles': 'c1cc2c3c(cccc3c1)CC2',  # Has two sp3 carbons
        'smarts': 'c1cc2c3c(cccc3c1)CC2',
        'num_atoms': 12,
        # IUPAC numbering for acenaphthene
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],  # Excluding the ethylene bridge
        'num_rings': 3,
    },
    'acenaphthylene': {
        'canonical_smiles': 'C1=Cc2cccc3cccc1c23',  # Fully unsaturated
        'smarts': 'C1=Cc2cccc3cccc1c23',
        'num_atoms': 12,
        # IUPAC numbering for acenaphthylene
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],
        'num_rings': 3,
    },
    'chrysene': {
        'canonical_smiles': 'c1ccc2c(c1)ccc1c3ccccc3ccc21',
        'smarts': 'c1ccc2c(c1)ccc1c3ccccc3ccc21',
        'num_atoms': 18,
        # IUPAC numbering for chrysene
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    # ==========================================================================
    # Tetracyclic (4 rings) - additional
    # ==========================================================================
    'tetracene': {
        # Also known as naphthacene - linear 4-ring PAH
        'canonical_smiles': 'c1ccc2cc3cc4ccccc4cc3cc2c1',
        'smarts': 'c1ccc2cc3cc4ccccc4cc3cc2c1',
        'num_atoms': 18,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    'triphenylene': {
        # Angular 4-ring PAH (three benzene rings sharing a central ring)
        'canonical_smiles': 'c1ccc2c(c1)c1ccccc1c1ccccc21',
        'smarts': 'c1ccc2c(c1)c1ccccc1c1ccccc21',
        'num_atoms': 18,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    'benz[a]anthracene': {
        # Bent 4-ring PAH (benzene fused to anthracene)
        'canonical_smiles': 'c1ccc2cc3c(ccc4ccccc43)cc2c1',
        'smarts': 'c1ccc2cc3c(ccc4ccccc43)cc2c1',
        'num_atoms': 18,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    'benzo[c]phenanthrene': {
        # Angular 4-ring PAH
        'canonical_smiles': 'c1ccc2c(c1)ccc1ccc3ccccc3c12',
        'smarts': 'c1ccc2c(c1)ccc1ccc3ccccc3c12',
        'num_atoms': 18,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    # ==========================================================================
    # Pentacyclic (5 rings)
    # ==========================================================================
    'pentacene': {
        # Linear 5-ring PAH
        'canonical_smiles': 'c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1',
        'smarts': 'c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1',
        'num_atoms': 22,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'num_rings': 5,
    },
    'picene': {
        # Angular 5-ring cata-fused PAH (fancene). v23 13B(a) S1: numbering is
        # supplied by the deterministic fusion-numbering engine (iupac_numbering
        # left empty); recognition added here so identify_polycyclic resolves it
        # (it contains a chrysene substructure, so it must be checked before
        # chrysene — the largest-first scan handles that).
        'canonical_smiles': 'c1ccc2c(c1)ccc1c2ccc2c3ccccc3ccc21',
        'smarts': 'c1ccc2c(c1)ccc1c2ccc2c3ccccc3ccc21',
        'num_atoms': 22,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'num_rings': 5,
    },
    'perylene': {
        # Peri-condensed PAH (two naphthalene units joined peri)
        'canonical_smiles': 'c1ccc2cccc3cc4c(c1)cc1cccc4c1c23',
        'smarts': 'c1ccc2cccc3cc4c(c1)cc1cccc4c1c23',
        'num_atoms': 20,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 6,  # RDKit counts 6 rings due to perception
    },
    'benzo[a]pyrene': {
        # Important carcinogen - benzene fused to pyrene
        'canonical_smiles': 'c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34',
        'smarts': 'c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34',
        'num_atoms': 20,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 5,
    },
    # ==========================================================================
    # Hexacyclic+ (6+ rings)
    # ==========================================================================
    'coronene': {
        # 7 rings, hexagonal symmetry
        'canonical_smiles': 'c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61',
        'smarts': 'c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61',
        'num_atoms': 24,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 7,
    },
    'benzo[ghi]perylene': {
        # P-52.2.4.2 large peri-fused PAH (6 rings, 22 atoms). AUTHORITATIVE
        # IUPAC numbering derived 2026-07-09 from OPSIN
        # `benzo[ghi]perylene -o extendedsmi` ($_AV: locants), mapped onto this
        # canonical SMILES via GetSubstructMatch.
        'canonical_smiles': 'c1cc2ccc3ccc4ccc5cccc6c(c1)c2c3c4c56',
        'smarts': 'c1cc2ccc3ccc4ccc5cccc6c(c1)c2c3c4c56',
        'num_atoms': 22,
        'iupac_numbering': {0: 9, 1: 10, 2: '10a', 3: 11, 4: 12, 5: '12a', 6: 1, 7: 2, 8: '2a', 9: 3, 10: 4, 11: '4a', 12: 5, 13: 6, 14: 7, 15: '7a', 16: '7b', 17: 8, 18: '12d', 19: '12b', 20: '12c', 21: '12e'},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 6,
    },
    'cyclobuta[1,7]indeno[5,6-b]naphthalene': {
        # P-25.5.2 three-component ortho/peri-fused PAH (18 atoms, 5 rings incl.
        # a cyclobuta ring). Less-senior-hydrocarbon-parent fusion name that the
        # bridged constructor delegates here (p5_bridged deferred it). Numbering
        # derived 2026-07-09 from OPSIN
        # `cyclobuta[1,7]indeno[5,6-b]naphthalene -o extendedsmi`.
        'canonical_smiles': 'C1=Cc2cc3cc4ccccc4cc3c3c2C1=C3',
        'smarts': 'C1=Cc2cc3cc4ccccc4cc3c3c2C1=C3',
        'num_atoms': 18,
        'iupac_numbering': {0: 2, 1: 3, 2: '3a', 3: 4, 4: '4a', 5: 5, 6: '5a', 7: 6, 8: 7, 9: 8, 10: 9, 11: '9a', 12: 10, 13: '10a', 14: '10b', 15: '10c', 16: '1a', 17: 1},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num_rings': 5,
    },
    # ==========================================================================
    # Partially saturated PAHs
    # ==========================================================================
    '9,10-dihydroanthracene': {
        # Anthracene with positions 9 and 10 saturated (sp3)
        'canonical_smiles': 'c1ccc2c(c1)Cc1ccccc1C2',
        'smarts': 'c1ccc2c(c1)Cc1ccccc1C2',
        'num_atoms': 14,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num_rings': 3,
    },
    '1,2-dihydronaphthalene': {
        # Naphthalene with positions 1 and 2 saturated
        'canonical_smiles': 'C1=Cc2ccccc2CC1',
        'smarts': 'C1=Cc2ccccc2CC1',
        'num_atoms': 10,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],
        'num_rings': 2,
    },
    # ── Wave-2 completion C (P-25.3.4.2.1 BB-cited PAH parents;
    # locants OPSIN-extendedsmi-derived). ──
    'cyclopenta[ij]pentaleno[2,1,6-cde]azulene': {
        'canonical_smiles': 'c1cc2ccc3cc4cc5ccc1c5c4c23',
        'smarts': 'c1cc2ccc3cc4cc5ccc1c5c4c23',
        'num_atoms': 16,
        'iupac_numbering': {0: 5, 1: 4, 2: '3a', 3: 3, 4: 2, 5: '1a', 6: 1, 7: '8a', 8: 8, 9: '7a', 10: 7, 11: 6, 12: '5a', 13: '8c', 14: '8b', 15: '8d'},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],
        'num_rings': 5,
    },
    'dibenzo[c,g]phenanthrene': {
        'canonical_smiles': 'c1ccc2c(c1)ccc1ccc3ccc4ccccc4c3c12',
        'smarts': 'c1ccc2c(c1)ccc1ccc3ccc4ccccc4c3c12',
        'num_atoms': 22,
        'iupac_numbering': {0: 13, 1: 12, 2: 11, 3: '10d', 4: '14a', 5: 14, 6: 1, 7: 2, 8: '2a', 9: 3, 10: 4, 11: '4a', 12: 5, 13: 6, 14: '6a', 15: 7, 16: 8, 17: 9, 18: 10, 19: '10a', 20: '10b', 21: '10c'},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'num_rings': 5,
    },
    "dinaphtho[1,2-c:2',1'-m]picene": {
        'canonical_smiles': 'c1ccc2c(c1)ccc1c2ccc2c1ccc1c2ccc2c3ccc4c5ccccc5ccc4c3ccc21',
        'smarts': 'c1ccc2c(c1)ccc1c2ccc2c1ccc1c2ccc2c3ccc4c5ccccc5ccc4c3ccc21',
        'num_atoms': 38,
        'iupac_numbering': {0: 3, 1: 2, 2: 1, 3: '22b', 4: '4a', 5: 4, 6: 5, 7: 6, 8: '6a', 9: '22a', 10: 22, 11: 21, 12: '20b', 13: '6b', 14: 7, 15: 8, 16: '8a', 17: '20a', 18: 20, 19: 19, 20: '18b', 21: '18a', 22: 18, 23: 17, 24: '16b', 25: '16a', 26: 16, 27: 15, 28: 14, 29: 13, 30: '12a', 31: 12, 32: 11, 33: '10b', 34: '10a', 35: 10, 36: 9, 37: '8b'},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        'num_rings': 9,
    },
    # W2F-P10 (P-25.5.3 core / P-25.3.4.1.2 identical-attached-components).
    # Curated named-fusion-parent catalog entries (same mechanism as
    # dinaphtho[1,2-c:2',1'-m]picene above, added fad413e7). The general P-25.3
    # polycomponent orientation/renumbering engine is not built (multi-week
    # subsystem); these fused/bridged PARENT ring systems are cataloged with
    # OPSIN-authoritative numbering (extendedsmi _AV labels mapped onto the
    # RDKit-canonical key via canonical-rank bijection; method reproduces the
    # hand-verified pentaphene entry identically; every mono-substituent
    # position round-trips through Orthonym's namer). Bare compounds match by
    # exact canonical SMILES (numbering not consulted); substituted derivatives
    # use the numbering with automorphism-min lowest-locant selection.
    # naphtho[2,3-a]pentaphene: single first-order attached component (P-25.3.1).
    "naphtho[2,3-a]pentaphene": {
        'canonical_smiles': 'c1ccc2cc3c(ccc4cc5ccc6cc7ccccc7cc6c5cc43)cc2c1',
        'smarts': 'c1ccc2cc3c(ccc4cc5ccc6cc7ccccc7cc6c5cc43)cc2c1',
        'num_atoms': 30,
        'iupac_numbering': {0: 3, 1: 2, 2: 1, 3: '18a', 4: 18, 5: '17b', 6: '5a', 7: 6, 8: 7, 9: '7a', 10: 8, 11: '8a', 12: 9, 13: 10, 14: '10a', 15: 11, 16: '11a', 17: 12, 18: 13, 19: 14, 20: 15, 21: '15a', 22: 16, 23: '16a', 24: '16b', 25: 17, 26: '17a', 27: 5, 28: '4a', 29: 4},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        'num_rings': 7,
    },
    # dinaphtho[2,3-a:2',3'-o]pentaphene: two IDENTICAL first-order attached
    # naphtho components on pentaphene (P-25.3.4.1.2); the bridge-free CORE of
    # the P-25.5.3 target (BB:14527).
    "dinaphtho[2,3-a:2',3'-o]pentaphene": {
        'canonical_smiles': 'c1ccc2cc3c(ccc4cc5ccc6cc7ccc8cc9ccccc9cc8c7cc6c5cc43)cc2c1',
        'smarts': 'c1ccc2cc3c(ccc4cc5ccc6cc7ccc8cc9ccccc9cc8c7cc6c5cc43)cc2c1',
        'num_atoms': 38,
        'iupac_numbering': {0: 20, 1: 19, 2: 18, 3: '17a', 4: 17, 5: '16b', 6: '22a', 7: 1, 8: 2, 9: '2a', 10: 3, 11: '3a', 12: 4, 13: 5, 14: '5a', 15: 6, 16: '6a', 17: 7, 18: 8, 19: '8a', 20: 9, 21: '9a', 22: 10, 23: 11, 24: 12, 25: 13, 26: '13a', 27: 14, 28: '14a', 29: '14b', 30: 15, 31: '15a', 32: '15b', 33: 16, 34: '16a', 35: 22, 36: '21a', 37: 21},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        'num_rings': 9,
    },
    # Full P-25.5.3 target (BB:14527): the bare bridged-fused parent ring system
    # 12,19:13,18-di(metheno)dinaphtho[2,3-a:2',3'-o]pentaphene. Two identical
    # one-carbon metheno bridges over the dinaphtho..pentaphene core; bridge
    # atoms numbered LAST (23,24) per P-25.4.4. All-aromatic mancude parent.
    "12,19:13,18-di(metheno)dinaphtho[2,3-a:2',3'-o]pentaphene": {
        'canonical_smiles': 'c1cc2cc3ccc4cc5ccc6cc7ccc8cc9ccc%10cc1c1cc%10c9cc8c7cc6c5cc4c3cc21',
        'smarts': 'c1cc2cc3ccc4cc5ccc6cc7ccc8cc9ccc%10cc1c1cc%10c9cc8c7cc6c5cc4c3cc21',
        'num_atoms': 40,
        'iupac_numbering': {0: 8, 1: 7, 2: '6a', 3: 6, 4: '5a', 5: 5, 6: 4, 7: '3a', 8: 3, 9: '2a', 10: 2, 11: 1, 12: '22a', 13: 22, 14: '21a', 15: 21, 16: 20, 17: 19, 18: 23, 19: 12, 20: 11, 21: 10, 22: '9a', 23: 9, 24: '8a', 25: '14a', 26: 14, 27: '13a', 28: 13, 29: 24, 30: 18, 31: '17a', 32: 17, 33: '16b', 34: '16a', 35: 16, 36: '15b', 37: '15a', 38: 15, 39: '14b'},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 24],
        'num_rings': 11,
    },
}


# Reverse lookup: canonical SMILES -> PAH name
_SMILES_TO_NAME: Dict[str, str] = {
    data['canonical_smiles']: name
    for name, data in POLYCYCLIC_DATA.items()
}


def get_polycyclic_by_smiles(canonical_smiles: str) -> Optional[Dict[str, Any]]:
    """
    Look up polycyclic data by canonical SMILES.

    Args:
        canonical_smiles: RDKit canonical SMILES string

    Returns:
        Dict with 'name' and all PAH data if found, None otherwise

    Example:
        >>> get_polycyclic_by_smiles('c1ccc2ccccc2c1')
        {'name': 'naphthalene', 'canonical_smiles': 'c1ccc2ccccc2c1', ...}
    """
    name = _SMILES_TO_NAME.get(canonical_smiles)
    if name is None:
        return None

    return {'name': name, **POLYCYCLIC_DATA[name]}


def get_polycyclic_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Look up polycyclic data by name.

    Args:
        name: PAH name (e.g., 'naphthalene', 'anthracene')

    Returns:
        Dict with PAH data if found, None otherwise
    """
    if name not in POLYCYCLIC_DATA:
        return None

    return {'name': name, **POLYCYCLIC_DATA[name]}


def is_polycyclic_aromatic(canonical_smiles: str) -> bool:
    """
    Check if a canonical SMILES represents a known polycyclic aromatic.

    Args:
        canonical_smiles: RDKit canonical SMILES string

    Returns:
        True if the SMILES matches a known PAH
    """
    return canonical_smiles in _SMILES_TO_NAME


def get_pah_names() -> List[str]:
    """
    Get list of all supported PAH names.

    Returns:
        List of PAH names
    """
    return list(POLYCYCLIC_DATA.keys())


def match_polycyclic_core(mol) -> Optional[Tuple[str, Dict[int, int]]]:
    """
    Match a molecule against known PAH cores using substructure matching.

    For substituted PAHs, finds the largest matching core and returns
    the atom mapping from molecule indices to IUPAC locants.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (pah_name, atom_mapping) where atom_mapping is {mol_idx: iupac_locant}
        Returns None if no PAH core found

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        >>> name, mapping = match_polycyclic_core(mol)
        >>> name
        'naphthalene'
    """
    # Lazy import to avoid circular dependency
    from rdkit import Chem

    # Sort PAHs by size (largest first) to find best match
    pah_by_size = sorted(
        POLYCYCLIC_DATA.items(),
        key=lambda x: x[1]['num_atoms'],
        reverse=True
    )

    for pah_name, pah_data in pah_by_size:
        smarts = pah_data['smarts']
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue

        matches = mol.GetSubstructMatches(pattern)
        if matches:
            # Found a match - build atom mapping
            match_atoms = matches[0]

            # Verify the match contains expected number of atoms
            if len(match_atoms) != pah_data['num_atoms']:
                continue

            # Build atom index to locant mapping
            # The mapping is based on the SMARTS match order
            # For now, use position in match as proxy for locant
            # More sophisticated mapping uses _map_pah_atoms_to_iupac in polycyclics.py
            atom_mapping = {match_atoms[i]: i + 1 for i in range(len(match_atoms))}

            return (pah_name, atom_mapping)

    return None


def get_pah_core_atoms(mol, pah_name: str) -> Optional[Set[int]]:
    """
    Get the atom indices that form a PAH core in a molecule.

    Args:
        mol: RDKit Mol object
        pah_name: Name of the PAH (e.g., 'naphthalene')

    Returns:
        Set of atom indices forming the PAH core, or None if no match

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        >>> core = get_pah_core_atoms(mol, 'naphthalene')
        >>> len(core)  # 10 atoms in naphthalene core
        10
    """
    from rdkit import Chem

    if pah_name not in POLYCYCLIC_DATA:
        return None

    pah_data = POLYCYCLIC_DATA[pah_name]
    smarts = pah_data['smarts']
    pattern = Chem.MolFromSmarts(smarts)

    if pattern is None:
        return None

    matches = mol.GetSubstructMatches(pattern)
    if matches:
        return set(matches[0])

    return None


def get_pah_substituent_positions(mol, pah_name: str) -> List[int]:
    """
    Get IUPAC locant positions where substituents are attached.

    Args:
        mol: RDKit Mol object with substituted PAH
        pah_name: Name of the PAH core (e.g., 'naphthalene')

    Returns:
        List of IUPAC locants with substituents

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        >>> get_pah_substituent_positions(mol, 'naphthalene')
        [2]  # Methyl at position 2
    """

    core_atoms = get_pah_core_atoms(mol, pah_name)
    if not core_atoms:
        return []

    result = match_polycyclic_core(mol)
    if not result or result[0] != pah_name:
        return []

    _, atom_mapping = result

    # Find atoms with non-core neighbors (these have substituents)
    substituted_positions = []
    for atom_idx in core_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in core_atoms:
                # This core atom has a substituent
                if atom_idx in atom_mapping:
                    substituted_positions.append(atom_mapping[atom_idx])
                break

    return sorted(set(substituted_positions))


# =============================================================================
# Complex Fusion Data
# Pre-computed fusion descriptors for known complex polycyclic systems
# =============================================================================

# Complex fusion data for multi-component and advanced fused systems
# Key: canonical name
# Value: dict with smiles, prefix, descriptor, parent, child, child_count
COMPLEX_FUSION_DATA: Dict[str, Dict[str, Any]] = {
    # Dibenzo compounds (two benzene rings fused to parent)
    'dibenzo[a,c]anthracene': {
        'smiles': 'c1ccc2c(c1)cc1ccc3cc4ccccc4cc3c1c2',
        'prefix': 'dibenzo',
        'descriptor': '[a,c]',
        'parent': 'anthracene',
        'child': 'benzene',
        'child_count': 2,
        'num_rings': 5,
    },
    'dibenzo[a,h]anthracene': {
        'smiles': 'c1ccc2c(c1)ccc1cc3ccc4ccccc4c3cc12',
        'prefix': 'dibenzo',
        'descriptor': '[a,h]',
        'parent': 'anthracene',
        'child': 'benzene',
        'child_count': 2,
        'num_rings': 5,
    },
    'dibenzo[a,j]anthracene': {
        'smiles': 'c1ccc2c(c1)c3ccc4ccccc4c3cc2c1ccccc1',
        'prefix': 'dibenzo',
        'descriptor': '[a,j]',
        'parent': 'anthracene',
        'child': 'benzene',
        'child_count': 2,
        'num_rings': 5,
    },

    # Naphtho compounds (naphthalene fused to heterocycle)
    'naphtho[2,3-b]furan': {
        'smiles': 'c1ccc2cc3occc3cc2c1',
        'prefix': 'naphtho',
        'descriptor': '[2,3-b]',
        'parent': 'furan',
        'child': 'naphthalene',
        'child_count': 1,
        'num_rings': 3,
    },
    'naphtho[1,2-b]furan': {
        'smiles': 'c1ccc2c(c1)cc1ccoc1c2',
        'prefix': 'naphtho',
        'descriptor': '[1,2-b]',
        'parent': 'furan',
        'child': 'naphthalene',
        'child_count': 1,
        'num_rings': 3,
    },
    'naphtho[2,3-b]thiophene': {
        'smiles': 'c1ccc2cc3sccc3cc2c1',
        'prefix': 'naphtho',
        'descriptor': '[2,3-b]',
        'parent': 'thiophene',
        'child': 'naphthalene',
        'child_count': 1,
        'num_rings': 3,
    },
    'naphtho[1,2-b]thiophene': {
        'smiles': 'c1ccc2c(c1)cc1ccsc1c2',
        'prefix': 'naphtho',
        'descriptor': '[1,2-b]',
        'parent': 'thiophene',
        'child': 'naphthalene',
        'child_count': 1,
        'num_rings': 3,
    },

    # Pyrido compounds (pyridine fused to pyrimidine/other heterocycles)
    'pyrido[2,3-d]pyrimidine': {
        'smiles': 'c1cnc2nccnc2c1',
        'prefix': 'pyrido',
        'descriptor': '[2,3-d]',
        'parent': 'pyrimidine',
        'child': 'pyridine',
        'child_count': 1,
        'num_rings': 2,
    },
    'pyrido[3,4-d]pyrimidine': {
        'smiles': 'c1cnc2ncncc2c1',
        'prefix': 'pyrido',
        'descriptor': '[3,4-d]',
        'parent': 'pyrimidine',
        'child': 'pyridine',
        'child_count': 1,
        'num_rings': 2,
    },
    'pyrido[4,3-d]pyrimidine': {
        'smiles': 'c1cnc2cncnc2c1',
        'prefix': 'pyrido',
        'descriptor': '[4,3-d]',
        'parent': 'pyrimidine',
        'child': 'pyridine',
        'child_count': 1,
        'num_rings': 2,
    },

    # Furo compounds
    'furo[2,3-b]pyridine': {
        'smiles': 'c1cc2ccoc2nc1',
        'prefix': 'furo',
        'descriptor': '[2,3-b]',
        'parent': 'pyridine',
        'child': 'furan',
        'child_count': 1,
        'num_rings': 2,
    },
    'furo[3,2-b]pyridine': {
        'smiles': 'c1cc2occc2nc1',
        'prefix': 'furo',
        'descriptor': '[3,2-b]',
        'parent': 'pyridine',
        'child': 'furan',
        'child_count': 1,
        'num_rings': 2,
    },

    # Thieno compounds
    'thieno[2,3-b]pyridine': {
        'smiles': 'c1cc2ccsc2nc1',
        'prefix': 'thieno',
        'descriptor': '[2,3-b]',
        'parent': 'pyridine',
        'child': 'thiophene',
        'child_count': 1,
        'num_rings': 2,
    },
    'thieno[3,2-b]pyridine': {
        'smiles': 'c1cc2sccc2nc1',
        'prefix': 'thieno',
        'descriptor': '[3,2-b]',
        'parent': 'pyridine',
        'child': 'thiophene',
        'child_count': 1,
        'num_rings': 2,
    },

    # Imidazo compounds
    'imidazo[1,2-a]pyridine': {
        'smiles': 'c1ccn2ccnc2c1',
        'prefix': 'imidazo',
        'descriptor': '[1,2-a]',
        'parent': 'pyridine',
        'child': 'imidazole',
        'child_count': 1,
        'num_rings': 2,
    },
    'imidazo[4,5-b]pyridine': {
        'smiles': 'c1cc2nc[nH]c2nc1',
        'prefix': 'imidazo',
        'descriptor': '[4,5-b]',
        'parent': 'pyridine',
        'child': 'imidazole',
        'child_count': 1,
        'num_rings': 2,
    },
}

# Reverse lookup: canonical SMILES -> complex fusion name
_COMPLEX_SMILES_TO_NAME: Dict[str, str] = {
    data['smiles']: name
    for name, data in COMPLEX_FUSION_DATA.items()
}


# Edge numbering data for common parent rings
# Maps parent ring name to dict of edge positions (0-indexed) to letters
# Edge 'a' is between IUPAC atoms 1-2, 'b' between 2-3, etc.
PARENT_RING_EDGES: Dict[str, Dict[int, str]] = {
    'naphthalene': {
        # 10 atoms, 10 edges (some are fusion edges, not substituable)
        # Edge labels based on IUPAC numbering
        0: 'a',  # between atoms 1-2
        1: 'b',  # between atoms 2-3
        2: 'c',  # between atoms 3-4 (peri-fusion)
        3: 'd',  # between atoms 4-4a
        4: 'e',  # between atoms 4a-5
        5: 'f',  # between atoms 5-6
        6: 'g',  # between atoms 6-7
        7: 'h',  # between atoms 7-8
        8: 'i',  # between atoms 8-8a
        9: 'j',  # between atoms 8a-1
    },
    'anthracene': {
        # 14 atoms, 14 edges
        0: 'a',   # between atoms 1-2
        1: 'b',   # between atoms 2-3
        2: 'c',   # between atoms 3-4
        3: 'd',   # between atoms 4-4a
        4: 'e',   # between atoms 4a-10
        5: 'f',   # between atoms 10-10a
        6: 'g',   # between atoms 10a-5
        7: 'h',   # between atoms 5-6
        8: 'i',   # between atoms 6-7
        9: 'j',   # between atoms 7-8
        10: 'k',  # between atoms 8-8a
        11: 'l',  # between atoms 8a-9
        12: 'm',  # between atoms 9-9a
        13: 'n',  # between atoms 9a-1
    },
    'phenanthrene': {
        # 14 atoms, 14 edges (angular arrangement)
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f',
        6: 'g',
        7: 'h',
        8: 'i',
        9: 'j',
        10: 'k',
        11: 'l',
        12: 'm',
        13: 'n',
    },
    'benzene': {
        # 6 atoms, 6 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f',
    },
    'furan': {
        # 5 atoms, 5 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
    },
    'thiophene': {
        # 5 atoms, 5 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
    },
    'pyrrole': {
        # 5 atoms, 5 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
    },
    'pyridine': {
        # 6 atoms, 6 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f',
    },
    'pyrimidine': {
        # 6 atoms, 6 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f',
    },
    'imidazole': {
        # 5 atoms, 5 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
    },
}


# Extended fusion prefixes (additions to existing FUSION_PREFIXES)
EXTENDED_FUSION_PREFIXES: Dict[str, str] = {
    # Additional prefixes not in main fusion_descriptors.py
    'phenanthro': 'phenanthro',  # explicit form
    'acenaphtho': 'acenaphtho',
    'acenaphtheno': 'acenaphtheno',  # for acenaphthene
    'fluoreno': 'fluoreno',
    'chryseno': 'chryseno',
    'triphenyleno': 'triphenyleno',
    'peryleno': 'peryleno',
    'coroneno': 'coroneno',
    'pyrazolo': 'pyrazolo',
    'isoxazolo': 'isoxazolo',
    'isothiazolo': 'isothiazolo',
    'oxazolo': 'oxazolo',
    'thiazolo': 'thiazolo',
    'triazolo': 'triazolo',
    'tetrazolo': 'tetrazolo',
    'pyrazino': 'pyrazino',
    'pyridazino': 'pyridazino',
    'triazino': 'triazino',
}


def get_complex_fusion_info(smiles: str) -> Optional[Tuple[str, str, str]]:
    """
    Look up pre-computed fusion descriptor for a known complex polycyclic.

    Args:
        smiles: SMILES string of the compound

    Returns:
        Tuple of (prefix, descriptor, parent) if found, None otherwise
        Example: ('dibenzo', '[a,c]', 'anthracene')

    Example:
        >>> get_complex_fusion_info('c1ccc2c(c1)cc1ccc3cc4ccccc4cc3c1c2')
        ('dibenzo', '[a,c]', 'anthracene')
    """
    # Try exact SMILES match first
    name = _COMPLEX_SMILES_TO_NAME.get(smiles)
    if name and name in COMPLEX_FUSION_DATA:
        data = COMPLEX_FUSION_DATA[name]
        return (data['prefix'], data['descriptor'], data['parent'])

    # Canonicalize and try again
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            name = _COMPLEX_SMILES_TO_NAME.get(canonical)
            if name and name in COMPLEX_FUSION_DATA:
                data = COMPLEX_FUSION_DATA[name]
                return (data['prefix'], data['descriptor'], data['parent'])
    except Exception:
        pass

    return None


def get_complex_fusion_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Look up complex fusion data by name.

    Args:
        name: Full fusion name (e.g., 'dibenzo[a,c]anthracene')

    Returns:
        Dict with fusion data if found, None otherwise
    """
    return COMPLEX_FUSION_DATA.get(name)


def get_edge_letter(ring_name: str, edge_index: int) -> str:
    """
    Get the IUPAC edge letter for a given edge index in a parent ring.

    Args:
        ring_name: Name of the parent ring (e.g., 'anthracene')
        edge_index: 0-indexed edge position

    Returns:
        Edge letter ('a', 'b', etc.) or empty string if not found

    Example:
        >>> get_edge_letter('anthracene', 0)
        'a'
        >>> get_edge_letter('anthracene', 2)
        'c'
    """
    if ring_name not in PARENT_RING_EDGES:
        return ''

    edges = PARENT_RING_EDGES[ring_name]
    return edges.get(edge_index, '')


def get_all_edge_letters(ring_name: str) -> List[str]:
    """
    Get all available edge letters for a parent ring.

    Args:
        ring_name: Name of the parent ring

    Returns:
        List of edge letters in order

    Example:
        >>> get_all_edge_letters('benzene')
        ['a', 'b', 'c', 'd', 'e', 'f']
    """
    if ring_name not in PARENT_RING_EDGES:
        return []

    edges = PARENT_RING_EDGES[ring_name]
    return [edges[i] for i in sorted(edges.keys())]
