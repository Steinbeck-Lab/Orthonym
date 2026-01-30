"""
Retained names for fused heterocyclic systems.

IUPAC 2013 prefers retained names (indole over benzo[b]pyrrole) for these
common fused heterocycles. ALWAYS check this lookup before applying
systematic fusion naming rules.

Keys are canonical SMILES (verified with RDKit), values contain:
- name: The IUPAC retained name
- tautomer_locant: Position of indicated hydrogen (e.g., 1 for 1H-indole), or None
- ring_system: Classification (benzo-5-membered, benzo-6-membered, tricyclic, etc.)
- parent_atoms: Number of heavy atoms in parent ring system
- iupac_locants: Mapping from canonical atom index to IUPAC peripheral locant
"""

from typing import Dict, Optional, Tuple, List, Any, Union
from rdkit import Chem


# Fused heterocycle data - canonical SMILES verified with RDKit
# Format: canonical_smiles -> {name, tautomer_locant, ring_system, parent_atoms, iupac_locants}
#
# IUPAC locant mappings (iupac_locants):
# - Maps canonical SMILES atom index -> IUPAC peripheral locant
# - Fusion atoms get string locants like '3a', '7a', '4a', '8a'
# - Peripheral atoms get integer locants 1, 2, 3, etc.
# - Used by match_fused_heterocycle_core() for correct substituent position naming

FUSED_HETEROCYCLE_DATA: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # BENZO-FUSED 5-MEMBERED RINGS (N-containing, aromatic)
    # IUPAC peripheral numbering: 1-2-3-3a-4-5-6-7-7a (9 positions)
    # Heteroatom at position 1 is adjacent to fusion 7a
    # =========================================================================

    # Indole: benzo[b]pyrrole - N at position 1
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-[nH](4)-c(5)-c(6)-c2(7)-c1(8)
    'c1ccc2[nH]ccc2c1': {
        'name': '1H-indole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # 1H-Isoindole: benzo[c]pyrrole - N at position 2 (aromatic tautomer)
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-c(4)-[nH](5)-c(6)-c2(7)-c1(8)
    'c1ccc2c[nH]cc2c1': {
        'name': '1H-isoindole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: 1, 5: 2, 6: 3, 7: '7a', 8: 7},
    },

    # 2H-Isoindole: non-aromatic in 5-ring
    # Canonical atom order: C1(0)=N(1)c2(2)c(3)c(4)c(5)c(6)c2(7)C1(8)
    'C1=Nc2ccccc2C1': {
        'name': '2H-isoindole',
        'tautomer_locant': 2,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 1, 1: 2, 2: '3a', 3: 4, 4: 5, 5: 6, 6: 7, 7: '7a', 8: 3},
    },

    # Indazole: benzo[c]pyrazole - N at positions 1,2
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-[nH](4)-n(5)-c(6)-c2(7)-c1(8)
    'c1ccc2[nH]ncc2c1': {
        'name': '1H-indazole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # Benzimidazole: benzo[d]imidazole - N at positions 1,3
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-[nH](4)-c(5)-n(6)-c2(7)-c1(8)
    'c1ccc2[nH]cnc2c1': {
        'name': '1H-benzimidazole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # Benzotriazole: benzo[d][1,2,3]triazole - N at positions 1,2,3
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-[nH](4)-n(5)-n(6)-c2(7)-c1(8)
    'c1ccc2[nH]nnc2c1': {
        'name': '1H-benzotriazole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # =========================================================================
    # BENZO-FUSED 5-MEMBERED RINGS (O/S-containing)
    # =========================================================================

    # Benzofuran: benzo[b]furan
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-o(4)-c(5)-c(6)-c2(7)-c1(8)
    'c1ccc2occc2c1': {
        'name': '1-benzofuran',
        'tautomer_locant': None,  # No tautomeric H
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # Benzothiophene: benzo[b]thiophene
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-s(4)-c(5)-c(6)-c2(7)-c1(8)
    'c1ccc2sccc2c1': {
        'name': '1-benzothiophene',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # Benzoxazole: benzo[d]oxazole
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-o(4)-c(5)-n(6)-c2(7)-c1(8)
    'c1ccc2ocnc2c1': {
        'name': '1,3-benzoxazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # Benzisoxazole: benzo[c]isoxazole (1,2-benzisoxazole)
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-n(4)-o(5)-c(6)-c2(7)-c1(8)
    'c1ccc2nocc2c1': {
        'name': '1,2-benzisoxazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # Benzothiazole: benzo[d]thiazole
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-s(4)-c(5)-n(6)-c2(7)-c1(8)
    'c1ccc2scnc2c1': {
        'name': '1,3-benzothiazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # Benzisothiazole: benzo[c]isothiazole (1,2-benzisothiazole)
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-n(4)-s(5)-c(6)-c2(7)-c1(8)
    'c1ccc2nscc2c1': {
        'name': '1,2-benzisothiazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # =========================================================================
    # BENZO-FUSED 6-MEMBERED RINGS
    # IUPAC peripheral numbering: 1-2-3-4-4a-5-6-7-8-8a (10 positions)
    # =========================================================================

    # Quinoline: benzo[b]pyridine
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-n(4)-c(5)-c(6)-c(7)-c2(8)-c1(9)
    'c1ccc2ncccc2c1': {
        'name': 'quinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },

    # Isoquinoline: benzo[c]pyridine
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-c(4)-n(5)-c(6)-c(7)-c2(8)-c1(9)
    'c1ccc2cnccc2c1': {
        'name': 'isoquinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },

    # Quinazoline: benzo[d]pyrimidine
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-n(4)-c(5)-n(6)-c(7)-c2(8)-c1(9)
    'c1ccc2ncncc2c1': {
        'name': 'quinazoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },

    # Quinoxaline: benzo[e]pyrazine
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-n(4)-c(5)-c(6)-n(7)-c2(8)-c1(9)
    'c1ccc2nccnc2c1': {
        'name': 'quinoxaline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },

    # Cinnoline: benzo[c]pyridazine
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-c(4)-n(5)-n(6)-c(7)-c2(8)-c1(9)
    'c1ccc2cnncc2c1': {
        'name': 'cinnoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },

    # Phthalazine: benzo[d]pyridazine
    # Canonical atom order: c1(0)-c(1)-c(2)-c2(3)-n(4)-n(5)-c(6)-c(7)-c2(8)-c1(9)
    'c1ccc2nnccc2c1': {
        'name': 'phthalazine',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },

    # =========================================================================
    # NAPHTHYRIDINES (pyridopyridines)
    # IUPAC peripheral numbering: 1-2-3-4-4a-5-6-7-8-8a (10 positions)
    # =========================================================================

    # 1,5-naphthyridine
    # Canonical: c1cnc2ccncc2c1
    # N atoms at idx 2 (position 1) and idx 6 (position 5)
    'c1cnc2ccncc2c1': {
        'name': '1,5-naphthyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '8a', 4: 8, 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4},
    },

    # 1,7-naphthyridine
    # Canonical: c1cc2ccncc2cn1
    # N atoms at idx 5 (position 7) and idx 9 (position 1)
    'c1cc2ccncc2cn1': {
        'name': '1,7-naphthyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
        'iupac_locants': {0: 2, 1: 3, 2: '4a', 3: 4, 4: 5, 5: 6, 6: 7, 7: '8a', 8: 8, 9: 1},
    },

    # 1,8-naphthyridine
    # Canonical: c1cnc2nccnc2c1
    # N atoms at idx 2, 4, 7
    'c1cnc2nccnc2c1': {
        'name': '1,8-naphthyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '8a', 4: 8, 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4},
    },

    # 2,7-naphthyridine
    # Canonical: c1cnc2cnccc2c1
    # N atoms at idx 2 (position 2) and idx 5 (position 7)
    'c1cnc2cnccc2c1': {
        'name': '2,7-naphthyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
        'iupac_locants': {0: 4, 1: 3, 2: 2, 3: '8a', 4: 1, 5: 8, 6: 7, 7: 6, 8: '4a', 9: 5},
    },

    # =========================================================================
    # TRICYCLIC SYSTEMS
    # =========================================================================

    # Carbazole: dibenzo[b,d]pyrrole
    # IUPAC peripheral numbering: 1-2-3-4-4a-4b-5-6-7-8-8a-9-9a (13 positions)
    # Canonical: c1ccc2c(c1)[nH]c1ccccc12
    'c1ccc2c(c1)[nH]c1ccccc12': {
        'name': '9H-carbazole',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '9a', 4: '4a', 5: 9, 6: '4b', 7: 5, 8: 6, 9: 7, 10: 8, 11: '8a', 12: 4},
    },

    # Acridine: dibenzo[b,e]pyridine
    # IUPAC peripheral numbering: 1-2-3-4-4a-9-9a-10-10a-5-6-7-8-8a (14 positions)
    # Canonical: c1ccc2nc3ccccc3cc2c1
    'c1ccc2nc3ccccc3cc2c1': {
        'name': 'acridine',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '9a', 4: 9, 5: '8a', 6: 8, 7: 7, 8: 6, 9: 5, 10: '10a', 11: 10, 12: '4a', 13: 4},
    },

    # Phenazine: dibenzo[b,e]pyrazine
    # Canonical: c1ccc2nc3ccccc3nc2c1
    'c1ccc2nc3ccccc3nc2c1': {
        'name': 'phenazine',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '10a', 4: 10, 5: '4a', 6: 4, 7: 5, 8: 6, 9: 7, 10: '5a', 11: 9, 12: '9a', 13: 8},
    },

    # Phenoxazine: dibenzo[b,e][1,4]oxazine
    # Canonical: c1ccc2c(c1)Nc1ccccc1O2
    'c1ccc2c(c1)Nc1ccccc1O2': {
        'name': '10H-phenoxazine',
        'tautomer_locant': 10,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '10a', 4: '4a', 5: 10, 6: '5a', 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: '9a', 13: 4},
    },

    # Phenothiazine: dibenzo[b,e][1,4]thiazine
    # Canonical: c1ccc2c(c1)Nc1ccccc1S2
    'c1ccc2c(c1)Nc1ccccc1S2': {
        'name': '10H-phenothiazine',
        'tautomer_locant': 10,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '10a', 4: '4a', 5: 10, 6: '5a', 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: '9a', 13: 4},
    },

    # Xanthene: dibenzo[b,e]pyran (9H-xanthene)
    # Canonical: c1ccc2c(c1)Cc1ccccc1O2
    'c1ccc2c(c1)Cc1ccccc1O2': {
        'name': '9H-xanthene',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '9a', 4: '4a', 5: 9, 6: '4b', 7: 5, 8: 6, 9: 7, 10: 8, 11: '8a', 12: 10, 13: 4},
    },

    # Thianthrene: dibenzo[b,e][1,4]dithiine
    # Canonical: c1ccc2c(c1)Sc1ccccc1S2
    'c1ccc2c(c1)Sc1ccccc1S2': {
        'name': 'thianthrene',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '10a', 4: '4a', 5: 10, 6: '5a', 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: '9a', 13: 4},
    },

    # =========================================================================
    # N-BRIDGEHEAD SYSTEMS
    # =========================================================================

    # Indolizine: pyrrolo[1,2-a]pyridine
    # IUPAC numbering: 1-2-3-3a-4-5-6-7-8-8a (N is 4)
    # Canonical: c1ccn2cccc2c1
    'c1ccn2cccc2c1': {
        'name': 'indolizine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 5},
    },

    # =========================================================================
    # IMIDAZOPYRIDINES (pharmaceutical scaffolds)
    # Common in drug molecules: zolpidem, alpidem class
    # =========================================================================

    # Imidazo[1,2-a]pyridine: N-bridgehead system (most common)
    # IUPAC: 1(N)-2(C)-3(C)-4(N bridgehead)-5-6-7-8(pyridine)-8a(fusion)
    # Canonical: c1ccn2ccnc2c1
    'c1ccn2ccnc2c1': {
        'name': 'imidazo[1,2-a]pyridine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: 4, 4: 3, 5: 2, 6: 1, 7: '8a', 8: 8},
    },

    # Imidazo[1,5-a]pyridine: N-bridgehead system (different fusion)
    # Canonical: c1ccn2cncc2c1
    'c1ccn2cncc2c1': {
        'name': 'imidazo[1,5-a]pyridine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: 4, 4: 3, 5: 2, 6: 1, 7: '8a', 8: 8},
    },

    # 3H-Imidazo[4,5-b]pyridine: 3-deazapurine analog
    # Has NH at position 3
    # Canonical: c1cnc2[nH]cnc2c1
    'c1cnc2[nH]cnc2c1': {
        'name': '3H-imidazo[4,5-b]pyridine',
        'tautomer_locant': 3,
        'ring_system': 'imidazopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '7a', 4: 3, 5: 2, 6: 1, 7: '3a', 8: 4},
    },

    # 3H-Imidazo[4,5-c]pyridine: another 3-deazapurine analog
    # Canonical: c1cc2nc[nH]cc-2n1
    'c1cc2nc[nH]cc-2n1': {
        'name': '3H-imidazo[4,5-c]pyridine',
        'tautomer_locant': 3,
        'ring_system': 'imidazopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: 7, 7: '7a', 8: 1},
    },

    # =========================================================================
    # AZAINDOLES (pyrrolopyridines) - kinase inhibitor scaffolds
    # =========================================================================

    # 1H-Pyrrolo[2,3-b]pyridine (7-azaindole)
    # Common scaffold in kinase inhibitors (vemurafenib class)
    # IUPAC numbering similar to indole: 1-2-3-3a-4-5-6-7-7a
    # Canonical: c1cnc2[nH]ccc2c1
    'c1cnc2[nH]ccc2c1': {
        'name': '1H-pyrrolo[2,3-b]pyridine',
        'tautomer_locant': 1,
        'ring_system': 'azaindole',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # 1H-Pyrrolo[3,2-b]pyridine (4-azaindole)
    # Canonical: c1cc2cc[nH]c2cn1
    'c1cc2cc[nH]c2cn1': {
        'name': '1H-pyrrolo[3,2-b]pyridine',
        'tautomer_locant': 1,
        'ring_system': 'azaindole',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: '4a', 3: 2, 4: 3, 5: 1, 6: '7a', 7: 7, 8: 4},
    },

    # =========================================================================
    # CHROMENES (benzopyrans) - flavonoid scaffolds
    # =========================================================================

    # 2H-Chromene (2H-1-benzopyran)
    # Oxygen at position 1, indicated H at position 2
    # Canonical: C1=Cc2ccccc2OC1
    'C1=Cc2ccccc2OC1': {
        'name': '2H-chromene',
        'tautomer_locant': 2,
        'ring_system': 'benzopyran',
        'parent_atoms': 10,
        'iupac_locants': {0: 4, 1: 3, 2: '4a', 3: 5, 4: 6, 5: 7, 6: 8, 7: '8a', 8: 1, 9: 2},
    },

    # 4H-Chromene (4H-1-benzopyran)
    # Oxygen at position 1, indicated H at position 4
    # Canonical: C1=COc2ccccc2C1
    'C1=COc2ccccc2C1': {
        'name': '4H-chromene',
        'tautomer_locant': 4,
        'ring_system': 'benzopyran',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '8a', 4: 8, 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4},
    },

    # =========================================================================
    # PURINES AND PTERIDINES (nucleobase-related)
    # =========================================================================

    # Purine: imidazo[4,5-d]pyrimidine
    # IUPAC numbering: N1-C2-N3-C4-C5-C6-N7-C8-N9 (9 atoms)
    # Canonical: c1ncc2nc[nH]c2n1
    # Verified with adenine (6-aminopurine): amino at C6 = idx 2
    'c1ncc2nc[nH]c2n1': {
        'name': '9H-purine',
        'tautomer_locant': 9,
        'ring_system': 'purine',
        'parent_atoms': 9,
        'iupac_locants': {0: 2, 1: 1, 2: 6, 3: 5, 4: 7, 5: 8, 6: 9, 7: 4, 8: 3},
    },

    # Pteridine: pyrimido[4,5-b]pyrazine
    # IUPAC numbering: 1-2-3-4-4a-5-6-7-8-8a (10 positions)
    # Canonical: c1cnc2ncncc2n1
    'c1cnc2ncncc2n1': {
        'name': 'pteridine',
        'tautomer_locant': None,
        'ring_system': 'pteridine',
        'parent_atoms': 10,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: 1, 8: '8a', 9: 8},
    },

    # Adenine: 6-aminopurine (retained name for nucleobase)
    # Canonical: Nc1ncnc2nc[nH]c12
    # Note: 10 atoms including the amino group
    'Nc1ncnc2nc[nH]c12': {
        'name': 'adenine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 10,
        'is_retained_name': True,
        'iupac_locants': {0: 'N6', 1: 6, 2: 1, 3: 2, 4: 3, 5: 4, 6: 7, 7: 8, 8: 9, 9: 5},
    },

    # Hypoxanthine: 6-oxopurine (retained name)
    # Canonical: O=c1[nH]cnc2nc[nH]c12
    # Note: 10 atoms including the oxo group
    'O=c1[nH]cnc2nc[nH]c12': {
        'name': 'hypoxanthine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 10,
        'is_retained_name': True,
        'iupac_locants': {0: 'O6', 1: 6, 2: 1, 3: 2, 4: 3, 5: 4, 6: 7, 7: 8, 8: 9, 9: 5},
    },

    # =========================================================================
    # MISCELLANEOUS PHARMACEUTICAL HETEROCYCLES
    # =========================================================================

    # Pyrazolo[1,5-a]pyrimidine: common kinase inhibitor scaffold
    # N-bridgehead system (no NH)
    # Canonical: c1cnc2ccnn2c1
    'c1cnc2ccnn2c1': {
        'name': 'pyrazolo[1,5-a]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 3, 5: 2, 6: 1, 7: '3a', 8: 4},
    },

    # Thieno[2,3-b]pyridine: thiophene-pyridine fusion
    # Canonical: c1cnc2sccc2c1
    'c1cnc2sccc2c1': {
        'name': 'thieno[2,3-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'thienopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # =========================================================================
    # PARTIALLY SATURATED (dihydro, tetrahydro) VARIANTS
    # Same numbering as aromatic parent, but some atoms are sp3
    # =========================================================================

    # Indoline: 2,3-dihydro-1H-indole
    # Same numbering as indole
    # Canonical: c1ccc2c(c1)CCN2
    'c1ccc2c(c1)CCN2': {
        'name': 'indoline',
        'tautomer_locant': None,  # Saturated, no tautomeric H
        'ring_system': 'benzo-5-saturated',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: '3a', 5: 3, 6: 2, 7: 1, 8: 4},
    },

    # Isoindoline: 1,3-dihydro-2H-isoindole
    # Same numbering as isoindole
    # Canonical: c1ccc2c(c1)CNC2
    'c1ccc2c(c1)CNC2': {
        'name': 'isoindoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-saturated',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: '3a', 5: 1, 6: 2, 7: 3, 8: 4},
    },

    # 1,2,3,4-Tetrahydroquinoline
    # Same numbering as quinoline
    # Canonical: c1ccc2c(c1)CCCN2
    'c1ccc2c(c1)CCCN2': {
        'name': '1,2,3,4-tetrahydroquinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 4, 6: 3, 7: 2, 8: 1, 9: 5},
    },

    # 1,2,3,4-Tetrahydroisoquinoline
    # Same numbering as isoquinoline
    # Canonical: c1ccc2c(c1)CCNC2
    'c1ccc2c(c1)CCNC2': {
        'name': '1,2,3,4-tetrahydroisoquinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 4, 6: 3, 7: 2, 8: 1, 9: 5},
    },

    # Chromane: 3,4-dihydro-2H-chromene
    # IUPAC chromene numbering: O at position 1
    # Canonical: c1ccc2c(c1)CCCO2
    'c1ccc2c(c1)CCCO2': {
        'name': 'chromane',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 4, 6: 3, 7: 2, 8: 1, 9: 5},
    },

    # Isochromane: 3,4-dihydro-1H-isochromene
    # Canonical: c1ccc2c(c1)CCOC2
    'c1ccc2c(c1)CCOC2': {
        'name': 'isochromane',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 4, 6: 3, 7: 1, 8: 2, 9: 5},
    },
}


# Build SMARTS patterns for substructure matching
# Use the same SMILES but allow variable substituents
def _build_smarts_lookup() -> Dict[str, str]:
    """
    Build SMARTS patterns from canonical SMILES for substructure matching.

    Returns dict mapping canonical SMILES to core name.
    """
    return {smiles: data['name'] for smiles, data in FUSED_HETEROCYCLE_DATA.items()}


# Cache for substructure matching patterns
_SUBSTRUCTURE_PATTERNS: Dict[str, Chem.Mol] = {}


def _get_substructure_patterns() -> Dict[str, Chem.Mol]:
    """
    Get cached substructure patterns.

    Returns dict mapping canonical SMILES to RDKit Mol patterns.
    """
    global _SUBSTRUCTURE_PATTERNS
    if not _SUBSTRUCTURE_PATTERNS:
        for smiles in FUSED_HETEROCYCLE_DATA:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                _SUBSTRUCTURE_PATTERNS[smiles] = mol
    return _SUBSTRUCTURE_PATTERNS


def get_fused_heterocycle_name(mol: Chem.Mol) -> Optional[Tuple[str, Optional[int]]]:
    """
    Get retained name for an exact fused heterocycle match.

    Checks if the molecule is an unsubstituted fused heterocycle with a
    retained name. For substituted molecules, use match_fused_heterocycle_core.

    Args:
        mol: RDKit molecule object

    Returns:
        Tuple of (name, tautomer_locant) if found, None otherwise.
        tautomer_locant is the position of indicated hydrogen (e.g., 1 for 1H-indole),
        or None if no tautomeric hydrogen.

    Example:
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        >>> get_fused_heterocycle_name(mol)
        ('1H-indole', 1)

        >>> mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')  # quinoline
        >>> get_fused_heterocycle_name(mol)
        ('quinoline', None)
    """
    if mol is None:
        return None

    # Canonicalize input
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

    # Check exact match
    if canonical_smiles in FUSED_HETEROCYCLE_DATA:
        data = FUSED_HETEROCYCLE_DATA[canonical_smiles]
        return (data['name'], data['tautomer_locant'])

    return None


def match_fused_heterocycle_core(
    mol: Chem.Mol
) -> Optional[Tuple[str, Dict[int, Union[int, str]], str]]:
    """
    Match a substituted molecule against fused heterocycle cores.

    Uses substructure matching to identify if a molecule contains a known
    fused heterocycle core. Returns the core name and atom index mapping
    for locant assignment using pre-computed IUPAC peripheral locants.

    Args:
        mol: RDKit molecule object

    Returns:
        Tuple of (core_name, atom_mapping, core_smiles) if a core is matched, None otherwise.
        atom_mapping maps mol atom indices to IUPAC locants (int or str like '3a').
        core_smiles is the canonical SMILES of the matched core.

    Example:
        >>> mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')  # 5-methylindole
        >>> name, mapping, smiles = match_fused_heterocycle_core(mol)
        >>> name
        '1H-indole'
        >>> mapping[1]  # mol atom idx 1 (where methyl attaches)
        5  # IUPAC position 5 - correct for 5-methylindole!
    """
    if mol is None:
        return None

    patterns = _get_substructure_patterns()
    best_match: Optional[Tuple[str, List[int], int, str]] = None

    # Find the largest matching core
    for smiles, pattern in patterns.items():
        if mol.HasSubstructMatch(pattern):
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                match = matches[0]  # Take first match
                data = FUSED_HETEROCYCLE_DATA[smiles]
                core_size = data['parent_atoms']

                # Keep the largest matching core
                if best_match is None or core_size > best_match[2]:
                    best_match = (data['name'], list(match), core_size, smiles)

    if best_match is None:
        return None

    name, match_atoms, _, core_smiles = best_match
    data = FUSED_HETEROCYCLE_DATA[core_smiles]

    # Get pre-computed IUPAC locant mapping
    iupac_locants = data.get('iupac_locants')

    if iupac_locants is None:
        # Fallback to old behavior if iupac_locants not defined
        atom_mapping = {atom_idx: locant + 1 for locant, atom_idx in enumerate(match_atoms)}
    else:
        # Use pre-computed IUPAC locants
        # match_atoms[pattern_idx] = mol_atom_idx
        # iupac_locants[pattern_idx] = iupac_locant
        atom_mapping = {}
        for pattern_idx, mol_atom_idx in enumerate(match_atoms):
            iupac_locant = iupac_locants.get(pattern_idx)
            if iupac_locant is not None:
                atom_mapping[mol_atom_idx] = iupac_locant

    return (name, atom_mapping, core_smiles)


def get_fused_heterocycle_info(canonical_smiles: str) -> Optional[Dict[str, Any]]:
    """
    Get full information about a fused heterocycle from canonical SMILES.

    Args:
        canonical_smiles: Canonical SMILES string

    Returns:
        Dict with name, tautomer_locant, ring_system, parent_atoms, iupac_locants
        or None if not found.
    """
    return FUSED_HETEROCYCLE_DATA.get(canonical_smiles)


def is_fused_heterocycle(mol: Chem.Mol) -> bool:
    """
    Check if molecule is a known fused heterocycle (exact match only).

    Args:
        mol: RDKit molecule object

    Returns:
        True if molecule is a known fused heterocycle
    """
    if mol is None:
        return False
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return canonical_smiles in FUSED_HETEROCYCLE_DATA


def get_ring_system_type(mol: Chem.Mol) -> Optional[str]:
    """
    Get the ring system type classification for a fused heterocycle.

    Args:
        mol: RDKit molecule object

    Returns:
        Ring system type string (e.g., 'benzo-5-membered', 'tricyclic'),
        or None if not a known fused heterocycle
    """
    if mol is None:
        return None
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    data = FUSED_HETEROCYCLE_DATA.get(canonical_smiles)
    if data:
        return data['ring_system']
    return None
