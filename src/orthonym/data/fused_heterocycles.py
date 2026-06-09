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

from collections import Counter
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

    # 1,6-naphthyridine  [WSD-03 (Phase 175): label corrected from the wrong
    # '1,5-naphthyridine'. OPSIN confirms this key (c1cnc2ccncc2c1) IS
    # 1,6-naphthyridine; the iupac_locants below already place the N's at
    # positions 1 (idx 2) and 6 (idx 6), so only the name/comment were wrong.
    # The true 1,5-naphthyridine is keyed c1cnc2cccnc2c1 below.]
    # Canonical: c1cnc2ccncc2c1
    # N atoms at idx 2 (position 1) and idx 6 (position 6)
    'c1cnc2ccncc2c1': {
        'name': '1,6-naphthyridine',
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
    # Empirically verified: N(idx 6)=9, Ring A peripheral: 2,1,0,5 -> 1,2,3,4
    # Ring B peripheral: 8,9,10,11 -> 5,6,7,8. Fusion: 4(=4a),3(=9a),7(=4b),12(=8a)
    'c1ccc2c(c1)[nH]c1ccccc12': {
        'name': '9H-carbazole',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {2: 1, 1: 2, 0: 3, 5: 4, 4: '4a', 3: '9a', 6: 9, 7: '4b', 12: '8a', 8: 5, 9: 6, 10: 7, 11: 8},
    },

    # Acridine: dibenzo[b,e]pyridine
    # IUPAC peripheral numbering: 1-2-3-4-4a-10(N)-10a-5-8a-6-7-8-9-9a (14 positions)
    # Canonical: c1ccc2nc3ccccc3cc2c1
    # Empirically verified: N(idx 4)=10. Ring A: 2,1,0,13 -> 1,2,3,4
    # Ring C: 9,8,7,6 -> 6,7,8,9. Fusion: 12(=4a),3(=10a),5(=9a),10(=8a)
    'c1ccc2nc3ccccc3cc2c1': {
        'name': 'acridine',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {2: 1, 1: 2, 0: 3, 13: 4, 12: '4a', 3: '10a', 4: 10, 11: 5, 5: '9a', 10: '8a', 9: 6, 8: 7, 7: 8, 6: 9},
    },

    # Phenazine: dibenzo[b,e]pyrazine
    # Canonical: c1ccc2nc3ccccc3nc2c1
    # Empirically verified: N(idx 4)=10, N(idx 11)=5. Ring A: 2,1,0,13 -> 1,2,3,4
    # Ring B: 9,8,7,6 -> 6,7,8,9. Fusion: 12(=4a),3(=10a),5(=9a),10(=5a)
    'c1ccc2nc3ccccc3nc2c1': {
        'name': 'phenazine',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {2: 1, 1: 2, 0: 3, 13: 4, 12: '4a', 3: '10a', 4: 10, 11: 5, 5: '9a', 10: '5a', 9: 6, 8: 7, 7: 8, 6: 9},
    },

    # Phenoxazine: dibenzo[b,e][1,4]oxazine
    # Canonical: c1ccc2c(c1)Nc1ccccc1O2
    # Empirically verified: N(idx 6)=10, O(idx 13)=5. Same topology as phenothiazine.
    # Ring A: 2,1,0,5 -> 1,2,3,4. Ring B: 8,9,10,11 -> 6,7,8,9
    # Fusion: 4(=4a),3(=10a),7(=5a),12(=9a)
    'c1ccc2c(c1)Nc1ccccc1O2': {
        'name': '10H-phenoxazine',
        'tautomer_locant': 10,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {2: 1, 1: 2, 0: 3, 5: 4, 4: '4a', 3: '10a', 6: 10, 13: 5, 7: '5a', 12: '9a', 8: 6, 9: 7, 10: 8, 11: 9},
    },

    # Phenothiazine: dibenzo[b,e][1,4]thiazine
    # Canonical: c1ccc2c(c1)Nc1ccccc1S2
    # Empirically verified: N(idx 6)=10, S(idx 13)=5. Same topology as phenoxazine.
    # Ring A: 2,1,0,5 -> 1,2,3,4. Ring B: 8,9,10,11 -> 6,7,8,9
    # Fusion: 4(=4a),3(=10a),7(=5a),12(=9a)
    'c1ccc2c(c1)Nc1ccccc1S2': {
        'name': '10H-phenothiazine',
        'tautomer_locant': 10,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {2: 1, 1: 2, 0: 3, 5: 4, 4: '4a', 3: '10a', 6: 10, 13: 5, 7: '5a', 12: '9a', 8: 6, 9: 7, 10: 8, 11: 9},
    },

    # Xanthene: dibenzo[b,e]pyran (9H-xanthene)
    # Canonical: c1ccc2c(c1)Cc1ccccc1O2
    # Empirically verified: O(idx 13) at position 5 (oxygen bridge), C(idx 6)=9.
    # Same angular tricyclic topology. Ring A: 2,1,0,5 -> 1,2,3,4
    # Ring B: 8,9,10,11 -> 5,6,7,8. Fusion: 4(=4a),3(=9a),7(=4b),12(=8a)
    'c1ccc2c(c1)Cc1ccccc1O2': {
        'name': '9H-xanthene',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {2: 1, 1: 2, 0: 3, 5: 4, 4: '4a', 3: '9a', 6: 9, 7: '4b', 12: '8a', 8: 5, 9: 6, 10: 7, 11: 8, 13: 10},
    },

    # Thianthrene: dibenzo[b,e][1,4]dithiine
    # Canonical: c1ccc2c(c1)Sc1ccccc1S2
    # Empirically verified: S(idx 6)=10, S(idx 13)=5. Same topology as phenothiazine.
    # Ring A: 2,1,0,5 -> 1,2,3,4. Ring B: 8,9,10,11 -> 6,7,8,9
    # Fusion: 4(=4a),3(=10a),7(=5a),12(=9a)
    'c1ccc2c(c1)Sc1ccccc1S2': {
        'name': 'thianthrene',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {2: 1, 1: 2, 0: 3, 5: 4, 4: '4a', 3: '10a', 6: 10, 13: 5, 7: '5a', 12: '9a', 8: 6, 9: 7, 10: 8, 11: 9},
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

    # Coumarin (2H-chromen-2-one) - lactone form of chromene
    # IUPAC: 2H-1-benzopyran-2-one
    # Numbering: O(1)-C(2)-C(3)-C(4)-C(4a)-C(5)-C(6)-C(7)-C(8)-C(8a)
    # Canonical atom order (11 atoms including =O):
    # idx 0: =O, idx 1: C2 (lactone), idx 2: C3, idx 3: C4, idx 4: C4a,
    # idx 5: C5, idx 6: C6, idx 7: C7, idx 8: C8, idx 9: C8a, idx 10: O1
    'O=c1ccc2ccccc2o1': {
        'name': 'coumarin',
        'systematic': '2H-chromen-2-one',
        'tautomer_locant': 2,
        'ring_system': 'benzo-6-membered-lactone',
        'parent_atoms': 11,
        'iupac_locants': {0: '=O', 1: 2, 2: 3, 3: 4, 4: '4a', 5: 5, 6: 6, 7: 7, 8: 8, 9: '8a', 10: 1},
    },

    # 2,3-Dihydro-1-benzofuran (coumaran)
    # Saturated 5-membered ring fused to benzene
    # Numbering: O(1)-C(2)-C(3)-C(3a)-C(4)-C(5)-C(6)-C(7)-C(7a)
    # Canonical: c1ccc2c(c1)CCO2 (9 atoms)
    # Canonical atom order: idx 0: C2, idx 1: C3, idx 2: C3a, idx 3: C4,
    # idx 4: C5, idx 5: C6, idx 6: C7, idx 7: C7a, idx 8: O1
    'c1ccc2c(c1)CCO2': {
        'name': '2,3-dihydro-1-benzofuran',
        'systematic': '2,3-dihydro-1-benzofuran',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered-saturated',
        'parent_atoms': 9,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: 7, 7: '7a', 8: 1},
    },

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
    # Adenine alternate tautomer (Phase 142: matches retained_names.py key)
    'Nc1ncnc2[nH]cnc12': {
        'name': 'adenine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 10,
        'is_retained_name': True,
        'iupac_locants': {0: 'N6', 1: 6, 2: 1, 3: 2, 4: 3, 5: 9, 6: 8, 7: 7, 8: 5, 9: 4},
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
    # Hypoxanthine alternate tautomer (Phase 142)
    'O=c1[nH]cnc2[nH]cnc12': {
        'name': 'hypoxanthine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 10,
        'is_retained_name': True,
        'iupac_locants': {0: 'O6', 1: 6, 2: 1, 3: 2, 4: 3, 5: 9, 6: 8, 7: 7, 8: 5, 9: 4},
    },

    # Guanine: 2-amino-1,9-dihydro-6H-purine-6-one (retained name)
    # Canonical: Nc1nc2[nH]cnc2c(=O)[nH]1
    # Note: 11 atoms including the amino and oxo groups
    'Nc1nc2[nH]cnc2c(=O)[nH]1': {
        'name': 'guanine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 11,
        'is_retained_name': True,
        'iupac_locants': {0: 'N2', 1: 2, 2: 3, 3: 9, 4: 8, 5: 7, 6: 5, 7: 6, 8: 'O6', 9: 1, 10: 4},
    },
    # Guanine alternate tautomer (Phase 142)
    'Nc1nc(=O)c2[nH]cnc2[nH]1': {
        'name': 'guanine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 11,
        'is_retained_name': True,
        'iupac_locants': {0: 'N2', 1: 2, 2: 6, 3: 'O6', 4: 5, 5: 9, 6: 8, 7: 7, 8: 4, 9: 3, 10: 1},
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
    # Pattern atoms: 0-5 = benzo ring, 6 = C(pos 1), 7 = N(pos 2), 8 = C(pos 3)
    'c1ccc2c(c1)CNC2': {
        'name': 'isoindoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-saturated',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: '3a', 5: 4, 6: 1, 7: 2, 8: 3},
    },

    # 1,2,3,4-Tetrahydroquinoline
    # Same numbering as quinoline: N at position 1
    # Canonical: c1ccc2c(c1)CCCN2
    'c1ccc2c(c1)CCCN2': {
        'name': '1,2,3,4-tetrahydroquinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },

    # 1,2,3,4-Tetrahydroisoquinoline
    # Same numbering as isoquinoline: N at position 2
    # Canonical: c1ccc2c(c1)CCNC2
    'c1ccc2c(c1)CCNC2': {
        'name': '1,2,3,4-tetrahydroisoquinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },

    # Chromane: 3,4-dihydro-2H-chromene
    # IUPAC chromene numbering: O at position 1
    # Canonical: c1ccc2c(c1)CCCO2
    'c1ccc2c(c1)CCCO2': {
        'name': 'chromane',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },

    # Isochromane: 3,4-dihydro-1H-isochromene
    # Canonical: c1ccc2c(c1)CCOC2
    'c1ccc2c(c1)CCOC2': {
        'name': 'isochromane',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },

    # =========================================================================
    # ADDITIONAL BENZO-FUSED HETEROCYCLES (Phase 8 expansion)
    # =========================================================================

    # 2,1,3-Benzothiadiazole: benzo[c][1,2,5]thiadiazole
    # Canonical: c1ccc2nsnc2c1
    'c1ccc2nsnc2c1': {
        'name': '2,1,3-benzothiadiazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # 2,1,3-Benzoxadiazole (benzofurazan): benzo[c][1,2,5]oxadiazole
    # Canonical: c1ccc2nonc2c1
    'c1ccc2nonc2c1': {
        'name': '2,1,3-benzoxadiazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # =========================================================================
    # NAPHTHO-FUSED HETEROCYCLES (Phase 8 expansion)
    # =========================================================================

    # Naphtho[1,2-b]furan
    # Canonical: c1ccc2c(c1)ccc1occc12 (13 atoms)
    'c1ccc2c(c1)ccc1occc12': {
        'name': 'naphtho[1,2-b]furan',
        'tautomer_locant': None,
        'ring_system': 'naphtho-fused',
        'parent_atoms': 13,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4b', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: '2a', 11: '8b', 12: 9},
    },

    # Naphtho[1,2-b]thiophene
    # Canonical: c1ccc2c(c1)ccc1sccc12 (13 atoms)
    'c1ccc2c(c1)ccc1sccc12': {
        'name': 'naphtho[1,2-b]thiophene',
        'tautomer_locant': None,
        'ring_system': 'naphtho-fused',
        'parent_atoms': 13,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4b', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: '2a', 11: '8b', 12: 9},
    },

    # Naphtho[1,2-b]pyrrole
    # Canonical: c1ccc2c(c1)ccc1[nH]ccc12 (13 atoms)
    'c1ccc2c(c1)ccc1[nH]ccc12': {
        'name': '1H-naphtho[1,2-b]pyrrole',
        'tautomer_locant': 1,
        'ring_system': 'naphtho-fused',
        'parent_atoms': 13,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4b', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: '2a', 11: '8b', 12: 9},
    },

    # Naphtho[2,3-b]furan
    # Canonical: c1ccc2cc3occc3cc2c1 (13 atoms)
    'c1ccc2cc3occc3cc2c1': {
        'name': 'naphtho[2,3-b]furan',
        'tautomer_locant': None,
        'ring_system': 'naphtho-fused',
        'parent_atoms': 13,
        'iupac_locants': {0: 7, 1: 8, 2: 9, 3: '9a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: '3a', 11: '9b', 12: 6},
    },

    # Naphtho[2,3-b]thiophene
    # Canonical: c1ccc2cc3sccc3cc2c1 (13 atoms)
    'c1ccc2cc3sccc3cc2c1': {
        'name': 'naphtho[2,3-b]thiophene',
        'tautomer_locant': None,
        'ring_system': 'naphtho-fused',
        'parent_atoms': 13,
        'iupac_locants': {0: 7, 1: 8, 2: 9, 3: '9a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: '3a', 11: '9b', 12: 6},
    },

    # Naphtho[2,3-b]pyrrole
    # Canonical: c1ccc2cc3[nH]ccc3cc2c1 (13 atoms)
    'c1ccc2cc3[nH]ccc3cc2c1': {
        'name': '1H-naphtho[2,3-b]pyrrole',
        'tautomer_locant': 1,
        'ring_system': 'naphtho-fused',
        'parent_atoms': 13,
        'iupac_locants': {0: 7, 1: 8, 2: 9, 3: '9a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: '3a', 11: '9b', 12: 6},
    },

    # =========================================================================
    # PYRIDO-FUSED SYSTEMS (Phase 8 expansion)
    # =========================================================================

    # Pyrido[2,3-d]pyrimidine
    # Canonical: c1cnc2ncncc2c1
    'c1cnc2ncncc2c1': {
        'name': 'pyrido[2,3-d]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'pyridopyrimidine',
        'parent_atoms': 10,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: 1, 8: '8a', 9: 8},
    },

    # 1,5-naphthyridine  [WSD-03 (Phase 175): label corrected from the wrong
    # 'pyrido[3,4-b]pyridine'. OPSIN confirms this key (c1cnc2cccnc2c1) IS
    # 1,5-naphthyridine (P-25.1.1 prefers the retained naphthyridine parent over
    # the pyrido[...]pyridine fusion construction). iupac_locants updated to the
    # 1,5 numbering: N (idx 2, idx 7) -> positions 1 and 5.]
    # Canonical: c1cnc2cccnc2c1   (N atoms at idx 2 -> position 1, idx 7 -> position 5)
    'c1cnc2cccnc2c1': {
        'name': '1,5-naphthyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '8a', 4: 8, 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4},
    },

    # Pyrido[2,3-b]pyrazine
    # Canonical: c1cnc2nccnc2c1
    'c1cnc2nccnc2c1': {
        'name': 'pyrido[2,3-b]pyrazine',
        'tautomer_locant': None,
        'ring_system': 'pyridopyrazine',
        'parent_atoms': 10,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: 1, 8: '8a', 9: 8},
    },

    # Pyrido[3,2-b]pyridine (1,7-naphthyridine isomer)
    # Canonical: c1cc2ccncc2cn1
    'c1cc2ccncc2cn1': {
        'name': 'pyrido[3,2-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
        'iupac_locants': {0: 2, 1: 3, 2: '4a', 3: 4, 4: 5, 5: 6, 6: 7, 7: '8a', 8: 8, 9: 1},
    },

    # Pyrido[3,4-b]pyridazine
    # Canonical: c1cnc2ccnnc2c1
    'c1cnc2ccnnc2c1': {
        'name': 'pyrido[3,4-b]pyridazine',
        'tautomer_locant': None,
        'ring_system': 'pyridopyridazine',
        'parent_atoms': 10,
        'iupac_locants': {0: 4, 1: 3, 2: 2, 3: '8a', 4: 8, 5: 7, 6: 6, 7: 5, 8: '4a', 9: 1},
    },

    # =========================================================================
    # IMIDAZO-FUSED SYSTEMS (Phase 8 expansion)
    # =========================================================================

    # Imidazo[1,2-b]pyridazine
    # Canonical: c1cnc2nccn2c1
    'c1cnc2nccn2c1': {
        'name': 'imidazo[1,2-b]pyridazine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '8a', 4: 3, 5: 2, 6: 1, 7: '3a', 8: 7},
    },

    # Imidazo[4,5-d]pyrimidine (purine core without NH)
    # Canonical: c1ncc2[nH]cnc2n1
    'c1ncc2[nH]cnc2n1': {
        'name': '1H-imidazo[4,5-d]pyrimidine',
        'tautomer_locant': 1,
        'ring_system': 'purine-related',
        'parent_atoms': 9,
        'iupac_locants': {0: 2, 1: 1, 2: 6, 3: 5, 4: 7, 5: 8, 6: 9, 7: 4, 8: 3},
    },

    # 1H-Imidazo[4,5-c]pyridine
    # Canonical: c1cnc2[nH]cnc2c1
    'c1cnc2[nH]cnc2c1': {
        'name': '1H-imidazo[4,5-c]pyridine',
        'tautomer_locant': 1,
        'ring_system': 'imidazopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 1, 5: 2, 6: 3, 7: '7a', 8: 4},
    },

    # =========================================================================
    # THIENO-FUSED SYSTEMS (Phase 8 expansion)
    # =========================================================================

    # Thieno[3,2-b]pyridine
    # Canonical: c1cnc2ccsc2c1
    'c1cnc2ccsc2c1': {
        'name': 'thieno[3,2-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'thienopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: '7a', 8: 1},
    },

    # Thieno[2,3-d]pyrimidine
    # Canonical: c1cnc2ncsc2c1
    'c1cnc2ncsc2c1': {
        'name': 'thieno[2,3-d]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'thienopyrimidine',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: '7a', 8: 1},
    },

    # =========================================================================
    # PYRROLO-FUSED SYSTEMS (Phase 8 expansion)
    # =========================================================================

    # 1H-Pyrrolo[3,2-c]pyridine
    # Canonical: c1cnc2cc[nH]c2c1
    'c1cnc2cc[nH]c2c1': {
        'name': '1H-pyrrolo[3,2-c]pyridine',
        'tautomer_locant': 1,
        'ring_system': 'azaindole',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 1, 7: '7a', 8: 2},
    },

    # =========================================================================
    # FURO-FUSED SYSTEMS (Phase 8 expansion)
    # =========================================================================

    # Furo[3,2-b]pyridine
    # Canonical: c1cnc2ccoc2c1
    'c1cnc2ccoc2c1': {
        'name': 'furo[3,2-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'furopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: '7a', 8: 1},
    },

    # Furo[2,3-b]pyridine
    # Canonical: c1cnc2occc2c1
    'c1cnc2occc2c1': {
        'name': 'furo[2,3-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'furopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },

    # =========================================================================
    # OXAZOLO-FUSED SYSTEMS (Phase 8 expansion)
    # =========================================================================

    # Oxazolo[4,5-b]pyridine
    # Canonical: c1cnc2ocnc2c1
    'c1cnc2ocnc2c1': {
        'name': 'oxazolo[4,5-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'oxazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: '7a', 8: 1},
    },

    # =========================================================================
    # TRIAZOLO-FUSED SYSTEMS (Phase 8 expansion)
    # =========================================================================

    # [1,2,4]Triazolo[1,5-a]pyridine
    # Canonical: c1cnc2nncn2c1
    'c1cnc2nncn2c1': {
        'name': '[1,2,4]triazolo[1,5-a]pyridine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 3, 5: 2, 6: 1, 7: '3a', 8: 4},
    },

    # =========================================================================
    # TRICYCLIC HETEROCYCLES (Phase 89 additions)
    # =========================================================================

    # Phenanthridine: benz[c]isoquinoline - angular tricyclic, N at position 5
    # IUPAC numbering: 1-2-3-4-4a-4b-5(N)-6-7-8-8a-9-10-10a (14 atoms)
    # Canonical: c1ccc2c(c1)cnc1ccccc12 (FIXED: was c1ccc2c(c1)ccc1cccnc12)
    'c1ccc2c(c1)cnc1ccccc12': {
        'name': 'phenanthridine',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {12: 1, 11: 2, 10: 3, 9: 4, 8: '4a', 4: '4b', 7: 5, 6: 6, 5: 7, 0: 8, 3: '8a', 1: 9, 2: 10, 13: '10a'},
    },

    # 9H-beta-Carboline: pyrido[3,4-b]indole - tricyclic, same topology as carbazole
    # IUPAC numbering: 1-2-3-4-4a-4b-5-6-7-8-8a-9-9a (13 atoms, 2N)
    # Canonical: c1ccc2c(c1)[nH]c1cnccc12
    # Empirically verified: N(idx 6)=9, N(idx 9)=6. Same topology as carbazole.
    'c1ccc2c(c1)[nH]c1cnccc12': {
        'name': '9H-beta-carboline',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {2: 1, 1: 2, 0: 3, 5: 4, 4: '4a', 3: '9a', 6: 9, 7: '4b', 12: '8a', 8: 5, 9: 6, 10: 7, 11: 8},
    },

    # Acridone: acridin-9(10H)-one - tricyclic, exocyclic =O at C-9, NH at pos 10
    # IUPAC numbering: follows acridine with =O at position 9
    # Canonical: O=c1c2ccccc2[nH]c2ccccc12
    'O=c1c2ccccc2[nH]c2ccccc12': {
        'name': 'acridone',
        'tautomer_locant': 10,
        'ring_system': 'tricyclic',
        'is_retained_name': True,  # Functional derivative (acridin-9(10H)-one), not a base ring system
        'parent_atoms': 15,
        'iupac_locants': {0: '=O', 1: 9, 2: '4a', 3: 4, 4: 3, 5: 2, 6: 1, 7: '9a', 8: 10, 9: '8a', 10: 8, 11: 7, 12: 6, 13: 5, 14: '10a'},
    },

    # =========================================================================
    # PHASE 107 ADDITIONS: Missing common tricyclic heterocycles
    # =========================================================================

    # Dibenzofuran: dibenzo[b,d]furan - two benzo rings fused to furan
    # Same topology as carbazole (NH -> O), IUPAC numbering same pattern
    # Canonical: c1ccc2c(c1)oc1ccccc12
    'c1ccc2c(c1)oc1ccccc12': {
        'name': 'dibenzofuran',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '9a', 4: '4a', 5: 4, 6: 9, 7: '4b', 8: 5, 9: 6, 10: 7, 11: 8, 12: '8a'},
    },

    # Dibenzothiophene: dibenzo[b,d]thiophene - two benzo rings fused to thiophene
    # Same topology as dibenzofuran (O -> S)
    # Canonical: c1ccc2c(c1)sc1ccccc12
    'c1ccc2c(c1)sc1ccccc12': {
        'name': 'dibenzothiophene',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '9a', 4: '4a', 5: 4, 6: 9, 7: '4b', 8: 5, 9: 6, 10: 7, 11: 8, 12: '8a'},
    },

    # Benzo[f]quinoline: angular tricyclic, N at position 1
    # IUPAC numbering: 1(N)-2-3-4-4a-5-6-6a-7-8-9-10-10a-10b
    # Canonical: c1ccc2c(c1)ccc1ncccc12
    'c1ccc2c(c1)ccc1ncccc12': {
        'name': 'benzo[f]quinoline',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {9: 1, 10: 2, 11: 3, 12: 4, 13: '4a', 7: 5, 6: 6, 4: '6a', 5: 7, 0: 8, 1: 9, 2: 10, 3: '10a', 8: '10b'},
    },

    # Benzo[h]quinoline: linear tricyclic, N at position 1
    # IUPAC numbering: 1(N)-2-3-4-4a-5-5a-6-7-8-9-9a-10-10a
    # Canonical: c1ccc2cc3ncccc3cc2c1
    'c1ccc2cc3ncccc3cc2c1': {
        'name': 'benzo[h]quinoline',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {6: 1, 7: 2, 8: 3, 9: 4, 10: '4a', 11: 5, 12: '5a', 13: 6, 0: 7, 1: 8, 2: 9, 3: '9a', 4: 10, 5: '10a'},
    },

    # Benzo[g]quinoline: linear tricyclic, N at position 3 (different from benzo[h])
    # IUPAC numbering follows standard peripheral path
    # Canonical: c1ccc2cc3cnccc3cc2c1
    'c1ccc2cc3cnccc3cc2c1': {
        'name': 'benzo[g]quinoline',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {7: 1, 8: 2, 9: 3, 6: 4, 5: '4a', 10: '5a', 11: 5, 12: '9a', 13: 6, 0: 7, 1: 8, 2: 9, 3: '9a', 4: 10},
    },

    # 9H-Fluoren-9-one (fluorenone): two benzo rings fused to cyclopentanone
    # Exocyclic =O at position 9
    # Canonical: O=C1c2ccccc2-c2ccccc21
    'O=C1c2ccccc2-c2ccccc21': {
        'name': '9H-fluoren-9-one',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'is_retained_name': True,
        'parent_atoms': 14,
        'iupac_locants': {0: '=O', 1: 9, 2: '9a', 3: 1, 4: 2, 5: 3, 6: 4, 7: '4a', 8: '4b', 9: 5, 10: 6, 11: 7, 12: 8, 13: '8a'},
    },

    # Phenanthridin-6(5H)-one: phenanthridine with =O at C-6, NH at N-5
    # Canonical: O=c1[nH]c2ccccc2c2ccccc12
    'O=c1[nH]c2ccccc2c2ccccc12': {
        'name': 'phenanthridin-6(5H)-one',
        'tautomer_locant': 5,
        'ring_system': 'tricyclic',
        'is_retained_name': True,
        'parent_atoms': 15,
        'iupac_locants': {0: '=O', 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: 1, 8: '10a', 9: '4b', 10: 7, 11: 8, 12: 9, 13: 10, 14: '10b'},
    },

    # =========================================================================
    # N-BRIDGEHEAD SYSTEMS (Phase 89 additions)
    # =========================================================================

    # 4H-Quinolizine: pyrido[1,2-a]pyridine analog, N bridgehead
    # IUPAC numbering: 1-2-3-4-4a(N)-6-7-8-9-9a (10 atoms, position 5 skipped)
    # Canonical: C1=CCN2C=CC=CC2=C1 (FIXED: was C1=CC2=CCC=CN2C=C1)
    'C1=CCN2C=CC=CC2=C1': {
        'name': '4H-quinolizine',
        'tautomer_locant': 4,
        'ring_system': 'bridgehead',
        'parent_atoms': 10,
        'iupac_locants': {9: 1, 0: 2, 1: 3, 2: 4, 3: '4a', 4: 6, 5: 7, 6: 8, 7: 9, 8: '9a'},
    },

    # Quinolizidine: decahydroquinolizine, fully saturated N-bridgehead
    # IUPAC numbering: same as quinolizine but fully saturated
    # Canonical: C1CCN2CCCCC2C1
    'C1CCN2CCCCC2C1': {
        'name': 'quinolizidine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 10,
        'iupac_locants': {9: 1, 0: 2, 1: 3, 2: 4, 3: '4a', 4: 5, 5: 6, 6: 7, 7: 8, 8: '8a'},
    },

    # Pyrrolizine: pyrrolo[1,2-a]pyrrole - bicyclic, N bridgehead
    # IUPAC numbering: 1-2-3-3a(N)-5-6-7-7a (8 atoms, position 4 skipped)
    # Canonical: C1=Cn2cccc2C1
    'C1=Cn2cccc2C1': {
        'name': 'pyrrolizine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 8,
        'iupac_locants': {7: 1, 0: 2, 1: 3, 2: '3a', 3: 5, 4: 6, 5: 7, 6: '7a'},
    },

    # =========================================================================
    # PHASE 101 ADDITIONS: New fused heterocycle entries
    # =========================================================================

    # Xanthone: 9H-xanthen-9-one - tricyclic with ring O and exocyclic =O
    # IUPAC numbering: 1-2-3-4-4a-4b-5-6-7-8-8a-9-9a (ring O at 10a, =O at 9)
    # Canonical: O=c1c2ccccc2oc2ccccc12
    'O=c1c2ccccc2oc2ccccc12': {
        'name': 'xanthone',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'is_retained_name': True,
        'parent_atoms': 15,
        'iupac_locants': {0: '=O', 1: 9, 2: '9a', 3: 1, 4: 2, 5: 3, 6: 4, 7: '4a', 8: '10a', 9: '4b', 10: 5, 11: 6, 12: 7, 13: 8, 14: '8a'},
    },

    # Thioxanthone: 9H-thioxanthen-9-one - sulfur analogue of xanthone
    # IUPAC numbering: same as xanthone but S instead of O in ring
    # Canonical: O=c1c2ccccc2sc2ccccc12
    'O=c1c2ccccc2sc2ccccc12': {
        'name': 'thioxanthone',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'is_retained_name': True,
        'parent_atoms': 15,
        'iupac_locants': {0: '=O', 1: 9, 2: '9a', 3: 1, 4: 2, 5: 3, 6: 4, 7: '4a', 8: '10a', 9: '4b', 10: 5, 11: 6, 12: 7, 13: 8, 14: '8a'},
    },

    # 1,10-Phenanthroline: angular tricyclic diazine, N at positions 1 and 10
    # IUPAC numbering: 1(N)-2-3-4-4a-4b-5-6-7-8-8a-9-10(N)-10a (14 atoms)
    # Canonical: c1cnc2c(c1)ccc1cccnc12
    'c1cnc2c(c1)ccc1cccnc12': {
        'name': '1,10-phenanthroline',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {12: 1, 11: 2, 10: 3, 9: 4, 8: '4a', 4: '4b', 7: 5, 6: 6, 5: 7, 0: 8, 3: '8a', 1: 9, 2: 10, 13: '10a'},
    },

    # 1H-Perimidine: 1H-perimidin-2-amine parent - peri-fused tricyclic
    # Naphtho[1,8-de] fused imidazoline
    # IUPAC numbering: 1-2-3-3a-4-5-6-6a-7-8-9-9a-9b (13 atoms)
    # Canonical: C1=Nc2cccc3cccc(c23)N1
    'C1=Nc2cccc3cccc(c23)N1': {
        'name': '1H-perimidine',
        'tautomer_locant': 1,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {12: 1, 0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: '6a', 7: 7, 8: 8, 9: 9, 10: '9a', 11: '9b'},
    },

    # Thioxanthene: dibenzo[b,e]thiopyran (9H-thioxanthene) - S analogue of xanthene
    # IUPAC numbering: same as xanthene but S at position 10 (bridge)
    # Canonical: c1ccc2c(c1)Cc1ccccc1S2
    # Empirically verified: S(idx 13)=10, C(idx 6)=9(CH2). Same topology as xanthene.
    'c1ccc2c(c1)Cc1ccccc1S2': {
        'name': '9H-thioxanthene',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {2: 1, 1: 2, 0: 3, 5: 4, 4: '4a', 3: '9a', 6: 9, 7: '4b', 12: '8a', 8: 5, 9: 6, 10: 7, 11: 8, 13: 10},
    },

    # =========================================================================
    # PHASE 109 EXPANSION: 60+ new entries to reach 150+ total
    # =========================================================================

    # Isobenzofuran: benzo[c]furan
    'c1ccc2cocc2c1': {
        'name': 'isobenzofuran',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 2,1-Benzisoxazole
    'c1ccc2oncc2c1': {
        'name': '2,1-benzisoxazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 1,3-Benzoselenazole
    'c1ccc2[se]cnc2c1': {
        'name': '1,3-benzoselenazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # Note: Naphthalene omitted to avoid substructure matching conflicts with naphtho-fused entries.
    # 1H-Indene
    'C1=Cc2ccccc2C1': {
        'name': '1H-indene',
        'tautomer_locant': 1,
        'ring_system': 'bicyclic-carbocyclic',
        'parent_atoms': 9,
        'iupac_locants': {8: 1, 0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: 7, 7: '7a'},
    },
    # Indane (2,3-dihydro-1H-indene)
    'c1ccc2c(c1)CCC2': {
        'name': 'indane',
        'tautomer_locant': None,
        'ring_system': 'bicyclic-saturated',
        'parent_atoms': 9,
        'iupac_locants': {8: 1, 7: 2, 6: 3, 4: '3a', 5: 4, 0: 5, 1: 6, 2: 7, 3: '7a'},
    },
    # Note: Tetralin (1,2,3,4-tetrahydronaphthalene) intentionally NOT included here.
    # It is a pure carbocyclic compound named via systematic naming (1,2,3,4-tetrahydronaphthalene).
    # Including it would override the systematic naming with the retained name "tetralin".
    # Note: Anthracene, phenanthrene, 9H-fluorene omitted to avoid substructure matching
    # conflicts with their derivatives (naphthacene, chrysene, fluorenone, etc.).
    # Biphenylene
    'c1ccc2c(c1)-c1ccccc1-2': {
        'name': 'biphenylene',
        'tautomer_locant': None,
        'ring_system': 'non-benzenoid',
        'parent_atoms': 12,
        'iupac_locants': {5: 1, 0: 2, 1: 3, 2: 4, 3: '4a', 11: '4b', 10: 5, 9: 6, 8: 7, 7: 8, 6: '8a', 4: '8b'},
    },
    # Acenaphthylene
    'C1=Cc2cccc3cccc1c23': {
        'name': 'acenaphthylene',
        'tautomer_locant': None,
        'ring_system': 'tricyclic-carbocyclic',
        'parent_atoms': 12,
        'iupac_locants': {0: 1, 1: 2, 2: '2a', 3: 3, 4: 4, 5: 5, 6: '5a', 7: 6, 8: 7, 9: 8, 10: '8a', 11: '8b'},
    },
    # 1H-Phenalene
    'C1=Cc2cccc3cccc(c23)C1': {
        'name': '1H-phenalene',
        'tautomer_locant': 1,
        'ring_system': 'tricyclic-carbocyclic',
        'parent_atoms': 13,
        'iupac_locants': {12: 1, 0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: '6a', 7: 7, 8: 8, 9: 9, 10: '9a', 11: '9b'},
    },
    # Pyrene
    'c1cc2ccc3cccc4ccc(c1)c2c34': {
        'name': 'pyrene',
        'tautomer_locant': None,
        'ring_system': 'tetracyclic-carbocyclic',
        'parent_atoms': 16,
        'iupac_locants': {1: 1, 0: 2, 13: 3, 12: '3a', 11: 4, 10: 5, 9: '5a', 8: 6, 7: 7, 6: 8, 5: '8a', 4: 9, 3: 10, 2: '10a', 14: '10b', 15: '10c'},
    },
    # Chrysene
    'c1ccc2c(c1)ccc1c3ccccc3ccc21': {
        'name': 'chrysene',
        'tautomer_locant': None,
        'ring_system': 'tetracyclic-carbocyclic',
        'parent_atoms': 18,
        'iupac_locants': {0: 2, 1: 3, 2: 4, 3: '4a', 4: '12a', 5: 1, 6: 12, 7: 11, 8: '10b', 9: '10a', 10: 10, 11: 9, 12: 8, 13: 7, 14: '6a', 15: 6, 16: 5, 17: '4b'},
    },
    # Triphenylene
    'c1ccc2c(c1)c1ccccc1c1ccccc21': {
        'name': 'triphenylene',
        'tautomer_locant': None,
        'ring_system': 'tetracyclic-carbocyclic',
        'parent_atoms': 18,
        'iupac_locants': {2: 1, 1: 2, 0: 3, 5: 4, 4: '4a', 6: '4b', 7: 5, 8: 6, 9: 7, 10: 8, 11: '8a', 12: '8b', 13: 9, 14: 10, 15: 11, 16: 12, 17: '12a', 3: '12b'},
    },
    # Naphthacene (tetracene)
    'c1ccc2cc3cc4ccccc4cc3cc2c1': {
        'name': 'naphthacene',
        'tautomer_locant': None,
        'ring_system': 'tetracyclic-carbocyclic',
        'parent_atoms': 18,
        'iupac_locants': {2: 1, 1: 2, 0: 3, 17: 4, 16: '4a', 15: 5, 14: '5a', 13: 6, 12: '6a', 11: 7, 10: 8, 9: 9, 8: 10, 7: '10a', 6: 11, 5: '11a', 4: 12, 3: '12a'},
    },
    # s-Indacene
    'C1=Cc2cc3c(cc2=C1)C=CC=3': {
        'name': 's-indacene',
        'tautomer_locant': None,
        'ring_system': 'non-benzenoid',
        'parent_atoms': 12,
        'iupac_locants': {1: 1, 0: 2, 8: 3, 7: '3a', 6: 4, 5: '4a', 9: 5, 10: 6, 11: 7, 4: '7a', 3: 8, 2: '8a'},
    },
    # as-Indacene
    'C1=Cc2c3c(ccc2=C1)=CC=C3': {
        'name': 'as-indacene',
        'tautomer_locant': None,
        'ring_system': 'non-benzenoid',
        'parent_atoms': 12,
        'iupac_locants': {1: 1, 0: 2, 8: 3, 7: '3a', 6: 4, 5: 5, 4: '5a', 9: 6, 10: 7, 11: 8, 3: '8a', 2: '8b'},
    },
    # Heptalene
    'C1=CC=C2C=CC=CC=C2C=1': {
        'name': 'heptalene',
        'tautomer_locant': None,
        'ring_system': 'non-benzenoid',
        'parent_atoms': 11,
        'iupac_locants': {3: 1, 2: 2, 1: 3, 0: 4, 10: '4a', 9: 5, 8: 6, 7: 7, 6: 8, 5: '8a', 4: 9},
    },
    # 1H-Pyrazolo[3,4-b]pyridine
    'c1cnc2[nH]ncc2c1': {
        'name': '1H-pyrazolo[3,4-b]pyridine',
        'tautomer_locant': 1,
        'ring_system': 'pyrazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # Oxazolo[5,4-b]pyridine
    'c1cnc2ncoc2c1': {
        'name': 'oxazolo[5,4-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'oxazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # Thiazolo[5,4-b]pyridine
    'c1cnc2scnc2c1': {
        'name': 'thiazolo[5,4-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'thiazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # Isoxazolo[5,4-b]pyridine
    'c1cnc2oncc2c1': {
        'name': 'isoxazolo[5,4-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'isoxazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # Thieno[3,2-d]pyrimidine
    'c1cnc2scnc2n1': {
        'name': 'thieno[3,2-d]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'thienopyrimidine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # Pyrido[2,3-c]pyridazine
    'c1cnc2cnncc2c1': {
        'name': 'pyrido[2,3-c]pyridazine',
        'tautomer_locant': None,
        'ring_system': 'pyridopyridazine',
        'parent_atoms': 10,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: 1, 8: '8a', 9: 8},
    },
    # Pyrazino[2,3-b]pyrazine
    'c1cnc2nccnc2n1': {
        'name': 'pyrazino[2,3-b]pyrazine',
        'tautomer_locant': None,
        'ring_system': 'pyrazinopyrazine',
        'parent_atoms': 10,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: 1, 8: '8a', 9: 8},
    },
    # Pyrido[3,2-d]pyrimidine
    'c1cc2ncncc2cn1': {
        'name': 'pyrido[3,2-d]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'pyridopyrimidine',
        'parent_atoms': 10,
        'iupac_locants': {0: 2, 1: 3, 2: '4a', 3: 4, 4: 5, 5: 6, 6: 7, 7: '8a', 8: 8, 9: 1},
    },
    # Pyrido[4,3-d]pyrimidine
    'c1cc2nccnc2cn1': {
        'name': 'pyrido[4,3-d]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'pyridopyrimidine',
        'parent_atoms': 10,
        'iupac_locants': {0: 2, 1: 3, 2: '4a', 3: 4, 4: 5, 5: 6, 6: 7, 7: '8a', 8: 8, 9: 1},
    },
    # Thieno[2,3-c]pyridine
    'c1cc2ccsc2cn1': {
        'name': 'thieno[2,3-c]pyridine',
        'tautomer_locant': None,
        'ring_system': 'thienopyridine',
        'parent_atoms': 9,
        'iupac_locants': {5: 1, 4: 2, 3: 3, 2: '3a', 1: 4, 0: 5, 8: 6, 7: 7, 6: '7a'},
    },
    # Thieno[3,4-b]pyridine
    'c1cc2sccc2cn1': {
        'name': 'thieno[3,4-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'thienopyridine',
        'parent_atoms': 9,
        'iupac_locants': {5: 1, 4: 2, 3: 3, 2: '3a', 1: 4, 0: 5, 8: 6, 7: 7, 6: '7a'},
    },
    # Furo[2,3-c]pyridine
    'c1cc2occc2cn1': {
        'name': 'furo[2,3-c]pyridine',
        'tautomer_locant': None,
        'ring_system': 'furopyridine',
        'parent_atoms': 9,
        'iupac_locants': {5: 1, 4: 2, 3: 3, 2: '3a', 1: 4, 0: 5, 8: 6, 7: 7, 6: '7a'},
    },
    # 1H-Pyrrolo[2,3-c]pyridine
    'c1cc2[nH]ccc2cn1': {
        'name': '1H-pyrrolo[2,3-c]pyridine',
        'tautomer_locant': 1,
        'ring_system': 'azaindole',
        'parent_atoms': 9,
        'iupac_locants': {5: 1, 4: 2, 3: 3, 2: '3a', 1: 4, 0: 5, 8: 6, 7: 7, 6: '7a'},
    },
    # Pyrazolo[1,5-a]pyridine
    'c1cnn2cccc2c1': {
        'name': 'pyrazolo[1,5-a]pyridine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 5},
    },
    # [1,2,3,4]Tetrazolo[1,5-a]pyridine
    'c1ccn2nnnc2c1': {
        'name': '[1,2,3,4]tetrazolo[1,5-a]pyridine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 5},
    },
    # [1,2,4]Triazolo[4,3-a]pyridine
    'c1ccn2ncnc2c1': {
        'name': '[1,2,4]triazolo[4,3-a]pyridine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 5},
    },
    # Imidazo[2,1-b]thiazole (5+5 bridgehead, 8 atoms)
    'c1cn2ccsc2n1': {
        'name': 'imidazo[2,1-b]thiazole',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 8,
        'iupac_locants': {0: 6, 1: 5, 2: '5a', 3: 3, 4: 2, 5: 1, 6: '3a', 7: 7},
    },
    # Pyrrolo[1,2-a]pyrimidine
    'c1cc2ccncn2c1': {
        'name': 'pyrrolo[1,2-a]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: '7a', 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: '8a'},
    },
    # Pyrrolo[1,2-a]pyrazine
    'c1cc2cnccn2c1': {
        'name': 'pyrrolo[1,2-a]pyrazine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: '7a', 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: '8a'},
    },
    # Pyrrolizidine (fully saturated, 8 atoms)
    'C1CC2CCCN2C1': {
        'name': 'pyrrolizidine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 8,
        'iupac_locants': {7: 1, 0: 2, 1: 3, 2: '3a', 3: 5, 4: 6, 5: 7, 6: '7a'},
    },
    # Indolizidine (fully saturated, 9 atoms)
    'C1CCN2CCCC2C1': {
        'name': 'indolizidine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {8: 1, 0: 2, 1: 3, 2: 4, 3: '4a', 4: 5, 5: 6, 6: 7, 7: '8a'},
    },
    # Phenoxathiin: dibenzo[b,e][1,4]oxathiine
    # Empirically verified: O(idx 6)=10, S(idx 13)=5. Same topology as phenothiazine.
    'c1ccc2c(c1)Oc1ccccc1S2': {
        'name': 'phenoxathiin',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {2: 1, 1: 2, 0: 3, 5: 4, 4: '4a', 3: '10a', 6: 10, 13: 5, 7: '5a', 12: '9a', 8: 6, 9: 7, 10: 8, 11: 9},
    },
    # Furo[3,2-b]quinoline
    'c1ccc2c(c1)oc1cccnc12': {
        'name': 'furo[3,2-b]quinoline',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '9a', 4: '4a', 5: 4, 6: 9, 7: '4b', 8: 5, 9: 6, 10: 7, 11: 8, 12: '8a'},
    },
    # Thieno[2,3-b]thiophene
    'c1cc2sccc2s1': {
        'name': 'thieno[2,3-b]thiophene',
        'tautomer_locant': None,
        'ring_system': 'thienothiophene',
        'parent_atoms': 8,
        'iupac_locants': {0: 3, 1: 2, 2: '3a', 3: 4, 4: 5, 5: 6, 6: '6a', 7: 1},
    },
    # Thieno[3,2-b]thiophene
    'c1cc2ccsc2s1': {
        'name': 'thieno[3,2-b]thiophene',
        'tautomer_locant': None,
        'ring_system': 'thienothiophene',
        'parent_atoms': 8,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: '6a', 7: 1},
    },
    # Xanthine (3,7-dihydro-1H-purine-2,6-dione)
    'O=c1[nH]c(=O)c2nc[nH]c2[nH]1': {
        'name': 'xanthine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 11,
        'is_retained_name': True,
        'iupac_locants': {0: 'O6', 1: 6, 2: 1, 3: 2, 4: 'O2', 5: 5, 6: 7, 7: 8, 8: 9, 9: 4, 10: 3},
    },
    # Xanthine alternate tautomer (Phase 142: canonical form from common SMILES input)
    'O=c1[nH]c(=O)c2[nH]cnc2[nH]1': {
        'name': 'xanthine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 11,
        'is_retained_name': True,
        'iupac_locants': {0: 'O6', 1: 6, 2: 1, 3: 2, 4: 'O2', 5: 3, 6: 9, 7: 8, 8: 7, 9: 5, 10: 4},
    },
    # Alloxazine (benzo[g]pteridine-2,4(1H,3H)-dione)
    'O=c1[nH]c(=O)c2nc3ccccc3nc2[nH]1': {
        'name': 'alloxazine',
        'tautomer_locant': None,
        'ring_system': 'pteridine-related',
        'parent_atoms': 16,
        'is_retained_name': True,
        'iupac_locants': {0: 'O2', 1: 2, 2: 3, 3: 4, 4: 'O4', 5: '4a', 6: 5, 7: '5a', 8: 6, 9: 7, 10: 8, 11: 9, 12: '9a', 13: 10, 14: '10a', 15: 1},
    },
    # Pyrimido[4,5-d]pyrimidine
    'c1ncc2ncncc2n1': {
        'name': 'pyrimido[4,5-d]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'pyrimidopyrimidine',
        'parent_atoms': 10,
        'iupac_locants': {9: 1, 0: 2, 1: 3, 2: 4, 3: '4a', 4: 5, 5: 6, 6: 7, 7: 8, 8: '8a'},
    },
    # 1H-Pyrrolo[2,3-b]pyrazine
    'c1ncc2[nH]ccc2n1': {
        'name': '1H-pyrrolo[2,3-b]pyrazine',
        'tautomer_locant': 1,
        'ring_system': 'pyrrolopyrazine',
        'parent_atoms': 9,
        'iupac_locants': {0: 2, 1: 1, 2: 6, 3: 5, 4: 7, 5: 8, 6: 9, 7: 4, 8: 3},
    },
    # 2H-Isothiochromene
    'C1=Cc2ccccc2SC1': {
        'name': '2H-isothiochromene',
        'tautomer_locant': 2,
        'ring_system': 'benzothiopyran',
        'parent_atoms': 10,
        'iupac_locants': {0: 4, 1: 3, 2: '4a', 3: 5, 4: 6, 5: 7, 6: 8, 7: '8a', 8: 1, 9: 2},
    },
    # Thiochromane
    'c1ccc2c(c1)CCCS2': {
        'name': 'thiochromane',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # 2,3-Dihydro-1,4-benzodioxine
    'c1ccc2c(c1)OCCO2': {
        'name': '2,3-dihydro-1,4-benzodioxine',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # 2,3-Dihydro-1-benzothiophene
    'c1ccc2c(c1)CCS2': {
        'name': '2,3-dihydro-1-benzothiophene',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-saturated',
        'parent_atoms': 9,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: 7, 7: '7a', 8: 1},
    },
    # 1,2,3,4-Tetrahydroquinazoline
    'c1ccc2c(c1)NCCN2': {
        'name': '1,2,3,4-tetrahydroquinazoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # 3,4-Dihydroisoquinoline
    'C1=NCc2ccccc2C1': {
        'name': '3,4-dihydroisoquinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 1, 1: 2, 2: 3, 9: 4, 8: '4a', 7: 5, 6: 6, 5: 7, 4: 8, 3: '8a'},
    },
    # 2,3-Dihydro-1H-pyrrolo[2,3-b]pyridine
    'c1cnc2c(c1)CCN2': {
        'name': '2,3-dihydro-1H-pyrrolo[2,3-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-saturated',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: '3a', 5: 4, 6: 3, 7: 2, 8: 1},
    },
    # Pyrazolo[1,5-a]pyrazine (bridgehead, 9 atoms)
    'c1cnn2ccnc2c1': {
        'name': 'pyrazolo[1,5-a]pyrazine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 5},
    },
    # 1H-Pyrazolo[3,4-c]pyridine
    'c1cc2[nH]ncc2cn1': {
        'name': '1H-pyrazolo[3,4-c]pyridine',
        'tautomer_locant': 1,
        'ring_system': 'pyrazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {5: 1, 4: 2, 3: 3, 2: '3a', 1: 4, 0: 5, 8: 6, 7: 7, 6: '7a'},
    },
    # Thieno[2,3-d]pyridazine
    'c1cc2ccsc2nn1': {
        'name': 'thieno[2,3-d]pyridazine',
        'tautomer_locant': None,
        'ring_system': 'thienopyridazine',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # Oxazolo[5,4-c]pyridine
    'c1cc2ncoc2cn1': {
        'name': 'oxazolo[5,4-c]pyridine',
        'tautomer_locant': None,
        'ring_system': 'oxazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {5: 1, 4: 2, 3: 3, 2: '3a', 1: 4, 0: 5, 8: 6, 7: 7, 6: '7a'},
    },
    # Thiazolo[5,4-c]pyridine
    'c1cc2ncsc2cn1': {
        'name': 'thiazolo[5,4-c]pyridine',
        'tautomer_locant': None,
        'ring_system': 'thiazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {5: 1, 4: 2, 3: 3, 2: '3a', 1: 4, 0: 5, 8: 6, 7: 7, 6: '7a'},
    },
    # Isoxazolo[4,5-b]pyridine
    'c1cnc2nocc2c1': {
        'name': 'isoxazolo[4,5-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'isoxazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # Pyrazino[2,3-c]pyridazine
    'c1cc2nccnc2nn1': {
        'name': 'pyrazino[2,3-c]pyridazine',
        'tautomer_locant': None,
        'ring_system': 'pyrazinopyridazine',
        'parent_atoms': 10,
        'iupac_locants': {0: 2, 1: 3, 2: '4a', 3: 4, 4: 5, 5: 6, 6: 7, 7: '8a', 8: 8, 9: 1},
    },
}


# =========================================================================
# PREFIX STEM DERIVATION (Phase 78)
# =========================================================================


def _derive_prefix_stem(name: str) -> str:
    """Derive IUPAC prefix stem from a fused heterocycle retained name.

    Rule: Drop terminal 'e' from names ending in 'e'.
    This covers all standard fused heterocycle naming patterns:
      quinoline → quinolin, 1H-indole → 1H-indol, acridine → acridin

    Names not ending in 'e' are returned unchanged:
      1-benzofuran → 1-benzofuran, coumarin → coumarin

    Args:
        name: The IUPAC retained name of the fused heterocycle.

    Returns:
        The prefix stem suitable for -yl suffix attachment.
    """
    if name.endswith('e'):
        return name[:-1]
    return name


# Entries that should not be included in prefix stems:
# - Retained names (adenine, hypoxanthine): functional derivatives, not ring systems
# - Lactones (coumarin): need special "oxo-chromen" prefix, deferred to Phase 80
_PREFIX_STEM_EXCLUDES = {'is_retained_name', 'benzo-6-membered-lactone'}

# Module-level derivation: compute all prefix stems at import time (O(1) lookup)
FUSED_HETEROCYCLE_PREFIX_STEMS: Dict[str, str] = {
    smiles: _derive_prefix_stem(data['name'])
    for smiles, data in FUSED_HETEROCYCLE_DATA.items()
    if not data.get('is_retained_name', False)
    and data.get('ring_system') not in _PREFIX_STEM_EXCLUDES
}


def get_fused_heterocycle_prefix(
    core_smiles: str,
    attachment_atom_idx: int,
    atom_mapping: Dict[int, Union[int, str]],
) -> Optional[str]:
    """Get prefix form for a fused heterocycle substituent.

    Static O(1) dict lookup — no recursive naming calls.
    Returns the IUPAC P-57.1.5 systematic prefix form:
      stem-locant-yl  (e.g., "quinolin-2-yl", "1H-indol-3-yl")

    Args:
        core_smiles: Canonical SMILES of the matched fused heterocycle core.
        attachment_atom_idx: Mol atom index where the ring attaches to parent.
        atom_mapping: Mapping from mol atom index → IUPAC locant (from
            match_fused_heterocycle_core()).

    Returns:
        Prefix string like "quinolin-2-yl" or None if core_smiles not recognized
        or attachment atom not mapped.
    """
    stem = FUSED_HETEROCYCLE_PREFIX_STEMS.get(core_smiles)
    if stem is None:
        return None

    locant = atom_mapping.get(attachment_atom_idx)
    if locant is None:
        return None

    return f"{stem}-{locant}-yl"


def get_substituted_fused_het_prefix(
    core_smiles: str,
    attachment_atom_idx: int,
    atom_mapping: Dict[int, Union[int, str]],
    inner_substituent_prefixes: str,
) -> Optional[str]:
    """Get compound prefix for a substituted fused heterocycle substituent.

    When a fused heterocycle ring has its own substituents (e.g., 5-methyl on
    indole) AND the whole ring is a substituent on another parent, this produces
    the compound prefix form like "(5-methyl-1H-indol-3-yl)".

    Static O(1) dict lookup for the stem — no recursive naming calls.

    Args:
        core_smiles: Canonical SMILES of the fused heterocycle core.
        attachment_atom_idx: Mol atom index where the ring attaches to parent.
        atom_mapping: Mapping from mol atom index → IUPAC locant.
        inner_substituent_prefixes: Pre-formatted inner substituent string
            (e.g., "5-methyl-" or "5,6-dimethyl-"). Already includes locants
            and multiplicative prefixes. May or may not end with hyphen.

    Returns:
        Compound prefix in parentheses, e.g., "(5-methyl-1H-indol-3-yl)"
        or None if core_smiles not recognized.
    """
    stem = FUSED_HETEROCYCLE_PREFIX_STEMS.get(core_smiles)
    if stem is None:
        return None

    locant = atom_mapping.get(attachment_atom_idx)
    if locant is None:
        return None

    # Ensure inner prefix ends with hyphen for proper concatenation
    inner = inner_substituent_prefixes.rstrip('-')
    if inner:
        prefix = f"({inner}-{stem}-{locant}-yl)"
    else:
        prefix = f"({stem}-{locant}-yl)"

    return prefix


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


# =========================================================================
# HYG-01 (Phase 173): connectivity-hash-bucketed ring lookup
# -------------------------------------------------------------------------
# match_fused_heterocycle_core previously scanned ALL ~150 catalog patterns
# with HasSubstructMatch for every query (O(N)). HYG-01 buckets the
# canonical-SMILES-keyed catalog (the RING-03 key) by a connectivity hash
# (cycle rank + heteroatom-element set) and prunes by necessary substructure
# conditions, so a query only tests the patterns that *could* match.
#
# Byte-identical by construction: every condition used to bucket/prune is a
# proven NECESSARY condition for subgraph isomorphism, so no real match is
# ever discarded; candidates are scanned in catalog insertion order with the
# same strict-`>` largest-match-wins / first-at-max tie-break as the old loop.
# Proofs (each a necessary condition for pattern ⊑ query):
#   * cycle rank: circuit rank is monotonic under subgraph (the cycle space of
#     a subgraph is a subspace of the host's) ⇒ rank(pattern) ≤ rank(query).
#   * per-element heavy-atom count: each pattern atom maps to a distinct query
#     atom of the SAME element ⇒ count(e, pattern) ≤ count(e, query) ∀e
#     (this subsumes the heteroatom-set ⊆ condition used for bucketing).
#   * total heavy-atom count: ≤, for the same reason.
# A HARD old-vs-new equivalence test (tests/unit/data/) gates this.
# =========================================================================

class _PatternRec:
    """A catalog pattern enriched with the descriptors HYG-01 buckets/prunes on."""

    __slots__ = ('index', 'smiles', 'pattern', 'name', 'parent_atoms',
                 'elem_counts', 'num_heavy', 'rank', 'hetset', 'has_exocyclic')

    def __init__(self, index, smiles, pattern, name, parent_atoms,
                 elem_counts, num_heavy, rank, hetset, has_exocyclic):
        self.index = index
        self.smiles = smiles
        self.pattern = pattern
        self.name = name
        self.parent_atoms = parent_atoms
        self.elem_counts = elem_counts
        self.num_heavy = num_heavy
        self.rank = rank
        self.hetset = hetset
        self.has_exocyclic = has_exocyclic


# Lazily-built connectivity-hash index. Bucket key = (cycle_rank, hetset).
_PATTERN_BUCKETS: Dict[Tuple[int, frozenset], List['_PatternRec']] = {}
_REC_BY_SMILES: Dict[str, '_PatternRec'] = {}
# Fast-path index: catalog rec keyed by ring_skeleton_key(pattern). Computed
# with the SAME function used on the query, so the keys are comparable.
_REC_BY_SKELETON: Dict[str, '_PatternRec'] = {}


def _build_pattern_index() -> None:
    """Build the bucketed pattern index once (mirrors _SUBSTRUCTURE_PATTERNS).

    Iterates FUSED_HETEROCYCLE_DATA in INSERTION ORDER and records each pattern's
    catalog index so the bucket scan can reproduce the old loop's tie-break.
    """
    global _PATTERN_BUCKETS, _REC_BY_SMILES, _REC_BY_SKELETON
    if _PATTERN_BUCKETS:
        return
    for idx, (smiles, data) in enumerate(FUSED_HETEROCYCLE_DATA.items()):
        pattern = Chem.MolFromSmiles(smiles)
        if pattern is None:
            # Match _get_substructure_patterns: skip unparseable keys. The
            # catalog index still advances so ordering stays aligned with the
            # original full-scan order.
            continue
        elem_counts = Counter(a.GetSymbol() for a in pattern.GetAtoms())
        hetset = frozenset(s for s in elem_counts if s not in ('C', 'H'))
        rank = pattern.GetRingInfo().NumRings()
        has_exocyclic = any(not a.IsInRing() for a in pattern.GetAtoms())
        rec = _PatternRec(
            index=idx,
            smiles=smiles,
            pattern=pattern,
            name=data['name'],
            parent_atoms=data['parent_atoms'],
            elem_counts=elem_counts,
            num_heavy=pattern.GetNumAtoms(),  # heavy atoms (implicit H)
            rank=rank,
            hetset=hetset,
            has_exocyclic=has_exocyclic,
        )
        _PATTERN_BUCKETS.setdefault((rank, hetset), []).append(rec)
        _REC_BY_SMILES[smiles] = rec
        # Index all-ring entries by their skeleton key for the O(1) fast path.
        # First-writer-wins keeps the lowest catalog index on a key collision,
        # mirroring the bucket scan's first-at-max tie-break.
        if not has_exocyclic:
            skel = ring_skeleton_key(pattern)
            if skel is not None and skel not in _REC_BY_SKELETON:
                _REC_BY_SKELETON[skel] = rec


def _gather_candidates(q_rank: int, q_hetset: frozenset) -> List['_PatternRec']:
    """Return catalog patterns whose bucket key clears the necessary conditions.

    A pattern can substructure-match the query only if its cycle rank is ≤ the
    query's and its heteroatom-element set is ⊆ the query's. Both are encoded in
    the bucket key, so we collect only the qualifying buckets (a tiny constant
    number) instead of scanning the whole table. Returned in catalog insertion
    order to preserve the largest-match / first-at-max tie-break.
    """
    _build_pattern_index()
    out: List['_PatternRec'] = []
    for (rank, hetset), recs in _PATTERN_BUCKETS.items():
        if rank <= q_rank and hetset <= q_hetset:
            out.extend(recs)
    out.sort(key=lambda rec: rec.index)
    return out


def ring_skeleton_key(mol: Chem.Mol) -> Optional[str]:
    """Return the RDKit canonical SMILES of the molecule's ring system (RING-03 key).

    Extracts the subgraph induced by ring atoms + ring bonds and canonicalises
    it — a path-independent identifier for the ring skeleton of a (possibly
    substituted) molecule. This is the substructure analogue of the
    whole-molecule canonical-SMILES key used by get_fused_heterocycle_name, and
    the key HYG-01's O(1) fast path looks up. Returns None for acyclic input.
    """
    if mol is None:
        return None
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() == 0:
        return None
    ring_atoms = set()
    for ring in ring_info.AtomRings():
        ring_atoms.update(ring)
    if not ring_atoms:
        return None
    ring_bonds = [
        b.GetIdx() for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in ring_atoms and b.GetEndAtomIdx() in ring_atoms
    ]
    if not ring_bonds:
        return None
    # Use MolFragmentToSmiles (NOT PathToSubmol + MolToSmiles): it writes the
    # fragment SMILES from the parent mol's ALREADY-PERCEIVED aromaticity, so it
    # never re-sanitizes a fresh fragment. Re-sanitizing an extracted ring whose
    # aromaticity only kekulizes in the full-molecule context aborts at the C++
    # level (uncatchable core dump) — this avoids that entirely. The fast path
    # is opportunistic: any non-match (or None) simply falls through to the
    # byte-identical bucket scan, so a divergent key never affects correctness.
    try:
        return Chem.MolFragmentToSmiles(
            mol, atomsToUse=sorted(ring_atoms), bondsToUse=ring_bonds,
            canonical=True,
        )
    except Exception:
        return None


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

    # ---- query descriptors (computed once) ----
    q_rank = mol.GetRingInfo().NumRings()
    if q_rank == 0:
        # Every catalog pattern is a ring system (rank >= 2); the cycle-rank
        # necessary condition means none can match an acyclic query. The old
        # full scan returned None here too — fast exit, byte-identical.
        return None
    q_elem = Counter(a.GetSymbol() for a in mol.GetAtoms())
    q_hetset = frozenset(s for s in q_elem if s not in ('C', 'H'))
    q_heavy = mol.GetNumAtoms()

    candidates = _gather_candidates(q_rank, q_hetset)

    def _prefilter(rec: '_PatternRec') -> bool:
        # Necessary conditions for rec.pattern ⊑ mol (never prunes a real match).
        if rec.num_heavy > q_heavy:
            return False
        for sym, cnt in rec.elem_counts.items():
            if q_elem.get(sym, 0) < cnt:
                return False
        return True

    # ---- O(1) ring-skeleton fast path (RING-03 key) ----
    # If the query's ring skeleton is an exact catalog entry, that entry is the
    # winner UNLESS some entry that is larger — or equal-size but earlier in
    # catalog order — also matches (which would win/tie under the old loop).
    # We confirm against only that small superior set; otherwise fall through.
    # Restricted to all-ring skeleton entries: an exocyclic-bearing entry's
    # standalone canonical SMILES need not equal the bare ring skeleton.
    skel = ring_skeleton_key(mol)
    if skel is not None:
        skel_rec = _REC_BY_SKELETON.get(skel)
        if (skel_rec is not None and not skel_rec.has_exocyclic
                and _prefilter(skel_rec)
                and mol.HasSubstructMatch(skel_rec.pattern)):
            beaten = False
            for rec in candidates:
                if rec is skel_rec:
                    continue
                superior = (rec.parent_atoms > skel_rec.parent_atoms
                            or (rec.parent_atoms == skel_rec.parent_atoms
                                and rec.index < skel_rec.index))
                if superior and _prefilter(rec) and mol.HasSubstructMatch(rec.pattern):
                    beaten = True
                    break
            if not beaten:
                skel_matches = mol.GetSubstructMatches(skel_rec.pattern)
                if skel_matches:
                    return _build_core_result(
                        skel_rec.name, list(skel_matches[0]), skel_rec.smiles)

    # ---- bucket-pruned scan (byte-identical to the old full scan) ----
    best_match: Optional[Tuple[str, List[int], int, str]] = None

    # Find the largest matching core (insertion order; first-at-max wins)
    for rec in candidates:
        if best_match is not None and rec.parent_atoms < best_match[2]:
            # Strictly smaller than the current best can never replace it
            # (strict-`>` update) nor tie it — skip the expensive match.
            continue
        if not _prefilter(rec):
            continue
        if mol.HasSubstructMatch(rec.pattern):
            matches = mol.GetSubstructMatches(rec.pattern)
            if matches:
                match = matches[0]  # Take first match
                core_size = rec.parent_atoms

                # Keep the largest matching core
                if best_match is None or core_size > best_match[2]:
                    best_match = (rec.name, list(match), core_size, rec.smiles)

    if best_match is None:
        return None

    name, match_atoms, _, core_smiles = best_match
    return _build_core_result(name, match_atoms, core_smiles)


def _build_core_result(
    name: str, match_atoms: List[int], core_smiles: str
) -> Optional[Tuple[str, Dict[int, Union[int, str]], str]]:
    """Build the (name, atom_mapping, core_smiles) result from a matched core.

    Shared by both match_fused_heterocycle_core paths; preserves the exact
    pre-computed-IUPAC-locant mapping logic (and the missing-mapping warning)
    of the original implementation.
    """
    data = FUSED_HETEROCYCLE_DATA[core_smiles]

    # Get pre-computed IUPAC locant mapping
    iupac_locants = data.get('iupac_locants')

    if iupac_locants is None:
        # All entries should have iupac_locants -- if this triggers, the entry
        # is incomplete and must be fixed. Return None to avoid incorrect numbering.
        import warnings
        warnings.warn(
            f"Fused heterocycle '{name}' (core: {core_smiles}) missing iupac_locants mapping. "
            f"This entry must be completed in FUSED_HETEROCYCLE_DATA.",
            stacklevel=2,
        )
        return None
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


def _validate_all_entries() -> bool:
    """Verify all FUSED_HETEROCYCLE_DATA entries have complete iupac_locants.

    This is intended to be called from tests, not at import time, to avoid
    import overhead. Returns True if all entries are valid.
    """
    missing = []
    for smiles, data in FUSED_HETEROCYCLE_DATA.items():
        locants = data.get('iupac_locants')
        if locants is None or (isinstance(locants, dict) and len(locants) == 0):
            missing.append(data.get('name', smiles))
    if missing:
        import warnings
        warnings.warn(
            f"FUSED_HETEROCYCLE_DATA entries missing iupac_locants: {missing}",
            stacklevel=2,
        )
    return len(missing) == 0
