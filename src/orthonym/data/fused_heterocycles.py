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

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

from rdkit import Chem

from ..assembly.fragment_naming import (  # M2.5 macrocycle-hang budgets
    spend_analysis_call,
    spend_perf_work,
)
from ..perception.molcache import (  # audit 2026-09-03 (S2): per-call atom/bond tuples
    atoms_of,
    bonds_of,
)

# Fused heterocycle data - canonical SMILES verified with RDKit
# Format: canonical_smiles -> {name, tautomer_locant, ring_system, parent_atoms, iupac_locants}
#
# IUPAC locant mappings (iupac_locants):
# - Maps canonical SMILES atom index -> IUPAC peripheral locant
# - Fusion atoms get string locants like '3a', '7a', '4a', '8a'
# - Peripheral atoms get integer locants 1, 2, 3, etc.
# - Used by match_fused_heterocycle_core() for correct substituent position naming

FUSED_HETEROCYCLE_DATA: Dict[str, Dict[str, Any]] = {
    # 1H-indole
    'c1ccc2[nH]ccc2c1': {
        'name': '1H-indole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 2H-isoindole
    'c1ccc2c[nH]cc2c1': {
        'name': '2H-isoindole',
        'tautomer_locant': 2,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 3H-indole
    'C1=Nc2ccccc2C1': {
        'name': '3H-indole',
        'tautomer_locant': 3,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 2, 1: 1, 2: '7a', 3: 7, 4: 6, 5: 5, 6: 4, 7: '3a', 8: 3},
    },
    # 1H-indazole
    'c1ccc2[nH]ncc2c1': {
        'name': '1H-indazole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 1H-benzimidazole
    'c1ccc2[nH]cnc2c1': {
        'name': '1H-benzimidazole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 1H-benzotriazole
    'c1ccc2[nH]nnc2c1': {
        'name': '1H-benzotriazole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 1-benzofuran
    'c1ccc2occc2c1': {
        'name': '1-benzofuran',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 1-benzothiophene
    'c1ccc2sccc2c1': {
        'name': '1-benzothiophene',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 1,3-benzoxazole
    'c1ccc2ocnc2c1': {
        'name': '1,3-benzoxazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 2,1-benzisoxazole
    'c1ccc2nocc2c1': {
        'name': '2,1-benzisoxazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 1,3-benzothiazole
    'c1ccc2scnc2c1': {
        'name': '1,3-benzothiazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 2,1-benzothiazole
    'c1ccc2nscc2c1': {
        'name': '2,1-benzothiazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # [1,3,2]benzodioxathiole (spiro-component form; the ring S is the spiro
    # atom, cited with a λ token in the spiro PIN — P-24.8.2). Canonical key uses
    # the λ4 [SH2] valence as it appears in the extracted spiro component.
    # OPSIN numbering: O1, S2, O3, C3a, C4-C7, C7a (verified via -o extendedsmi).
    'c1ccc2c(c1)O[SH2]O2': {
        'name': '[1,3,2]benzodioxathiole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {6: 1, 7: 2, 8: 3, 3: '3a', 2: 4, 1: 5, 0: 6, 5: 7, 4: '7a'},
    },
    # quinoline
    'c1ccc2ncccc2c1': {
        'name': 'quinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },
    # isoquinoline
    'c1ccc2cnccc2c1': {
        'name': 'isoquinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },
    # quinazoline
    'c1ccc2ncncc2c1': {
        'name': 'quinazoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },
    # quinoxaline
    'c1ccc2nccnc2c1': {
        'name': 'quinoxaline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },
    # phthalazine
    'c1ccc2cnncc2c1': {
        'name': 'phthalazine',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },
    # cinnoline
    'c1ccc2nnccc2c1': {
        'name': 'cinnoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },
    # 1,6-naphthyridine
    'c1cnc2ccncc2c1': {
        'name': '1,6-naphthyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '8a', 4: 8, 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4},
    },
    # 2,7-naphthyridine
    'c1cc2ccncc2cn1': {
        'name': '2,7-naphthyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 4, 2: '4a', 3: 5, 4: 6, 5: 7, 6: 8, 7: '8a', 8: 1, 9: 2},
    },
    # pyrido[2,3-b]pyrazine
    'c1cnc2nccnc2c1': {
        'name': 'pyrido[2,3-b]pyrazine',
        'tautomer_locant': None,
        'ring_system': 'pyridopyrazine',
        'parent_atoms': 10,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: 1, 8: '8a', 9: 8},
    },
    # 1,7-naphthyridine
    'c1cnc2cnccc2c1': {
        'name': '1,7-naphthyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '8a', 4: 8, 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4},
    },
    # 9H-carbazole
    'c1ccc2c(c1)[nH]c1ccccc12': {
        'name': '9H-carbazole',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {0: 2, 1: 3, 2: 4, 3: '4a', 4: '9a', 5: 1, 6: 9, 7: '8a', 8: 8, 9: 7, 10: 6, 11: 5, 12: '4b'},
    },
    # acridine
    'c1ccc2nc3ccccc3cc2c1': {
        'name': 'acridine',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 2, 1: 3, 2: 4, 3: '4a', 4: 10, 5: '10a', 6: 5, 7: 6, 8: 7, 9: 8, 10: '8a', 11: 9, 12: '9a', 13: 1},
    },
    # phenazine
    'c1ccc2nc3ccccc3nc2c1': {
        'name': 'phenazine',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '10a', 4: 10, 5: '9a', 6: 9, 7: 8, 8: 7, 9: 6, 10: '5a', 11: 5, 12: '4a', 13: 4},
    },
    # 10H-phenoxazine
    'c1ccc2c(c1)Nc1ccccc1O2': {
        'name': '10H-phenoxazine',
        'tautomer_locant': 10,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 2, 1: 3, 2: 4, 3: '4a', 4: '10a', 5: 1, 6: 10, 7: '9a', 8: 9, 9: 8, 10: 7, 11: 6, 12: '5a', 13: 5},
    },
    # 10H-phenothiazine
    'c1ccc2c(c1)Nc1ccccc1S2': {
        'name': '10H-phenothiazine',
        'tautomer_locant': 10,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 2, 1: 3, 2: 4, 3: '4a', 4: '10a', 5: 1, 6: 10, 7: '9a', 8: 9, 9: 8, 10: 7, 11: 6, 12: '5a', 13: 5},
    },
    # 9H-xanthene
    'c1ccc2c(c1)Cc1ccccc1O2': {
        'name': '9H-xanthene',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 2, 1: 3, 2: 4, 3: '4a', 4: '9a', 5: 1, 6: 9, 7: '8a', 8: 8, 9: 7, 10: 6, 11: 5, 12: '10a', 13: 10},
    },
    # 6H-benzo[c]chromene -> PIN 6H-dibenzo[b,d]pyran (the Blue Book '6H-dibenzo[b,d]pyran (PIN)
    #... not... 6H-benzo[c]chromene'). Same ring system + numbering (OPSIN extendedsmi verified) -> rename only.
    'c1ccc2c(c1)COc1ccccc1-2': {
        'name': '6H-dibenzo[b,d]pyran',
        'tautomer_locant': 6,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 8, 1: 9, 2: 10, 3: '10a', 4: '6a', 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4, 10: 3, 11: 2, 12: 1, 13: '10b'},
    },
    # thianthrene
    'c1ccc2c(c1)Sc1ccccc1S2': {
        'name': 'thianthrene',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '10a', 4: '4a', 5: 4, 6: 10, 7: '5a', 8: 6, 9: 7, 10: 8, 11: 9, 12: '9a', 13: 5},
    },
    # indolizine
    'c1ccn2cccc2c1': {
        'name': 'indolizine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: '8a', 8: 8},
    },
    # imidazo[1,2-a]pyridine
    'c1ccn2ccnc2c1': {
        'name': 'imidazo[1,2-a]pyridine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: '8a', 8: 8},
    },
    # imidazo[1,5-a]pyridine
    'c1ccn2cncc2c1': {
        'name': 'imidazo[1,5-a]pyridine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: '8a', 8: 8},
    },
    # 3H-imidazo[4,5-b]pyridine
    'c1cnc2[nH]cnc2c1': {
        'name': '3H-imidazo[4,5-b]pyridine',
        'tautomer_locant': 3,
        'ring_system': 'imidazopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: 3, 5: 2, 6: 1, 7: '7a', 8: 7},
    },
    # 3H-pyrrolo[3,2-d]pyrimidine
    'c1cc2nc[nH]cc-2n1': {
        'name': '3H-pyrrolo[3,2-d]pyrimidine',
        'tautomer_locant': 3,
        'ring_system': 'imidazopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: '7a', 3: 1, 4: 2, 5: 3, 6: 4, 7: '4a', 8: 5},
    },
    # 1H-pyrrolo[2,3-b]pyridine
    'c1cnc2[nH]ccc2c1': {
        'name': '1H-pyrrolo[2,3-b]pyridine',
        'tautomer_locant': 1,
        'ring_system': 'azaindole',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 1H-pyrrolo[2,3-c]pyridine
    'c1cc2cc[nH]c2cn1': {
        'name': '1H-pyrrolo[2,3-c]pyridine',
        'tautomer_locant': 1,
        'ring_system': 'azaindole',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 4, 2: '3a', 3: 3, 4: 2, 5: 1, 6: '7a', 7: 7, 8: 6},
    },
    # coumarin -> PIN 2H-1-benzopyran-2-one (P-19(d) line 1736: '1-benzopyran' is
    # the PIN ring parent, not 'chromene'; '2H-1-benzopyran-2-one' OPSIN-RT-verified)
    'O=c1ccc2ccccc2o1': {
        'name': '2H-1-benzopyran-2-one',
        'tautomer_locant': 2,
        'ring_system': 'benzo-6-membered-lactone',
        'parent_atoms': 11,
        'iupac_locants': {0: '=O', 1: 2, 2: 3, 3: 4, 4: '4a', 5: 5, 6: 6, 7: 7, 8: 8, 9: '8a', 10: 1},
        'systematic': '2H-1-benzopyran-2-one',
    },
    # 2,3-dihydro-1-benzofuran
    'c1ccc2c(c1)CCO2': {
        'name': '2,3-dihydro-1-benzofuran',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered-saturated',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: '3a', 5: 4, 6: 3, 7: 2, 8: 1},
        'systematic': '2,3-dihydro-1-benzofuran',
    },
    # 2H-chromene -> PIN 2H-1-benzopyran (P-19(d) line 1736; line 24647 '2H-1-benzopyran (PIN)';
    # line 19453 '2H-1-benzopyran (PIN) 2H-chromene'). Same ring system => iupac_locants unchanged.
    'C1=Cc2ccccc2OC1': {
        'name': '2H-1-benzopyran',
        'tautomer_locant': 2,
        'ring_system': 'benzopyran',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 4, 2: '4a', 3: 5, 4: 6, 5: 7, 6: 8, 7: '8a', 8: 1, 9: 2},
    },
    # 4H-chromene -> PIN 4H-1-benzopyran
    'C1=COc2ccccc2C1': {
        'name': '4H-1-benzopyran',
        'tautomer_locant': 4,
        'ring_system': 'benzopyran',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '8a', 4: 8, 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4},
    },
    # 9H-purine
    'c1ncc2nc[nH]c2n1': {
        'name': '9H-purine',
        'tautomer_locant': 9,
        'ring_system': 'purine',
        'parent_atoms': 9,
        'iupac_locants': {0: 2, 1: 1, 2: 6, 3: 5, 4: 7, 5: 8, 6: 9, 7: 4, 8: 3},
    },
    # pteridine
    'c1cnc2ncncc2n1': {
        'name': 'pteridine',
        'tautomer_locant': None,
        'ring_system': 'pteridine',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },
    # adenine
    'Nc1ncnc2nc[nH]c12': {
        'name': 'adenine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 10,
        'iupac_locants': {0: 'N6', 1: 6, 2: 1, 3: 2, 4: 3, 5: 4, 6: 7, 7: 8, 8: 9, 9: 5},
        'is_retained_name': True,
    },
    # adenine
    'Nc1ncnc2[nH]cnc12': {
        'name': 'adenine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 10,
        'iupac_locants': {0: 'N6', 1: 6, 2: 1, 3: 2, 4: 3, 5: 9, 6: 8, 7: 7, 8: 5, 9: 4},
        'is_retained_name': True,
    },
    # hypoxanthine
    'O=c1[nH]cnc2nc[nH]c12': {
        'name': 'hypoxanthine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 10,
        'iupac_locants': {0: 'O6', 1: 6, 2: 1, 3: 2, 4: 3, 5: 4, 6: 7, 7: 8, 8: 9, 9: 5},
        'is_retained_name': True,
    },
    # hypoxanthine
    'O=c1[nH]cnc2[nH]cnc12': {
        'name': 'hypoxanthine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 10,
        'iupac_locants': {0: 'O6', 1: 6, 2: 1, 3: 2, 4: 3, 5: 9, 6: 8, 7: 7, 8: 5, 9: 4},
        'is_retained_name': True,
    },
    # guanine
    'Nc1nc2[nH]cnc2c(=O)[nH]1': {
        'name': 'guanine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 11,
        'iupac_locants': {0: 'N2', 1: 2, 2: 3, 3: 9, 4: 8, 5: 7, 6: 5, 7: 6, 8: 'O6', 9: 1, 10: 4},
        'is_retained_name': True,
    },
    # guanine
    'Nc1nc(=O)c2[nH]cnc2[nH]1': {
        'name': 'guanine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 11,
        'iupac_locants': {0: 'N2', 1: 2, 2: 6, 3: 'O6', 4: 5, 5: 9, 6: 8, 7: 7, 8: 4, 9: 3, 10: 1},
        'is_retained_name': True,
    },
    # pyrazolo[1,5-a]pyrimidine
    'c1cnc2ccnn2c1': {
        'name': 'pyrazolo[1,5-a]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: 3, 5: 2, 6: 1, 7: 8, 8: 7},
    },
    # thieno[2,3-b]pyridine
    'c1cnc2sccc2c1': {
        'name': 'thieno[2,3-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'thienopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # ------------------------------------------------------------------
    # a phase (B3b) -- two-ring hetero/hetero ortho-fused PINs the
    # systematic fusion builder cannot yet spell (it lacks the 1,3-dioxole /
    # 1,2-oxazine / 1,3-oxathiole / 1,2,4-triazine / 1,2,3-oxathiazole
    # components AND never computes fusion-path indicated hydrogen). Each
    # `name` is a verbatim Blue-Book (PIN) example; each `iupac_locants` map
    # is OPSIN-authoritative (derived via opsin_atom_locant_map, all round-trip
    # inchi_match=True). Substituted forms of the 3-heteroatom / indicated-H
    # systems (triazine, oxathiazolo, thieno-imidazole) remain a named residual
    # for the general fusion engine -- the bare parents below are exact-match
    # only.
    # 2H-furo[2,3-d][1,3]dioxole -- P-25.3.2.4 (the Blue Book '2H-furo[2,3-d][1,3]dioxole (PIN)'
    # [dioxole (2 heteroatoms) preferred to furan (1 heteroatom)]).
    'c1cc2c(o1)OCO2': {
        'name': '2H-furo[2,3-d][1,3]dioxole',
        'tautomer_locant': 2,
        'ring_system': 'furo-dioxole',
        'parent_atoms': 8,
        'iupac_locants': {0: 5, 1: 6, 2: '6a', 3: '3a', 4: 4, 5: 3, 6: 2, 7: 1},
    },
    # 5H-pyrido[2,3-d][1,2]oxazine -- P-25.3.3.1.2 (the Blue Book
    # '5H-pyrido[2,3-d][1,2]oxazine (PIN)' [oxazine (2 het) preferred to pyridine]).
    'C1=NOCc2cccnc21': {
        'name': '5H-pyrido[2,3-d][1,2]oxazine',
        'tautomer_locant': 5,
        'ring_system': 'pyrido-oxazine',
        'parent_atoms': 10,
        'iupac_locants': {0: 8, 1: 7, 2: 6, 3: 5, 4: '4a', 5: 4, 6: 3, 7: 2, 8: 1, 9: '8a'},
    },
    # 2H,4H-[1,3]oxathiolo[5,4-b]pyrrole -- verbatim (PIN) at the Blue Book. Two
    # indicated H (CH2 at 2, NH at 4); tautomer_locant stores the lower (2) --
    # the baked `name` carries the full '2H,4H' string on the bare-parent path.
    'c1cc2c([nH]1)OCS2': {
        'name': '2H,4H-[1,3]oxathiolo[5,4-b]pyrrole',
        'tautomer_locant': 2,
        'ring_system': 'oxathiolo-pyrrole',
        'parent_atoms': 8,
        'iupac_locants': {0: 5, 1: 6, 2: '6a', 3: '3a', 4: 4, 5: 3, 6: 2, 7: 1},
    },
    # imidazo[1,2-b][1,2,4]triazine -- verbatim (PIN) at the Blue Book (P-25.3.3.1.2(c),
    # the Blue Book "the locant '4a' is lower than '8a'"). Fully mancude, no indicated H.
    'c1cnn2ccnc2n1': {
        'name': 'imidazo[1,2-b][1,2,4]triazine',
        'tautomer_locant': None,
        'ring_system': 'imidazo-triazine',
        'parent_atoms': 9,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: 8, 4: 7, 5: 6, 6: 5, 7: '4a', 8: 4},
    },
    # 3H,5H-[1,3,2]oxathiazolo[4,5-d][1,2,3]oxathiazole -- verbatim (PIN) at
    # the Blue Book ("locants '1,2,3' are lower than '1,3,2'"). Two indicated H
    # (NH at 3 and 5); tautomer_locant stores the lower (3).
    '[nH]1oc2os[nH]c=2s1': {
        'name': '3H,5H-[1,3,2]oxathiazolo[4,5-d][1,2,3]oxathiazole',
        'tautomer_locant': 3,
        'ring_system': 'oxathiazolo-oxathiazole',
        'parent_atoms': 8,
        'iupac_locants': {0: 5, 1: 6, 2: '6a', 3: 1, 4: 2, 5: 3, 6: '3a', 7: 4},
    },
    # [1,3]selenazolo[5,4-d][1,3]thiazole -- verbatim (PIN) at the Blue Book
    # (P-25.3.2.4(f); S,N senior to Se,N so [1,3]thiazole is the base). No iH.
    'c1nc2[se]cnc2s1': {
        'name': '[1,3]selenazolo[5,4-d][1,3]thiazole',
        'tautomer_locant': None,
        'ring_system': 'selenazolo-thiazole',
        'parent_atoms': 8,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: '6a', 7: 1},
    },
    # 1H-thieno[2,3-d]imidazole -- verbatim (PIN) at the Blue Book (P-25.3.3.1.2).
    # NH indicated H at 1. Replaces the systematic path's wrong '[3,2-d]'/no-iH spelling.
    'c1nc2sccc2[nH]1': {
        'name': '1H-thieno[2,3-d]imidazole',
        'tautomer_locant': 1,
        'ring_system': 'thieno-imidazole',
        'parent_atoms': 8,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: '6a', 7: 1},
    },
    # ------------------------------------------------------------------
    # a phase (B2 / CW-2 remainder) -- two-ring fused parents (4 hetero,
    # 1 pure carbocycle) the systematic fusion / von-Baeyer path degrades: it
    # lacks these small-ring components / fusion-path indicated hydrogen and so
    # emits a von-Baeyer name (e.g. '2,7-dioxabicyclo[4.3.0]nona-...') instead of
    # the fusion PIN. Each `name` is a verbatim Blue-Book (PIN); each
    # `iupac_locants` map is OPSIN-authoritative (opsin_atom_locant_map, all
    # round-trip inchi_match=True). Exact whole-molecule canonical-SMILES keys, so
    # substituted forms (different canonical SMILES) never over-match the bare
    # parents. Mirrors 6f1fe6333 (a phase B3b). Row 4 (cyclopenta[8]annulene) is a
    # pure carbocycle and confirmed to route through this same lookup (indene /
    # pyrene positives).
    # 2H-furo[3,2-b]pyran -- P-25.3.2.4 (the Blue Book '2H-furo[3,2-b]pyran (PIN)
    # [pyran (6 ring) preferred to furan (5 ring)]'). iH at 2.
    'C1=COC2=CCOC2=C1': {
        'name': '2H-furo[3,2-b]pyran',
        'tautomer_locant': 2,
        'ring_system': 'furo-pyran',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: 3, 5: 2, 6: 1, 7: '7a', 8: 7},
    },
    # 2H-1,3-benzoxathiole -- verbatim (PIN) at the Blue Book (P-25.7.1.1). iH at 2.
    'c1ccc2c(c1)OCS2': {
        'name': '2H-1,3-benzoxathiole',
        'tautomer_locant': 2,
        'ring_system': 'benzoxathiole',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: '7a', 5: 7, 6: 1, 7: 2, 8: 3},
    },
    # pyrrolo[3,2-b]pyrrole -- verbatim (PIN) at the Blue Book (P-25.7.1.1). Fully
    # mancude (both N are pyridine-type =N-, no NH; pentalene analog) -> no
    # indicated hydrogen, so gold correctly carries none.
    'C1=CC2=NC=CC2=N1': {
        'name': 'pyrrolo[3,2-b]pyrrole',
        'tautomer_locant': None,
        'ring_system': 'pyrrolo-pyrrole',
        'parent_atoms': 8,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: '6a', 7: 1},
    },
    # 1H-cyclopenta[8]annulene -- verbatim (PIN) at the Blue Book (P-25.3.8.1). Pure
    # carbocycle; iH at 1. Confirmed to reach this lookup via the carbocyclic
    # route (indene / pyrene are named through the same table).
    'C1=CC=CC2=C(C=C1)C=CC2': {
        'name': '1H-cyclopenta[8]annulene',
        'tautomer_locant': 1,
        'ring_system': 'cyclopenta-annulene',
        'parent_atoms': 11,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: 9, 4: '9a', 5: '3a', 6: 4, 7: 5, 8: 3, 9: 2, 10: 1},
    },
    # 1H,3H-thieno[3,4-c]thiophene -- verbatim (PIN) at the Blue Book (P-25.7.1.3.2).
    # Two indicated H (CH2 at 1 and 3); tautomer_locant stores the lower (1).
    # (the Blue Book lists a separate 2lambda4,5lambda4 form -- not this molecule.)
    'c1scc2c1CSC2': {
        'name': '1H,3H-thieno[3,4-c]thiophene',
        'tautomer_locant': 1,
        'ring_system': 'thieno-thiophene',
        'parent_atoms': 8,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: '6a', 5: 1, 6: 2, 7: 3},
    },
    # ------------------------------------------------------------------
    # indoline
    'c1ccc2c(c1)CCN2': {
        # The name below IS the PIN. P-54.4.3.2 (the Blue Book) names the retained form
        # verbatim as non-preferred -- "The retained names for the partially saturated
        # heterocycles, indane, indoline, isoindoline, and chromane, isochromane and
        # their chalcogen analogues are not used as preferred IUPAC names" -- and
        # the Blue Book print the three PINs (2,3-dihydro-1H-indene /
        # -1H-indole / -1H-isoindole).
        #
        # ⚠ HISTORY, so this is not re-broken: the rename was attempted once and the
        # gate correctly rejected it (1 protect + 3 target regressions). This `name`
        # field feeds TWO consumers. Standalone naming wants the saturated PIN, but
        # `rules/spiro.py:_name_spirobi_core` embedded it as the SPIRO COMPONENT, and
        # P-24.3.1 (the Blue Book) requires the bracket to hold the MANCUDE component ring
        # system with hydrogen cited OUTSIDE it -- so the bare rename produced
        # `1,2'-spirobi[2,3-dihydro-1H-indene]`. The obvious shortcut is WORSE, not
        # merely wrong: `1,2'-spirobi[1H-indene]` denotes the UNSATURATED molecule --
        # a wrong STRUCTURE, not a wrong spelling (session a project rule).
        #
        # ✅ UNBLOCKED (Phase C). `_name_spirobi_core` no longer reads saturation
        # out of this string at all: it derives the hydro prefixes and any indicated
        # hydrogen from the GRAPH (maximum noncumulative double-bond assignment over
        # the assembled skeleton, the spiro atom excluded) and hoists them in front of
        # the spiro locants, per P-24.3.2 (the Blue Book) and the the Blue Book template
        # `1,3'-dihydro-3H-1lambda6,1'-spirobi[[2,1]benzoxathiole]`. It now emits
        # `1',2,3,3'-tetrahydro-1,2'-spirobi[indene]`. The spiro consumer is therefore
        # INDEPENDENT of this field, and renaming here is safe.
        'name': '2,3-dihydro-1H-indole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-saturated',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: '3a', 5: 4, 6: 3, 7: 2, 8: 1},
    },
    # isoindoline
    'c1ccc2c(c1)CNC2': {
        # The name below IS the PIN. P-54.4.3.2 (the Blue Book) names the retained form
        # verbatim as non-preferred -- "The retained names for the partially saturated
        # heterocycles, indane, indoline, isoindoline, and chromane, isochromane and
        # their chalcogen analogues are not used as preferred IUPAC names" -- and
        # the Blue Book print the three PINs (2,3-dihydro-1H-indene /
        # -1H-indole / -1H-isoindole).
        #
        # ⚠ HISTORY, so this is not re-broken: the rename was attempted once and the
        # gate correctly rejected it (1 protect + 3 target regressions). This `name`
        # field feeds TWO consumers. Standalone naming wants the saturated PIN, but
        # `rules/spiro.py:_name_spirobi_core` embedded it as the SPIRO COMPONENT, and
        # P-24.3.1 (the Blue Book) requires the bracket to hold the MANCUDE component ring
        # system with hydrogen cited OUTSIDE it -- so the bare rename produced
        # `1,2'-spirobi[2,3-dihydro-1H-indene]`. The obvious shortcut is WORSE, not
        # merely wrong: `1,2'-spirobi[1H-indene]` denotes the UNSATURATED molecule --
        # a wrong STRUCTURE, not a wrong spelling (session a project rule).
        #
        # ✅ UNBLOCKED (Phase C). `_name_spirobi_core` no longer reads saturation
        # out of this string at all: it derives the hydro prefixes and any indicated
        # hydrogen from the GRAPH (maximum noncumulative double-bond assignment over
        # the assembled skeleton, the spiro atom excluded) and hoists them in front of
        # the spiro locants, per P-24.3.2 (the Blue Book) and the the Blue Book template
        # `1,3'-dihydro-3H-1lambda6,1'-spirobi[[2,1]benzoxathiole]`. It now emits
        # `1',2,3,3'-tetrahydro-1,2'-spirobi[indene]`. The spiro consumer is therefore
        # INDEPENDENT of this field, and renaming here is safe.
        'name': '2,3-dihydro-1H-isoindole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-saturated',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: '3a', 5: 4, 6: 1, 7: 2, 8: 3},
    },
    # 1,2,3,4-tetrahydroquinoline
    'c1ccc2c(c1)CCCN2': {
        'name': '1,2,3,4-tetrahydroquinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # 1,2,3,4-tetrahydroisoquinoline
    'c1ccc2c(c1)CCNC2': {
        'name': '1,2,3,4-tetrahydroisoquinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # chromane -> PIN 3,4-dihydro-2H-1-benzopyran (the Blue Book). The IH-01h cyclic-oxo
    # engine now names chromanone derivatives correctly (carbonyl-at-IH mancude parent,
    # P-64.2.2.2.2), so the rename is safe.
    'c1ccc2c(c1)CCCO2': {
        'name': '3,4-dihydro-2H-1-benzopyran',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # isochromane -> PIN 3,4-dihydro-1H-2-benzopyran (the Blue Book)
    'c1ccc2c(c1)CCOC2': {
        'name': '3,4-dihydro-1H-2-benzopyran',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # isothiochromane -> PIN 3,4-dihydro-1H-2-benzothiopyran (BB Table 3.1, line 17018;
    # retained name isothiochromane is NOT a PIN, P-31.2.3.3.1 line 16509 / P-54.4.3.2 line 24256)
    'c1ccc2c(c1)CCSC2': {
        'name': '3,4-dihydro-1H-2-benzothiopyran',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # isoselenochromane -> PIN 3,4-dihydro-1H-2-benzoselenopyran (BB Table 3.1, line 17020)
    'c1ccc2c(c1)CC[Se]C2': {
        'name': '3,4-dihydro-1H-2-benzoselenopyran',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # isotellurochromane -> PIN 3,4-dihydro-1H-2-benzotelluropyran (BB Table 3.1, line 17022)
    'c1ccc2c(c1)CC[Te]C2': {
        'name': '3,4-dihydro-1H-2-benzotelluropyran',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # isochromene -> PIN 1H-2-benzopyran (P-19(d); the Blue Book 'the PIN is 1H-2-benzopyran').
    # New catalog entry (was named only via the OPSIN-import alias); locants from OPSIN extendedsmi (O at 2).
    'C1=Cc2ccccc2CO1': {
        'name': '1H-2-benzopyran',
        'tautomer_locant': 1,
        'ring_system': 'benzopyran',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 4, 2: '4a', 3: 5, 4: 6, 5: 7, 6: 8, 7: '8a', 8: 1, 9: 2},
    },
    # 2,1,3-benzothiadiazole
    'c1ccc2nsnc2c1': {
        'name': '2,1,3-benzothiadiazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 2,1,3-benzoxadiazole
    'c1ccc2nonc2c1': {
        'name': '2,1,3-benzoxadiazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # naphtho[2,1-b]furan
    'c1ccc2c(c1)ccc1occc12': {
        'name': 'naphtho[2,1-b]furan',
        'tautomer_locant': None,
        'ring_system': 'naphtho-fused',
        'parent_atoms': 13,
        'iupac_locants': {0: 7, 1: 8, 2: 9, 3: '9a', 4: '5a', 5: 6, 6: 5, 7: 4, 8: '3a', 9: 3, 10: 2, 11: 1, 12: '9b'},
    },
    # naphtho[2,1-b]thiophene
    'c1ccc2c(c1)ccc1sccc12': {
        'name': 'naphtho[2,1-b]thiophene',
        'tautomer_locant': None,
        'ring_system': 'naphtho-fused',
        'parent_atoms': 13,
        'iupac_locants': {0: 7, 1: 8, 2: 9, 3: '9a', 4: '5a', 5: 6, 6: 5, 7: 4, 8: '3a', 9: 3, 10: 2, 11: 1, 12: '9b'},
    },
    # 3H-benzo[e]indole
    'c1ccc2c(c1)ccc1[nH]ccc12': {
        'name': '3H-benzo[e]indole',
        'tautomer_locant': 3,
        'ring_system': 'naphtho-fused',
        'parent_atoms': 13,
        'iupac_locants': {0: 7, 1: 8, 2: 9, 3: '9a', 4: '5a', 5: 6, 6: 5, 7: 4, 8: '3a', 9: 3, 10: 2, 11: 1, 12: '9b'},
    },
    # naphtho[2,3-b]furan
    'c1ccc2cc3occc3cc2c1': {
        'name': 'naphtho[2,3-b]furan',
        'tautomer_locant': None,
        'ring_system': 'naphtho-fused',
        'parent_atoms': 13,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 9, 5: '9a', 6: 1, 7: 2, 8: 3, 9: '3a', 10: 4, 11: '4a', 12: 5},
    },
    # naphtho[2,3-b]thiophene
    'c1ccc2cc3sccc3cc2c1': {
        'name': 'naphtho[2,3-b]thiophene',
        'tautomer_locant': None,
        'ring_system': 'naphtho-fused',
        'parent_atoms': 13,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 9, 5: '9a', 6: 1, 7: 2, 8: 3, 9: '3a', 10: 4, 11: '4a', 12: 5},
    },
    # 1H-naphtho[2,3-b]pyrrole
    'c1ccc2cc3[nH]ccc3cc2c1': {
        'name': '1H-naphtho[2,3-b]pyrrole',
        'tautomer_locant': 1,
        'ring_system': 'naphtho-fused',
        'parent_atoms': 13,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 9, 5: '9a', 6: 1, 7: 2, 8: 3, 9: '3a', 10: 4, 11: '4a', 12: 5},
    },
    # pyrido[2,3-d]pyrimidine
    'c1cnc2ncncc2c1': {
        'name': 'pyrido[2,3-d]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'pyridopyrimidine',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: '4a', 9: 5},
    },
    # 1,5-naphthyridine
    'c1cnc2cccnc2c1': {
        'name': '1,5-naphthyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '8a', 4: 8, 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4},
    },
    # pyrido[3,2-c]pyridazine
    'c1cnc2ccnnc2c1': {
        'name': 'pyrido[3,2-c]pyridazine',
        'tautomer_locant': None,
        'ring_system': 'pyridopyridazine',
        'parent_atoms': 10,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: 1, 8: '8a', 9: 8},
    },
    # imidazo[1,2-a]pyrimidine
    'c1cnc2nccn2c1': {
        'name': 'imidazo[1,2-a]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: 5},
    },
    # 1H-imidazo[4,5-d]pyrimidine
    'c1ncc2[nH]cnc2n1': {
        'name': '1H-imidazo[4,5-d]pyrimidine',
        'tautomer_locant': 1,
        'ring_system': 'purine-related',
        'parent_atoms': 9,
        'iupac_locants': {0: 2, 1: 1, 2: 6, 3: 5, 4: 7, 5: 8, 6: 9, 7: 4, 8: 3},
    },
    # thieno[3,2-b]pyridine
    'c1cnc2ccsc2c1': {
        'name': 'thieno[3,2-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'thienopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: 3, 5: 2, 6: 1, 7: '7a', 8: 7},
    },
    # thiazolo[4,5-b]pyridine
    'c1cnc2ncsc2c1': {
        'name': 'thiazolo[4,5-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'thienopyrimidine',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: 3, 5: 2, 6: 1, 7: '7a', 8: 7},
    },
    # 1H-pyrrolo[3,2-b]pyridine
    'c1cnc2cc[nH]c2c1': {
        'name': '1H-pyrrolo[3,2-b]pyridine',
        'tautomer_locant': 1,
        'ring_system': 'azaindole',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: 3, 5: 2, 6: 1, 7: '7a', 8: 7},
    },
    # furo[3,2-b]pyridine
    'c1cnc2ccoc2c1': {
        'name': 'furo[3,2-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'furopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: 3, 5: 2, 6: 1, 7: '7a', 8: 7},
    },
    # P-25.3.5.3 multiparent difuran (a multiparent name is preferred to a
    # two-component fused name furo[3,4-f][1]benzofuran). Numbering derived
    # 2026-07-09 from OPSIN `benzo[1,2-b:4,5-c']difuran -o extendedsmi`.
    'c1cc2cc3cocc3cc2o1': {
        'name': "benzo[1,2-b:4,5-c']difuran",
        'tautomer_locant': None,
        'ring_system': 'benzodifuran',
        'parent_atoms': 12,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 4, 4: '4a', 5: 5, 6: 6, 7: 7, 8: '7a', 9: 8, 10: '8a', 11: 1},
    },
    # P-25.3.7.1 multiparent difuran (b' isomer): one interparent benzene
    # component; primed letters + colon-separated locant sets. DISTINCT canonical
    # key from the c' isomer above. Numbering derived 2026-07-09 from OPSIN
    # `benzo[1,2-b:4,5-b']difuran -o extendedsmi`.
    'c1cc2cc3occc3cc2o1': {
        'name': "benzo[1,2-b:4,5-b']difuran",
        'tautomer_locant': None,
        'ring_system': 'benzodifuran',
        'parent_atoms': 12,
        'iupac_locants': {0: 6, 1: 7, 2: '7a', 3: 8, 4: '8a', 5: 1, 6: 2, 7: 3, 8: '3a', 9: 4, 10: '4a', 11: 5},
    },
    # P-25.3.7.3 three+-interparent multiparent (BB-verbatim PIN,
    # the Blue Book): double-primed benzo interparent, dicyclobuta
    # first-order interparent, difuran parents. Numbering derived 2026-07-09
    # from OPSIN `<PIN> -o extendedsmi`.
    'c1cc2c3cc4c5cocc5c4cc3c2o1': {
        'name': "benzo[1'',2'':3,4;4'',5'':3',4']dicyclobuta[1,2-b:1',2'-c']difuran",
        'tautomer_locant': None,
        'ring_system': 'benzodicyclobutadifuran',
        'parent_atoms': 16,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: '3b', 4: 4, 5: '4a', 6: '4b', 7: 5, 8: 6, 9: 7, 10: '7a', 11: '7b', 12: 8, 13: '8a', 14: '8b', 15: 1},
    },
    # P-25.5.1.2 three-component: skeletal-replacement 'a' heteroatoms +
    # methano bridge on a cyclopenta[cd]azulene residual (the bare residual is
    # not yet nameable by the fusion engine; catalog the closed structure).
    # p5_bridged DEFERRED this to the fused catalog (xfail). Numbering derived
    # 2026-07-09 from OPSIN `2,3,9-trioxa-5,8-methanocyclopenta[cd]azulene
    # -o extendedsmi`.
    'c1cc2oc1-c1coc3occ-2c13': {
        'name': "2,3,9-trioxa-5,8-methanocyclopenta[cd]azulene",
        'tautomer_locant': None,
        'ring_system': 'trioxamethanocyclopentaazulene',
        'parent_atoms': 13,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: 9, 4: 8, 5: '8a', 6: 1, 7: 2, 8: '2a', 9: 3, 10: 4, 11: '4a', 12: '8b'},
    },
    # P-25.4.2.3.1 composite epoxymethano bridge on a fused-heterocycle parent.
    # The bare residual furo[3,4-b]pyran is not deterministically nameable by the
    # algorithmic fusion engine (it emits a wrong '[4,3-b]' descriptor, SELF-01-
    # suppressed), so the closed bridged structure is cataloged verbatim (same
    # pattern as the 2,3,9-trioxa-5,8-methanocyclopenta[cd]azulene entry above).
    # p5_bridged DEFERRED this to the fused catalog (xfail). Numbering derived
    # 2026-07-11 from OPSIN `2H-3,5-(epoxymethano)furo[3,4-b]pyran -o extendedsmi`.
    'C1=C2COc3coc(c31)CO2': {
        'name': "2H-3,5-(epoxymethano)furo[3,4-b]pyran",
        # The name carries an indicated hydrogen, so the metadata must record it;
        # it read None, which is the one entry in this table where the two
        # disagree. Metadata only -- the sole consumer (rules/fused_rings.py:1290)
        # discards the value with "Name already includes tautomer locant if
        # present", so no emitted name can change.
        'tautomer_locant': 2,
        'ring_system': 'epoxymethanofuropyran',
        'parent_atoms': 11,
        'iupac_locants': {0: 4, 1: 3, 2: 2, 3: 1, 4: '7a', 5: 7, 6: 6, 7: 5, 8: '4a', 9: 8, 10: 9},
    },
    # furo[2,3-b]pyridine
    'c1cnc2occc2c1': {
        'name': 'furo[2,3-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'furopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # oxazolo[5,4-b]pyridine
    'c1cnc2ocnc2c1': {
        'name': 'oxazolo[5,4-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'oxazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: 3, 5: 2, 6: 1, 7: '7a', 8: 7},
    },
    # [1,2,4]triazolo[4,3-a]pyrimidine
    'c1cnc2nncn2c1': {
        'name': '[1,2,4]triazolo[4,3-a]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: 1, 5: 2, 6: 3, 7: 4, 8: 5},
    },
    # phenanthridine
    'c1ccc2c(c1)cnc1ccccc12': {
        'name': 'phenanthridine',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 8, 1: 9, 2: 10, 3: '8a', 4: '4b', 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4, 10: 3, 11: 2, 12: 1, 13: '10a'},
    },
    # 9H-beta-carboline
    'c1ccc2c(c1)[nH]c1cnccc12': {
        'name': '9H-beta-carboline',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4b', 4: '8a', 5: 8, 6: 9, 7: '9a', 8: 1, 9: 2, 10: 3, 11: 4, 12: '4a'},
    },
    # acridone
    'O=c1c2ccccc2[nH]c2ccccc12': {
        'name': 'acridone',
        'tautomer_locant': 10,
        'ring_system': 'tricyclic',
        'parent_atoms': 15,
        'iupac_locants': {0: '=O', 1: 9, 2: '4a', 3: 4, 4: 3, 5: 2, 6: 1, 7: '9a', 8: 10, 9: '8a', 10: 8, 11: 7, 12: 6, 13: 5, 14: '10a'},
        'is_retained_name': True,
    },
    # dibenzofuran
    # D-FOLLOWON item 4 (DATA-01 remnant): stored grid put the O bridge (idx 6) at
    # locant 9 and the second benzo ring (idx 8-11) at 5-8; OPSIN's authoritative
    # numbering (`-o extendedsmi` $_AV:) puts O at 5, the second benzo ring at 6-9,
    # and fusion atoms at 9b/5a/9a. The O/S anchor fixes the C2v axis, so the old
    # grid was NOT a valid automorphic labeling -> di-substituents spanning both
    # rings (2,8-dimethyl) emerged shifted (2,7) and SELF-01-suppressed. Re-derived
    # from OPSIN extendedsmi (independently reproduced).
    'c1ccc2c(c1)oc1ccccc12': {
        # PIN requires the fusion-locant descriptor [b,d] (P-25.3.1.3); bare
        # 'dibenzofuran' is general/retained-style. dibenzofuran is NOT in
        # Table 2.8 retained heterocycles, so PIN style needs the descriptor.
        # BB verbatim: 'dibenzo[b,d]furan-1-yl (PIN)' (line 7475, 'not
        #...dibenzofuran'). The fixed skeleton always fuses on sides b (2,3)
        # and d (4,5), so [b,d] is a constant of this entry.
        'name': 'dibenzo[b,d]furan',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '9b', 4: '4a', 5: 4, 6: 5, 7: '5a', 8: 6, 9: 7, 10: 8, 11: 9, 12: '9a'},
    },
    # dibenzo[b,d]thiophene (same skeleton, S substitutes for O at idx 6 -> identical grid)
    # PIN needs the [b,d] descriptor (P-25.3.1.3); not in Table 2.8. BB verbatim
    # 'dibenzo[b,d]thiophene] (PIN)' (line 11250).
    'c1ccc2c(c1)sc1ccccc12': {
        'name': 'dibenzo[b,d]thiophene',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '9b', 4: '4a', 5: 4, 6: 5, 7: '5a', 8: 6, 9: 7, 10: 8, 11: 9, 12: '9a'},
    },
    # benzo[f]quinoline
    'c1ccc2c(c1)ccc1ncccc12': {
        'name': 'benzo[f]quinoline',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 8, 1: 9, 2: 10, 3: '10a', 4: '6a', 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4, 10: 3, 11: 2, 12: 1, 13: '10b'},
    },
    # benzo[g]quinoline
    'c1ccc2cc3ncccc3cc2c1': {
        'name': 'benzo[g]quinoline',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 7, 1: 8, 2: 9, 3: '9a', 4: 10, 5: '10a', 6: 1, 7: 2, 8: 3, 9: 4, 10: '4a', 11: 5, 12: '5a', 13: 6},
    },
    # benzo[g]isoquinoline
    'c1ccc2cc3cnccc3cc2c1': {
        'name': 'benzo[g]isoquinoline',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 7, 1: 8, 2: 9, 3: '9a', 4: 10, 5: '10a', 6: 1, 7: 2, 8: 3, 9: 4, 10: '4a', 11: 5, 12: '5a', 13: 6},
    },
    # 9H-fluoren-9-one
    'O=C1c2ccccc2-c2ccccc21': {
        'name': '9H-fluoren-9-one',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: '=O', 1: 9, 2: '9a', 3: 1, 4: 2, 5: 3, 6: 4, 7: '4a', 8: '4b', 9: 5, 10: 6, 11: 7, 12: 8, 13: '8a'},
        'is_retained_name': True,
    },
    # phenanthridin-6(5H)-one
    'O=c1[nH]c2ccccc2c2ccccc12': {
        'name': 'phenanthridin-6(5H)-one',
        'tautomer_locant': 5,
        'ring_system': 'tricyclic',
        'parent_atoms': 15,
        'iupac_locants': {0: '=O', 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: 1, 8: '10a', 9: '4b', 10: 7, 11: 8, 12: 9, 13: 10, 14: '10b'},
        'is_retained_name': True,
    },
    # 4H-quinolizine
    'C1=CCN2C=CC=CC2=C1': {
        'name': '4H-quinolizine',
        'tautomer_locant': 4,
        'ring_system': 'bridgehead',
        'parent_atoms': 10,
        'iupac_locants': {0: 2, 1: 3, 2: 4, 3: '4a', 4: 6, 5: 7, 6: 8, 7: 9, 8: '9a', 9: 1},
    },
    # quinolizidine
    'C1CCN2CCCCC2C1': {
        'name': 'quinolizidine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 10,
        'iupac_locants': {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: '9a', 9: 1},
    },
    # pyrrolizine
    'C1=Cn2cccc2C1': {
        'name': 'pyrrolizine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 8,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 5, 4: 6, 5: 7, 6: '7a', 7: 1},
    },
    # xanthone
    'O=c1c2ccccc2oc2ccccc12': {
        'name': 'xanthone',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 15,
        'iupac_locants': {0: '=O', 1: 9, 2: '9a', 3: 1, 4: 2, 5: 3, 6: 4, 7: '4a', 8: '10a', 9: '4b', 10: 5, 11: 6, 12: 7, 13: 8, 14: '8a'},
        'is_retained_name': True,
    },
    # thioxanthone
    'O=c1c2ccccc2sc2ccccc12': {
        'name': 'thioxanthone',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 15,
        'iupac_locants': {0: '=O', 1: 9, 2: '9a', 3: 1, 4: 2, 5: 3, 6: 4, 7: '4a', 8: '10a', 9: '4b', 10: 5, 11: 6, 12: 7, 13: 8, 14: '8a'},
        'is_retained_name': True,
    },
    # 1,10-phenanthroline
    'c1cnc2c(c1)ccc1cccnc12': {
        'name': '1,10-phenanthroline',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 8, 1: 9, 2: 10, 3: '8a', 4: '4b', 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4, 10: 3, 11: 2, 12: 1, 13: '10a'},
    },
    # 1H-perimidine
    'C1=Nc2cccc3cccc(c23)N1': {
        'name': '1H-perimidine',
        'tautomer_locant': 1,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: '6a', 7: 7, 8: 8, 9: 9, 10: '9a', 11: '9b', 12: 1},
    },
    # 9H-thioxanthene
    'c1ccc2c(c1)Cc1ccccc1S2': {
        'name': '9H-thioxanthene',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 2, 1: 3, 2: 4, 3: '4a', 4: '9a', 5: 1, 6: 9, 7: '8a', 8: 8, 9: 7, 10: 6, 11: 5, 12: '10a', 13: 10},
    },
    # 2-benzofuran (PIN; isobenzofuran/benzo[c]furan are non-PIN synonyms —
    # Blue Book P-25 line 11829: "2-benzofuran (PIN) isobenzofuran benzo[c]furan";
    # cf. its dione = phthalic anhydride = 2-benzofuran-1,3-dione, the Blue Book)
    'c1ccc2cocc2c1': {
        'name': '2-benzofuran',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # Wave2 T4 (P-25.2.2.4 benzo-heterocycles + P-25.7.1.3.2 fused indicated-H):
    # bare fused-name catalog entries that were fail-closed 'unknown' at HEAD.
    # OPSIN-RT + numbering verified (extendedsmi mapped onto RDKit-canonical,
    # confirmed via methyl-isomer attachment). The benzoxepines/benzothiepine
    # have a divalent O/S -> no indicated H; the benzazepines and the pyrrole-
    # fused pairs carry a MANDATORY indicated hydrogen baked into 'name'.
    'C1=COc2ccccc2C=C1': {
        'name': '1-benzoxepine',
        'tautomer_locant': None,
        'ring_system': 'benzo-7-membered',
        'parent_atoms': 11,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '9a', 4: 9, 5: 8, 6: 7, 7: 6, 8: '5a', 9: 5, 10: 4},
    },
    'C1=COC=c2ccccc2=C1': {
        'name': '2-benzoxepine',
        'tautomer_locant': None,
        'ring_system': 'benzo-7-membered',
        'parent_atoms': 11,
        'iupac_locants': {0: 4, 1: 3, 2: 2, 3: 1, 4: '9a', 5: 9, 6: 8, 7: 7, 8: 6, 9: '5a', 10: 5},
    },
    'C1=Cc2ccccc2C=CO1': {
        'name': '3-benzoxepine',
        'tautomer_locant': None,
        'ring_system': 'benzo-7-membered',
        'parent_atoms': 11,
        'iupac_locants': {0: 2, 1: 1, 2: '9a', 3: 9, 4: 8, 5: 7, 6: 6, 7: '5a', 8: 5, 9: 4, 10: 3},
    },
    'C1=Cc2ccccc2C=CS1': {
        'name': '3-benzothiepine',
        'tautomer_locant': None,
        'ring_system': 'benzo-7-membered',
        'parent_atoms': 11,
        'iupac_locants': {0: 2, 1: 1, 2: '9a', 3: 9, 4: 8, 5: 7, 6: 6, 7: '5a', 8: 5, 9: 4, 10: 3},
    },
    'C1=CNc2ccccc2C=C1': {
        'name': '1H-1-benzazepine',
        'tautomer_locant': 1,
        'ring_system': 'benzo-7-membered',
        'parent_atoms': 11,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '9a', 4: 9, 5: 8, 6: 7, 7: 6, 8: '5a', 9: 5, 10: 4},
    },
    'C1=CNC=c2ccccc2=C1': {
        'name': '2H-2-benzazepine',
        'tautomer_locant': 2,
        'ring_system': 'benzo-7-membered',
        'parent_atoms': 11,
        'iupac_locants': {0: 4, 1: 3, 2: 2, 3: 1, 4: '9a', 5: 9, 6: 8, 7: 7, 8: 6, 9: '5a', 10: 5},
    },
    'C1=Cc2ccccc2C=CN1': {
        'name': '3H-3-benzazepine',
        'tautomer_locant': 3,
        'ring_system': 'benzo-7-membered',
        'parent_atoms': 11,
        'iupac_locants': {0: 2, 1: 1, 2: '9a', 3: 9, 4: 8, 5: 7, 6: 6, 7: '5a', 8: 5, 9: 4, 10: 3},
    },
    'c1cc2occc2[nH]1': {
        'name': '4H-furo[3,2-b]pyrrole',
        'tautomer_locant': 4,
        'ring_system': 'bicyclic-5-5',
        'parent_atoms': 8,
        'iupac_locants': {0: '5', 1: '6', 2: '6a', 3: '1', 4: '2', 5: '3', 6: '3a', 7: '4'},
    },
    'c1cc2sccc2[nH]1': {
        'name': '4H-thieno[3,2-b]pyrrole',
        'tautomer_locant': 4,
        'ring_system': 'bicyclic-5-5',
        'parent_atoms': 8,
        'iupac_locants': {0: '5', 1: '6', 2: '6a', 3: '1', 4: '2', 5: '3', 6: '3a', 7: '4'},
    },
    'c1cc2ccoc2[nH]1': {
        'name': '6H-furo[2,3-b]pyrrole',
        'tautomer_locant': 6,
        'ring_system': 'bicyclic-5-5',
        'parent_atoms': 8,
        'iupac_locants': {0: '5', 1: '4', 2: '3a', 3: '3', 4: '2', 5: '1', 6: '6a', 7: '6'},
    },
    # 1,6-dihydropyrrolo[2,3-b]pyrrole (Wave-2 completion, P-25.7.1.3.2).
    # The two-NH compound is the DIHYDRO derivative: the bare mancude
    # 'pyrrolo[2,3-b]pyrrole' is a different molecule (4 noncumulative double
    # bonds, no NH) — the ledger's bare-name expectation was a lenient OPSIN
    # parse. Locants OPSIN-derived (extended-SMILES atom values).
    'c1cc2cc[nH]c2[nH]1': {
        'name': '1,6-dihydropyrrolo[2,3-b]pyrrole',
        'tautomer_locant': None,
        'ring_system': 'bicyclic-5-5',
        'parent_atoms': 8,
        'iupac_locants': {0: '5', 1: '4', 2: '3a', 3: '3', 4: '2', 5: '1', 6: '6a', 7: '6'},
    },
    # 1,3-dihydro-2-benzofuran (phthalan) — the 1,3-dihydro form of 2-benzofuran
    # (parallel to the cataloged 2,3-dihydro-1-benzofuran). PIN per BB P-25;
    # iupac_locants OPSIN-derived (O at 2, CH2 at 1 and 3).
    'c1ccc2c(c1)COC2': {
        'name': '1,3-dihydro-2-benzofuran',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered-saturated',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: '7a', 5: 7, 6: 1, 7: 2, 8: 3},
        'systematic': '1,3-dihydro-2-benzofuran',
    },
    # 1,2-benzisoxazole
    'c1ccc2oncc2c1': {
        'name': '1,2-benzisoxazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 1,3-benzoselenazole
    'c1ccc2[se]cnc2c1': {
        'name': '1,3-benzoselenazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # 1H-indene
    'C1=Cc2ccccc2C1': {
        'name': '1H-indene',
        'tautomer_locant': 1,
        'ring_system': 'bicyclic-carbocyclic',
        'parent_atoms': 9,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: 7, 7: '7a', 8: 1},
    },
    # indane
    'c1ccc2c(c1)CCC2': {
        # The name below IS the PIN. P-54.4.3.2 (the Blue Book) names the retained form
        # verbatim as non-preferred -- "The retained names for the partially saturated
        # heterocycles, indane, indoline, isoindoline, and chromane, isochromane and
        # their chalcogen analogues are not used as preferred IUPAC names" -- and
        # the Blue Book print the three PINs (2,3-dihydro-1H-indene /
        # -1H-indole / -1H-isoindole).
        #
        # ⚠ HISTORY, so this is not re-broken: the rename was attempted once and the
        # gate correctly rejected it (1 protect + 3 target regressions). This `name`
        # field feeds TWO consumers. Standalone naming wants the saturated PIN, but
        # `rules/spiro.py:_name_spirobi_core` embedded it as the SPIRO COMPONENT, and
        # P-24.3.1 (the Blue Book) requires the bracket to hold the MANCUDE component ring
        # system with hydrogen cited OUTSIDE it -- so the bare rename produced
        # `1,2'-spirobi[2,3-dihydro-1H-indene]`. The obvious shortcut is WORSE, not
        # merely wrong: `1,2'-spirobi[1H-indene]` denotes the UNSATURATED molecule --
        # a wrong STRUCTURE, not a wrong spelling (session a project rule).
        #
        # ✅ UNBLOCKED (Phase C). `_name_spirobi_core` no longer reads saturation
        # out of this string at all: it derives the hydro prefixes and any indicated
        # hydrogen from the GRAPH (maximum noncumulative double-bond assignment over
        # the assembled skeleton, the spiro atom excluded) and hoists them in front of
        # the spiro locants, per P-24.3.2 (the Blue Book) and the the Blue Book template
        # `1,3'-dihydro-3H-1lambda6,1'-spirobi[[2,1]benzoxathiole]`. It now emits
        # `1',2,3,3'-tetrahydro-1,2'-spirobi[indene]`. The spiro consumer is therefore
        # INDEPENDENT of this field, and renaming here is safe.
        'name': '2,3-dihydro-1H-indene',
        'tautomer_locant': None,
        'ring_system': 'bicyclic-saturated',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: '3a', 5: 4, 6: 3, 7: 2, 8: 1},
    },
    # biphenylene
    'c1ccc2c(c1)-c1ccccc1-2': {
        'name': 'biphenylene',
        'tautomer_locant': None,
        'ring_system': 'non-benzenoid',
        'parent_atoms': 12,
        'iupac_locants': {0: 2, 1: 3, 2: 4, 3: '4a', 4: '8b', 5: 1, 6: '8a', 7: 8, 8: 7, 9: 6, 10: 5, 11: '4b'},
    },
    # acenaphthylene
    'C1=Cc2cccc3cccc1c23': {
        'name': 'acenaphthylene',
        'tautomer_locant': None,
        'ring_system': 'tricyclic-carbocyclic',
        'parent_atoms': 12,
        'iupac_locants': {0: 1, 1: 2, 2: '2a', 3: 3, 4: 4, 5: 5, 6: '5a', 7: 6, 8: 7, 9: 8, 10: '8a', 11: '8b'},
    },
    # 1H-phenalene
    'C1=Cc2cccc3cccc(c23)C1': {
        'name': '1H-phenalene',
        'tautomer_locant': 1,
        'ring_system': 'tricyclic-carbocyclic',
        'parent_atoms': 13,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: '6a', 7: 7, 8: 8, 9: 9, 10: '9a', 11: '9b', 12: 1},
    },
    # pyrene
    'c1cc2ccc3cccc4ccc(c1)c2c34': {
        'name': 'pyrene',
        'tautomer_locant': None,
        'ring_system': 'tetracyclic-carbocyclic',
        'parent_atoms': 16,
        'iupac_locants': {0: 2, 1: 1, 2: '10a', 3: 10, 4: 9, 5: '8a', 6: 8, 7: 7, 8: 6, 9: '5a', 10: 5, 11: 4, 12: '3a', 13: 3, 14: '10b', 15: '10c'},
    },
    # chrysene
    'c1ccc2c(c1)ccc1c3ccccc3ccc21': {
        'name': 'chrysene',
        'tautomer_locant': None,
        'ring_system': 'tetracyclic-carbocyclic',
        'parent_atoms': 18,
        'iupac_locants': {0: 2, 1: 3, 2: 4, 3: '4a', 4: '12a', 5: 1, 6: 12, 7: 11, 8: '10b', 9: '10a', 10: 10, 11: 9, 12: 8, 13: 7, 14: '6a', 15: 6, 16: 5, 17: '4b'},
    },
    # triphenylene
    'c1ccc2c(c1)c1ccccc1c1ccccc21': {
        'name': 'triphenylene',
        'tautomer_locant': None,
        'ring_system': 'tetracyclic-carbocyclic',
        'parent_atoms': 18,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '12b', 4: '4a', 5: 4, 6: '4b', 7: 5, 8: 6, 9: 7, 10: 8, 11: '8a', 12: '8b', 13: 9, 14: 10, 15: 11, 16: 12, 17: '12a'},
    },
    # naphthacene
    'c1ccc2cc3cc4ccccc4cc3cc2c1': {
        'name': 'naphthacene',
        'tautomer_locant': None,
        'ring_system': 'tetracyclic-carbocyclic',
        'parent_atoms': 18,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '12a', 4: 12, 5: '11a', 6: 11, 7: '10a', 8: 10, 9: 9, 10: 8, 11: 7, 12: '6a', 13: 6, 14: '5a', 15: 5, 16: '4a', 17: 4},
    },
    # s-indacene
    'C1=Cc2cc3c(cc2=C1)C=CC=3': {
        'name': 's-indacene',
        'tautomer_locant': None,
        'ring_system': 'non-benzenoid',
        'parent_atoms': 12,
        'iupac_locants': {0: 2, 1: 1, 2: '8a', 3: 8, 4: '7a', 5: '4a', 6: 4, 7: '3a', 8: 3, 9: 5, 10: 6, 11: 7},
    },
    # as-indacene
    'C1=Cc2c3c(ccc2=C1)=CC=C3': {
        'name': 'as-indacene',
        'tautomer_locant': None,
        'ring_system': 'non-benzenoid',
        'parent_atoms': 12,
        'iupac_locants': {0: 2, 1: 1, 2: '8b', 3: '8a', 4: '5a', 5: 5, 6: 4, 7: '3a', 8: 3, 9: 6, 10: 7, 11: 8},
    },
    # heptalene
    'C1=CC=C2C=CC=CC=C2C=C1': {
        'name': 'heptalene',
        'tautomer_locant': None,
        'ring_system': 'non-benzenoid',
        'parent_atoms': 12,
        'iupac_locants': {0: 3, 1: 4, 2: 5, 3: '5a', 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: '10a', 10: 1, 11: 2},
    },
    # 1H-pyrazolo[3,4-b]pyridine
    'c1cnc2[nH]ncc2c1': {
        'name': '1H-pyrazolo[3,4-b]pyridine',
        'tautomer_locant': 1,
        'ring_system': 'pyrazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # oxazolo[4,5-b]pyridine
    'c1cnc2ncoc2c1': {
        'name': 'oxazolo[4,5-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'oxazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: 3, 5: 2, 6: 1, 7: '7a', 8: 7},
    },
    # thiazolo[5,4-b]pyridine
    'c1cnc2scnc2c1': {
        'name': 'thiazolo[5,4-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'thiazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: 4, 3: '3a', 4: 3, 5: 2, 6: 1, 7: '7a', 8: 7},
    },
    # isoxazolo[5,4-b]pyridine
    'c1cnc2oncc2c1': {
        'name': 'isoxazolo[5,4-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'isoxazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # thiazolo[4,5-b]pyrazine
    'c1cnc2scnc2n1': {
        'name': 'thiazolo[4,5-b]pyrazine',
        'tautomer_locant': None,
        'ring_system': 'thienopyrimidine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # pyrido[2,3-d]pyridazine
    'c1cnc2cnncc2c1': {
        'name': 'pyrido[2,3-d]pyridazine',
        'tautomer_locant': None,
        'ring_system': 'pyridopyridazine',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '8a', 4: 8, 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4},
    },
    # pyrazino[2,3-b]pyrazine
    'c1cnc2nccnc2n1': {
        'name': 'pyrazino[2,3-b]pyrazine',
        'tautomer_locant': None,
        'ring_system': 'pyrazinopyrazine',
        'parent_atoms': 10,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: '4a', 4: 4, 5: 3, 6: 2, 7: 1, 8: '8a', 9: 8},
    },
    # pyrido[4,3-d]pyrimidine
    'c1cc2ncncc2cn1': {
        'name': 'pyrido[4,3-d]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'pyridopyrimidine',
        'parent_atoms': 10,
        'iupac_locants': {0: 7, 1: 8, 2: '8a', 3: 1, 4: 2, 5: 3, 6: 4, 7: '4a', 8: 5, 9: 6},
    },
    # pyrido[3,4-b]pyrazine
    'c1cc2nccnc2cn1': {
        'name': 'pyrido[3,4-b]pyrazine',
        'tautomer_locant': None,
        'ring_system': 'pyridopyrimidine',
        'parent_atoms': 10,
        'iupac_locants': {0: 7, 1: 8, 2: '8a', 3: 1, 4: 2, 5: 3, 6: 4, 7: '4a', 8: 5, 9: 6},
    },
    # thieno[2,3-c]pyridine
    'c1cc2ccsc2cn1': {
        'name': 'thieno[2,3-c]pyridine',
        'tautomer_locant': None,
        'ring_system': 'thienopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 4, 2: '3a', 3: 3, 4: 2, 5: 1, 6: '7a', 7: 7, 8: 6},
    },
    # thieno[3,2-c]pyridine
    'c1cc2sccc2cn1': {
        'name': 'thieno[3,2-c]pyridine',
        'tautomer_locant': None,
        'ring_system': 'thienopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: '7a', 3: 1, 4: 2, 5: 3, 6: '3a', 7: 4, 8: 5},
    },
    # furo[3,2-c]pyridine
    'c1cc2occc2cn1': {
        'name': 'furo[3,2-c]pyridine',
        'tautomer_locant': None,
        'ring_system': 'furopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: '7a', 3: 1, 4: 2, 5: 3, 6: '3a', 7: 4, 8: 5},
    },
    # 1H-pyrrolo[3,2-c]pyridine
    'c1cc2[nH]ccc2cn1': {
        'name': '1H-pyrrolo[3,2-c]pyridine',
        'tautomer_locant': 1,
        'ring_system': 'azaindole',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: '7a', 3: 1, 4: 2, 5: 3, 6: '3a', 7: 4, 8: 5},
    },
    # pyrrolo[1,2-b]pyridazine
    'c1cnn2cccc2c1': {
        'name': 'pyrrolo[1,2-b]pyridazine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: 8, 4: 7, 5: 6, 6: 5, 7: '4a', 8: 4},
    },
    # [1,2,3,4]tetrazolo[1,5-a]pyridine
    'c1ccn2nnnc2c1': {
        'name': '[1,2,3,4]tetrazolo[1,5-a]pyridine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: '8a', 8: 8},
    },
    # [1,2,4]triazolo[1,5-a]pyridine
    'c1ccn2ncnc2c1': {
        'name': '[1,2,4]triazolo[1,5-a]pyridine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: '8a', 8: 8},
    },
    # imidazo[2,1-b]thiazole
    'c1cn2ccsc2n1': {
        'name': 'imidazo[2,1-b]thiazole',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 8,
        'iupac_locants': {0: 6, 1: 5, 2: '5a', 3: 3, 4: 2, 5: 1, 6: '3a', 7: 7},
    },
    # pyrrolo[1,2-c]pyrimidine
    'c1cc2ccncn2c1': {
        'name': 'pyrrolo[1,2-c]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 5, 2: '4a', 3: 4, 4: 3, 5: 2, 6: 1, 7: 8, 8: 7},
    },
    # pyrrolo[1,2-a]pyrazine
    'c1cc2cnccn2c1': {
        'name': 'pyrrolo[1,2-a]pyrazine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 8, 2: '8a', 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6},
    },
    # pyrrolizidine
    'C1CC2CCCN2C1': {
        'name': 'pyrrolizidine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 8,
        'iupac_locants': {0: 2, 1: 1, 2: '7a', 3: 7, 4: 6, 5: 5, 6: 4, 7: 3},
    },
    # indolizidine
    'C1CCN2CCCC2C1': {
        'name': 'indolizidine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: '8a', 8: 8},
    },
    # phenoxathiine -- P-25.2.2.3 (the Blue Book) prints "X = S
    # phenoxathiine (PIN)"; the terminal 'e' is part of the PIN stem (the whole
    # C4OS-C6-C6 family: phenoxathiine/phenoxaselenine/phenoxaphosphinine at
    # the Blue Book). This catalog value is the on-path source for the bare parent
    # (trace-verified: replacing it makes the OPSIN gate reject the emission).
    'c1ccc2c(c1)Oc1ccccc1S2': {
        'name': 'phenoxathiine',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '10a', 4: '4a', 5: 4, 6: 10, 7: '5a', 8: 6, 9: 7, 10: 8, 11: 9, 12: '9a', 13: 5},
    },
    # benzofuro[3,2-b]pyridine
    'c1ccc2c(c1)oc1cccnc12': {
        'name': 'benzofuro[3,2-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {0: 7, 1: 8, 2: 9, 3: '9a', 4: '5a', 5: 6, 6: 5, 7: '4a', 8: 4, 9: 3, 10: 2, 11: 1, 12: '9b'},
    },
    # thieno[3,2-b]thiophene
    'c1cc2sccc2s1': {
        'name': 'thieno[3,2-b]thiophene',
        'tautomer_locant': None,
        'ring_system': 'thienothiophene',
        'parent_atoms': 8,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: '6a', 7: 1},
    },
    # thieno[2,3-b]thiophene
    'c1cc2ccsc2s1': {
        'name': 'thieno[2,3-b]thiophene',
        'tautomer_locant': None,
        'ring_system': 'thienothiophene',
        'parent_atoms': 8,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: 4, 4: 5, 5: 6, 6: '6a', 7: 1},
    },
    # xanthine
    'O=c1[nH]c(=O)c2nc[nH]c2[nH]1': {
        'name': 'xanthine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 11,
        'iupac_locants': {0: 'O6', 1: 6, 2: 1, 3: 2, 4: 'O2', 5: 5, 6: 7, 7: 8, 8: 9, 9: 4, 10: 3},
        'is_retained_name': True,
    },
    # xanthine
    'O=c1[nH]c(=O)c2[nH]cnc2[nH]1': {
        'name': 'xanthine',
        'tautomer_locant': None,
        'ring_system': 'purine',
        'parent_atoms': 11,
        'iupac_locants': {0: 'O6', 1: 6, 2: 1, 3: 2, 4: 'O2', 5: 3, 6: 9, 7: 8, 8: 7, 9: 5, 10: 4},
        'is_retained_name': True,
    },
    # benzo[g]pteridine-2,4(1H,3H)-dione
    'O=c1[nH]c(=O)c2nc3ccccc3nc2[nH]1': {
        'name': 'benzo[g]pteridine-2,4(1H,3H)-dione',
        'tautomer_locant': None,
        'ring_system': 'pteridine-related',
        'parent_atoms': 16,
        'iupac_locants': {0: 'O2', 1: 2, 2: 3, 3: 4, 4: 'O4', 5: '4a', 6: 5, 7: '5a', 8: 6, 9: 7, 10: 8, 11: 9, 12: '9a', 13: 10, 14: '10a', 15: 1},
        'is_retained_name': True,
    },
    # pyrimido[5,4-d]pyrimidine
    'c1ncc2ncncc2n1': {
        'name': 'pyrimido[5,4-d]pyrimidine',
        'tautomer_locant': None,
        'ring_system': 'pyrimidopyrimidine',
        'parent_atoms': 10,
        'iupac_locants': {0: 2, 1: 3, 2: 4, 3: '4a', 4: 5, 5: 6, 6: 7, 7: 8, 8: '8a', 9: 1},
    },
    # 5H-pyrrolo[3,2-d]pyrimidine
    'c1ncc2[nH]ccc2n1': {
        'name': '5H-pyrrolo[3,2-d]pyrimidine',
        'tautomer_locant': 5,
        'ring_system': 'pyrrolopyrazine',
        'parent_atoms': 9,
        'iupac_locants': {0: 2, 1: 3, 2: 4, 3: '4a', 4: 5, 5: 6, 6: 7, 7: '7a', 8: 1},
    },
    # thiochromene -> PIN 2H-1-benzothiopyran (P-19(d); the Blue Book 'the PIN is 2H-1-benzothiopyran')
    'C1=Cc2ccccc2SC1': {
        'name': '2H-1-benzothiopyran',
        'tautomer_locant': 2,
        'ring_system': 'benzothiopyran',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 4, 2: '4a', 3: 5, 4: 6, 5: 7, 6: 8, 7: '8a', 8: 1, 9: 2},
    },
    # 4H-1-benzothiopyran (4H isomer) — the mancude oxo-parent for thiochroman-4-one
    # (carbonyl at the 4H indicated position, P-64.2.2.2.2). Locants from OPSIN extendedsmi (S@1).
    'C1=CSc2ccccc2C1': {
        'name': '4H-1-benzothiopyran',
        'tautomer_locant': 4,
        'ring_system': 'benzothiopyran',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '8a', 4: 8, 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4},
    },
    # thiochromane -> PIN 3,4-dihydro-2H-1-benzothiopyran (the Blue Book)
    'c1ccc2c(c1)CCCS2': {
        'name': '3,4-dihydro-2H-1-benzothiopyran',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # selenochromane -> PIN 3,4-dihydro-2H-1-benzoselenopyran (BB Table 3.1, line 17008;
    # retained name selenochromane is NOT a PIN, P-31.2.3.3.1 line 16509 / P-54.4.3.2 line 24256)
    'c1ccc2c(c1)CCC[Se]2': {
        'name': '3,4-dihydro-2H-1-benzoselenopyran',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # tellurochromane -> PIN 3,4-dihydro-2H-1-benzotelluropyran (BB Table 3.1, line 17010;
    # retained name tellurochromane is NOT a PIN, P-31.2.3.3.1 line 16509 / P-54.4.3.2 line 24256)
    'c1ccc2c(c1)CCC[Te]2': {
        'name': '3,4-dihydro-2H-1-benzotelluropyran',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # 2,3-dihydro-1,4-benzodioxine
    'c1ccc2c(c1)OCCO2': {
        'name': '2,3-dihydro-1,4-benzodioxine',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # 2,3-dihydro-1-benzothiophene
    'c1ccc2c(c1)CCS2': {
        'name': '2,3-dihydro-1-benzothiophene',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-saturated',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: '3a', 5: 4, 6: 3, 7: 2, 8: 1},
    },
    # 1,2,3,4-tetrahydroquinoxaline
    'c1ccc2c(c1)NCCN2': {
        'name': '1,2,3,4-tetrahydroquinoxaline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 6, 1: 7, 2: 8, 3: '8a', 4: '4a', 5: 5, 6: 4, 7: 3, 8: 2, 9: 1},
    },
    # 1,4-dihydroisoquinoline
    'C1=NCc2ccccc2C1': {
        'name': '1,4-dihydroisoquinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 2, 2: 1, 3: '8a', 4: 8, 5: 7, 6: 6, 7: 5, 8: '4a', 9: 4},
    },
    # 2,3-dihydro-1H-pyrrolo[2,3-b]pyridine
    'c1cnc2c(c1)CCN2': {
        'name': '2,3-dihydro-1H-pyrrolo[2,3-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-saturated',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: '3a', 5: 4, 6: 3, 7: 2, 8: 1},
    },
    # imidazo[1,2-b]pyridazine
    'c1cnn2ccnc2c1': {
        'name': 'imidazo[1,2-b]pyridazine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: '8a', 8: 8},
    },
    # 1H-pyrazolo[4,3-c]pyridine
    'c1cc2[nH]ncc2cn1': {
        'name': '1H-pyrazolo[4,3-c]pyridine',
        'tautomer_locant': 1,
        'ring_system': 'pyrazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: '7a', 3: 1, 4: 2, 5: 3, 6: '3a', 7: 4, 8: 5},
    },
    # thieno[2,3-c]pyridazine
    'c1cc2ccsc2nn1': {
        'name': 'thieno[2,3-c]pyridazine',
        'tautomer_locant': None,
        'ring_system': 'thienopyridazine',
        'parent_atoms': 9,
        'iupac_locants': {0: 3, 1: 4, 2: '4a', 3: 5, 4: 6, 5: 7, 6: '7a', 7: 1, 8: 2},
    },
    # oxazolo[5,4-c]pyridine
    'c1cc2ncoc2cn1': {
        'name': 'oxazolo[5,4-c]pyridine',
        'tautomer_locant': None,
        'ring_system': 'oxazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: '7a', 3: 1, 4: 2, 5: 3, 6: '3a', 7: 4, 8: 5},
    },
    # thiazolo[5,4-c]pyridine
    'c1cc2ncsc2cn1': {
        'name': 'thiazolo[5,4-c]pyridine',
        'tautomer_locant': None,
        'ring_system': 'thiazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 6, 1: 7, 2: '7a', 3: 1, 4: 2, 5: 3, 6: '3a', 7: 4, 8: 5},
    },
    # isoxazolo[3,4-b]pyridine
    'c1cnc2nocc2c1': {
        'name': 'isoxazolo[3,4-b]pyridine',
        'tautomer_locant': None,
        'ring_system': 'isoxazolopyridine',
        'parent_atoms': 9,
        'iupac_locants': {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4},
    },
    # pyrazino[2,3-c]pyridazine
    'c1cc2nccnc2nn1': {
        'name': 'pyrazino[2,3-c]pyridazine',
        'tautomer_locant': None,
        'ring_system': 'pyrazinopyridazine',
        'parent_atoms': 10,
        'iupac_locants': {0: 3, 1: 4, 2: '4a', 3: 5, 4: 6, 5: 7, 6: 8, 7: '8a', 8: 1, 9: 2},
    },

    # ── Wave-2 completion C (P-25.3.4/.5/.6/.8 BB-cited multi-component
    # fusion parents; the general multi-prime descriptor ENGINE remains
    # unbuilt — these are the specific BB-named parents, locants
    # OPSIN-extendedsmi-derived; T4 pentaphene/hexahelicene precedent). ──
    'c1ccc2nc3cc4nc5c(nc4cc3nc2c1)nc1ccccn15': {
        'name': "pyrido[1'',2'':1',2']imidazo[4',5':5,6]pyrazino[2,3-b]phenazine",
        'tautomer_locant': None,
        'ring_system': 'polycyclic',
        'parent_atoms': 25,
        'iupac_locants': {0: 10, 1: 11, 2: 12, 3: '12a', 4: 13, 5: '13a', 6: 14, 7: '14a', 8: 15, 9: '15a', 10: '5a', 11: 6, 12: '6a', 13: 7, 14: '7a', 15: 8, 16: '8a', 17: 9, 18: 5, 19: '4a', 20: 4, 21: 3, 22: 2, 23: 1, 24: 16},
    },
    'c1cc2c3ccc4cncc5ccc(c6ccc7cncc1c7c26)c3c45': {
        'name': "anthra[2,1,9-def:6,5,10-d'e'f']diisoquinoline",
        'tautomer_locant': None,
        'ring_system': 'polycyclic',
        'parent_atoms': 26,
        'iupac_locants': {0: 11, 1: 12, 2: '12a', 3: '12b', 4: 13, 5: 14, 6: '14a', 7: 1, 8: 2, 9: 3, 10: '3a', 11: 4, 12: 5, 13: '5a', 14: '5b', 15: 6, 16: 7, 17: '7a', 18: 8, 19: 9, 20: 10, 21: '10a', 22: '14d', 23: '14e', 24: '14c', 25: '14b'},
    },
    'C1=Cc2c(c3ccccc3c3ncoc23)C1': {
        'name': '8H-cyclopenta[3,4]naphtho[1,2-d][1,3]oxazole',
        'tautomer_locant': 8,   # was None; the name's own indicated H (metadata only)
        'ring_system': 'polycyclic',
        'parent_atoms': 16,
        'iupac_locants': {0: 9, 1: 10, 2: '10a', 3: '7b', 4: '7a', 5: 7, 6: 6, 7: 5, 8: 4, 9: '3b', 10: '3a', 11: 3, 12: 2, 13: 1, 14: '10b', 15: 8},
    },
    'c1ccc2c(c1)ccc1c3ccsc3c3ccc4ccccc4c3c21': {
        'name': "naphtho[2',1':3,4]phenanthro[1,2-b]thiophene",
        'tautomer_locant': None,
        'ring_system': 'polycyclic',
        'parent_atoms': 25,
        'iupac_locants': {0: 9, 1: 10, 2: 11, 3: '11a', 4: '7a', 5: 8, 6: 7, 7: 6, 8: '5b', 9: '5a', 10: 5, 11: 4, 12: 3, 13: '2b', 14: '2a', 15: 2, 16: 1, 17: '15a', 18: 15, 19: 14, 20: 13, 21: 12, 22: '11d', 23: '11c', 24: '11b'},
    },
    'C1=CN2C=CC3=CNOC3=C2O1': {
        'name': '2H-[1,2]oxazolo[5,4-c][1,3]oxazolo[3,2-a]pyridine',
        'tautomer_locant': 2,   # was None; the name's own indicated H (metadata only)
        'ring_system': 'polycyclic',
        'parent_atoms': 12,
        'iupac_locants': {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: '3a', 6: 3, 7: 2, 8: 1, 9: '9b', 10: '9a', 11: 9},
    },
    'C1=NC=c2ccc3c(c21)N=c1cc2occc2cc1=3': {
        'name': 'furo[3,2-h]pyrrolo[3,4-a]carbazole',
        'tautomer_locant': None,
        'ring_system': 'polycyclic',
        'parent_atoms': 19,
        'iupac_locants': {0: 1, 1: 2, 2: 3, 3: '3a', 4: 4, 5: 5, 6: '5a', 7: '11a', 8: '11b', 9: 11, 10: '10a', 11: 10, 12: '9a', 13: 9, 14: 8, 15: 7, 16: '6a', 17: 6, 18: '5b'},
    },
    'c1cc2cc3csnc3cc2s1': {
        'name': 'thieno[3,2-f][2,1]benzothiazole',
        'tautomer_locant': None,
        'ring_system': 'polycyclic',
        'parent_atoms': 12,
        'iupac_locants': {0: 6, 1: 5, 2: '4a', 3: 4, 4: '3a', 5: 3, 6: 2, 7: 1, 8: '8a', 9: 8, 10: '7a', 11: 7},
    },
    'c1cc2c(o1)oc1cc3cocc3nc12': {
        'name': "furo[3,4-b]furo[3',2':4,5]furo[2,3-e]pyridine",
        'tautomer_locant': None,
        'ring_system': 'polycyclic',
        'parent_atoms': 15,
        'iupac_locants': {0: 2, 1: 3, 2: '3a', 3: '9a', 4: 1, 5: 9, 6: '8a', 7: 8, 8: '7a', 9: 7, 10: 6, 11: 5, 12: '4a', 13: 4, 14: '3b'},
    },
    'C1=CC2=c3ccncc3=NC2=C1': {
        'name': 'cyclopenta[4,5]pyrrolo[2,3-c]pyridine',
        'tautomer_locant': None,
        'ring_system': 'polycyclic',
        'parent_atoms': 12,
        'iupac_locants': {0: 6, 1: 5, 2: '4b', 3: '4a', 4: 4, 5: 3, 6: 2, 7: 1, 8: '8a', 9: 8, 10: '7a', 11: 7},
    },
    'c1cc2ccc3ccnc4ccc(c1)c2c34': {
        'name': 'naphtho[2,1,8-def]quinoline',
        'tautomer_locant': None,
        'ring_system': 'polycyclic',
        'parent_atoms': 16,
        'iupac_locants': {0: 7, 1: 6, 2: '5a', 3: 5, 4: 4, 5: '3a', 6: 3, 7: 2, 8: 1, 9: '10a', 10: 10, 11: 9, 12: '8a', 13: 8, 14: '10c', 15: '10b'},
    },
    # M6 Kind-C catalog expansion (2026-08-29): 2-ring fused cores measured
    # ABSENT from this catalog (`fused_component_uncatalogued`, guard
    # spiro.py:1468 / composer.py's plain ortho-fused branch), sized in
    # internal notes + M6-KINDB-CONFIRM.md. The routing
    # to `match_fused_heterocycle_core` already exists and self-activates once
    # these are cataloged. Each name below OPSIN-round-trips (constitution-only
    # InChI match) to exactly this bare core before being added; the
    # `iupac_locants` map is re-anchored FROM the OPSIN name via
    # `validation.opsin_roundtrip.opsin_atom_locant_map` (name -> extendedsmi ->
    # substructure isomorphism onto this canonical SMILES), never hand-derived.
    #
    # 9H-fluorene (bare parent; witness ring system from an N-triflyloxime
    # substituent, e.g. O=S(=O)(ON=C(c1ccc2c(c1)Cc1ccccc1-2)C(F)(F)F)C(F)(F)F).
    'c1ccc2c(c1)Cc1ccccc1-2': {
        'name': '9H-fluorene',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
        'iupac_locants': {5: 1, 0: 2, 1: 3, 2: 4, 3: '4a', 12: '4b', 11: 5, 10: 6, 9: 7, 8: 8, 7: '8a', 6: 9, 4: '9a'},
    },
    # 3,4-dihydro-2H-1,4-benzoxazine (benzomorpholine parent; witness ring
    # system from a spiro-decorated derivative, e.g.
    # CN1CCC(n2cc(-c3ccc4c(c3)OCC3(...)N4)cn2)CC1). OPSIN also accepts the
    # bracket form "3,4-dihydro-2H-benzo[b][1,4]oxazine" for the same
    # structure; this catalog uses the classical-locant spelling to match
    # the sibling entries '3,4-dihydro-2H-1-benzopyran' /
    # '3,4-dihydro-2H-1-benzothiopyran' already in this table.
    'c1ccc2c(c1)NCCO2': {
        'name': '3,4-dihydro-2H-1,4-benzoxazine',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
        'iupac_locants': {9: 1, 8: 2, 7: 3, 6: 4, 4: '4a', 3: '8a', 2: 8, 1: 7, 0: 6, 5: 5},
    },
    # pyrazolo[1,5-a]pyrazine (bridgehead-N 5-6 fusion; witness ring system
    # from Cn1ccc(-c2cn3nccc3c(...)n2)n1). Bridgehead N takes a plain
    # peripheral locant (8) and the carbon fusion atom takes the lettered
    # locant ('3a'), matching the indolizine / imidazo[1,2-a]pyridine
    # convention already in this table.
    'c1cn2nccc2cn1': {
        'name': 'pyrazolo[1,5-a]pyrazine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
        'iupac_locants': {3: 1, 4: 2, 5: 3, 6: '3a', 2: 8, 1: 7, 0: 6, 8: 5, 7: 4},
    },
    # 5,6-dihydro-[1,2,4]triazolo[3,4-b][1,3,4]thiadiazole (partially
    # saturated 5-5 heteroaromatic fusion; witness ring system from
    # [O-][NH+](O)c1ccc([C@@H]2Nn3c(nnc3-c3cccnc3)S2)cc1, a common
    # 6-aryl-5,6-dihydrotriazolothiadiazole scaffold). OPSIN rejects the
    # mirror-bracket "[1,3,4]thiadiazolo[2,3-b][1,2,4]triazole" spelling
    # and the [3,2-b] fusion-locant variant names a DIFFERENT structure
    # (verified non-match) -- only the [3,4-b] form round-trips to this core.
    'c1nnc2n1NCS2': {
        'name': '5,6-dihydro-[1,2,4]triazolo[3,4-b][1,3,4]thiadiazole',
        'tautomer_locant': None,
        'ring_system': 'triazolothiadiazole',
        'parent_atoms': 8,
        'iupac_locants': {2: 1, 1: 2, 0: 3, 4: 4, 3: '7a', 7: 7, 6: 6, 5: 5},
    },
}


# =========================================================================
# PREFIX STEM DERIVATION (a phase)
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
# - Lactones (coumarin): need special "oxo-chromen" prefix, deferred to a phase
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
      stem-locant-yl (e.g., "quinolin-2-yl", "1H-indol-3-yl")

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
# HYG-01 (a phase): connectivity-hash-bucketed ring lookup
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
# * cycle rank: circuit rank is monotonic under subgraph (the cycle space of
# a subgraph is a subspace of the host's) ⇒ rank(pattern) ≤ rank(query).
# * per-element heavy-atom count: each pattern atom maps to a distinct query
# atom of the SAME element ⇒ count(e, pattern) ≤ count(e, query) ∀e
# (this subsumes the heteroatom-set ⊆ condition used for bucketing).
# * total heavy-atom count: ≤, for the same reason.
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
        b.GetIdx() for b in bonds_of(mol)
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
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1') # indole
        >>> get_fused_heterocycle_name(mol)
        ('1H-indole', 1)

        >>> mol = Chem.MolFromSmiles('c1ccc2ncccc2c1') # quinoline
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


def _match_covers_ring_systems(mol, atom_mapping) -> bool:
    """Does the matched catalog core cover every atom of the fused ring
    system(s) it sits in?

    Mirrors ``rules.fused_rings._core_covers_ring_system`` (kept local to avoid a
    data->rules import); ``get_ring_systems`` is imported lazily (perception does
    not import data, so there is no cycle). ``int`` keys in ``atom_mapping`` are
    the matched core atoms; a pendant ring joined by a single (non-ring) bond is a
    SEPARATE ring system, so a legitimate cyclic substituent does not trip this.

    This is the guard that stops a SMALLER catalog pattern from matching a LARGER
    fused ring system as a substructure (e.g. the 4-ring naphthacene pattern
    matching inside 5-ring pentacene, or chrysene inside picene) — a structure-
    loss hallucination that otherwise yields a wrong parent name. 13B(a).
    """
    core_atoms = {k for k in (atom_mapping or {}) if isinstance(k, int)}
    if not core_atoms:
        return True
    from ..perception.rings import get_ring_systems
    for rs in get_ring_systems(mol):
        rs = set(rs)
        if (core_atoms & rs) and not rs.issubset(core_atoms):
            return False
    return True


# RISK-5: elements that can carry an indicated hydrogen (a mancude saturated
# position). Mirrors ``rules.ring_assemblies._INDICATED_H_ATOM_ELEMENTS`` (kept
# local to avoid a data->rules import): B, C, N, Si, P, Ge, As, Sn, Sb, Bi. The
# divalent chalcogens O/S/Se are deliberately EXCLUDED — a ring O/S/Se is always
# two-coordinate and never bears an indicated H, so it must not be read as a
# saturated position.
_FUSED_INDICATED_H_ELEMENTS = frozenset({5, 6, 7, 14, 15, 32, 33, 50, 51, 83})

# Leading indicated-hydrogen descriptor of a catalog core name: ``1H-``,
# ``2H,4H-``, ``10H-``. Captures the whole comma-joined descriptor before the
# first hyphen so a genuinely-stated iH set can be compared to the input's.
_LEADING_INDICATED_H_RE = re.compile(r"^((?:\d+[a-z]*H,)*\d+[a-z]*H)-")


def _input_indicated_h_locants(
    mol: Chem.Mol, atom_mapping: Dict[int, Union[int, str]]
) -> Optional[List[int]]:
    """Integer indicated-hydrogen locants the INPUT actually carries, in the
    matched core's own numbering.

    IUPAC P-25.7.1.3 (the Blue Book): a fused-system PIN must cite indicated hydrogen
    at the ring atom that is saturated where the mancude parent would carry a
    double bond. The catalog's ``atom_mapping`` (an OPSIN-derived atom->locant
    map for ONE reference tautomer) is applied by ``match_fused_heterocycle_core``
    via a substructure match that ignores H position, so a DIFFERENT tautomer of
    the same skeleton (NH on another ring atom) matches and inherits the reference
    tautomer's baked indicated-H locant — a wrong PIN (RISK-5). This returns the
    input's true saturated positions so the caller can detect that mismatch.

    A saturated (indicated-H) position is a ring atom of an iH-capable element
    (``_FUSED_INDICATED_H_ELEMENTS``) that, in the KEKULIZED structure, has NO
    ring double bond and NO exocyclic double bond. Substituted-at-iH atoms (a
    substituent or ring bond in place of the H) are KEPT — ``1-methyl-1H-indole``
    still cites 1H — so a 0-H position is not excluded. An exocyclic double bond
    IS excluded, which keeps an oxo/added-hydrogen position (``9H-fluoren-9-one``,
    ``acridone``) out of the set so those are not mis-flagged as a tautomer swap.

    Only INTEGER locants are returned; a fusion-atom locant (``'3a'``) is skipped,
    so an indicated H at a fusion position is invisible here and the caller stays
    conservative (it does not fire for a name whose iH descriptor carries a letter).

    Returns the sorted integer locants, or ``None`` if the molecule cannot be
    kekulized (caller then leaves the name unchanged).
    """
    core_atoms = {k for k in atom_mapping if isinstance(k, int)}
    try:
        km = Chem.Mol(mol)
        Chem.Kekulize(km, clearAromaticFlags=True)
    except Exception:
        return None
    out: List[int] = []
    for idx in core_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() not in _FUSED_INDICATED_H_ELEMENTS:
            continue
        katom = km.GetAtomWithIdx(idx)
        has_ring_double = any(
            bond.GetOtherAtomIdx(idx) in core_atoms
            and bond.GetBondType() == Chem.BondType.DOUBLE
            for bond in katom.GetBonds()
        )
        if has_ring_double:
            continue
        has_exo_double = any(
            bond.GetOtherAtomIdx(idx) not in core_atoms
            and bond.GetBondType() == Chem.BondType.DOUBLE
            for bond in katom.GetBonds()
        )
        if has_exo_double:
            continue
        loc = atom_mapping.get(idx)
        if isinstance(loc, int):
            out.append(loc)
    return sorted(out)


def _correct_indicated_h_tautomer(
    mol: Chem.Mol, name: str, atom_mapping: Dict[int, Union[int, str]]
) -> Optional[str]:
    """Guard a catalog core hit against the wrong-tautomer defect (RISK-5).

    The substructure matcher accepts a hit by heavy-atom skeleton and returns the
    reference tautomer's baked indicated-H locant. When the INPUT is a different
    tautomer, that baked locant states an indicated hydrogen the molecule does not
    have (``c1[nH]c2sccc2n1`` — the 3H tautomer — matching the 1H thieno[2,3-d]-
    imidazole entry and emitting ``1H-``). P-25.7.1.3 (the Blue Book) requires the
    indicated H at the atom that actually bears it.

    Fires only for a genuine tautomer conflict: the name STATES an integer
    indicated-H set, the input carries a NON-EMPTY set, and the two differ. A
    clean single-locant swap is CORRECTED by re-anchoring the descriptor to the
    input's true iH locant (``1H-`` -> ``3H-``); the map is the authority, so the
    corrected name places the iH at exactly the atom that bears it and round-trips
    0-wrong. Anything less clean (multi-iH mismatch, letter-suffix locants) returns
    ``None`` so the caller falls through to the algorithmic path rather than
    emitting a wrong PIN.

    Returns the (possibly corrected) name, or ``None`` to reject the hit.
    """
    m = _LEADING_INDICATED_H_RE.match(name)
    if not m:
        return name  # name states no indicated H (acridone, quinoline): nothing to verify
    tokens = [t[:-1] for t in m.group(1).split(",")]  # strip trailing 'H'
    if any(not t.isdigit() for t in tokens):
        return name  # letter-suffix iH (e.g. '3aH'): stay conservative, leave unchanged
    name_ih = {int(t) for t in tokens}
    input_ih = _input_indicated_h_locants(mol, atom_mapping)
    if not input_ih:
        return name  # oxo / added-hydrogen consumed the position: keep as-is
    if set(input_ih) == name_ih:
        return name  # correct tautomer
    # Wrong tautomer. Re-anchor a clean single-iH swap to the true locant.
    if len(input_ih) == 1 and len(name_ih) == 1:
        return f"{input_ih[0]}H-{name[m.end():]}"
    return None  # ambiguous multi-iH mismatch: reject, let the caller degrade


def match_fused_heterocycle_core(
    mol: Chem.Mol
) -> Optional[Tuple[str, Dict[int, Union[int, str]], str]]:
    """Match a molecule against the cataloged fused ring-system cores.

    Thin coverage-guarded wrapper over ``_match_fused_heterocycle_core_impl``: a
    catalog match is accepted ONLY if it covers the whole fused ring system it
    touches (P-25.3 — a base/retained ring system name must span the entire fused
    system, not a sub-part). Without this, a larger non-cataloged PAH whose
    skeleton contains a cataloged subset (pentacene contains naphthacene, picene
    contains chrysene) spuriously matched the smaller entry. 13B(a) S1.

     RISK-5: also verifies the input's indicated hydrogen matches the matched
    entry's, correcting a clean single-locant tautomer swap (``1H-`` -> ``3H-``)
    and rejecting an ambiguous mismatch — see ``_correct_indicated_h_tautomer``.
    """
    result = _match_fused_heterocycle_core_impl(mol)
    if result is None:
        return None
    if not _match_covers_ring_systems(mol, result[1]):
        return None
    name, atom_mapping, key = result
    corrected = _correct_indicated_h_tautomer(mol, name, atom_mapping)
    if corrected is None:
        return None
    if corrected != name:
        return (corrected, atom_mapping, key)
    return result


def _match_fused_heterocycle_core_impl(
    mol: Chem.Mol
) -> Optional[Tuple[str, Dict[int, Union[int, str]], str]]:
    """ perf lever — a per-molecule memo over the fused-core matcher.

    Within ONE molecule's naming this matcher is re-run on the SAME ring-bearing
    fragment 100-390x (measured: the recursive substituent enumeration re-derives
    the same fragment thousands of times — vancomycin 1059 core-matcher calls, a
    thiopeptide 1198, and the perf-lever probe found 10-392x redundancy over 1-15
    distinct fragments per molecule). Each call charges the per-molecule
    ``spend_analysis_call`` / ``spend_perf_work`` macrocycle-hang budgets, so the
    redundancy alone exhausts them and abstains a NAMEABLE macrocycle. The cache
    turns a repeat into an O(1) hit that does no work and charges no budget, so the
    molecule finishes under budget and reclaims.

    0-wrong / byte-identity: the key is ``(non-canonical SMILES, atom OUTPUT ORDER)``
    (see the key line below) — the non-canonical SMILES alone does NOT pin the
    index->atom correspondence (a review P3 Crit-1), so the output order is required to
    make the key COMPLETE for the atom-INDEX-keyed ``atom_mapping``. Validated:
    ``ORTHONYM_MEMO=on`` vs ``off`` emits the identical name over 258 PubChem-ORDERED
    fused molecules (atom order != RDKit DFS — the class that exposed the bug), and
    ``ORTHONYM_MEMO=verify`` records **0** ``fused_core`` mismatches over the same set
    via ``memo.verify_mismatch_count()`` (the raised ``MemoMismatch`` is swallowed by
    the naming cascade, so the COUNTER — not the exception — is the real detector,
    Crit-2). A cache MISS or no open scope simply recomputes; a hit can never ship a
    wrong name once the key is complete. Deterministic: the key (string + output
    order) is deterministic for a given input and the cache is per-top-level-call.
    The compute is unchanged below.
    """
    if mol is None:
        return None
    from ..assembly.memo import cache_or_compute
    try:
        _smi = Chem.MolToSmiles(mol, canonical=False)
        # (a review P3 Crit-1): the non-canonical SMILES ALONE does NOT pin the
        # index->atom correspondence -- RDKit's non-canonical writer DFS-walks from
        # atom 0 taking the lowest-index neighbour, so any two labelings that make
        # the same branch choices produce the SAME string with DIFFERENT atom
        # numbers. The result's ``atom_mapping`` is index-keyed, so a bare-string hit
        # would ship the FIRST caller's mapping to a differently-labeled mol -- a
        # WRONG LOCANT (measured: a PubChem-ordered quinazolinone got locant 4 vs 3;
        # the engine presents one molecule as both MolFromSmiles(input) and
        # MolFromSmiles(canonical) within one scope). Pin the labeling with the atom
        # OUTPUT ORDER ``MolToSmiles`` just recorded -- same key construction the
        # M1 substituent memo uses (assembly/substituent_enumerator.py). Same
        # string + same output order == identical indexed graph.
        key = (_smi, mol.GetProp("_smilesAtomOutputOrder"))
    except Exception:
        # An unkeyable mol (should not happen) -> recompute, never cache-corrupt.
        return _match_fused_heterocycle_core_compute(mol)
    return cache_or_compute(
        "fused_core", key,
        lambda: _match_fused_heterocycle_core_compute(mol))


def _match_fused_heterocycle_core_compute(
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
        >>> mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1') # 5-methylindole
        >>> name, mapping, smiles = match_fused_heterocycle_core(mol)
        >>> name
        '1H-indole'
        >>> mapping[1] # mol atom idx 1 (where methyl attaches)
        5 # IUPAC position 5 - correct for 5-methylindole!
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
    q_elem = Counter(a.GetSymbol() for a in atoms_of(mol))
    q_hetset = frozenset(s for s in q_elem if s not in ('C', 'H'))
    q_heavy = mol.GetNumAtoms()

    candidates = _gather_candidates(q_rank, q_hetset)

    # M2.5: charge one ANALYSIS-CALL unit per core-matcher call. This matcher is
    # re-run per ring-bearing fragment throughout the recursive substituent
    # enumeration; a large cyclic glyco-/thio-peptide re-invokes it THOUSANDS of
    # times (measured: vancomycin 1059, thiopeptide 1198 vs chlorophyll 128).
    # The call budget is armed at the outermost name() and shared with the
    # von-Baeyer analyze site, so the call-count explosion class exhausts it.
    spend_analysis_call()

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
                # uniquify=False: a C2v/Cs-symmetric core's automorphic matches
                # share one atom SET, which uniquify=True (default) would collapse
                # to a single match — hiding the alternative numbering from the
                # substituent-locant minimization (the acridine 2-vs-7 bug).
                skel_matches = mol.GetSubstructMatches(skel_rec.pattern, uniquify=False)
                if skel_matches:
                    # E1/DD4: choose the automorphism minimizing substituent
                    # locants (was matches[0] — non-deterministic for symmetric
                    # cores like acridine; byte-identical for asymmetric ones).
                    skel_iul = FUSED_HETEROCYCLE_DATA[skel_rec.smiles].get('iupac_locants') or {}
                    best_skel = _select_lowest_locant_match(mol, skel_matches, skel_iul)
                    return _build_core_result(
                        skel_rec.name, list(best_skel), skel_rec.smiles)

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
        spend_perf_work()  # M2.5: charge each (post-prefilter) substructure match attempt
        if mol.HasSubstructMatch(rec.pattern):
            # uniquify=False so symmetry-equivalent automorphic matches are all
            # available for the substituent-locant minimization (see fast-path).
            matches = mol.GetSubstructMatches(rec.pattern, uniquify=False)
            if matches:
                core_size = rec.parent_atoms

                # Keep the largest matching core
                if best_match is None or core_size > best_match[2]:
                    # E1/DD4: choose the automorphism minimizing substituent
                    # locants (was matches[0]). Byte-identical when only one
                    # match (asymmetric core) or no substituent.
                    rec_iul = FUSED_HETEROCYCLE_DATA[rec.smiles].get('iupac_locants') or {}
                    match = _select_lowest_locant_match(mol, matches, rec_iul)
                    best_match = (rec.name, list(match), core_size, rec.smiles)

    if best_match is None:
        return None

    name, match_atoms, _, core_smiles = best_match
    return _build_core_result(name, match_atoms, core_smiles)


def _coerce_locant_for_compare(locant):
    """Coerce a catalog locant ('4a', 3,...) to the int / (int, str) form that
    ``compare_locant_sets`` orders. Returns None for an unparseable value."""
    import re as _re
    if isinstance(locant, int):
        return locant
    m = _re.match(r'^(\d+)([a-z]*)$', str(locant))
    if m is None:
        return None
    base, suffix = int(m.group(1)), m.group(2)
    return (base, suffix) if suffix else base


def _loc_to_float(coerced) -> float:
    """Normalize a coerced locant (int, or (base, suffix) tuple from
    _coerce_locant_for_compare) to a single sortable float so the P-14.4(g)
    alpha-tiebreak tuple is uniformly comparable across automorphic matches."""
    if isinstance(coerced, tuple):
        base, suffix = coerced
        return float(base) + (ord(suffix[0]) - 96) * 0.01 if suffix else float(base)
    return float(coerced)


def _match_substituent_alpha_key(mol, match, core_atoms, sub_atoms, iupac_locants):
    """P-14.4(g) / P-14.5.2 alpha tiebreak key for one automorphic match.

    For each substituent-bearing core atom, name its exocyclic substituent
    fragment and pair the name with the atom's locant in THIS match's
    orientation; return ``(alpha_sort_key(name), locant)`` pairs sorted by name
    and flattened. Lexicographic comparison of two matches' keys then gives the
    alphabetically-first substituent the lowest locant -- the deterministic PIN
    choice when the substituent LOCANT SET ties between mirror orientations
    (e.g. 4-bromo-6-methyldibenzofuran, NOT 6-bromo-4-methyl). Mirrors
    rules.locants._p45_alpha_key. Fail-soft: an unnameable fragment falls back to
    a carbon-count / element-symbol proxy (which still orders halo<alkyl etc.)."""
    from ..assembly.naming_utils import alpha_sort_key, get_alkyl_name

    entries = []
    for pattern_idx, mol_idx in enumerate(match):
        if mol_idx not in sub_atoms:
            continue
        coerced = _coerce_locant_for_compare(iupac_locants.get(pattern_idx))
        if coerced is None:
            continue
        # Gather the exocyclic substituent fragment hanging off this core atom.
        frag, seen, stack = [], set(core_atoms), [
            nb.GetIdx() for nb in mol.GetAtomWithIdx(mol_idx).GetNeighbors()
            if nb.GetIdx() not in core_atoms
        ]
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            frag.append(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                if nb.GetIdx() not in seen:
                    stack.append(nb.GetIdx())
        name = None
        if frag and any(mol.GetAtomWithIdx(a).IsInRing() for a in frag):
            try:
                attach = next(
                    (a for a in frag
                     if any(nbr.GetIdx() == mol_idx
                            for nbr in mol.GetAtomWithIdx(a).GetNeighbors())),
                    None,
                )
                if attach is not None:
                    from ..assembly.substituent_enumerator import name_substituent
                    name = name_substituent(mol, list(frag), attach)
            except Exception:
                name = None
        if not name:
            ccount = sum(1 for a in frag if mol.GetAtomWithIdx(a).GetSymbol() == 'C')
            if ccount > 0:
                name = get_alkyl_name(ccount)
            elif frag:
                name = mol.GetAtomWithIdx(frag[0]).GetSymbol().lower()
            else:
                name = 'zzz'
        entries.append((alpha_sort_key(name or 'zzz'), _loc_to_float(coerced)))
    entries.sort()
    return tuple(item for pair in entries for item in pair)


def _select_lowest_locant_match(mol, matches, iupac_locants):
    """Pick the automorphic substructure match giving the substituent-bearing
    core atoms the lowest locants (P-14.3.5 / P-14.4 / P-25.3.3.1.2(a)).

     Phase E1 / DD4. For an asymmetric core there is exactly one match, so
    this returns ``matches[0]`` (byte-identical to the legacy first-match
    behaviour). For a C2v/Cs-symmetric core (acridine, carbazole,
    phenanthridine,...) with a substituent, the multiple automorphic matches
    map the substituted atom onto its symmetry orbit; choosing the match that
    minimizes the substituent locant set yields the deterministic PIN
    numbering instead of an input-order-dependent first match.

    Bare (unsubstituted) symmetric cores tie on the empty substituent set and
    return ``matches[0]`` — their output name carries no locants, so the choice
    is immaterial and byte-identical.
    """
    matches = [list(m) for m in matches]
    if not matches:                       # IN-04: self-defensive (callers guard too)
        return []
    if len(matches) <= 1:
        return matches[0]
    core_atoms = set(matches[0])
    sub_atoms = {
        idx for idx in core_atoms
        if any(nb.GetIdx() not in core_atoms
               for nb in mol.GetAtomWithIdx(idx).GetNeighbors())
    }
    if not sub_atoms or not iupac_locants:
        return matches[0]

    from ..rules.locants import compare_numbering

    best, best_cand = None, None
    for m in matches:
        locs = []
        scorable = True
        for pattern_idx, mol_idx in enumerate(m):
            if mol_idx in sub_atoms:
                coerced = _coerce_locant_for_compare(iupac_locants.get(pattern_idx))
                if coerced is None:
                    # WR-02 fail-closed: a substituent atom whose catalog locant
                    # is non-numeric (exocyclic '=O'/'N6'/... — latent today)
                    # cannot be scored. Excluding the whole match is correct;
                    # silently dropping the locant would shorten the set and let
                    # "shorter set wins" pick a wrong numbering.
                    scorable = False
                    break
                locs.append(coerced)
        if not scorable:
            continue
        # P-14.4(f) lowest substituent-locant SET, then P-14.4(g)/P-14.3.5 lowest
        # locant to the alphabetically-first substituent (the alpha tier breaks a
        # mirror-orientation tie deterministically AND PIN-correctly, e.g.
        # 4-bromo-6-methyldibenzofuran rather than the input-order-dependent
        # 6-bromo-4-methyl).
        cand = {'substituents': locs,
                'alpha': _match_substituent_alpha_key(mol, m, core_atoms, sub_atoms, iupac_locants)}
        if best is None or compare_numbering(cand, best_cand) < 0:
            best, best_cand = m, cand
    # If every match was unscorable, fall back to the first (byte-identical to
    # the legacy first-match rather than crashing).
    return best if best is not None else matches[0]


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
