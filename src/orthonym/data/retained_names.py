"""
Retained (trivial) names that are preferred over systematic names.

These names are recognized by IUPAC as Preferred IUPAC Names (PINs).
ALWAYS check this lookup before applying systematic naming rules!

Keys are canonical SMILES, values are retained names.
"""

from typing import Optional

# Common retained names (canonical SMILES -> name)
# These take precedence over systematic names
RETAINED_NAMES = {
    # === SIMPLE ALKANES ===
    "C": "methane",
    "CC": "ethane",
    "CCC": "propane",
    "CCCC": "butane",

    # === AROMATIC HYDROCARBONS ===
    "c1ccccc1": "benzene",
    "Cc1ccccc1": "toluene",
    "CCc1ccccc1": "ethylbenzene",
    "C=Cc1ccccc1": "styrene",  # ethenylbenzene
    "CC(C)c1ccccc1": "cumene",  # isopropylbenzene
    # NOTE: xylene isomers are NOT retained names in IUPAC 2013 PIN
    # Use systematic: 1,2-dimethylbenzene, 1,3-dimethylbenzene, 1,4-dimethylbenzene
    "c1ccc2ccccc2c1": "naphthalene",
    # a phase: pentalene (PIN per Blue Book line 11459; bicyclo[3.3.0] fully
    # mancude non-benzenoid). HEAD dropped one ring -> 'cyclopenta-1,3-diene'
    # (-suppressed to 'unknown'); the retained name keeps both rings.
    # OPSIN-RT verified ('pentalene' -> same molecule).
    "C1=CC2=CC=CC2=C1": "pentalene",
    # / (the Blue Book "pentalene (PIN) octalene (PIN)"):
    # the mancude bicyclo[6.6.0] all-carbon fused system is the retained PIN
    # 'octalene', not a von Baeyer name. Exact-SMILES retained parent.
    "c1cccc2ccccccc-2cc1": "octalene",
    "c1cc2ccc3cccc4ccc(c1)c2c34": "pyrene",
    "c1ccc2cc3ccccc3cc2c1": "anthracene",
    "c1ccc2c(c1)ccc1ccccc12": "phenanthrene",
    "c1ccc2c(c1)Cc1ccccc1-2": "fluorene",  # Has sp3 carbon (methylene bridge)
    "c1cc2c3c(cccc3c1)CC2": "acenaphthene",  # Has two sp3 carbons
    "C1=Cc2cccc3cccc1c23": "acenaphthylene",  # Fully aromatic
    "c1ccc2c(c1)ccc1c3ccccc3ccc21": "chrysene",
    "c1cc2cccc3c4cccc5cccc(c(c1)c23)c54": "perylene",
    "c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1": "pentacene",

    # === SIMPLE ALCOHOLS ===
    "CO": "methanol",
    "CCO": "ethanol",
    "OCC(O)CO": "glycerol",

    # === CARBOXYLIC ACIDS ===
    "O=CO": "formic acid",  # Canonical SMILES for HCOOH
    "CC(=O)O": "acetic acid",
    "CCC(=O)O": "propanoic acid",  # propionic acid is also accepted
    "CCCC(=O)O": "butanoic acid",  # butyric acid is also accepted
    # Task 1.10 (PIN-policy): canonicalized key fix. The old non-canonical
    # key OC(=O)C(O)=O never matched (every input canonicalizes to
    # O=C(O)C(=O)O), so the canonical SMILES resolved to the OPSIN-import
    # 'dihydroxalate' instead. Re-key to canonical + deny 'dihydroxalate' in
    # iupac_2013_pin_list.json so this retained PIN wins the merge.
    "O=C(O)C(=O)O": "oxalic acid",
    # Task 1.10 (PIN-policy): oxamic acid is a retained PIN with no
    # prior key (systematic engine emits '1-carbamoylmethanoic acid' otherwise).
    "NC(=O)C(=O)O": "oxamic acid",
    "OC(=O)CC(=O)O": "malonic acid",
    "OC(=O)CCC(=O)O": "succinic acid",
    "OC(=O)CCCC(=O)O": "glutaric acid",
    "OC(=O)CCCCC(=O)O": "adipic acid",
    "O=C(O)c1ccccc1": "benzoic acid",  # Canonical SMILES (was OC(=O)c1ccccc1)
    "OC(=O)CC(O)(CC(=O)O)C(=O)O": "citric acid",

    # === ALDEHYDES ===
    "C=O": "formaldehyde",
    "CC=O": "acetaldehyde",
    "O=Cc1ccccc1": "benzaldehyde",

    # === KETONES ===
    # NOTE: acetone removed -- IUPAC 2013 PIN is "propan-2-one"
    #: `chalcone` is the ONE retained ketone name that is a PIN.
    # (the Blue Book), under `### **** Retained names`:
    # "The name 'chalcone' is the only retained name as a preferred IUPAC name
    # and is limited to ring substitution only by characteristic groups lower
    # than 'ketone'. Chalcone refers only to the trans- or (E)- stereoisomer."
    # The (PIN) line follows at:28299 -- `chalcone (PIN) (2E)-1,3-diphenyl-
    # prop-2-en-1-one`. (:28307) then closes the GENERAL-only ketone
    # list and ends "Substitutive names, systematically constructed, are the
    # preferred IUPAC names for ketones" -- which is why acetophenone and
    # benzophenone below are `_PIN_DENY_HC`-demoted to general-only while this
    # row is not.
    #
    # The key is the ISOMERIC canonical SMILES (`_name_impl` builds its lookup
    # key with `Chem.MolToSmiles(mol, canonical=True)`, isomeric by default), so
    # this single row enforces ALL THREE limits the rule imposes, exactly rather
    # than approximately:
    # (1) (E) only -- the (Z) isomer canonicalises to
    # `O=C(/C=C\c1ccccc1)c1ccccc1` and the stereo-
    # UNSPECIFIED molecule to `O=C(C=Cc1ccccc1)c1ccccc1`;
    # three distinct keys, so neither can match.
    # (2) ring substitution only, by groups junior to ketone, and
    # (3) no chain substitution
    # -- ANY substituent changes the whole-molecule key, so
    # the BB's own worked counter-example
    # `2',4'-dihydroxychalcone-4-carboxamide` (:28305,
    # "(not...)"; also in bluebook_not_names.py:171) is
    # structurally unreachable from here.
    #
    # SCOPE: substituted chalcones that ARE PINs (e.g. `2',4'-dihydroxy-3,3'-
    # dimethoxychalcone`,:28303) are deliberately NOT emitted -- they need
    # primed-locant machinery across two rings plus a ketone-seniority test, and
    # a wrong construction would ship a WRONG name where falling through ships
    # the valid systematic one. Fail-open to systematic is the safe direction.
    #
    # NB the OPSIN import surface already carries this exact SMILES key as the
    # STEM `chalcon` (no trailing 'e'); it never promoted because
    # `_is_complete_name('chalcon')` is False. Hand-curated wins the merge
    # (`ALL_RETAINED_NAMES = {**_OPSIN_NAMES, **_HAND_CURATED_GATED}`), so this
    # row is what ships. OPSIN 2.9.0 parses `chalcone` -> InChIKey
    # an InChIKey, identical to the input's (round-trip verified).
    "O=C(/C=C/c1ccccc1)c1ccccc1": "chalcone",
    "CC(=O)c1ccccc1": "acetophenone",
    "O=C(c1ccccc1)c1ccccc1": "benzophenone",

    # === AMINES ===
    # /: methylamine / ethylamine / trimethylamine /
    # triethylamine are GENERAL-nomenclature functional-class names, NOT PINs.
    # The PINs are the substitutive forms (methanamine, ethanamine,
    # N,N-dimethylmethanamine, N,N-diethylethanamine), which the systematic
    # path already produces once these retained entries are absent. Removed from
    # the PIN-headline path (DD1 Fix 4 / H5). 'aniline' IS a retained PIN
    # and stays.
    "Nc1ccccc1": "aniline",

    # === PHENOLS ===
    "Oc1ccccc1": "phenol",
    # NOTE: cresol isomers removed -- IUPAC 2013 PINs are "2-methylphenol",
    # "3-methylphenol", "4-methylphenol" (produced by systematic naming pipeline)
    "Oc1ccc(O)cc1": "hydroquinone",
    "Oc1cccc(O)c1": "resorcinol",
    "Oc1ccccc1O": "catechol",

    # === 5-MEMBERED AROMATIC HETEROCYCLES ===
    "c1ccoc1": "furan",
    "c1ccsc1": "thiophene",
    # Azoles carry a leading indicated hydrogen in the PIN: the
    # NH is the indicated-H position, cited as 1H-. Each SMILES key fixes a
    # specific tautomer, so the indicated-H locant is determined per key
    # (all OPSIN-2.9.0 round-trip verified,).
    "c1cc[nH]c1": "1H-pyrrole",
    "c1c[nH]cn1": "1H-imidazole",
    "c1cnc[nH]1": "1H-imidazole",
    "c1cn[nH]c1": "1H-pyrazole",   # Canonical SMILES for pyrazole
    "c1cc[nH]n1": "1H-pyrazole",   # Alternate input form
    "c1cocn1": "1,3-oxazole",   # HW-locant PIN /; 'oxazole' is general-only
    "c1cnco1": "1,3-oxazole",   # Alternate input form
    "c1cnoc1": "1,2-oxazole",   # PIN (was 'isoxazole')
    "c1ccno1": "1,2-oxazole",   # Alternate input form
    "c1cscn1": "1,3-thiazole",  # PIN (was 'thiazole')
    "c1cncs1": "1,3-thiazole",  # Alternate input form
    "c1cnsc1": "1,2-thiazole",  # PIN (was 'isothiazole')
    "c1ccsn1": "1,2-thiazole",  # Alternate input form
    "c1nnn[nH]1": "1H-tetrazole",  # this tautomer = 1H- (OPSIN-RT verified)
    # The OPSIN import surface carries this key as 's-triazole', a general
    # (CAS-style) name that is NOT a PIN, and it was live: `c1nc[nH]n1` emitted
    # `s-triazole`, and once the N-substituted form began resolving here it
    # assembled as the malformed `1-methyls-triazole`. The PIN is the HW name
    # plus its indicated hydrogen: (`the Blue Book`) "in a
    # preferred IUPAC name a locant and the symbol 'H' must be cited", over the
    # HW name the Blue Book's own glossary spells out at `:1832`
    # ("for example 1,2,4-triazole and 1,2-oxazole"); `:42460` prints the parent
    # inside a PIN verbatim -- `N,1,4-triphenyl-1H-1,2,4-triazol-4-ium-3-aminide
    # (PIN)`. Hand-curated wins the merge, so this row is what ships.
    "c1nc[nH]n1": "1H-1,2,4-triazole",
    # 5-membered heteroarenes with THREE heteroatoms (>=1 N): Hantzsch-Widman
    # PINs; furazan is general-only, PIN = 1,2,5-oxadiazole per
    # / the Blue Book). No indicated H (all ring N are
    # pyridine-type). Each key is canonical(OPSIN(name)) => bare name AND the
    # free-valence substituent locant round-trip (T2, OPSIN-RT verified).
    "c1ncon1": "1,2,4-oxadiazole",
    "c1nnco1": "1,3,4-oxadiazole",
    "c1cnon1": "1,2,5-oxadiazole",
    "c1csnn1": "1,2,3-thiadiazole",
    "c1ncsn1": "1,2,4-thiadiazole",
    "c1cnsn1": "1,2,5-thiadiazole",
    "c1nncs1": "1,3,4-thiadiazole",

    # === 6-MEMBERED AROMATIC HETEROCYCLES ===
    "c1ccncc1": "pyridine",
    "c1ccnnc1": "pyridazine",
    "c1cncnc1": "pyrimidine",
    "c1cnccn1": "pyrazine",
    "c1cnncn1": "1,2,4-triazine",
    "c1ncncn1": "1,3,5-triazine",

    # === FUSED HETEROCYCLES ===
    # IUPAC 2013 PIN includes tautomer locant (indicated hydrogen) where applicable
    "c1ccc2ncccc2c1": "quinoline",
    "c1ccc2cnccc2c1": "isoquinoline",
    "c1ccc2[nH]ccc2c1": "1H-indole",  # 1H-indole is IUPAC 2013 PIN
    # === GROUP-15 (As/P) FUSED RING RETAINED PARENTS ===
    # the Blue Book "arsenic or phosphorus atoms replace nitrogen atoms";
    # the arsenic/phosphorus analogues of indole/isoindole/indolizine/quinolizine
    # are retained PINs spelled with the terminal 'e' (the Blue Book "arsindole (PIN)
    # phosphindole (PIN)", the Blue Book "isoarsindole (PIN) isophosphindole (PIN)",
    # the Blue Book "arsindolizine (PIN) phosphindolizine (PIN)", the Blue Book
    # "phosphinolizine (PIN)"). The OPSIN aryl-group import supplies the stem
    # WITHOUT the 'e' (arsindol/phosphindol/...); these hand-curated entries win
    # the ALL_RETAINED_NAMES merge (: hand-curated wins) and fix the spelling.
    # The indolizine/quinolizine analogues have no OPSIN whole-molecule name, so
    # the engine otherwise emits a von Baeyer name -- these add the retained PIN.
    "C1=Cc2ccccc2[AsH]1": "arsindole",         # the Blue Book
    "C1=c2ccccc2=C[AsH]1": "isoarsindole",      # the Blue Book
    "c1ccc2[pH]ccc2c1": "phosphindole",         # the Blue Book
    "c1ccc2c[pH]cc2c1": "isophosphindole",      # the Blue Book
    "C1=CC2=CC=C[As]2C=C1": "arsindolizine",    # the Blue Book (5+6 indolizine-type, fully mancude: NO indicated H)
    "c1ccp2cccc2c1": "phosphindolizine",        # the Blue Book (5+6 indolizine-type, fully mancude: NO indicated H)
    # phosphinolizine is the 6+6 QUINOLIZINE analogue (Table 2.9 row 15), so it
    # carries indicated hydrogen exactly like its N parent: the Blue Book "the PIN is
    # 4H-quinolizine"; (the Blue Book) + (the Blue Book) require the
    # indicated H to be cited. The bare "phosphinolizine (PIN)" in Table 2.9 is
    # shorthand (as "quinolizine (PIN)" is). OPSIN parses 4H-phosphinolizine to the
    # identical structure (RT-verified). a review.
    "C1=CCP2C=CC=CC2=C1": "4H-phosphinolizine",  # the Blue Book / the Blue Book (6+6 quinolizine analogue)
    "c1ccc2[nH]cnc2c1": "1H-benzimidazole",  # 1H-benzimidazole is IUPAC 2013 PIN
    # (the Blue Book) "for preferred IUPAC names locants must be
    # cited": the (PIN) example prints "1-benzofuran (PIN) benzofuran" -- the bare
    # stem is a retained general name, the PIN carries the O locant. The FUSED
    # catalog already spells these '1-*', but this hand-curated retained entry wins
    # the ALL_RETAINED_NAMES merge, so the bare-parent PIN is fixed here.
    "c1ccc2occc2c1": "1-benzofuran",
    # the Blue Book "(2,1-benzothiazole is senior to 1-benzothiophene)"
    # confirms the PIN stem is '1-benzothiophene' locant citation).
    "c1ccc2sccc2c1": "1-benzothiophene",
    "c1cnc2ccccc2n1": "quinazoline",
    "c1ccc2nccnc2c1": "quinoxaline",
    "c1ncnc2[nH]cnc12": "7H-purine",  # 7H-purine is IUPAC 2013 PIN

    # === RETAINED TRICYCLIC RING PARENTS / ===
    # As/Sb/P/Se fused-ring heterocycles are organic ring nomenclature and
    # ARE IN SCOPE (out-of-scope is organometallics only). Bare parents; the
    # name-only ALL_RETAINED_NAMES path preempts the errors.py "<metal> compound
    # (not supported)" last-resort fallback (trace-verified). OPSIN 2.9.0 round-trips
    # every one to the input InChIKey. Keys are RDKit-canonical.
    # (heteroatom analogues of acridine/phenanthridine):
    "C1=c2ccccc2=c2ccccc2=[As]1": "arsanthridine",     # the Blue Book arsanthridine (PIN)
    "C1=c2ccccc2=[As]c2ccccc21": "acridarsine",        # the Blue Book acridarsine (PIN)
    "c1ccc2pc3ccccc3cc2c1": "acridophosphine",         # the Blue Book acridophosphine (PIN)
    # (phenoxa-/phenothia- C4OX-C6-C6 family, the Blue Book table):
    # phenoxathiine's bare parent is emitted from ALL_RETAINED_NAMES, where the
    # OPSIN import (_OPSIN_NAMES) carries the non-PIN 'phenoxathiin' (trace-verified
    # winning source); this hand-curated entry overrides it with the PIN
    # 'phenoxathiine' (the Blue Book "X = S phenoxathiine (PIN)"). The FUSED
    # catalog value is fixed in lockstep for the substituted/fused path.
    "c1ccc2c(c1)Oc1ccccc1S2": "phenoxathiine",         # the Blue Book phenoxathiine (PIN)
    "c1ccc2c(c1)Oc1ccccc1[Se]2": "phenoxaselenine",    # the Blue Book phenoxaselenine (PIN)
    # The 4 X-H (P/As/Sb) tricyclics carry a saturated X-H at ring position 10, but the
    # BB prints these as "<name> (PIN, 10H-isomer shown)" (the Blue Book) —
    # "10H-isomer shown" says which drawn isomer the PIN denotes, it is NOT part of the
    # PIN string. Contrast the isostructural N-cases, which spell it out explicitly as
    # "the PIN is 10H-phenoxazine" / "the PIN is 10H-phenothiazine" (the Blue Book) — a
    # different phrasing for a different fact. So the P/As/Sb PINs are the bare parent
    # names, no 10H- prefix. (phenoxaselenine/phenoxathiine have divalent Se/S = no H.)
    "c1ccc2c(c1)Oc1ccccc1P2": "phenoxaphosphinine",    # the Blue Book phenoxaphosphinine (PIN, 10H-isomer shown)
    "c1ccc2c(c1)Oc1ccccc1[AsH]2": "phenoxarsinine",    # the Blue Book phenoxarsinine (PIN, 10H-isomer shown)
    "c1cc[c]2c(c1)Oc1cccc[c]1[SbH]2": "phenoxastibinine",  # the Blue Book phenoxastibinine (PIN, 10H-isomer shown)
    "c1ccc2c(c1)Sc1ccccc1[AsH]2": "phenothiarsinine",  # the Blue Book phenothiarsinine (PIN, 10H-isomer shown)

    # === UNSATURATED 6-MEMBERED O-HETEROCYCLES (pyrans) ===
    # IUPAC 2013 prefers "2H-pyran" / "4H-pyran" over HW systematic "oxine"
    # OPSIN does not recognize "oxine"; these retained names ensure compatibility
    "C1=CCOC=C1": "2H-pyran",            # 2H-pyran (two C=C bonds)
    "C1=COC=CC1": "4H-pyran",            # 4H-pyran (two C=C bonds)
    "C1=COCCC1": "3,4-dihydro-2H-pyran", # dihydropyran (one C=C bond)
    "C1=CCOCC1": "3,6-dihydro-2H-pyran", # dihydropyran (one C=C bond)

    # === SATURATED HETEROCYCLES ===
    "C1CO1": "oxirane",
    "C1CN1": "aziridine",
    "C1CS1": "thiirane",
    "C1COC1": "oxetane",
    "C1CNC1": "azetidine",
    "C1CSC1": "thietane",
    "C1CCOC1": "oxolane",
    "C1CCNC1": "pyrrolidine",
    "C1CCSC1": "thiolane",
    "C1CCOCC1": "oxane",
    "C1CCNCC1": "piperidine",
    "C1CCSCC1": "thiane",
    "C1COCCN1": "morpholine",
    "C1CNCCN1": "piperazine",
    # PA1 sweep: the row `"O=C1CNCC(=O)N1": "piperazine-2,5-dione"` was DELETED.
    # WRONG STRUCTURE: in that SMILES both carbonyls sit adjacent to the SAME ring
    # nitrogen, i.e. it is piperazine-2,6-dione (InChIKey CYJAWBVQRMVFEO).
    # Piperazine-2,5-dione (glycine anhydride) is O=C1CNC(=O)CN1
    # (BXRNXXXXHLBUKK), verified by independent hand-construction from the ring
    # definition. It WAS live: O=C1CNCC(=O)N1 emitted "piperazine-2,5-dione".
    # The true 2,5-dione is named correctly by another path and is unaffected.

    # === CYCLOALKANES ===
    "C1CC1": "cyclopropane",
    "C1CCC1": "cyclobutane",
    "C1CCCC1": "cyclopentane",
    "C1CCCCC1": "cyclohexane",
    "C1CCCCCC1": "cycloheptane",
    "C1CCCCCCC1": "cyclooctane",

    # === AMIDES ===
    "NC=O": "formamide",
    "CC(=O)N": "acetamide",
    "CC(N)=O": "acetamide",  # canonical form of CC(=O)N

    # === HYDRAZIDES: retained acyl-stem + hydrazide) ===
    # C1: formic acid -> formohydrazide (H-C(=O)-NH-NH2)
    "NNC=O": "formohydrazide",
    # C2: acetic acid -> acetohydrazide (CH3-C(=O)-NH-NH2)
    "CC(=O)NN": "acetohydrazide",
    # Aromatic: benzoic acid -> benzohydrazide (C6H5-C(=O)-NH-NH2)
    "NNC(=O)c1ccccc1": "benzohydrazide",
    # C2 diacid: oxalic acid -> oxalohydrazide (H2N-NH-CO-CO-NH-NH2). One of the
    # five retained hydrazide PINs, the Blue Book "acetohydrazide (PIN)
    # benzohydrazide (PIN) oxalohydrazide (PIN)"); the systematic form is
    # 'ethanedihydrazide'. Oxalic acid has no substitutable position, so only the
    # bare molecule is in scope -> exact-SMILES retained key.
    "NNC(=O)C(=O)NN": "oxalohydrazide",

    # === CARBONIC / DIAZENE HYDRAZIDES (W3-P01: exact-SMILES retained PINs) ===
    # These carbonic-acid derivatives have PINs the substitutive engine cannot
    # derive; the OPSIN-import trivials (is_pin=False) are denied in
    # iupac_2013_pin_list.json and routed to --trivial. Phosgene model.
    # W3-P01-4: H2N-NH-C(=O)-NH-NH2. PIN hydrazinecarbohydrazide
    # (the Blue Book); the Blue Book 'the names carbonohydrazide, carbohydrazide, and
    # carbazide are not recommended'. Canonical key of NNC(=O)NN.
    "NNC(=O)NN": "hydrazinecarbohydrazide",
    # W3-P01-5: HN=N-C(=O)-NH-NH2. PIN diazenecarbohydrazide
    # (the Blue Book); the Blue Book 'The name carbazone is not recommended'. Canonical
    # key of N=NC(=O)NN.
    "N=NC(=O)NN": "diazenecarbohydrazide",
    # W3-P01-6: HN=N-C(=O)-N=NH. PIN bis(diazenyl)methanone
    # (the Blue Book '...(PIN) [not 1,1'-carbonylbis(diazene)]'); the -one suffix is
    # senior to the diazene parent. REPRODUCE-FIRST: the substitutive
    # engine fails closed (unknown) on this symmetric diazenyl ketone, so the
    # exact-SMILES retained PIN supplies it; trivial 'carbodiazone' denied in
    # iupac_2013_pin_list.json (--trivial). Canonical key of N=NC(=O)N=N.
    "N=NC(=O)N=N": "bis(diazenyl)methanone",

    # === PA1 a lever/a lever: PIN word-forms for the deprecated-trivial replacements
    # (phosgene model). Each trivial name below is denied in
    # iupac_2013_pin_list.json with its verbatim Blue Book citation. REPRODUCE-
    # FIRST FINDING that made these entries mandatory rather than optional:
    # denying the trivial name ALONE does not fail closed here -- it unmasks
    # pre-existing generators that emit a WRONG name, several of which silently
    # DROP atoms. Measured, deny-only, before these rows were added:
    # N=C=N -> "methane" (loses N2!)
    # [C-]#[N+]O -> "methane" (loses N,O!)
    # NC(=S)SC(N)=S -> "methanamine" (loses most of it!)
    # OCC(CO)(CO)CO -> "2,2-di(hydroxymethyl)propane-1,2,3-triol"
    # (a 5-OH structure)
    # CC(=O)N(C(C)=O)C(C)=O -> "N-acetyl-1-acetamidoethan-1-imide"
    # N=C(N)NC(=N)NC(=N)N -> "N-guanidinomethaniminylguanidine"
    # N=C(N)S -> "sulfanylmethanimidamide" (right structure,
    # not the PIN)
    # NC(=S)SSC(N)=S -> "1-aminosulfanylidenemethyldisulfanyl
    # disulfanediylmethanamine"
    # So a deny-only change would have replaced 8 deprecated names with 8 names
    # that are worse. Supplying the PIN word-form is the root-cause fix; the
    # underlying generator defects are logged separately as their own work items.
    # (fulminic acid, diacetamide and pinacol are NOT listed here -- their
    # systematic engines already derive the correct PIN unaided, verified.)
    #
    # / -- HO-N=C:; the Blue Book 'N-hydroxy-λ2-methanamine (PIN)'
    # for the structure drawn at the Blue Book. The trivial 'isofulminic acid' is
    # state (c) NOT ACCEPTABLE (the Blue Book). The λ-superscript is rendered with a
    # plain digit exactly as the on-disk Blue Book renders it in running text
    # ('λ2 -methylidenehydroxylamine'); this is the first λ descriptor in an
    # emitted name, and validation/name_morphemes.py already recognises both
    # 'lambda' and 'λ'. InChI normalises the charge-separated [C-]#[N+]O and the
    # neutral [C]=NO to one key (OTXBWGUYZNKPMG), so this key IS the BB's HO-N=C:.
    "[C-]#[N+]O": "N-hydroxy-λ2-methanamine",
    # -- the Blue Book 'carbamimidothioic acid (PIN)' / '(not
    # isothiourea)'; the Blue Book 'H2N-C(=NH)-SH carbamimidothioic acid (PIN)'.
    "N=C(N)S": "carbamimidothioic acid",
    # -- the Blue Book "1,2,3-trithiodicarbonic diamide (PIN) (not
    # 'thiuram monosulfide')".
    "NC(=S)SC(N)=S": "1,2,3-trithiodicarbonic diamide",
    # -- the Blue Book "2-dithioperoxy-1,3-dithiodicarbonic diamide (PIN)
    # (not 'thiuram disulfide')".
    "NC(=S)SSC(N)=S": "2-dithioperoxy-1,3-dithiodicarbonic diamide",
    # -- the Blue Book 'The names biguanide, triguanide, etc., are no
    # longer recommended... named systematically as the diamides of
    # imidodicarbonimidic acid, diimidotricarbonimidic acid, and
    # triimidotetracarbonimidic acid.' triguanide is n=3. The sibling n=2
    # (N=C(N)NC(=N)N) already emits 'imidodicarbonimidic diamide' unaided, which
    # is why only the n=3 homologue needs a word-form here.
    "N=C(N)NC(=N)NC(=N)N": "diimidotricarbonimidic diamide",
    # -- the Blue Book condemns 'triacetamide'; the (R-CO)3N shape is named
    # N,N-diacyl + parent carboxamide per the Blue Book, and the Blue Book
    # 'N-acetyl-N-cyclopentylacetamide (PIN)' fixes the acetyl skeleton.
    "CC(=O)N(C(C)=O)C(C)=O": "N,N-diacetylacetamide",
    # -- the Blue Book 'pentaerythritol / 2,2-bis(hydroxymethyl)propane-
    # 1,3-diol (PIN)'. Note the deny-only fallback named it a 1,2,3-TRIOL, i.e. a
    # different structure, so this row is a correctness fix and not cosmetic.
    "OCC(CO)(CO)CO": "2,2-bis(hydroxymethyl)propane-1,3-diol",
    # -- the Blue Book 'The name carbodiimide, for HN=C=NH, is retained but
    # only for general nomenclature... The systematic name, methanediimine, is
    # the preferred IUPAC name.'
    "N=C=N": "methanediimine",

    # === COMMON SOLVENTS AND REAGENTS ===
    "ClCCl": "dichloromethane",
    "ClC(Cl)Cl": "chloroform",
    "ClC(Cl)(Cl)Cl": "carbon tetrachloride",
    # Task 1.10 (PIN-policy): carbonyl dichloride IS the PIN for
    # phosgene (functional-class name the substitutive engine cannot derive;
    # it goes to unknown otherwise). Canonical key of ClC(=O)Cl. The trivial
    # 'phosgene' (OPSIN import) is denied in iupac_2013_pin_list.json.
    "O=C(Cl)Cl": "carbonyl dichloride",
    # (Wave-2 completion C): functional-class IS the PIN for
    # carbonic-acid nitriles; the engine cannot derive these word-forms.
    "N#CC(=O)C#N": "carbonyl dicyanide",
    # W3-P01-2: methaneperoxoic acid IS the PIN for HC(=O)-O-OH
    # (the Blue Book '...(PIN) peroxyformic acid performic acid'). Unlike the C>=2
    # case (CCC(=O)OO->propaneperoxoic acid works via the systematic peroxoic
    # suffix), the C1 formic edge case is mis-perceived as an aldehyde and the
    # engine emits the wrong '1-hydroperoxymethanal'. Exact-SMILES retained PIN
    # (phosgene model); the trivial 'performic acid' is denied in
    # iupac_2013_pin_list.json (--trivial fallback). Canonical key of O=COO.
    "O=COO": "methaneperoxoic acid",
    # W3-P01-3: carbononitridic chloride IS the PIN for N#C-Cl
    # (the Blue Book 'NC-Cl carbononitridic chloride (PIN) cyanic chloride';
    # the Blue Book 'Method (1) generates preferred IUPAC names'). The substitutive
    # engine cannot derive the 'carbononitridic' word-form (same rationale as
    # phosgene->carbonyl dichloride), so exact-SMILES retained PIN. The trivial
    # 'cyanogen chloride' (OPSIN simpleGroup, is_pin=False) is denied in
    # iupac_2013_pin_list.json (--trivial fallback). Canonical key of N#CCl.
    "N#CCl": "carbononitridic chloride",
    "CCOC(C)=O": "ethyl acetate",
    "COC(C)=O": "methyl acetate",
    "CC#N": "acetonitrile",
    # (the Blue Book): 'dimethylformamide (PIN)' (:32782) --
    # formamide's substitutable hydrogens are its two N-H (a C-substituted formamide
    # is named on another parent: 'carbonochloridic amide (PIN) (not
    # 1-chloroformamide)',:32707), so complete substitution in the same way omits
    # the N locants,:3007).
    "CN(C)C=O": "dimethylformamide",

    # === SULFUR COMPOUNDS (a phase) ===
    # Disulfane (S-S bond, no carbon)
    "SS": "disulfane",
    # Thiols
    "CS": "methanethiol",
    "CCS": "ethanethiol",
    # Sulfides (thioethers)
    "CSC": "dimethyl sulfide",
    "CCSCC": "diethyl sulfide",
    # Sulfoxides (canonical SMILES form)
    "CS(C)=O": "dimethyl sulfoxide",  # DMSO
    # Sulfones (canonical SMILES form)
    "CS(C)(=O)=O": "dimethyl sulfone",
    # Sulfonic acids (canonical SMILES form)
    "CS(=O)(=O)O": "methanesulfonic acid",
    "CCS(=O)(=O)O": "ethanesulfonic acid",
    "O=S(=O)(O)c1ccccc1": "benzenesulfonic acid",

    # === PHOSPHORUS COMPOUNDS (a phase) ===
    # Phosphines (use IUPAC 2013 "phosphane" not "phosphine")
    "CP": "methylphosphane",
    "CCP": "ethylphosphane",
    "CP(C)C": "trimethylphosphane",
    "CCP(CC)CC": "triethylphosphane",
    "c1ccc(P(c2ccccc2)c2ccccc2)cc1": "triphenylphosphane",
    # Phosphine oxides
    "CP(C)(C)=O": "trimethylphosphane oxide",
    "CCP(=O)(CC)CC": "triethylphosphane oxide",
    "O=P(c1ccccc1)(c1ccccc1)c1ccccc1": "triphenylphosphane oxide",
    # Phosphonic acids (a phase: substituent-prefix PIN — the
    # 'methane'/'ethane' parent-hydride-stem forms are explicitly rejected)
    "CP(=O)(O)O": "methylphosphonic acid",
    "CCP(=O)(O)O": "ethylphosphonic acid",
    "O=P(O)(O)c1ccccc1": "phenylphosphonic acid",
    # Phosphinic acids
    "CP(C)(=O)O": "dimethylphosphinic acid",
    "CCP(=O)(O)CC": "diethylphosphinic acid",
    # Phosphate esters (functional class naming).
    # Mono-/di-ester rows for methyl/ethyl were REMOVED here : "methyl
    # phosphate"/"dimethyl phosphate"/"ethyl phosphate"/"diethyl phosphate"
    # denote the ANION (OPSIN parses them back to the deprotonated dianion),
    # not the neutral acid-ester SMILES these rows were keyed on. The neutral
    # forms need the free -OH cited ("methyl dihydrogen phosphate", "dimethyl
    # hydrogen phosphate",...) which `rules/phosphorus.py::name_phosphate_ester`
    # already produces correctly (per-OH protonation word derived from the
    # structure) once dispatch is no longer shadowed by this table. The
    # tri-ester rows below are correct as-is (0 free -OH -> no protonation word).
    "COP(=O)(OC)OC": "trimethyl phosphate",
    "CCOP(=O)(OCC)OCC": "triethyl phosphate",

    # === RETAINED NITROGEN COMPOUNDS ===
    "NC(N)=O": "urea",
    # Wave-2 completion /: thiourea is the retained
    # functional-parent PIN for the standalone NC(=S)N (thio-analogue of urea).
    # The thiourea-as-substituent prefix 'carbamothioylamino' is separate.
    "NC(N)=S": "thiourea",
    # Wave-2 completion: pentazolidine — the saturated homogeneous
    # all-nitrogen 5-ring (HW-preselected). RDKit aromatises the all-NH ring to
    # [nH]1[nH][nH][nH][nH]1, distinct from 1H-pentazole; exact-SMILES lookup.
    "[nH]1[nH][nH][nH][nH]1": "pentazolidine",
    "N=C(N)N": "guanidine",
    # Wave2 /: formazan HN=N-CH=N-NH2 is the
    # retained PIN (fully substitutable; substituted formazans + the
    # formazan-N-yl prefixes are a deferred follow-on,. The
    # systematic fall-through emitted the RT-correct but non-PIN functional-
    # class name '1-diazenylmethanal hydrazone'.
    "N=NC=NN": "formazan",

    # === OTHER COMMON COMPOUNDS ===
    "O": "water",
    "N": "ammonia",
    # Wave2 /: bare H2N-OH — retained parent
    # hydride 'hydroxylamine'. The perception SMARTS requires a C on N (the
    # handler names substituted forms), so the bare parent fell through to
    # 'unknown'. Exact-SMILES match, same tier as water/ammonia.
    "NO": "hydroxylamine",
    "O=C=O": "carbon dioxide",
    # (the Blue Book "HCN formonitrile(PIN) methanenitrile
    # hydrogen cyanide"): the PIN for HCN is 'formonitrile', not the retained
    # functional-class 'hydrogen cyanide'. Exact-SMILES whole-molecule parent.
    "C#N": "formonitrile",
    # (the Blue Book "NC-CN oxalonitrile (PIN)
    # ethanedinitrile"): the PIN for NC-CN is 'oxalonitrile'. The systematic
    # engine derives 'ethanedinitrile' (right molecule, non-PIN spelling);
    # exact-SMILES retained PIN wins at RETAINED_NAME.
    "N#CC#N": "oxalonitrile",
    "O=S=O": "sulfur dioxide",
    "N#N": "dinitrogen",
    "O=O": "dioxygen",
    # Task M1 (v51): O2 written as a diradical (`[O][O]`) is a DISTINCT
    # RDKit-canonical SMILES from `O=O` (same InChIKey MYMOFIZGZYHOMD), so the
    # exact-SMILES lookup above missed it and it abstained ("inorganic compound
    # (not supported)"). OPSIN parses `dioxygen` -> O=O (identical skeleton),
    # so this diradical spelling round-trips too. Root cause: the table carried
    # only one of the two canonical spellings of the same molecule.
    "[O][O]": "dioxygen",

    # === INORGANIC ACIDS (a phase fix) ===
    "O=[N+]([O-])O": "nitric acid",
    "O=[N+]([O-])OO": "peroxynitric acid",

    # === NITRILES - AROMATIC (a phase fix) ===
    "N#Cc1ccccc1": "benzonitrile",  # C6H5CN - PIN per

    # === AMIDES - AROMATIC (a phase suffix FG fix) ===
    "NC(=O)c1ccccc1": "benzamide",  # C6H5CONH2 - PIN per

    # === THIAZOLIDINES (a phase fix) ===
    # 1,3-thiazolidine: S at 1, N at 3 (not adjacent)
    "C1CSCN1": "thiazolidine",
    # 1,2-isothiazolidine: S at 1, N at 2 (adjacent)
    "C1CNSC1": "isothiazolidine",

    # === AMINO ACIDS (common) ===
    # Glycine is the sole ACHIRAL standard amino acid (no alpha-carbon
    # stereocentre), so a bare, context-free SMILES match never asserts a
    # configuration the input lacks -- safe to keep in this dumb, stereo-blind
    # dict.
    "NCC(=O)O": "glycine",
    # a phase (stereo honesty, LIVE 0-wrong): the other 19 common amino
    # acids used to have a flat/stereo-free entry HERE too. This dict is keyed
    # by EXACT canonical SMILES with no stereo awareness at all, and
    # `_handle_retained_name` (routing/dispatch_table.py, StoutClass.RETAINED_
    # NAME, priority 1300) runs BEFORE `_handle_amino_acid` (StoutClass.AMINO_
    # ACID, priority 1400) -- so for any INPUT with an undefined alpha-carbon
    # (no wedge/parity), this flat entry matched FIRST and silently emitted the
    # bare retained name (e.g. `NC(CC(N)=O)C(=O)O` -> "asparagine"), which
    # OPSIN parses back to the L stereoisomer -- a stereo the input never
    # defined (full RT-fail). `## **** The stereodescriptors 'D' and 'L'`
    # (the Blue Book): "The stereodescriptor 'xi' (Greek letter xi)
    # indicates unknown configuration" -- so an unresolved centre must never
    # be spelled with the plain (implicit-L) name. `data.amino_acids.
    # STANDARD_AMINO_ACIDS` already carries every one of these 19 keys (same
    # structures, RT-equivalent), reached via the AMINO_ACID dispatch entry
    # (`get_amino_acid_name(..., with_descriptor=True)`), which now correctly
    # DEFERS (returns None -> falls through to the systematic namer) for a
    # genuinely stereo-undefined alpha-carbon and still emits the bare name
    # for the achiral case / correct L-/D- for a defined one. Deleting the
    # duplicates here (rather than gating dispatch order) removes the second,
    # stereo-blind copy at its root and costs no coverage -- every deleted key
    # has a live equivalent in `STANDARD_AMINO_ACIDS`.

    # === ADDITIONAL SATURATED HETEROCYCLES (a phase expansion) ===
    "C1COCCO1": "1,4-dioxane",
    # / Table 2.3: morpholine chalcogen-replacement parents.
    "C1CSCCO1": "1,4-oxathiane",        # O+S ring (was MISLABELED thiomorpholine)
    "C1CSCCN1": "thiomorpholine",       # S-for-O morpholine (PIN; wins OPSIN 'thiamorpholine')
    "C1C[Se]CCN1": "selenomorpholine",  # Se-for-O morpholine (PIN)
    "C1C[Te]CCN1": "telluromorpholine", # Te-for-O morpholine (PIN)
    "C1CN2CCC1CC2": "quinuclidine",
    "C1CCC2NCCCC2C1": "decahydroquinoline",
    # PA1 sweep: the row `"C1CCN2CCCCC2C1": "decahydroisoquinoline"` was DELETED.
    # WRONG STRUCTURE: that SMILES puts the nitrogen at a RING-FUSION atom, which
    # makes it quinolizidine / octahydro-2H-quinolizine (InChIKey LJPZHJUSICYOIX).
    # Decahydroisoquinoline has N at position 2, not at a fusion position:
    # C1CCC2CNCCC2C1 (NENLYAQPNATJSU), verified by independent hand-construction.
    # It WAS live: C1CCN2CCCCC2C1 emitted "decahydroisoquinoline". Deleted rather
    # than re-keyed to the true structure, because adding a NEW retained-name
    # emission is a separate decision (the 2013 PIN would carry the full hydro
    # locant set, and this change must not smuggle one in.

    # === FATTY ACIDS (a phase expansion) ===
    "CCCCC(=O)O": "pentanoic acid",
    "CCCCCC(=O)O": "hexanoic acid",
    "CCCCCCCC(=O)O": "octanoic acid",
    "CCCCCCCCCC(=O)O": "decanoic acid",
    "CCCCCCCCCCCC(=O)O": "dodecanoic acid",
    "CCCCCCCCCCCCCC(=O)O": "tetradecanoic acid",
    "CCCCCCCCCCCCCCCC(=O)O": "hexadecanoic acid",
    "CCCCCCCCCCCCCCCCCC(=O)O": "octadecanoic acid",

    # === BRANCHED CARBOXYLIC ACIDS (a phase expansion) ===
    # NOTE: isobutyric acid removed -- PIN is "2-methylpropanoic acid"
    # NOTE: isovaleric acid removed -- PIN is "3-methylbutanoic acid"
    # NOTE: pivalic acid removed -- PIN is "2,2-dimethylpropanoic acid"

    # === UNSATURATED ACIDS (a phase expansion) ===
    # PF sweep: `crotonic acid` and `sorbic acid` DELETED, and `crotonaldehyde`
    # below with them. Two independent reasons, either sufficient:
    #
    # 1. NOT BLUE BOOK NAMES. All three occur **0** times in the Blue Book under a
    # markup/OCR-tolerant search whose known-positive control passes in the same
    # run (acetic acid 178, benzoic acid 185, but-2-enoic acid 4). Their PIN
    # status is therefore unestablished -- the same ground on which `isobutyric
    # acid`, `isovaleric acid`, `pivalic acid` and `pinacolone` were removed
    # above. And the replacement is not merely systematic, it is marked PIN
    # verbatim: "but-2-enoic acid (PIN)".
    # 2. EACH OVER-ASSERTS DOUBLE-BOND GEOMETRY THE KEY LEAVES UNDEFINED. The keys
    # `CC=CC(=O)O` / `CC=CC=CC(=O)O` / `CC=CC=O` specify no geometry, while
    # crotonic acid IS (E)-but-2-enoic acid (the Z isomer is isocrotonic acid)
    # and sorbic acid IS (2E,4E). So the name named a stereoisomer the input
    # never claimed -- invisible to, which compares the InChIKey
    # skeleton block and is stereo-insensitive by design.
    #
    # Measured at 27160d6a: all three now emit the PIN and round-trip EXACTLY to the
    # input (but-2-enoic acid, hexa-2,4-dienoic acid, but-2-enal); +3 round-trip, 0
    # lost, and 0 gold-oracle rows assert any of the three names.
    #
    # ⚠ DO NOT extend this deletion to the 19 unqualified amino-acid entries
    # (alanine, leucine, isoleucine, threonine,...). They look identical to a
    # round-trip metric -- OPSIN resolves a bare `isoleucine` to the L-form, so they
    # fail RT too -- but they are BLUE-BOOK-CORRECT. "The
    # stereodescriptors 'D' and 'L'" (the Blue Book) spells configuration with
    # the prefix: ":54324 L-isoleucine (symbols 'Ile', 'I') (2S,3S)-2-amino-3-
    # methylpentanoic acid", and the amino-acid table at:54191 lists bare
    # `alanine` against a structure drawn WITHOUT stereochemistry
    # ("CH3-CH(NH2)-COOH"). Deleting them would buy 2 round-trip points by emitting
    # a less preferred name -- Goodharting the metric against conformance.

    # === ALDEHYDES (a phase expansion) ===
    # Note: butanal/pentanal preferred over butyraldehyde/valeraldehyde for consistency

    # === KETONES (a phase expansion) ===
    # PA1 sweep: the row `"CC(=O)CC(C)(C)C": "pinacolone"` was DELETED. WRONG
    # STRUCTURE, and unambiguously so -- the formulae differ: that key is C7H14O
    # (4,4-dimethylpentan-2-one, AZASWMGVGQEVCS), while pinacolone is C6H12O
    # (3,3-dimethylbutan-2-one, CC(=O)C(C)(C)C, PJGSXYOJTGTZAV). One CH2 too many.
    # It WAS live: CC(=O)CC(C)(C)C emitted "pinacolone". CORRECTION TO THE PA1
    # AUDIT, which recorded pinacolone as "currently moot: the molecule names as
    # 'unknown organic compound' (fail-closed), so nothing wrong ships" -- it did
    # ship. Deleted rather than re-keyed: 'pinacolone' does not occur anywhere in
    # the Blue Book, so its PIN status is unestablished and it must not be newly
    # promoted onto the correct structure. True pinacolone keeps the name it
    # already gets from the OPSIN-import surface; 4,4-dimethylpentan-2-one now
    # gets its systematic name.
    "CC(=O)C=C(C)C": "mesityl oxide",

    # === DIOLS AND POLYOLS (a phase expansion) ===
    # a phase: ethylene/propylene/trimethylene glycol removed (deprecated
    # "glycol" names, not PINs). Systematic ethane-1,2-diol / propane-1,2-diol /
    # propane-1,3-diol now emitted; also enforced by _PIN_DENY (recurrence guard).
    # See docs/retained_name_conflicts.md § "a phase".
    "OCCCCO": "butane-1,4-diol",

    # === UNSATURATED ALCOHOLS (a phase expansion) ===
    "C=CCO": "allyl alcohol",
    "C#CCO": "propargyl alcohol",

    # === TERPENES (a phase expansion) ===
    "C=C(C)C1CC=C(C)CC1": "limonene",
    "CC12CCC(CC1=O)C2(C)C": "camphor",

    # === NAPHTHOLS AND BIPHENYLS (a phase expansion) ===
    "Oc1ccc2ccccc2c1": "2-naphthol",
    "Oc1ccc(-c2ccccc2)cc1": "4-phenylphenol",

    # === COMMON PHARMACEUTICALS (a phase expansion) ===
    # a phase: "aspirin" (brand name) removed; PIN 2-acetyloxybenzoic
    # acid now emitted; also enforced by _PIN_DENY. See retained_name_conflicts.md.

    # === CYCLIC IMIDES (a phase, expanded a phase) ===
    "O=C1CCC(=O)N1": "succinimide",
    "O=C1C=CC(=O)N1": "maleimide",
    "O=C1CCCC(=O)N1": "glutarimide",
    "O=C1NC(=O)c2ccccc21": "phthalimide",

    # === COMMONLY ENCOUNTERED RETAINED NAMES (a phase) ===
    # Source: IUPAC 2013 Blue Book, various sections
    # OPSIN RT verified 2026-03-08
    "c1ccc(-c2ccccc2)cc1": "biphenyl",  # general nomenclature
    "C#C": "acetylene",  # PIN for unsubstituted ethyne
    "COc1ccccc1": "anisole",  # PIN
    #: "caprolactam" WITHDRAWN from the PIN path (demoted to
    # GENERAL_RETAINED_NAMES by the pin:false row in iupac_2013_pin_list.json).
    # The PIN is `azepan-2-one` -- (the Blue Book) "Cyclic anhydrides, esters
    # and amides are named as pseudoketones; the resulting names are preferred
    # IUPAC names", whose own example prints it at the Blue Book: `azepan-2-one (PIN)
    # hexano-6-lactam (see `. It was the only break in the systematic
    # series azetidin-2-one / pyrrolidin-2-one / piperidin-2-one / _ / azocan-2-one.
    # ⚠ The former inline citation here read " retained lactam name";
    # (the Blue Book) is "Bi- and polycyclic von Baeyer parent hydrides" and
    # licenses nothing of the sort. `caprolactam` has 0 the Blue Book hits.
    "O=C1CCCCCN1": "caprolactam",

    # === NUCLEOSIDES (a phase) ===
    # Retained names per carbohydrate nomenclature conventions
    # Each nucleoside has fixed beta stereochemistry at anomeric position
    # OPSIN RT verified 2026-03-08 (all 8 pass)
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O": "adenosine",
    "Nc1nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1": "guanosine",
    "Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1": "cytidine",
    "Cc1cn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]c1=O": "thymidine",
    "O=c1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1": "uridine",
    # a phase: 2'-deoxynucleoside PINs (the prime is REQUIRED — 'deoxyadenosine'
    # is ambiguous; '2'-deoxyadenosine' is the PIN, / carbohydrate.
    # All four OPSIN-RT; 2'-deoxyuridine was MISSING. Old bare 'deoxy…' keys renamed.
    "Nc1ncnc2c1ncn2[C@H]1C[C@H](O)[C@@H](CO)O1": "2'-deoxyadenosine",
    "Nc1nc2c(ncn2[C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1": "2'-deoxyguanosine",
    "Nc1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)n1": "2'-deoxycytidine",
    "O=c1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1": "2'-deoxyuridine",
    #: nucleotide retained names re-admitted on the hand-curated
    # side (the adenylic JSON deny entries carry hc_override). These are
    # correct-but-NON-PIN names (the PIN is the full systematic
    # adenosine/inosine 5'-(dihydrogen phosphate)); re-admitting them stops the
    # egregious atom-dropping mis-name AMP→'6-aminopyrimidine' /
    # IMP→'6-oxo-1,3-diazine'. A10 honest-fail-on-data: a correct retained name
    # beats a wrong systematic one when no PIN is computable. The inosinic KETO
    # tautomer is added explicitly (OPSIN normalises the name to the enol form,
    # which already promotes via the OPSIN import). OPSIN-RT verified.
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O": "5'-adenylic acid",
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1OP(=O)(O)O": "2'-adenylic acid",
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](OP(=O)(O)O)[C@H]1O": "3'-adenylic acid",
    "O=c1[nH]cnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O": "5'-inosinic acid",
    # a phase: inosine (the hypoxanthine nucleoside) was MISSING (adenosine/
    # guanosine/cytidine/uridine all present) -> 'unknown'. Added via the KETO
    # tautomer canonical (mirrors hypoxanthine and 5'-inosinic acid above): RDKit
    # canonicalises a keto input to this form, while OPSIN normalises the name
    # 'inosine' to the 6-hydroxy ENOL — same skeleton InChIKey (UGQMRVRMYYASKQ),
    # so the self-consistency gate accepts it (tautomer-tolerant). Correct-but-
    # NON-PIN retained name (PIN is systematic); OPSIN-RT (skeleton) verified.
    "O=c1[nH]cnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O": "inosine",

    # === DISACCHARIDES (a phase) ===
    # NOTE: OPSIN cannot parse most disaccharide names -- InChI validation used
    # instead of OPSIN RT. Canonical SMILES from PubChem + RDKit canonicalization.
    "OC[C@@H]1O[C@@](CO)(O[C@H]2[C@H](O)[C@@H](O)[C@@H](O)O[C@@H]2CO)[C@@H](O)[C@H]1O": "sucrose",
    "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)C(O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O": "maltose",
    "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)C(O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@H]1O": "lactose",
    "OC[C@H]1O[C@@H](O[C@@H]2[C@@H](O)[C@H](O)[C@@H](O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O": "cellobiose",
    "OC[C@H]1O[C@H](O[C@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@H](O)[C@H]1O": "trehalose",

    # === MODIFIED SUGARS (a phase) ===
    # N-acetylneuraminic acid (sialic acid / Neu5Ac)
    # Unique 9-carbon structure, not standard pyranose/furanose
    # OPSIN RT verified 2026-03-08
    "CC(=O)N[C@H]1[C@H]([C@H](O)[C@H](O)CO)OC(O)(C(=O)O)C[C@@H]1O": "N-acetylneuraminic acid",

    # === ADDITIONAL POLYCYCLIC AROMATICS (a phase expansion) ===
    # All canonical SMILES verified via Chem.CanonSmiles + OPSIN RT 2026-03-16
    "c1ccc2c(c1)c1ccccc1c1ccccc21": "triphenylene",
    "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61": "coronene",
    "c1ccc2c(c1)-c1cccc3cccc-2c13": "fluoranthene",

    # === BENZOIC ACID DERIVATIVES (a phase expansion) ===
    # Note: salicylic acid and gallic acid omitted -- existing tests expect
    # systematic names (2-hydroxybenzoic acid, 3,4,5-trihydroxybenzoic acid)
    "O=C(O)c1cccc(C(=O)O)c1": "isophthalic acid",
    "Nc1ccccc1C(=O)O": "anthranilic acid",
    "Nc1ccc(C(=O)O)cc1": "4-aminobenzoic acid",
    "COc1cc(C(=O)O)ccc1O": "vanillic acid",

    # === CYCLIC ANHYDRIDES -> heterocyclic-pseudoketone (dione) PINs ===
    # D-FOLLOWON item 6 method 1): the PIN for a cyclic anhydride is
    # the heterocyclic dione, NOT the functional-class '{diacid} anhydride'.
    # Retargeted from the non-PIN retained names 'maleic anhydride'/'phthalic
    # anhydride' (general nomenclature only). The anhydride handler  emits
    # these via get_retained_name; succinic/glutaric (no retained entry) take the
    # saturated-oxa-dione path (oxolane-2,5-dione / oxane-2,6-dione).
    "O=C1C=CC(=O)O1": "furan-2,5-dione",
    "O=C1OC(=O)c2ccccc21": "2-benzofuran-1,3-dione",
    # W8-P4: the mancude (aromatic-benzo-fused) thio analogue of
    # phthalic anhydride. BB verbatim gives the SATURATED benzo form
    # "hexahydro-2-benzothiophene-1,3-dione (PIN)" (BB 32546); dropping
    # "hexahydro" for this fully-aromatic ring is the direct BB-sanctioned
    # extension (exactly mirroring furan-2,5-dione/2-benzofuran-1,3-dione
    # above, S replacing O). OPSIN-2.9 RT-confirmed: name -> canonical SMILES
    # matches this key exactly.
    "O=C1SC(=O)c2ccccc21": "2-benzothiophene-1,3-dione",

    # === HETEROCYCLE DERIVATIVES (a phase expansion) ===
    "O=c1ccc2ccccc2o1": "coumarin",       # 2H-chromen-2-one
    "O=c1ccoc2ccccc12": "chromone",       # 4H-chromen-4-one
    "O=c1c2ccccc2oc2ccccc12": "xanthone",  # 9H-xanthen-9-one
    "O=C(O)c1cccnc1": "nicotinic acid",   # pyridine-3-carboxylic acid
    "O=C(O)c1ccncc1": "isonicotinic acid",  # pyridine-4-carboxylic acid
    "O=C(O)c1ccccn1": "picolinic acid",   # pyridine-2-carboxylic acid
    "NC(=O)c1cccnc1": "nicotinamide",     # pyridine-3-carboxamide
    "O=C1NS(=O)(=O)c2ccccc21": "saccharin",  # 1,1-dioxo-1,2-benzothiazol-3-one

    # === ADDITIONAL AMINES (a phase expansion) ===
    "NCCCCN": "putrescine",    # butane-1,4-diamine
    "NCCCCCN": "cadaverine",   # pentane-1,5-diamine

    # === ADDITIONAL SOLVENTS (a phase expansion) ===
    "C1COCO1": "1,3-dioxolane",

    # === NUCLEOBASES (a phase expansion) ===
    "O=c1cc[nH]c(=O)[nH]1": "uracil",
    "Cc1c[nH]c(=O)[nH]c1=O": "thymine",
    "Nc1cc[nH]c(=O)n1": "cytosine",
    "Nc1ncnc2[nH]cnc12": "adenine",
    "Nc1nc2[nH]cnc2c(=O)[nH]1": "guanine",
    "Nc1nc(=O)c2[nH]cnc2[nH]1": "guanine",  # alternate tautomer (a phase)

    # === PURINE DERIVATIVES (a phase expansion) ===
    "O=c1[nH]c(=O)c2[nH]cnc2[nH]1": "xanthine",  # 3,7-dihydro-1H-purine-2,6-dione
    "O=c1[nH]c(=O)c2nc[nH]c2[nH]1": "xanthine",  # alternate tautomer
    "O=c1[nH]cnc2[nH]cnc12": "hypoxanthine",  # 1,9-dihydro-6H-purine-6-one
    "O=c1[nH]cnc2nc[nH]c12": "hypoxanthine",  # alternate tautomer
    "O=C(O)c1cc(=O)[nH]c(=O)[nH]1": "orotic acid",  # pyrimidine-2,4(1H,3H)-dione-6-carboxylic acid
    "c1ccc2nc3ccccc3cc2c1": "acridine",  # dibenzo[b,e]pyridine

    # === LONG-CHAIN DIACIDS (a phase expansion) ===
    "O=C(O)CCCCCCC(=O)O": "suberic acid",   # octanedioic acid
    "O=C(O)CCCCCCCC(=O)O": "azelaic acid",  # nonanedioic acid
    "O=C(O)CCCCCCCCC(=O)O": "sebacic acid",  # decanedioic acid

    # === AROMATIC DERIVATIVES (a phase expansion) ===
    "c1ccc(Nc2ccccc2)cc1": "diphenylamine",

    # === BENZALDEHYDE DERIVATIVES (a phase expansion) ===
    "COc1cc(C=O)ccc1O": "vanillin",           # 4-hydroxy-3-methoxybenzaldehyde
    "O=Cc1ccc(O)cc1": "4-hydroxybenzaldehyde",
    "COc1ccc(C=O)cc1": "anisaldehyde",        # 4-methoxybenzaldehyde
    "O=Cc1cccnc1": "nicotinaldehyde",         # pyridine-3-carbaldehyde
    "O=Cc1ccncc1": "isonicotinaldehyde",      # pyridine-4-carbaldehyde

    # === NAPHTHOL (a phase expansion) ===
    "Oc1cccc2ccccc12": "1-naphthol",

    # === MISCELLANEOUS AROMATICS (a phase expansion) ===
    "OC(c1ccccc1)c1ccccc1": "benzhydrol",             # diphenylmethanol
    "O=c1cc(-c2ccccc2)oc2ccccc12": "flavone",         # 2-phenyl-4H-chromen-4-one
    "Nc1ccc(N)cc1": "1,4-phenylenediamine",           # benzene-1,4-diamine
    "Oc1cccc(O)c1O": "pyrogallol",                    # benzene-1,2,3-triol
    "CC(=O)c1ccc(O)cc1": "4-hydroxyacetophenone",

    # === ADDITIONAL COMMON COMPOUNDS (a phase expansion) ===
    # All canonical SMILES verified via Chem.CanonSmiles 2026-03-16
    "C1CCC2CCCCC2C1": "decahydronaphthalene",         # decalin
    "O=C(O)c1ccco1": "furan-2-carboxylic acid",       # furoic acid
    # (Wave-2 completion) The old "bipyridyl" row here keyed the canonical
    # SMILES of 2,4'-bipyridine to the NAME "2,2'-bipyridine" — a wrong-key
    # data bug (suppressed it to unknown). Bipyridines are named
    # systematically by rules/ring_assemblies.py; no retained row.
    "Oc1cc(O)cc(O)c1": "phloroglucinol",              # benzene-1,3,5-triol
    # Keys checked against OPSIN 2.9.0's reading of each name: the
    # fusion name denotes the structure): benzo[f] fuses the benzo ring to the
    # quinoline 5,6-bond, benzo[h] to the 7,8-bond next to N1.
    "c1ccc2c(c1)ccc1ncccc12": "benzo[f]quinoline",
    "c1ccc2c(c1)ccc1cccnc12": "benzo[h]quinoline",
    "Oc1ccc2c(c1)OCO2": "sesamol",                    # 3,4-methylenedioxyphenol
    "O=Cc1ccc2c(c1)OCO2": "piperonal",                # 3,4-methylenedioxybenzaldehyde
    "C=CCc1ccc2c(c1)OCO2": "safrole",
    "CC(=O)c1ccco1": "2-acetylfuran",
    # Note: tetracene SMILES was actually benz[a]anthracene (angular) - removed
    "Cc1cc(C)c(O)c(C)c1": "mesitol",                  # 2,4,6-trimethylphenol
    "CC(C)(C)c1ccccc1": "tert-butylbenzene",
    "c1ccc(CCc2ccccc2)cc1": "1,2-diphenylethane",     # bibenzyl
    "C=Cc1ccc(C=C)cc1": "1,4-divinylbenzene",
    "Cc1cc(C)c(C)cc1C": "durene",                     # 1,2,4,5-tetramethylbenzene
    "c1ccc(Cc2ccccc2)cc1": "diphenylmethane",
    "c1ccc(C(c2ccccc2)c2ccccc2)cc1": "triphenylmethane",
    "C1=Cc2ccccc2C1": "1H-indene",
    "c1ccc(SSc2ccccc2)cc1": "diphenyl disulfide",
    "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]": "2,4,6-trinitrotoluene",
    "O=Cc1ccco1": "furfural",                         # furan-2-carbaldehyde
}


def get_retained_name(canonical_smiles: str) -> Optional[str]:
    """
    Get retained name for a canonical SMILES if one exists.

    a phase + RESEARCH ADDITION 1 root-cause fix: consults the
    merged ALL_RETAINED_NAMES dict via late binding (avoids circular
    import at module load time). Resolves 2 function-import consumers
    in one edit: ring_assemblies.py:296, heterocycles.py:36.

    Args:
        canonical_smiles: Canonical SMILES string (must be canonicalized!)

    Returns:
        Retained name string, or None if not found
    """
    try:
        # wp7: the shared accessor also serves an OPSIN-import trivial name
        # without PIN evidence, recorded as a non-PIN fragment.
        from orthonym.data import get_retained_name as _merged_get
        return _merged_get(canonical_smiles)
    except ImportError:
        return RETAINED_NAMES.get(canonical_smiles)


def is_retained_name_compound(canonical_smiles: str) -> bool:
    """
    Check if compound has a retained name.

    a phase + RESEARCH ADDITION 1 root-cause fix: consults
    merged ALL_RETAINED_NAMES via late binding.

    Args:
        canonical_smiles: Canonical SMILES string

    Returns:
        True if compound has a retained name
    """
    try:
        from orthonym.data import ALL_RETAINED_NAMES
        return canonical_smiles in ALL_RETAINED_NAMES
    except ImportError:
        return canonical_smiles in RETAINED_NAMES


def add_retained_name(canonical_smiles: str, name: str) -> None:
    """DEPRECATED: prefer ``orthonym.data.register_retained_name``.

    a phase + REVIEW root-cause fix: ``ALL_RETAINED_NAMES``
    is built once at import time (``data/__init__.py``); runtime mutators
    must keep the HC dict and the merged dict synchronised so the public
    ``get_retained_name`` lookup sees new entries.

    This wrapper updates BOTH the hand-curated ``RETAINED_NAMES`` dict
    (for legacy callers that read it directly) and delegates to
    ``register_retained_name`` (which mutates ``ALL_RETAINED_NAMES``).

    Args:
        canonical_smiles: Canonical SMILES string
        name: Retained IUPAC name
    """
    RETAINED_NAMES[canonical_smiles] = name
    try:
        from orthonym.data import register_retained_name
        register_retained_name(canonical_smiles, name)
    except ImportError:
        pass
