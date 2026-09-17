"""
Functional group seniority rules for principal group selection.

Based on IUPAC 2013 Blue Book to.
The principal group (highest seniority) becomes the suffix;
all other groups become prefixes.
"""

from typing import Dict, List, Optional, Tuple

# (2026-04-25): per-FG SMARTS attachment atom override.
#
# Most functional-group SMARTS lead with the locant-bearing atom (the C of
# -COOH in [CX3](=O)[OX2H1]; the C of -OH in [CX4][OX2H1]). Parent-selection
# locant comparators historically used SMARTS-match index 0 as the canonical
# attachment for that FG instance.
#
# For PGs whose SMARTS leads with a flanking (non-locant-bearing) atom, this
# heuristic is wrong. The motivating case is ``disulfide`` (``[#6][SX2][SX2][#6]``):
# atom 0 is a flanking carbon, but IUPAC says the locant set for a
# multi-atom PG uses the heteroatoms (S, S). When a disulfide lives in a ring,
# using the flanking C's locant produces a wrong ring-vs-chain decision in
# the (f) cascade, silently dropping the ring sulfurs from the name.
#
# Each entry maps FG name -> list of SMARTS-match indices to use as
# locant-bearing atoms. When the helper sees multiple indices, downstream
# comparators take ``min(locants)`` to match. Default for any FG
# not present is ``[0]`` (preserves existing behaviour).
#
# Source: internal notes
PG_ATTACHMENT_INDICES: Dict[str, List[int]] = {
    "disulfide": [1, 2],  # [#6][SX2][SX2][#6] -- the two S sulfurs
    # task 9: the ketone family SMARTS '[#6][CX3](=X)[#6]' leads with a
    # FLANKING carbon; the locant-bearing atom is the carbonyl/thiocarbonyl
    # carbon at match index 1. Without these entries every on-ring/on-chain/
    # locant computation used the flanking atom (an aryl ketone's "attachment"
    # was the benzene ring atom itself -> false pg_on_ring -> wrong parent).
    "ketone": [1],
    "thioketone": [1],
    "selenoketone": [1],
    "telluroketone": [1],
    # Wave2: the -ol/-thiol/-selenol/-tellurol/-amine family
    # SMARTS lead with the characteristic HETEROATOM; the locant-bearing atom
    # is the attached carbon. Alcohol-class matches reach
    # principal_group_atoms normalized to (O, bearing-C) 2-tuples
    # (_normalize_pcg_match), so index 1 is the carbinol carbon for every
    # subtype. Without these, extending SKELETAL_SUFFIX_PGS to this family
    # would false-negative every genuine ring suffix (4-methylcyclohexan-1-ol
    # would lose its ring parent). Secondary/tertiary amines list every C on
    # the N: the parent atom may be any of them.
    "alcohol": [1],
    "primary_alcohol": [1],
    "secondary_alcohol": [1],
    "tertiary_alcohol": [1],
    "thiol": [1],
    "selenol": [1],
    "tellurol": [1],
    "primary_amine": [1],
    "secondary_amine": [1, 2],
    "tertiary_amine": [1, 2, 3],
    # C2: the aromatic/enolic members of the same -ol/-amine family were
    # left out of the Wave2 block above, so they fell through to the
    # default index 0 -- which for all three SMARTS is the HETEROATOM:
    # phenol [OX2H][cX3] -> 0 is O, the aromatic C is 1
    # aromatic_amine [NX3H2][cX3] -> 0 is N, the aromatic C is 1
    # enol [OX2H][CX3]=[CX3] -> 0 is O, the hydroxy C is 1
    # A heteroatom is never a parent-hydride skeletal atom, so every consumer
    # asking "is the PG on this ring/chain, and at which locant?" got 'no'.
    # After general_engine._inline_suffix_locant began refusing on an empty
    # anchor set (Task 5, 5f46d505) that silently destroyed correct names.
    #
    # (the Blue Book Blue Book) -- "Primary amines, R-NH2,
    # are systematically named in the following ways: (1) by adding the suffix
    # 'amine' to the name of the parent hydride". The suffix attaches to the
    # PARENT HYDRIDE; its examples are `quinolin-4-amine (PIN)` and
    # `1-benzofuran-2-amine (PIN)` -- the locant is the ring carbon.
    # "Systematic names of alcohols, phenols, enols, and ynols"
    # (:26826) -- "(1) substitutively, using the suffix 'ol'... the starting
    # point and the direction of numbering... are chosen so as to give lowest
    # locants to the 'ol' suffixes"; `naphthalen-1-ol (PIN)` (:26820) and
    # `2-nitrobenzene-1,3-diol (PIN)` cite ring carbons -- the hydroxy oxygen
    # has no skeletal locant at all.
    # "Citation of locants" (:2869) -- "the name 2-chloroethan-1-ol is
    # the PIN": the '1' is the carbon bearing the -OH, not the oxygen.
    "phenol": [1],
    "aromatic_amine": [1],
    "enol": [1],
    # imine `[CX3]=[NX2;...]` ALREADY leads with the carbon, so [0] is exactly
    # the default and this entry changes no behaviour today. Pinned anyway:
    # parent_selection.py:52 and general_engine.py:66 both assert "imine
    # already leads with C" in prose, and that is an unenforced dependency on
    # SMARTS atom order -- reordering the pattern would move every imine locant
    # onto the nitrogen with no test to catch it.
    "imine": [0],
}

# Functional group seniority order (highest priority first)
# Groups in this list can be expressed as suffixes when principal
SENIORITY_ORDER = [
    # Acids (highest priority)
    "carboxylic_acid",
    # Peroxy acid / Table 4.3) -- directly below its parent acid,
    # above the chalcogen replacements
    "peroxy_acid",
    # Thiocarboxylic acids (IUPAC -- just below carboxylic acid
    "thioic_S_acid",
    "thioic_O_acid",
    "dithioic_acid",
    # a phase Tier FRN-A: chalcogen acids — additive per internal notes;
    # position locked by AUDIT-FRN; parallel to thioic_S_acid/O_acid/dithioic
    "selenoic_Se_acid",
    "selenoic_O_acid",
    "diselenoic_acid",
    "telluroic_Te_acid",
    "telluroic_O_acid",
    "ditelluroic_acid",
    # Imidic acid -- N-replacement ranks after the O/S/Se/Te
    # chalcogen-replacement block
    "imidic_acid",
    # Hydrazonic acid: =O -> =N-NH2 replacement acid; ranks
    # immediately below imidic acid per the sub-order.
    "hydrazonic_acid",
    # Hydroximic acid: =O -> =N-OH replacement acid; ranks below
    # hydrazonic acid. PIN is the N-hydroxy derivative of the imidic
    # acid, emitted by the dedicated hydroximic_acid handler
    # (SUFFIX_FORMS None, like hydroxamic_acid).
    "hydroximic_acid",
    # Carbamic acid (IUPAC -- retained acid name, rank with carboxylic acids
    "carbamic_acid",
    "sulfonic_acid",
    # W3-P04): FRN -OO- modification of the sulfonic acid
    # suffix. BB orders replacements 'unmodified acids followed by
    # -OO- > S > Se > Te', so it ranks DIRECTLY BELOW unmodified sulfonic_acid —
    # exactly as peroxy_acid sits directly below carboxylic_acid. (Governs only
    # near-zero-corpus S-vs-S polyfunctional ties; additive insert.)
    "sulfonoperoxoic_acid",
    # W3-P04): FRN -S- (=S/-SH) modification of the sulfonic
    # acid suffix. BB orders 'unmodified > -OO- > S > Se > Te', so it
    # ranks below sulfonoperoxoic_acid (-OO-) and above the =NH imidic block.
    "sulfonothioic_S_acid",
    # W3-P04): FRN =NH (imidic) modification of the sulfonic
    # acid suffix (-S(=O)(=NH)-OH). =NH ranks below the chalcogen modifications
    # (BB lists =NH after -OO-/S/Se/Te); grouped with its sulfonic parent,
    # above the sulfinic block (sulfonic > sulfinic).
    "sulfonimidic_acid",
    "sulfinic_acid",
    # W3-P04): FRN =NH (imidic) modification of the sulfinic
    # acid suffix (-S(=NH)-OH). Imidic (=NH) replacement ranks below the
    # unmodified acid (parallel to imidic_acid below the carboxylic chalcogen
    # block); grouped with its sulfinic parent.
    "sulfinimidic_acid",
    "sulfenic_acid",   # IUPAC: between sulfinic and phosphonic
    # a phase: Se/Te chalcogen-suffix acids, parallel to
    # sulfonic/sulfinic. Standalone naming is rank-independent; this relative
    # position only governs polyfunctional Se/Te-vs-S ties (S>Se>Te, -onic>-inic;
    # near-zero corpus) — additive, so the byte-positions of the rows above hold.
    "selenonic_acid",
    "seleninic_acid",
    "telluronic_acid",
    "tellurinic_acid",
    "phosphonic_acid",
    "phosphinic_acid",
    # organo-oxoacids of the heavier pnictogens, ranked immediately after
    # their phosphorus analogues per the element order (P > As > Sb) and
    # -onic before -inic within each element. Additive, so the byte-positions of
    # every row above are unchanged.
    "arsonic_acid",
    "arsinic_acid",
    "stibonic_acid",
    "stibinic_acid",
    # a phase: the trivalent -ous organo-oxoacids, ranked immediately after
    # their -onic/-inic (-ic) siblings per (an -ic acid is senior to its -ous
    # analogue) and P > As > Sb. Additive — byte-positions above are unchanged.
    "phosphonous_acid",
    "phosphinous_acid",
    "arsonous_acid",
    "arsinous_acid",
    "stibonous_acid",
    "stibinous_acid",
    "boronic_acid",    # boron acid
    # functional-group perception fix (169.7): free inorganic oxoacids — functional parents. Ranked in
    # the acid tier so they win PCG when perceived; NAMING is (downstream). Listed
    # so get_principal_group recognizes them (no KeyError) and they are not silently dropped.
    "phosphoric_acid",
    "sulfuric_acid",
    "nitric_acid",
    "carbonic_acid",

    # Acid derivatives
    "anhydride",
    "ester",
    # a phase Tier FRN-D: iminoester at ester-tier per AUDIT-FRN LOCK
    # ("alkyl alkanimidate" functional-class parallel to "alkyl alkanoate" per
    "iminoester",
    "thioester",
    # a phase Tier FRN-E: chalcogen-esters extension) — additive per internal notes
    "selenoester",
    "telluroester",
    # W3-P07 /: pseudoester (R-CO-O-Z, Z a Group-13/14/15
    # organyl) at the ester tier. Below the carboxylic 'ester' block; only governs
    # the near-zero-corpus pseudoester-vs-other-ester tie. Functional-class only.
    "pseudoester",
    # W3-P07: sulfonic/sulfinic esters (R-SO2-O-R' / R-S(=O)-O-R').
    # Below the carboxylic ester block, mirroring sulfonic_acid < carboxylic_acid
    #. Functional-class only ('alkyl alkanesulfonate' / '-sulfinate').
    "sulfonic_ester",
    "sulfinic_ester",
    "acid_chloride",
    "acid_bromide",
    "acid_fluoride",
    "acid_iodide",    # IUPAC: parallel to other acid halides
    # Wave2 T6c: acyl pseudohalides. class 10 orders
    # halides before pseudohalides, pseudohalogens as N3 > CN > NC > NCO.
    "acyl_azide",
    "acyl_cyanide",
    "acyl_isocyanate",
    # W3-P11 / /: acid halides of sulfonic /
    # sulfinic acids. Acid-halide tier class 8), below the carbon acyl
    # halides (S oxoacid-derivatives named after their C analogues) and above
    # amides. Functional-class handler-emitted ('ethanesulfonyl chloride').
    "sulfonyl_halide",
    "sulfinyl_halide",
    # -3: the cyanide of a sulfonic acid (R-SO2-C#N ->
    # 'methanesulfonyl cyanide'). Cyanide/pseudohalide functional class, ranked in
    # the acid-halide tier beside sulfonyl_halide and above the plain nitrile, so
    # the functional-class handler wins over the substitutive '-methanenitrile'.
    "sulfonyl_cyanide",
    # a phase: acyl halides of the imido / chalcogeno analogues of
    # carboxylic acid — R-C(=NH)-X / R-C(=S)-X / R-C(=Se)-X. Acid-halide tier
    # class 8), functional-class handler-emitted ('cyclohexanecarboximidoyl
    # chloride' / 'cyclohexanecarbothioyl chloride').
    "imidoyl_halide",
    "carbothioyl_halide",
    "carboselenoyl_halide",

    # Nitrogen acid derivatives
    "primary_amide",
    "secondary_amide",
    "tertiary_amide",
    # a phase Tier FRN-B: chalcogen amides + — additive per internal notes
    # thioamide ranks below amide and above sulfonamide; selenoamide/telluroamide parallel
    "thioamide",
    "selenoamide",
    "telluroamide",
    "primary_sulfonamide",
    "secondary_sulfonamide",
    "tertiary_sulfonamide",
    # Wave2 item 20): sulfonimidamide ranks just below
    # sulfonamide (item 19).
    "sulfonimidamide",
    # Wave-2 P1AM: Se analogue of sulfonimidamide, ranks with it.
    "selenonimidamide",
    # Task Y item 24): sulfinamide -SO-NH2. Table 6.1
    # orders 19 sulfonamide > 20 sulfonimidamide > 21 sulfonodiimidamide >
    # 22 sulfonohydrazonamide > 23 sulfonodihydrazonamide > 24 SULFINAMIDE >
    # 25 sulfinimidamide, so it sits BELOW sulfonimidamide and ABOVE
    # sulfinimidamide -- which is exactly here.
    "primary_sulfinamide",
    "secondary_sulfinamide",
    "tertiary_sulfinamide",
    # Wave2 item 25): sulfinimidamide -S(=NH)-NH2 ranks
    # just below sulfinamide (item 24).
    "sulfinimidamide",
    # Wave-2 P1AM: Se analogue of sulfinimidamide, ranks with it.
    "seleninimidamide",
    # C1: sulfono N-analogue; ranks with the sulfonamide family
    "sulfonohydrazide",
    "sulfinohydrazonohydrazide",   # (Table 6.1 item 41)
    # Wave2: hydrazidine's nomenclatural properties are those of
    # hydrazides; rank just ABOVE hydrazide so a molecule reading as both is the
    # hydrazidine.
    "hydrazidine",
    "hydrazide",
    # Wave2: thiohydrazide = chalcogen analogue of hydrazide, same
    # rank region (just below the oxo hydrazide).
    "thiohydrazide",
    "hydroxamic_acid",  # IUPAC: between hydrazides and imides
    "imide",
    "amidine",     # IUPAC: between imide and nitrile
    # Wave2, BB Table 6.1 item 18): amidrazone / hydrazonamide ranks
    # immediately below amidine (item 17).
    "hydrazonamide",
    # Wave-2 P1AM Task 7, BB 34460): the imidohydrazide amidrazone
    # tautomer — an imidamide (amidine) is senior to an imidohydrazide; keep it
    # adjacent to hydrazonamide (same amidrazone class, tautomer pair).
    "imidohydrazide",

    # tail #17: an ester of a phosphonic acid R-P(=O)(OR')2. Per an
    # ester ranks above a nitrile, so the phosphonate diester is the senior
    # parent when it co-occurs with a nitrile / isocyanide (diisopropyl
    # (1-cyano-1-isocyanoethyl)phosphonate). Functional-class named via
    # name_phosphate_ester (its phosphonate stem). Phosphonate diesters were
    # otherwise UNPERCEIVED (no matching FG) -> garbage / abstain.
    "phosphonate_diester",

    # a review: the ester of a phosphinic/arsinic/stibinic acid
    # R2E(=O)(OR') class 9, senior to a ring's hydroxy/amine).
    # Functional-class named via name_pnictogen_inate_ester ('methyl
    # diphenylphosphinate' / 'methyl diphenylarsinate'). Kept beside the other
    # pnictogen esters so the whole ester class ranks together.
    "phosphinate_ester",
    "arsinate_ester",
    "stibinate_ester",

    # Nitriles
    "nitrile",
    "isocyanide",

    # Carbonyls
    "aldehyde",
    "ketone",
    "thioaldehyde",
    # a phase Tier FRN-C aldehydes — additive per internal notes
    "selenoaldehyde",
    "telluroaldehyde",
    "thioketone",
    # a phase Tier FRN-C ketones — additive per internal notes
    "selenoketone",
    "telluroketone",

    # Alcohols and analogs
    "primary_alcohol",
    "secondary_alcohol",
    "tertiary_alcohol",
    "phenol",
    "enol",
    "alcohol",         #: generic catch-all, same seniority tier as other alcohols
    # Class 17 "Hydroxy compounds and chalcogen analogues" (alcohols/phenols/thiol/
    # selenol/tellurol) — all senior to class 18 hydroperoxide. name-hygiene fix/DEF- (169.7):
    # hydroperoxide MOVED below the chalcogen-ols (was incorrectly above thiol with a
    # factually-wrong "Class 19" comment). Verified the Blue Book lines
    # ~18190-18191: "17 Hydroxy compounds and chalcogen analogues" then "18 Hydroperoxides".
    "thiol",
    "selenol",
    "tellurol",        # functional-group perception fix (169.7): Te analogue of -ol/-selenol
    "hydroperoxide",   # class 18 (peroxol -OOH); junior to class-17 hydroxy/thiol
    # DD2 Fix C (Phase D): chalcogen hydroperoxol analogues / (3)),
    # same class-18 tier as hydroperoxide, in the chalcogen sub-order
    # -OOH (peroxol) > -SOH > -OSH > -SSH.
    "so_thioperoxol",  # R-S-OH -> -SO-thioperoxol (replaces retired sulfenic acid)
    "os_thioperoxol",  # R-O-SH -> -OS-thioperoxol
    "dithioperoxol",   # R-S-SH -> dithioperoxol (suffix) / disulfanyl (prefix)

    # Hydroxylamines (functional-group perception fix, 169.7: class 21; ranked just above amines)
    "hydroxylamine",

    # Amines
    "primary_amine",
    "secondary_amine",
    "tertiary_amine",
    "aromatic_amine",

    # Imines
    "imine",
    "oxime",
    "hydrazone",

    # Pseudohalides and special groups (prefix-only in IUPAC 2013)
    # Placed here so they are recognized but always junior to suffix-capable groups
    "azido",        #: prefix-only
    "azo",          #: prefix-only
    "cyanate",      #: prefix-only (pseudohalide)
    "thiocyanate",  #: prefix-only (pseudohalide)
    "diazo",           #: prefix-only
    "disulfide",       #: prefix-only
    "peroxide",        # DD2 Fix B (Phase D): R-OO-R' substitutive (R)peroxy prefix (1))
    "hydrazine_fg",    #: prefix-only
    # Wave2 T2b: terminal N-heteroatom preselected prefixes /
    # / — prefix-only, junior to all suffix groups
    "aminooxy",        # -O-NH2 aminooxy; no o-elision)
    "nitrooxy",        # -O-NO2 nitrooxy preselected; nitric-acid ester)
    "diazenyl",        # HN=N- diazenyl preselected)
    "n_fluoroamine",   # -NH-F fluoroamino compound prefix)
    "n_chloroamine",   # -NH-Cl chloroamino
    "n_bromoamine",    # -NH-Br bromoamino
    "n_iodoamine",     # -NH-I iodoamino

    # Sulfur oxidation states (functional class naming, lower seniority than amines)
    "sulfoxide",
    "sulfone",
    # -6I, the Blue Book): Se/Te analogues, same tier as sulfoxide/sulfone.
    "selenoxide",
    "selenone",
    "telluroxide",
    "tellurone",
    "thioether",  # Also called sulfide
    # functional-group perception fix (169.7): Se/Te ether analogues, same tier as thioether
    "selenoether",
    "telluroether",

    # Phosphorus compounds (functional class or substitutive naming)
    "phosphine_oxide",
    "phosphate_triester",
    "phosphate_diester",
    "phosphate_monoester",
    "phosphite_triester",
    # v50 B2: esters of sulfuric acid, functional-class named
    # ('dimethyl sulfate' / 'methyl hydrogen sulfate'). Kept beside the phosphate
    # esters (the sulfur analogue of an inorganic-oxoacid ester) so the whole
    # oxoacid-ester class ranks together and below the carbon acids: a sulfate
    # ester that co-occurs with a senior carboxylic acid is demoted (sulfooxy).
    "sulfate_diester",
    "sulfate_monoester",
    "tertiary_phosphine",
    "secondary_phosphine",
    "primary_phosphine",
]

# functional-group perception fix/ (a phase): groups present in SENIORITY_ORDER for ranking but
# which are ALWAYS detachable prefixes — never a principal (suffix) group.
# get_principal_group skips these so a prefix-only-only molecule is named with the
# group as a prefix (azidomethane) instead of dropping it. These mirror the
# "Pseudohalides and special groups (prefix-only)" block in SENIORITY_ORDER above.
_PREFIX_ONLY_PRINCIPAL = frozenset({
    "azido",         #
    "azo",           #
    "cyanate",       #
    "thiocyanate",   #
    "diazo",         #
    "isocyanide",    # (Wave-2 C2): PIN is the substitutive 'isocyano' prefix;
                     # no suffix form — pg='isocyanide' made the general handler
                     # misread N#C as an amine ('(methylamino)methane', suppressed)
    "disulfide",     #
    "peroxide",      # DD2 Fix B (Phase D): R-OO-R' is prefix-only ((R)peroxy), never a suffix
    "hydrazine_fg",  #
    # Wave2: sulfoxides/sulfones have NO suffix form
    # (SUFFIX_FORMS None) — the PIN is substitutive with sulfinyl/sulfonyl
    # PREFIXES on the senior parent (1-(methanesulfinyl)-2-(methylsulfanyl)-
    # ethane, BB 18284). Claiming PG starved the polyfunctional path
    # (functional-class 'ethyl methyl sulfoxide' dropped the -S-CH3).
    # Single-group molecules are named by the dedicated handler, whose
    # predicate now keys on FG presence with no PCG.
    "sulfoxide",
    "sulfone",
    # -6I, the Blue Book): Se/Te oxides are prefix-only exactly like
    # their S kin — the PIN is substitutive (seleninyl/selenonyl prefix on the
    # senior parent), never a suffix. Keeping them out would let
    # get_principal_group claim the FG as PCG and starve the dedicated handler.
    "selenoxide",
    "selenone",
    "telluroxide",
    "tellurone",
    # Wave2 D7c /: thioether/selenoether/telluroether have NO
    # suffix form (SUFFIX_FORMS None) — the PIN is substitutive (sulfanyl/
    # selanyl/tellanyl PREFIXES) or functional-class (dialkyl sulfide), never a
    # suffix. Keeping them out of _PREFIX_ONLY_PRINCIPAL let get_principal_group
    # claim 'thioether' as the PCG, which made select_parent treat -CH2-S-CH2-
    # O-CH3 on benzene as a chain PCG and pick an acyclic parent over the senior
    # ring, building the wrong '(heptylsulfanyl)methoxymethane'
    # (-suppressed to unknown). With no PCG claimed, the benzene ring is
    # correctly senior and name_benzene_derivative yields the nested PIN
    # '{[(methoxymethyl)sulfanyl]methyl}benzene'. The dedicated thioether handler
    # now keys on FG-presence-with-no-PCG (mirrors sulfoxide/sulfone above), so
    # simple dialkyl sulfides keep their functional-class name.
    "thioether",
    "selenoether",
    "telluroether",
    # Wave2 T2b: terminal N-heteroatom preselected prefixes /
    # /
    "aminooxy",
    "nitrooxy",
    "diazenyl",
    "n_fluoroamine",
    "n_chloroamine",
    "n_bromoamine",
    "n_iodoamine",
})

# /: Map subtypes to canonical parent for seniority comparison.
# IUPAC: All alcohol types have equal seniority; all amine types have equal seniority.
# Used only in get_principal_group for equalization; SENIORITY_ORDER stays intact per.
_SENIORITY_PARENT = {
    "primary_alcohol": "alcohol",
    "secondary_alcohol": "alcohol",
    "tertiary_alcohol": "alcohol",
    "phenol": "alcohol",
    "enol": "alcohol",
    "primary_amine": "amine",
    "secondary_amine": "amine",
    "tertiary_amine": "amine",
    "aromatic_amine": "amine",
}

# DD5: reverse index class -> subtypes, in SENIORITY_ORDER order.
# get_principal_group returns the UNION of match tuples over EVERY present subtype
# of the chosen group's equal-seniority class, so the whole class is expressed in
# ONE multiplied suffix (a primary + a secondary OH on the chain ->...-1,n-diol,
# not n-hydroxy-...-1-ol). Classes NOT in _SENIORITY_PARENT (acids, carbonyls, …)
# are their own singleton class -> the union is byte-identical for them.
_SENIORITY_CLASS_MEMBERS: Dict[str, List[str]] = {}
for _sub in SENIORITY_ORDER:
    _cls = _SENIORITY_PARENT.get(_sub)
    if _cls is not None:
        _SENIORITY_CLASS_MEMBERS.setdefault(_cls, []).append(_sub)
del _sub, _cls

# Characteristic heteroatom (atomic number) of each equalized PCG class, used to
# normalize the union's heterogeneous subtype match tuples to the senior subtype's
# (heteroatom, bearing-carbon) 2-tuple shape .
_CLASS_CHARACTERISTIC_Z: Dict[str, int] = {"alcohol": 8, "amine": 7}

# DD5 scope: classes for which get_principal_group unions across subtypes
# (and chains/prefixes count/skip the class). SCOPED to alcohols: a mixed
# primary+secondary OH chain is a diol. Amines are EXCLUDED — a mixed
# primary+secondary amine carries N-substituents on the secondary N (N-methyl…)
# that the general acyclic path does not express (it would mis-name the N-methyl
# as a 'methylamino' prefix); that needs the amine-assembler N-substituent path,
# a documented follow-on. Same-subtype amine diamines (NCC(N)C -> propane-1,2-
# diamine) already work via the single-subtype count and are unaffected.
# Wave-2 completion C2: amine joins the class union so a
# mixed primary+secondary diamine (CNCCCN) collects ALL amine matches and the
# polyamine assembler emits N-methylpropane-1,3-diamine instead of the
# double-counted '3-amino-3-(methylamino)propan-1-amine' (-suppressed).
_RC4_UNION_CLASSES = frozenset({"alcohol", "amine"})


def _normalize_pcg_match(mol, match, het_z: Optional[int]) -> tuple:
    """Normalize a PCG SMARTS match tuple to the canonical
    ``(characteristic_heteroatom, bearing_carbon)`` 2-tuple.

    Different alcohol/amine subtypes emit different-shaped SMARTS matches
    (``(O, C)`` vs ``(O, C, C, C)``); the union in get_principal_group returns
    them all under one ``pg_name``, so they must share ONE shape. Falls back to
    the original tuple when the heteroatom / bearing carbon cannot be located.
    """
    if het_z is None:
        return tuple(match)
    match_set = set(match)
    het = next(
        (a for a in match if mol.GetAtomWithIdx(a).GetAtomicNum() == het_z), None
    )
    if het is None:
        return tuple(match)
    # Wave-2 completion C2: among in-match carbon neighbours prefer the one
    # with the most heavy neighbours (the CHAIN carbon), tie-broken by index.
    # 'next(first-in-match)' picked an arbitrary carbon — for a secondary
    # amine in a diamine union (CNCCCN) it chose the METHYL, so the suffix
    # anchored off-chain and the branch was double-expressed downstream.
    _cands = [nb.GetIdx() for nb in mol.GetAtomWithIdx(het).GetNeighbors()
              if nb.GetAtomicNum() == 6 and nb.GetIdx() in match_set]
    carbon = None
    if _cands:
        carbon = min(
            _cands,
            key=lambda ci: (-sum(1 for nb in mol.GetAtomWithIdx(ci).GetNeighbors()
                                 if nb.GetAtomicNum() > 1), ci),
        )
    return (het, carbon) if carbon is not None else (het,)

# Suffix forms for principal groups
# Format: (chain_terminal_suffix, ring_attached_suffix), or None for functional-class-only
SUFFIX_FORMS = {
    "carboxylic_acid": ("oic acid", "carboxylic acid"),
    "peroxy_acid": ("peroxoic acid", "carboperoxoic acid"),  # / Table 4.3
    "thioic_S_acid": ("thioic S-acid", "carbothioic S-acid"),
    "thioic_O_acid": ("thioic O-acid", "carbothioic O-acid"),
    "dithioic_acid": ("dithioic acid", "carbodithioic acid"),
    # a phase Tier FRN-A: chalcogen-acid SUFFIX_FORMS (parallel to thioic_*_acid) per AUDIT-FRN
    "selenoic_Se_acid": ("selenoic Se-acid", "carboselenoic Se-acid"),
    "selenoic_O_acid": ("selenoic O-acid", "carboselenoic O-acid"),
    "diselenoic_acid": ("diselenoic acid", "carbodiselenoic acid"),
    "telluroic_Te_acid": ("telluroic Te-acid", "carbotelluroic Te-acid"),
    "telluroic_O_acid": ("telluroic O-acid", "carbotelluroic O-acid"),
    "ditelluroic_acid": ("ditelluroic acid", "carboditelluroic acid"),
    "imidic_acid": ("imidic acid", "carboximidic acid"),  # / Table 4.3
    # W3-P02-3 / Table 4.3): hydrazonic acid, the =N-NH2 analogue of
    # imidic acid. Chain 'hydrazonic acid' (methanehydrazonic acid); ring/appended-C
    # 'carbohydrazonic acid'. The C is always chain-terminal (see TERMINAL_FG_TYPES).
    "hydrazonic_acid": ("hydrazonic acid", "carbohydrazonic acid"),
    # W3-P02-5 Note): the '-hydroximic acid' suffix is general-only.
    # The PIN is the N-hydroxy derivative of the imidic acid (N-hydroxyethanimidic
    # acid), built by the dedicated hydroximic_acid handler — so NO substitutive
    # suffix here (None, exactly like hydroxamic_acid).
    "hydroximic_acid": None,
    "carbamic_acid": ("carbamic acid", "carbamic acid"),  # Retained name, same for chain/ring
    "sulfonic_acid": ("sulfonic acid", "sulfonic acid"),
    # W3-P04): FRN peroxy-modified sulfonic acid suffix; chain
    # stem + 'sulfonoperoxoic acid' (methanesulfonoperoxoic acid, PIN).
    "sulfonoperoxoic_acid": ("sulfonoperoxoic acid", "sulfonoperoxoic acid"),
    # W3-P04): FRN -SH modified sulfonic acid. The tautomer
    # symbol italic 'S' sits before 'acid' in the suffix (BB '-SO2-SH
    # sulfonothioic S-acid'); chain stem + 'sulfonothioic S-acid'
    # (ethanesulfonothioic S-acid, PIN).
    "sulfonothioic_S_acid": ("sulfonothioic S-acid", "sulfonothioic S-acid"),
    # W3-P04): FRN =NH modified sulfonic acid; chain stem +
    # 'sulfonimidic acid' (methanesulfonimidic acid). The N-hydroxy derivative
    # prepends 'N-hydroxy' via the dedicated handler.
    "sulfonimidic_acid": ("sulfonimidic acid", "sulfonimidic acid"),
    # W3-P04): FRN =NH modified sulfinic acid; chain stem +
    # 'sulfinimidic acid' (methanesulfinimidic acid, PIN).
    "sulfinimidic_acid": ("sulfinimidic acid", "sulfinimidic acid"),
    "sulfinic_acid": ("sulfinic acid", "sulfinic acid"),
    # a phase: Se/Te suffix-acid forms (chain stem + suffix, e.g.
    # ethaneselenonic acid), parallel to sulfonic.
    "selenonic_acid": ("selenonic acid", "selenonic acid"),
    "seleninic_acid": ("seleninic acid", "seleninic acid"),
    "telluronic_acid": ("telluronic acid", "telluronic acid"),
    "tellurinic_acid": ("tellurinic acid", "tellurinic acid"),
    "phosphonic_acid": ("phosphonic acid", "phosphonic acid"),
    "phosphinic_acid": ("phosphinic acid", "phosphinic acid"),
    "arsonic_acid": ("arsonic acid", "arsonic acid"),
    "arsinic_acid": ("arsinic acid", "arsinic acid"),
    "stibonic_acid": ("stibonic acid", "stibonic acid"),
    "stibinic_acid": ("stibinic acid", "stibinic acid"),
    # a phase: the trivalent -ous analogues (direct-return handlers own the
    # organyl forms, but the FGs must resolve here or get_principal_group KeyErrors).
    "phosphonous_acid": ("phosphonous acid", "phosphonous acid"),
    "phosphinous_acid": ("phosphinous acid", "phosphinous acid"),
    "arsonous_acid": ("arsonous acid", "arsonous acid"),
    "arsinous_acid": ("arsinous acid", "arsinous acid"),
    "stibonous_acid": ("stibonous acid", "stibonous acid"),
    "stibinous_acid": ("stibinous acid", "stibinous acid"),
    "anhydride": ("oic anhydride", "carboxylic anhydride"),
    "ester": ("oate", "carboxylate"),
    "acid_chloride": ("oyl chloride", "carbonyl chloride"),
    "acid_bromide": ("oyl bromide", "carbonyl bromide"),
    "acid_fluoride": ("oyl fluoride", "carbonyl fluoride"),
    # Wave2 T6c: acyl pseudohalides functional-class PINs)
    # W3-P11: sulfonyl/sulfinyl halide are functional-class handler-emitted
    # (name_sulfonyl_halide builds the two-word name directly); the generic
    # get_suffix path is never reached, so SUFFIX_FORMS is None (like the FRN
    # functional-class groups). PREFIX_FORMS is likewise None -> a demoted case
    # (a senior group co-present) fails closed rather than emit a wrong prefix.
    "sulfonyl_halide": None,
    "sulfinyl_halide": None,
    # -3: sulfonyl cyanide is functional-class handler-emitted
    # ('methanesulfonyl cyanide'); the generic get_suffix path is never reached.
    "sulfonyl_cyanide": None,
    # a phase: imidoyl / carbothioyl / carboselenoyl halides are functional-
    # class handler-emitted (name_imidoyl_thioyl_halide builds the two-word name);
    # None -> the generic get_suffix path is never used, and a demoted case fails
    # closed rather than emit a wrong prefix.
    "imidoyl_halide": None,
    "carbothioyl_halide": None,
    "carboselenoyl_halide": None,
    "acyl_azide": ("oyl azide", "carbonyl azide"),
    "acyl_cyanide": ("oyl cyanide", "carbonyl cyanide"),
    "acyl_isocyanate": ("oyl isocyanate", "carbonyl isocyanate"),
    "primary_amide": ("amide", "carboxamide"),
    "secondary_amide": ("amide", "carboxamide"),
    "tertiary_amide": ("amide", "carboxamide"),
    # a phase Tier FRN-B: chalcogen-amide SUFFIX_FORMS + PIN)
    "thioamide": ("thioamide", "carbothioamide"),
    "selenoamide": ("selenoamide", "carboselenoamide"),
    "telluroamide": ("telluroamide", "carbotelluroamide"),
    "primary_sulfonamide": ("sulfonamide", "sulfonamide"),
    "secondary_sulfonamide": ("sulfonamide", "sulfonamide"),
    "tertiary_sulfonamide": ("sulfonamide", "sulfonamide"),
    # Task Y, Table 6.1 item 24): same word for chain and ring,
    # exactly like sulfonamide -- 'methane' + 'sulfinamide' = methanesulfinamide,
    # 'benzene' + 'sulfinamide' = benzenesulfinamide. Worked PINs:
    # `butane-2-sulfinamide` , `N-hydroxypropane-1-sulfinamide` .
    "primary_sulfinamide": ("sulfinamide", "sulfinamide"),
    "secondary_sulfinamide": ("sulfinamide", "sulfinamide"),
    "tertiary_sulfinamide": ("sulfinamide", "sulfinamide"),
    # Wave2 item 20): same word for chain/ring (like sulfonamide);
    # methanesulfonimidamide / benzenesulfonimidamide.
    "sulfonimidamide": ("sulfonimidamide", "sulfonimidamide"),
    "selenonimidamide": ("selenonimidamide", "selenonimidamide"),  # (Se analogue of sulfonimidamide)
    "seleninimidamide": ("seleninimidamide", "seleninimidamide"),  # (Se analogue of sulfinimidamide)
    # C1: sulfono N-analogue -- parent stem + 'sulfonohydrazide'
    # ('methane' + 'sulfonohydrazide' = 'methanesulfonohydrazide'), exactly like
    # methanesulfonamide. Both simple and ring form are the same word.
    "sulfonohydrazide": ("sulfonohydrazide", "sulfonohydrazide"),
    "sulfinohydrazonohydrazide": ("sulfinohydrazonohydrazide", "sulfinohydrazonohydrazide"),  #
    "nitrile": ("nitrile", "carbonitrile"),
    "aldehyde": ("al", "carbaldehyde"),
    "ketone": ("one", "one"),
    "thioaldehyde": ("thial", "carbothialdehyde"),
    "thioketone": ("thione", "thione"),
    # a phase Tier FRN-C: chalcogen-aldehyde/ketone SUFFIX_FORMS per AUDIT-FRN LOCK
    # PIN short form -selenal / -tellural (parallel to -thial per
    # C- (V-2): the added-carbon ("carbo*") forms are carboselenaldehyde /
    # carbotelluraldehyde (Blue Book Table 28, BB ~line 18827/18829) — NOT the
    # "carboseleno-/carbotelluro-aldehyde" (extra 'o') typo, which mirrored the
    # acid/amide infix form by mistake. cf. the correct "carbothialdehyde" above.
    "selenoaldehyde": ("selenal", "carboselenaldehyde"),
    "telluroaldehyde": ("tellural", "carbotelluraldehyde"),
    # PIN suffix form -selone / -tellone (dialkyl-word form fails OPSIN per AUDIT LOCK + RESEARCH)
    "selenoketone": ("selone", "selone"),
    "telluroketone": ("tellone", "tellone"),
    "primary_alcohol": ("ol", "ol"),
    "secondary_alcohol": ("ol", "ol"),
    "tertiary_alcohol": ("ol", "ol"),
    "phenol": ("ol", "ol"),
    "enol": ("ol", "ol"),
    "alcohol": ("ol", "ol"),  #: generic catch-all suffix form
    "thiol": ("thiol", "thiol"),
    "selenol": ("selenol", "selenol"),
    "tellurol": ("tellurol", "tellurol"),  # functional-group perception fix (169.7): Te -ol analogue
    "hydroperoxide": ("peroxol", "peroxol"),
    # DD2 Fix C (Phase D, / (3)): chalcogen peroxol analogues.
    # The italic chalcogen-pair descriptor ('SO'/'OS') is carried in the suffix
    # string; the emission layer inserts the carbon locant before it
    # (methane-SO-thioperoxol; propane-1-SO-thioperoxol).
    "so_thioperoxol": ("SO-thioperoxol", "SO-thioperoxol"),
    "os_thioperoxol": ("OS-thioperoxol", "OS-thioperoxol"),
    "dithioperoxol": ("dithioperoxol", "dithioperoxol"),
    # functional-group perception fix (169.7): hydroxylamine + free oxoacids + Se/Te ethers are named via
    # their handler / functional-parent / / functional-class paths,
    # NOT as a chain suffix — None per the SUFFIX_FORMS contract (like thioether).
    "hydroxylamine": None,
    "phosphoric_acid": None,
    "sulfuric_acid": None,
    "nitric_acid": None,
    "carbonic_acid": None,
    "selenoether": None,
    "telluroether": None,
    "primary_amine": ("amine", "amine"),
    "secondary_amine": ("amine", "amine"),
    "tertiary_amine": ("amine", "amine"),
    "aromatic_amine": ("amine", "amine"),
    "imine": ("imine", "imine"),
    "oxime": ("oxime", "oxime"),           # functional class suffix
    "hydrazone": ("hydrazone", "hydrazone"),  # functional class suffix
    "boronic_acid": ("boronic acid", "boronic acid"),  # retained acid form
    # R8c /: hydroxamic acid is an amide with N-hydroxy;
    # PIN is 'N-hydroxy<stem>amide', not the retained 'hydroxamic acid' suffix.
    # Suffix path suppressed (None) — composer dispatches via _assemble_hydroxamic_name.
    "hydroxamic_acid": None,
    # --- a phase-01: close SENIORITY_ORDER suffix gaps ---
    # FGs with real suffix forms
    #: acyclic hydrazide suffix is stem + 'hydrazide' (no '-o-' infix);
    # 'pentanoic acid' -> 'pentanehydrazide' (h is consonant, no vowel elision).
    # Ring form is 'carbohydrazide' (appended-carbon nomenclature, e.g. cyclopentane-
    # carbohydrazide). The characteristic C is always terminal (chain-end), so
    # locant-1 is implicit and elided (see TERMINAL_FG_TYPES in naming_utils.py).
    "hydrazide": ("hydrazide", "carbohydrazide"),  # IUPAC
    # Wave2: thiohydrazide, chalcogen analogue of hydrazide.
    "thiohydrazide": ("thiohydrazide", "carbothiohydrazide"),
    # Wave2: hydrazidine — chain 'hydrazonohydrazide',
    # ring/appended-C 'carbohydrazonohydrazide' (parallel to hydrazide).
    "hydrazidine": ("hydrazonohydrazide", "carbohydrazonohydrazide"),
    "imide": ("imide", "dicarboximide"),  # IUPAC
    # FGs with functional class naming only (no substitutive suffix)
    "thioester": None,           # IUPAC: functional class naming (S-alkyl alkanethioate)
    # a phase Tier FRN-D + FRN-E: functional-class (handler-emitted) — SUFFIX_FORMS = None
    "iminoester": None,          #: emitted via handlers/imidate.py per internal notes
    "selenoester": None,         #: functional-class "Se-alkyl alkaneselenoate"
    "telluroester": None,        #: functional-class "Te-alkyl alkanetelluroate"
    "pseudoester": None,         # W3-P07: functional-class "Zyl acylate" (handler-emitted)
    "sulfonic_ester": None,      # W3-P07: functional-class "alkyl alkanesulfonate"
    "sulfinic_ester": None,      # W3-P07: functional-class "alkyl alkanesulfinate"
    "isocyanide": None,          # IUPAC 2013: prefix-only (isocyano)
    "sulfoxide": None,           # IUPAC: functional class naming (dialkyl sulfoxide)
    "sulfone": None,             # IUPAC: functional class naming (dialkyl sulfone)
    # -6I, the Blue Book): Se/Te oxides have no suffix form (parallel to S)
    "selenoxide": None,
    "selenone": None,
    "telluroxide": None,
    "tellurone": None,
    "thioether": None,           # IUPAC: functional class naming (dialkyl sulfide)
    "phosphine_oxide": None,     # IUPAC: functional class naming
    "phosphate_triester": None,  # IUPAC: functional class naming
    "phosphate_diester": None,   # IUPAC: functional class naming
    "phosphate_monoester": None, # IUPAC: substitutive prefix only (phosphonooxy)
    "phosphite_triester": None,  # IUPAC: functional class naming (... phosphite)
    "phosphonate_diester": None, # IUPAC: functional class naming (... phosphonate)
    # v50 B2: sulfuric-acid esters -> functional-class naming (no chain suffix)
    "sulfate_diester": None,     # IUPAC ('dialkyl sulfate')
    "sulfate_monoester": None,   # IUPAC ('alkyl hydrogen sulfate')
    # a review: functional-class ester names (no suffix form)
    "phosphinate_ester": None,   # IUPAC (... phosphinate)
    "arsinate_ester": None,      # IUPAC (... arsinate)
    "stibinate_ester": None,     # IUPAC (... stibinate)
    "tertiary_phosphine": None,  # IUPAC: parent hydride naming (phosphane)
    "secondary_phosphine": None, # IUPAC: parent hydride naming
    "primary_phosphine": None,   # IUPAC: parent hydride naming
    # Prefix-only FGs added to SENIORITY_ORDER (no suffix form)
    "azido": None,               # IUPAC: prefix-only
    "azo": None,                 # IUPAC: prefix-only
    "cyanate": None,             # IUPAC: prefix-only (pseudohalide)
    "thiocyanate": None,         # IUPAC: prefix-only (pseudohalide)
    # a phase: 6 new FG classes
    "acid_iodide": ("oyl iodide", "carbonyl iodide"),  # IUPAC
    "amidine": ("imidamide", "carboximidamide"),        # IUPAC
    # Wave2, BB Table 6.1 item 18): amidrazone — chain
    # 'hydrazonamide' (ethanehydrazonamide), ring/appended-C 'carbohydrazonamide'
    # (benzenecarbohydrazonamide).
    "hydrazonamide": ("hydrazonamide", "carbohydrazonamide"),
    # Wave-2 P1AM Task 7: amidrazone =NH tautomer — chain
    # 'imidohydrazide', ring/appended-C 'carboximidohydrazide'.
    "imidohydrazide": ("imidohydrazide", "carboximidohydrazide"),
    # Wave2 item 25): sulfinimidamide, same word chain/ring
    # (like sulfonamide/sulfonimidamide).
    "sulfinimidamide": ("sulfinimidamide", "sulfinimidamide"),
    # DD2 Fix C (Phase D): R-S-OH is now perceived as `so_thioperoxol` and emitted
    # as `-SO-thioperoxol` PIN). `sulfenic_acid` is retired from the PIN path
    # (perception suppresses it on overlap) — this legacy suffix is unreachable for
    # R-S-OH and kept only as a defensive label.
    "sulfenic_acid": ("sulfenic acid", "sulfenic acid"),  # IUPAC (retired PIN; see so_thioperoxol)
    "diazo": None,               #: prefix-only
    "disulfide": None,           #: prefix-only
    "peroxide": None,            # DD2 Fix B (Phase D): prefix-only ((R)peroxy)
    "hydrazine_fg": None,        #: prefix-only
    # Wave2 T2b: terminal N-heteroatom preselected prefixes — prefix-only
    "aminooxy": None,            #
    "nitrooxy": None,            # (prefix-only)
    "diazenyl": None,            #
    "n_fluoroamine": None,       #
    "n_chloroamine": None,       #
    "n_bromoamine": None,        #
    "n_iodoamine": None,         #
}

# Prefix forms for non-principal groups
# Audit (a phase-01): verified all SENIORITY_ORDER entries have a PREFIX_FORMS
# key. None entries are genuinely functional-class-only (no IUPAC prefix form).
PREFIX_FORMS = {
    "carboxylic_acid": "carboxy",
    "thioic_S_acid": "sulfanylcarbonyl",  # IUPAC: S-acid prefix (-C(=O)SH)
    "thioic_O_acid": "carbothioyl",      # IUPAC: O-acid prefix (-C(=S)OH)
    "dithioic_acid": "dithiocarboxy",     # IUPAC: dithioic acid prefix (-C(=S)SH)
    # a phase Tier FRN-A: chalcogen-acid PREFIX_FORMS (parallel to thioic_*_acid) per AUDIT-FRN
    "selenoic_Se_acid": "selanylcarbonyl",
    "selenoic_O_acid": "carboselenoyl",
    "diselenoic_acid": "diselenocarboxy",
    "telluroic_Te_acid": "tellanylcarbonyl",
    "telluroic_O_acid": "carbotelluroyl",
    "ditelluroic_acid": "ditellurocarboxy",
    "carbamic_acid": "carbamoyloxy",  # When not principal group
    "sulfonic_acid": "sulfo",
    # W3-P04: demoted-prefix case of the FRN sulfur-oxo-acids fails
    # closed (None), exactly like imidic_acid — these are in-scope only as the
    # principal group; a wrong prefix would be worse than fail-closed abstention.
    "sulfonoperoxoic_acid": None,
    "sulfonothioic_S_acid": None,  # W3-P04: demoted-prefix fails closed
    "sulfonimidic_acid": None,     # W3-P04: demoted-prefix fails closed
    "sulfinimidic_acid": None,     # W3-P04: demoted-prefix fails closed
    "sulfinic_acid": "sulfino",
    # a phase: Se/Te prefix forms (demoted when a senior group present)
    "selenonic_acid": "selenono",
    "seleninic_acid": "selenino",
    "telluronic_acid": "tellurono",
    "tellurinic_acid": "tellurino",
    "aldehyde": "oxo",  # or "formyl" for terminal
    "ketone": "oxo",
    "thioketone": "sulfanylidene",  #: =S as non-principal prefix
    # a phase Tier FRN-C: =Se / =Te ketone non-principal prefix (parallel to sulfanylidene) per AUDIT-FRN
    "selenoketone": "selanylidene",
    "telluroketone": "tellanylidene",
    "primary_alcohol": "hydroxy",
    "secondary_alcohol": "hydroxy",
    "tertiary_alcohol": "hydroxy",
    "phenol": "hydroxy",
    "enol": "hydroxy",
    "alcohol": "hydroxy",  #: generic catch-all prefix form
    "thiol": "sulfanyl",
    "selenol": "selanyl",
    "tellurol": "tellanyl",   # functional-group perception fix (169.7): Te analogue of sulfanyl/selanyl
    "hydroperoxide": "hydroperoxy",
    # DD2 Fix C (Phase D): demoted-prefix forms of the chalcogen peroxol analogues
    # /. -S-OH -> hydroxysulfanyl; -O-SH -> sulfanyloxy;
    # -S-SH -> disulfanyl (terminal disulfide,.
    "so_thioperoxol": "hydroxysulfanyl",
    "os_thioperoxol": "sulfanyloxy",
    "dithioperoxol": "disulfanyl",
    # functional-group perception fix (169.7): hydroxylamine + free oxoacids are functional parents
    # /; not expressed as detachable prefixes (None, like thioether).
    "hydroxylamine": None,
    "phosphoric_acid": None,
    "sulfuric_acid": None,
    "nitric_acid": None,
    "carbonic_acid": None,
    "primary_amine": "amino",
    "secondary_amine": "amino",
    "tertiary_amine": "amino",
    "aromatic_amine": "amino",
    "imine": "imino",
    "oxime": "hydroxyimino",
    "hydrazone": "hydrazinylidene",
    "isocyanate": "isocyanato",
    "isothiocyanate": "isothiocyanato",
    "urea": "carbamoylamino",
    "thiourea": "carbamothioylamino",  # Wave-2 completion (thio-urea prefix)
    "guanidine": "guanidino",
    # Wave2 T2b: terminal N-heteroatom preselected prefixes. BB verbatim:
    # 'aminooxy (preselected prefix) (note that there is no elision of the
    # final letter o of amino)' + '2-(aminooxy)ethan-1-amine (PIN)'
    #; 'diazenyl (preselected prefix; see '
    #; '-NH-Cl chloroamino (preselected prefix)'.
    "aminooxy": "aminooxy",
    # W3-P11: -O-NO2 preselected prefix 'nitrooxy' (BB verbatim);
    # compound '(nitrooxy)' takes enclosing marks (see needs_brackets).
    "nitrooxy": "nitrooxy",
    "diazenyl": "diazenyl",
    "n_fluoroamine": "fluoroamino",
    "n_chloroamine": "chloroamino",
    "n_bromoamine": "bromoamino",
    "n_iodoamine": "iodoamino",
    "carbamate": None,                  # functional class only
    "boronic_acid": "borono",           #: -B(OH)2 preselected prefix (not 'dihydroxyboranyl')
    "n_oxide_aromatic": None,           # functional class only
    "n_oxide_aliphatic": None,          # functional class only
    "primary_sulfonamide": "sulfamoyl",
    "secondary_sulfonamide": "sulfamoyl",
    "tertiary_sulfonamide": "sulfamoyl",
    # Task Y: -S(=O)-NH2 as a detachable prefix. NOT the naive parallel to
    # 'sulfamoyl' -- BB lists `aminosulfinyl* (not sulfinamoyl) | H2N-S(O)- |
    # `, i.e. it names 'sulfinamoyl' explicitly as the non-preferred
    # form, and redirects "sulfinamoyl: see aminosulfinyl*". Confirmed in
    # use by the PIN at, `3-[(aminosulfinyl)oxy]propanoic acid (PIN)`.
    # Required so a DEMOTED sulfinamide is not silently dropped by the
    # `no_fg_prefix_form` skip (substituent_no_prefix_form, _handler_shared.py /
    # polyfunctional.py) when a senior group takes the suffix.
    "primary_sulfinamide": "aminosulfinyl",
    "secondary_sulfinamide": "aminosulfinyl",
    "tertiary_sulfinamide": "aminosulfinyl",
    # C1: -SO2-NH-NH2 as a substituent prefix (defensive; target
    # compounds are mono-functional so it is normally the principal suffix).
    "sulfonohydrazide": "hydrazinesulfonyl",
    # (BB 34346/55487, W2E-P1FG Task 13): the PRESELECTED prefix
    # for -S(O)(=NH)-NH2 is 'S-aminosulfonimidoyl' (the italic 'S' locant
    # disambiguates substitution on the imido N); bare 'sulfonimidoyl' is the
    # divalent connector, so it must NOT be the FG prefix here. Mirrors the
    # sulfinimidamide row ('S-aminosulfinimidoyl') above.
    "sulfonimidamide": "S-aminosulfonimidoyl",
    "nitrile": "cyano",
    "isocyanide": "isocyano",
    # Amides as non-principal group prefix (IUPAC method 2)
    # Only primary_amide (-CONH2) gets carbamoyl here; secondary/tertiary amides
    # are already handled via the acylamino naming pathway in the pipeline.
    # Adding carbamoyl for sec/tert causes double-naming (e.g., "ethanoylamino" + "carbamoyl").
    "primary_amide": "carbamoyl",
    # a phase Tier FRN-B: chalcogen-amide PREFIX_FORMS (parallel to primary_amide carbamoyl) per AUDIT-FRN
    "thioamide": "carbamothioyl",
    "selenoamide": "carbamoselenoyl",
    "telluroamide": "carbamotelluroyl",
    # Halogens (always prefixes)
    "fluoro": "fluoro",
    "chloro": "chloro",
    "bromo": "bromo",
    "iodo": "iodo",
    # Other
    "nitro": "nitro",
    "nitroso": "nitroso",
    "azido": "azido",
    # Acid halides as non-principal group prefix (IUPAC
    "acid_chloride": "carbonochloridoyl",
    "acid_bromide": "bromocarbonyl",
    "acid_fluoride": "fluorocarbonyl",
    # Wave2 T6c: acyl-pseudohalide prefixes, parallel to
    # bromocarbonyl/fluorocarbonyl above)
    # W3-P11: demoted sulfonyl/sulfinyl halide prefix fails closed (None) — the
    # -SO2-X substituent prefix ('{halide}sulfonyl') is halide-dependent and the
    # target compounds are mono-functional (always the principal group).
    "sulfonyl_halide": None,
    "sulfinyl_halide": None,
    # -3: demoted sulfonyl cyanide prefix fails closed (None) — the target
    # compounds are mono-functional (always the principal group).
    "sulfonyl_cyanide": None,
    # a phase: demoted imidoyl / carbothioyl / carboselenoyl halide prefix
    # fails closed (None) — the target compounds are mono-functional (the halide is
    # always the principal group), and a halide-dependent prefix is not derived here.
    "imidoyl_halide": None,
    "carbothioyl_halide": None,
    "carboselenoyl_halide": None,
    "acyl_azide": "azidocarbonyl",
    "acyl_cyanide": "cyanocarbonyl",
    "acyl_isocyanate": "isocyanatocarbonyl",
    # Esters (handled specially in polyfunctional.py as acyloxy prefixes)
    "ester": None,  # Esters use alkoxycarbonyl prefix (generated in polyfunctional.py get_fg_prefix_form)
    "pseudoester": None,  # W3-P07: functional-class only (no principal-group prefix)
    "sulfonic_ester": None,  # W3-P07: functional-class only
    "sulfinic_ester": None,  # W3-P07: functional-class only
    # Ethers and thioethers
    "ether": None,  # Named by substitution: methoxy, ethoxy, etc.
    "thioether": None,  # IUPAC: functional class naming (dialkyl sulfide)
    # functional-group perception fix (169.7): Se/Te ethers use the dynamic (alkyl)selanyl/tellanyl generator
    # in substituent_prefix_forms.get_substituent_prefix_form (parallel to thioether).
    "selenoether": None,  # IUPAC: (alkyl)selanyl substitutive prefix
    "telluroether": None,  # IUPAC: (alkyl)tellanyl substitutive prefix
    # Sulfur oxidation states (IUPAC
    "sulfoxide": "sulfinyl",  # IUPAC: bivalent prefix for -S(=O)-
    "sulfone": "sulfonyl",    # IUPAC: bivalent prefix for -S(=O)2-
    # -6I, the Blue Book): Se/Te bivalent prefixes, from the
    # seleninic/selenonic (tellurinic/telluronic) acid stems.
    "selenoxide": "seleninyl",   # -Se(=O)-
    "selenone": "selenonyl",     # -Se(=O)2-
    "telluroxide": "tellurinyl",  # -Te(=O)-
    "tellurone": "telluronyl",    # -Te(=O)2-
    # Phosphorus compounds
    "phosphonic_acid": "phosphono",
    "phosphinic_acid": "phosphino",
    # heavier-pnictogen analogues. BB L36030-36031 gives -As(O)(OH)2 ->
    # 'arsono' and -Sb(O)(OH)2 -> 'stibono' as PRESELECTED prefixes; BB L15858
    # admits 'arsino'/'stibino' alongside 'phosphino'.
    "arsonic_acid": "arsono",
    "arsinic_acid": "arsino",
    "stibonic_acid": "stibono",
    "stibinic_acid": "stibino",
    # a phase: the trivalent -ous acids are principal-group producers
    # (direct-return handlers in inner_dispatch -2370); no verified static
    # substituent-prefix spelling, so None here (degrade to the dynamic/functional
    # path if ever seen as a non-principal substituent) — parallel to phosphine_oxide.
    "phosphonous_acid": None,
    "phosphinous_acid": None,
    "arsonous_acid": None,
    "arsinous_acid": None,
    "stibonous_acid": None,
    "stibinous_acid": None,
    # Phosphine oxide and phosphates use functional class naming
    "phosphine_oxide": None,
    "phosphate_triester": None,
    "phosphate_diester": None,
    "phosphate_monoester": "phosphonooxy",  # IUPAC
    "phosphite_triester": None,  # IUPAC: functional class naming
    "phosphonate_diester": None,  # IUPAC: functional class naming
    # v50 B2: sulfuric-acid esters. Functional-class named as a whole molecule;
    # a demoted sulfate ester on a senior parent is a 'sulfooxy'-type oxy prefix
    # emitted by the dedicated rules/sulfur_oxoacid.py path, NOT a naive per-FG
    # string here (which cannot carry the ester organyl / attachment). Kept None.
    "sulfate_diester": None,      # IUPAC functional class naming
    "sulfate_monoester": None,    # IUPAC functional class naming
    # a review: functional-class ester (no prefix form)
    "phosphinate_ester": None,   # IUPAC functional class naming
    "arsinate_ester": None,      # IUPAC functional class naming
    "stibinate_ester": None,     # IUPAC functional class naming

    "tertiary_phosphine": None,
    "secondary_phosphine": None,
    "primary_phosphine": None,
    #: New FG prefix forms
    "hydroxamic_acid": "N-hydroxyamido",  # IUPAC
    "cyanate": "cyanato",                 # IUPAC
    "thiocyanate": "thiocyanato",         # IUPAC
    "azo": "diazenyl",                    # IUPAC
    # --- a phase-01: close SENIORITY_ORDER prefix gaps ---
    "anhydride": None,            # IUPAC: functional class naming only
    "secondary_amide": None,      # Named via acylamino pathway in universal pipeline
    "tertiary_amide": None,       # Named via acylamino pathway in universal pipeline
    "hydrazide": "hydrazinecarbonyl",  # IUPAC
    # Wave2 defensive non-principal prefixes (target compounds are
    # mono-functional -> the suffix path is used; these guard the demoted case).
    "thiohydrazide": "hydrazinecarbothioyl",       # / Table 4.4
    "hydrazidine": "hydrazinecarbohydrazonoyl",    # (BB 56105)
    "hydrazonamide": "carbamohydrazonoyl",         # (BB 34498)
    "imidohydrazide": None,  # demoted case: Task 6 owns the chain-end split; ring prefix not yet built -- fail closed
    # (BB 34352/55484): the preselected prefix for the WHOLE
    # H2N-S(=NH)- group is 'S-aminosulfinimidoyl' -- the old bare
    # 'sulfinimidoyl' is the divalent -S(=NH)- connector and
    # silently dropped the amino N (wrong name, RT-gate suppressed).
    "sulfinimidamide": "S-aminosulfinimidoyl",
    "peroxy_acid": None,          # demoted case fails closed
    "imidic_acid": None,          # demoted case fails closed
    # W3-P02-4: demoted hydrazonic acid at a chain end splits into
    # 'hydroxy' + 'hydrazinylidene' prefixes on the geminal locant — a two-prefix
    # decomposition emitted by name_polyfunctional's chain-end block, NOT a single
    # static prefix. None here (handled specially, like imidic_acid).
    "hydrazonic_acid": None,
    # W3-P02-6: demoted hydroximic acid at a chain end splits into
    # 'hydroxy' + 'hydroxyimino' prefixes on the geminal locant (two-prefix
    # decomposition emitted by name_polyfunctional's chain-end block). None here.
    "hydroximic_acid": None,
    "sulfinohydrazonohydrazide": None,  # demoted case fails closed
    "selenonimidamide": None,   # demoted case fails closed
    "seleninimidamide": None,   # demoted case fails closed
    "imide": None,                # Named as heterocyclic ring substituent
    "thioaldehyde": "thioxo",     # IUPAC: =S as non-principal prefix (parallel to "oxo")
    # a phase Tier FRN-C: =Se / =Te non-principal prefix (parallel to thioxo) per AUDIT-FRN
    "selenoaldehyde": "selenoxo",
    "telluroaldehyde": "telluroxo",
    "thioester": None,            # W3-P08: functional-class via chalcogen_ester handler ("S-alkyl alkanethioate"); no prefix form
    # a phase Tier FRN-D + FRN-E: functional-class — no prefix form
    "iminoester": None,
    "selenoester": None,
    "telluroester": None,
    # a phase: 6 new FG classes
    "acid_iodide": "iodocarbonyl",     # IUPAC
    "amidine": "carbamimidoyl",         # IUPAC (was "amidino" — wrong per BB
    "sulfenic_acid": "sulfeno",        # IUPAC
    "diazo": "diazo",                  #
    "disulfide": "disulfanediyl",      # (divalent bridge; substitutive (R)disulfanyl via get_disulfanyl_prefix)
    # DD2 Fix B (Phase D): R-OO-R' substituent prefix is (R)peroxy, generated
    # dynamically by substituent_prefix_forms.get_peroxy_prefix (None here, like ether/ester).
    "peroxide": None,                  # (1): (R)peroxy via get_peroxy_prefix
    "hydrazine_fg": "hydrazinyl",      #
}


def get_principal_group(
    mol,
    functional_groups: Dict[str, List[tuple]]
) -> Tuple[Optional[str], List[tuple]]:
    """
    Determine the principal characteristic group.

    The principal group is the highest-seniority functional group
    that will be expressed as a suffix in the name.

    Args:
        mol: RDKit Mol object
        functional_groups: Dict from detect_functional_groups

    Returns:
        Tuple of (group_name, list_of_atom_index_tuples)
        Returns (None, ) if no suffix-capable group found
    """
    for fg_name in SENIORITY_ORDER:
        if fg_name in functional_groups and functional_groups[fg_name]:
            # functional-group perception fix/ (169.7): a principal characteristic group MUST be
            # suffix-capable. The pseudohalide / special prefix-only groups
            # (azido/azo/cyanate/thiocyanate/diazo/disulfide/hydrazine) are kept in
            # SENIORITY_ORDER for RANKING but can never be the principal group — they
            # are always detachable prefixes /////.
            # Skipping them here makes a molecule whose ONLY group is prefix-only return
            # (None, ), so the FG is emitted as a prefix (e.g. azidomethane) instead
            # of being consumed-as-principal-then-dropped (-> bare 'methane'). Matches
            # this function's contract: "the group that will be expressed as a suffix".
            if fg_name in _PREFIX_ONLY_PRINCIPAL:
                continue
            # DD5: the principal characteristic group is the WHOLE
            # equal-seniority class. When fg_name maps to an equalized class
            # (alcohol/amine), return the UNION of match tuples over every present
            # subtype, normalized to the senior subtype's (heteroatom, bearing-C)
            # shape, so downstream suffix counting (a primary + secondary OH ->
            # pentane-1,4-diol) and chain selection see all instances. Non-equalized
            # groups (acids, carbonyls, …) have no class entry -> unchanged.
            parent_class = _SENIORITY_PARENT.get(fg_name)
            members = _SENIORITY_CLASS_MEMBERS.get(parent_class)
            if members is None or parent_class not in _RC4_UNION_CLASSES:
                return fg_name, functional_groups[fg_name]
            het_z = _CLASS_CHARACTERISTIC_Z.get(parent_class)
            matches: List[tuple] = []
            seen = set()
            for subtype in members:
                for match in functional_groups.get(subtype, []):
                    norm = _normalize_pcg_match(mol, match, het_z)
                    key = norm[0] if norm else tuple(match)
                    if key not in seen:
                        seen.add(key)
                        matches.append(norm)
            return fg_name, matches

    return None, []


def get_suffix(fg_name: str, is_ring: bool = False) -> Optional[str]:
    """
    Get the suffix form for a functional group.

    Args:
        fg_name: Name of functional group
        is_ring: True if the group is attached to a ring (not terminal on chain)

    Returns:
        Suffix string, or None if group has no suffix form
    """
    if fg_name not in SUFFIX_FORMS:
        return None

    entry = SUFFIX_FORMS[fg_name]
    if entry is None:
        return None  # Functional class naming only -- no substitutive suffix

    chain_suffix, ring_suffix = entry
    return ring_suffix if is_ring else chain_suffix


def get_prefix(fg_name: str) -> Optional[str]:
    """
    Get the prefix form for a functional group.

    Used when the group is not the principal group.

    Args:
        fg_name: Name of functional group

    Returns:
        Prefix string, or None if group has no standard prefix form
    """
    return PREFIX_FORMS.get(fg_name)


def is_suffix_group(fg_name: str) -> bool:
    """
    Check if a functional group can be expressed as a suffix.

    Args:
        fg_name: Name of functional group

    Returns:
        True if group can be a suffix (is in seniority order)
    """
    return fg_name in SENIORITY_ORDER


def compare_seniority(fg1: str, fg2: str) -> int:
    """
    Compare seniority of two functional groups.

    Args:
        fg1, fg2: Names of functional groups

    Returns:
        -1 if fg1 is higher seniority
         0 if equal seniority (or both not in list)
         1 if fg2 is higher seniority
    """
    try:
        idx1 = SENIORITY_ORDER.index(fg1)
    except ValueError:
        idx1 = len(SENIORITY_ORDER)  # Not in list = lowest priority

    try:
        idx2 = SENIORITY_ORDER.index(fg2)
    except ValueError:
        idx2 = len(SENIORITY_ORDER)

    if idx1 < idx2:
        return -1  # fg1 is higher (lower index = higher priority)
    elif idx1 > idx2:
        return 1   # fg2 is higher
    return 0       # Equal


def get_all_prefix_groups(
    functional_groups: Dict[str, List[tuple]],
    principal_group: Optional[str]
) -> Dict[str, List[tuple]]:
    """
    Get all functional groups that should be named as prefixes.

    This includes all groups except the principal group.

    Args:
        functional_groups: Dict from detect_functional_groups
        principal_group: Name of the principal group (or None)

    Returns:
        Dict of functional group names to their atom indices,
        excluding the principal group
    """
    prefix_groups = {}

    for fg_name, matches in functional_groups.items():
        if fg_name == principal_group:
            continue
        if matches:
            prefix_groups[fg_name] = matches

    return prefix_groups
