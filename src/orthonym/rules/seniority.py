"""
Functional group seniority rules for principal group selection.

Based on IUPAC 2013 Blue Book to.
The principal group (highest seniority) becomes the suffix;
all other groups become prefixes.
"""

from typing import Dict, List, Optional, Tuple

from .n_substituted_acids import CN_GROUPS, group_at_rank, rank_group
from .amide_parent import senior_carboxamide_group

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
    #: '[OX2H][OX2][#6]' leads with the hydroxy oxygen; the locant-bearing atom is the carbon (index 2)
    "hydroperoxide": [2],
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
    # Hydroximic acid, the Blue Book): the N-hydroxy derivative of the imidic
    # acid, so it has the imidic acid's rank class 7a,:18172: acids "each followed
    # in turn by the corresponding peroxy, imidic, and hydrazonic acids. Chalcogen analogues follow
    # each of the corresponding oxygen acids"). The rank of one with an S, Se or Te acid group is that
    # of the imidothioic, imidoselenoic or imidotelluroic acid: ``_head_principal_group`` ranks the
    # group by the chalcogen, this entry is the O acid's place and the one the rankers that read the
    # index see. (It stood below the hydrazonic acids, a place the group has no claim to.)
    "hydroximic_acid",
    # The N-substituted imidic acid, the Blue Book; 'N-sulfanylcyclopentanecarboximidic acid
    # (PIN)',:30271) has the rank of its imidic acid, as the hydroximic acid does: the O acid's place here,
    # ``_head_principal_group`` ranks it by its chalcogen.
    "n_substituted_imidic_acid",
    # (the Blue Book-:18540): the carboximidic acids
    # modified by S, Se, Te follow the imidic acid, in that order, and come before
    # the hydrazonic acid; the hydrazonic acids modified by S, Se, Te follow it.
    "imidothioic_acid",
    "imidoselenoic_acid",
    "imidotelluroic_acid",
    # Hydrazonic acid: =O -> =N-NH2 replacement acid; ranks
    # immediately below imidic acid per the sub-order.
    "hydrazonic_acid",
    "hydrazonothioic_acid",
    "hydrazonoselenoic_acid",
    "hydrazonotelluroic_acid",
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
    "acyl_isothiocyanate",
    # W3-P11 / /: acid halides of sulfonic /
    # sulfinic acids. Acid-halide tier class 8), below the carbon acyl
    # halides (S oxoacid-derivatives named after their C analogues) and above
    # amides. Functional-class handler-emitted ('ethanesulfonyl chloride').
    #
    # "Seniority order for classes" class 10 (the Blue Book): "Acid halides and
    # pseudohalides [first in the order of the corresponding acid given above, then in the order of
    # the halogen atoms...; then in the following order of pseudohalogen groups". The order of the
    # acids is class 7a (:18172): "carboxylic..., sulfonic, sulfinic, selenonic, seleninic,
    # telluronic, tellurinic". So the halide and the cyanide of one acid are adjacent (halide first)
    # and the acids follow in that order: sulfonic, sulfinic, selenonic, seleninic, telluronic,
    # tellurinic.
    "sulfonyl_halide",
    # -3: the cyanide of a sulfonic acid (R-SO2-C#N ->
    # 'methanesulfonyl cyanide'). Cyanide/pseudohalide functional class, ranked in
    # the acid-halide tier beside sulfonyl_halide and above the plain nitrile, so
    # the functional-class handler wins over the substitutive '-methanenitrile'.
    "sulfonyl_cyanide",
    "sulfinyl_halide",
    # (the Blue Book): the same functional-class cyanide for the sulfinic,
    # selenonic, seleninic, telluronic and tellurinic acids ('benzeneseleninyl cyanide',:34813).
    "sulfinyl_cyanide",
    "selenonyl_cyanide",
    "seleninyl_cyanide",
    "telluronyl_cyanide",
    "tellurinyl_cyanide",
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
    "diselenoperoxol",   # R-Se-SeH (Table 6.1,:27972)
    "ditelluroperoxol",  # R-Te-TeH

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
    # the esters of the class 7d acids, the partial ones ('dihydrogen phosphate', 'hydrogen sulfate') included, are esters of
    # class 9 ranked by their acid (``ESTER_ACID_ORDER``, ``_promoted_oxoacid_ester``): below the carboxylic esters and the carbamate
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
    # The generic 'alcohol' key is where perception files every -OH
    # that fits no subtype, a geminal diol's two OH among them (C(O)O is
    # neither primary nor secondary). It belongs to its own equal-seniority
    # class: without this row the class union below left it out, so in
    # OC(O)CO the principal 'primary_alcohol' suffix counted one OH and the
    # prefix loop skipped the other two as same-class ('ethan-1-ol', a
    # different molecule) instead of 'ethane-1,1,2-triol'.
    "alcohol": "alcohol",
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
    # (the Blue Book): 'propanimidothioic acid (PIN)' (:30267),
    # 'butanehydrazonoselenoic acid (PIN)' (:30269), 'N-hydroxycyclohexane-
    # carboximidoselenoic acid (PIN)' (:30273); Table 4.3 (:18537-:18538).
    "imidothioic_acid": ("imidothioic acid", "carboximidothioic acid"),
    "imidoselenoic_acid": ("imidoselenoic acid", "carboximidoselenoic acid"),
    "imidotelluroic_acid": ("imidotelluroic acid", "carboximidotelluroic acid"),
    "hydrazonothioic_acid": ("hydrazonothioic acid", "carbohydrazonothioic acid"),
    "hydrazonoselenoic_acid": ("hydrazonoselenoic acid", "carbohydrazonoselenoic acid"),
    "hydrazonotelluroic_acid": ("hydrazonotelluroic acid", "carbohydrazonotelluroic acid"),
    # W3-P02-5 Note): the '-hydroximic acid' suffix is general-only.
    # The PIN is the N-hydroxy derivative of the imidic acid (N-hydroxyethanimidic
    # acid), built by the dedicated hydroximic_acid handler — so NO substitutive
    # suffix here (None, exactly like hydroxamic_acid).
    "hydroximic_acid": None,
    "n_substituted_imidic_acid": None,    # written with the suffix of its imidic acid (n_substituted_acids.suffix_group)
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
    #: the cyanides of the other S / Se / Te oxoacids, handler-emitted the same way.
    "sulfinyl_cyanide": None,
    "seleninyl_cyanide": None,
    "selenonyl_cyanide": None,
    "tellurinyl_cyanide": None,
    "telluronyl_cyanide": None,
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
    "acyl_isothiocyanate": ("oyl isothiocyanate", "carbonyl isothiocyanate"),
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
    "diselenoperoxol": ("diselenoperoxol", "diselenoperoxol"),
    "ditelluroperoxol": ("ditelluroperoxol", "ditelluroperoxol"),
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
    "diselenoperoxol": "diselanyl",
    "ditelluroperoxol": "ditellanyl",
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
    # (the Blue Book-34268): "the following prefixes are used. The
    # prefix guanidino may be used in general nomenclature." -- 'carbamimidoylamino
    # (preferred prefix)' for H2N-C(=NH)-NH-; 7. Prefixes (g) (:1700) "The prefix
    # 'guanidino' is no longer acceptable in preferred IUPAC names". The (H2N)2C=N-
    # tautomer takes '(diaminomethylidene)amino' (:34270-34272); the atom-aware choice is
    # substituent_prefix_forms.get_guanidine_prefix.
    "guanidine": "carbamimidoylamino",
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
    #: the demoted prefix of the other S / Se / Te oxoacid cyanides fails closed too.
    "sulfinyl_cyanide": None,
    "seleninyl_cyanide": None,
    "selenonyl_cyanide": None,
    "tellurinyl_cyanide": None,
    "telluronyl_cyanide": None,
    # a phase: demoted imidoyl / carbothioyl / carboselenoyl halide prefix
    # fails closed (None) — the target compounds are mono-functional (the halide is
    # always the principal group), and a halide-dependent prefix is not derived here.
    "imidoyl_halide": None,
    "carbothioyl_halide": None,
    "carboselenoyl_halide": None,
    "acyl_azide": "azidocarbonyl",
    "acyl_cyanide": "cyanocarbonyl",
    "acyl_isocyanate": "isocyanatocarbonyl",
    "acyl_isothiocyanate": "isothiocyanatocarbonyl",
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
    # a demoted DIESTER on a senior parent needs its second organyl in the prefix
    # ('[(methoxysulfonyl)oxy]'), which a per-FG string cannot carry: kept None.
    # A demoted MONOESTER has no second organyl -- the parent is its only one --
    # so it is the fixed preselected prefix 'sulfooxy', as the phosphate
    # monoester is 'phosphonooxy' below: (the Blue Book)
    # "3-(sulfooxy)propanoic acid (PIN)" (:36488); 'HO-SO2-O- sulfooxy
    # (preselected prefix)' (:31342). The PIN-vocabulary guard keeps it off the
    # PIN label under a junior parent (pin_vocabulary._oxoacid_ester_prefix_re).
    "sulfate_diester": None,      # IUPAC functional class naming
    "sulfate_monoester": "sulfooxy",  # IUPAC
    # a review: functional-class ester (no prefix form)
    "phosphinate_ester": None,   # IUPAC functional class naming
    "arsinate_ester": None,      # IUPAC functional class naming
    "stibinate_ester": None,     # IUPAC functional class naming

    "tertiary_phosphine": None,
    "secondary_phosphine": None,
    "primary_phosphine": None,
    #: New FG prefix forms
    "hydroxamic_acid": "hydroxycarbamoyl",  # method (2), the Blue Book (not "N-hydroxyamido")
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
    # The demoted N-substituted imidic acid has no prefix the book prints for a carbon group on the N: fails closed.
    "n_substituted_imidic_acid": None,
    #: a demoted chalcogen imidic / hydrazonic acid ending a chain
    # splits into its simple prefix pair ('imino' + 'sulfanyl',...) in
    # name_polyfunctional's chain-end block; no single static prefix.
    "imidothioic_acid": None,
    "imidoselenoic_acid": None,
    "imidotelluroic_acid": None,
    "hydrazonothioic_acid": None,
    "hydrazonoselenoic_acid": None,
    "hydrazonotelluroic_acid": None,
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


#: The acyclic esters of the oxoacids of classes 7c and 7d of (:data:`ESTER_ACID_ORDER`), in the order of
#: their acids (class 7c before 7d; tri- before di- before mono-, the ``SENIORITY_ORDER`` order), that rank with the esters
#: when their functional-class name can be built (``get_principal_group``).
_OXOACID_ESTERS_PROMOTED = (
    "phosphonate_diester", "phosphinate_ester", "arsinate_ester", "stibinate_ester",
    "phosphate_triester", "phosphate_diester", "phosphate_monoester", "phosphite_triester",
    "sulfate_diester", "sulfate_monoester",
)

#: the symbol of the central atom of the esters of ``_OXOACID_ESTERS_PROMOTED`` that one namer reads for all three
_PNICTOGEN_INATE_ATOM = {"phosphinate_ester": "P", "arsinate_ester": "As", "stibinate_ester": "Sb"}

#: Molecules (by ``id``) whose oxoacid-ester offer is being built right now, with the ester it is built for. The
#: functional-class namer names the owners through the substituent namer, which may ask for the principal group of the
#: same molecule again; that nested question gets the head answer (:func:`offered_principal_group` is the hypothesis, for
#: the one reader that needs it).
_OXOACID_OFFER_ACTIVE: dict = {}


def offered_principal_group(mol):
    """``(group_name, matches)`` of the oxoacid ester whose name is being built for ``mol`` right now (the offer of
    ``get_principal_group``), else None. The amido prefix of an acylated nitrogen is cited in the name of that ester
    (``assembly.n_acyl_amido.group_is_principal``), so the question of whether the amide is the principal group is asked
    of the ester being offered, which outranks it."""
    return _OXOACID_OFFER_ACTIVE.get(id(mol))


def _oxoacid_ester_offered(mol, fg_name, matches) -> bool:
    """True when the functional-class name of the acyclic ester ``fg_name`` (``matches``) can be built for ``mol``."""
    if fg_name.startswith("sulfate"):
        from .sulfur_oxoacid import sulfate_ester_offered
        return sulfate_ester_offered(mol, matches)
    if fg_name in _PNICTOGEN_INATE_ATOM:
        from .phosphorus import pnictogen_inate_ester_offered
        return pnictogen_inate_ester_offered(mol, matches, _PNICTOGEN_INATE_ATOM[fg_name])
    from .phosphorus import phosphate_ester_offered
    return phosphate_ester_offered(mol, matches)


def _promoted_oxoacid_ester(mol, functional_groups, head_name, head_matches=()):
    """The acyclic ester of a phosphorus or sulfur oxoacid to rank as the principal group, or None.

     "SENIORITY ORDER FOR CLASSES" (the Blue Book), Table 4.1: "9 Esters
    (functional class names are given to noncyclic esters; lactones and other cyclic esters
    are named as heterocycles; see 16 below)" (:18182) ranks above "10 Acid halides and pseudohalides",
    "11 Amides" (:18183), "12 Hydrazides", "13 Imides", "14 Nitriles", "15 Aldehydes", "16 Ketones" (:18189),
    "17 Hydroxy compounds" (:18190) and "19 Amines" (:18192); the esters rank among themselves by their acid
    (:data:`ESTER_ACID_ORDER`, "Esters cited as prefixes",:31696), so the esters of the acids of classes 7c
    and 7d rank below the carboxylic and sulfonic esters (7a) and the carbamates (7b). "Esters of mononuclear
    noncarbon oxoacids" (:35916) names 'trimethyl phosphate (PIN)' (:35936), 'methyl dihydrogen phosphate (PIN)'
    (:35940) and 'methyl hydrogen sulfate (PIN)' as functional-class esters; a partial ester keeps the ranks of its
    acid ('hydrogen'/'dihydrogen' of the name are the acid hydrogens), and every group below the class 9 esters is
    cited as a prefix of its organyl group ('2-acetamidoethyl hydrogen sulfate', not 'N-[2-(sulfooxy)ethyl]acetamide').

    Decided in the final round of lane/l17: a PARTIAL ester ('dihydrogen phosphate', 'hydrogen sulfate') is an ESTER of
    class 9 ranked by its acid, not an acid of class 7 above the carbamate and the carboxylic esters (the reading that the
    free hydrogen is a free acid group). "When anions are different, two methods are used" (the Blue Book:
    31831): "The seniority order of anions is corresponding to that of acids (see seniority order of acids in ", the
    anion of a partial ester ('hydrogen sulfate') being one of the anions; "Esters of mononuclear noncarbon
    oxoacids" (:35916): "Partial acid esters of polybasic acids are named by citing alkyl groups... followed by the word
    'hydrogen'", an ester name; and the Blue Book's own phosphatidylethanolamine, (:55180), a partial phosphate
    ester beside two carboxylic esters, is named on the carboxylic esters with the phosphate as a prefix
    ('(2R)-3-{[(2-aminoethoxy)hydroxyphosphoryl]oxy}propane-1,2-diyl dihexadecanoate'). The free acid group that outranks
    an ester is the one of a carboxylic or sulfonic acid, a suffix ('5-butoxy-2-methyl-5-oxopentanoic acid (PIN)',
    ,:31935), which the free hydrogen of a phosphorus or sulfur oxoacid ester is not. So the carbamate (7b) is the
    parent beside a partial phosphate or sulfate ester (7d) and the carboxylic ester (7a) too: 'methyl [4-(phosphonooxy)
    phenyl]carbamate', 'methyl 3-(sulfooxy)propanoate'; the order of the acids is the ``ESTER_ACID_ORDER`` of this module.

    The class is OFFERED, not returned: it is ranked above the head group only when the functional-class name of
    :func:`orthonym.rules.phosphorus.name_phosphate_ester` (or its sulfur and pnictogen siblings) can be built for
    the molecule. A suffix-less functional-class principal group that cannot be built would demote the next class's
    suffix to a prefix ('propan-2-ol' to '2-hydroxy...propane'), so an ester the namer declines keeps the head order.
    A central atom in a ring is a cyclic ester, named as a heterocycle (the same note of Table 4.1), and is not
    ranked here. Only a head group below the esters of the carboxylic and sulfonic acids (the last of them,
    ``sulfinic_ester``) is outranked: the acids, the carboxylic and sulfonic esters and the anhydrides keep their head
    rank, as does a molecule that has a carbamate (carbamic acid is a carbonic acid, class 7b, senior to these). A head
    group that is only a cyclic ester (``head_matches``: a lactone, a sultone) is a pseudoketone, class 16
    "Cyclic anhydrides, esters and amides are named as pseudoketones",:29314), and is outranked like a ketone:
    '2-(2-oxooxolan-3-yl)ethyl hydrogen sulfate'.

    A molecule whose ester outranks the head but cannot be offered is named on the head group; that name is not the
    PIN, and the label rule for it is the phosphoryl-oxy prefix test of
    ``rules.pin_vocabulary.non_pin_vocabulary`` (a name decides it, so it does not depend on which tier built the name).

    Only for a whole molecule (``is_top_level_naming``): a fragment named inside another name
    keeps the substitutive principal group that name needs, because a functional-class name
    cannot take the substituents of the other fragments. Follow-up (item 42c): the same rule
    moves a decomposition fragment of a large ester between two names that are both correct
    and neither better (the stereo-stripped ceramide phosphoinositol: the cyclohexane-1,2,3,
    4,5-pentol name with an N-(2-hydroxytetracosanoyl) group becomes the skeletal-replacement
    form with a '3-hydroxy-3-oxo-4-(2,3,4,5,6-pentahydroxycyclohexyl)-2,4-dioxa-3-phosphabutyl'
    group; both are systematic_verified): the fragment case needs its own measurement."""
    if head_name is None or head_name in _OXOACID_ESTERS_PROMOTED:
        return None
    candidates = [n for n in _OXOACID_ESTERS_PROMOTED if functional_groups.get(n)]
    if not candidates or functional_groups.get("carbamate"):
        return None                          # no oxoacid ester (or a carbamate, the senior ester): the head answer
    # Only a whole molecule: a fragment named inside another name (a decomposition fragment, a
    # substituent's capped fragment) keeps the substitutive principal group its enclosing name
    # needs, because a functional-class name ('X Y hydrogen phosphate') cannot take the
    # substituents of the other fragments (the same reasoning as name_general_multiplicative).
    from ..assembly.fragment_naming import is_top_level_naming
    if not is_top_level_naming():
        return None
    try:
        floor = SENIORITY_ORDER.index("sulfinic_ester")
        if SENIORITY_ORDER.index(head_name) <= floor and not (
                head_name in _RING_ESTER_CLASSES and head_matches
                and all(_is_ring_ester(mol, m) for m in head_matches)):
            return None
    except ValueError:
        return None
    if id(mol) in _OXOACID_OFFER_ACTIVE:
        return None
    for fg_name in candidates:
        matches = functional_groups.get(fg_name)
        _OXOACID_OFFER_ACTIVE[id(mol)] = (fg_name, matches)
        try:
            if _oxoacid_ester_offered(mol, fg_name, matches):
                return fg_name, matches
        finally:
            _OXOACID_OFFER_ACTIVE.pop(id(mol), None)
    return None


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
    head = _head_principal_group(mol, functional_groups)
    head = senior_carboxamide_group(mol, functional_groups, head)   # one class of amides, the parent by /
    head = _senior_pseudoester_group(mol, functional_groups, head)  # a pseudoester ranks as an ester, by its acid
    promoted = _promoted_oxoacid_ester(mol, functional_groups, head[0], head[1])
    return promoted if promoted is not None else head


#: the groups of class 7a (carboxylic acids and their imidic and chalcogen analogues) that a pseudoester of a carboxylic acid
#: is ranked against: the esters of the carboxylic acid itself, by their acid, the esters of its analogues below them
_ANALOGUE_ESTER_CLASSES = ("iminoester", "thioester", "selenoester", "telluroester")

#: Molecules (by ``id``) whose pseudoester offer is being built right now (the functional-class name is made through the
#: nested naming of the free acid, which may ask for the principal group of the same molecule again)
_PSEUDOESTER_OFFER_ACTIVE: set = set()


def _senior_pseudoester_group(mol, functional_groups, head):
    """``head``, or the pseudoester group when the pseudoester is the senior ester of the molecule.

     "Definitions" (the Blue Book): "Pseudoesters are compounds having the generic formula R-E(=O)x(OZ)...
    Z is not a carbon atom but an element from the following list: B, Al, In, Ga, Tl, Si, Ge, Sn, Pb, N(cyclic), P, As, Sb,
    Bi. Pseudoesters are ranked as esters in the seniority order of classes (see." Class 9 of Table 4.1 (:18182)
    ranks the esters among themselves by their acid, and "Esters cited as prefixes" (:31696) cites the ester of
    the junior acid as a prefix of the name of the senior one. The pseudoester ranked after every other ester of the order
    of ``SENIORITY_ORDER`` (a convention for the groups that stand alone), so the acetate of a molecule that also holds the
    trimethylsilyl ester of a benzoic acid was its principal group, and the silyl ester a prefix, spelled
    '4-{oxo[(trimethylsilyl)oxy]methyl}phenyl acetate'. The two compare as the esters of ``rules.ester_seniority`` do, the
    acid component (the senior acid is the ring acid, the longer chain, the more substituted); the esters of the
    analogues of the carboxylic acid (the imidic and chalcogen ones,:18172) are junior to the ester of the acid itself.
    The ester of ``head`` keeps its place on a tie. The pseudoester is OFFERED, as the oxoacid esters are
    (:func:`_promoted_oxoacid_ester`): a functional-class name that cannot be built would demote the next group to a prefix.
    Only for a whole molecule, as there."""
    pseudo = [m for m in functional_groups.get("pseudoester") or [] if len(m) >= 4 and not _is_ring_ester(mol, m)]
    if not pseudo or head[0] is None or id(mol) in _PSEUDOESTER_OFFER_ACTIVE:
        return head
    if head[0] != "ester" and head[0] not in _ANALOGUE_ESTER_CLASSES:
        return head
    from ..assembly.fragment_naming import is_top_level_naming
    if not is_top_level_naming():
        return head
    if head[0] == "ester":
        from .ester_seniority import ester_seniority_key
        acyclic = [m for m in head[1] if not _is_ring_ester(mol, m)]
        if not acyclic or min(ester_seniority_key(mol, m)[0] for m in pseudo) >= min(
                ester_seniority_key(mol, m)[0] for m in acyclic):
            return head                 # the acid component only: the alcohol side of a pseudoester is no organyl chain
    if not _pseudoester_offered(mol, pseudo):
        return head
    return "pseudoester", pseudo


def _pseudoester_offered(mol, matches) -> bool:
    """True when the functional-class name of the pseudoester of ``matches`` can be built
    (:func:`orthonym.rules.esters.name_noncarbon_ester`, the producer of its handler). Memoised for the naming call
    (``assembly.memo``), keyed on the atom order of the molecule and the matches."""
    from ..assembly.memo import cache_or_compute
    from rdkit import Chem

    def build():
        from .esters import name_noncarbon_ester
        _PSEUDOESTER_OFFER_ACTIVE.add(id(mol))
        try:
            return bool(name_noncarbon_ester(mol, matches[0], tuple(matches[1:])))
        except Exception:  # noqa: BLE001 - an offer that cannot be built is not made
            return False
        finally:
            _PSEUDOESTER_OFFER_ACTIVE.discard(id(mol))
    return bool(cache_or_compute("pseudoester_offer", (Chem.MolToSmiles(mol, canonical=False), tuple(matches)), build))


#: The amide classes a ring amide (a lactam) is perceived as; the lactam is a pseudoketone (class 16).
_RING_AMIDE_CLASSES = ("secondary_amide", "tertiary_amide")


def _head_principal_group(
    mol,
    functional_groups: Dict[str, List[tuple]]
) -> Tuple[Optional[str], List[tuple]]:
    """The first group of ``SENIORITY_ORDER`` that is present and suffix-capable (the answer
    before the phosphate esters are offered; see ``get_principal_group``).

    A ring amide (a lactam) is a pseudoketone, class 16 of Table 4.1 (the Blue Book), and "Cyclic
    anhydrides, esters and amides are named as pseudoketones" (:29314): a thioamide, a hydrazide, an amidine, a
    nitrile, an aldehyde or an amide of the chain (classes 11 to 15,:18162-18188) is senior to it. The order of
    ``SENIORITY_ORDER`` ranks the ring amide among the amides, so those groups were demoted to prefixes
    ('1-(2-hydrazinyl-2-oxoethyl)pyrrolidin-2-one'). The ring amides step aside, and a group senior to the ketone
    that remains is the principal group; with none the ring amide keeps its place. A cyclic ester (a lactone, a sultone,
    a cyclic phosphate) is a heterocycle (:func:`_is_ring_ester`, Table 4.1 class 9 note,:18182) and steps aside in the
    same way."""
    ring = {cls: [m for m in functional_groups.get(cls) or [] if _is_ring_amide(mol, m)]
            for cls in _RING_AMIDE_CLASSES}
    for cls in _RING_ESTER_CLASSES:
        ring[cls] = [m for m in functional_groups.get(cls) or [] if _is_ring_ester(mol, m)]
    if any(ring.values()):
        rest = {k: ([m for m in v if m not in ring[k]] if k in ring else v) for k, v in functional_groups.items()}
        group, matches = _first_principal_group(mol, rest)
        if (group is not None and SENIORITY_ORDER.index(group) < SENIORITY_ORDER.index("ketone")
                and (group != "ester" or _acyclic_ester_offered(mol, matches))):
            return group, matches
    return _first_principal_group(mol, functional_groups)


#: Molecules (by ``id``) whose acyclic carboxylic ester is being offered right now (the owners of the functional-class
#: name are named through the substituent namer, which may ask for the principal group of the same molecule again)
_ESTER_OFFER_ACTIVE: set = set()


def _acyclic_ester_offered(mol, matches) -> bool:
    """True when the functional-class name of the acyclic carboxylic ester of ``matches`` can be built
    (:func:`orthonym.rules.esters.name_ester`, the producer of the ester handlers): the offer behind the step-aside of a
    lactone for an acyclic ester. class 9 "Esters" (the Blue Book): "functional class names are given
    to noncyclic esters; lactones and other cyclic esters are named as heterocycles; see 16 below". An ester whose
    functional-class name cannot be built (the alcohol component a polycyclic lactone) would be the principal group of a
    name that does not cite it, or cites it as a prefix of a ring parent with no ester suffix to carry it, so the lactone
    keeps its place and the ester is cited as an acyloxy prefix of the ring parent, as it was. Memoised for the naming
    call (``assembly.memo``), keyed on the atom order of the molecule and the matches."""
    if id(mol) in _ESTER_OFFER_ACTIVE or not matches:
        return False
    from ..assembly.memo import cache_or_compute
    from rdkit import Chem

    def build():
        from .esters import name_ester
        _ESTER_OFFER_ACTIVE.add(id(mol))
        try:
            return any(name_ester(mol, m) for m in matches)
        except Exception:  # noqa: BLE001 - an offer that cannot be built is not made
            return False
        finally:
            _ESTER_OFFER_ACTIVE.discard(id(mol))
    return bool(cache_or_compute("acyclic_ester_offer", (Chem.MolToSmiles(mol, canonical=False), tuple(matches)), build))


def _is_ring_ester(mol, match) -> bool:
    """An ester match whose acid atom (``match[0]``) is bonded to an ester oxygen -O-C by a ring bond: a cyclic ester
    (a lactone, a sultone, a cyclic phosphate), named as a heterocycle. class 9 "Esters" (the Blue Book):
    "functional class names are given to noncyclic esters; lactones and other cyclic esters are named as heterocycles; see
    16 below", and class 16 (:18189): "See Classes 8, 9, 11 and 13, on lactones, lactams, anhydrides, imides"."""
    centre = mol.GetAtomWithIdx(match[0])
    for bond in centre.GetBonds():
        o = bond.GetOtherAtom(centre)
        if (o.GetAtomicNum() == 8 and bond.GetBondTypeAsDouble() == 1.0 and bond.IsInRing() and o.GetDegree() == 2
                and any(n.GetIdx() != centre.GetIdx() and n.GetAtomicNum() in (6, 16) for n in o.GetNeighbors())):
            return True
    return False


def _is_ring_amide(mol, match) -> bool:
    """An amide match ``(C, O, N,...)`` whose C-N bond is in a ring: a cyclic amide."""
    bond = mol.GetBondBetweenAtoms(match[0], match[2]) if len(match) > 2 else None
    return bond is not None and bond.IsInRing()


def _first_principal_group(
    mol,
    functional_groups: Dict[str, List[tuple]]
) -> Tuple[Optional[str], List[tuple]]:
    """The first group of ``SENIORITY_ORDER`` that is present and suffix-capable in ``functional_groups``."""
    for fg_name in SENIORITY_ORDER:
        if fg_name in CN_GROUPS:
            continue                  # ranked by its chalcogen, with the imidic acids (below)
        _n_group, _n_matches = group_at_rank(mol, functional_groups, fg_name)
        if _n_matches and not functional_groups.get(fg_name):
            return _n_group, _n_matches    # the rank of its imidic acid, class 7a (:18172)
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


#: the amidine and its amidrazone relatives, whose carbon and nitrogens are ring atoms when the ring holds the
#: whole unit. "Amidrazone suffixes" (the Blue Book): the suffixes 'hydrazonamide' and
#: 'carbohydrazonamide', 'imidohydrazide' and 'carboximidohydrazide' name R-C(NH2)=N-NH2 on a parent hydride;
#: the 'carbo' form is the one for "a cyclic parent hydride", whose group carbon is outside the ring, so a
#: -C(=N-N)-N- unit with every atom in the ring is skeletal like the amidine of
_AMIDINE_LIKE_CLASSES = ("amidine", "hydrazonamide", "hydrazidine", "imidohydrazide")


def without_ring_skeleton_amidine(
    functional_groups: Dict[str, List[tuple]],
    ring_atoms,
) -> Dict[str, List[tuple]]:
    """``functional_groups`` with the 'amidine' class removed when every amidine of the molecule
    has its carbon and both nitrogens among ``ring_atoms`` (a ring or ring system), else the
    dict unchanged.

     "Suffixes for amidines" (the Blue Book): the amidine group is the
    characteristic group "-C(=NH)-NH2" "attached to a cyclic parent hydride" (:34200); a
    ring that holds the whole N=C-N unit has those three atoms as skeletal atoms of the
    heterocycle (a tetrahydropyrimidine, the octahydropyrimido[1,2-a]azepine of
    1,8-diazabicyclo[5.4.0]undec-7-ene), so no '-imidamide' suffix stands on it and the ring
    system is the parent. A group of another kind that the ring carries is still seen once
    the skeletal amidine is set aside."""
    drop = {cls for cls in _AMIDINE_LIKE_CLASSES
            if functional_groups.get(cls)
            and all(all(i in ring_atoms for i in match) for match in functional_groups[cls])}
    if not drop:
        return functional_groups
    return {k: v for k, v in functional_groups.items() if k not in drop}


def guanidine_parent_matches(mol, matches) -> List[tuple]:
    """The guanidine matches (N, C, =N, N) of ``matches`` whose carbon is not a ring atom: the
    ones the retained parent 'guanidine' can name.

     "Guanidine and its derivatives" (the Blue Book): "The preferred IUPAC
    name for the 'amidine' related to carbonic acid, H2N-C(=NH)-NH2, is the retained name
    'guanidine'", an acyclic parent hydride with the locants N, N' and N''. The carbon of a
    guanidine has three nitrogen neighbours and no other, so a ring that runs through it holds two
    of those nitrogens as skeletal atoms: the unit is part of a heterocycle (the ring system of
    1,3,4,6,7,8-hexahydro-2H-pyrimido[1,2-a]pyrimidine, the 4,5-dihydro-1H-imidazole of 2-amino-
    imidazoline), the ring system is the parent "Systems composed of rings and chains",
    the Blue Book,:19340 "Within the same class, a ring or ring system has seniority over a
    chain"), and no guanidine stands on it."""
    return [m for m in matches if not mol.GetAtomWithIdx(m[1]).IsInRing()]


def ring_skeleton_units_resolved(
    mol,
    functional_groups: Dict[str, List[tuple]],
    ring_atoms,
) -> Dict[str, List[tuple]]:
    """``functional_groups`` as seen from the ring system ``ring_atoms``: an amidine or a guanidine
    whose carbon is one of its skeletal atoms is a feature of the ring, not a group of its own.

    A unit with all of its nitrogens among ``ring_atoms`` is the heterocycle itself and is set aside
    (``without_ring_skeleton_amidine``, ``guanidine_parent_matches``). A unit whose one exocyclic
    nitrogen is a terminal single-bonded -NH2 has its other nitrogens in the ring and that -NH2
    is the amino group of an amine: "Primary amines, R-NH2, are systematically named in
    the following ways" (the Blue Book), substitutive '-amine' (1-benzofuran-2-amine (PIN),
    :26193; quinolin-4-amine (PIN),:26199), so the unit is replaced by a 'primary_amine' match
    (N, C) and ranks as one. A unit with any other exocyclic part (a substituted amino, the =NH of an
    imine, an acylated nitrogen) is kept unchanged: it is not the plain amine this function can
    name. Groups of other kinds are untouched."""
    groups = without_ring_skeleton_amidine(functional_groups, ring_atoms)
    ring_atoms = set(ring_atoms)
    drop = set()                    # (class, match) set aside
    amines = []                     # (N, C) of the -NH2 that becomes an amine
    for cls, c_pos, n_pos in (("amidine", 0, (1, 2)), ("guanidine", 1, (0, 2, 3))):
        for match in groups.get(cls) or ():
            c_idx = match[c_pos]
            if c_idx not in ring_atoms:
                continue
            exo = [match[i] for i in n_pos if match[i] not in ring_atoms]
            if not exo:
                drop.add((cls, match))
                continue
            if len(exo) != 1:
                continue
            n = mol.GetAtomWithIdx(exo[0])
            bond = mol.GetBondBetweenAtoms(exo[0], c_idx)
            if (n.GetDegree() == 1 and n.GetTotalNumHs() == 2 and n.GetFormalCharge() == 0
                    and bond is not None and bond.GetBondTypeAsDouble() == 1.0):
                drop.add((cls, match))
                amines.append((exo[0], c_idx))
    if not drop:
        return groups
    out = {k: [m for m in v if (k, m) not in drop] for k, v in groups.items()}
    if amines:
        out["primary_amine"] = list(out.get("primary_amine") or []) + amines
    return out


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


def seniority_index(fg_name: str, mol=None, matches=None) -> int:
    """The place of ``fg_name`` in ``SENIORITY_ORDER`` (ValueError for a name that is not in it).

    THE one reader of the order for a group's rank: a hydroximic or N-substituted imidic acid has the rank of
    the imidic acid of its acid chalcogen class 7a, the Blue Book; ``hydroximic_acid`` in
    the order is the O acid's place), so ``mol`` and the group's ``matches`` give it the place of its
    imidic, imidothioic, imidoselenoic or imidotelluroic acid, and two groups of one rank share an
    index. Without them the group is the O acid's. Every other name is its own place."""
    return SENIORITY_ORDER.index(rank_group(mol, fg_name, matches))


#: class 9 "Esters" (the Blue Book) rank by the acid they are esters of: "Esters cited as
#: prefixes" (:31696) keeps the ester of the senior acid as the parent and cites the other as a prefix ('ethyl 2-[(ethoxy-
#: carbonyl)oxy]-4,4-dimethyl-3-oxopentanoate (PIN)', the carbonic ester in the prefix of the carboxylic one; 'methyl 4-
#: (phenoxysulfinothioyl)naphthalene-1-carboxylate (PIN)', that of a sulfur acid), "Seniority for numbering follows that
#: for acids". The order of the acids is class 7 of Table 4.1 (:18170ff) and: class 7a "CLASS 7a.
#: ACIDS EXPRESSED BY SUFFIXES", carboxylic with its imidic and chalcogen analogues, sulfonic, sulfinic...) > class 7b
#: "CLASS 7b. CARBON ACIDS WITH NO SUBSTITUTABLE HYDROGEN ATOMS": carbonic acid, to which carbamic acid belongs,
#: > class 7c, phosphonic, phosphinic, arsonic...) > class 7d, phosphoric, sulfuric,...). The
#: functional-group keys of the esters of each class, senior class first; the one reader is:func:`ester_acid_rank`.
ESTER_ACID_ORDER = (
    ("7a", frozenset({"ester", "iminoester", "thioester", "selenoester", "telluroester", "pseudoester",
                      "sulfonic_ester", "sulfinic_ester"})),
    ("7b", frozenset({"carbamate", "carbamimidate"})),
    ("7c", frozenset({"phosphonate_diester", "phosphinate_ester", "arsinate_ester", "stibinate_ester"})),
    ("7d", frozenset({"phosphate_triester", "phosphate_diester", "phosphate_monoester", "phosphite_triester",
                      "sulfate_diester", "sulfate_monoester"})),
)


#: the ester classes (:data:`ESTER_ACID_ORDER`) whose ring form, a lactone, a sultone or a cyclic phosphate, is a heterocycle
_RING_ESTER_CLASSES = tuple(sorted({g for _, groups in ESTER_ACID_ORDER for g in groups}))


def ester_acid_rank(fg_name: str) -> Optional[int]:
    """The place of the acid of the class-9 ester group ``fg_name`` in:data:`ESTER_ACID_ORDER` (0 is the senior class);
    None for a group that is not an ester there."""
    for rank, (_, groups) in enumerate(ESTER_ACID_ORDER):
        if fg_name in groups:
            return rank
    return None


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
        idx1 = seniority_index(fg1)
    except ValueError:
        idx1 = len(SENIORITY_ORDER)  # Not in list = lowest priority

    try:
        idx2 = seniority_index(fg2)
    except ValueError:
        idx2 = len(SENIORITY_ORDER)

    if idx1 < idx2:
        return -1  # fg1 is higher (lower index = higher priority)
    elif idx1 > idx2:
        return 1   # fg2 is higher
    return 0       # Equal
