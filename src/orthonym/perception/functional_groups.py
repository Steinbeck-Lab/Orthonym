"""
Functional group detection using SMARTS patterns.

Groups are ordered by IUPAC seniority (highest priority first).
The principal group (highest seniority) becomes the suffix;
all others become prefixes.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
from rdkit import Chem


# SMARTS patterns ordered by IUPAC seniority (P-41 to P-43)
# First match = highest priority = principal group
FUNCTIONAL_GROUP_SMARTS = {
    # === CHARGED CHARACTERISTIC GROUPS (BBR-PERC/DEF-1, 169.7: P-41 classes 4/6) ===
    # Added so the NEUTRAL FG layer can PERCEIVE ionic groups too — a charged molecule
    # that reaches this detector with a residual charge (the D-1 mis-route exposure) no
    # longer silently loses its group. The normal charged path neutralizes BEFORE this
    # detector runs (charged_router re-enters Orthonym().name on the neutral form), so
    # these fire only on the mis-route case. Every pattern requires a formal charge, so
    # NEUTRAL molecules never match. Suffixes: P-72 (anions) / P-73 (cations).
    "carboxylate": "[#6][CX3](=O)[O-]",            # P-72.2.2.2.1.1 (-oate)
    "sulfonate": "[#6][SX4](=O)(=O)[O-]",          # P-72.2.2.2.1.1 (-sulfonate)
    "phosphonate": "[#6][PX4](=O)([O-,OX2H1])[O-]",# P-72 (-phosphonate; mono/di-deprotonated)
    "thiolate": "[#6][S-]",                        # P-72.2.2.2.2 (-thiolate)
    "phenolate": "[c][O-]",                        # P-72.2.2.2.2 (aromatic-O anion)
    "ammonium": "[NX4+]",                          # P-73.1.2.1 (protonated/quaternary N+)
    # === ACIDS (highest priority) ===
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    # Peroxy acid R-C(=O)-OOH (P-43.1 / Table 4.3 'peroxoic acid'); the [#6]
    # guard keeps OOC(=O)O routing to the exact-SMILES carbonoperoxoic entry.
    "peroxy_acid": "[CX3;$([CX3]([#6])(=[OX1])[OX2][OX2H1])](=[OX1])[OX2][OX2H1]",
    # Thiocarboxylic acids (IUPAC P-65.3) -- rank just below carboxylic acid
    "thioic_S_acid": "[CX3](=O)[SX2H1]",    # R-C(=O)-SH -> thioic S-acid
    "thioic_O_acid": "[CX3](=S)[OX2H1]",    # R-C(=S)-OH -> thioic O-acid
    "dithioic_acid": "[CX3](=S)[SX2H1]",    # R-C(=S)-SH -> dithioic acid
    # Phase 163 Tier FRN-A: chalcogen-on-acid (P-65.3) -- additive per CONTEXT D-08
    "selenoic_Se_acid": "[CX3](=O)[SeX2H1]",     # R-C(=O)-SeH (P-65.3; AUDIT-FRN § 2)
    "selenoic_O_acid": "[CX3](=[SeX1])[OX2H1]",  # R-C(=Se)-OH (P-65.3; AUDIT-FRN § 2)
    "diselenoic_acid": "[CX3](=[SeX1])[SeX2H1]", # R-C(=Se)-SeH (P-65.3; AUDIT-FRN § 2)
    "telluroic_Te_acid": "[CX3](=O)[TeX2H1]",    # R-C(=O)-TeH (P-65.3 parallel; AUDIT-FRN § 2)
    "telluroic_O_acid": "[CX3](=[TeX1])[OX2H1]", # R-C(=Te)-OH (P-65.3 parallel; AUDIT-FRN § 2)
    "ditelluroic_acid": "[CX3](=[TeX1])[TeX2H1]",# R-C(=Te)-TeH (P-65.3 parallel; AUDIT-FRN § 2)
    # Carbamic acid (IUPAC P-65.2.3): N-C(=O)-OH (free acid, not ester)
    # Wave2 T3d: the amino N must NOT itself bear a second N — else N-N-C(=O)-OH
    # (a hydrazinecarboxylic acid, P-66.3.5.1) is mis-read as carbamic acid and the
    # senior carbamic path DROPS the terminal N (wrong structure). The negated
    # recursive keeps ordinary carbamic acids (NC(=O)O / CNC(=O)O — N has no N-neighbour).
    # W3-P05 (P-65.1.8.2): the amino N must be a NEUTRAL amine N with NO oxygen
    # neighbour — else the nitro N of O2N-C(=O)-OH (nitroformic acid) false-matches
    # and the senior carbamic path emits a WRONG 'carbamic acid' (structure loss:
    # the nitro group is dropped). ``+0`` excludes the cationic nitro [N+];
    # ``!$([NX3]~[OX1])`` excludes any N bearing a terminal =O/[O-] (nitro, N-oxide).
    # Both constraints are inside atom-1's brackets, so the 4-atom match tuple is
    # unchanged. Nitroformic acid is named by its own exact-SMILES @40 key.
    "carbamic_acid": "[NX3;+0;!$([NX3][NX3]);!$([NX3]~[OX1])][CX3](=O)[OX2H1]",  # R2N-C(=O)-OH -> carbamic acid
    # BBR-PERC/DEF-4 (169.7): require a C neighbour on S/P (P-65.3: sulfonic/phosphonic
    # are CARBON acids). Recursive-env `$(...)` adds the constraint WITHOUT changing the
    # match-tuple arity, so inorganic oxoacids (sulfamic NS(=O)(=O)O, phosphoric OP(=O)(O)O)
    # stop false-matching while C-attached acids still match. Free inorganic oxoacids are
    # perceived by their own keys below (P-67) and named as functional parents (P-42/P-67).
    "sulfonic_acid": "[SX4;$([SX4][#6])](=O)(=O)[OX2H1]",
    # W3-P04 (P-65.3.1.2 / P-65.3.1.3 / P-65.3.1.4): FRN-modified sulfur-oxo-acid
    # suffix acids. Each is a C-attached ([#6] guard) sulfonic/sulfinic acid whose
    # -OH oxygen or =O is functionally replaced, so the generic sulfonic_acid/
    # sulfinic_acid SMARTS above CANNOT match (their [OX2H1] must sit DIRECTLY on S).
    # Checked (and suppressed over the generics) in the disambiguation resolver below.
    # -SO2-OOH sulfonoperoxoic acid (P-65.3.1.2; BB @31184). Match tuple
    # (S, =O, =O, inner-O, hydroperoxy-OH). The inner O is on S (not C), so the
    # hydroperoxide SMARTS [OX2H][OX2][#6] cannot claim the -OOH either.
    "sulfonoperoxoic_acid": "[SX4;$([SX4][#6])](=O)(=O)[OX2][OX2H1]",
    # -SO2-SH sulfonothioic S-acid (P-65.3.1.3; BB @31198, preselected suffix).
    # The -OH of sulfonic acid is replaced by -SH. Match tuple (S, =O, =O, S-H).
    # The thiol SMARTS [SX2H][#6] cannot claim the -SH (its S is on S, not C).
    "sulfonothioic_S_acid": "[SX4;$([SX4][#6])](=O)(=O)[SX2H1]",
    # -S(=NH)-OH sulfinimidic acid (P-65.3.1.4; BB @31220, =NH replacement of the
    # sulfinic-acid =O). SX3 (sulfinic-derived). Match tuple (S, =NH, -OH). The
    # required [OX2H1] keeps this from stealing sulfinimidamide -S(=NH)-NH2 (the
    # third neighbour is N, not OH there).
    "sulfinimidic_acid": "[SX3;$([SX3][#6])](=[NX2])[OX2H1]",
    # -S(=O)(=NH)-OH sulfonimidic acid (P-65.3.1.4; BB @31234, one =O of the
    # sulfonic acid replaced by =NH). SX4 (sulfonic-derived). Match tuple
    # (S, =O, =NH, -OH). Required [OX2H1] keeps it from stealing sulfonimidamide
    # -S(=O)(=NH)-NH2. The N may itself bear an -OH (=N-OH): that is the
    # N-hydroxy hydroximic derivative (P-65.3.1.5), handled downstream.
    "sulfonimidic_acid": "[SX4;$([SX4][#6])](=O)(=[NX2])[OX2H1]",
    "sulfinic_acid": "[SX3;$([SX3][#6])](=O)[OX2H1]",
    "sulfenic_acid": "[SX2]([OX2H])[#6]",  # DATA-05d: IUPAC P-65.3.1.4 R-S-OH
    # v23 Phase 9 (P-65.3 / Table 6.2): selenium & tellurium analogues of the
    # sulfonic/sulfinic suffix acids, C-attached (parallel [#6] guard) so the
    # free chalcogen oxoacids are not mis-claimed -> ethaneselenonic acid etc.
    "selenonic_acid": "[SeX4;$([SeX4][#6])](=O)(=O)[OX2H1]",  # R-Se(=O)2-OH
    "seleninic_acid": "[SeX3;$([SeX3][#6])](=O)[OX2H1]",      # R-Se(=O)-OH
    "telluronic_acid": "[TeX4;$([TeX4][#6])](=O)(=O)[OX2H1]", # R-Te(=O)2-OH
    "tellurinic_acid": "[TeX3;$([TeX3][#6])](=O)[OX2H1]",     # R-Te(=O)-OH
    "phosphonic_acid": "[PX4;$([PX4][#6])](=O)([OX2H1])[OX2H1]",
    # Phosphinic acid: R2P(=O)(OH) - two C attached to P
    "phosphinic_acid": "[PX4](=O)([OX2H1])([#6])[#6]",
    # === FREE INORGANIC OXOACIDS (BBR-PERC/DEF-2, 169.7: P-67 functional parents) ===
    # Perceived so they are NOT silently dropped or mis-cast as carbon acids (the old
    # `OP(=O)(O)O → trihydrophosphate` malformed bug; now blocked by the [#6]-tightened
    # phosphonic_acid above). NO carbon on the central atom (that is what distinguishes
    # them from the carbon acids). NAMING is P-67 functional-parent (downstream); here
    # the goal is perception completeness + correct mis-cast prevention. The resolver
    # below suppresses the carbon-acid generics on their atoms for robustness.
    "phosphoric_acid": "[OX2H1][PX4](=O)([OX2H1])[OX2H1]",   # HO-P(=O)(OH)OH (P-67)
    "sulfuric_acid": "[OX2H1][SX4](=O)(=O)[OX2H1]",          # HO-S(=O)2-OH (P-67)
    "nitric_acid": "[OX2H1][NX3+](=O)[O-]",                  # HO-N(+)(=O)O- (P-67)
    "carbonic_acid": "[OX2H1][CX3](=O)[OX2H1]",              # HO-C(=O)-OH (P-65.2.1)

    # === ACID DERIVATIVES ===
    "anhydride": "[CX3](=O)[OX2][CX3](=O)",
    "ester": "[CX3](=O)[OX2][#6]",
    # Phase 163 Tier FRN-D: iminoester / imidate (P-65.1.7) -- additive per CONTEXT D-08
    # AUDIT DECISION (AUDIT-FRN § 2.4): [NX2H1] only (=NH form); N-substituted iminoesters
    # (R-C(=NR')-O-R'') deferred to Phase 163.1 per Open Question 4. Free imidic acid form
    # (R-C(=NH)-OH) deferred per CONTEXT line 120. Cyclic imidates deferred per RESEARCH §5.4.
    "iminoester": "[CX3](=[NX2H1])[OX2][#6]",     # R-C(=NH)-O-R' (P-65.1.7; "alkyl alkanimidate")
    # W2F-P6 (P-66.1.6.1.2.1): N-substituted carbamimidate ester
    # R''2N-C(=NR')-O-R -> "R N'-R'-N,N-R''2-carbamimidate". A DEDICATED pattern
    # (imino N-substitution ALLOWED, plus a REQUIRED second amino N) so the
    # restricted [NX2H1] iminoester pattern above stays untouched (its
    # AUDIT-FRN 2.4 =NH guard holds). Atom order: (central_C, imino_N, amino_N,
    # ester_O, alkyl_C).
    "carbamimidate": "[CX3](=[NX2])([NX3])[OX2][#6]",  # R''2N-C(=NR')-O-R (P-66.1.6.1.2.1)
    # Imidic acid R-C(=NH)-OH (P-65.1.3.1 / Table 4.3 'imidic acid'). W3-P02-1
    # (P-65.1.3.1.1): the imidic C may bear an H (methanimidic HC(=NH)-OH, PIN
    # 'methanimidic acid'), so the third neighbour is NO LONGER required to be
    # [#6] — it may be H or C. The `!$([CX3]([#7,#8])...)` guard still keeps
    # N=C(O)O / N=C(N)O routing to the exact-SMILES carbonimidic/carbamimidic
    # inorganic-acid entries (third neighbour = O/N -> excluded). Match tuple
    # stays (C, imino-N, hydroxyl-O).
    "imidic_acid": "[CX3;$([CX3](=[NX2H1])[OX2H1]);!$([CX3]([#7,#8])(=[NX2H1])[OX2H1])](=[NX2H1])[OX2H1]",
    # Hydrazonic acid R-C(=N-NH2)-OH (P-65.1.3.2 / Table 4.3 'hydrazonic acid'),
    # the =O -> =N-NH2 replacement analogue of imidic acid. W3-P02-3. The
    # geminal C-OH (sp2 C) is NOT perceived by the [OX2H][CX4] alcohol pattern,
    # so without this FG the group mis-reads as a plain hydrazone and the -OH is
    # dropped. Match tuple (C, imino-N, amino-N, hydroxyl-O). The terminal
    # amino N is [NX3H2] (unsubstituted hydrazono); the
    # `!$([CX3]([#7,#8])...)` guard keeps the third C neighbour H or carbon
    # (excludes an extra N/O, e.g. hydrazono-carbonic/amidrazone hybrids).
    "hydrazonic_acid": "[CX3;!$([CX3]([#7,#8])(=[NX2][NX3])[OX2H1])](=[NX2][NX3H2])[OX2H1]",
    # Hydroximic acid R-C(=N-OH)-OH (P-65.1.3.3), the =O -> =N-OH replacement
    # analogue. W3-P02-5. Per P-65.1.3.3.1 the PIN is the N-hydroxy derivative
    # of the corresponding imidic acid (N-hydroxyethanimidic acid), NOT the
    # general-only '-hydroximic acid' suffix; the dedicated handler builds that.
    # The geminal C-OH (sp2 C) is not seen by the alcohol pattern, so without
    # this FG the group mis-reads as a plain oxime and the -OH is dropped. Match
    # tuple (C, imino-N, O-on-N, hydroxyl-O). The `!$([CX3]([#7,#8])...)` guard
    # keeps the third C neighbour H or carbon (excludes amidoxime CC(=NO)N and
    # any hydroxyimino-carbonic hybrid).
    "hydroximic_acid": "[CX3;!$([CX3]([#7,#8])(=[NX2][OX2H1])[OX2H1])](=[NX2][OX2H1])[OX2H1]",
    "thioester": "[CX3](=O)[SX2][#6]",
    # Phase 163 Tier FRN-E: chalcogen-ester (P-65.6 ester extension) -- additive per CONTEXT D-08
    "selenoester": "[CX3](=O)[SeX2][#6]",         # R-C(=O)-Se-R' (P-65.6; Se-alkyl alkaneselenoate)
    "telluroester": "[CX3](=O)[TeX2][#6]",        # R-C(=O)-Te-R' (P-65.6; Te-alkyl alkanetelluroate)
    "acid_chloride": "[CX3](=O)[Cl]",
    "acid_bromide": "[CX3](=O)[Br]",
    "acid_fluoride": "[CX3](=O)[F]",
    "acid_iodide": "[CX3](=O)[I]",  # DATA-04: IUPAC P-65.5.1
    # Wave2 T6c: acyl PSEUDOhalides (P-65.5.2.1 — functional-class PINs
    # 'butanoyl azide' / 'propanoyl cyanide' / 'acetyl isocyanate'). Without
    # these the bare azido/isocyanate/nitrile SMARTS claim only the tail and
    # the acyl C=O is DROPPED (1-azidobutane / isocyanatoethane — wrong
    # constitution). P-41 class-10 pseudohalogen seniority: N3 > CN > NCO.
    "acyl_azide": "[CX3](=[OX1])[NX2]=[NX2,NX3+]=[NX1-]",
    "acyl_cyanide": "[CX3;!$([CX3][OX2]);!$([CX3][NX3])](=[OX1])[CX2]#[NX1]",
    "acyl_isocyanate": "[CX3](=[OX1])[NX2]=[CX2]=[OX1]",

    # === NITROGEN ACID DERIVATIVES ===
    "primary_amide": "[CX3](=O)[NX3H2]",
    "secondary_amide": "[CX3](=O)[NX3H1][#6]",
    "tertiary_amide": "[CX3](=O)[NX3]([#6])[#6]",
    "hydrazide": "[CX3](=O)[NX3][NX3]",
    # Wave2 T3d (P-66.3.4): thiohydrazide R-C(=S)-NH-NH2 (chalcogen analogue of
    # hydrazide). Cascade-suppresses thioamide + hydrazine_fg on its atoms.
    "thiohydrazide": "[CX3](=S)[NX3][NX3]",
    "imide": "[CX3](=O)[NX3][CX3](=O)",
    # Phase 163 Tier FRN-B: chalcogen-on-amide (P-66.1.4.1.1 + P-66.6.3) -- additive per CONTEXT D-08
    # AUDIT DECISION (AUDIT-FRN § 2.2): single-permissive [NX3] (NOT 3-way primary/secondary/tertiary
    # split) captures all 3 N-substitution levels per Open Question 2 + RESEARCH §3.2 line 265.
    # CR-fix (post-merge regression closure): require explicit C neighbor on the chalcogen-carbonyl
    # carbon so thiocarbamates (R-O-C(=S)-N-, R-S-C(=S)-N-) and chalcogen-ureas (N-C(=S)-N) route
    # through their own pathways rather than over-matching as thio-/seleno-/telluro-amides.
    "thioamide": "[CX3;$([CX3]([#6])(=S)[NX3])](=S)[NX3]",     # R-C(=S)-N(H,R), R=C only (P-66.1.4.1.1)
    "selenoamide": "[CX3;$([CX3]([#6])(=[SeX1])[NX3])](=[SeX1])[NX3]",  # R-C(=Se)-N, R=C only (P-66.6.3)
    "telluroamide": "[CX3;$([CX3]([#6])(=[TeX1])[NX3])](=[TeX1])[NX3]", # R-C(=Te)-N, R=C only (P-66.6.3 parallel)

    # === SULFONAMIDES ===
    "primary_sulfonamide": "[SX4](=O)(=O)[NX3H2]",
    "secondary_sulfonamide": "[SX4](=O)(=O)[NX3H1][#6]",
    "tertiary_sulfonamide": "[SX4](=O)(=O)[NX3]([#6])[#6]",
    # C1 fix (P-65.3.1): R-SO2-NH-NH2 is a sulfonohydrazide, the N-analogue of
    # sulfonic acid. The carbonyl-based hydrazide SMARTS above never matches a
    # sulfonyl S; mirror primary_sulfonamide but with the second N.
    "sulfonohydrazide": "[SX4](=O)(=O)[NX3][NX3]",
    # Wave2 T3d (P-66.1.1 Table 6.1 item 20): sulfonimidamide -S(=O)(=NH)-NH2,
    # a preselected suffix ranked just below sulfonamide (BB 18773). The whole
    # group used to drop (CS(=O)(=N)N -> 'unknown'). SMARTS mirrors
    # primary_sulfonamide with one =O replaced by the imido =NH. On sulfur, so
    # NO overlap with the carbon-based imine/oxime/hydrazone SMARTS (the
    # composite-N carbon families that need the perception-priority prototype).
    "sulfonimidamide": "[SX4](=[OX1])(=[NX2])[NX3H2]",
    # Wave2 T3d (P-66.1.1 Table 6.1 item 25): sulfinimidamide -S(=NH)-NH2 = the
    # shipped sulfonimidamide with the S(=O) removed (SX4->SX3, one fewer O).
    # SX3-vs-SX4 makes it disjoint from every shipped SMARTS -> no suppression.
    "sulfinimidamide": "[SX3](=[NX2])[NX3H2]",
    # Wave-2 P1AM (P-66.4.1.1, BB 34173): Se analogues named "similarly" --
    # -seleninimidamide (preselected) / -selenonimidamide. Te analogues are
    # DEFERRED until an OPSIN oracle exists for the -telluron- form
    # (methanetellurinimidamide parses; the selenon/telluron =O forms were
    # only OPSIN-verified for Se during planning). Fail-closed by absence.
    "seleninimidamide": "[SeX3](=[NX2])[NX3H2]",
    "selenonimidamide": "[SeX4](=[OX1])(=[NX2])[NX3H2]",
    # P-66.4.3.2: -S(=N-NH2)-NH-NH2 (cannot collide with sulfinimidamide --
    # the S-NH- nitrogen here is H1, not the [NX3H2] amide N).
    "sulfinohydrazonohydrazide": "[SX3](=[NX2][NX3])[NX3][NX3]",

    # === CARBAMATES (must check before esters -- N-C(=O)-O is more specific) ===
    "carbamate": "[NX3][CX3](=O)[OX2][#6]",

    # === UREA (must check before amides -- N-C(=O)-N is more specific) ===
    "urea": "[NX3][CX3](=O)[NX3]",

    # === THIOUREA (Wave-2 completion, P-66.1.6.1.3.3): thio-analogue of urea,
    # N-C(=S)-N. Prefix form 'carbamothioylamino' (mirror of urea's
    # carbamoylamino). More specific than thioamide (which requires a C
    # neighbour) and primary_amine — suppresses those on its atoms below. ===
    "thiourea": "[NX3][CX3](=[SX1])[NX3]",

    # === GUANIDINE (must check before imines -- N-C(=N)-N is more specific) ===
    "guanidine": "[NX3][CX3](=[NX2])[NX3]",

    # === CYANAMIDE (AM-1, P-66.1.6.2): H2N-C#N and its N-substituted
    # derivatives use 'cyanamide' as the retained parent ((propan-2-yl)cyanamide,
    # dimethylcyanamide). ACYCLIC amine N only ([NX3;!R]) — a ring N-C#N stays
    # 'piperidine-1-carbonitrile'. Collision resolver below suppresses the
    # nitrile + amine reads on these atoms so principal_group becomes None. ===
    "cyanamide": "[NX3;!R][CX2]#[NX1]",

    # === AMIDINE (IUPAC P-66.4.1: C(=N)N, less specific than guanidine) ===
    # D1: broadened from [CX3](=[NX2H])[NX3H2] (both N unsubstituted) to
    # [CX3](=[NX2])[NX3] so N-/N'-substituted amidines are classified too. This
    # matches the benzene handler's proven SMARTS (benzene.py). The broadened
    # pattern also matches the guanidine central C; the atom-overlap
    # guanidine-suppresses-amidine rule below removes those matches. Match-tuple
    # order stays (amidine_C, imino_N, amino_N) -> chains.py _TERMINAL_C_FGS
    # ['amidine']=0 remains correct. =[NX2] (not =O) still excludes urea; the N
    # (not O) neighbour still excludes imidates (COC(=N)C).
    "amidine": "[CX3](=[NX2])[NX3]",  # DATA-03 / D1
    # Wave2 T3d (P-66.4.2, BB Table 6.1 items 17-18): amidrazone / hydrazonamide
    # R-C(=N-NH2)-NH2 — an amidine whose imido =NH is a hydrazono =N-NH2. The
    # =[NX2][NX2,NX3] (imino N bonded to another N) is the discriminant vs plain
    # amidine (=NH, no N-neighbour). Ranks just below amidine; cascade-suppresses
    # amidine/hydrazone/primary_amine on its atoms.
    "hydrazonamide": "[CX3](=[NX2][NX2,NX3])[NX3]",
    # Wave2 T3d (P-66.4.3, BB 34574): hydrazidine / hydrazonohydrazide
    # R-C(=N-NH2)-NH-NH2 — the [NX3][NX3] hydrazido side is the discriminant vs
    # amidrazone's single [NX3]. Cascade-suppresses hydrazone/amidine/hydrazonamide/
    # hydrazine_fg on its atoms.
    "hydrazidine": "[CX3](=[NX2][NX3])[NX3][NX3]",
    # Wave-2 P1AM Task 7 (P-66.4.2.1, BB 34430): the R-C(=NH)-NH-NH2
    # amidrazone tautomer -> 'imidohydrazide' suffix. Terminal =NH (D1)
    # distinguishes it from hydrazonamide (C=N-N) and hydrazidine
    # (C=N-N + N-N). Cascade-suppresses amidine/hydrazine_fg.
    "imidohydrazide": "[CX3](=[NX2;D1])[NX3][NX3H2]",

    # === ISOCYANATES/ISOTHIOCYANATES (cumulated double bonds) ===
    "isocyanate": "[#6][NX2]=[CX2]=[OX1]",
    "isothiocyanate": "[#6][NX2]=[CX2]=[SX1]",

    # === N-OXIDES ===
    "n_oxide_aromatic": "[n+][O-]",
    "n_oxide_aliphatic": "[NX4+]([#6])([#6])([#6])[O-]",

    # === BORONIC ACIDS ===
    "boronic_acid": "[#6][BX3]([OX2H])([OX2H])",

    # === NITRILES ===
    "nitrile": "[CX2]#[NX1]",
    "isocyanide": "[#6][NX2]#[CX1]",
    
    # === CARBONYLS ===
    # Aldehyde: carbonyl C with 1 or 2 H (PERC-01: H2 added for formaldehyde;
    # collision resolution suppresses false positives on amides/acids)
    "aldehyde": "[CX3;H1,H2](=O)",
    "ketone": "[#6][CX3](=O)[#6]",
    "thioaldehyde": "[CX3H1](=S)",
    "thioketone": "[#6][CX3](=S)[#6]",
    # Phase 163 Tier FRN-C: chalcogen-on-aldehyde/ketone (P-66.6.3) -- additive per CONTEXT D-08
    "selenoaldehyde": "[CX3H1](=[SeX1])",         # R-C(=Se)H (P-66.6.3)
    "telluroaldehyde": "[CX3H1](=[TeX1])",        # R-C(=Te)H (P-66.6.3 parallel)
    "selenoketone": "[#6][CX3](=[SeX1])[#6]",     # R-C(=Se)-R' (P-66.6.3); selone PIN suffix form
    "telluroketone": "[#6][CX3](=[TeX1])[#6]",    # R-C(=Te)-R' (P-66.6.3 parallel); tellone PIN suffix form

    # === ALCOHOLS AND ANALOGS ===
    # Generic catch-all: any OH on sp3 carbon (IUPAC P-63.1)
    # Complements specific sub-type patterns below; ensures detection of
    # OH on carbons with non-carbon neighbors (halogens, nitrogen, sulfur)
    "alcohol": "[OX2H][CX4]",  # PERC-05: generic catch-all per D-01
    "primary_alcohol": "[OX2H][CX4H2]",
    "secondary_alcohol": "[OX2H][CX4H1]([#6])[#6]",
    "tertiary_alcohol": "[OX2H][CX4]([#6])([#6])[#6]",
    "phenol": "[OX2H][cX3]",
    "enol": "[OX2H][CX3]=[CX3]",
    "thiol": "[SX2H][#6]",
    # Wave2 T3a: carbon in match (parallel to thiol/tellurol) so
    # PG_ATTACHMENT_INDICES["selenol"] = [1] can point at the locant-bearing C.
    "selenol": "[SeX2H][#6]",
    # BBR-PERC/DEF-2 (169.7): tellurol — Te analogue of -ol/-thiol/-selenol (P-63.1.5).
    # Defines the resolver ref at line ~298 (was a dead ref, audit Dim-02 §4 #2).
    "tellurol": "[TeX2H1][#6]",

    # === HYDROPEROXIDES ===
    "hydroperoxide": "[OX2H][OX2][#6]",
    "peroxide": "[#6][OX2][OX2][#6]",
    # === CHALCOGEN HYDROPEROXOL ANALOGUES (DD2 Fix C, Phase D: P-63.4.2 / P-33.2.2(3)) ===
    # The -OOH (peroxol) sulfur/mixed analogues. Each is BOTH a suffix-capable
    # principal group (-SO-/-OS-/dithioperoxol) and, when demoted, a substituent
    # prefix (hydroxysulfanyl / sulfanyloxy / disulfanyl). The -S-OH key REPLACES
    # the Blue-Book-retired `sulfenic_acid` PIN (P-56.2 verbatim, BlueBookV2
    # line ~24326: `CH3-S-OH -> methane-SO-thioperoxol (PIN) (not methanesulfenic
    # acid)`); the collision resolver below suppresses `sulfenic_acid` on its atoms.
    # W3-P03-4 (P-65.1.5.3): the leading [#6] must be a NON-acyl carbon. An acyl
    # carbon (CH3-CO-S-OH / CH3-CO-O-SH) is a (thioperoxoic) SO-/OS-acid, NOT an
    # alcohol-class thioperoxol; matching it here emitted a WRONG name
    # ('ethane-1-OS-thioperoxol' for CC(=O)OS). Excluding acyl/thioacyl carbons
    # fails those closed (-> 'unknown') until the peroxoic-acid FRN suffix path
    # gains S/Se-infix + italic OS/SO letter-locants (OPSIN has no oracle for that
    # word-form). Genuine sp3 R-O-SH / R-S-OH (CCOS / CCSO) still match.
    "so_thioperoxol": "[#6;!$([CX3]=[OX1]);!$([CX3]=[SX1])][SX2][OX2H]",   # R-S-OH  -> -SO-thioperoxol (was retired sulfenic acid)
    "os_thioperoxol": "[#6;!$([CX3]=[OX1]);!$([CX3]=[SX1])][OX2][SX2H]",   # R-O-SH  -> -OS-thioperoxol
    "dithioperoxol": "[#6][SX2][SX2H]",    # R-S-SH  -> dithioperoxol (suffix) / disulfanyl (prefix)

    # === HYDROXYLAMINES (BBR-PERC/DEF-2, 169.7: P-68.3 class 21) ===
    # R-NH-OH / R2N-OH — the N bears an -OH and >=1 carbon. MUST be checked before
    # amines/alcohol (resolver suppresses those on its atoms). Excludes hydroxamic
    # acid (C(=O)-NH-OH, carbonyl present) and oxime (C=N-OH) via the carbonyl/=*
    # recursive exclusions. The O is on N (not C) so the `[OX2H][CX4]` alcohol never
    # matches it. Specifying only the OH + one C neighbour leaves the 3rd N connection
    # free (implicit H for R-NH-OH, or a 2nd C for R2N-OH) — so it matches BOTH forms.
    "hydroxylamine": "[OX2H1][NX3;!$([NX3][CX3]=[OX1,SX1,SeX1,TeX1]);!$([NX3]=*)][#6]",
    # Wave2 T2b: -O-NH2 on carbon = 'aminooxy' preselected prefix
    # (P-68.3.1.1.1.5; BB PIN '2-(aminooxy)ethan-1-amine'; no o-elision).
    # NX3H2 pins the terminal unsubstituted N (its only heavy neighbour is the
    # O), so N-substituted forms R'-NH-O-R stay unperceived → fail-closed
    # (their PIN is O-substituted-hydroxylamine territory, P-68.3.1.1.1.2).
    "aminooxy": "[NX3H2][OX2][#6]",
    # Wave2 T2b: -NH-X N-haloamines → compound prefixes fluoroamino/
    # chloroamino/bromoamino/iodoamino (P-35.3.1; BB verbatim '-NH-Cl
    # chloroamino (preselected prefix)'). NX3H1 with the halogen + one C pins
    # the mono-halo N-H form; N,N-dihalo and N-halo-N-alkyl stay unperceived
    # → fail-closed.
    "n_fluoroamine": "F[NX3H1][#6]",
    "n_chloroamine": "Cl[NX3H1][#6]",
    "n_bromoamine": "Br[NX3H1][#6]",
    "n_iodoamine": "I[NX3H1][#6]",
    # === AMINES ===
    # WS-A task 9 (P-66.6.1): the N carries !R — a RING nitrogen is a
    # skeletal heteroatom of a ring parent hydride (morpholine, pyrrolidine,
    # piperidine), never an amine characteristic group. Without it the
    # ring N matched tertiary_amine, became the principal group, and the
    # amine emitter CUT the ring ('N,N-dibutyl-N-cyclohexylcyclohexan-1-
    # amine' for cyclohexylmorpholine). Only the N is !R: the carbon
    # neighbours may be ring atoms (N-cyclohexylamines still match).
    "primary_amine": "[NX3;H2;!R;!$([NX3][CX3]=O);!$([NX3][CX3]=[NX2])][#6]",  # PERC-02: sp2/sp3, excludes amide/urea/guanidine N
    "secondary_amine": "[NX3;H1;!R;!$([NX3][CX3]=O);!$([NX3][CX3]=[NX2])]([CX4,cX3,$([CX3]=[CX3;!R])])[CX4,cX3,$([CX3]=[CX3;!R])]",  # DATA-02+PERC-07: sp3, aromatic, or acyclic vinyl C; exclude amides/guanidines
    "tertiary_amine": "[NX3;H0;!R;!$([NX3][CX3]=O);!$([NX3][CX3]=[NX2])]([CX4,cX3,$([CX3]=[CX3;!R])])([CX4,cX3,$([CX3]=[CX3;!R])])[CX4,cX3,$([CX3]=[CX3;!R])]",  # DATA-02+PERC-07: sp3, aromatic, or acyclic vinyl C; exclude amides/guanidines
    "aromatic_amine": "[NX3H2][cX3]",
    
    # === IMINES ===
    # Wave2 T2a (P-62.3.1.1): broadened from [CX3]=[NX2H] to admit
    # N-SUBSTITUTED imines R-CH=N-R' (BB VERBATIM 'N-methylethanimine
    # (PIN)'). Guards keep the old blast radius: !R on N — a ring C=N is a
    # skeletal feature of the ring parent (kekulized azirine/dihydroazine
    # rings must NOT gain an imine FG); the $-exclusions keep oxime
    # (C=N-OH), oxime ethers (C=N-OR), hydrazones (C=N-N<) and N-halo
    # imines (P-62.4: named as amides) out — each is its own class.
    "imine": "[CX3]=[NX2;!R;!$([NX2][OX2]);!$([NX2][NX2]);!$([NX2][NX3]);!$([NX2][F,Cl,Br,I])]",
    "oxime": "[CX3]=[NX2][OX2H]",
    "hydrazone": "[CX3]=[NX2][NX3]",
    "hydrazine_fg": "[NX3;H1;!$([NX3][CX3]=O)][NX3H2]",  # DATA-05c: P-62.4 -NH-NH2, excludes hydrazides
    
    # === ETHERS (no suffix - substitutive naming) ===
    "ether": "[OX2]([CX4])[CX4]",
    "vinyl_ether": "[OX2]([#6])[CX3]=[CX3]",
    "aromatic_ether": "[OX2]([#6])[cX3]",
    "thioether": "[SX2]([#6])[#6]",
    # BBR-PERC/DEF-2 (169.7): selenide/telluride — Se/Te ether analogues (P-63.6).
    # Named selenoether/telluroether to match (and make LIVE) the dead resolver refs
    # at lines ~284-285 (audit Dim-02 §4 #2). Prefix form = (alkyl)selanyl/tellanyl.
    "selenoether": "[SeX2]([#6])[#6]",   # R-Se-R' (P-63.6); selanyl prefix
    "telluroether": "[TeX2]([#6])[#6]",  # R-Te-R' (P-63.6 parallel); tellanyl prefix
    "disulfide": "[#6][SX2][SX2][#6]",  # DATA-05b: P-63.6.2

    # === PHOSPHORUS COMPOUNDS (check more specific first) ===
    # Phosphate esters (C-O-P bonds, not C-P bonds) - check before phosphine oxide
    "phosphate_triester": "[PX4](=O)([OX2][#6])([OX2][#6])[OX2][#6]",
    "phosphate_diester": "[PX4](=O)([OX2][#6])([OX2][#6])[OX2H1]",
    "phosphate_monoester": "[PX4](=O)([OX2][#6])([OX2H1])[OX2H1]",
    # Phosphine oxide: R3P=O - three C attached to P(V)
    "phosphine_oxide": "[PX4](=O)([#6])([#6])[#6]",
    # Phosphines (P(III)) - check last as parent hydride
    "tertiary_phosphine": "[PX3]([#6])([#6])[#6]",
    "secondary_phosphine": "[PX3H1]([#6])[#6]",
    "primary_phosphine": "[PX3H2][#6]",

    # === SULFUR OXIDATION STATES (check more specific first) ===
    # Sulfone: S with 2 =O and 2 C neighbors (R-SO2-R')
    "sulfone": "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    # Sulfoxide: S with 1 =O and 2 C neighbors (R-SO-R')
    "sulfoxide": "[SX3](=[OX1])([#6])[#6]",
    
    # === UNSATURATION ===
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    
    # === HALOGENS (always prefixes) ===
    "fluoro": "[FX1][#6]",
    "chloro": "[ClX1][#6]",
    "bromo": "[BrX1][#6]",
    "iodo": "[IX1][#6]",

    # === HYDROXAMIC ACIDS (PERC-03: P-65.3.3) ===
    "hydroxamic_acid": "[CX3](=O)[NX3;H1][OX2H]",  # R-C(=O)-NH-OH: free hydroxamic acid only

    # === CYANATES / THIOCYANATES (PERC-03: P-65.5) ===
    "cyanate": "[OX2][CX2]#[NX1]",
    "thiocyanate": "[SX2][CX2]#[NX1]",

    # === AZO (PERC-03: P-67.2) ===
    "azo": "[#6][NX2]=[NX2][#6]",
    # Wave2 T2b: terminal HN=N- = 'diazenyl' preselected prefix (P-35.2.2;
    # BB 'diazenyl (preselected prefix; see P-12.2)', ledger 8-diazenyl-
    # octanoic acid). NX2H1 pins the terminal =N-H (its only heavy neighbour
    # is the other N); internal R-N=N-R' stays 'azo' (C on both ends, above).
    "diazenyl": "[NX2H1]=[NX2][#6]",

    # === OTHER ===
    # BBR-PERC/DEF-4 (169.7): C-attached nitro only (P-65.3.1) via recursive-env so the
    # match-tuple arity is unchanged; stops nitrate ESTERS (R-O-NO2, CCO[N+](=O)[O-]) from
    # false-matching as nitro. Aliphatic + aromatic C both satisfy [#6].
    "nitro": "[NX3+;$([NX3+][#6])](=O)[O-]",
    # WSD-05 (PERC-03): [#6] C-attachment guard, uniform with the sibling `nitro`.
    # Without it the N=O of a nitrite ester (R-O-N=O) false-matched `nitroso` and
    # won the collision -> `nitrosoethane` for ethyl nitrite. The guard limits
    # `nitroso` to a genuine C-nitroso (N bonded to carbon, P-66.5.2).
    "nitroso": "[NX2;$([NX2][#6])]=[OX1]",
    "azido": "[N;+0]=[N+]=[N-]",  # SUB-01/C2: organic azide R-N=[N+]=[N-] (attach N is NX2 neutral; the old [NX1]=... never matched RDKit canonical azides)
    "diazo": "[#6]=[NX2+]=[NX1-]",  # DATA-05a: P-61.5 diazo group
    # BBR-PERC/DEF-3 (169.7): nitrite ester R-O-N=O (P-65.5). Added so its O,N are
    # excludable from the skeletal-replacement backbone (see get_chain_excluded_atoms).
    "nitrite": "[#6][OX2][NX2]=[OX1]",
}

# Pre-compile all SMARTS patterns once at module load (avoid recompilation per molecule)
_COMPILED_FG_SMARTS = {}
for _fg_name, _smarts in FUNCTIONAL_GROUP_SMARTS.items():
    _pat = Chem.MolFromSmarts(_smarts)
    if _pat is not None:
        _COMPILED_FG_SMARTS[_fg_name] = _pat


# W3-P06 (P-65.7.1 / P-65.7.3 / P-65.7.4): anhydride BRIDGE variants whose bridge
# is NOT the carbon/O-only single O of the base ANHYDRIDE_SMARTS
# ([CX3](=O)[OX2][CX3](=O)). Each match is FOLDED into the 'anhydride'
# functional-group bucket in detect_functional_groups() (below) so
# get_principal_group() returns 'anhydride' and rules.anhydrides.name_anhydride
# (which re-detects the bridge type on the mol) produces the functional-class PIN.
# Kept SEPARATE from FUNCTIONAL_GROUP_SMARTS so the base-FG count contract and the
# per-FG collision resolver are untouched; the folded tuples are shape-heterogeneous
# but every 'anhydride' consumer (get_anhydride_consumed_atoms, _filter_consumed,
# the resolver) iterates atoms only.
_ANHYDRIDE_BRIDGE_SMARTS = {
    # sulfonic anhydride R-SO2-O-SO2-R' (P-65.7.1; BB 32247 'benzenesulfonic anhydride')
    "sulfonic_anhydride": "[SX4](=O)(=O)[OX2][SX4](=O)(=O)",
}
_COMPILED_ANHYDRIDE_BRIDGE = {}
for _bk, _bsmarts in _ANHYDRIDE_BRIDGE_SMARTS.items():
    _bpat = Chem.MolFromSmarts(_bsmarts)
    if _bpat is not None:
        _COMPILED_ANHYDRIDE_BRIDGE[_bk] = _bpat


def detect_functional_groups(mol) -> Dict[str, List[Tuple[int, ...]]]:
    """
    Detect all functional groups in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Dictionary mapping functional group names to lists of atom index tuples.
        Each tuple contains the indices of atoms in one instance of that group.

    Example:
        >>> mol = Chem.MolFromSmiles("CC(=O)O")  # acetic acid
        >>> groups = detect_functional_groups(mol)
        >>> "carboxylic_acid" in groups
        True
        >>> len(groups["carboxylic_acid"])
        1
    """
    results = defaultdict(list)

    for fg_name, pattern in _COMPILED_FG_SMARTS.items():
        matches = mol.GetSubstructMatches(pattern, uniquify=True)
        for match in matches:
            results[fg_name].append(match)

    # A ring-internal C=N-N is the backbone of an azoline/azole ring (e.g.
    # 4,5-dihydro-1H-pyrazole), NOT a hydrazone principal group (a hydrazone is
    # the acyclic R2C=N-NH2). Drop hydrazone matches whose C=N AND N-N bonds are
    # BOTH ring bonds, so the ring namer (not the hydrazone PCG path) handles
    # them. Acyclic hydrazones keep their match; aromatic azoles never match the
    # explicit-'=' SMARTS. (P-31.1.4 ring features vs P-66.6 hydrazone PCG.)
    if results.get("hydrazone"):
        kept = []
        for match in results["hydrazone"]:
            c_idx, n1_idx, n2_idx = match[0], match[1], match[2]
            b_cn = mol.GetBondBetweenAtoms(c_idx, n1_idx)
            b_nn = mol.GetBondBetweenAtoms(n1_idx, n2_idx)
            ring_internal = (b_cn is not None and b_cn.IsInRing()
                             and b_nn is not None and b_nn.IsInRing())
            if not ring_internal:
                kept.append(match)
        if kept:
            results["hydrazone"] = kept
        else:
            del results["hydrazone"]

    # W3-P06 (P-65.7.1/.3/.4): fold anhydride bridge-variant matches into the
    # 'anhydride' bucket so principal_group -> 'anhydride' (name_anhydride
    # re-detects the bridge type). Run BEFORE the collision resolver so the
    # ('anhydride', [...]) suppression rule clears the sub-component reads
    # (thioester/peroxide/ketone) on the folded atoms.
    for _bk, _bpat in _COMPILED_ANHYDRIDE_BRIDGE.items():
        for _bm in mol.GetSubstructMatches(_bpat, uniquify=True):
            results["anhydride"].append(_bm)

    # Post-processing: remove generic FG matches that overlap with more-specific FGs
    results = _resolve_fg_collisions(results)

    return dict(results)


def detect_features(mol) -> Dict[str, List[Tuple[int, ...]]]:
    """BBR-PERC (Phase 169.7) — the single shared feature-perception entry point.

    The ONE place every route (neutral and charged) reads functional-group classes
    from, so a mis-route can never produce a DIFFERENT feature set (audit Dim-02 §4
    #1: "two parallel detectors with disjoint coverage"). Today it is a thin wrapper
    over ``detect_functional_groups`` (which now includes the charged + missing-class
    SMARTS); the orthogonal charge-SITE scan (``ions.get_ion_sites``) stays separate
    — it answers "where are the charges", not "what FG classes are present". Keeping
    a single accessor lets future consolidation route through one function.
    """
    return detect_functional_groups(mol)


# BBR-PERC/DEF-3 (Phase 169.7): prefix-only characteristic groups whose heteroatoms
# (N/O) are NOT skeletal/parent-chain atoms (P-59 / P-65.5 / P-61). They must never be
# walked into an aza/oxa chain (e.g. azidomethane CN=[N+]=[N-] -> wrong '2,3-diazabutane').
_CHAIN_EXCLUDED_FG = frozenset({
    "azido", "diazo", "nitroso", "nitrite", "nitro",
    "n_oxide_aromatic", "n_oxide_aliphatic",
})


def get_chain_excluded_atoms(mol) -> Set[int]:
    """Return the N/O atom indices of every prefix-only characteristic group
    (``_CHAIN_EXCLUDED_FG``) present in *mol*.

    These graph indices, derived from perception's OWN ``detect_functional_groups``
    matches (NOT a per-FG SMARTS blocklist — CONTEXT D-04), are the atoms that must
    be excluded from skeletal/parent-chain finding. Only the heteroatoms (N=7, O=8)
    of each match are returned, so the carbon attachment point is preserved (the
    chain can still terminate there).
    """
    results = detect_functional_groups(mol)
    excluded: Set[int] = set()
    for fg_name in _CHAIN_EXCLUDED_FG:
        for match in results.get(fg_name, []):
            for idx in match:
                if mol.GetAtomWithIdx(idx).GetAtomicNum() in (7, 8):
                    excluded.add(idx)
    return excluded


def _resolve_fg_collisions(results):
    """Remove generic FG matches that overlap with more-specific FGs.

    Collision rules:
    - urea atoms should NOT also be detected as primary_amide/secondary_amide/tertiary_amide
    - guanidine atoms should NOT also be detected as imine
    - carbamate atoms should NOT also be detected as ester or amide (primary/secondary/tertiary)
    - isocyanate/isothiocyanate atoms should NOT also be detected as nitrile or primary_amide
    """
    for fg_specific, fg_generic_list in [
        ('urea', ['primary_amide', 'secondary_amide', 'tertiary_amide']),
        # Wave-2 completion (P-66.1.6.1.3.3): thiourea N-C(=S)-N owns its whole
        # unit — suppress the terminal-N amine perception and any thioamide/
        # thioketone match on its atoms (thiourea is the more specific FG).
        ('thiourea', ['thioamide', 'thioketone',
                      'primary_amine', 'secondary_amine', 'tertiary_amine']),
        ('guanidine', ['imine']),
        # AM-1 (P-66.1.6.2): cyanamide N-C#N owns its whole unit — suppress the
        # nitrile and amine reads on its atoms so principal_group -> None and the
        # cyanamide handler names it. Atom-scoped: a separate nitrile/amine
        # elsewhere is untouched.
        ('cyanamide', ['nitrile', 'primary_amine', 'secondary_amine',
                       'tertiary_amine']),
        ('carbamate', ['ester', 'primary_amide', 'secondary_amide', 'tertiary_amide']),
        ('isocyanate', ['nitrile', 'primary_amide']),
        ('isothiocyanate', ['nitrile', 'primary_amide']),
        # Wave2 T6c: acyl pseudohalides (P-65.5.2.1) own their whole
        # C(=O)-pseudohalogen unit — atom-overlap-scoped, so a separate
        # azide/nitrile/ketone elsewhere in the molecule is untouched.
        ('acyl_azide', ['azido', 'ketone', 'aldehyde', 'imine', 'azo']),
        ('acyl_cyanide', ['nitrile', 'ketone', 'aldehyde']),
        ('acyl_isocyanate', ['isocyanate', 'ketone', 'aldehyde', 'imine',
                             'nitrile', 'primary_amide', 'secondary_amide']),
        # Carbamic acid: N-C(=O)-OH must NOT also match carboxylic_acid or amide
        ('carbamic_acid', ['carboxylic_acid', 'primary_amide', 'secondary_amide', 'tertiary_amide']),
        # Thiocarboxylic acids: SH in C(=O)SH or C(=S)SH must NOT match thiol
        # C(=O)SH must NOT match thioester either (C(=O)S is substructure of both)
        ('thioic_S_acid', ['thiol', 'thioester']),
        # W3-P03-3 (P-65.1.5.1): the C1 case H-C(=S)-SH / H-C(=S)-OH bears an H on
        # the thiocarbonyl C, so the broad thioaldehyde SMARTS [CX3H1](=S) also
        # matches it (exactly parallel to thioic_S_acid suppressing the broad
        # aldehyde below). The thio/dithio-acid FG owns that carbon -> suppress
        # thioaldehyde so C1 acids don't gain a spurious '1-thioxo' prefix
        # (e.g. methanethioic O-acid OC=S; the C2+ cases never matched thioaldehyde
        # because their carbon has no H).
        ('dithioic_acid', ['thiol', 'thioketone', 'thioaldehyde']),
        # C(=S)OH should not collide with carboxylic_acid (different SMARTS: =S vs =O)
        # but suppress thioketone/thioaldehyde matches on the C=S carbon
        ('thioic_O_acid', ['thioketone', 'thioaldehyde']),
        # Phase 163 Tier FRN-A: chalcogen-acid suppressions
        # (mirror thioic_S_acid -> thiol+thioester at line 228 above; AUDIT-FRN § 2.1)
        # Forward-reference note: selenoester/telluroester/tellurol added in
        # commits 163-02-03 (chalcogen-ketones) + 163-02-05 (chalcogen-esters);
        # unknown FG names are harmlessly no-op'd by the resolver loop below.
        ('selenoic_Se_acid', ['selenol', 'selenoester', 'thioester']),
        ('diselenoic_acid', ['selenol', 'selenoketone']),
        ('selenoic_O_acid', ['selenoketone', 'carboxylic_acid']),
        ('telluroic_Te_acid', ['tellurol', 'telluroester', 'thioester']),
        ('ditelluroic_acid', ['tellurol', 'telluroketone']),
        ('telluroic_O_acid', ['telluroketone', 'carboxylic_acid']),
        # Phase 163 Tier FRN-B: chalcogen-amide suppressions (AUDIT-FRN § 2.2 + RESEARCH §3.2).
        # Single-permissive [NX3] match captures =[S,Se,Te]-N(H,R) at all 3 N-degrees;
        # downstream N-degree inspection happens at assembly time.
        ('thioamide', ['thioketone', 'primary_amine', 'secondary_amine', 'tertiary_amine']),
        ('selenoamide', ['selenoketone', 'primary_amine', 'secondary_amine', 'tertiary_amine']),
        ('telluroamide', ['telluroketone', 'primary_amine', 'secondary_amine', 'tertiary_amine']),
        # Phase 163 Tier FRN-C: chalcogen-aldehyde/ketone suppressions (AUDIT-FRN § 2.3 + RESEARCH §3.3).
        # Defensive suppression mirrors existing thioaldehyde discipline; =O vs =Se/=Te should not
        # overlap structurally but the cascade preserves parallelism for downstream safety.
        ('selenoaldehyde', ['aldehyde', 'ketone']),
        ('telluroaldehyde', ['aldehyde', 'ketone']),
        ('selenoketone', ['selenoether', 'selenoester']),
        ('telluroketone', ['telluroether', 'telluroester']),
        # Phase 163 Tier FRN-D: iminoester suppressions (AUDIT-FRN § 2.4 + RESEARCH §3.4 + §9 Risk B).
        # Risk B mitigation: iminoester -> ester defensive suppression mirrors the existing thio*
        # cascade discipline; imine/primary_amine suppress the =NH from being double-claimed;
        # ether suppresses the -O-C portion from being double-claimed.
        ('iminoester', ['ester', 'imine', 'primary_amine', 'ether']),
        # W2F-P6 (P-66.1.6.1.2.1): the N-substituted carbamimidate ester
        # R''2N-C(=NR')-O-R owns its whole -O-C(=N-)-N unit. Suppress the
        # constituent ester (O-C), imine/amine (the two N's), and ether reads so
        # they are not double-claimed. Also suppress the FALSE-POSITIVE amidine
        # match: a C bearing =N + N + an ester -O-C is a carbamimidate ESTER,
        # not a plain amidine (amidine is R-C(=NH)-NH2, no O). This makes the
        # imidate handler win over the amidine namer for the N-substituted case
        # (the unsubstituted case already routes via iminoester, which outranks
        # amidine in SENIORITY_ORDER). Atom-overlap-scoped -> a separate amidine
        # elsewhere is untouched.
        ('carbamimidate', ['ester', 'imine', 'primary_amine',
                           'secondary_amine', 'tertiary_amine', 'ether',
                           'amidine']),
        # Phase 163 Tier FRN-E: chalcogen-ester suppressions (AUDIT-FRN § 2.5 + RESEARCH §3.5).
        # Note: Tier FRN-A suppressions at lines above already declared
        # ('selenoic_Se_acid', ['selenol', 'selenoester', 'thioester']) and
        # ('telluroic_Te_acid', ['tellurol', 'telluroester', 'thioester']) anticipating
        # the existence of selenoester/telluroester per RESEARCH §3.5 line 306. Now that
        # those SMARTS exist, the suppression cascade works as designed.
        ('selenoester', ['selenol']),
        ('telluroester', ['tellurol']),
        # Ester O-Ar bond should NOT also match aromatic_ether.
        # The ester oxygen in -C(=O)-O-Ar is part of the ester, not a separate
        # aromatic ether; without this, phenyl esters double-name as both
        # "phenoxy" and "phenoxycarbonyl".
        ('ester', ['aromatic_ether']),
        # PERC-01: broadened aldehyde [CX3;H1,H2](=O) now matches C=O in acid
        # derivatives (formates, formamides, etc.); suppress aldehyde on overlap
        ('carboxylic_acid', ['aldehyde']),
        ('ester', ['aldehyde']),
        ('anhydride', ['aldehyde']),
        ('acid_chloride', ['aldehyde']),
        ('acid_bromide', ['aldehyde']),
        ('acid_fluoride', ['aldehyde']),
        ('primary_amide', ['aldehyde']),
        ('secondary_amide', ['aldehyde']),
        ('tertiary_amide', ['aldehyde']),
        ('hydrazide', ['aldehyde']),
        ('imide', ['aldehyde']),
        ('thioic_S_acid', ['aldehyde']),
        ('carbamic_acid', ['aldehyde']),
        # USUB-11: imide suppresses overlapping amide matches to prevent
        # double-counting C=O groups (one as amide, one as ketone)
        ('imide', ['primary_amide', 'secondary_amide', 'tertiary_amide']),
        # PERC-02: broadened amine pattern overlaps with aromatic_amine on aromatic carbons
        ('aromatic_amine', ['primary_amine']),
        # PERC-03: hydroxamic acid suppresses amide + alcohol false positives
        ('hydroxamic_acid', ['primary_amide', 'secondary_amide', 'primary_alcohol',
                             'secondary_alcohol', 'tertiary_alcohol', 'carboxylic_acid']),
        # PERC-03: cyanate/thiocyanate suppress ether/thioether + nitrile
        ('cyanate', ['ether', 'nitrile']),
        ('thiocyanate', ['thioether', 'nitrile']),
        # PERC-03: azo suppresses imine
        ('azo', ['imine']),
        # DATA-03: guanidine suppresses amidine (guanidine is more specific)
        ('guanidine', ['amidine']),
        # Wave2 completion C (P-61.4): the diazo C=[N+] matches the imine
        # SMARTS, injecting a bogus 'imino' prefix that the validity gate
        # then suppressed (ethyl diazoacetate / diazomethane were unknown).
        ('diazo', ['imine']),
        # Wave2 completion (P-43.1): the acyl -OOH matches the hydroperoxide
        # SMARTS; the peroxy-acid FG owns those atoms.
        ('peroxy_acid', ['hydroperoxide', 'peroxide', 'ester', 'ketone',
                         'aldehyde', 'carboxylic_acid']),
        # Wave2 completion (P-65.1.3.1): the C=NH of an imidic acid matches
        # the imine SMARTS and its -OH the alcohol/enol patterns.
        ('imidic_acid', ['imine', 'alcohol', 'primary_alcohol',
                         'secondary_alcohol', 'tertiary_alcohol', 'enol']),
        # W3-P02-3 (P-65.1.3.2): the C=N-NH2 of a hydrazonic acid matches the
        # hydrazone SMARTS; the hydrazonic_acid FG owns the whole geminal
        # C(=N-NH2)(OH) unit (suffix 'hydrazonic acid' / demoted hydroxy +
        # hydrazinylidene). Also defensively clears imine/alcohol/enol on its
        # atoms (the sp2 C keeps alcohol/enol from matching anyway).
        ('hydrazonic_acid', ['hydrazone', 'imine', 'alcohol', 'primary_alcohol',
                             'secondary_alcohol', 'tertiary_alcohol', 'enol']),
        # W3-P02-5 (P-65.1.3.3): the C=N-OH of a hydroximic acid matches the
        # oxime SMARTS; the hydroximic_acid FG owns the whole geminal
        # C(=N-OH)(OH) unit (PIN = N-hydroxy + imidic acid; demoted = hydroxy +
        # hydroxyimino). Clears oxime + defensively imine/alcohol/enol.
        ('hydroximic_acid', ['oxime', 'imine', 'alcohol', 'primary_alcohol',
                             'secondary_alcohol', 'tertiary_alcohol', 'enol']),
        # Wave2 completion (P-66.4.3.2): the S-hydrazido/hydrazono N pairs
        # match the hydrazine/hydrazone/imine/amine patterns.
        ('sulfinohydrazonohydrazide', ['hydrazine_fg', 'sulfinimidamide',
                                       'hydrazone', 'imine', 'primary_amine']),
        # Wave2 T3d composite-N precedence (ORDER LOAD-BEARING — the resolver
        # applies top-to-bottom on live results):
        #  1. hydrazidine (R-C(=N-NH2)-NH-NH2, the most specific: 2-N hydrazido
        #     side) clears hydrazonamide/hydrazone/amidine/hydrazine_fg/primary_amine/
        #     imine on its atoms.
        # Wave-2 P1AM Task 7 (P-66.4.2.1): the imidohydrazide tautomer
        # R-C(=NH)-NH-NH2 -- terminal =NH (D1) -- clears the amidine (the
        # C=NH + first NH) and hydrazine_fg (the NH-NH2) matches on its atoms.
        # Placed BEFORE hydrazidine so the most-specific composite wins first.
        ('imidohydrazide', ['amidine', 'hydrazine_fg', 'imine']),
        ('hydrazidine', ['hydrazonamide', 'hydrazone', 'amidine',
                         'hydrazine_fg', 'primary_amine', 'imine']),
        #  2. hydrazonamide (amidrazone, 1-N amino side) clears amidine/hydrazone/
        #     primary_amine. Placed BEFORE ('guanidine',['hydrazonamide']) so on the
        #     carbonic-diamide NC(=NN)N it strips amidine/hydrazone first, THEN
        #     guanidine strips hydrazonamide -> guanidine cleanly wins.
        ('hydrazonamide', ['amidine', 'hydrazone', 'primary_amine']),
        #  3. guanidine (3-N carbon) owns the carbonic-diamide case.
        ('guanidine', ['hydrazonamide', 'hydrazidine']),
        # DATA-03 + Wave2 T3d: amidine suppresses imine/primary_amine, AND oxime
        # (the amidoxime -C(=N-OH)-NH2 is a P-66.4.4 N'-hydroxy amidine, NOT an
        # oxime; atom-scoped so standalone oximes CC=NO are untouched).
        ('amidine', ['imine', 'primary_amine', 'oxime']),
        # W3-P02-7 (P-65.1.7.2.2 / P-41): an N-acyl amidine R-C(=NH)-N(R')-C(=O)-R''
        # is a carboxAMIDE bearing an N-imidoyl substituent, NOT a free amidine —
        # carboxamide is SENIOR to carboximidamide (P-41). When an amidine's
        # amino-N is (also) an amide nitrogen (secondary/tertiary amide match
        # shares that N and/or the imidoyl C), suppress the amidine so the amide
        # is the sole principal group and the C(=NH)- becomes the N-'{stem}animidoyl'
        # substituent (BB 30462: 'N-ethanimidoyl-N-methylacetamide'). Atom-overlap-
        # scoped: a genuinely separate amidine elsewhere is untouched. Placed AFTER
        # the amidine rule above so amidine's own imine/oxime suppressions fire first.
        ('secondary_amide', ['amidine']),
        ('tertiary_amide', ['amidine']),
        # Wave2 T3d (P-66.3.4): thiohydrazide suppresses thioamide + hydrazine_fg.
        ('thiohydrazide', ['thioamide', 'hydrazine_fg']),
        # DATA-04: acid iodide suppresses aldehyde (parallel to other acid halides)
        ('acid_iodide', ['aldehyde']),
        # DATA-05b: disulfide suppresses thioether if S atoms overlap
        ('disulfide', ['thioether']),
        # DD2 Fix C (Phase D, P-63.4.2 / P-56.2): the chalcogen hydroperoxol
        # analogues supersede the legacy/retired generics on their atoms.
        # -S-OH (`so_thioperoxol`) REPLACES the Blue-Book-retired `sulfenic acid`
        # PIN; it also pre-empts any `thiol`/`thioether` read on its divalent S.
        ('so_thioperoxol', ['sulfenic_acid', 'thiol', 'thioether', 'alcohol',
                            'primary_alcohol', 'secondary_alcohol', 'tertiary_alcohol']),
        # -O-SH (`os_thioperoxol`): the S bears the H but is bonded to O, not C, so
        # `thiol` would not match anyway; suppress any `ether`/`thioether` read.
        ('os_thioperoxol', ['thiol', 'ether', 'thioether']),
        # -S-SH (`dithioperoxol`): terminal disulfanyl. The H-bearing S is bonded
        # to S (not C) so `thiol` cannot match; the inner S-S has only one C end so
        # the carbon-flanked `disulfide` cannot match — suppress defensively anyway.
        ('dithioperoxol', ['thiol', 'thioether', 'disulfide']),
        # DATA-05c: hydrazine suppresses primary_amine on its -NH2 nitrogen
        ('hydrazine_fg', ['primary_amine']),
        # DATA-05c: hydrazide is more specific than hydrazine_fg
        ('hydrazide', ['hydrazine_fg']),
        # C1 fix (P-65.3.1): sulfonohydrazide (R-SO2-NH-NH2) is the most specific
        # read on its N-N atoms -- suppress the generic hydrazine and any
        # sulfonamide/hydrazide read that shares those atoms. (The carbonyl-based
        # hydrazide SMARTS cannot match a sulfonyl anyway; suppress defensively.)
        ('sulfonohydrazide', ['hydrazine_fg', 'primary_sulfonamide',
                              'secondary_sulfonamide', 'tertiary_sulfonamide',
                              'hydrazide']),
        # PERC-05: specific alcohol subtypes suppress generic "alcohol" on same atoms
        ('primary_alcohol', ['alcohol']),
        ('secondary_alcohol', ['alcohol']),
        ('tertiary_alcohol', ['alcohol']),
        ('phenol', ['alcohol']),
        ('enol', ['alcohol']),
        # PERC-05: hydroxamic acid suppresses generic alcohol too
        ('hydroxamic_acid', ['alcohol']),
        # BBR-PERC (169.7): hydroxylamine (R-NH-OH) suppresses amine + alcohol on its
        # N/O atoms (the N is an amine-N and the O an -OH to the generic patterns).
        ('hydroxylamine', ['primary_amine', 'secondary_amine', 'tertiary_amine',
                           'aromatic_amine', 'alcohol', 'primary_alcohol',
                           'secondary_alcohol', 'tertiary_alcohol']),
        # BBR-PERC (169.7): carbonic acid (HO-C(=O)-OH) suppresses the carboxylic_acid
        # + ester generics on its carbon (it is a P-65.2.1 functional parent, not a
        # carboxylic acid).
        ('carbonic_acid', ['carboxylic_acid', 'ester']),
        # BBR-PERC (169.7): free inorganic oxoacids suppress the (now [#6]-tightened)
        # carbon-acid + ester generics on their atoms for robustness (P-67 parents).
        ('phosphoric_acid', ['phosphonic_acid', 'phosphate_monoester',
                             'phosphate_diester', 'phosphate_triester']),
        ('sulfuric_acid', ['sulfonic_acid']),
        # W3-P04 (P-65.3.1.2): the -OO- modified sulfonic acid owns its whole
        # R-SO2-OOH unit. The generics cannot structurally match (verified: the
        # -OH is not on S, the inner peroxo-O is on S not C, the S bears only one
        # C) but suppress defensively so a future SMARTS broadening cannot steal
        # the group or double-name the -OOH as a hydroperoxy prefix.
        ('sulfonoperoxoic_acid', ['sulfonic_acid', 'sulfuric_acid', 'sulfone',
                                  'hydroperoxide', 'peroxide']),
        # W3-P04 (P-65.3.1.3): the -SH-modified sulfonic acid owns its whole
        # R-SO2-SH unit. Generics cannot match (the -SH S is on S not C; the
        # sulfonyl S bears one C) but suppress defensively.
        ('sulfonothioic_S_acid', ['sulfonic_acid', 'sulfuric_acid', 'sulfone',
                                  'thiol', 'thioether', 'disulfide']),
        # W3-P04 (P-65.3.1.4): the =NH-modified sulfinic acid owns its
        # R-S(=NH)-OH unit; suppress the generic sulfinic read and any imine
        # (S=N) / alcohol (S-OH) reads defensively.
        ('sulfinimidic_acid', ['sulfinic_acid', 'imine', 'alcohol',
                               'primary_alcohol', 'secondary_alcohol',
                               'tertiary_alcohol']),
        # W3-P04 (P-65.3.1.4 / P-65.3.1.5): the =NH-modified sulfonic acid owns
        # its R-S(=O)(=NH)-OH unit; suppress the generic sulfonic/sulfone read
        # and any imine (S=N) / oxime (=N-OH, the N-hydroxy variant) /
        # hydroxylamine / alcohol reads on its atoms.
        ('sulfonimidic_acid', ['sulfonic_acid', 'sulfuric_acid', 'sulfone',
                               'imine', 'oxime', 'hydroxylamine', 'alcohol',
                               'primary_alcohol', 'secondary_alcohol',
                               'tertiary_alcohol']),
        ('nitric_acid', ['nitro', 'nitroso']),
        # WSD-05 (PERC-03): a nitrite ester (R-O-N=O) suppresses any residual
        # `nitroso` on its atoms — makes the O-vs-C precedence explicit/robust on
        # top of the [#6] guard above.
        ('nitrite', ['nitroso']),
    ]:
        if fg_specific in results:
            specific_atoms = set()
            for match in results[fg_specific]:
                specific_atoms.update(match)

            for fg_generic in fg_generic_list:
                if fg_generic in results:
                    # Remove generic matches where ANY atom overlaps with specific
                    results[fg_generic] = [
                        m for m in results[fg_generic]
                        if not any(atom in specific_atoms for atom in m)
                    ]
                    # Clean up empty lists
                    if not results[fg_generic]:
                        del results[fg_generic]

    # Phosphate specificity: more-specific phosphate esters suppress
    # less-specific phosphate esters for the same P atom.
    # NOTE: phosphonic_acid is NOT suppressed -- it remains as principal group
    # candidate for molecules with C-O-P(=O)(OH)2 because the naming pipeline
    # uses its "phosphono" prefix. Full phosphate principal group naming is
    # deferred to a future phase.
    _phosphate_suppress = [
        ('phosphate_triester', ['phosphate_diester', 'phosphate_monoester']),
        ('phosphate_diester', ['phosphate_monoester']),
    ]
    for fg_specific, fg_generic_list in _phosphate_suppress:
        if fg_specific in results:
            # Collect P atom indices from specific matches (P is always first atom in SMARTS)
            specific_p_atoms = {m[0] for m in results[fg_specific]}
            for fg_generic in fg_generic_list:
                if fg_generic in results:
                    results[fg_generic] = [
                        m for m in results[fg_generic]
                        if m[0] not in specific_p_atoms
                    ]
                    if not results[fg_generic]:
                        del results[fg_generic]

    return results


def has_functional_group(mol, fg_name: str) -> bool:
    """
    Check if molecule contains a specific functional group.
    
    Args:
        mol: RDKit Mol object
        fg_name: Name of functional group (must be in FUNCTIONAL_GROUP_SMARTS)
        
    Returns:
        True if functional group is present
    """
    pattern = _COMPILED_FG_SMARTS.get(fg_name)
    if pattern is None:
        return False

    return mol.HasSubstructMatch(pattern)


def get_functional_group_atoms(mol, fg_name: str) -> List[Tuple[int, ...]]:
    """
    Get atom indices for all instances of a specific functional group.
    
    Args:
        mol: RDKit Mol object
        fg_name: Name of functional group
        
    Returns:
        List of tuples of atom indices
    """
    pattern = _COMPILED_FG_SMARTS.get(fg_name)
    if pattern is None:
        return []

    return list(mol.GetSubstructMatches(pattern, uniquify=True))


def count_functional_groups(mol) -> Dict[str, int]:
    """
    Count occurrences of each functional group.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Dictionary mapping functional group names to counts
    """
    groups = detect_functional_groups(mol)
    return {name: len(matches) for name, matches in groups.items()}
