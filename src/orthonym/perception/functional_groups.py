"""
Functional group detection using SMARTS patterns.

Groups are ordered by IUPAC seniority (highest priority first).
The principal group (highest seniority) becomes the suffix;
all others become prefixes.
"""

import weakref
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from rdkit import Chem

# SMARTS patterns ordered by IUPAC seniority to
# First match = highest priority = principal group
FUNCTIONAL_GROUP_SMARTS = {
    # === CHARGED CHARACTERISTIC GROUPS (functional-group perception fix/, 169.7: classes 4/6) ===
    # Added so the NEUTRAL FG layer can PERCEIVE ionic groups too — a charged molecule
    # that reaches this detector with a residual charge (the mis-route exposure) no
    # longer silently loses its group. The normal charged path neutralizes BEFORE this
    # detector runs (charged_router re-enters Orthonym.name on the neutral form), so
    # these fire only on the mis-route case. Every pattern requires a formal charge, so
    # NEUTRAL molecules never match. Suffixes: (anions) / (cations).
    "carboxylate": "[#6][CX3](=O)[O-]",            # (-oate)
    "sulfonate": "[#6][SX4](=O)(=O)[O-]",          # (-sulfonate)
    "phosphonate": "[#6][PX4](=O)([O-,OX2H1])[O-]",# (-phosphonate; mono/di-deprotonated)
    "thiolate": "[#6][S-]",                        # (-thiolate)
    "phenolate": "[c][O-]",                        # (aromatic-O anion)
    "ammonium": "[NX4+]",                          # (protonated/quaternary N+)
    # === ACIDS (highest priority) ===
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    # Peroxy acid R-C(=O)-OOH / Table 4.3 'peroxoic acid'); the [#6]
    # guard keeps OOC(=O)O routing to the exact-SMILES carbonoperoxoic entry.
    "peroxy_acid": "[CX3;$([CX3]([#6])(=[OX1])[OX2][OX2H1])](=[OX1])[OX2][OX2H1]",
    # Thiocarboxylic acids (IUPAC -- rank just below carboxylic acid
    "thioic_S_acid": "[CX3](=O)[SX2H1]",    # R-C(=O)-SH -> thioic S-acid
    "thioic_O_acid": "[CX3](=S)[OX2H1]",    # R-C(=S)-OH -> thioic O-acid
    "dithioic_acid": "[CX3](=S)[SX2H1]",    # R-C(=S)-SH -> dithioic acid
    # a phase Tier FRN-A: chalcogen-on-acid -- additive per internal notes
    "selenoic_Se_acid": "[CX3](=O)[SeX2H1]",     # R-C(=O)-SeH; AUDIT-FRN)
    "selenoic_O_acid": "[CX3](=[SeX1])[OX2H1]",  # R-C(=Se)-OH; AUDIT-FRN)
    "diselenoic_acid": "[CX3](=[SeX1])[SeX2H1]", # R-C(=Se)-SeH; AUDIT-FRN)
    "telluroic_Te_acid": "[CX3](=O)[TeX2H1]",    # R-C(=O)-TeH parallel; AUDIT-FRN)
    "telluroic_O_acid": "[CX3](=[TeX1])[OX2H1]", # R-C(=Te)-OH parallel; AUDIT-FRN)
    "ditelluroic_acid": "[CX3](=[TeX1])[TeX2H1]",# R-C(=Te)-TeH parallel; AUDIT-FRN)
    # Carbamic acid (IUPAC: N-C(=O)-OH (free acid, not ester)
    # Wave2 T3d: the amino N must NOT itself bear a second N — else N-N-C(=O)-OH
    # (a hydrazinecarboxylic acid, is mis-read as carbamic acid and the
    # senior carbamic path DROPS the terminal N (wrong structure). The negated
    # recursive keeps ordinary carbamic acids (NC(=O)O / CNC(=O)O — N has no N-neighbour).
    # W3-P05: the amino N must be a NEUTRAL amine N with NO oxygen
    # neighbour — else the nitro N of O2N-C(=O)-OH (nitroformic acid) false-matches
    # and the senior carbamic path emits a WRONG 'carbamic acid' (structure loss:
    # the nitro group is dropped). ``+0`` excludes the cationic nitro [N+];
    # ``!$([NX3]~[OX1])`` excludes any N bearing a terminal =O/[O-] (nitro, N-oxide).
    # Both constraints are inside atom-1's brackets, so the 4-atom match tuple is
    # unchanged. Nitroformic acid is named by its own exact-SMILES @40 key.
    "carbamic_acid": "[NX3;+0;!$([NX3][NX3]);!$([NX3]~[OX1])][CX3](=O)[OX2H1]",  # R2N-C(=O)-OH -> carbamic acid
    # functional-group perception fix/ (169.7): require a C neighbour on S/P: sulfonic/phosphonic
    # are CARBON acids). Recursive-env `$(...)` adds the constraint WITHOUT changing the
    # match-tuple arity, so inorganic oxoacids (sulfamic NS(=O)(=O)O, phosphoric OP(=O)(O)O)
    # stop false-matching while C-attached acids still match. Free inorganic oxoacids are
    # perceived by their own keys below and named as functional parents /.
    "sulfonic_acid": "[SX4;$([SX4][#6])](=O)(=O)[OX2H1]",
    # W3-P04 / /: FRN-modified sulfur-oxo-acid
    # suffix acids. Each is a C-attached ([#6] guard) sulfonic/sulfinic acid whose
    # -OH oxygen or =O is functionally replaced, so the generic sulfonic_acid/
    # sulfinic_acid SMARTS above CANNOT match (their [OX2H1] must sit DIRECTLY on S).
    # Checked (and suppressed over the generics) in the disambiguation resolver below.
    # -SO2-OOH sulfonoperoxoic acid; BB). Match tuple
    # (S, =O, =O, inner-O, hydroperoxy-OH). The inner O is on S (not C), so the
    # hydroperoxide SMARTS [OX2H][OX2][#6] cannot claim the -OOH either.
    "sulfonoperoxoic_acid": "[SX4;$([SX4][#6])](=O)(=O)[OX2][OX2H1]",
    # -SO2-SH sulfonothioic S-acid; BB, preselected suffix).
    # The -OH of sulfonic acid is replaced by -SH. Match tuple (S, =O, =O, S-H).
    # The thiol SMARTS [SX2H][#6] cannot claim the -SH (its S is on S, not C).
    "sulfonothioic_S_acid": "[SX4;$([SX4][#6])](=O)(=O)[SX2H1]",
    # -S(=NH)-OH sulfinimidic acid; BB, =NH replacement of the
    # sulfinic-acid =O). SX3 (sulfinic-derived). Match tuple (S, =NH, -OH). The
    # required [OX2H1] keeps this from stealing sulfinimidamide -S(=NH)-NH2 (the
    # third neighbour is N, not OH there).
    "sulfinimidic_acid": "[SX3;$([SX3][#6])](=[NX2])[OX2H1]",
    # -S(=O)(=NH)-OH sulfonimidic acid; BB, one =O of the
    # sulfonic acid replaced by =NH). SX4 (sulfonic-derived). Match tuple
    # (S, =O, =NH, -OH). Required [OX2H1] keeps it from stealing sulfonimidamide
    # -S(=O)(=NH)-NH2. The N may itself bear an -OH (=N-OH): that is the
    # N-hydroxy hydroximic derivative, handled downstream.
    "sulfonimidic_acid": "[SX4;$([SX4][#6])](=O)(=[NX2])[OX2H1]",
    "sulfinic_acid": "[SX3;$([SX3][#6])](=O)[OX2H1]",
    "sulfenic_acid": "[SX2]([OX2H])[#6]",  #: IUPAC R-S-OH
    # a phase / Table 6.2): selenium & tellurium analogues of the
    # sulfonic/sulfinic suffix acids, C-attached (parallel [#6] guard) so the
    # free chalcogen oxoacids are not mis-claimed -> ethaneselenonic acid etc.
    "selenonic_acid": "[SeX4;$([SeX4][#6])](=O)(=O)[OX2H1]",  # R-Se(=O)2-OH
    "seleninic_acid": "[SeX3;$([SeX3][#6])](=O)[OX2H1]",      # R-Se(=O)-OH
    "telluronic_acid": "[TeX4;$([TeX4][#6])](=O)(=O)[OX2H1]", # R-Te(=O)2-OH
    "tellurinic_acid": "[TeX3;$([TeX3][#6])](=O)[OX2H1]",     # R-Te(=O)-OH
    "phosphonic_acid": "[PX4;$([PX4][#6])](=O)([OX2H1])[OX2H1]",
    # Phosphinic acid: R2P(=O)(OH) - two C attached to P
    "phosphinic_acid": "[PX4](=O)([OX2H1])([#6])[#6]",
    # --- organo-oxoacids of the heavier pnictogens (As, Sb) -------------
    # Exact analogues of the two phosphorus patterns above. BB L36051-36054
    # gives all four as PRESELECTED names:
    # AsH(O)(OH)2 arsonic acid AsH2(O)OH arsinic acid
    # SbH(O)(OH)2 stibonic acid SbH2(O)OH stibinic acid
    # These are NOT organometallics -- never applies to them. The
    # `$([...][#6])` carbon guard mirrors phosphonic_acid and is what keeps the
    # FREE inorganic oxoacids (arsoric acid As(O)(OH)3 / stiboric acid, which
    # are separate preselected names) out of the organo-acid class.
    # Bismuth has NO oxoacid analogue in the BB, so the family stops at Sb.
    "arsonic_acid": "[AsX4;$([AsX4][#6])](=O)([OX2H1])[OX2H1]",
    "arsinic_acid": "[AsX4](=O)([OX2H1])([#6])[#6]",
    "stibonic_acid": "[SbX4;$([SbX4][#6])](=O)([OX2H1])[OX2H1]",
    "stibinic_acid": "[SbX4](=O)([OX2H1])([#6])[#6]",
    # --- trivalent -ous organo-oxoacids of P, As, Sb (a phase) --
    # The trivalent (X3, NO =O) analogues of the -onic/-inic block above. BB gives
    # the free parents as PRESELECTED (BB L35499-35507, L35413-35443):
    # HP(OH)2 phosphonous acid H2P(OH) phosphinous acid
    # HAs(OH)2 arsonous acid H2As(OH) arsinous acid
    # HSb(OH)2 stibonous acid H2Sb(OH) stibinous acid
    # and the substituent-prefix PINs verbatim: (C6H5)2As(OH) diphenylarsinous
    # acid (BB L35465), C6H5-Sb(OH)2 phenylstibonous acid (BB L35467),
    # (naphthalene-2,6-diyl)bis(phosphonous acid) (BB L35472). The `$([...][#6])`
    # carbon guard on -onous (and the two literal [#6] on -inous) is what keeps the
    # FREE inorganic -ous oxoacids -- phosphorous P(OH)3, arsorous As(OH)3, stiborous
    # Sb(OH)3 (separate preselected names, tabled in inorganic_acids.py) -- out of
    # the organo class, exactly as the -onic/-inic guard does. Trivalent (no =O)
    # distinguishes -ous from -ic; a P=N/P=S form is a different FG and never X3.
    "phosphonous_acid": "[PX3;$([PX3][#6])]([OX2H1])[OX2H1]",   # R-P(OH)2
    "phosphinous_acid": "[PX3]([OX2H1])([#6])[#6]",             # R2P-OH
    "arsonous_acid": "[AsX3;$([AsX3][#6])]([OX2H1])[OX2H1]",    # R-As(OH)2
    "arsinous_acid": "[AsX3]([OX2H1])([#6])[#6]",               # R2As-OH
    "stibonous_acid": "[SbX3;$([SbX3][#6])]([OX2H1])[OX2H1]",   # R-Sb(OH)2
    "stibinous_acid": "[SbX3]([OX2H1])([#6])[#6]",              # R2Sb-OH
    # === FREE INORGANIC OXOACIDS (functional-group perception fix/, 169.7: functional parents) ===
    # Perceived so they are NOT silently dropped or mis-cast as carbon acids (the old
    # `OP(=O)(O)O → trihydrophosphate` malformed bug; now blocked by the [#6]-tightened
    # phosphonic_acid above). NO carbon on the central atom (that is what distinguishes
    # them from the carbon acids). NAMING is functional-parent (downstream); here
    # the goal is perception completeness + correct mis-cast prevention. The resolver
    # below suppresses the carbon-acid generics on their atoms for robustness.
    "phosphoric_acid": "[OX2H1][PX4](=O)([OX2H1])[OX2H1]",   # HO-P(=O)(OH)OH
    "sulfuric_acid": "[OX2H1][SX4](=O)(=O)[OX2H1]",          # HO-S(=O)2-OH
    "nitric_acid": "[OX2H1][NX3+](=O)[O-]",                  # HO-N(+)(=O)O-
    "carbonic_acid": "[OX2H1][CX3](=O)[OX2H1]",              # HO-C(=O)-OH

    # === ACID DERIVATIVES ===
    "anhydride": "[CX3](=O)[OX2][CX3](=O)",
    "ester": "[CX3](=O)[OX2][#6]",
    # a phase Tier FRN-D: iminoester / imidate -- additive per internal notes
    # AUDIT DECISION (AUDIT-FRN): [NX2H1] only (=NH form); N-substituted iminoesters
    # (R-C(=NR')-O-R'') deferred to a phase per Open Question 4. Free imidic acid form
    # (R-C(=NH)-OH) deferred per internal notes. Cyclic imidates deferred per RESEARCH
    "iminoester": "[CX3](=[NX2H1])[OX2][#6]",     # R-C(=NH)-O-R'; "alkyl alkanimidate")
    # W2F-P6: N-substituted carbamimidate ester
    # R''2N-C(=NR')-O-R -> "R N'-R'-N,N-R''2-carbamimidate". A DEDICATED pattern
    # (imino N-substitution ALLOWED, plus a REQUIRED second amino N) so the
    # restricted [NX2H1] iminoester pattern above stays untouched (its
    # AUDIT-FRN 2.4 =NH guard holds). Atom order: (central_C, imino_N, amino_N,
    # ester_O, alkyl_C).
    "carbamimidate": "[CX3](=[NX2])([NX3])[OX2][#6]",  # R''2N-C(=NR')-O-R
    # Imidic acid R-C(=NH)-OH / Table 4.3 'imidic acid'). W3-P02-1
    #: the imidic C may bear an H (methanimidic HC(=NH)-OH, PIN
    # 'methanimidic acid'), so the third neighbour is NO LONGER required to be
    # [#6] — it may be H or C. The `!$([CX3]([#7,#8])...)` guard still keeps
    # N=C(O)O / N=C(N)O routing to the exact-SMILES carbonimidic/carbamimidic
    # inorganic-acid entries (third neighbour = O/N -> excluded). Match tuple
    # stays (C, imino-N, hydroxyl-O).
    "imidic_acid": "[CX3;$([CX3](=[NX2H1])[OX2H1]);!$([CX3]([#7,#8])(=[NX2H1])[OX2H1])](=[NX2H1])[OX2H1]",
    # Hydrazonic acid R-C(=N-NH2)-OH / Table 4.3 'hydrazonic acid'),
    # the =O -> =N-NH2 replacement analogue of imidic acid. W3-P02-3. The
    # geminal C-OH (sp2 C) is NOT perceived by the [OX2H][CX4] alcohol pattern,
    # so without this FG the group mis-reads as a plain hydrazone and the -OH is
    # dropped. Match tuple (C, imino-N, amino-N, hydroxyl-O). The terminal
    # amino N is [NX3H2] (unsubstituted hydrazono); the
    # `!$([CX3]([#7,#8])...)` guard keeps the third C neighbour H or carbon
    # (excludes an extra N/O, e.g. hydrazono-carbonic/amidrazone hybrids).
    "hydrazonic_acid": "[CX3;!$([CX3]([#7,#8])(=[NX2][NX3])[OX2H1])](=[NX2][NX3H2])[OX2H1]",
    # Hydroximic acid R-C(=N-OH)-OH, the =O -> =N-OH replacement
    # analogue. W3-P02-5. Per the PIN is the N-hydroxy derivative
    # of the corresponding imidic acid (N-hydroxyethanimidic acid), NOT the
    # general-only '-hydroximic acid' suffix; the dedicated handler builds that.
    # The geminal C-OH (sp2 C) is not seen by the alcohol pattern, so without
    # this FG the group mis-reads as a plain oxime and the -OH is dropped. Match
    # tuple (C, imino-N, O-on-N, hydroxyl-O). The `!$([CX3]([#7,#8])...)` guard
    # keeps the third C neighbour H or carbon (excludes amidoxime CC(=NO)N and
    # any hydroxyimino-carbonic hybrid).
    "hydroximic_acid": "[CX3;!$([CX3]([#7,#8])(=[NX2][OX2H1])[OX2H1])](=[NX2][OX2H1])[OX2H1]",
    "thioester": "[CX3](=O)[SX2][#6]",
    # a phase Tier FRN-E: chalcogen-ester ester extension) -- additive per internal notes
    "selenoester": "[CX3](=O)[SeX2][#6]",         # R-C(=O)-Se-R'; Se-alkyl alkaneselenoate)
    "telluroester": "[CX3](=O)[TeX2][#6]",        # R-C(=O)-Te-R'; Te-alkyl alkanetelluroate)
    # W3-P07 /: PSEUDOESTER — a carboxylic acid whose
    # ester oxygen carries a Group-13/14/15 organyl (Si/Ge/Sn/Pb/B/Al/Ga/In/Tl/
    # P/As/Sb/Bi) instead of carbon (CH3-CO-O-Si(CH3)3 -> 'trimethylsilyl
    # acetate'). The generic 'ester' pattern requires [#6] on the ester O and
    # never matches these, so without this FG the molecule reads as 'unknown'.
    # Match tuple: (carbonyl_c, carbonyl_o, ester_o, Z). Named by the shared
    # non-carbon-ester handler (build the neutral carboxylic acid -> '-ate' +
    # organyl-Z substituent). Ranked at the ester tier.
    "pseudoester": "[CX3](=O)[OX2][Si,Ge,Sn,Pb,B,Al,Ga,In,Tl,P,As,Sb,Bi]",
    # W3-P07: SULFONIC / SULFINIC esters R-SO2-O-R' / R-S(=O)-O-R'
    # ('methyl methanesulfonate'). No carbonyl -> the 'ester' pattern never
    # matches; without these the molecule reads 'unknown'. The recursive
    # $([S..][#6]) requires a CARBON on S (C-S bond of a genuine sulfonic/sulfinic
    # acid) so sulfate/sulfite esters (C-O-S-O-C, no C-S) are excluded. Match
    # tuples: sulfonate (S, =O, =O, ester_o, alkyl_c); sulfinate (S, =O, ester_o,
    # alkyl_c). Named by the shared non-carbon-ester handler.
    "sulfonic_ester": "[SX4;$([SX4][#6])](=O)(=O)[OX2][#6]",
    "sulfinic_ester": "[SX3;$([SX3][#6])](=O)[OX2][#6]",
    "acid_chloride": "[CX3](=O)[Cl]",
    "acid_bromide": "[CX3](=O)[Br]",
    "acid_fluoride": "[CX3](=O)[F]",
    "acid_iodide": "[CX3](=O)[I]",  #: IUPAC
    # W3-P11 / /: acid halide of a sulfonic /
    # sulfinic acid — R-SO2-X / R-S(=O)-X (X = F,Cl,Br,I). Named by the two-word
    # functional-class grammar '{stem}sulfonyl {halide}' / '{stem}sulfinyl
    # {halide}' (BB 'ethanesulfonyl chloride', 'propane-1-sulfonyl
    # chloride', '4-isocyanatobenzene-1-sulfonyl chloride (PIN)'). The
    # $([SX4/SX3][#6]) guard requires a CARBON on S (a genuine C-sulfonic/sulfinic
    # acid halide), so sulfuryl/thionyl halides (no C-S) are excluded. Neither the
    # sulfone (2 C on S) nor the halogen 'chloro' ([ClX1][#6], halide on C) SMARTS
    # can claim these atoms, so R-SO2-X was previously perceived as nothing ->
    # 'unknown'. Match tuples: sulfonyl (S, =O, =O, X); sulfinyl (S, =O, X).
    "sulfonyl_halide": "[SX4;$([SX4][#6])](=[OX1])(=[OX1])[F,Cl,Br,I]",
    "sulfinyl_halide": "[SX3;$([SX3][#6])](=[OX1])[F,Cl,Br,I]",
    # -3, the Blue Book 'CH3-SO2-CN methanesulfonyl cyanide
    # (PIN)'): the cyanide of a sulfonic acid, named by functional class because
    # a nitrile cannot be expressed substitutively over a sulfonyl centre. The
    # $([SX4][#6]) guard requires a C-S (a genuine C-sulfonic acid), matching the
    # sulfonyl_halide pattern; the -C#N replaces the -OH. Match tuple
    # (S, =O, =O, cyanide-C, N).
    "sulfonyl_cyanide": "[SX4;$([SX4][#6])](=[OX1])(=[OX1])[CX2]#[NX1]",
    # a phase: acyl halides of the CHALCOGEN / IMIDO analogues of
    # carboxylic acid — R-C(=NH)-X (imidoyl), R-C(=S)-X (carbothioyl), R-C(=Se)-X
    # (carboselenoyl). Named by the two-word functional class '{parent}carbo{imidoyl
    # |thioyl|selenoyl} {halide}' for a ring parent, or '{chain-stem}{imidoyl|thioyl}
    # {halide}' for a chain (BB 31438 'cyclohexanecarboximidoyl chloride (PIN)',
    # 31442 'cyclohexanecarbothioyl chloride (PIN)'). The $([CX3][#6]) carbon guard
    # keeps this to a genuine C-acyl halide (excludes carbonimidoyl/carbamimidoyl
    # halides, whose acyl C bears N/O, and the 1-carbon methanimidoyl edge). The
    # broad 'imine' / 'thioketone' / halo SMARTS otherwise claimed only the fragments
    # and left the acyl unnamed -> abstain. Match tuple (C, =X, halide).
    "imidoyl_halide": "[CX3;$([CX3][#6])](=[NX2])[F,Cl,Br,I]",
    "carbothioyl_halide": "[CX3;$([CX3][#6])](=[SX1])[F,Cl,Br,I]",
    "carboselenoyl_halide": "[CX3;$([CX3][#6])](=[SeX1])[F,Cl,Br,I]",
    # Wave2 T6c: acyl PSEUDOhalides — functional-class PINs
    # 'butanoyl azide' / 'propanoyl cyanide' / 'acetyl isocyanate'). Without
    # these the bare azido/isocyanate/nitrile SMARTS claim only the tail and
    # the acyl C=O is DROPPED (1-azidobutane / isocyanatoethane — wrong
    # constitution). class-10 pseudohalogen seniority: N3 > CN > NCO.
    "acyl_azide": "[CX3](=[OX1])[NX2]=[NX2,NX3+]=[NX1-]",
    "acyl_cyanide": "[CX3;!$([CX3][OX2]);!$([CX3][NX3])](=[OX1])[CX2]#[NX1]",
    "acyl_isocyanate": "[CX3](=[OX1])[NX2]=[CX2]=[OX1]",

    # === NITROGEN ACID DERIVATIVES ===
    "primary_amide": "[CX3](=O)[NX3H2]",
    "secondary_amide": "[CX3](=O)[NX3H1][#6]",
    "tertiary_amide": "[CX3](=O)[NX3]([#6])[#6]",
    "hydrazide": "[CX3](=O)[NX3][NX3]",
    # Wave2: thiohydrazide R-C(=S)-NH-NH2 (chalcogen analogue of
    # hydrazide). Cascade-suppresses thioamide + hydrazine_fg on its atoms.
    "thiohydrazide": "[CX3](=S)[NX3][NX3]",
    "imide": "[CX3](=O)[NX3][CX3](=O)",
    # a phase Tier FRN-B: chalcogen-on-amide + -- additive per internal notes
    # AUDIT DECISION (AUDIT-FRN): single-permissive [NX3] (NOT 3-way primary/secondary/tertiary
    # split) captures all 3 N-substitution levels per Open Question 2 + RESEARCH line 265.
    # CR-fix (post-merge regression closure): require explicit C neighbor on the chalcogen-carbonyl
    # carbon so thiocarbamates (R-O-C(=S)-N-, R-S-C(=S)-N-) and chalcogen-ureas (N-C(=S)-N) route
    # through their own pathways rather than over-matching as thio-/seleno-/telluro-amides.
    "thioamide": "[CX3;$([CX3]([#6])(=S)[NX3])](=S)[NX3]",     # R-C(=S)-N(H,R), R=C only
    "selenoamide": "[CX3;$([CX3]([#6])(=[SeX1])[NX3])](=[SeX1])[NX3]",  # R-C(=Se)-N, R=C only
    "telluroamide": "[CX3;$([CX3]([#6])(=[TeX1])[NX3])](=[TeX1])[NX3]", # R-C(=Te)-N, R=C only parallel)

    # === SULFONAMIDES ===
    "primary_sulfonamide": "[SX4](=O)(=O)[NX3H2]",
    "secondary_sulfonamide": "[SX4](=O)(=O)[NX3H1][#6]",
    "tertiary_sulfonamide": "[SX4](=O)(=O)[NX3]([#6])[#6]",
    # C1 fix: R-SO2-NH-NH2 is a sulfonohydrazide, the N-analogue of
    # sulfonic acid. The carbonyl-based hydrazide SMARTS above never matches a
    # sulfonyl S; mirror primary_sulfonamide but with the second N.
    "sulfonohydrazide": "[SX4](=O)(=O)[NX3][NX3]",
    # Wave2 item 20): sulfonimidamide -S(=O)(=NH)-NH2,
    # a preselected suffix ranked just below sulfonamide (BB 18773). The whole
    # group used to drop (CS(=O)(=N)N -> 'unknown'). SMARTS mirrors
    # primary_sulfonamide with one =O replaced by the imido =NH. On sulfur, so
    # NO overlap with the carbon-based imine/oxime/hydrazone SMARTS (the
    # composite-N carbon families that need the perception-priority prototype).
    "sulfonimidamide": "[SX4](=[OX1])(=[NX2])[NX3H2]",
    # Task Y, Table 6.1 item 24): sulfinamide -SO-NH2, a
    # PRESELECTED suffix -- "Sulfonamides, sulfinamides, and the analogous
    # selenium and tellurium amides are named substitutively using the following
    # suffixes:... -SO-NH2 sulfinamide (preselected suffix)" (/),
    # with the worked PIN `butane-2-sulfinamide` . The whole class used
    # to be unnameable (CS(=O)N -> 'unknown organic compound').
    #
    # Mirrors the primary/secondary/tertiary split of sulfonamide above, and the
    # split is LOAD-BEARING, not cosmetic: a single permissive [NX3] bucket would
    # send CS(=O)NC down the generic suffix path and silently DROP the N-methyl
    # carbon -- exactly the defect recorded in rules/sulfonamides.py. With the
    # split, secondary/tertiary have no producer yet and so fail CLOSED instead.
    #
    # The $([SX3][#6]) carbon guard is REQUIRED (parallels sulfinic_acid /
    # sulfinyl_halide): without it the pattern claims H2N-S(=O)-OH, and
    # states "the name sulfinamidic acid is not an approved name". It also keeps
    # out the S-free sulfurous diamide O=S(N)N.
    #
    # Measured disjoint (20-molecule RDKit matrix, zero shared matches) from
    # sulfinic_acid, sulfinic_ester, sulfinyl_halide, sulfinimidamide, sulfoxide,
    # primary/secondary/tertiary_amine, primary_sulfonamide,
    # sulfinohydrazonohydrazide and sulfenic_acid -> no suppression rows needed.
    # The split also excludes CS(=O)NN (sulfinohydrazide, N is H1 with no C) and
    # CS(=O)NO (N-hydroxy), so neither is misnamed.
    "primary_sulfinamide": "[SX3;$([SX3][#6])](=[OX1])[NX3H2]",
    "secondary_sulfinamide": "[SX3;$([SX3][#6])](=[OX1])[NX3H1][#6]",
    "tertiary_sulfinamide": "[SX3;$([SX3][#6])](=[OX1])[NX3]([#6])[#6]",
    # Wave2 item 25): sulfinimidamide -S(=NH)-NH2 = the
    # shipped sulfonimidamide with the S(=O) removed (SX4->SX3, one fewer O).
    # SX3-vs-SX4 makes it disjoint from every shipped SMARTS -> no suppression.
    "sulfinimidamide": "[SX3](=[NX2])[NX3H2]",
    # Wave-2 P1AM, BB 34173): Se analogues named "similarly" --
    # -seleninimidamide (preselected) / -selenonimidamide. Te analogues are
    # DEFERRED until an OPSIN oracle exists for the -telluron- form
    # (methanetellurinimidamide parses; the selenon/telluron =O forms were
    # only OPSIN-verified for Se during planning). Fail-closed by absence.
    "seleninimidamide": "[SeX3](=[NX2])[NX3H2]",
    "selenonimidamide": "[SeX4](=[OX1])(=[NX2])[NX3H2]",
    #: -S(=N-NH2)-NH-NH2 (cannot collide with sulfinimidamide --
    # the S-NH- nitrogen here is H1, not the [NX3H2] amide N).
    "sulfinohydrazonohydrazide": "[SX3](=[NX2][NX3])[NX3][NX3]",

    # === CARBAMATES (must check before esters -- N-C(=O)-O is more specific) ===
    "carbamate": "[NX3][CX3](=O)[OX2][#6]",

    # === UREA (must check before amides -- N-C(=O)-N is more specific) ===
    "urea": "[NX3][CX3](=O)[NX3]",

    # === THIOUREA (Wave-2 completion,: thio-analogue of urea,
    # N-C(=S)-N. Prefix form 'carbamothioylamino' (mirror of urea's
    # carbamoylamino). More specific than thioamide (which requires a C
    # neighbour) and primary_amine — suppresses those on its atoms below.
    #
    # R3: broadened from [SX1] to the whole chalcogen set.
    # (" Chalcogen analogues of urea and isourea", the Blue Book
    #:33439): "Chalcogen analogues of urea are named by functional replacement
    # nomenclature using the prefixes 'thio', 'seleno', and 'telluro'." The
    # section's own worked example is a Se compound — ':33451'
    # "N-(butan-2-yl)selenourea (PIN)". Before the broadening the Se/Te
    # analogues were not perceived at all, so the collision resolver below never
    # suppressed the terminal-amine read and NC(=[Se])N was named 'methanamine'
    # — a SILENT CHALCOGEN DROP (gate-suppressed in production, but fabricated).
    # The FG name stays 'thiourea'; consumers select the stem from the actual
    # chalcogen (see composer._try_name_thiourea) and the PREFIX path fails
    # closed for Se/Te, whose prefixes does not enumerate.
    "thiourea": "[NX3][CX3](=[S,Se,Te;X1])[NX3]",

    # === GUANIDINE (must check before imines -- N-C(=N)-N is more specific) ===
    "guanidine": "[NX3][CX3](=[NX2])[NX3]",

    # === CYANAMIDE (,: H2N-C#N and its N-substituted
    # derivatives use 'cyanamide' as the retained parent ((propan-2-yl)cyanamide,
    # dimethylcyanamide). ACYCLIC amine N only ([NX3;!R]) — a ring N-C#N stays
    # 'piperidine-1-carbonitrile'. Collision resolver below suppresses the
    # nitrile + amine reads on these atoms so principal_group becomes None. ===
    "cyanamide": "[NX3;!R][CX2]#[NX1]",

    # === AMIDINE (IUPAC: C(=N)N, less specific than guanidine) ===
    # D1: broadened from [CX3](=[NX2H])[NX3H2] (both N unsubstituted) to
    # [CX3](=[NX2])[NX3] so N-/N'-substituted amidines are classified too. This
    # matches the benzene handler's proven SMARTS (benzene.py). The broadened
    # pattern also matches the guanidine central C; the atom-overlap
    # guanidine-suppresses-amidine rule below removes those matches. Match-tuple
    # order stays (amidine_C, imino_N, amino_N) -> chains.py _TERMINAL_C_FGS
    # ['amidine']=0 remains correct. =[NX2] (not =O) still excludes urea; the N
    # (not O) neighbour still excludes imidates (COC(=N)C).
    # W3-P15: the amino-N guard ``!$([NX3]~[OX1])`` excludes an N
    # bearing a terminal =O/[O-] (a nitro / nitroso / N-oxide nitrogen). Without
    # it a nitrolic acid R-C(=N-OH)-NO2 false-matched amidine (its nitro N is an
    # [NX3]), suppressing the real oxime and abstaining. Real amidine amino-N's
    # never carry an oxygen, so no amidine regresses. Match-tuple arity unchanged
    # (C, imino_N, amino_N).
    "amidine": "[CX3](=[NX2])[NX3;!$([NX3]~[OX1])]",  # / D1 / W3-P15
    # Wave2, BB Table 6.1 items 17-18): amidrazone / hydrazonamide
    # R-C(=N-NH2)-NH2 — an amidine whose imido =NH is a hydrazono =N-NH2. The
    # =[NX2][NX2,NX3] (imino N bonded to another N) is the discriminant vs plain
    # amidine (=NH, no N-neighbour). Ranks just below amidine; cascade-suppresses
    # amidine/hydrazone/primary_amine on its atoms.
    "hydrazonamide": "[CX3](=[NX2][NX2,NX3])[NX3]",
    # Wave2, BB 34574): hydrazidine / hydrazonohydrazide
    # R-C(=N-NH2)-NH-NH2 — the [NX3][NX3] hydrazido side is the discriminant vs
    # amidrazone's single [NX3]. Cascade-suppresses hydrazone/amidine/hydrazonamide/
    # hydrazine_fg on its atoms.
    "hydrazidine": "[CX3](=[NX2][NX3])[NX3][NX3]",
    # Wave-2 P1AM Task 7, BB 34430): the R-C(=NH)-NH-NH2
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
    # W4-I4 /: imine (nitrone) N-oxide — an sp2 N+ double-
    # bonded to C and single-bonded to a terminal O- (named '<imine> N-oxide').
    "n_oxide_imine": "[NX2,NX3;+](=[#6])[OX1-]",

    # === BORONIC ACIDS ===
    "boronic_acid": "[#6][BX3]([OX2H])([OX2H])",

    # === NITRILES ===
    "nitrile": "[CX2]#[NX1]",
    "isocyanide": "[#6][NX2]#[CX1]",

    # === CARBONYLS ===
    # Aldehyde: carbonyl C with 1 or 2 H (: H2 added for formaldehyde;
    # collision resolution suppresses false positives on amides/acids)
    "aldehyde": "[CX3;H1,H2](=O)",
    "ketone": "[#6][CX3](=O)[#6]",
    "thioaldehyde": "[CX3H1](=S)",
    "thioketone": "[#6][CX3](=S)[#6]",
    # a phase Tier FRN-C: chalcogen-on-aldehyde/ketone -- additive per internal notes
    "selenoaldehyde": "[CX3H1](=[SeX1])",         # R-C(=Se)H
    "telluroaldehyde": "[CX3H1](=[TeX1])",        # R-C(=Te)H parallel)
    "selenoketone": "[#6][CX3](=[SeX1])[#6]",     # R-C(=Se)-R'; selone PIN suffix form
    "telluroketone": "[#6][CX3](=[TeX1])[#6]",    # R-C(=Te)-R' parallel); tellone PIN suffix form

    # === ALCOHOLS AND ANALOGS ===
    # Generic catch-all: any OH on sp3 carbon (IUPAC
    # Complements specific sub-type patterns below; ensures detection of
    # OH on carbons with non-carbon neighbors (halogens, nitrogen, sulfur)
    "alcohol": "[OX2H][CX4]",  #: generic catch-all per
    "primary_alcohol": "[OX2H][CX4H2]",
    "secondary_alcohol": "[OX2H][CX4H1]([#6])[#6]",
    "tertiary_alcohol": "[OX2H][CX4]([#6])([#6])[#6]",
    "phenol": "[OX2H][cX3]",
    "enol": "[OX2H][CX3]=[CX3]",
    "thiol": "[SX2H][#6]",
    # Wave2 T3a: carbon in match (parallel to thiol/tellurol) so
    # PG_ATTACHMENT_INDICES["selenol"] = [1] can point at the locant-bearing C.
    "selenol": "[SeX2H][#6]",
    # functional-group perception fix/ (169.7): tellurol — Te analogue of -ol/-thiol/-selenol.
    # Defines the resolver ref at line ~298 (was a dead ref, audit Dim-02 #2).
    "tellurol": "[TeX2H1][#6]",

    # === HYDROPEROXIDES ===
    "hydroperoxide": "[OX2H][OX2][#6]",
    "peroxide": "[#6][OX2][OX2][#6]",
    # === CHALCOGEN HYDROPEROXOL ANALOGUES (DD2 Fix C, Phase D: / (3)) ===
    # The -OOH (peroxol) sulfur/mixed analogues. Each is BOTH a suffix-capable
    # principal group (-SO-/-OS-/dithioperoxol) and, when demoted, a substituent
    # prefix (hydroxysulfanyl / sulfanyloxy / disulfanyl). The -S-OH key REPLACES
    # the Blue-Book-retired `sulfenic_acid` PIN verbatim, the Blue Book
    # line ~24326: `CH3-S-OH -> methane-SO-thioperoxol (PIN) (not methanesulfenic
    # acid)`); the collision resolver below suppresses `sulfenic_acid` on its atoms.
    # W3-P03-4: the leading [#6] must be a NON-acyl carbon. An acyl
    # carbon (CH3-CO-S-OH / CH3-CO-O-SH) is a (thioperoxoic) SO-/OS-acid, NOT an
    # alcohol-class thioperoxol; matching it here emitted a WRONG name
    # ('ethane-1-OS-thioperoxol' for CC(=O)OS). Excluding acyl/thioacyl carbons
    # fails those closed (-> 'unknown') until the peroxoic-acid FRN suffix path
    # gains S/Se-infix + italic OS/SO letter-locants (OPSIN has no oracle for that
    # word-form). Genuine sp3 R-O-SH / R-S-OH (CCOS / CCSO) still match.
    "so_thioperoxol": "[#6;!$([CX3]=[OX1]);!$([CX3]=[SX1])][SX2][OX2H]",   # R-S-OH -> -SO-thioperoxol (was retired sulfenic acid)
    "os_thioperoxol": "[#6;!$([CX3]=[OX1]);!$([CX3]=[SX1])][OX2][SX2H]",   # R-O-SH -> -OS-thioperoxol
    "dithioperoxol": "[#6][SX2][SX2H]",    # R-S-SH -> dithioperoxol (suffix) / disulfanyl (prefix)

    # === HYDROXYLAMINES (functional-group perception fix/, 169.7: class 21) ===
    # R-NH-OH / R2N-OH — the N bears an -OH and >=1 carbon. MUST be checked before
    # amines/alcohol (resolver suppresses those on its atoms). Excludes hydroxamic
    # acid (C(=O)-NH-OH, carbonyl present) and oxime (C=N-OH) via the carbonyl/=*
    # recursive exclusions. The O is on N (not C) so the `[OX2H][CX4]` alcohol never
    # matches it. Specifying only the OH + one C neighbour leaves the 3rd N connection
    # free (implicit H for R-NH-OH, or a 2nd C for R2N-OH) — so it matches BOTH forms.
    "hydroxylamine": "[OX2H1][NX3;!$([NX3][CX3]=[OX1,SX1,SeX1,TeX1]);!$([NX3]=*)][#6]",
    # Wave2 T2b: -O-NH2 on carbon = 'aminooxy' preselected prefix
    #; BB PIN '2-(aminooxy)ethan-1-amine'; no o-elision).
    # NX3H2 pins the terminal unsubstituted N (its only heavy neighbour is the
    # O), so N-substituted forms R'-NH-O-R stay unperceived → fail-closed
    # (their PIN is O-substituted-hydroxylamine territory,.
    "aminooxy": "[NX3H2][OX2][#6]",
    # W3-P11: -O-NO2 nitrooxy preselected prefix (ester of nitric
    # acid, named by concatenating the acyl 'nitro' onto 'oxy'). BB verbatim
    # '-O-NO2 nitrooxy (preselected prefix)' -> '3-(nitrooxy)propanoic acid'.
    # The nitro N+ is bonded to the ester O (not carbon), so the C-attached
    # `nitro` SMARTS [NX3+;$([NX3+][#6])] cannot claim it and there is no overlap
    # with any existing FG (R-O-NO2 was previously perceived as nothing ->
    # 'unknown'). Match tuple (O-, N+, =O, ester-O, C); the C attachment is last,
    # mirroring aminooxy. Only the O/N heteroatoms enter get_chain_excluded_atoms.
    "nitrooxy": "[OX1-][NX3+](=[OX1])[OX2][#6]",
    # Wave2 T2b: -NH-X N-haloamines → compound prefixes fluoroamino/
    # chloroamino/bromoamino/iodoamino; BB verbatim '-NH-Cl
    # chloroamino (preselected prefix)'). NX3H1 with the halogen + one C pins
    # the mono-halo N-H form; N,N-dihalo and N-halo-N-alkyl stay unperceived
    # → fail-closed.
    "n_fluoroamine": "F[NX3H1][#6]",
    "n_chloroamine": "Cl[NX3H1][#6]",
    "n_bromoamine": "Br[NX3H1][#6]",
    "n_iodoamine": "I[NX3H1][#6]",
    # === AMINES ===
    # task 9: the N carries !R — a RING nitrogen is a
    # skeletal heteroatom of a ring parent hydride (morpholine, pyrrolidine,
    # piperidine), never an amine characteristic group. Without it the
    # ring N matched tertiary_amine, became the principal group, and the
    # amine emitter CUT the ring ('N,N-dibutyl-N-cyclohexylcyclohexan-1-
    # amine' for cyclohexylmorpholine). Only the N is !R: the carbon
    # neighbours may be ring atoms (N-cyclohexylamines still match).
    "primary_amine": "[NX3;H2;!R;!$([NX3][CX3]=O);!$([NX3][CX3]=[NX2])][#6]",  #: sp2/sp3, excludes amide/urea/guanidine N
    "secondary_amine": "[NX3;H1;!R;!$([NX3][CX3]=O);!$([NX3][CX3]=[NX2])]([CX4,cX3,$([CX3]=[CX3;!R])])[CX4,cX3,$([CX3]=[CX3;!R])]",  # +: sp3, aromatic, or acyclic vinyl C; exclude amides/guanidines
    "tertiary_amine": "[NX3;H0;!R;!$([NX3][CX3]=O);!$([NX3][CX3]=[NX2])]([CX4,cX3,$([CX3]=[CX3;!R])])([CX4,cX3,$([CX3]=[CX3;!R])])[CX4,cX3,$([CX3]=[CX3;!R])]",  # +: sp3, aromatic, or acyclic vinyl C; exclude amides/guanidines
    "aromatic_amine": "[NX3H2][cX3]",

    # === IMINES ===
    # Wave2: broadened from [CX3]=[NX2H] to admit
    # N-SUBSTITUTED imines R-CH=N-R' (BB VERBATIM 'N-methylethanimine
    # (PIN)'). Guards keep the old blast radius: !R on N — a ring C=N is a
    # skeletal feature of the ring parent (kekulized azirine/dihydroazine
    # rings must NOT gain an imine FG); the $-exclusions keep oxime
    # (C=N-OH), oxime ethers (C=N-OR), hydrazones (C=N-N<) and N-halo
    # imines: named as amides) out — each is its own class.
    "imine": "[CX3]=[NX2;!R;!$([NX2][OX2]);!$([NX2][NX2]);!$([NX2][NX3]);!$([NX2][F,Cl,Br,I])]",
    "oxime": "[CX3]=[NX2][OX2H]",
    "hydrazone": "[CX3]=[NX2][NX3]",
    "hydrazine_fg": "[NX3;H1;!$([NX3][CX3]=O)][NX3H2]",  #: -NH-NH2, excludes hydrazides

    # === ETHERS (no suffix - substitutive naming) ===
    "ether": "[OX2]([CX4])[CX4]",
    "vinyl_ether": "[OX2]([#6])[CX3]=[CX3]",
    "aromatic_ether": "[OX2]([#6])[cX3]",
    "thioether": "[SX2]([#6])[#6]",
    # functional-group perception fix/ (169.7): selenide/telluride — Se/Te ether analogues.
    # Named selenoether/telluroether to match (and make LIVE) the dead resolver refs
    # at lines ~284-285 (audit Dim-02 #2). Prefix form = (alkyl)selanyl/tellanyl.
    "selenoether": "[SeX2]([#6])[#6]",   # R-Se-R'; selanyl prefix
    "telluroether": "[TeX2]([#6])[#6]",  # R-Te-R' parallel); tellanyl prefix
    "disulfide": "[#6][SX2][SX2][#6]",  #:

    # === PHOSPHORUS COMPOUNDS (check more specific first) ===
    # Phosphate esters (C-O-P bonds, not C-P bonds) - check before phosphine oxide
    "phosphate_triester": "[PX4](=O)([OX2][#6])([OX2][#6])[OX2][#6]",
    "phosphate_diester": "[PX4](=O)([OX2][#6])([OX2][#6])[OX2H1]",
    "phosphate_monoester": "[PX4](=O)([OX2][#6])([OX2H1])[OX2H1]",
    # tail #16: phosphorous-acid TRIESTER -- a TRIVALENT P (no P=O)
    # bearing three ester oxygens (distinct from the PX4 phosphates above). The
    # ester O may carry a carbon (triethyl phosphite) or a sulfenyl -S-R
    # (tris(dodecylsulfanyl) phosphite). Trivalent-P phosphites were otherwise
    # UNPERCEIVED (pg=None) -> a garbage skeletal-replacement name / abstention.
    # name_phosphate_ester already carries the phosphite stem ((False,0)); this
    # is the perception that finally reaches it. Each OX2 excludes P-O-P bridges.
    "phosphite_triester": "[PX3]([OX2][#6,#16])([OX2][#6,#16])[OX2][#6,#16]",
    # tail #17: phosphonic-acid DIESTER R-P(=O)(OR')2 -- a P=O with
    # exactly ONE P-C ligand and two ester oxygens (the '$([PX4][#6])' P-C guard
    # separates it from a phosphate triester, which has none). name_phosphate_ester
    # names the P-C ligand as the stem's carbon prefix (diisopropyl
    # (1-cyano-1-isocyanoethyl)phosphonate). Phosphonate diesters were otherwise
    # unperceived; the free-acid 'phosphonic_acid' row (2 OH) does not match.
    "phosphonate_diester": "[PX4;$([PX4][#6])](=O)([OX2][#6])[OX2][#6]",
    # 11-FABLEFIX class 9): the phosphinic/arsinic/stibinic
    # acid ESTER R2E(=O)(OR') -- TWO E-C bonds, one =O, one ester -O-C. This is the
    # exact shape the multiplicative pnictogen guard declines (two identical/aryl
    # C-E bonds), so declining the multiplicative name (1,1'-(methoxyphosphoryl)-
    # dibenzene) hands the molecule to name_pnictogen_inate_ester -> the PIN
    # 'methyl diphenylphosphinate' / 'methyl diphenylarsinate'. As/Sb had no ester
    # perception at all; the two E-C guards separate this from the 1-C phosphonate
    # diester and the 0-C phosphate triester above.
    "phosphinate_ester": "[PX4]([#6])([#6])(=O)[OX2][#6]",
    "arsinate_ester": "[AsX4]([#6])([#6])(=O)[OX2][#6]",
    "stibinate_ester": "[SbX4]([#6])([#6])(=O)[OX2][#6]",
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
    # -6I, the Blue Book): Se/Te analogues of sulfone/sulfoxide.
    # the Blue Book "selenium and tellurium... named in the same way" (class names
    # selenoxide/selenone, telluroxide/tellurone). SeX4/TeX4 = R-Se(=O)2-R'
    # (selenonyl), SeX3/TeX3 = R-Se(=O)-R' (seleninyl). Two carbons are required
    # so the free seleninic/selenonic acids (one C + OH) are not mis-claimed.
    "selenone": "[SeX4](=[OX1])(=[OX1])([#6])[#6]",
    "selenoxide": "[SeX3](=[OX1])([#6])[#6]",
    "tellurone": "[TeX4](=[OX1])(=[OX1])([#6])[#6]",
    "telluroxide": "[TeX3](=[OX1])([#6])[#6]",

    # === UNSATURATION ===
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",

    # === HALOGENS (always prefixes) ===
    "fluoro": "[FX1][#6]",
    "chloro": "[ClX1][#6]",
    "bromo": "[BrX1][#6]",
    "iodo": "[IX1][#6]",

    # === HYDROXAMIC ACIDS (: ===
    "hydroxamic_acid": "[CX3](=O)[NX3;H1][OX2H]",  # R-C(=O)-NH-OH: free hydroxamic acid only

    # === CYANATES / THIOCYANATES (: ===
    "cyanate": "[OX2][CX2]#[NX1]",
    "thiocyanate": "[SX2][CX2]#[NX1]",

    # === AZO (: ===
    "azo": "[#6][NX2]=[NX2][#6]",
    # Wave2 T2b: terminal HN=N- = 'diazenyl' preselected prefix;
    # BB 'diazenyl (preselected prefix; see ', ledger 8-diazenyl-
    # octanoic acid). NX2H1 pins the terminal =N-H (its only heavy neighbour
    # is the other N); internal R-N=N-R' stays 'azo' (C on both ends, above).
    "diazenyl": "[NX2H1]=[NX2][#6]",

    # === OTHER ===
    # functional-group perception fix/ (169.7): C-attached nitro only via recursive-env so the
    # match-tuple arity is unchanged; stops nitrate ESTERS (R-O-NO2, CCO[N+](=O)[O-]) from
    # false-matching as nitro. Aliphatic + aromatic C both satisfy [#6].
    "nitro": "[NX3+;$([NX3+][#6])](=O)[O-]",
    # -05 : [#6] C-attachment guard, uniform with the sibling `nitro`.
    # Without it the N=O of a nitrite ester (R-O-N=O) false-matched `nitroso` and
    # won the collision -> `nitrosoethane` for ethyl nitrite. The guard limits
    # `nitroso` to a genuine C-nitroso (N bonded to carbon,.
    "nitroso": "[NX2;$([NX2][#6])]=[OX1]",
    "azido": "[N;+0]=[N+]=[N-]",  # /C2: organic azide R-N=[N+]=[N-] (attach N is NX2 neutral; the old [NX1]=... never matched RDKit canonical azides)
    "diazo": "[#6]=[NX2+]=[NX1-]",  #: diazo group
    # functional-group perception fix/ (169.7): nitrite ester R-O-N=O. Added so its O,N are
    # excludable from the skeletal-replacement backbone (see get_chain_excluded_atoms).
    "nitrite": "[#6][OX2][NX2]=[OX1]",
}

# Pre-compile all SMARTS patterns once at module load (avoid recompilation per molecule)
_COMPILED_FG_SMARTS = {}
for _fg_name, _smarts in FUNCTIONAL_GROUP_SMARTS.items():
    _pat = Chem.MolFromSmarts(_smarts)
    if _pat is not None:
        _COMPILED_FG_SMARTS[_fg_name] = _pat


# W3-P06 / /: anhydride BRIDGE variants whose bridge
# is NOT the carbon/O-only single O of the base ANHYDRIDE_SMARTS
# ([CX3](=O)[OX2][CX3](=O)). Each match is FOLDED into the 'anhydride'
# functional-group bucket in detect_functional_groups (below) so
# get_principal_group returns 'anhydride' and rules.anhydrides.name_anhydride
# (which re-detects the bridge type on the mol) produces the functional-class PIN.
# Kept SEPARATE from FUNCTIONAL_GROUP_SMARTS so the base-FG count contract and the
# per-FG collision resolver are untouched; the folded tuples are shape-heterogeneous
# but every 'anhydride' consumer (get_anhydride_consumed_atoms, _filter_consumed,
# the resolver) iterates atoms only.
_ANHYDRIDE_BRIDGE_SMARTS = {
    # sulfonic anhydride R-SO2-O-SO2-R'; BB 32247 'benzenesulfonic anhydride')
    "sulfonic_anhydride": "[SX4](=O)(=O)[OX2][SX4](=O)(=O)",
    # chalcogen-bridge anhydride R-CO-X-CO-R', X = S/Se/Te; BB 32292
    # 'benzoic thioanhydride'). The bridge is S/Se/Te, not O, so the base core misses it.
    "chalcogen_anhydride": "[CX3](=O)[SX2,SeX2,TeX2][CX3](=O)",
    # peroxy anhydride R-CO-OO-CO-R'; BB 32344 'acetic peroxyanhydride').
    # The -O-O- bridge cannot match the single-O base core (no single O bonded to
    # both carbonyls), so this NEVER double-fires as a plain anhydride.
    "peroxy_anhydride": "[CX3](=O)[OX2][OX2][CX3](=O)",
    # W8-P4 /: mixed organic/cyanic(-thio) acid anhydride
    # R-CO-O-C#N / R-CO-S-C#N (BB 30979/32279 'CH3-CO-O-CN acetic cyanic
    # anhydride (PIN)'). Neither side is a second carbonyl, so the base
    # ANHYDRIDE_SMARTS never matches this; without folding it into 'anhydride'
    # the molecule perceives as ester+cyanate/thiocyanate and mis-names.
    "cyanic_anhydride": "[CX3](=O)[OX2][CX2]#[NX1]",
    "thiocyanic_anhydride": "[CX3](=O)[SX2][CX2]#[NX1]",
    # D /.2/.3, /.3): thio/seleno-ACYL anhydrides
    # R-C(=X1)-Y-C(=X2)-R' where an acyl chalcogen X1/X2 is S or Se (a
    # '...thioic'/'...selenoic' acid component, BB 32451/32455/32467), with an
    # O bridge ('propanethioic anhydride') or an S/Se/Te bridge
    # ('ethanethioic propanethioic thioanhydride'). The existing bridge SMARTS
    # all require [CX3](=O) acyls, so the C(=S)/C(=Se) case is missed entirely
    # (a trace: principal_group=None, name_anhydride never reached). These broad
    # SMARTS also match the (=O,=O) acyls already covered above, so the fold
    # below DEDUPS them by atom-set against the existing 'anhydride' matches --
    # only the genuinely-new thio/seleno-acyl motifs are added. Named by
    # rules.anhydrides._name_thioacyl_anhydride (component acid affix from the
    # acyl chalcogen; class word from the bridge chalcogen).
    "thioacyl_oxo_anhydride":
        "[CX3](=[OX1,SX1,SeX1])[OX2][CX3](=[OX1,SX1,SeX1])",
    "thioacyl_chalcogen_anhydride":
        "[CX3](=[OX1,SX1,SeX1])[SX2,SeX2,TeX2][CX3](=[OX1,SX1,SeX1])",
}
# D: the two broad thio/seleno-acyl SMARTS overlap the base O-bridge and
# chalcogen-bridge matches, so their fold is DEDUPED by atom-set (below).
_BROAD_ACYL_ANHYDRIDE = {"thioacyl_oxo_anhydride", "thioacyl_chalcogen_anhydride"}
_COMPILED_ANHYDRIDE_BRIDGE = {}
for _bk, _bsmarts in _ANHYDRIDE_BRIDGE_SMARTS.items():
    _bpat = Chem.MolFromSmarts(_bsmarts)
    if _bpat is not None:
        _COMPILED_ANHYDRIDE_BRIDGE[_bk] = _bpat


def _add_resonance_twin_matches(mol, results: Dict[str, List[Tuple[int, ...]]]) -> None:
    """ADD the resonance-shifted twins of ``azido``/``diazo`` that the fixed
    single-bond-order SMARTS above (:568-569) cannot see (a phase a trace,
    `internal notes` Q1/Q2/Q3). Purely
    additive: the existing SMARTS matches are kept untouched, and a chain the
    SMARTS already found is skipped here (atom-set dedup), so canonical-
    drawing detection -- and its downstream naming -- is byte-identical.

    Match-tuple SHAPE mirrors each SMARTS exactly, so every existing consumer
    (attach-point discovery, consumed-atom accounting) needs no changes:
      * ``azido``: 3 N atoms, attach-adjacent first (the SMARTS never
        includes the external attach atom).
      * ``diazo``: attach atom + 2 N atoms (the SMARTS's ``[#6]=...`` DOES
        include the attach carbon).

    Diazonium is intentionally not added here -- see
    ``perception.ions._resonance_twin_internal_atoms`` for why.
    """
    from ..data.resonance_templates import find_resonance_chains
    already = {
        'azido': {frozenset(m) for m in results.get('azido', ())},
        'diazo': {frozenset(m) for m in results.get('diazo', ())},
    }
    for cls, chain in find_resonance_chains(mol):
        if cls == 'azido':
            match: Tuple[int, ...] = tuple(chain[1:])
        elif cls == 'diazo':
            match = tuple(chain)
        else:
            continue
        key = frozenset(match)
        if key in already[cls]:
            continue
        results[cls].append(match)
        already[cls].add(key)


def _detect_functional_groups_impl(mol) -> Dict[str, List[Tuple[int, ...]]]:
    """
    Detect all functional groups in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Dictionary mapping functional group names to lists of atom index tuples.
        Each tuple contains the indices of atoms in one instance of that group.

    Example:
        >>> mol = Chem.MolFromSmiles("CC(=O)O") # acetic acid
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

    _add_resonance_twin_matches(mol, results)

    # A ring-internal C=N-N is the backbone of an azoline/azole ring (e.g.
    # 4,5-dihydro-1H-pyrazole), NOT a hydrazone principal group (a hydrazone is
    # the acyclic R2C=N-NH2). Drop hydrazone matches whose C=N AND N-N bonds are
    # BOTH ring bonds, so the ring namer (not the hydrazone PCG path) handles
    # them. Acyclic hydrazones keep their match; aromatic azoles never match the
    # explicit-'=' SMARTS. ring features vs hydrazone PCG.)
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

    # W3-P06 /.3/.4): fold anhydride bridge-variant matches into the
    # 'anhydride' bucket so principal_group -> 'anhydride' (name_anhydride
    # re-detects the bridge type). Run BEFORE the collision resolver so the
    # ('anhydride', [...]) suppression rule clears the sub-component reads
    # (thioester/peroxide/ketone) on the folded atoms.
    #
    #: a chalcogen-bridge anhydride (-CO-S/Se/Te-CO-) that
    # is SUBORDINATE to a senior principal characteristic group (e.g. a free
    # carboxylic acid — ranked above 'anhydride' in SENIORITY_ORDER) is NOT a
    # 'carboxylic anhydride': the senior acid owns the suffix and the linkage is
    # expressed substitutively (in-chain acyl C -> oxo, acyl-S -> acylsulfanyl,
    #. So SKIP the chalcogen-anhydride fold when a senior-to-anhydride
    # group is present; the underlying thio/seleno/telluro-ester base FG then
    # survives the (atom-scoped) collision resolver and the existing thioester
    # group-split emits e.g. '9-(acetylsulfanyl)-9-oxononanoic acid'. Genuine
    # anhydrides (no senior co-group) fold exactly as before -> byte-identical.
    # Scoped to the chalcogen bridge (its substitutive ester read + oxo/acyl-
    # sulfanyl split are already built); the O/peroxy/cyanic bridges keep folding
    # unconditionally (their subordinate substitutive path is not wired -> they
    # stay fail-closed, never wrong).
    from orthonym.rules.seniority import SENIORITY_ORDER as _SEN_ORDER
    _anh_rank = _SEN_ORDER.index("anhydride")
    _senior_to_anh_present = any(
        fg in _SEN_ORDER and _SEN_ORDER.index(fg) < _anh_rank for fg in results
    )
    # D: the S/Se/Te-bridge THIOACYL variant joins the subordinate skip
    # (like chalcogen_anhydride, its substitutive read is owned by a senior
    # co-group when one is present). The O-bridge thioacyl variant folds
    # unconditionally, matching the base O-bridge (fail-closed, never wrong).
    _SUBSTITUTIVE_BRIDGE = {"chalcogen_anhydride", "thioacyl_chalcogen_anhydride"}
    # D: dedup the BROAD acyl folds by atom-set against everything already
    # in the 'anhydride' bucket (base O-bridge FG + the earlier bridge keys), so
    # a (=O,=O) match is not double-added -- only the new thio/seleno-acyl motifs.
    _seen_anh = {frozenset(m) for m in results["anhydride"]}
    for _bk, _bpat in _COMPILED_ANHYDRIDE_BRIDGE.items():
        if _bk in _SUBSTITUTIVE_BRIDGE and _senior_to_anh_present:
            continue  # subordinate chalcogen-anhydride -> keep substitutive read
        _dedup = _bk in _BROAD_ACYL_ANHYDRIDE
        for _bm in mol.GetSubstructMatches(_bpat, uniquify=True):
            _fs = frozenset(_bm)
            if _dedup and _fs in _seen_anh:
                continue
            results["anhydride"].append(_bm)
            _seen_anh.add(_fs)

    # Post-processing: remove generic FG matches that overlap with more-specific FGs
    results = _resolve_fg_collisions(results)

    # +: a ring-carbon exocyclic =O on a MANCUDE ring
    # (residual unsaturation) is a heterone — the senior principal-characteristic
    # group: "Ketones, pseudoketones and heterones... are senior to
    #... amines, and imines in the seniority order of classes"). The ketone
    # SMARTS "[#6][CX3](=O)[#6]" needs two C neighbours, so a ring c=O adjacent to
    # a ring heteroatom (pyridinone/pyrimidinone/quinolinone/...) is missed and the
    # amine wrongly wins the principal-group slot. Register it here (AFTER the
    # collision resolver, so the resolver never strips it) so get_principal_group
    # ranks it (ketone > amine, seniority.py) and the mancude-oxo namer
    # partial_saturation.name_cyclic_oxo_compound fires with the retained ring name
    # + parenthetical added-H. SCOPED to residual-unsaturation rings so fully
    # saturated lactams/lactones/ketones (piperidin-2-one, oxolan-2-one,
    # barbituric acid) are untouched — they route to their existing saturated
    # handlers.
    _ri = mol.GetRingInfo()
    _ring_atoms = set(a for r in _ri.AtomRings() for a in r)
    if _ring_atoms:
        _het_matches = []
        _carbonyl_cs = set()
        for _a in _ring_atoms:
            _at = mol.GetAtomWithIdx(_a)
            if _at.GetSymbol() != 'C':
                continue
            for _b in _at.GetBonds():
                _o = _b.GetOtherAtom(_at)
                if (_o.GetIdx() not in _ring_atoms
                        and _o.GetSymbol() == 'O'
                        and _b.GetBondTypeAsDouble() == 2.0
                        and _o.GetDegree() == 1):
                    # tuple (carbonylC, carbonylC, O): index [1] is the principal
                    # atom per PG_ATTACHMENT_INDICES['ketone'] == [1] (seniority.py:32).
                    # carbonylC is used at both [0] and [1] to avoid the flanking-atom
                    # locant misread (a heterone C is flanked by ring N, not C).
                    _het_matches.append((_a, _a, _o.GetIdx()))
                    _carbonyl_cs.add(_a)
        if _het_matches:
            # residual-unsaturation gate (mirrors name_cyclic_oxo_compound scope):
            # aromatic ring, or a ring-internal C=C not part of a carbonyl.
            _res = any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in _ring_atoms)
            if not _res:
                for _b in mol.GetBonds():
                    _i, _j = _b.GetBeginAtomIdx(), _b.GetEndAtomIdx()
                    if (_b.GetBondTypeAsDouble() == 2.0
                            and _i in _ring_atoms and _j in _ring_atoms
                            and _i not in _carbonyl_cs and _j not in _carbonyl_cs):
                        _res = True
                        break
            # Only supply the heterone when the SMARTS found no genuine 2-carbon
            # ketone (never clobber a real acyclic ketone read).
            if _res and 'ketone' not in results:
                results['ketone'] = _het_matches

    _reclassify_ring_heterols(mol, results)

    return dict(results)


def _reclassify_ring_heterols(mol, results) -> None:
    """ (Heterols) / (Hydroperoxides): an -OH or -OOH attached to a
    RING heteroatom is a suffix-eligible hydroxy/hydroperoxy compound
    (``piperidin-1-ol``, ``pyrrolidine-1-peroxol``, ``3,4,5,6-tetrahydro-1λ4,2-thiazin-
    1-ol``), NOT the non-suffix ``hydroxylamine`` / ``sulfinimidic_acid`` functional
    parent -- those structurally cannot exist once the heteroatom is a ring atom (its
    other valences are ring bonds, so there is no open functional-parent linkage). Move
    such matches to the generic ``alcohol`` / ``hydroperoxide`` classes so the ordinary
    ``-ol``/``-peroxol`` suffix machinery and the demotion cascade handle them
    (demoting to a ``hydroxy``/``hydroperoxy`` prefix when a senior group is present,
    e.g. ``1-hydroxypiperidine-3-carbonitrile``). Mutates ``results`` in place.
    """
    def _in_ring(idx):
        return mol.GetAtomWithIdx(idx).IsInRing()

    # (a) hydroxylamine [OX2H1][NX3...][#6] where the N is a RING atom -> heterol -ol.
    # alcohol tuple is (O, bearer); attachment index 1 = the ring N (PG_ATTACHMENT
    # ['alcohol'] == [1]).
    if 'hydroxylamine' in results:
        keep, moved = [], []
        for m in results['hydroxylamine']:
            o_idx, n_idx = m[0], m[1]
            if _in_ring(n_idx) and not _in_ring(o_idx):
                moved.append((o_idx, n_idx))
            else:
                keep.append(m)
        if moved:
            results.setdefault('alcohol', []).extend(moved)
            if keep:
                results['hydroxylamine'] = keep
            else:
                del results['hydroxylamine']

    # (b) sulfinimidic_acid [SX3](=[NX2])[OX2H1] where S and N are BOTH ring atoms
    # (intra-ring S=N) -> the -OH on the ring S is a heterol -ol. Match is
    # (S, N, O); build the alcohol tuple (O, S).
    if 'sulfinimidic_acid' in results:
        keep, moved = [], []
        for m in results['sulfinimidic_acid']:
            s_idx, n_idx, o_idx = m[0], m[1], m[2]
            if _in_ring(s_idx) and _in_ring(n_idx) and not _in_ring(o_idx):
                moved.append((o_idx, s_idx))
            else:
                keep.append(m)
        if moved:
            results.setdefault('alcohol', []).extend(moved)
            if keep:
                results['sulfinimidic_acid'] = keep
            else:
                del results['sulfinimidic_acid']

    # (c) -OOH on a ring heteroatom (no carbon-only SMARTS covers it) -> hydroperoxide
    # -peroxol. Detect structurally: a terminal O(H)-O- whose inner O is bonded to
    # a ring N/S/... heteroatom. hydroperoxide tuple is (O_outer, O_inner, bearer).
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'O' or atom.GetTotalNumHs() != 1 or atom.IsInRing():
            continue
        nbrs = [nb for nb in atom.GetNeighbors()]
        if len(nbrs) != 1:
            continue
        inner = nbrs[0]
        if inner.GetSymbol() != 'O' or inner.IsInRing():
            continue
        bearers = [nb for nb in inner.GetNeighbors() if nb.GetIdx() != atom.GetIdx()]
        if len(bearers) != 1:
            continue
        bearer = bearers[0]
        if bearer.GetSymbol() in ('C', 'H') or not bearer.IsInRing():
            continue  # carbon bearer is the ordinary hydroperoxide SMARTS' job
        tup = (atom.GetIdx(), inner.GetIdx(), bearer.GetIdx())
        results.setdefault('hydroperoxide', [])
        if tup not in results['hydroperoxide']:
            results['hydroperoxide'].append(tup)


# Per-molecule memoization (perf, 2026-08-28). `detect_functional_groups` was
# measured at ~37 calls PER MOLECULE (decomposition + retry/recovery cascades each
# re-perceive), the #1 self-time hot spot on the best-effort path. The result is a
# pure function of the molecule (SMARTS matches + ring reads, no context flags), so
# it is safe to compute once per mol object and reuse. Keyed by the RDKit Mol via a
# WeakKeyDictionary (mols are weakref-able), so the entry is freed when the mol is;
# distinct Mol objects for the same structure simply miss (correctness preserved).
# A fresh shallow copy is returned each call so a caller mutating its dict/lists can
# never corrupt the cached value (the tuples inside are immutable).
_FG_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def detect_functional_groups(mol) -> Dict[str, List[Tuple[int, ...]]]:
    """
    Detect all functional groups in a molecule (memoized per mol object).

    Returns a dictionary mapping functional group names to lists of atom index
    tuples. Each tuple contains the indices of atoms in one instance of that group.

    Example:
        >>> mol = Chem.MolFromSmiles("CC(=O)O") # acetic acid
        >>> groups = detect_functional_groups(mol)
        >>> "carboxylic_acid" in groups
        True
        >>> len(groups["carboxylic_acid"])
        1
    """
    cached = _FG_CACHE.get(mol)
    if cached is None:
        cached = _detect_functional_groups_impl(mol)
        try:
            _FG_CACHE[mol] = cached
        except TypeError:                     # mol not weakref-able (defensive) — skip cache
            return cached
    return {k: list(v) for k, v in cached.items()}


def detect_features(mol) -> Dict[str, List[Tuple[int, ...]]]:
    """functional-group perception fix (a phase) — the single shared feature-perception entry point.

    The ONE place every route (neutral and charged) reads functional-group classes
    from, so a mis-route can never produce a DIFFERENT feature set (audit Dim-02
    #1: "two parallel detectors with disjoint coverage"). Today it is a thin wrapper
    over ``detect_functional_groups`` (which now includes the charged + missing-class
    SMARTS); the orthogonal charge-SITE scan (``ions.get_ion_sites``) stays separate
    — it answers "where are the charges", not "what FG classes are present". Keeping
    a single accessor lets future consolidation route through one function.
    """
    return detect_functional_groups(mol)


# functional-group perception fix/ (a phase): prefix-only characteristic groups whose heteroatoms
# (N/O) are NOT skeletal/parent-chain atoms / /. They must never be
# walked into an aza/oxa chain (e.g. azidomethane CN=[N+]=[N-] -> wrong '2,3-diazabutane').
_CHAIN_EXCLUDED_FG = frozenset({
    "azido", "diazo", "nitroso", "nitrite", "nitro",
    "n_oxide_aromatic", "n_oxide_aliphatic", "n_oxide_imine",
    # W3-P11: nitrooxy -O-NO2 is a prefix-only characteristic
    # group; its O/N heteroatoms must never be walked into an oxa/aza skeletal
    # chain (parallel to nitro/nitrite). Only the carbon attachment is preserved.
    "nitrooxy",
})


def get_chain_excluded_atoms(mol) -> Set[int]:
    """Return the N/O atom indices of every prefix-only characteristic group
    (``_CHAIN_EXCLUDED_FG``) present in *mol*.

    These graph indices, derived from perception's OWN ``detect_functional_groups``
    matches (NOT a per-FG SMARTS blocklist — internal notes), are the atoms that must
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


# For a collision fg_specific whose SMARTS captures a trailing ANCHOR atom that is
# not part of the characteristic group, restrict the suppression atom set to the real
# group atoms. hydroxylamine SMARTS is [OX2H1][NX3...][#6] -> only O(0),N(1) belong to
# the group; the [#6] anchor(2) must not suppress an unrelated FG on that carbon.
_SUPPRESS_ATOM_SLICE = {
    'hydroxylamine': slice(0, 2),
}


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
        # Wave-2 completion: thiourea N-C(=S)-N owns its whole
        # unit — suppress the terminal-N amine perception and any thioamide/
        # thioketone match on its atoms (thiourea is the more specific FG).
        ('thiourea', ['thioamide', 'thioketone',
                      'primary_amine', 'secondary_amine', 'tertiary_amine']),
        ('guanidine', ['imine']),
        #: cyanamide N-C#N owns its whole unit — suppress the
        # nitrile and amine reads on its atoms so principal_group -> None and the
        # cyanamide handler names it. Atom-scoped: a separate nitrile/amine
        # elsewhere is untouched.
        ('cyanamide', ['nitrile', 'primary_amine', 'secondary_amine',
                       'tertiary_amine']),
        ('carbamate', ['ester', 'primary_amide', 'secondary_amide', 'tertiary_amide']),
        ('isocyanate', ['nitrile', 'primary_amide']),
        ('isothiocyanate', ['nitrile', 'primary_amide']),
        # Wave2 T6c: acyl pseudohalides own their whole
        # C(=O)-pseudohalogen unit — atom-overlap-scoped, so a separate
        # azide/nitrile/ketone elsewhere in the molecule is untouched.
        # a phase: imidoyl / carbothioyl / carboselenoyl halides are
        # named by their handler ONLY when they are the principal group (a bare
        # cycloalkane/chain acyl halide). We deliberately DO NOT suppress the broad
        # imine (=NH) / thioketone / selenoketone / halo reads on the acyl C=X:
        # when a SENIOR group (e.g. a carboxylic acid) is the parent, the acyl-halide
        # FG is non-principal and has no substitutive prefix form, so the substituent
        # layer must fall back to the atom-level 'chloro' + 'imino'/'sulfanylidene'
        # prefixes ('5-chloro-5-iminopentanoic acid') rather than drop the group and
        # abstain essential-substituent citation; degrade, never silence).
        # In the principal case the handler builds the whole name and ignores these
        # co-present reads, so keeping them is harmless there.
        ('acyl_azide', ['azido', 'ketone', 'aldehyde', 'imine', 'azo']),
        ('acyl_cyanide', ['nitrile', 'ketone', 'aldehyde']),
        # -3: a sulfonyl cyanide R-SO2-C#N owns its whole
        # unit -- suppress the plain nitrile read on the -C#N and the sulfone read
        # on the S(=O)2 (S bears methyl + cyanide C) so the sulfonyl_cyanide
        # handler names it 'methanesulfonyl cyanide'. Atom-scoped.
        ('sulfonyl_cyanide', ['nitrile', 'sulfone']),
        ('acyl_isocyanate', ['isocyanate', 'ketone', 'aldehyde', 'imine',
                             'nitrile', 'primary_amide', 'secondary_amide']),
        # Carbamic acid: N-C(=O)-OH must NOT also match carboxylic_acid or amide
        ('carbamic_acid', ['carboxylic_acid', 'primary_amide', 'secondary_amide', 'tertiary_amide']),
        # Thiocarboxylic acids: SH in C(=O)SH or C(=S)SH must NOT match thiol
        # C(=O)SH must NOT match thioester either (C(=O)S is substructure of both)
        ('thioic_S_acid', ['thiol', 'thioester']),
        # W3-P03-3: the C1 case H-C(=S)-SH / H-C(=S)-OH bears an H on
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
        # a phase Tier FRN-A: chalcogen-acid suppressions
        # (mirror thioic_S_acid -> thiol+thioester at line 228 above; AUDIT-FRN)
        # Forward-reference note: selenoester/telluroester/tellurol added in
        # commits 163-02-03 (chalcogen-ketones) + 163-02-05 (chalcogen-esters);
        # unknown FG names are harmlessly no-op'd by the resolver loop below.
        ('selenoic_Se_acid', ['selenol', 'selenoester', 'thioester']),
        ('diselenoic_acid', ['selenol', 'selenoketone']),
        ('selenoic_O_acid', ['selenoketone', 'carboxylic_acid']),
        ('telluroic_Te_acid', ['tellurol', 'telluroester', 'thioester']),
        ('ditelluroic_acid', ['tellurol', 'telluroketone']),
        ('telluroic_O_acid', ['telluroketone', 'carboxylic_acid']),
        # a phase Tier FRN-B: chalcogen-amide suppressions (AUDIT-FRN + RESEARCH).
        # Single-permissive [NX3] match captures =[S,Se,Te]-N(H,R) at all 3 N-degrees;
        # downstream N-degree inspection happens at assembly time.
        ('thioamide', ['thioketone', 'primary_amine', 'secondary_amine', 'tertiary_amine']),
        ('selenoamide', ['selenoketone', 'primary_amine', 'secondary_amine', 'tertiary_amine']),
        ('telluroamide', ['telluroketone', 'primary_amine', 'secondary_amine', 'tertiary_amine']),
        # a phase Tier FRN-C: chalcogen-aldehyde/ketone suppressions (AUDIT-FRN + RESEARCH).
        # Defensive suppression mirrors existing thioaldehyde discipline; =O vs =Se/=Te should not
        # overlap structurally but the cascade preserves parallelism for downstream safety.
        ('selenoaldehyde', ['aldehyde', 'ketone']),
        ('telluroaldehyde', ['aldehyde', 'ketone']),
        ('selenoketone', ['selenoether', 'selenoester']),
        ('telluroketone', ['telluroether', 'telluroester']),
        # a phase Tier FRN-D: iminoester suppressions (AUDIT-FRN + RESEARCH + Risk B).
        # Risk B mitigation: iminoester -> ester defensive suppression mirrors the existing thio*
        # cascade discipline; imine/primary_amine suppress the =NH from being double-claimed;
        # ether suppresses the -O-C portion from being double-claimed.
        ('iminoester', ['ester', 'imine', 'primary_amine', 'ether']),
        # W2F-P6: the N-substituted carbamimidate ester
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
        # a phase Tier FRN-E: chalcogen-ester suppressions (AUDIT-FRN + RESEARCH).
        # Note: Tier FRN-A suppressions at lines above already declared
        # ('selenoic_Se_acid', ['selenol', 'selenoester', 'thioester']) and
        # ('telluroic_Te_acid', ['tellurol', 'telluroester', 'thioester']) anticipating
        # the existence of selenoester/telluroester per RESEARCH line 306. Now that
        # those SMARTS exist, the suppression cascade works as designed.
        ('selenoester', ['selenol']),
        ('telluroester', ['tellurol']),
        # Ester O-Ar bond should NOT also match aromatic_ether.
        # The ester oxygen in -C(=O)-O-Ar is part of the ester, not a separate
        # aromatic ether; without this, phenyl esters double-name as both
        # "phenoxy" and "phenoxycarbonyl".
        ('ester', ['aromatic_ether']),
        #: broadened aldehyde [CX3;H1,H2](=O) now matches C=O in acid
        # derivatives (formates, formamides, etc.); suppress aldehyde on overlap
        ('carboxylic_acid', ['aldehyde']),
        ('ester', ['aldehyde']),
        # W3-P06 /.4): a chalcogen-bridge anhydride folded into
        # 'anhydride' also matches thio/seleno/telluro-ester (CO-X-CO); a peroxy
        # anhydride (CO-OO-CO) matches 'peroxide' on its bridge. Suppress those
        # sub-component reads (+ defensively ketone) so the whole bridge unit is
        # owned by the anhydride handler. W8-P4: the cyanic/thiocyanic mixed-
        # anhydride fold also matches 'ester' (acyl-O-) and 'cyanate'/
        # 'thiocyanate' (-O-C#N/-S-C#N) on the SAME atoms; suppress those too so
        # 'acetic cyanic anhydride' is not mis-perceived as an ester.
        ('anhydride', ['aldehyde', 'thioester', 'selenoester', 'telluroester',
                       'ketone', 'peroxide', 'ester', 'cyanate', 'thiocyanate']),
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
        #: imide suppresses overlapping amide matches to prevent
        # double-counting C=O groups (one as amide, one as ketone)
        ('imide', ['primary_amide', 'secondary_amide', 'tertiary_amide']),
        #: broadened amine pattern overlaps with aromatic_amine on aromatic carbons
        ('aromatic_amine', ['primary_amine']),
        #: hydroxamic acid suppresses amide + alcohol false positives
        ('hydroxamic_acid', ['primary_amide', 'secondary_amide', 'primary_alcohol',
                             'secondary_alcohol', 'tertiary_alcohol', 'carboxylic_acid']),
        #: cyanate/thiocyanate suppress ether/thioether + nitrile
        ('cyanate', ['ether', 'nitrile']),
        ('thiocyanate', ['thioether', 'nitrile']),
        #: azo suppresses imine
        ('azo', ['imine']),
        #: guanidine suppresses amidine (guanidine is more specific)
        ('guanidine', ['amidine']),
        # Wave2 completion C: the diazo C=[N+] matches the imine
        # SMARTS, injecting a bogus 'imino' prefix that the validity gate
        # then suppressed (ethyl diazoacetate / diazomethane were unknown).
        ('diazo', ['imine']),
        # Wave2 completion: the acyl -OOH matches the hydroperoxide
        # SMARTS; the peroxy-acid FG owns those atoms.
        ('peroxy_acid', ['hydroperoxide', 'peroxide', 'ester', 'ketone',
                         'aldehyde', 'carboxylic_acid']),
        # Wave2 completion: the C=NH of an imidic acid matches
        # the imine SMARTS and its -OH the alcohol/enol patterns.
        ('imidic_acid', ['imine', 'alcohol', 'primary_alcohol',
                         'secondary_alcohol', 'tertiary_alcohol', 'enol']),
        # W3-P02-3: the C=N-NH2 of a hydrazonic acid matches the
        # hydrazone SMARTS; the hydrazonic_acid FG owns the whole geminal
        # C(=N-NH2)(OH) unit (suffix 'hydrazonic acid' / demoted hydroxy +
        # hydrazinylidene). Also defensively clears imine/alcohol/enol on its
        # atoms (the sp2 C keeps alcohol/enol from matching anyway).
        ('hydrazonic_acid', ['hydrazone', 'imine', 'alcohol', 'primary_alcohol',
                             'secondary_alcohol', 'tertiary_alcohol', 'enol']),
        # W3-P02-5: the C=N-OH of a hydroximic acid matches the
        # oxime SMARTS; the hydroximic_acid FG owns the whole geminal
        # C(=N-OH)(OH) unit (PIN = N-hydroxy + imidic acid; demoted = hydroxy +
        # hydroxyimino). Clears oxime + defensively imine/alcohol/enol.
        ('hydroximic_acid', ['oxime', 'imine', 'alcohol', 'primary_alcohol',
                             'secondary_alcohol', 'tertiary_alcohol', 'enol']),
        # Wave2 completion: the S-hydrazido/hydrazono N pairs
        # match the hydrazine/hydrazone/imine/amine patterns.
        ('sulfinohydrazonohydrazide', ['hydrazine_fg', 'sulfinimidamide',
                                       'hydrazone', 'imine', 'primary_amine']),
        # Wave2 composite-N precedence (ORDER LOAD-BEARING — the resolver
        # applies top-to-bottom on live results):
        # 1. hydrazidine (R-C(=N-NH2)-NH-NH2, the most specific: 2-N hydrazido
        # side) clears hydrazonamide/hydrazone/amidine/hydrazine_fg/primary_amine/
        # imine on its atoms.
        # Wave-2 P1AM Task 7: the imidohydrazide tautomer
        # R-C(=NH)-NH-NH2 -- terminal =NH (D1) -- clears the amidine (the
        # C=NH + first NH) and hydrazine_fg (the NH-NH2) matches on its atoms.
        # Placed BEFORE hydrazidine so the most-specific composite wins first.
        ('imidohydrazide', ['amidine', 'hydrazine_fg', 'imine']),
        ('hydrazidine', ['hydrazonamide', 'hydrazone', 'amidine',
                         'hydrazine_fg', 'primary_amine', 'imine']),
        # 2. hydrazonamide (amidrazone, 1-N amino side) clears amidine/hydrazone/
        # primary_amine. Placed BEFORE ('guanidine',['hydrazonamide']) so on the
        # carbonic-diamide NC(=NN)N it strips amidine/hydrazone first, THEN
        # guanidine strips hydrazonamide -> guanidine cleanly wins.
        ('hydrazonamide', ['amidine', 'hydrazone', 'primary_amine']),
        # 3. guanidine (3-N carbon) owns the carbonic-diamide case.
        ('guanidine', ['hydrazonamide', 'hydrazidine']),
        # + Wave2 T3d: amidine suppresses imine/primary_amine, AND oxime
        # (the amidoxime -C(=N-OH)-NH2 is a N'-hydroxy amidine, NOT an
        # oxime; atom-scoped so standalone oximes CC=NO are untouched).
        ('amidine', ['imine', 'primary_amine', 'oxime']),
        # W3-P02-7 /: an N-acyl amidine R-C(=NH)-N(R')-C(=O)-R''
        # is a carboxAMIDE bearing an N-imidoyl substituent, NOT a free amidine —
        # carboxamide is SENIOR to carboximidamide. When an amidine's
        # amino-N is (also) an amide nitrogen (secondary/tertiary amide match
        # shares that N and/or the imidoyl C), suppress the amidine so the amide
        # is the sole principal group and the C(=NH)- becomes the N-'{stem}animidoyl'
        # substituent (BB 30462: 'N-ethanimidoyl-N-methylacetamide'). Atom-overlap-
        # scoped: a genuinely separate amidine elsewhere is untouched. Placed AFTER
        # the amidine rule above so amidine's own imine/oxime suppressions fire first.
        ('secondary_amide', ['amidine']),
        ('tertiary_amide', ['amidine']),
        # Wave2: thiohydrazide suppresses thioamide + hydrazine_fg.
        ('thiohydrazide', ['thioamide', 'hydrazine_fg']),
        #: acid iodide suppresses aldehyde (parallel to other acid halides)
        ('acid_iodide', ['aldehyde']),
        #: disulfide suppresses thioether if S atoms overlap
        ('disulfide', ['thioether']),
        # DD2 Fix C (Phase D, /: the chalcogen hydroperoxol
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
        #: hydrazine suppresses primary_amine on its -NH2 nitrogen
        ('hydrazine_fg', ['primary_amine']),
        #: hydrazide is more specific than hydrazine_fg
        ('hydrazide', ['hydrazine_fg']),
        # C1 fix: sulfonohydrazide (R-SO2-NH-NH2) is the most specific
        # read on its N-N atoms -- suppress the generic hydrazine and any
        # sulfonamide/hydrazide read that shares those atoms. (The carbonyl-based
        # hydrazide SMARTS cannot match a sulfonyl anyway; suppress defensively.)
        ('sulfonohydrazide', ['hydrazine_fg', 'primary_sulfonamide',
                              'secondary_sulfonamide', 'tertiary_sulfonamide',
                              'hydrazide']),
        #: specific alcohol subtypes suppress generic "alcohol" on same atoms
        ('primary_alcohol', ['alcohol']),
        ('secondary_alcohol', ['alcohol']),
        ('tertiary_alcohol', ['alcohol']),
        ('phenol', ['alcohol']),
        ('enol', ['alcohol']),
        #: hydroxamic acid suppresses generic alcohol too
        ('hydroxamic_acid', ['alcohol']),
        # functional-group perception fix (169.7): hydroxylamine (R-NH-OH) suppresses amine + alcohol on its
        # N/O atoms (the N is an amine-N and the O an -OH to the generic patterns).
        ('hydroxylamine', ['primary_amine', 'secondary_amine', 'tertiary_amine',
                           'aromatic_amine', 'alcohol', 'primary_alcohol',
                           'secondary_alcohol', 'tertiary_alcohol']),
        # functional-group perception fix (169.7): carbonic acid (HO-C(=O)-OH) suppresses the carboxylic_acid
        # + ester generics on its carbon (it is a functional parent, not a
        # carboxylic acid).
        ('carbonic_acid', ['carboxylic_acid', 'ester']),
        # functional-group perception fix (169.7): free inorganic oxoacids suppress the (now [#6]-tightened)
        # carbon-acid + ester generics on their atoms for robustness parents).
        ('phosphoric_acid', ['phosphonic_acid', 'phosphate_monoester',
                             'phosphate_diester', 'phosphate_triester']),
        ('sulfuric_acid', ['sulfonic_acid']),
        # W3-P04: the -OO- modified sulfonic acid owns its whole
        # R-SO2-OOH unit. The generics cannot structurally match (verified: the
        # -OH is not on S, the inner peroxo-O is on S not C, the S bears only one
        # C) but suppress defensively so a future SMARTS broadening cannot steal
        # the group or double-name the -OOH as a hydroperoxy prefix.
        ('sulfonoperoxoic_acid', ['sulfonic_acid', 'sulfuric_acid', 'sulfone',
                                  'hydroperoxide', 'peroxide']),
        # W3-P04: the -SH-modified sulfonic acid owns its whole
        # R-SO2-SH unit. Generics cannot match (the -SH S is on S not C; the
        # sulfonyl S bears one C) but suppress defensively.
        ('sulfonothioic_S_acid', ['sulfonic_acid', 'sulfuric_acid', 'sulfone',
                                  'thiol', 'thioether', 'disulfide']),
        # W3-P04: the =NH-modified sulfinic acid owns its
        # R-S(=NH)-OH unit; suppress the generic sulfinic read and any imine
        # (S=N) / alcohol (S-OH) reads defensively.
        ('sulfinimidic_acid', ['sulfinic_acid', 'imine', 'alcohol',
                               'primary_alcohol', 'secondary_alcohol',
                               'tertiary_alcohol']),
        # W3-P04 /: the =NH-modified sulfonic acid owns
        # its R-S(=O)(=NH)-OH unit; suppress the generic sulfonic/sulfone read
        # and any imine (S=N) / oxime (=N-OH, the N-hydroxy variant) /
        # hydroxylamine / alcohol reads on its atoms.
        ('sulfonimidic_acid', ['sulfonic_acid', 'sulfuric_acid', 'sulfone',
                               'imine', 'oxime', 'hydroxylamine', 'alcohol',
                               'primary_alcohol', 'secondary_alcohol',
                               'tertiary_alcohol']),
        ('nitric_acid', ['nitro', 'nitroso']),
        # -05 : a nitrite ester (R-O-N=O) suppresses any residual
        # `nitroso` on its atoms — makes the O-vs-C precedence explicit/robust on
        # top of the [#6] guard above.
        ('nitrite', ['nitroso']),
    ]:
        if fg_specific in results:
            specific_atoms = set()
            _slice = _SUPPRESS_ATOM_SLICE.get(fg_specific)
            for match in results[fg_specific]:
                specific_atoms.update(match[_slice] if _slice else match)

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
