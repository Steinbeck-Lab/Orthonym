"""
Benzene naming rules according to IUPAC 2013 (Blue Book).

Handles:
- Monosubstituted benzenes (chlorobenzene, nitrobenzene)
- Polysubstituted benzenes with numeric locants (1,4-dimethylbenzene)
- Benzene ring orientation for lowest locants
- Alphabetical ordering of substituents
- Suffix functional groups on benzene (carboxylic acid, amide, sulfonamide, etc.)

IUPAC 2013 PIN Rules:
- Numeric locants are REQUIRED (not ortho/meta/para)
- Toluene is retained ONLY for unsubstituted methylbenzene
- Substituted methylbenzene uses "methylbenzene" (not "toluene")
- Position 1 assigned to give lowest locants via first-point-of-difference
- Ring-attached principal groups use suffix form
- benzamide = retained name for C6H5CONH2
"""

import re
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

from ..assembly.naming_utils import (
    alpha_sort_key,
    format_substituent_prefix,
    get_alkyl_name,
    get_multiplier_prefix,
    get_suffix_multiplier_prefix,
    is_complex_substituent,
    prefix_citation_sort_key,
    should_omit_locant_one,
)
from .locants import compare_locant_sets as _compare_locant_sets  #
from ..perception.molcache import atoms_of, inchikey_of
from ..perception.smarts_cache import compiled as _compiled_smarts

# Mapping from substituent atom symbol/pattern to prefix name
# Key: (symbol, hybridization/bond_info) or simple symbol
# Value: prefix name
SUBSTITUENT_PREFIXES = {
    # Halogens
    "F": "fluoro",
    "Cl": "chloro",
    "Br": "bromo",
    "I": "iodo",
    # Common groups - these are detected by the get_benzene_substituents function
    # based on the substituent structure
}

# Priority order for suffix functional groups on benzene
# Highest priority first (carboxylic acid > sulfonamide > amide > nitrile > aldehyde)
_SUFFIX_PRIORITY = [
    'carboxylic acid',
    # Phase C: peroxy acid, rank 1 in SENIORITY_ORDER (directly after
    # carboxylic_acid at rank 0). the Blue Book benzenecarboperoxoic acid (PIN).
    'carboperoxoic acid',
    # Suite fix j6 (TRIAGE g8 C23): imidic_acid is rank 11 in SENIORITY_ORDER,
    # after the carboxylic/peroxy/chalcogen C-acids and before sulfonic_acid
    # (rank 15), so it sits here. 'carbothioic S-acid' (rank 2) is not in this
    # list; _assemble_benzene_with_suffix hands the suffix to a co-occurring
    # thioic S-acid rather than let the junior imidic acid win.
    'carboximidic acid',
    'sulfonic acid',
    # D-FOLLOWON item 5: P/Se/Te + sulfinic ring oxoacids, junior to carboxylic
    # acid and sulfonic acid acid seniority C-acids > S > Se/Te/P oxoacids) so
    # a co-occurring -COOH stays principal and these demote to their prefixes.
    'sulfinic acid',
    'phosphonic acid',
    'selenonic acid',
    # W3-P11: Se/Te -inic ring oxoacids, junior to their -onic parents
    # (-onic > -inic) — near-zero-corpus tie only; standalone naming is
    # rank-independent (benzeneseleninic acid is the sole suffix here).
    'seleninic acid',
    'telluronic acid',
    'tellurinic acid',
    'sulfonamide',
    # Wave-2 P1AM: sulfonimidamide ranks just below sulfonamide.
    'sulfonimidamide',
    # Task Y, Table 6.1 item 24): sulfinamide -SO-NH2 ranks
    # below sulfonimidamide (item 20) and above sulfonohydrazide (item 34).
    'sulfinamide',
    # C1: sulfonohydrazide ranks with the sulfonamide family, above
    # the carbon carboxamide/carbohydrazide (S oxoacid-derivatives are named after
    # the S parent; here it is the sole principal group in the target set).
    'sulfonohydrazide',
    # Wave2 completion, Table 6.1 item 41): -S(=NNH2)-NHNH2
    # ranks below sulfonohydrazide (item 34).
    'sulfinohydrazonohydrazide',
    'carbonyl chloride',
    'carboxamide',
    # R7, BB 18762): ring-attached thioamide -C(=S)-NH2 ->
    # '-carbothioamide'. The S-analogue of carboxamide; junior to the O-amide
    # chalcogen seniority O > S) so a co-occurring carboxamide/acid stays
    # principal. Rank among the amide cluster is standalone-independent for the
    # sole-group target (benzenecarbothioamide).
    'carbothioamide',
    # C1: ring-attached -C(=O)NN -> '-carbohydrazide'. Ranks with
    # the carboxamide (both are added-carbon C-suffixes on the ring); placed after
    # carboxamide so a co-occurring amide would stay principal.
    'carbohydrazide',
    # Wave2: ring-attached hydrazidine -C(=N-NH2)-NH-NH2 ->
    # '-carbohydrazonohydrazide'. Ranks with the hydrazide family.
    'carbohydrazonohydrazide',
    # Wave2: ring-attached thiohydrazide -C(=S)-NH-NH2 ->
    # '-carbothiohydrazide'.
    'carbothiohydrazide',
    # C2 / seniority): amidine ranks below amide and above
    # nitrile, so '-carboximidamide' sits between 'carboxamide' and
    # 'carbonitrile'. A co-occurring amide/acid therefore stays principal.
    'carboximidamide',
    # Wave2: ring-attached amidrazone -C(=N-NH2)-NH2 ->
    # '-carbohydrazonamide'. Ranks just below carboximidamide (amidine).
    'carbohydrazonamide',
    'carbonitrile',
    'carbaldehyde',
    # C- (V-2): chalcogen analogues of the aldehyde-on-ring suffix
    #. Seniority O > S > Se > Te. Ring-attached -CH=S/Se/Te was
    # silently dropped (the C read as a methyl -> 'methylbenzene'); these emit
    # benzenecarbothialdehyde / benzenecarboselenaldehyde / benzenecarbotelluraldehyde.
    'carbothialdehyde',
    'carboselenaldehyde',
    'carbotelluraldehyde',
    'ol',  #: hydroxyl as suffix when principal group on benzene
    # Phase C tranche C: thiol is rank 94 in SENIORITY_ORDER, between the alcohols
    # (88/89) and phenol (91) above it and the amines (102+) below. the Blue Book
    # `benzenethiol (PIN) (not thiophenol)`. Placed here, NOT appended at the end --
    # putting it after 'amine' would make a thiol lose to an amine, inverting.
    'thiol',
    'selenol',
    'tellurol',
    # C4 /: amine is JUNIOR to alcohol and all C/S/Se/Te/P
    # acids, amides, nitriles, aldehydes, ketones. It MUST be the LAST entry so
    # it only becomes the ring parent suffix ('aniline') when nothing more
    # senior is present; any co-present senior group keeps the amine demoted to
    # the 'amino'/(N-alkylamino) prefix (4-aminophenol, 4-aminobenzoic acid).
    'amine',
]

# Prefix forms for suffix FGs when they are NOT the principal group
_SUFFIX_TO_PREFIX = {
    # Phase C tranche C: needed so a `-thiol` that loses the suffix slot to the
    # senior `-ol` demotes to its correct prefix rather than being dropped.
    # the Blue Book `benzenethiol (PIN)`; the demoted form is `sulfanyl`
    # (e.g. `2-sulfanylphenol`).
    'thiol': 'sulfanyl',
    'selenol': 'selanyl',
    'tellurol': 'tellanyl',
    'carboxylic acid': 'carboxy',
    'sulfonic acid': 'sulfo',
    # D-FOLLOWON item 5 (prefix forms; seniority.py FG_PREFIXES already has these).
    'sulfinic acid': 'sulfino',
    'phosphonic acid': 'phosphono',
    'selenonic acid': 'selenono',
    'seleninic acid': 'selenino',  # W3-P11: -inic prefix form
    'telluronic acid': 'tellurono',
    'tellurinic acid': 'tellurino',  # W3-P11: -inic prefix form
    'sulfonamide': 'sulfamoyl',
    # Task Y: NOT 'sulfinamoyl' -- gives `aminosulfinyl* (not sulfinamoyl)`.
    'sulfinamide': 'aminosulfinyl',
    'sulfonohydrazide': 'hydrazinesulfonyl',  # C1
    'carbonyl chloride': 'carbonochloridoyl',
    'carboxamide': 'carbamoyl',
    'carbothioamide': 'carbamothioyl',  # R7 / BB 55681)
    'carbohydrazide': 'hydrazinecarbonyl',  # C1
    'carbohydrazonohydrazide': 'hydrazinecarbohydrazonoyl',  # Wave2
    'carbothiohydrazide': 'hydrazinecarbothioyl',  # Wave2
    'carboximidamide': 'carbamimidoyl',  # C2
    # Suite fix j6: '4-(C-hydroxycarbonimidoyl)benzoic acid (PIN)',
    # the Blue Book).
    'carboximidic acid': 'C-hydroxycarbonimidoyl',
    'carbohydrazonamide': 'carbamohydrazonoyl',  # Wave2
    'carbonitrile': 'cyano',
    'carbaldehyde': 'formyl',
    'ol': 'hydroxy',  #: when OH is not principal, use prefix form
    'amine': 'amino',  # C4: demoted amine -> 'amino' prefix when a senior group wins
}


# FGs whose presence blocks the prefix->suffix promotion below, keyed on the
# DETECTOR's vocabulary (``perception.functional_groups.FUNCTIONAL_GROUP_SMARTS``).
#
# ⚠ Phase C Task 9b: this set carried two keys that could never fire --
# ``'azide'`` and ``'selenocyanate'`` -- because the detector emits ``'azido'`` and
# has no selenocyanate pattern at all. That is "presence in a lookup table is not
# evidence the table is REACHED" in its exact shape, so ``_assert_registry_keys``
# below now fails at import if a member is not a detector key, and the dead keys are
# gone rather than repaired: azides must NOT block the promotion.
# §** "AZIDES"** (``the Blue Book Blue Book``) states *"Compounds
# containing a -N3... group attached to a parent hydride, are named using
# substitutive nomenclature and the prefix 'azido'. This method gives preferred
# IUPAC names rather than names based on the class name 'azido' in functional class
# nomenclature"*, and ``:25997`` gives the decisive worked PIN
# ``3-azidonaphthalene-2-sulfonic acid (PIN)`` -- azido cited as a detachable prefix
# while a SUFFIX governs the parent. So ``6-azidobenzene-1,2,3,4,5-pentol`` is the
# PIN and the dead key was accidentally producing the right answer for the wrong
# reason.
#
# ⚠ KNOWN RESIDUAL DEFECT, derived but deliberately NOT changed here (it is a
# suffix-SELECTION question, outside this task's numbering scope): by the same
# argument the remaining members are wrong too. ``:1710`` item (p) records that
# *"The -N=C=O group, its chalcogen analogues, and the -NC group have been added to
# the list of characteristic groups that are always cited as prefixes in
# substitutive nomenclature"*, ``:26003`` repeats it for isocyanates, and ``:26014``
# gives ``4-isocyanatobenzene-1-sulfonyl chloride (PIN)`` -- isocyanato as a prefix
# with a suffix on the parent. A group that can never BE a suffix cannot outrank
# -ol, so ``Oc1c(O)c(O)c(O)c(O)c1N=C=O`` should be
# ``6-isocyanatobenzene-1,2,3,4,5-pentol`` and today emits
# ``1,2,3,4,5-pentahydroxy-6-isocyanatobenzene``. Removing the guard wholesale
# changes the benzene SUFFIX layer, so it is recorded here with its citations and
# pinned by an xfail-strict test rather than guessed at.
_FUNCTIONAL_CLASS_FGS = frozenset({
    'isocyanate', 'isothiocyanate', 'diazo', 'cyanate', 'thiocyanate',
})

# Suffixes JUNIOR to -ol: SENIORITY_ORDER ranks phenol 91, thiol 94,
# selenol/tellurol just after, amines 102+. A junior suffix holding the slot must
# not block the -ol promotion.
#
# ⚠ Phase C Task 9b: two comments used to claim this set was "shared with the
# (c) anchor", which was FALSE -- the anchor never referenced it. The claim is
# now TRUE rather than deleted: it is read by ``benzene_prefix_suffix_promotion``
# below, which is the single authority both the anchor and the promotion consult.
_OL_JUNIOR_SUFFIXES = frozenset({'thiol', 'selenol', 'tellurol', 'amine'})


def _assert_registry_keys() -> None:
    """Every ``_FUNCTIONAL_CLASS_FGS`` member must be a key the DETECTOR can emit.

    A blacklist keyed on a vocabulary nothing produces is silently inert, and an
    inert guard reads as a working one. Called at import so the failure is loud and
    immediate rather than a name that is quietly wrong.
    """
    from ..perception.functional_groups import FUNCTIONAL_GROUP_SMARTS
    dead = sorted(_FUNCTIONAL_CLASS_FGS - set(FUNCTIONAL_GROUP_SMARTS))
    if dead:
        raise AssertionError(
            '_FUNCTIONAL_CLASS_FGS members absent from '
            'perception.functional_groups.FUNCTIONAL_GROUP_SMARTS (they can never '
            'fire): %s' % dead
        )


_assert_registry_keys()


def _has_functional_class_fg(detected_fgs) -> bool:
    """True when a functional-class-named FG is present (blocks suffix promotion)."""
    if not detected_fgs:
        return False
    return any(fg in _FUNCTIONAL_CLASS_FGS for fg in detected_fgs)


def benzene_prefix_suffix_promotion(
    suffix_names,
    prefix_names,
    detected_fgs=None,
    has_amine_candidates: bool = False,
) -> Tuple[Optional[str], frozenset]:
    """THE single authority on which detachable prefix becomes the ring suffix.

    Two questions must give the same answer and were previously computed twice:

    1. ``name_substituted_benzene`` -- which prefix does the NAME promote to the
       suffix? (a ring -OH is perceived as the prefix ``hydroxy`` and only becomes
       ``-ol`` here; likewise a promotable amine becomes ``-amine``/aniline.)
    2. ``principal_group_ring_atoms`` -- which ring atoms may criterion (c)
       therefore minimise the locants of?

    ⚠ Phase C Task 9b, root cause of C-2: (2) used to answer *"the principal
    group has a prefix form that appears on this ring"*, which is true for **76** of
    the 136 seniority names (every entry with both a ring suffix and a prefix) while
    (1) only ever promotes **two** of them. For the other 74, criterion (c)
    minimised the locant of a group the name still spells as a detachable prefix --
    INVERTING (f)/(g). Two answers that must agree and are computed twice is the
    defect shape, so both callers now read this one function.

    Args:
        suffix_names: the suffix names already present (``suffix_groups`` keys).
        prefix_names: the detachable prefix names present (``prefix_groups`` keys).
        detected_fgs: ``features.functional_groups``, for the functional-class guard.
        has_amine_candidates: whether any ring amine carries the
            ``amine_candidate`` payload ``_identify_nitrogen_group`` sets.

    Returns:
        ``(promoted_suffix, promoted_prefix_names)``. ``(None, frozenset)`` when
        no promotion happens. For the amine promotion the prefix set is empty
        because the promoted atoms are identified by the ``amine_candidate`` marker
        (the prefix may be ``amino``, ``(N-methylamino)``,...), not by one name.
    """
    if _has_functional_class_fg(detected_fgs):
        return None, frozenset()

    suffix_names = set(suffix_names or ())
    prefix_names = set(prefix_names or ())

    #: the guard is "no suffix SENIOR to -ol", not "no suffix at all" -- a
    # junior -thiol holding the slot is demoted to `sulfanyl` instead of blocking.
    senior_to_ol = {s for s in suffix_names if s not in _OL_JUNIOR_SUFFIXES}
    if 'hydroxy' in prefix_names and not senior_to_ol:
        return 'ol', frozenset({'hydroxy'})

    # The amine (aniline) promotion runs only when NOTHING holds the suffix -- and
    # if the -ol promotion had fired, ``suffix_groups`` would hold 'ol' by now, so
    # reaching here already means it did not.
    if has_amine_candidates and not suffix_names:
        return 'amine', frozenset()

    return None, frozenset()


# Pre-compiled SMARTS for suffix FG identification (avoid per-call recompilation)
_BENZENE_FG_SMARTS = {
    'acid': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),
    # Phase C: aryl PEROXY acid -C(=O)-O-OH -> '-carboperoxoic acid'.
    # Disjoint from 'acid' above by construction: that SMARTS requires the O bonded to
    # the carbon to carry the H ([OX2H1]), and in a peroxy acid that O is bonded to a
    # second O instead. Checked first anyway, so the intent is explicit.
    'peroxy_acid': Chem.MolFromSmarts('[CX3](=O)[OX2][OX2H1]'),
    # Phase C: aryl THIOL -SH -> '-thiol'. / the Blue Book + the Blue Book both print
    # 'C6H5-SH benzenethiol (PIN) (not thiophenol)'.
    'ring_thiol': Chem.MolFromSmarts('[SX2H1]'),
    'hydroxamic': Chem.MolFromSmarts('[CX3](=O)[NX3;H1][OX2H]'),
    'amide': Chem.MolFromSmarts('[CX3](=O)[NX3H2]'),
    'sec_amide': Chem.MolFromSmarts('[CX3](=O)[NX3H1][#6]'),
    'tert_amide': Chem.MolFromSmarts('[CX3](=O)[NX3]([#6])[#6]'),
    # C1: ring-attached hydrazide -C(=O)-NH-NH2. The amide N is
    # bonded to N (not H2 / not [#6]), so none of the amide SMARTS above match it;
    # check this pattern BEFORE the amide patterns for cleanliness.
    'hydrazide': Chem.MolFromSmarts('[CX3](=O)[NX3][NX3]'),
    'aldehyde': Chem.MolFromSmarts('[CX3H1](=O)'),
    # C- (V-2): chalcogen aldehydes (-CH=S / -CH=Se / -CH=Te). The H1
    # requirement excludes 0-H carbons (chalcogen ketones/amides), so a
    # selenobenzamide ring-C(=Se)NH2 is NOT matched here.
    'thioaldehyde': Chem.MolFromSmarts('[CX3H1](=[SX1])'),
    'selenoaldehyde': Chem.MolFromSmarts('[CX3H1](=[SeX1])'),
    'telluroaldehyde': Chem.MolFromSmarts('[CX3H1](=[TeX1])'),
    'nitrile': Chem.MolFromSmarts('[CX2]#[NX1]'),
    # C2: ring-attached amidine -C(=N)N -> '-carboximidamide' suffix.
    # =[NX2] (not =O) restricts it to true amidines so C(=O)N (amide) never
    # matches; matched AFTER the amide/hydrazide blocks in the detector so amide
    # wins on seniority. The C(=N)N guanidine/urea attach via N (start_idx would
    # be N, not this C), so the C-branch is never reached for them.
    'amidine': Chem.MolFromSmarts('[CX3](=[NX2])[NX3]'),
    # Wave2 T3d: ring-attached composite-N added-carbon suffixes. Matched BEFORE
    # amidine/hydrazide in the detector (they are more specific: amidine's
    # [CX3](=[NX2])[NX3] and hydrazide's [CX3](=O)[NX3][NX3] both partially match
    # these, so amidrazone/hydrazidine/thiohydrazide must be tested first).
    'hydrazidine_ring': Chem.MolFromSmarts('[CX3](=[NX2][NX3])[NX3][NX3]'),
    'hydrazonamide_ring': Chem.MolFromSmarts('[CX3](=[NX2][NX2,NX3])[NX3]'),
    'thiohydrazide_ring': Chem.MolFromSmarts('[CX3](=S)[NX3][NX3]'),
    # R7: ring-attached PRIMARY thioamide -C(=S)-NH2. The [NX3H2]
    # (unsubstituted amide N) is deliberate: this bare-suffix path cannot cite an
    # N-substituent, so restricting to -NH2 makes an N-mono/N,N-di/N-aryl thioamide
    # NOT match here (both the suffix and the carbamothioyl-prefix roles) rather
    # than silently DROP the N-substituent and emit a methyl-less name (a no-Java
    # leak; masks it only when Java is present). Disjoint from the
    # thioaldehyde [CX3H1](=[SX1]) (0-H carbon) and from thiohydrazide's -NH-NH2
    # (its amide N is NX3H1, not NX3H2). N-substituted thioamides fall through to
    # the pre-R7 behaviour (fail-closed / general substituent naming).
    'thioamide': Chem.MolFromSmarts('[CX3](=[SX1])[NX3H2]'),
    'acid_cl': Chem.MolFromSmarts('[CX3](=O)[Cl]'),
    'thio_acid': Chem.MolFromSmarts('[CX3](=O)[SX2H1]'),
    # a phase Group A / Table 4.3): ring-attached hydrazonic acid
    # -C(=N-NH2)-OH -> '-carbohydrazonic acid'. Perception already fires
    # (`hydrazonic_acid` FG), but the benzene ring handler had no suffix routing,
    # so `NN=C(O)c1ccccc1` fell to systematic. =N-NH2 (not =O / not single N) is
    # disjoint from amide/hydrazide/amidine here. No italic chalcogen locant
    # (hydrazonic acid has a single acid oxygen).
    'hydrazonic_acid': Chem.MolFromSmarts('[CX3](=[NX2][NX3H2])[OX2H1]'),
    # Suite fix j6 (TRIAGE g8 C23;, the Blue Book 'benzene-
    # carboximidic acid (PIN)'): ring-attached imidic acid -C(=NH)-OH ->
    # '-carboximidic acid'. =[NX2H1] (an UNSUBSTITUTED imino N, as perception's
    # `imidic_acid` FG requires) keeps an N-substituted imidate/imine out; the
    # -OH (OX2H1) keeps the imidate ester (O-alkyl) and the amidine out.
    'imidic_acid': Chem.MolFromSmarts('[CX3](=[NX2H1])[OX2H1]'),
    'sulfonamide': Chem.MolFromSmarts('[SX4](=O)(=O)[NX3H2]'),
    # F-B: N-substituted ring sulfonamide -S(=O)(=O)-NR'R''. The
    # NX3 has fewer than two H, so it is disjoint from the primary 'sulfonamide'
    # (NX3H2) above and matched only after it. The italic-N substituents are named
    # by _detect_sulfonamide_n_substituents and merged with the ring locants.
    'n_sub_sulfonamide': Chem.MolFromSmarts('[SX4](=O)(=O)[NX3;!H2;!$([NX3]~[!#6;!S])]'),
    # Task Y): ring-attached sulfinamide -S(=O)-NH2. SX3 (one
    # fewer O) keeps it disjoint from every SX4 pattern here; the carbon guard
    # parallels sulfinic_acid and excludes H2N-S(=O)-OH ("the name
    # sulfinamidic acid is not an approved name").
    'sulfinamide': Chem.MolFromSmarts('[SX3;$([SX3][#6])](=[OX1])[NX3H2]'),
    # Wave-2 P1AM, BB 34173): ring-attached sulfonimidamide
    # -S(=O)(=NH)-NH2. The imido =N breaks the sulfonamide (=O)(=O) SMARTS so
    # there is no overlap; checked BEFORE sulfonamide (more specific).
    'sulfonimidamide': Chem.MolFromSmarts('[SX4](=[OX1])(=[NX2])[NX3H2]'),
    # C1: ring-attached sulfonohydrazide -SO2-NH-NH2. The N is
    # [NX3H1] bonded to N (not H2), so the sulfonamide SMARTS above never matches
    # it; check this pattern first for cleanliness.
    'sulfonohydrazide': Chem.MolFromSmarts('[SX4](=O)(=O)[NX3][NX3]'),
    # Wave2 completion: -S(=N-NH2)-NH-NH2 (SX3 keeps it disjoint
    # from the SX4 sulfonohydrazide).
    'sulfinohydrazonohydrazide':
        Chem.MolFromSmarts('[SX3](=[NX2][NX3])[NX3][NX3]'),
    'sulfonic': Chem.MolFromSmarts('[SX4](=O)(=O)[OX2H1]'),
    # D-FOLLOWON item 5 /: the P/Se/Te ring oxoacids + ring
    # sulfinic acid as demotable suffix FGs (mirrors the sulfonic/borono ring path;
    # the Phase-9a fix was the acyclic CHAIN path only). The attach atom is at
    # match[0] (guarded == start_idx downstream). Each pattern matches ONLY the
    # true oxoacid (an -OH present), never a phosphine / sulfoxide / oxoacid ester.
    'sulfinic': Chem.MolFromSmarts('[SX3](=O)[OX2H1]'),
    'phosphonic': Chem.MolFromSmarts('[PX4](=O)([OX2H1])[OX2H1]'),
    'selenonic': Chem.MolFromSmarts('[SeX4](=O)(=O)[OX2H1]'),
    'telluronic': Chem.MolFromSmarts('[TeX4](=O)(=O)[OX2H1]'),
    # W3-P11 Se/Te -inic acids): the SeX3/TeX3 seleninic/tellurinic ring
    # oxoacids (R-Se(=O)-OH / R-Te(=O)-OH), the -inic analogues of the SeX4/TeX4
    # -onic acids above. The chain path already names ethaneseleninic acid; only
    # the benzene ring-suffix handler lacked the SeX3/TeX3 pattern -> the target
    # phenyl-Se(=O)-OH was 'unknown'. -> benzeneseleninic acid.
    'seleninic': Chem.MolFromSmarts('[SeX3](=O)[OX2H1]'),
    'tellurinic': Chem.MolFromSmarts('[TeX3](=O)[OX2H1]'),
}


def is_benzene_ring(mol, ring_atoms: Tuple[int, ...]) -> bool:
    """
    Check if a ring is a benzene ring (6-membered aromatic carbocycle).

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring

    Returns:
        True if ring is benzene (6 aromatic carbons)
    """
    if len(ring_atoms) != 6:
        return False

    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        # Must be carbon
        if atom.GetSymbol() != 'C':
            return False
        # Must be aromatic
        if not atom.GetIsAromatic():
            return False

    return True


def didehydro_benzene_name(mol, ring_atoms) -> Optional[str]:
    """ (Wave-2 completion): a bare benzene ring in which 2 (or 4)
    ring carbons carry NO hydrogen (RDKit perceives benzyne as an aromatic C6
    ring with a triple bond) is the didehydrobenzene parent:
    '1,2-didehydrobenzene' (BB verbatim; was 'benzene' -> unknown).

    Fail-closed: exactly 6 neutral ring carbons, no exocyclic heavy neighbour
    anywhere (substituted didehydrobenzenes are not built), every ring atom
    has 0 or 1 H, dehydro count in {2, 4}. Locants = the minimal tuple over
    all 12 ring walks.
    """
    if ring_atoms is None or len(ring_atoms) != 6:
        return None
    ring = list(ring_atoms)
    ring_set = set(ring)
    dehydro = []
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C' or atom.GetFormalCharge() != 0:
            return None
        if any(n.GetIdx() not in ring_set and n.GetAtomicNum() > 1
               for n in atom.GetNeighbors()):
            return None  # substituent -- class not built, stay fail-closed
        h = atom.GetTotalNumHs()
        if h == 0:
            dehydro.append(idx)
        elif h != 1:
            return None
    mult = {2: 'di', 4: 'tetra'}.get(len(dehydro))
    if mult is None:
        return None
    # Order the ring as a cycle walk.
    order = [ring[0]]
    prev = None
    cur = ring[0]
    while len(order) < 6:
        nxt = [n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
               if n.GetIdx() in ring_set and n.GetIdx() != prev]
        nxt = [i for i in nxt if i not in order]
        if not nxt:
            return None
        prev, cur = cur, nxt[0]
        order.append(cur)
    dehydro_set = set(dehydro)
    best = None
    for start in range(6):
        for step in (1, -1):
            locs = tuple(sorted(
                (pos % 6) + 1
                for pos, i in enumerate(range(0, 6 * step, step))
                if order[(start + i) % 6] in dehydro_set
            ))
            if best is None or locs < best:
                best = locs
    locant_str = ','.join(str(l) for l in best)
    return f"{locant_str}-{mult}dehydrobenzene"


def get_benzene_ring(mol) -> Optional[Tuple[int, ...]]:
    """
    Find the benzene ring in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of atom indices in the benzene ring, or None if not found
    """
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if is_benzene_ring(mol, ring):
            return ring
    return None


def get_benzene_substituents(mol, ring_atoms: Tuple[int, ...]) -> Dict[int, List[Dict]]:
    """
    Find substituents attached to each benzene carbon.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the benzene ring

    Returns:
        Dict mapping ring atom index to list of substituent info dicts.
        Each dict has keys: 'name' (str), 'atoms' (list of atom indices)
    """
    from ..perception.rings import get_containing_ring_system

    # Use the complete ring system as BFS boundary (IUPAC
    # Prevents walking into fused partner rings
    ring_set = set(get_containing_ring_system(mol, ring_atoms))
    substituents: Dict[int, List[Dict]] = defaultdict(list)

    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip ring atoms
            if nbr_idx in ring_set:
                continue

            # Identify the substituent
            sub_info = _identify_substituent(mol, nbr_idx, ring_set)
            if sub_info:
                substituents[ring_idx].append(sub_info)
            else:
                # Wave2 constitution-conservation guard: an exocyclic
                # branch NO recognizer could name used to be silently
                # dropped — the assembled name then described a different
                # molecule (bare 'benzoic acid' for the Ar-CH2-SiH2-CH2-Ar'
                # witness) and only the OPSIN-dependent oracle
                # caught it. Record an explicit unnameable sentinel: the
                # name assemblers decline (fail closed) when one is present,
                # and the atom-coverage accounting still sees the atoms.
                substituents[ring_idx].append({
                    'name': 'unknown',
                    'atoms': _bfs_substituent_atoms(mol, nbr_idx, ring_set),
                    'unnameable': True,
                })

    return dict(substituents)


def _bfs_substituent_atoms(mol, start_idx: int, ring_atoms: Set[int]) -> List[int]:
    """
    BFS to collect all atom indices in a substituent starting from start_idx.

    Does not constrain to pure alkyl -- collects all atoms outside ring_atoms.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the first atom in the substituent
        ring_atoms: Set of ring atom indices to exclude

    Returns:
        List of all atom indices in the substituent (including start_idx)
    """
    visited = {start_idx}
    queue = deque([start_idx])
    all_atoms = []

    while queue:
        current_idx = queue.popleft()
        all_atoms.append(current_idx)

        current_atom = mol.GetAtomWithIdx(current_idx)
        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_atoms:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    return all_atoms


def _identify_suffix_fg_on_benzene(
    mol, start_idx: int, sub_atoms: List[int], ring_atoms: Set[int]
) -> Optional[Dict]:
    """
    Identify suffix-type functional groups attached to benzene ring.

    Checks if the substituent starting at start_idx is a functional group
    that should be expressed as a suffix on the benzene parent name.

    Handles both C-based FGs (carboxylic acid, amide, aldehyde, nitrile, acid chloride)
    and S-based FGs (sulfonamide, sulfonic acid).

    Args:
        mol: RDKit Mol object
        start_idx: Index of the atom directly attached to ring
        sub_atoms: All atom indices in the substituent
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name', 'suffix_name', 'is_suffix', 'atoms', and optionally
        'n_substituents' for N-substituted amides; or None if not a suffix FG.
    """
    start_atom = mol.GetAtomWithIdx(start_idx)
    symbol = start_atom.GetSymbol()

    # Carbon-based suffix FGs
    if symbol == 'C':
        # Phase C: peroxy acid -C(=O)-O-OH -> '-carboperoxoic acid'.
        # the Blue Book verbatim: 'C6H5-CO-OOH benzenecarboperoxoic acid (PIN)
        # peroxybenzoic acid perbenzoic acid'. Before this, benzene had NO peroxy-acid
        # suffix form at all, so the group fell through and the molecule was named from
        # the OPSIN-import trivial name 'perbenzoic acid' -- itself doubly non-preferred
        # (BB prints the PIN beside it at:30178 and:29797, and the Blue Book states "The
        # prefix 'per-' is no longer recommended"). Denying that trivial name alone
        # produced an ABSTENTION, not the PIN -- session a project rule -- which is why the
        # suffix form had to be built rather than the name merely denied.
        # Seniority: SENIORITY_ORDER puts peroxy_acid at rank 1, directly after
        # carboxylic_acid at rank 0, and _SUFFIX_PRIORITY mirrors that.
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['peroxy_acid']):
            if match[0] == start_idx:
                return {
                    'name': 'carboperoxoic acid',
                    'suffix_name': 'carboperoxoic acid',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Carboxylic acid: C(=O)(OH) -- check before amide!
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['acid']):
            if match[0] == start_idx:
                return {
                    'name': 'carboxylic acid', 'suffix_name': 'carboxylic acid',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Hydrazonic acid: C(=N-NH2)(OH) -> '-carbohydrazonic acid'.
        # Checked BEFORE the amide/amidine blocks: =N-NH2 + -OH is an ACID (senior
        # to amide/amidine) and disjoint from their =O / single-N SMARTS.
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['hydrazonic_acid']):
            if match[0] == start_idx:
                return {
                    'name': 'carbohydrazonic acid',
                    'suffix_name': 'carbohydrazonic acid',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Imidic acid: C(=NH)(OH) -> '-carboximidic acid'. An
        # ACID (senior to amide/amidine); disjoint from their =O / -N SMARTS and
        # from the hydrazonic acid above (=N-NH2, not =NH).
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['imidic_acid']):
            if match[0] == start_idx:
                return {
                    'name': 'carboximidic acid',
                    'suffix_name': 'carboximidic acid',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Hydroxamic acid: C(=O)(NH-OH) — checked BEFORE primary/secondary amide
        # because N has H1 bonded to O (not H2 or NHC), so neither 'amide' nor
        # 'sec_amide' SMARTS would match. n_substituents=['hydroxy'] propagates
        # through the existing _name_substituted_benzamide machinery unchanged.
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['hydroxamic']):
            if match[0] == start_idx:
                return {
                    'name': 'carboxamide', 'suffix_name': 'carboxamide',
                    'is_suffix': True, 'atoms': sub_atoms,
                    'n_substituents': ['hydroxy'],
                }

        # C1: ring-attached hydrazide C(=O)-NH-NH2 -> '-carbohydrazide'.
        # Checked BEFORE the amide patterns: the hydrazide N is bonded to another
        # N so the amide SMARTS never match it, but the explicit ordering keeps
        # the intent clear. Emits e.g. 'benzene-1,4-dicarbohydrazide'.
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['hydrazide']):
            if match[0] == start_idx:
                return {
                    'name': 'carbohydrazide', 'suffix_name': 'carbohydrazide',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Primary amide: C(=O)(NH2)
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['amide']):
            if match[0] == start_idx:
                return {
                    'name': 'carboxamide', 'suffix_name': 'carboxamide',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Secondary amide: C(=O)(NHR)
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['sec_amide']):
            if match[0] == start_idx:
                n_subs = _detect_n_substituents(mol, match, ring_atoms)
                return {
                    'name': 'carboxamide', 'suffix_name': 'carboxamide',
                    'is_suffix': True, 'atoms': sub_atoms,
                    'n_substituents': n_subs,
                }

        # Tertiary amide: C(=O)(NR2)
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['tert_amide']):
            if match[0] == start_idx:
                n_subs = _detect_n_substituents(mol, match, ring_atoms)
                return {
                    'name': 'carboxamide', 'suffix_name': 'carboxamide',
                    'is_suffix': True, 'atoms': sub_atoms,
                    'n_substituents': n_subs,
                }

        # C2: ring-attached amidine -C(=N)N -> '-carboximidamide'.
        # Placed AFTER all amide/hydrazide blocks so a C(=O)N group is claimed as
        # the (senior) amide and never falls here; the =[NX2] in the SMARTS also
        # excludes C=O structurally. Defensive guanidine skip: a guanidino C
        # (N-C(=N)-N) attaches to the ring via N, so its C is never start_idx, but
        # guard anyway. N-substituted amidines route through the N-aware
        # substituent path (FIX c) when amidine is NOT the principal group; as a
        # ring SUFFIX the plain -carboximidamide base is emitted here and any
        # N/N'-substituents are carried on 'n_substituents'.
        # Wave2 T3d: ring-attached composite-N suffixes, tested BEFORE amidine
        # (more specific — amidine/hydrazide partially match these). Each is the
        # added-carbon carbo* form on the ring; N/N'-substituent citation is out
        # of scope here (fail-closed to the bare suffix, keeps honest).
        for _fg, _suffix in (
            ('hydrazidine_ring', 'carbohydrazonohydrazide'),
            ('hydrazonamide_ring', 'carbohydrazonamide'),
            ('thiohydrazide_ring', 'carbothiohydrazide'),
            # R7: thioamide AFTER thiohydrazide (thiohydrazide's
            # =S-N-N is more specific and would else be claimed as -carbothioamide).
            ('thioamide', 'carbothioamide'),
        ):
            for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS[_fg]):
                if match[0] == start_idx:
                    return {
                        'name': _suffix, 'suffix_name': _suffix,
                        'is_suffix': True, 'atoms': sub_atoms,
                    }

        _guanidine_patt = _compiled_smarts('[NX3][CX3](=[NX2])[NX3]')
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['amidine']):
            if match[0] != start_idx:
                continue
            # Skip if this C is a guanidine carbon (both flanking atoms are N).
            is_guanidino = any(
                gm[1] == start_idx
                for gm in mol.GetSubstructMatches(_guanidine_patt)
            )
            if is_guanidino:
                continue
            # D1 /: detect N/N'-substituents so the
            # already-wired _build_amidine_n_prefix (ring-suffix path) and the
            # demotion path can cite them. Fail-closed: on any non-nameable
            # N-substituent the detector returns None -> emit the bare suffix and
            # let keep it honest (never a wrong name).
            n_subs = _detect_amidine_n_substituents(mol, start_idx, ring_atoms)
            result = {
                'name': 'carboximidamide', 'suffix_name': 'carboximidamide',
                'is_suffix': True, 'atoms': sub_atoms,
            }
            if n_subs:
                result['n_substituents'] = n_subs
            # (BB 34490/34496, plan P1AM Task 6): ring-attached
            # -C(=NH)-NH-NH2 is 'hydrazinecarboximidoyl' (PIN) when demoted;
            # plain carbamimidoyl would DROP the hydrazino N (structure-wrong).
            # Detect: the amidine C's single-bonded N carries exactly one
            # terminal NH2 and no other heavy substituent -> record the swapped
            # prefix stem. A terminal-N branch PLUS anything else -> fail
            # closed (mark unnameable so the caller declines).
            _amc = start_idx  # match[0] == the amidine carbon
            _sgl = next(
                (nb.GetIdx() for nb in mol.GetAtomWithIdx(_amc).GetNeighbors()
                 if nb.GetSymbol() == 'N'
                 and mol.GetBondBetweenAtoms(
                     _amc, nb.GetIdx()).GetBondTypeAsDouble() == 1.0),
                None)
            if _sgl is not None:
                _sgl_heavy = [
                    nb.GetIdx()
                    for nb in mol.GetAtomWithIdx(_sgl).GetNeighbors()
                    if nb.GetIdx() != _amc and nb.GetAtomicNum() > 1]
                if (len(_sgl_heavy) == 1
                        and mol.GetAtomWithIdx(_sgl_heavy[0]).GetSymbol() == 'N'
                        and mol.GetAtomWithIdx(_sgl_heavy[0]).GetDegree() == 1):
                    #: 'hydrazinecarboximidoyl' is a compound
                    # substituent prefix -> enclose in parentheses so it
                    # alphabetises on its complete name and cites cleanly
                    # ('3-(hydrazinecarboximidoyl)benzoic acid').
                    result['demoted_prefix_override'] = '(hydrazinecarboximidoyl)'
                elif len(_sgl_heavy) >= 1 and any(
                        mol.GetAtomWithIdx(_h).GetSymbol() == 'N'
                        and mol.GetAtomWithIdx(_h).GetDegree() == 1
                        for _h in _sgl_heavy):
                    # hydrazino N present but with extra substitution -> the
                    # N/N-prime semantics of a decorated hydrazinecarboximidoyl
                    # are not built; fail closed rather than drop the N.
                    result['unnameable'] = True
            return result

        # Aldehyde: C(=O)H
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['aldehyde']):
            if match[0] == start_idx:
                return {
                    'name': 'carbaldehyde', 'suffix_name': 'carbaldehyde',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # C- (V-2): chalcogen aldehydes on the ring (-CH=S/Se/Te).
        # added-carbon suffixes carbothialdehyde/carboselenaldehyde/
        # carbotelluraldehyde. HEAD dropped the =chalcogen and read the carbon
        # as a methyl (-> 'methylbenzene', a different molecule).
        for _fg, _suffix in (
            ('thioaldehyde', 'carbothialdehyde'),
            ('selenoaldehyde', 'carboselenaldehyde'),
            ('telluroaldehyde', 'carbotelluraldehyde'),
        ):
            for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS[_fg]):
                if match[0] == start_idx:
                    return {
                        'name': _suffix, 'suffix_name': _suffix,
                        'is_suffix': True, 'atoms': sub_atoms,
                    }

        # Nitrile: C#N
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['nitrile']):
            if match[0] == start_idx:
                return {
                    'name': 'carbonitrile', 'suffix_name': 'carbonitrile',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Acid chloride: C(=O)Cl
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['acid_cl']):
            if match[0] == start_idx:
                return {
                    'name': 'carbonyl chloride', 'suffix_name': 'carbonyl chloride',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Thiocarboxylic S-acid: C(=O)(SH)
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['thio_acid']):
            if match[0] == start_idx:
                return {
                    'name': 'carbothioic S-acid', 'suffix_name': 'carbothioic S-acid',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

    # Sulfur-based suffix FGs
    if symbol == 'S':
        # Phase C tranche C: ring -SH -> '-thiol'. the Blue Book and the Blue Book BOTH print
        # 'C6H5-SH benzenethiol (PIN) (not thiophenol)'. We emitted 'sulfanylbenzene'
        # because benzene had no thiol SUFFIX form, so the SH was demoted to a
        # 'sulfanyl' prefix -- a missing suffix, not a locant defect, which is why the
        # (c) licence alone could never have produced the right name.
        # Placed FIRST in this branch but guarded on SX2H1, which is disjoint from every
        # oxidised-sulfur pattern below (all SX3/SX4).
        # Seniority: thiol is rank 94 in SENIORITY_ORDER, between the alcohols (88-91)
        # and the amines (102+); _SUFFIX_PRIORITY places it between 'ol' and 'amine'.
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['ring_thiol']):
            if match[0] == start_idx:
                return {
                    'name': 'thiol', 'suffix_name': 'thiol',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Wave2 completion: -S(=N-NH2)-NH-NH2 ->
        # '-sulfinohydrazonohydrazide' (BB verbatim benzene example). SX3 vs
        # SX4 keeps this disjoint from the sulfonohydrazide match below.
        for match in mol.GetSubstructMatches(
                _BENZENE_FG_SMARTS['sulfinohydrazonohydrazide']):
            if match[0] == start_idx:
                return {
                    'name': 'sulfinohydrazonohydrazide',
                    'suffix_name': 'sulfinohydrazonohydrazide',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # C1: ring-attached sulfonohydrazide S(=O)(=O)-NH-NH2 ->
        # '-sulfonohydrazide'. Checked BEFORE sulfonamide (the N-N form is more
        # specific; the sulfonamide SMARTS requires [NX3H2] so it never matches).
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['sulfonohydrazide']):
            if match[0] == start_idx:
                return {
                    'name': 'sulfonohydrazide', 'suffix_name': 'sulfonohydrazide',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Wave-2 P1AM, BB 34173): ring-attached sulfonimidamide
        # -S(=O)(=NH)-NH2 -> '-sulfonimidamide'. Checked BEFORE sulfonamide
        # (the imido =N makes it disjoint from the sulfonamide (=O)(=O) SMARTS).
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['sulfonimidamide']):
            if match[0] == start_idx:
                return {
                    'name': 'sulfonimidamide', 'suffix_name': 'sulfonimidamide',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Sulfonamide: S(=O)(=O)(NH2)
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['sulfonamide']):
            if match[0] == start_idx:
                return {
                    'name': 'sulfonamide', 'suffix_name': 'sulfonamide',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # F-B: N-substituted ring sulfonamide -S(=O)(=O)-NR'R''.
        # Detected AFTER the primary (disjoint NX3 H-count). The italic-N
        # substituents are carried on 'n_substituents' and merged with the ring
        # locants by _name_substituted_benzenesulfonamide, exactly as the amide
        # path does for N,4-dimethylbenzamide. If the N-shape is outside the class
        # (ring N / heteroatom N-substituent / unnameable branch) the detector
        # returns None and this FG is marked unnameable so the whole name fails
        # closed rather than dropping the N-substituent's atoms.
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['n_sub_sulfonamide']):
            if match[0] == start_idx:
                n_subs = _detect_sulfonamide_n_substituents(mol, start_idx, ring_atoms)
                if n_subs is None:
                    return {
                        'name': 'sulfonamide', 'suffix_name': 'sulfonamide',
                        'is_suffix': True, 'atoms': sub_atoms, 'unnameable': True,
                    }
                return {
                    'name': 'sulfonamide', 'suffix_name': 'sulfonamide',
                    'is_suffix': True, 'atoms': sub_atoms, 'n_substituents': n_subs,
                }

        # Task Y: sulfinamide S(=O)(NH2) -- checked after sulfonamide; the SX3/SX4
        # split means neither can steal the other's match.
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['sulfinamide']):
            if match[0] == start_idx:
                return {
                    'name': 'sulfinamide', 'suffix_name': 'sulfinamide',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Sulfonic acid: S(=O)(=O)(OH)
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['sulfonic']):
            if match[0] == start_idx:
                return {
                    'name': 'sulfonic acid', 'suffix_name': 'sulfonic acid',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Sulfinic acid: S(=O)(OH) -- demotes to 'sulfino' (D-FOLLOWON item 5,
        # the ring analogue of the chain sulfinic-locant item 1).
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['sulfinic']):
            if match[0] == start_idx:
                return {
                    'name': 'sulfinic acid', 'suffix_name': 'sulfinic acid',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

    # Phosphorus / selenium / tellurium ring oxoacids (D-FOLLOWON item 5,
    # /. Demote to phosphono / selenono / tellurono when a
    # senior carboxylic acid is the principal group; emit benzenephosphonic /
    # benzeneselenonic / benzenetelluronic acid as the bare principal suffix.
    if symbol in ('P', 'Se', 'Te'):
        for _sym, _fg, _suffix in (
            ('P', 'phosphonic', 'phosphonic acid'),
            ('Se', 'selenonic', 'selenonic acid'),
            # W3-P11: SeX3/TeX3 -inic ring oxoacids. Ordered AFTER the
            # -onic (SeX4/TeX4) pattern so a genuine SeX4 acid matches -onic first
            # (the SeX3 SMARTS cannot match an SeX4 atom anyway; belt-and-braces).
            ('Se', 'seleninic', 'seleninic acid'),
            ('Te', 'telluronic', 'telluronic acid'),
            ('Te', 'tellurinic', 'tellurinic acid'),
        ):
            if symbol != _sym:
                continue
            for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS[_fg]):
                if match[0] == start_idx:
                    return {
                        'name': _suffix, 'suffix_name': _suffix,
                        'is_suffix': True, 'atoms': sub_atoms,
                    }

    return None


def _detect_n_substituents(mol, amide_match: Tuple[int, ...], ring_atoms: Set[int]) -> List[str]:
    """
    Detect N-alkyl substituents on an amide nitrogen.

    Args:
        mol: RDKit Mol object
        amide_match: SMARTS match tuple (C, O/N indices depending on pattern)
        ring_atoms: Set of ring atom indices

    Returns:
        List of alkyl names attached to nitrogen (e.g., ['methyl'] or ['methyl', 'methyl'])
    """
    # Find the nitrogen atom in the amide
    c_idx = amide_match[0]
    c_atom = mol.GetAtomWithIdx(c_idx)

    n_atom = None
    for nbr in c_atom.GetNeighbors():
        if nbr.GetSymbol() == 'N' and nbr.GetIdx() not in ring_atoms:
            n_atom = nbr
            break

    if n_atom is None:
        return []

    n_idx = n_atom.GetIdx()
    alkyl_names = []

    for nbr in n_atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == c_idx or nbr_idx in ring_atoms:
            continue
        if nbr.GetSymbol() == 'C':
            # Collect alkyl chain
            alkyl_atoms, carbon_count = _collect_pure_alkyl(
                mol, nbr_idx, ring_atoms | {n_idx, c_idx}
            )
            if alkyl_atoms is not None and carbon_count > 0:
                from ..assembly.substituent_naming import name_substituent_fragment
                rec_name = name_substituent_fragment(
                    mol, alkyl_atoms, nbr_idx, list(ring_atoms | {n_idx, c_idx})
                )
                if rec_name:
                    alkyl_names.append(rec_name)
                else:
                    try:
                        alkyl_names.append(get_alkyl_name(carbon_count))
                    except (ValueError, KeyError):
                        pass

    alkyl_names.sort()
    return alkyl_names


def _detect_sulfonamide_n_substituents(
    mol, s_idx: int, ring_atoms: Set[int]
) -> Optional[List[str]]:
    """N-substituent names on a sulfonamide nitrogen, or None to FAIL CLOSED.

    F-B. The amide detector above walks from the carbonyl
    carbon; a sulfonamide N hangs off the SULFONYL sulfur, so this is its
    S-rooted twin. It returns None (never a partial list) whenever the shape is
    outside the class the merge is built for, so ``name_substituted_benzene``
    marks the substituent ``unnameable`` and declines rather than dropping an
    N-substituent's atoms:

    * the N is in a ring (a sultam is a ring parent, 1-(arenesulfonyl)…), or
      charged / isotopically labelled;
    * an N-substituent is a heteroatom (N-hydroxy / N-amino / N-N is
       territory, a different construction);
    * an N-substituent branch loops back into the ring or the sulfur;
    * a branch cannot be named by the shared fragment namer;
    * more than two N-substituents (a trivalent amide N cannot carry three).

    An unsubstituted N (primary sulfonamide) returns None too — the primary
    suffix path names it and this twin must not compete.
    """
    from ..assembly.substituent_naming import name_substituent_fragment

    s_atom = mol.GetAtomWithIdx(s_idx)
    n_idx = next(
        (nb.GetIdx() for nb in s_atom.GetNeighbors() if nb.GetAtomicNum() == 7),
        None,
    )
    if n_idx is None:
        return None
    n_atom = mol.GetAtomWithIdx(n_idx)
    if n_atom.IsInRing() or n_atom.GetFormalCharge() != 0 or n_atom.GetIsotope():
        return None

    blocked = set(ring_atoms) | {n_idx, s_idx}
    names: List[str] = []
    for nb in n_atom.GetNeighbors():
        nb_idx = nb.GetIdx()
        if nb_idx == s_idx:
            continue
        if nb.GetAtomicNum() != 6:
            return None
        branch = _bfs_substituent_atoms(mol, nb_idx, blocked)
        if s_idx in branch or any(a in ring_atoms for a in branch):
            return None
        nm = name_substituent_fragment(
            mol, branch, nb_idx, [n_idx, s_idx] + list(ring_atoms)
        )
        if not nm:
            return None
        names.append(nm)

    if not names or len(names) > 2:
        return None
    return names


def _identify_boron_group(
    mol, start_idx: int, ring_atoms: Set[int]
) -> Optional[Dict]:
    """Identify a ``-B(OH)2`` boronic-acid substituent on a ring as the
    preselected ``borono`` prefix.

    Graph classifier (NOT SMARTS broadening, per feedback_smarts_and_seniority):
    a neutral three-coordinate boron bonded to exactly one ring atom and two
    TERMINAL hydroxy oxygens (-OH), nothing else. The multivalent boranediyl /
    dimethylboranyl forms are out of scope and fail closed to
    None (the borono ring-propagation here is the Phase-8 target). Returns the
    prefix dict, or None.
    """
    b = mol.GetAtomWithIdx(start_idx)
    if b.GetSymbol() != 'B' or b.GetFormalCharge() != 0 or b.GetDegree() != 3:
        return None
    ring_nbrs = 0
    oh_nbrs = 0
    for nbr in b.GetNeighbors():
        if nbr.GetIdx() in ring_atoms:
            ring_nbrs += 1
            continue
        bond = mol.GetBondBetweenAtoms(start_idx, nbr.GetIdx())
        if (nbr.GetSymbol() == 'O' and nbr.GetFormalCharge() == 0
                and nbr.GetDegree() == 1 and nbr.GetTotalNumHs() == 1
                and bond is not None
                and bond.GetBondType() == Chem.BondType.SINGLE):
            oh_nbrs += 1
        else:
            return None  # any other neighbour (alkyl/=O/chalcogen) -> not borono
    if ring_nbrs == 1 and oh_nbrs == 2:
        sub_atoms = _bfs_substituent_atoms(mol, start_idx, ring_atoms)
        return {'name': 'borono', 'atoms': sub_atoms}
    return None


def _identify_ring_heteroatom_fg(mol, start_idx: int,
                                 ring_atoms: Set[int]) -> Optional[Dict]:
    """Wave-2 completion ring-FG root): recognize a small closed set of
    hypervalent-iodine / oxophosphanyl ring substituents that the plain-symbol
    branches would otherwise mis-name or drop. Returns the prefix dict or None
    (fail-closed -> the caller continues its normal per-symbol dispatch).

    Handled (all OPSIN-RT verified):
      * iodine with one terminal =O -> 'iodosyl'
      * iodine with two terminal =O -> 'iodyl'
      * phosphorus with one terminal =O (bare -PH=O) -> 'oxophosphanyl'
    """
    atom = mol.GetAtomWithIdx(start_idx)
    sym = atom.GetSymbol()
    if sym not in ('I', 'P'):
        return None
    if atom.GetFormalCharge() != 0:
        return None
    oxo = []
    other_heavy = 0
    for nbr in atom.GetNeighbors():
        if nbr.GetIdx() in ring_atoms:
            continue
        bond = mol.GetBondBetweenAtoms(start_idx, nbr.GetIdx())
        if (nbr.GetSymbol() == 'O' and nbr.GetDegree() == 1
                and bond.GetBondType() == Chem.BondType.DOUBLE):
            oxo.append(nbr.GetIdx())
        else:
            other_heavy += 1
    if other_heavy != 0:
        return None  # any extra substituent -> not this clean class
    if sym == 'I':
        if len(oxo) == 1:
            return {'name': 'iodosyl', 'atoms': [start_idx] + oxo}
        if len(oxo) == 2:
            return {'name': 'iodyl', 'atoms': [start_idx] + oxo}
    elif sym == 'P':
        # bare -P(=O)H2 attached to the ring -> oxophosphanyl
        if len(oxo) == 1:
            return {'name': 'oxophosphanyl', 'atoms': [start_idx] + oxo}
    return None


def _nitrile_oxide_prefix(mol, start_idx: int,
                          ring_atoms: Set[int]) -> Optional[Dict]:
    """ lambda-branch: ring substituent -C#[N+]-[O-] (nitrile
    oxide, ON#C-) -> preferred prefix '(oxo-λ5-azanylidyne)methyl'
    (not isofulminato).

    Three-state return:
      * prefix dict — pattern matched AND the fragment being named is an
        ANION (net formal charge < 0): class-2 anion parent outranks the
        zwitterionic nitrile oxide, so the prefix form is the PIN (BB
        'sodium 4-[(oxo-λ5-azanylidyne)methyl]benzoate', 34897).
      * {'name': None} — pattern matched in a NON-anion context: the nitrile
        oxide itself is senior zwitterion) and the PIN is the
        functional-class '...nitrile oxide' SUFFIX form, unbuilt -> the
        caller must FAIL CLOSED (today's walker silently mis-names the group).
      * None — pattern absent: continue normal dispatch.
    """
    atom = mol.GetAtomWithIdx(start_idx)
    if atom.GetSymbol() != 'C' or atom.GetFormalCharge() != 0:
        return None
    if atom.GetTotalNumHs() != 0:
        return None
    triple_n = None
    for nbr in atom.GetNeighbors():
        if nbr.GetIdx() in ring_atoms:
            continue
        bond = mol.GetBondBetweenAtoms(start_idx, nbr.GetIdx())
        if (nbr.GetSymbol() == 'N' and nbr.GetFormalCharge() == 1
                and bond.GetBondType() == Chem.BondType.TRIPLE):
            if triple_n is not None:
                return None
            triple_n = nbr
        else:
            return None  # any other exocyclic decoration -> not this class
    if triple_n is None:
        return None
    o_minus = [n for n in triple_n.GetNeighbors()
               if n.GetIdx() != start_idx]
    if (len(o_minus) != 1 or o_minus[0].GetSymbol() != 'O'
            or o_minus[0].GetFormalCharge() != -1
            or o_minus[0].GetDegree() != 1):
        return None
    if Chem.GetFormalCharge(mol) < 0:  # anion context (see investigation)
        from .lambda_convention import LAMBDA
        return {
            'name': f'(oxo-{LAMBDA}5-azanylidyne)methyl',
            'atoms': [start_idx, triple_n.GetIdx(), o_minus[0].GetIdx()],
            'is_complex': True,
        }
    return {'name': None}


_CHALCOGEN_LINK_SYMBOLS = frozenset(('O', 'S', 'Se', 'Te'))


def _chalcogen_linked_carbon_substituent(
    mol, e_idx: int, ring_atoms: Set[int]
) -> Optional[Dict]:
    """R-E- on the ring, E a neutral divalent chalcogen (O, S, Se, Te) joined by
    single bonds to the ring and to one carbon atom of R.

     (the Blue Book), method (1) (:27808): "by prefixing the names of
    the substituent groups R'-S-, R'-Se-, or R'-Te-, i.e., R'-sulfanyl, R'-selanyl,
    and R'-tellanyl, respectively, to that of the parent hydride, RH"; "Method (1),
    substitutive nomenclature, gives preferred IUPAC names" (:27817).
    '(cyclopentylselanyl)benzene (PIN)' (:27834), '1-chloro-4-[(chloromethyl)
    selanyl]benzene (PIN)' (:27850); for R-O-, '1-(chloromethoxy)-4-nitrobenzene
    (PIN)',:27711). The prefix is R's substituent name with the
    chalcogen's linker, built by the shared substituent namer from the whole
    R-E- fragment, and it is enclosed by the rule (:7232) "Parentheses
    are used around compound... and complex... prefixes".

    The dedicated O and S branches build the plain-alkyl R (and the other R-O-
    shapes they recognise) first; this covers the R they decline and every R-Se- /
    R-Te-. Declines (None) when R holds an aromatic atom: a second benzene ring in R
    raises the multiplicative choice (1,1'-sulfanediyldibenzene (PIN),
    :27826), which the multiplicative producer decides, not this collector.
    """
    e_atom = mol.GetAtomWithIdx(e_idx)
    if (e_atom.GetSymbol() not in _CHALCOGEN_LINK_SYMBOLS
            or e_atom.GetFormalCharge() != 0
            or e_atom.GetDegree() != 2
            or e_atom.GetTotalNumHs() != 0
            or e_atom.GetNumRadicalElectrons() != 0):
        return None
    if any(b.GetBondType() != Chem.BondType.SINGLE for b in e_atom.GetBonds()):
        return None
    others = [n for n in e_atom.GetNeighbors() if n.GetIdx() not in ring_atoms]
    if len(others) != 1 or others[0].GetSymbol() != 'C':
        return None
    sub_atoms = _bfs_substituent_atoms(mol, e_idx, ring_atoms)
    if any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in sub_atoms):
        return None
    from ..assembly.naming_utils import enclose_if_compound
    from ..assembly.substituent_naming import name_substituent_fragment
    try:
        name = name_substituent_fragment(mol, sub_atoms, e_idx, list(ring_atoms))
    except Exception:
        return None
    if not name or name == 'substituent' or ' ' in name:
        return None
    enclosed = enclose_if_compound(name)
    return {'name': enclosed, 'atoms': sub_atoms, 'is_complex': enclosed != name}


def _identify_substituent(mol, start_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify a substituent starting from an atom attached to the ring.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the atom attached to the ring
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name' and 'atoms', or None if unknown
    """
    start_atom = mol.GetAtomWithIdx(start_idx)
    symbol = start_atom.GetSymbol()

    # Wave-2 completion ring-FG root): explicit hypervalent-iodine and
    # oxophosphanyl ring substituents, recognized BEFORE the plain-halogen
    # branch so a =O-bearing iodine is not mis-emitted as bare 'iodo' (which
    # would drop the oxide and name a different molecule). O=I- -> iodosyl,
    # O=I(=O)- -> iodyl; O=PH- -> oxophosphanyl. All
    # OPSIN-RT verified; fail-closed (fall through) on any other decoration.
    _ring_fg = _identify_ring_heteroatom_fg(mol, start_idx, ring_atoms)
    if _ring_fg is not None:
        return _ring_fg

    #: nitrile oxide -C#[N+]-[O-]. Must run BEFORE the generic C
    # branches, which mis-name the group (silent structure loss). Prefix form
    # is PIN only in anion context; neutral context fails closed.
    _no = _nitrile_oxide_prefix(mol, start_idx, ring_atoms)
    if _no is not None:
        return _no if _no.get('name') else None

    # Halogens - single atom substituents
    if symbol in SUBSTITUENT_PREFIXES:
        return {
            'name': SUBSTITUENT_PREFIXES[symbol],
            'atoms': [start_idx]
        }

    # For C, S, and the P/Se/Te oxoacid hubs, check suffix FGs first. P/Se/Te
    # added by D-FOLLOWON item 5 so a ring phosphonic/selenonic/telluronic acid is
    # recognized BEFORE the phosphanyl P branch / Se-Te fall-through-None below
    # (genuine phosphines etc. don't match the oxoacid SMARTS -> still fall through).
    if symbol in ('C', 'S', 'P', 'Se', 'Te'):
        sub_atoms = _bfs_substituent_atoms(mol, start_idx, ring_atoms)
        suffix_fg = _identify_suffix_fg_on_benzene(mol, start_idx, sub_atoms, ring_atoms)
        if suffix_fg:
            return suffix_fg

    # DD2 Fix B (Phase D, (1)): a peroxy (-O-O-R) / disulfanyl (-S-S-R) /
    # thioperoxol (-S-O-R, -O-S-R) substituent on the ring. The attach atom is a
    # divalent (all-single-bond, neutral) chalcogen bonded outside the ring to a
    # second divalent chalcogen — the peroxide/disulfide linkage. Route through the
    # shared substituent namer ((R)peroxy / (R)disulfanyl) instead of dropping it
    # (peroxide -> 'benzene') or collapsing it to a bare sulfanyl (disulfide ->
    # 'sulfanylbenzene'). The divalent-single-bond guard excludes sulfinyl/sulfonyl.
    if symbol in ('O', 'S') and start_atom.GetFormalCharge() == 0 and all(
        b.GetBondType() == Chem.BondType.SINGLE for b in start_atom.GetBonds()
    ):
        _partner = None
        for nbr in start_atom.GetNeighbors():
            if nbr.GetIdx() in ring_atoms:
                continue
            if (nbr.GetSymbol() in ('O', 'S') and nbr.GetFormalCharge() == 0 and all(
                b.GetBondType() == Chem.BondType.SINGLE for b in nbr.GetBonds()
            )):
                _partner = nbr
                break
        if _partner is not None:
            _sub_atoms = _bfs_substituent_atoms(mol, start_idx, ring_atoms)
            from ..assembly.naming_utils import needs_brackets
            from ..assembly.substituent_enumerator import name_substituent
            _pname = name_substituent(mol, set(_sub_atoms), start_idx)
            if _pname and _pname not in ("substituent", "sulfanyl", "hydroxy"):
                if needs_brackets(_pname) and not (
                    _pname.startswith('(') and _pname.endswith(')')):
                    _pname = f'({_pname})'
                return {'name': _pname, 'atoms': _sub_atoms, 'is_complex': True}

    # Nitrogen-based groups
    if symbol == 'N':
        return _identify_nitrogen_group(mol, start_idx, ring_atoms)

    # Oxygen-based groups
    if symbol == 'O':
        return _identify_oxygen_group(mol, start_idx, ring_atoms)

    # Sulfur-based groups (non-suffix fallback)
    if symbol == 'S':
        return _identify_sulfur_group(mol, start_idx, ring_atoms)

    # Selanyl / tellanyl ethers R-Se- / R-Te- method (1)); the oxoacid
    # suffixes on Se/Te were recognised above.
    if symbol in ('Se', 'Te'):
        return _chalcogen_linked_carbon_substituent(mol, start_idx, ring_atoms)

    # Boron-based groups: the borono prefix -B(OH)2 preselected
    # substituent prefix). HEAD dropped the whole boron unit on ring parents
    # (4-boronobenzoic acid -> 'benzoic acid', a different molecule) because no
    # 'B' branch existed here — only the acyclic chain path emitted borono.
    if symbol == 'B':
        boron_result = _identify_boron_group(mol, start_idx, ring_atoms)
        if boron_result:
            return boron_result

    # Group-14 silyl/germyl substituents: -SiH3 -> silyl, -Si(OH)3 ->
    # trihydroxysilyl, -Si(CH3)3 -> trimethylsilyl. HEAD had no 'Si'/'Ge' branch
    # here, so the whole silyl/germyl unit was dropped on ring parents
    # (4-silylbenzoic acid -> 'benzoic acid', a different molecule). Routes through
    # the shared _name_group14_substituent (fail-closed; multivalent / complex
    # centres -> None -> fall through unchanged).
    if symbol in ('Si', 'Ge'):
        from ..assembly.substituent_naming import _name_group14_substituent
        sub_atoms = _bfs_substituent_atoms(mol, start_idx, ring_atoms)
        g14 = _name_group14_substituent(mol, sub_atoms, start_idx)
        if g14:
            # Enclosing marks are added downstream by the benzene assembler via
            # is_complex_substituent (silyl-with-prefixes is now complex).
            return {'name': g14, 'atoms': sub_atoms}

    # Phosphorus-based groups: generate phosphanyl prefix (IUPAC
    if symbol == 'P':
        from ..rules.phosphorus import (
            get_phosphanyl_prefix, name_acyl_prefix_substituent)
        sub_atoms = _bfs_substituent_atoms(mol, start_idx, ring_atoms)
        # a phase: a P ACYL group (a =E'-bearing P, e.g.
        # -P(=S)Me2) is cited as the acyl prefix
        # ('dimethylphosphinothioyl'), NOT the trivalent 'phosphanyl' — which
        # would silently drop the =E' chalcogen. Consumes the shared table; the
        # recognizer returns None for a plain trivalent P, so phosphanyl below
        # still owns -PR2. Fail-closed.
        acyl_prefix = name_acyl_prefix_substituent(mol, sub_atoms, start_idx)
        if acyl_prefix:
            return {'name': acyl_prefix, 'atoms': sub_atoms}
        # Exclude parent ring atoms from P's substituent count so the
        # attachment carbon is not counted (e.g., PPh2 on benzene -> diphenylphosphanyl)
        prefix = get_phosphanyl_prefix(mol, start_idx, exclude_atoms=ring_atoms)
        if prefix:
            return {'name': prefix, 'atoms': sub_atoms}

    # W3-P10: arsenic-based substituent -As(OH)2 -> dihydroxyarsanyl,
    # cited as a prefix when a senior organic group (-COOH) is the parent PCG. HEAD
    # had no 'As' branch, so the whole As unit fell through unnamed and the molecule
    # failed closed to 'inorganic compound (not supported)'. Fail-closed (falls
    # through) for any decorated-As shape the arsanyl namer declines.
    if symbol in ('As', 'Sb', 'Bi'):
        from ..rules.mononuclear_hydrides import name_arsanyl_substituent
        from ..rules.phosphorus import name_acyl_prefix_substituent
        sub_atoms = _bfs_substituent_atoms(mol, start_idx, ring_atoms)
        # a phase: an As/Sb ACYL group (=E'-bearing, e.g.
        # -As(=O)(OH)) is cited as the acyl prefix ('hydroxyarsoryl'),
        # not the trivalent 'arsanyl' — which would drop the =E'. Fail-closed; a
        # plain trivalent As/Sb (no =E') falls through to the arsanyl namer.
        acyl_prefix = name_acyl_prefix_substituent(mol, sub_atoms, start_idx)
        if acyl_prefix:
            return {'name': acyl_prefix, 'atoms': sub_atoms}
        as_name = name_arsanyl_substituent(mol, sub_atoms, start_idx)
        if as_name:
            return {'name': as_name, 'atoms': sub_atoms}

    # Carbon-based groups (alkyl or functionalized chain) - fallback for non-suffix C
    if symbol == 'C':
        # Check for nitrile C#N pattern FIRST (fix)
        nitrile_result = _identify_nitrile_group(mol, start_idx, ring_atoms)
        if nitrile_result:
            return nitrile_result

        # +: a co-present ESTER decoration -C(=O)-O-R on a
        # benzene whose PRINCIPAL group is a senior ACID demotes to the
        # alkoxycarbonyl prefix (methoxycarbonyl / ethoxycarbonyl /
        # phenoxycarbonyl). Acids are class 7 in the seniority order
        # (:18170), esters class 9 (:18182), so a free acid always owns the
        # suffix word ("an acid is senior to an ester",:36540). The
        # aliphatic-ring collector already does this
        # (rules/ring_substituents.py:3726); the benzene collector had NO
        # ester branch, so '2-(methoxycarbonyl)benzoic acid' fell to the
        # carbonyl-guarded universal fallback (:1417) -> 'unnameable'
        # sentinel -> abstain. Gated on a co-present acid so it never fires
        # when the ester itself is (or would be) the PCG (methyl benzoate,
        # dimethyl phthalate); fail-closed via get_alkoxycarbonyl_prefix
        # (aryl/branched/hetero R -> None -> fall through unchanged).
        ester_prefix = _identify_ester_as_alkoxycarbonyl(
            mol, start_idx, ring_atoms)
        if ester_prefix is not None:
            return ester_prefix

        # Try simple alkyl
        alkyl_result = _identify_alkyl_group(mol, start_idx, ring_atoms)
        if alkyl_result:
            return alkyl_result

        # Wave2: ketone-acyl branch DEMOTED to a retained/
        # systematic acyl prefix — ONLY when another branch on this ring bears
        # a SENIOR suffix FG (acid/amide/nitrile/... — all senior to ketone,
        #, so the ketone can never be the PCG here: 4-acetylbenzoic
        # acid, 4-acetylbenzamide. Without a senior suffix (acetophenone
        # class) the gate is False and the chain-parent ketone PIN
        # (1-phenylethan-1-one) keeps ownership. Fail-closed: branched/
        # unsaturated/hetero acyls return None -> unnameable sentinel.
        if _ring_bears_senior_suffix(mol, ring_atoms, start_idx):
            acyl_result = _identify_acyl_group(mol, start_idx, ring_atoms)
            if acyl_result:
                return acyl_result

        # Try functionalized chain (chains with FG like -CCCC(=O)O)
        func_chain = _identify_functionalized_chain(mol, start_idx, ring_atoms)
        if func_chain:
            return func_chain

        # Fallback: collect all atoms and produce a generic substituent name
        # Handles haloalkyl groups like CF3 (trifluoromethyl)
        sub_atoms = _bfs_substituent_atoms(mol, start_idx, ring_atoms)
        generic = _identify_generic_carbon_substituent(mol, start_idx, sub_atoms, ring_atoms)
        if generic:
            return generic

        # a phase: Universal pipeline fallback for complex C-substituents.
        # By this point, suffix FG detection (line 387-389) has already returned
        # for recognized suffix patterns (acid, amide, nitrile, aldehyde, etc.).
        # We only reach here for C-substituents that are NOT suffix FGs and NOT
        # simple alkyls/haloalkyls -- e.g., cyanomethyl, carbamoylmethyl, etc.
        #
        # Guards:
        # 1. Carbonyl guard: if start C has C=O, it's a carbonyl carbon
        # (ester C(=O)OR, ketone C(=O)R). These are FG features handled
        # by dedicated handlers; naming them as benzene substituents causes
        # routing regressions. All suffix-type carbonyls (acid, amide,
        # aldehyde, acid chloride) are already caught by suffix FG check.
        # 2. Size guard: fragments larger than 10 atoms are major structural
        # features (fused systems, long chains), not genuine substituents.
        if sub_atoms:
            # Check if start carbon is a carbonyl (has C=O bond)
            _is_carbonyl = False
            for _nbr in start_atom.GetNeighbors():
                if _nbr.GetIdx() in ring_atoms:
                    continue
                if _nbr.GetSymbol() == 'O':
                    _bond = mol.GetBondBetweenAtoms(start_idx, _nbr.GetIdx())
                    if _bond and _bond.GetBondType() == Chem.BondType.DOUBLE:
                        _is_carbonyl = True
                        break

            if not _is_carbonyl:
                _MAX_FALLBACK_ATOMS = 10
                if len(sub_atoms) <= _MAX_FALLBACK_ATOMS:
                    from ..assembly.naming_utils import needs_brackets
                    from ..assembly.substituent_enumerator import name_substituent
                    prefix_name = name_substituent(mol, set(sub_atoms), start_idx)
                    if prefix_name and prefix_name != "substituent":
                        # Wrap MARKLESS compound names in parentheses per IUPAC
                        #. w2f p1 (research C.6): a name already
                        # carrying an enclosing mark ('(4-bromophenyl)(chloro)
                        # methyl') must NOT take plain parens here — the naive
                        # startswith('(')/endswith(')') test misfires on
                        # interior/trailing-stem marks (the non-PIN '((...)...)'
                        # parens-in-parens). Left BARE, the escalating consumer
                        # (format_substituent_prefix / _omit branch) applies the
                        # brackets '[(4-bromophenyl)(chloro)methyl]'.
                        is_compound = needs_brackets(prefix_name)
                        if (is_compound and '(' not in prefix_name
                                and '[' not in prefix_name):
                            # Lane L2 (R2): the marks keep the writer's record
                            from ..assembly.prefix_derivation import carried
                            prefix_name = carried(f'({prefix_name})', like=prefix_name)
                        return {
                            'name': prefix_name,
                            'atoms': sub_atoms,
                            'is_complex': is_compound,
                        }

    return None


def _ring_bears_senior_suffix(
    mol, ring_atoms: Set[int], exclude_branch_start: int
) -> bool:
    """True iff ANOTHER exocyclic branch of this ring is a suffix-forming FG
    (carboxylic acid / carboxamide / carbonitrile / sulfonic acid ... — every
    record `_identify_suffix_fg_on_benzene` emits is senior to ketone, P-41).
    Gates the Wave2 T5c acyl-prefix demotion so a lone aryl ketone (the PCG
    case, acetophenone class) never routes here.

    Scoped to C-ATTACHED suffix branches (carboxylic acid / carboxamide /
    carbonitrile / carbaldehyde ...): the S-suffix assembly currently drops
    ring locants when combined with an acyl prefix ('acetylbenzenesulfonic
    acid', wrong constitution), so acetyl+sulfonic stays on its HEAD path
    (fail-closed) until that assembly is fixed — documented deferral."""
    for ra in ring_atoms:
        for nbr in mol.GetAtomWithIdx(ra).GetNeighbors():
            ni = nbr.GetIdx()
            if ni in ring_atoms or nbr.GetAtomicNum() <= 1:
                continue
            if ni == exclude_branch_start:
                continue
            if nbr.GetSymbol() != 'C':
                continue
            sub_atoms = _bfs_substituent_atoms(mol, ni, ring_atoms)
            if _identify_suffix_fg_on_benzene(mol, ni, sub_atoms, ring_atoms):
                return True
    return False


def _ring_bears_acid_suffix(
    mol, ring_atoms: Set[int], exclude_branch_start: int
) -> bool:
    """True iff ANOTHER exocyclic branch of this ring is a suffix-forming FG
    of the ACID class (carboxylic / sulfonic / sulfinic / phosphonic /
    selenonic / carbothioic S-acid / carboperoxoic... — every acid the
    benzene suffix detector emits carries the word 'acid' in its
    ``suffix_name``).

    Narrower than ``_ring_bears_senior_suffix`` on purpose: that helper gates
    the KETONE-acyl demotion and fires on ANY suffix (amide / nitrile /
    aldehyde included), which is correct because a ketone is junior to all of
    them. An ESTER, by contrast, is class 9 in 's order (:18182) and is
    SENIOR to amides (11), nitriles (12) and aldehydes (13) — it only demotes
    to a prefix when a class-7 ACID (:18170) owns the parent word ("an acid is
    senior to an ester",:36540). So the ester-demotion gate must see an acid
    specifically, not merely any suffix."""
    for ra in ring_atoms:
        for nbr in mol.GetAtomWithIdx(ra).GetNeighbors():
            ni = nbr.GetIdx()
            if ni in ring_atoms or nbr.GetAtomicNum() <= 1:
                continue
            if ni == exclude_branch_start:
                continue
            if nbr.GetSymbol() not in ('C', 'S', 'P', 'Se', 'Te'):
                continue
            sub_atoms = _bfs_substituent_atoms(mol, ni, ring_atoms)
            rec = _identify_suffix_fg_on_benzene(mol, ni, sub_atoms, ring_atoms)
            if rec and 'acid' in (rec.get('suffix_name') or rec.get('name') or ''):
                return True
    return False


def _identify_ester_as_alkoxycarbonyl(
    mol, start_idx: int, ring_atoms: Set[int]
) -> Optional[Dict]:
    """Recognize a ring-attached ester branch -C(=O)-O-R and, when a senior
    ACID owns the ring's suffix word, return its alkoxycarbonyl PREFIX
    record ('methoxycarbonyl', 'ethoxycarbonyl', 'phenoxycarbonyl',
    '(benzyloxy)carbonyl'; IUPAC.

    Anchored on the KNOWN ester carbonyl carbon (never a scanned atom set):
    the attach carbon must carry exactly one terminal C=O and exactly one
    single-bonded -O- that goes onward to a carbon (the OR alkyl), with no
    other exocyclic neighbour. That shape excludes free acids (-OH, an H on
    the second O -> caught by the carboxy suffix branch above), ketones/acyls
    (no second O), aldehydes and amides. Fail-closed (None) for any other
    decoration, and for aryl/branched/hetero R the shared
    ``get_alkoxycarbonyl_prefix`` producer returns None so no partial name
    leaks. The returned 'atoms' cover the ENTIRE ester unit (carbonyl C, both
    O, and the whole R fragment) so the exact-coverage gate accounts for
    every atom."""
    start_atom = mol.GetAtomWithIdx(start_idx)
    if start_atom.GetSymbol() != 'C' or start_atom.GetFormalCharge() != 0:
        return None
    if start_atom.IsInRing():
        return None

    carbonyl_o = None
    ester_o = None
    alkyl_c = None
    for nbr in start_atom.GetNeighbors():
        ni = nbr.GetIdx()
        if ni in ring_atoms:
            continue
        if nbr.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(start_idx, ni)
        order = bond.GetBondTypeAsDouble()
        if (nbr.GetSymbol() == 'O' and order == 2.0 and nbr.GetDegree() == 1
                and nbr.GetTotalNumHs() == 0 and nbr.GetFormalCharge() == 0):
            if carbonyl_o is not None:
                return None
            carbonyl_o = ni
        elif (nbr.GetSymbol() == 'O' and order == 1.0 and nbr.GetDegree() == 2
                and nbr.GetTotalNumHs() == 0 and nbr.GetFormalCharge() == 0):
            onward = [x for x in nbr.GetNeighbors() if x.GetIdx() != start_idx]
            if len(onward) == 1 and onward[0].GetSymbol() == 'C':
                if ester_o is not None:
                    return None
                ester_o = ni
                alkyl_c = onward[0].GetIdx()
            else:
                return None
        else:
            # A ketone R, a second heteroatom, a thio/seleno ester, etc.:
            # not a clean -C(=O)-O-C ester -> leave to the other branches.
            return None
    if carbonyl_o is None or ester_o is None or alkyl_c is None:
        return None

    # gate: an ester demotes to a prefix ONLY when a senior ACID owns the
    # parent suffix word (see _ring_bears_acid_suffix docstring).
    if not _ring_bears_acid_suffix(mol, ring_atoms, start_idx):
        return None

    from ..assembly.naming_utils import needs_brackets
    from ..assembly.substituent_prefix_forms import get_alkoxycarbonyl_prefix
    prefix = get_alkoxycarbonyl_prefix(
        mol, (start_idx, carbonyl_o, ester_o, alkyl_c), None)
    if not prefix:
        return None
    sub_atoms = _bfs_substituent_atoms(mol, start_idx, ring_atoms)
    return {
        'name': prefix,
        'atoms': sub_atoms,
        'is_complex': needs_brackets(prefix),
    }


def _identify_acyl_group(mol, start_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """Recognize a ring-attached ketone-acyl branch -C(=O)-R and return its
    retained/systematic acyl PREFIX record (Wave2 T5c, /:

      R = unbranched saturated all-C chain -> acetyl / propanoyl / {stem}anoyl
      R = plain (unsubstituted) benzene -> benzoyl

    Fail-closed None for everything else (branched / unsaturated / hetero /
    substituted-aryl acyls, formyl [aldehyde suffix territory], esters/acids
    [second O]) so the unnameable sentinel keeps the molecule honest. The
    returned 'atoms' cover the ENTIRE branch (carbonyl C + O + R) — no
    silent atom drops."""
    start_atom = mol.GetAtomWithIdx(start_idx)
    if start_atom.GetSymbol() != 'C' or start_atom.GetFormalCharge() != 0:
        return None
    if start_atom.IsInRing():
        return None
    carbonyl_o = None
    r_neighbors = []
    for nbr in start_atom.GetNeighbors():
        ni = nbr.GetIdx()
        if ni in ring_atoms:
            continue
        if nbr.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(start_idx, ni)
        if (nbr.GetSymbol() == 'O' and bond.GetBondTypeAsDouble() == 2.0
                and nbr.GetDegree() == 1 and nbr.GetFormalCharge() == 0):
            if carbonyl_o is not None:
                return None
            carbonyl_o = ni
        else:
            r_neighbors.append(nbr)
    if carbonyl_o is None or len(r_neighbors) != 1:
        return None  # formyl (no R) is aldehyde-suffix territory; acids/esters have a 2nd O
    r0 = r_neighbors[0]
    if mol.GetBondBetweenAtoms(
            start_idx, r0.GetIdx()).GetBondTypeAsDouble() != 1.0:
        return None

    # R = plain benzene -> benzoyl (retained acyl prefix,
    if r0.GetIsAromatic() and r0.IsInRing():
        r_ring = None
        for ring in mol.GetRingInfo().AtomRings():
            if r0.GetIdx() in ring:
                r_ring = ring
                break
        if r_ring is None or len(r_ring) != 6:
            return None
        for a in r_ring:
            atom = mol.GetAtomWithIdx(a)
            if not atom.GetIsAromatic() or atom.GetSymbol() != 'C':
                return None
            for nb in atom.GetNeighbors():
                if (nb.GetIdx() not in r_ring and nb.GetAtomicNum() > 1
                        and nb.GetIdx() != start_idx):
                    return None  # substituted aryl acyl -> fail closed
        return {
            'name': 'benzoyl',
            'atoms': [start_idx, carbonyl_o] + list(r_ring),
        }

    # R = unbranched saturated all-C chain -> acetyl / {stem}anoyl
    if r0.GetSymbol() != 'C' or r0.IsInRing() or r0.GetIsAromatic():
        return None
    chain = []
    prev = start_idx
    cur = r0
    while True:
        if cur.GetSymbol() != 'C' or cur.GetFormalCharge() != 0 \
                or cur.IsInRing() or cur.GetIsAromatic():
            return None
        nxt = []
        for nb in cur.GetNeighbors():
            ni = nb.GetIdx()
            if ni == prev or nb.GetAtomicNum() <= 1:
                continue
            if mol.GetBondBetweenAtoms(
                    cur.GetIdx(), ni).GetBondTypeAsDouble() != 1.0:
                return None
            if nb.GetSymbol() != 'C':
                return None  # hetero on the acyl chain -> fail closed
            nxt.append(nb)
        chain.append(cur.GetIdx())
        if not nxt:
            break
        if len(nxt) > 1:
            return None  # branched acyl -> fail closed
        prev, cur = cur.GetIdx(), nxt[0]

    n_acyl = len(chain) + 1  # + the carbonyl carbon
    if n_acyl == 2:
        acyl_name = 'acetyl'
    else:
        from ..data.chain_names import get_chain_prefix
        try:
            acyl_name = f"{get_chain_prefix(n_acyl)}anoyl"
        except (ValueError, KeyError):
            return None
    return {
        'name': acyl_name,
        'atoms': [start_idx, carbonyl_o] + chain,
    }


def _identify_nitrile_group(mol, c_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify nitrile (C#N) substituent attached to benzene.

    The nitrile carbon is directly attached to the ring. We check if this carbon
    has a triple bond to nitrogen and no other heavy atom neighbors (besides the ring).

    Args:
        mol: RDKit Mol object
        c_idx: Index of the carbon atom attached to the ring
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name': 'nitrile', 'type': 'functional', 'atoms': [...] or None
    """
    c_atom = mol.GetAtomWithIdx(c_idx)

    # Check neighbors of the nitrile carbon
    for neighbor in c_atom.GetNeighbors():
        if neighbor.GetIdx() in ring_atoms:
            continue

        # Check for triple bond to nitrogen
        bond = mol.GetBondBetweenAtoms(c_idx, neighbor.GetIdx())
        if (neighbor.GetSymbol() == 'N' and
                bond and bond.GetBondType() == Chem.BondType.TRIPLE):
            # Verify the nitrogen has no other heavy atom neighbors (just the C#N)
            n_atom = neighbor
            n_heavy_neighbors = [n for n in n_atom.GetNeighbors()
                                 if n.GetSymbol() != 'H' and n.GetIdx() != c_idx]
            if not n_heavy_neighbors:
                return {
                    'name': 'nitrile',
                    'type': 'functional',
                    'atoms': [c_idx, neighbor.GetIdx()]
                }

    return None


def _detect_benzene_nitrile(mol, ring_atoms: Tuple[int, ...]) -> Dict:
    """
    Detect if benzene ring has a nitrile substituent.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the benzene ring

    Returns:
        Dict with 'is_nitrile': bool, 'nitrile_positions': list of ring atom indices
        that have nitrile attached
    """
    ring_set = set(ring_atoms)
    nitrile_positions = []

    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in ring_set:
                continue

            # Check if this neighbor is a nitrile carbon
            if neighbor.GetSymbol() == 'C':
                result = _identify_nitrile_group(mol, nbr_idx, ring_set)
                if result and result.get('name') == 'nitrile':
                    nitrile_positions.append(ring_idx)
                    break  # Only count once per ring position

    return {
        'is_nitrile': len(nitrile_positions) > 0,
        'nitrile_positions': nitrile_positions
    }


def _identify_nitrogen_group(mol, n_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """Identify nitrogen-based substituent (amino, nitro, N-alkylamino, etc.)."""
    n_atom = mol.GetAtomWithIdx(n_idx)

    # Count neighbors (excluding ring)
    neighbors = [n for n in n_atom.GetNeighbors() if n.GetIdx() not in ring_atoms]

    # Check for nitro group: N with 2 oxygens, positive charge
    if n_atom.GetFormalCharge() == 1:
        o_count = sum(1 for n in neighbors if n.GetSymbol() == 'O')
        if o_count == 2:
            # Nitro group
            atoms = [n_idx] + [n.GetIdx() for n in neighbors if n.GetSymbol() == 'O']
            return {'name': 'nitro', 'atoms': atoms}

    h_count = n_atom.GetTotalNumHs()

    # Simple amino (-NH2)
    if h_count == 2 and len(neighbors) == 0:
        # C4: carry an amine_candidate payload so name_substituted_benzene can
        # promote this to the retained 'aniline' SUFFIX when the amine is the
        # molecule-level principal group. The existing 'amino' prefix name is
        # kept untouched so the NON-promoted (senior-group-present) path is
        # byte-identical (4-aminophenol / 4-aminobenzoic acid).
        return {
            'name': 'amino',
            'atoms': [n_idx],
            'amine_candidate': {'suffix_name': 'amine', 'n_substituents': []},
        }

    # Wave2: ring-attached amide N (R-CO-NH-ring) takes
    # the amido-family prefix — formamido / acetamido / {stem}anamido /
    # benzamido — method (1) generates the PIN (4-formamidobenzoic acid).
    # Without this recognizer the branch fell to the generic fragment
    # fallback below, which named it as if attached at the carbonyl C
    # ('carbamoyl'), claiming a different molecule. Fail closed (None) for
    # anything but a clean -N(H)-CO-R branch.
    if h_count == 1 and len(neighbors) == 1 and neighbors[0].GetSymbol() == 'C':
        _acyl_c = neighbors[0]
        _has_c_o = any(
            nb.GetSymbol() == 'O' and nb.GetIdx() not in ring_atoms
            and mol.GetBondBetweenAtoms(
                _acyl_c.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 2.0
            for nb in _acyl_c.GetNeighbors()
        )
        if _has_c_o:
            sub_atoms = _bfs_substituent_atoms(mol, n_idx, ring_atoms)
            from ..assembly.substituent_naming import (
                acyl_amido_prefix_from_branch,
            )
            _amido = acyl_amido_prefix_from_branch(
                mol, n_idx, _acyl_c.GetIdx(), sub_atoms
            )
            if _amido:
                _complex = any(ch.isdigit() for ch in _amido) or '-' in _amido
                return {
                    'name': f'({_amido})' if _complex else _amido,
                    'atoms': sub_atoms,
                    'is_complex': _complex,
                }
        else:
            # Wave2: ring-attached amidine via its AMINO N
            # (-NH-C(=NH)-R) -> the {stem}imidamido prefix (4-ethanimidamido-
            # benzoic acid). Parallel to the amido branch above but keyed on the
            # imino C=N. Guards inside imidoyl_amido_prefix_from_branch reject
            # guanidine (2nd amino N), amidrazone (=N-N), and Schiff bases.
            _has_c_n = any(
                nb.GetSymbol() == 'N' and nb.GetIdx() not in ring_atoms
                and mol.GetBondBetweenAtoms(
                    _acyl_c.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 2.0
                for nb in _acyl_c.GetNeighbors()
            )
            if _has_c_n:
                sub_atoms = _bfs_substituent_atoms(mol, n_idx, ring_atoms)
                from ..assembly.substituent_naming import (
                    imidoyl_amido_prefix_from_branch,
                )
                _imid = imidoyl_amido_prefix_from_branch(
                    mol, n_idx, _acyl_c.GetIdx(), sub_atoms
                )
                if _imid:
                    _complex = any(ch.isdigit() for ch in _imid) or '-' in _imid
                    return {
                        'name': f'({_imid})' if _complex else _imid,
                        'atoms': sub_atoms,
                        'is_complex': _complex,
                    }

    # / (plan P1AM Task 8): ring-attached
    # -NH-S(=N-NH2)(R)[=O]? -> '(...sulfino/sulfonohydrazonamido)'.
    if h_count == 1 and len(neighbors) == 1 and neighbors[0].GetSymbol() == 'S':
        sub_atoms = _bfs_substituent_atoms(mol, n_idx, ring_atoms)
        from ..assembly.substituent_naming import (
            sulfino_hydrazonoyl_amido_prefix_from_branch,
        )
        _shz = sulfino_hydrazonoyl_amido_prefix_from_branch(
            mol, n_idx, neighbors[0].GetIdx(), sub_atoms
        )
        if _shz:
            return {
                'name': f'({_shz})',
                'atoms': sub_atoms,
                'is_complex': True,
            }

    # (W2F-P8): -NH-(substituted benzene aryl) -> the decorated aryl ring
    # is cited as an 'N-(2,4-dibromophenyl)' substituent (via the shared
    # decorated-ring core namer), promoting the CURRENT benzene ring to the
    # 'aniline' parent (the parent-vs-N-aryl choice runs through parent selection,
    #. Only a carbocyclic benzene N-ring is handled: a BARE phenyl gives
    # None here (nothing to decorate) and falls through to the 'anilino'/
    # 'N-phenyl' path below, and a heteroaryl N-ring is left to the parent-
    # selection pipeline (the heterocycle may be the senior parent,.
    # Fail closed when the core cannot be built.
    if (h_count == 1 and len(neighbors) == 1
            and neighbors[0].GetSymbol() == 'C'
            and neighbors[0].GetIsAromatic() and neighbors[0].IsInRing()):
        _arylc = neighbors[0]
        _aryl_ring = None
        for _rng in mol.GetRingInfo().AtomRings():
            if (_arylc.GetIdx() in _rng and len(_rng) == 6
                    and not (set(_rng) & ring_atoms)
                    and all(mol.GetAtomWithIdx(i).GetIsAromatic()
                            and mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                            for i in _rng)):
                _aryl_ring = _rng
                break
        if _aryl_ring is not None:
            from .ring_substituents import (
                anilino_preferred_prefix,
                decorated_ring_substituent_name,
            )
            _sub_atoms = _bfs_substituent_atoms(mol, n_idx, ring_atoms)
            _dec = decorated_ring_substituent_name(
                mol, _aryl_ring, _arylc.GetIdx(),
                expected_atoms=set(_sub_atoms) - {n_idx},
            )
            if _dec is not None and _dec.endswith('phenyl'):
                # (the Blue Book): 'anilino' is the PREFERRED PREFIX for
                # C6H5-NH- with full substitution allowed; '(...phenyl)amino' is
                # the general-nomenclature column (the Blue Book '4-chloroanilino
                # (preferred prefix) | (4-chlorophenyl)amino'). The locants are
                # identical because decorated_ring_substituent_name numbers a
                # carbocycle with the attachment at 1, which is aniline's C-1.
                # The amine_candidate is DELIBERATELY unchanged: it feeds the
                # aniline-as-parent promotion ('N-(2,4-dibromophenyl)aniline'),
                # where the ring is an N-substituent and not an anilino prefix.
                _pref = anilino_preferred_prefix(_dec)
                from ..assembly.naming_utils import apply_enclosing_marks
                return {
                    'name': _pref if _pref is not None
                            else apply_enclosing_marks(f'({_dec})amino', -1),
                    'atoms': _sub_atoms,
                    'is_complex': True,
                    'amine_candidate': {
                        'suffix_name': 'amine', 'n_substituents': [f'({_dec})'],
                    },
                }

    # N-monoalkyl amino (-NHR): 1 H, 1 carbon neighbor
    # IUPAC 2013: N-alkylamino (e.g., N-methylamino, N-ethylamino)
    if h_count == 1 and len(neighbors) == 1 and neighbors[0].GetSymbol() == 'C':
        alkyl_atoms, carbon_count = _collect_pure_alkyl(mol, neighbors[0].GetIdx(), ring_atoms | {n_idx})
        if alkyl_atoms is not None and carbon_count > 0:
            from ..assembly.substituent_naming import name_substituent_fragment
            alkyl_name = name_substituent_fragment(
                mol, alkyl_atoms, neighbors[0].GetIdx(), list(ring_atoms | {n_idx})
            )
            if alkyl_name is None:
                try:
                    alkyl_name = get_alkyl_name(carbon_count)
                except (ValueError, KeyError):
                    alkyl_name = None
            if alkyl_name:
                # (the Blue Book) retained preferred prefix. Cited BARE:
                # the unsubstituted prefix carries no locant of its own, so no
                # enclosing marks (the Blue Book '3-anilinobenzoic acid (PIN)').
                # Routed through the shared primitive so this site and the
                # substituted ones cannot drift apart.
                if alkyl_name == 'phenyl':
                    from .ring_substituents import anilino_preferred_prefix
                    return {
                        'name': anilino_preferred_prefix('phenyl'),
                        'atoms': [n_idx] + alkyl_atoms,
                        'is_complex': False,
                        # C4: -NHPh promotes to 'N-phenylaniline' when the amine
                        # is the principal group; else stays the 'anilino' prefix.
                        'amine_candidate': {
                            'suffix_name': 'amine', 'n_substituents': ['phenyl'],
                        },
                    }
                # Amino prefixes cite the N-substituent without 'N-';
                # '4,4-bis(methylamino)butanoic acid (PIN)', the Blue Book;
                # '6-[(methylamino)sulfinyl]naphthalene-2-carboxylic acid (PIN)',:32989);
                # a compound alkyl prefix is enclosed ('2-[di(butan-2-yl)amino]butan-2-ol
                # (PIN)',:28170).
                from ..assembly.naming_utils import enclose_if_compound
                return {
                    'name': f'({enclose_if_compound(alkyl_name)}amino)',
                    'atoms': [n_idx] + alkyl_atoms,
                    'is_complex': True,
                    # C4: -NHR promotes to 'N-<alkyl>aniline' when principal.
                    'amine_candidate': {
                        'suffix_name': 'amine', 'n_substituents': [alkyl_name],
                    },
                }

    # (BB 18108, W2E-P1FC Task 6): complex substituent prefix by
    # SUBSTITUTION — an -NH-R amino whose R is a compound/substituted prefix
    # (e.g. -NH-CH2Cl chloromethyl) is the complex prefix '(chloromethyl)amino'.
    # The N-monoalkyl branch above only handles a PURE (all-carbon) alkyl via
    # _collect_pure_alkyl; a substituted branch (halomethyl, alkoxyalkyl,...)
    # skips it and previously mis-parsed as separate amino + chloromethyl. Name
    # the WHOLE N-substituent branch with the universal substituent namer and
    # wrap it as '({branch}amino)'. Fail-closed: decline (fall through) if the
    # branch cannot be fully named, so a partial/ambiguous prefix never leaks.
    if h_count == 1 and len(neighbors) == 1 and neighbors[0].GetSymbol() == 'C':
        _branch_atoms = _bfs_substituent_atoms(mol, n_idx, ring_atoms)
        _branch_atoms = [a for a in _branch_atoms if a != n_idx]
        # Fail-closed guard: this LOCAL complex-prefix build is only safe when
        # the N-substituent branch is a plain acyclic substituted alkyl (e.g.
        # -CH2Cl). A RING-bearing branch (-NH-pyridinyl, -NH-phenyl) triggers a
        # parent-hydride SENIORITY competition between the two ring systems
        # that must be resolved by the parent-selection pipeline, not
        # forced onto the benzene ring here — so decline and let it fall
        # through (N-phenylpyridin-4-amine, not [(pyridin-4-yl)amino]benzene).
        _branch_has_ring = any(mol.GetAtomWithIdx(a).IsInRing()
                               for a in _branch_atoms)
        if _branch_atoms and not _branch_has_ring:
            from ..assembly.substituent_naming import name_substituent_fragment
            _bname = name_substituent_fragment(
                mol, _branch_atoms, neighbors[0].GetIdx(),
                list(ring_atoms | {n_idx}),
            )
            if _bname and ' ' not in _bname:
                #: the inner substituted prefix ('chloromethyl') is
                # itself enclosed in parentheses, then concatenated with the
                # 'amino' compound prefix -> '(chloromethyl)amino'. Mark complex
                # so the outer composer escalates the enclosing marks to square
                # brackets on the senior parent -> '4-[(chloromethyl)amino]-
                # benzoic acid'. A bare (already-simple) branch name that is not
                # itself compound is left unwrapped inside (handled by
                # is_complex_substituent).
                from ..assembly.naming_utils import (
                    apply_enclosing_marks,
                    is_complex_substituent,
                )
                # Inner substituted prefix gets its own parens;
                # 'chloromethyl' -> '(chloromethyl)'. Suite fix j6 (TRIAGE g5
                # C11): the mark escalates past the inner name's own marks
                #, the Blue Book) -- a stereo-prefixed
                # '(2R)-3-oxobutan-2-yl' becomes '[(2R)-3-oxobutan-2-yl]', not
                # '((2R)-...)' (which also failed the OPSIN grammar check).
                _inner = (apply_enclosing_marks(_bname, -1)
                          if is_complex_substituent(_bname) else _bname)
                _complex_prefix = f'{_inner}amino'
                # Escalate the OUTER enclosing mark by nesting depth:
                # '(chloromethyl)amino' -> '[(chloromethyl)amino]'. Passing the
                # already-escalated form to format_substituent_prefix (which
                # only wraps-if-unwrapped) yields '4-[(chloromethyl)amino]-...'.
                # A SIMPLE inner prefix ('cyano') still makes 'cyanoamino' a
                # compound prefix, and (the Blue Book)
                # "Parentheses are used around compound... prefixes" -- the
                # formatter only wraps-if-unwrapped, so an unwrapped
                # 'cyanoamino' shipped as '4-cyanoaminobenzoic acid' at
                # pin_verified; '(methylamino)' (:21624), '(carbamoylamino)acetic
                # acid' (:33382). Both shapes are enclosed here.
                _wrapped = apply_enclosing_marks(_complex_prefix, -1)
                return {
                    'name': _wrapped,
                    'atoms': [n_idx] + list(_branch_atoms),
                    'is_complex': True,
                }

    # (the Blue Book) N,N-disubstituted ANILINO. -N(R)-C6H5 takes the
    # PREFERRED prefix 'N-<R>anilino': the Blue Book verbatim
    # '3-(N-methylanilino)phenol (PIN) 3-[methyl(phenyl)amino]phenol'
    # so the general-nomenclature column is the methyl(phenyl)amino family, and
    # HEAD's third spelling '(N-methyl-N-phenylamino)' is non-PIN by
    # (the Blue Book — a PIN requires the names of its COMPONENTS to be preferred names
    # too, even when the parent is right).
    #
    # The anilino ring is detected STRUCTURALLY, never from the branch's name: the
    # N,N-dialkyl path below falls back to get_alkyl_name(carbon_count) when the
    # universal namer declines, which named a 4-methylphenyl ring 'heptyl' — a
    # different molecule. Fails closed (falls through to the legacy path) whenever
    # the branch atoms are not exactly N + one bare benzene + one nameable branch.
    if h_count == 0 and len(neighbors) == 2 and all(
            nb.GetSymbol() == 'C' for nb in neighbors):
        from .ring_substituents import anilino_preferred_prefix
        _ri = mol.GetRingInfo()
        _aryl_pairs = []
        for _cn in neighbors:
            if not (_cn.GetIsAromatic() and _cn.IsInRing()):
                continue
            for _rng in _ri.AtomRings():
                if (_cn.GetIdx() in _rng and len(_rng) == 6
                        and not (set(_rng) & ring_atoms)
                        and all(_ri.NumAtomRings(i) == 1 for i in _rng)
                        and all(mol.GetAtomWithIdx(i).GetIsAromatic()
                                and mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                                and mol.GetAtomWithIdx(i).GetFormalCharge() == 0
                                for i in _rng)):
                    _aryl_pairs.append((_cn, _rng))
                    break
        _nn_sub_atoms = _bfs_substituent_atoms(mol, n_idx, ring_atoms)
        for _arylc, _rng in _aryl_pairs:
            _other = next(nb for nb in neighbors
                          if nb.GetIdx() != _arylc.GetIdx())
            _other_atoms = _bfs_substituent_atoms(
                mol, _other.GetIdx(), ring_atoms | {n_idx})
            # The anilino ring must be UNSUBSTITUTED: every branch atom is
            # accounted for by N + this ring + the other N-branch. (A decorated
            # ring here would need the ring and N locants merged into one
            # alphanumerical sequence, which anilino_preferred_prefix declines.)
            if set(_nn_sub_atoms) != {n_idx} | set(_rng) | set(_other_atoms):
                continue
            from ..assembly.substituent_naming import name_substituent_fragment
            _oname = name_substituent_fragment(
                mol, _other_atoms, _other.GetIdx(), list(ring_atoms | {n_idx}))
            if not _oname:
                continue
            _pref = anilino_preferred_prefix('phenyl', _oname)
            if _pref is None:
                continue
            return {
                'name': _pref,
                'atoms': _nn_sub_atoms,
                'is_complex': True,
                # UNCHANGED promotion payload: when the amine is the principal
                # group the ring is the 'aniline' parent and the two branches are
                # N-substituents ('N-methyl-N-phenylaniline'), not an anilino
                # prefix. Same list the legacy path below builds (it sorts too).
                'amine_candidate': {
                    'suffix_name': 'amine',
                    'n_substituents': sorted([_oname, 'phenyl']),
                },
            }

    # N,N-dialkyl amino (-NR2): 0 H, 2 carbon neighbors -> '(dimethylamino)',
    # '[ethyl(methyl)amino]' (no italic N locants on an amino prefix).
    if h_count == 0 and len(neighbors) == 2:
        c_neighbors = [n for n in neighbors if n.GetSymbol() == 'C']
        if len(c_neighbors) == 2:
            alkyl_names_list = []
            all_sub_atoms = [n_idx]
            for cn in c_neighbors:
                alkyl_atoms, carbon_count = _collect_pure_alkyl(mol, cn.GetIdx(), ring_atoms | {n_idx})
                if alkyl_atoms is None or carbon_count == 0:
                    break
                from ..assembly.substituent_naming import name_substituent_fragment
                aname = name_substituent_fragment(
                    mol, alkyl_atoms, cn.GetIdx(), list(ring_atoms | {n_idx})
                )
                if aname is None:
                    try:
                        aname = get_alkyl_name(carbon_count)
                    except (ValueError, KeyError):
                        break
                alkyl_names_list.append(aname)
                all_sub_atoms.extend(alkyl_atoms)
            else:
                # Both identified. The amino prefix takes NO italic N locants: the
                # nitrogen is a mononuclear parent, (the Blue Book)
                # 'the first cited substituent never has enclosing marks unless it
                # includes a locant. The second and further substituents are each
                # enclosed with parentheses'; 'bis(dimethylamino) (preferred prefix)'
                #, '4-(dimethylamino)-2-methylbutane-2-peroxol (PIN)'
                # (:27955), '5-methyl-2-[methyl(phenyl)carbamoyl]benzoic acid (PIN)'
                # (:32957). The ONE amino-prefix assembler owns order and marks.
                alkyl_names_list.sort()
                from ..assembly.composer import _assemble_decorated_amino_prefix
                prefix_name = _assemble_decorated_amino_prefix(
                    [(alkyl_names_list[0], False), (alkyl_names_list[1], False)])
                if not prefix_name:
                    return None
                return {
                    'name': prefix_name,
                    'atoms': all_sub_atoms,
                    'is_complex': True,
                    # C4: -NR2 promotes to 'N,N-<...>aniline' when principal.
                    'amine_candidate': {
                        'suffix_name': 'amine',
                        'n_substituents': list(alkyl_names_list),
                    },
                }

    # Nitroso (-NO)
    if h_count == 0 and len(neighbors) == 1 and neighbors[0].GetSymbol() == 'O':
        bond = mol.GetBondBetweenAtoms(n_idx, neighbors[0].GetIdx())
        if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
            return {
                'name': 'nitroso',
                'atoms': [n_idx, neighbors[0].GetIdx()]
            }

    # Wave-2 completion /: azido -N=[N+]=[N-] on the ring.
    # The ring-attached N is neutral with a single N neighbour (=[N+]); the
    # chain is exactly three N atoms. -> 'azido' (azidobenzene, OPSIN-RT).
    if h_count == 0 and len(neighbors) == 1 and neighbors[0].GetSymbol() == 'N':
        sub_atoms = _bfs_substituent_atoms(mol, n_idx, ring_atoms)
        if (len(sub_atoms) == 3
                and all(mol.GetAtomWithIdx(i).GetSymbol() == 'N' for i in sub_atoms)):
            return {'name': 'azido', 'atoms': sub_atoms}

    # Wave-2 completion: isocyano -[N+]#[C-] on the ring. The
    # ring-attached N is positively charged, triple-bonded to a terminal
    # negative C. -> 'isocyano' (isocyanobenzene, OPSIN-RT).
    if (n_atom.GetFormalCharge() == 1 and len(neighbors) == 1
            and neighbors[0].GetSymbol() == 'C' and neighbors[0].GetDegree() == 1):
        c_nbr = neighbors[0]
        bond = mol.GetBondBetweenAtoms(n_idx, c_nbr.GetIdx())
        if (bond and bond.GetBondType() == Chem.BondType.TRIPLE
                and c_nbr.GetFormalCharge() == -1):
            return {'name': 'isocyano', 'atoms': [n_idx, c_nbr.GetIdx()]}

    # W2E-D2 /: mixed additive-prefix N-substituents whose
    # attach atom is a heteroatom centre — (alkoxy)sulfinyl [-S(=O)-OR] and
    # [bis(sulfanyl)phosphoryl] [-P(=O)(SH)n]. The single N-neighbour is the S or
    # P (not a carbon), so the C/O/N branches above all decline. Name the whole
    # centre as a compound additive prefix and promote the amine to the 'aniline'
    # suffix (N-<prefix>aniline). Fail-closed inside the helper on any other shape.
    if h_count == 1 and len(neighbors) == 1:
        _het = neighbors[0]
        _het_sym = _het.GetSymbol()
        if _het_sym in ('S', 'P'):
            from ..assembly.substituent_prefix_forms import (
                get_alkoxysulfinyl_prefix,
                get_phosphoryl_prefix,
            )
            _mixed = None
            if _het_sym == 'S':
                _mixed = get_alkoxysulfinyl_prefix(mol, _het.GetIdx(), n_idx)
            else:
                _mixed = get_phosphoryl_prefix(mol, _het.GetIdx(), n_idx)
            if _mixed:
                _sub_atoms = _bfs_substituent_atoms(mol, n_idx, ring_atoms)
                return {
                    'name': f'({_mixed}amino)',   # no 'N-' in an amino prefix (:26318)
                    'atoms': _sub_atoms,
                    'is_complex': True,
                    # Promote to 'N-<prefix>aniline' when the amine is principal.
                    'amine_candidate': {
                        'suffix_name': 'amine', 'n_substituents': [_mixed],
                    },
                }

    # W3-P15: ring-attached hydroxylamine -NH-OH -> the
    # 'hydroxyamino' preselected prefix (BB 38428 '4-(hydroxyamino)phenol').
    # The prefix generator already exists (substituent_prefix_forms.get_prefix
    # returns 'hydroxyamino' for an unsubstituted -NH-OH); wire it here so the
    # co-perceived hydroxylamine FG is demoted to its prefix when a senior PCG
    # (phenol / -ol / acid) owns the ring, instead of falling to the generic
    # fragment fallback below (which emitted the wrong 'aminohydroxylyl').
    # Fail-closed to the fallback for N,O-disub / N-acyl forms (their prefix
    # needs a composed builder not built here): the O must be a bare -OH.
    if h_count == 1 and len(neighbors) == 1 and neighbors[0].GetSymbol() == 'O':
        _o = neighbors[0]
        _o_others = [x for x in _o.GetNeighbors() if x.GetIdx() != n_idx]
        if (_o.GetFormalCharge() == 0 and _o.GetTotalNumHs() == 1
                and not _o_others):
            from ..assembly.substituent_prefix_forms import (
                get_substituent_prefix_form,
            )
            _hp = get_substituent_prefix_form(
                'hydroxylamine', mol, (_o.GetIdx(), n_idx))
            if _hp:
                return {'name': _hp, 'atoms': [n_idx, _o.GetIdx()],
                        'is_complex': True}

    # Fallback: complex N-substituent (non-alkyl chains, heteroatom-containing
    # groups like guanidino, ureido, etc.). Collect all atoms via BFS and try
    # recursive naming.
    if neighbors:
        sub_atoms = _bfs_substituent_atoms(mol, n_idx, ring_atoms)
        if len(sub_atoms) > 1 and len(sub_atoms) <= 25:
            from ..assembly.substituent_naming import name_substituent_fragment
            frag_name = name_substituent_fragment(
                mol, sub_atoms, n_idx, list(ring_atoms)
            )
            if frag_name:
                return {
                    'name': frag_name,
                    'atoms': sub_atoms,
                    'is_complex': True,
                }

    return None


def _collect_pure_alkyl(mol, start_idx: int, excluded: Set[int]):
    """
    BFS to collect a pure alkyl group (only C/H atoms).

    Returns:
        Tuple of (list of atom indices, carbon_count) or (None, 0) if not pure alkyl.
    """
    visited = {start_idx}
    queue = deque([start_idx])
    all_atoms = []
    carbon_count = 0

    while queue:
        current_idx = queue.popleft()
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        if current_atom.GetSymbol() == 'C':
            carbon_count += 1
        elif current_atom.GetSymbol() != 'H':
            return None, 0  # Not pure alkyl

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in excluded:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    return all_atoms, carbon_count


def _identify_oxygen_group(mol, o_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """Identify oxygen-based substituent (hydroxy, methoxy, alkoxy, hydroperoxy, etc.)."""
    o_atom = mol.GetAtomWithIdx(o_idx)

    # Count neighbors (excluding ring)
    neighbors = [n for n in o_atom.GetNeighbors() if n.GetIdx() not in ring_atoms]

    # Simple hydroxy (-OH)
    h_count = o_atom.GetTotalNumHs()
    if h_count == 1 and len(neighbors) == 0:
        return {'name': 'hydroxy', 'atoms': [o_idx]}

    if len(neighbors) == 1:
        nbr = neighbors[0]
        nbr_symbol = nbr.GetSymbol()

        # Hydroperoxy (-O-OH): oxygen connected to another oxygen with H
        if nbr_symbol == 'O' and nbr.GetTotalNumHs() >= 1:
            return {'name': 'hydroperoxy', 'atoms': [o_idx, nbr.GetIdx()]}

        # (generalizes SP2.1'): -O-C(=O)-R is an ACYLOXY, not an alkoxy. The
        # alkoxy branch below collects a "pure alkyl" R via _collect_pure_alkyl,
        # which DROPS the carbonyl =O (and any further hetero) -- measured:
        # -O-C(=O)CH2C(CH3)2OH on benzoic acid -> '4-(pentyloxy)benzoic acid', a
        # WRONG molecule the gate then suppresses, so the whole molecule
        # abstains on the default tier (best-effort already emits the correct acyloxy
        # via name_substituent). trace (a project rule) refuted the SP2.1'
        # 'OPSIN-invalid' premise. Route a plain acyloxy ester to the SAME recognizer
        # name_substituent uses (composer._acyloxy_prefix_for_frag ->
        # rules.lipids._acyloxy_for_site -> the full acid engine) for the
        # '<acyl>oxy' PIN prefix. The recognizer is fail-closed (exactly
        # one terminal '=O' on the carbonyl C, <=1 all-carbon R, whole acyl side
        # self-contained, carbamate/carbonate/thiono excluded); on a non-plain shape
        # it returns None and we FAIL CLOSED here rather than fall through to the
        # alkyl mis-read (defer > wrong). A RETAINED acyloxy round-trips through the
        # same acid engine to the same string, so it stays byte-identical; the caller
        # RT-gates the assembled name downstream (0-wrong).
        if nbr_symbol == 'C' and any(
                b.GetBondTypeAsDouble() == 2.0
                and b.GetOtherAtom(nbr).GetAtomicNum() in (7, 8, 16)
                for b in nbr.GetBonds()):
            _acy = None
            try:
                from ..assembly.composer import _acyloxy_prefix_for_frag
                _acy_side = _bfs_substituent_atoms(
                    mol, nbr.GetIdx(), ring_atoms | {o_idx})
                _acy_frag = set(_acy_side) | {o_idx}
                _acy = _acyloxy_prefix_for_frag(mol, _acy_frag, o_idx)
            except Exception:
                _acy = None
            if _acy:
                return {'name': _acy, 'atoms': sorted(_acy_frag),
                        'is_complex': True}
            return None  # acyloxy we cannot name cleanly -> fail closed, never mis-read

        # Alkoxy (-O-C...): methoxy, ethoxy, propoxy, etc. An ARYLOXY
        # (-O-aromatic-ring-C) is NOT an alkoxy — route it to Case 1 below so a
        # bare ring gives 'phenoxy' and a decorated ring is built by the
        # substituted-aryloxy path. Without this guard _collect_pure_alkyl
        # greedily walks the aromatic ring as if it were a chain
        # (4-ethylphenoxy -> 'octoxy', a structure-dropping name suppressed to
        # 'unknown' by the validity gate).
        if nbr_symbol == 'C' and not (nbr.GetIsAromatic() and nbr.IsInRing()):
            c_atom = nbr
            # Collect the alkyl part
            alkyl_atoms, carbon_count = _collect_pure_alkyl(mol, c_atom.GetIdx(), ring_atoms | {o_idx})
            if alkyl_atoms is not None and carbon_count > 0:
                from ..assembly.substituent_naming import name_substituent_fragment
                alkyl_name = name_substituent_fragment(
                    mol, alkyl_atoms, c_atom.GetIdx(), list(ring_atoms | {o_idx})
                )
                if alkyl_name is None:
                    try:
                        alkyl_name = get_alkyl_name(carbon_count)
                    except (ValueError, KeyError):
                        alkyl_name = None
                if alkyl_name:
                    # morphology via composed_alkoxy_prefix: retained
                    # set contracts (butyl->butoxy), C5+ and rings keep the whole
                    # '-yl' ('pentyl'->'pentyloxy', 'cyclohexyl'->'cyclohexyloxy',
                    # NOT the mangled 'pentoxy'/'cyclohexoxy') (F-spell-oxy).
                    if alkyl_name.endswith('yl'):
                        from ..assembly.substituent_enumerator import alkoxy_prefix_from_substituent
                        oxy_name = alkoxy_prefix_from_substituent(alkyl_name)
                    else:
                        oxy_name = alkyl_name + 'oxy'
                    if oxy_name is None:
                        return None
                    return {'name': oxy_name, 'atoms': [o_idx] + alkyl_atoms}

            # /: _collect_pure_alkyl declines when the R
            # side carries an INTERIOR ether O / thioether S
            # (-O-CH2CH2-O-CH3). The Case-4 carbon count below would DROP that
            # heteroatom (constitutionally WRONG -> 'propoxy'). Name the R side
            # with the full substituent namer (nested (R-oxy)/(R-sulfanyl)
            # recursion) and cite '({R}oxy)'. Guard: acyclic, non-aromatic R
            # whose only heteroatoms are neutral divalent ether O / thioether S.
            r_side = _bfs_substituent_atoms(mol, c_atom.GetIdx(), ring_atoms | {o_idx})
            _ring_info = mol.GetRingInfo()
            _r_ok = bool(r_side) and all(
                _ring_info.NumAtomRings(a) == 0
                and not mol.GetAtomWithIdx(a).GetIsAromatic()
                and (
                    mol.GetAtomWithIdx(a).GetSymbol() == 'C'
                    or (mol.GetAtomWithIdx(a).GetSymbol() in ('O', 'S')
                        and mol.GetAtomWithIdx(a).GetFormalCharge() == 0
                        and mol.GetAtomWithIdx(a).GetTotalNumHs() == 0
                        and mol.GetAtomWithIdx(a).GetDegree() == 2)
                )
                for a in r_side
            )
            _r_has_hetero = any(
                mol.GetAtomWithIdx(a).GetSymbol() in ('O', 'S') for a in r_side
            )
            if _r_ok and _r_has_hetero:
                from ..assembly.naming_utils import apply_enclosing_marks
                from ..assembly.substituent_naming import name_substituent_fragment
                r_name = name_substituent_fragment(
                    mol, r_side, c_atom.GetIdx(), list(ring_atoms | {o_idx})
                )
                if r_name and ' ' not in r_name:
                    # R -> R-oxy via composed_alkoxy_prefix: '2-methoxyethyl' ->
                    # '2-methoxyethoxy' (contract) but a ring/locant-bearing
                    # '-yl' keeps its marks ('oxan-2-yl' -> '(oxan-2-yl)oxy',
                    # NOT 'oxan-2-oxy') (F-spell-oxy).
                    from ..assembly.substituent_enumerator import alkoxy_prefix_from_substituent
                    r_oxy = (alkoxy_prefix_from_substituent(r_name)
                             if r_name.endswith('yl') else r_name + 'oxy')
                    if r_oxy is None:
                        return None
                    return {'name': apply_enclosing_marks(r_oxy, -1),
                            'atoms': [o_idx] + r_side,
                            'is_complex': True}
                return None  # ether-bearing R not nameable -> fail closed (never drop a hetero)

        # Alkoxy fallback for O-C where C is not pure alkyl
        # (aromatic carbon, sugar ring carbon, carbonyl carbon, etc.)
        if nbr_symbol == 'C':
            c_atom = nbr
            sub_atoms = _bfs_substituent_atoms(mol, o_idx, ring_atoms)

            # Case 1: O -> aromatic C in benzene ring -> "phenoxy"
            if c_atom.GetIsAromatic() and c_atom.IsInRing():
                ring_info = mol.GetRingInfo()
                for ring in ring_info.AtomRings():
                    if c_atom.GetIdx() in ring and len(ring) == 6:
                        all_arom_c = all(
                            mol.GetAtomWithIdx(r).GetIsAromatic() and
                            mol.GetAtomWithIdx(r).GetSymbol() == 'C'
                            for r in ring
                        )
                        if all_arom_c:
                            # Wave-2 C2: a SUBSTITUTED aryloxy ring
                            # (O + C6H5 = 7 atoms; more means decoration) must
                            # carry its substituents — bare 'phenoxy' silently
                            # DROPPED them (2-phenoxy... for the dicyano BB
                            # example, -suppressed). Fail closed when
                            # the decorated name cannot be built.
                            if len(sub_atoms) > 7:
                                from .ring_substituents import (
                                    decorated_ring_substituent_name,
                                )
                                dec = decorated_ring_substituent_name(
                                    mol, ring, c_atom.GetIdx(),
                                    expected_atoms=set(sub_atoms) - {o_idx},
                                )
                                if dec is None or not dec.endswith('phenyl'):
                                    return None
                                return {'name': dec[:-6] + 'phenoxy',
                                        'atoms': sub_atoms,
                                        'is_complex': True}
                            return {'name': 'phenoxy', 'atoms': sub_atoms}

            # Case 2: O -> C(=O)R -> acyloxy group (ester linkage on benzene)
            for cn in c_atom.GetNeighbors():
                if cn.GetIdx() == o_idx:
                    continue
                if cn.GetSymbol() == 'O':
                    bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), cn.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                        # This is an ester: -O-C(=O)-R (acyloxy)
                        # Count carbons in the acyl chain (including the carbonyl C)
                        acyl_excluded = ring_atoms | {o_idx, cn.GetIdx()}
                        acyl_atoms_list, acyl_c = _collect_pure_alkyl(
                            mol, c_atom.GetIdx(), acyl_excluded
                        )
                        if acyl_atoms_list is not None and acyl_c > 0:
                            # W3-P07: use the RETAINED acyl name for
                            # the acyloxy prefix ('acetyloxy' is preferred to both
                            # the systematic 'ethanoyloxy' AND 'acetoxy'; formyloxy
                            # for HCO-O-). get_acyloxy_prefix consults the retained-
                            # acyl table, falling back to systematic '<stem>anoyloxy'
                            # for C>=3 (propanoyloxy,...). acyl_atoms_list already
                            # includes the carbonyl C; add the carbonyl O so the acid
                            # fragment is named as the free acid.
                            from .esters import (
                                get_acid_fragment_name,
                                get_acyloxy_prefix,
                            )
                            acid_atoms = list(acyl_atoms_list) + [cn.GetIdx()]
                            acid_name = get_acid_fragment_name(mol, acid_atoms)
                            if acid_name:
                                acyloxy = get_acyloxy_prefix(acid_name)
                                if acyloxy:
                                    return {
                                        'name': f"({acyloxy})",
                                        'atoms': sub_atoms,
                                        'is_complex': True,
                                    }
                        break

            # Case 3: O -> non-aromatic ring C (glycoside/sugar etc.)
            if c_atom.IsInRing():
                ring_info = mol.GetRingInfo()
                for ring in ring_info.AtomRings():
                    if c_atom.GetIdx() in ring:
                        ring_size = len(ring)
                        # 5- or 6-membered ring with O in ring -> likely glycoside
                        ring_has_o = any(
                            mol.GetAtomWithIdx(r).GetSymbol() == 'O'
                            for r in ring
                        )
                        if ring_has_o and ring_size in (5, 6):
                            # first: a recognized cyclic monosaccharide
                            # O-linked here is the compound prefix 'glycosyl'+'oxy'
                            # cited at this ring locant -- BB's own worked example
                            # is 1-[4-(beta-D-glucopyranosyloxy)phenyl]ethan-1-one
                            # (the Blue Book). Without this the skeleton-only
                            # fallback below names the sugar by its RING SIZE alone
                            # and silently discards every decoration on it (all the
                            # hydroxy groups and the CH2OH), while still reporting
                            # the whole fragment in 'atoms' -- so the coverage
                            # bookkeeping cannot see the loss.
                            from ..data.sugar_names import (
                                glycosyl_substituent_prefix,
                            )
                            _gly = glycosyl_substituent_prefix(
                                mol, set(sub_atoms), o_idx
                            )
                            if _gly:
                                return {
                                    'name': _gly,
                                    'atoms': sub_atoms,
                                    'is_complex': True,
                                }
                            # Use systematic heterocyclic names:
                            # 5-membered with O = oxolane (tetrahydrofuran)
                            # 6-membered with O = oxane (tetrahydropyran)
                            glyco_prefix = "(oxolan-2-yl)oxy" if ring_size == 5 else "(oxan-2-yl)oxy"
                            return {
                                'name': glyco_prefix,
                                'atoms': sub_atoms,
                                'is_complex': True,
                            }
                        break

            # An R that carries other atoms (a halogen,...) is named whole by
            # the shared substituent namer: '1-(chloromethoxy)-4-nitrobenzene
            # (PIN)', the Blue Book).
            _linked = _chalcogen_linked_carbon_substituent(mol, o_idx, ring_atoms)
            if _linked is not None:
                return _linked

            # Case 4: O-C that's not pure alkyl but not ring/aromatic
            # Try to count total carbons and make a best-effort alkoxy name
            all_sub = _bfs_substituent_atoms(mol, nbr.GetIdx(), ring_atoms | {o_idx})
            total_c = sum(
                1 for a in all_sub
                if mol.GetAtomWithIdx(a).GetSymbol() == 'C'
            )
            if total_c > 0:
                try:
                    alkyl_name = get_alkyl_name(total_c)
                    # composed_alkoxy_prefix: C5+ concatenates ('pentyl' ->
                    # 'pentyloxy', NOT 'pentoxy') (F-spell-oxy).
                    if alkyl_name.endswith('yl'):
                        from ..assembly.substituent_enumerator import alkoxy_prefix_from_substituent
                        oxy_name = alkoxy_prefix_from_substituent(alkyl_name)
                    else:
                        oxy_name = alkyl_name + 'oxy'
                    if oxy_name is not None:
                        return {'name': oxy_name, 'atoms': [o_idx] + all_sub}
                except (ValueError, KeyError):
                    from ..data.chain_names import get_chain_prefix
                    try:
                        cp = get_chain_prefix(total_c)
                        return {'name': cp + 'yloxy', 'atoms': [o_idx] + all_sub}
                    except (ValueError, KeyError):
                        pass

            # Last resort for O-C where all naming attempts failed.
            # If total_c > 0, use get_chain_prefix to build a systematic name.
            # Bare 'oxy' only for total_c == 0 (no carbon atoms, e.g. O-O or O-N
            # that somehow reached here -- normally handled by earlier branches).
            if total_c > 0:
                from ..data.chain_names import get_chain_prefix
                try:
                    cp = get_chain_prefix(total_c)
                    return {'name': cp + 'yloxy', 'atoms': [o_idx] + all_sub}
                except (ValueError, KeyError):
                    pass
            # total_c == 0: genuinely no carbon in substituent (O-O/O-N cases
            # normally handled by hydroperoxy/nitrooxy above).
            # Return None rather than bare 'oxy' -- OPSIN cannot parse standalone
            # 'oxy' and the caller skips None substituents gracefully.
            return None

        # Fallback for non-C neighbors (e.g., O-N in nitrate esters, O-S, O-P)
        sub_atoms = _bfs_substituent_atoms(mol, o_idx, ring_atoms)
        if len(sub_atoms) > 1:
            # For O-N(=O)=O: (nitrooxy) per IUPAC
            if nbr_symbol == 'N':
                n_atom = nbr
                o_neighbors_of_n = [
                    nb for nb in n_atom.GetNeighbors()
                    if nb.GetSymbol() == 'O' and nb.GetIdx() != o_idx
                ]
                if len(o_neighbors_of_n) >= 2:
                    return {'name': '(nitrooxy)', 'atoms': sub_atoms, 'is_complex': True}
                # O-N without multiple O on N: rare N-oxide-like linkage.
                # No standard IUPAC prefix; return None to avoid bare 'oxy'
                # which OPSIN cannot parse. Caller skips None gracefully.
                return None

            # O-S neighbors: sulfanyloxy / sulfinyloxy / sulfonyloxy
            if nbr_symbol == 'S':
                s_atom = nbr
                # Check oxidation state of sulfur (count =O bonds)
                s_double_o = sum(
                    1 for nb in s_atom.GetNeighbors()
                    if nb.GetSymbol() == 'O' and nb.GetIdx() != o_idx
                    and mol.GetBondBetweenAtoms(s_atom.GetIdx(), nb.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                )
                if s_double_o == 0:
                    # -O-S-: sulfanyloxy (IUPAC
                    return {'name': 'sulfanyloxy', 'atoms': sub_atoms}
                elif s_double_o == 1:
                    # -O-S(=O)-: sulfinyloxy
                    return {'name': 'sulfinyloxy', 'atoms': sub_atoms}
                elif s_double_o >= 2:
                    # -O-SO2-OH is the preselected prefix 'sulfooxy'
                    #, the Blue Book;:31342 "HO-SO2-O- sulfooxy
                    # (preselected prefix)"); 'sulfonyloxy' is -O-SO2- with a
                    # free valence on S, a different molecule, so an aryl
                    # hydrogen sulfate beside an -OH was refused
                    # ('4-nitro-2-(sulfonyloxy)phenol').
                    s_oh = [
                        nb for nb in s_atom.GetNeighbors()
                        if nb.GetIdx() != o_idx and nb.GetSymbol() == 'O'
                        and nb.GetDegree() == 1 and nb.GetTotalNumHs() == 1
                        and nb.GetFormalCharge() == 0
                        and mol.GetBondBetweenAtoms(
                            s_atom.GetIdx(), nb.GetIdx()).GetBondType()
                        == Chem.BondType.SINGLE
                    ]
                    if (len(s_oh) == 1 and s_atom.GetDegree() == 4
                            and s_atom.GetFormalCharge() == 0):
                        return {'name': 'sulfooxy', 'atoms': sub_atoms}
                    # -O-S(=O)(=O)-: sulfonyloxy
                    return {'name': 'sulfonyloxy', 'atoms': sub_atoms}

            # O-P neighbors: phosphonooxy (IUPAC
            if nbr_symbol == 'P':
                return {'name': 'phosphonooxy', 'atoms': sub_atoms}

            # Other non-C neighbors: return None to avoid bare 'oxy'
            # which OPSIN cannot parse. Caller skips None gracefully.
            return None

    return None


def _identify_sulfur_group(mol, s_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify sulfur-based substituent (thiol/sulfanyl, thioether, etc.).

    Args:
        mol: RDKit Mol object
        s_idx: Index of the sulfur atom attached to the ring
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name' and 'atoms', or None if unknown
    """
    s_atom = mol.GetAtomWithIdx(s_idx)
    neighbors = [n for n in s_atom.GetNeighbors() if n.GetIdx() not in ring_atoms]
    h_count = s_atom.GetTotalNumHs()

    # Simple thiol (-SH): IUPAC prefix = "sulfanyl"
    if h_count >= 1 and len(neighbors) == 0:
        return {'name': 'sulfanyl', 'atoms': [s_idx]}

    # Classify sulfur oxidation state by counting =O neighbors (non-ring)
    # Sulfoxide: S with 1 =O; Sulfone: S with 2 =O; Thioether: S with 0 =O
    o_double_neighbors = []
    c_neighbors = []
    for n in neighbors:
        if n.GetSymbol() == 'O':
            bond = mol.GetBondBetweenAtoms(s_idx, n.GetIdx())
            if bond and bond.GetBondTypeAsDouble() == 2.0:
                o_double_neighbors.append(n)
        elif n.GetSymbol() == 'C':
            c_neighbors.append(n)

    # Sulfoxide (-S(=O)-R) or Sulfone (-S(=O)(=O)-R): one C neighbor + =O neighbors
    if len(c_neighbors) == 1 and len(o_double_neighbors) in (1, 2):
        c_nbr = c_neighbors[0]
        alkyl_atoms, carbon_count = _collect_pure_alkyl(
            mol, c_nbr.GetIdx(), ring_atoms | {s_idx} | {o.GetIdx() for o in o_double_neighbors}
        )
        alkyl_name = None
        if alkyl_atoms is not None and carbon_count > 0:
            from ..assembly.substituent_naming import name_substituent_fragment
            alkyl_name = name_substituent_fragment(
                mol, alkyl_atoms, c_nbr.GetIdx(),
                list(ring_atoms | {s_idx} | {o.GetIdx() for o in o_double_neighbors})
            )
            if alkyl_name is None:
                try:
                    alkyl_name = get_alkyl_name(carbon_count)
                except (ValueError, KeyError):
                    alkyl_name = None
        if alkyl_name:
            o_atom_idxs = [o.GetIdx() for o in o_double_neighbors]
            all_sub_atoms = [s_idx] + o_atom_idxs + alkyl_atoms
            oxide_kind = 'sulfonyl' if len(o_double_neighbors) == 2 else 'sulfinyl'
            # PIN: a ring-attached sulfone/sulfoxide substituent is the
            # ACID-STEM oxide form (methanesulfonyl / ethanesulfonyl / methanesulfinyl),
            # NOT the 'alkyl'+'sulfonyl' concatenation (methylsulfonyl). Route through
            # the shared atom-aware builder -- the SAME producer the benzene-PARENT
            # path already uses (name_chalcogen_oxide_substitutive), so the two paths
            # agree. It re-derives the parent-hydride stem from the R' side of the
            # graph (atom-anchored, not a string rewrite); fall back to the
            # concatenated form only if it cannot classify that side (fail-safe).
            from ..assembly.substituent_prefix_forms import _acid_stem_oxide_prefix
            _pin = _acid_stem_oxide_prefix(mol, c_nbr.GetIdx(), s_idx, oxide_kind)
            return {'name': _pin if _pin else f'{alkyl_name}{oxide_kind}',
                    'atoms': all_sub_atoms}

    # Thioether (-S-R): named as alkylsulfanyl (methylsulfanyl, etc.)
    if len(c_neighbors) == 1 and len(o_double_neighbors) == 0:
        alkyl_atoms, carbon_count = _collect_pure_alkyl(
            mol, c_neighbors[0].GetIdx(), ring_atoms | {s_idx}
        )
        if alkyl_atoms is not None and carbon_count > 0:
            from ..assembly.substituent_naming import name_substituent_fragment
            alkyl_name = name_substituent_fragment(
                mol, alkyl_atoms, c_neighbors[0].GetIdx(), list(ring_atoms | {s_idx})
            )
            if alkyl_name is None:
                try:
                    alkyl_name = get_alkyl_name(carbon_count)
                except (ValueError, KeyError):
                    alkyl_name = None
            if alkyl_name:
                return {
                    'name': f'{alkyl_name}sulfanyl',
                    'atoms': [s_idx] + alkyl_atoms,
                }

    # Disulfanyl (-S-SH) or dithio linkages
    if len(neighbors) == 1 and neighbors[0].GetSymbol() == 'S':
        other_s = neighbors[0]
        if other_s.GetTotalNumHs() >= 1:
            return {'name': 'disulfanyl', 'atoms': [s_idx, other_s.GetIdx()]}

    # R-S- whose R carries other atoms ('[(chloromethyl)sulfanyl]',
    # method (1), the Blue Book): R is named whole by the shared namer.
    _linked = _chalcogen_linked_carbon_substituent(mol, s_idx, ring_atoms)
    if _linked is not None:
        return _linked

    # Generic fallback for other S-based substituents
    sub_atoms = _bfs_substituent_atoms(mol, s_idx, ring_atoms)
    if sub_atoms:
        return {'name': 'sulfanyl', 'atoms': sub_atoms}

    return None


def _identify_generic_carbon_substituent(
    mol, start_idx: int, sub_atoms: List[int], ring_atoms: Set[int]
) -> Optional[Dict]:
    """
    Fallback identification for carbon-based substituents that are not
    simple alkyl or known functionalized chains.

    Handles haloalkyl groups (trifluoromethyl, dichloromethyl, etc.)
    and other mixed-atom carbon substituents.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the carbon attached to the ring
        sub_atoms: All atom indices in the substituent
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name' and 'atoms', or None
    """
    start_atom = mol.GetAtomWithIdx(start_idx)

    # Count halogens and carbons
    halogen_counts = {'F': 0, 'Cl': 0, 'Br': 0, 'I': 0}
    carbon_count = 0
    other_count = 0

    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym == 'C':
            carbon_count += 1
        elif sym in halogen_counts:
            halogen_counts[sym] += 1
        elif sym != 'H':
            other_count += 1

    # Pure haloalkyl: only C and halogens (no other heteroatoms)
    if other_count == 0 and any(halogen_counts.values()):
        total_halogens = sum(halogen_counts.values())

        if carbon_count == 1:
            # Single C with halogens: trifluoromethyl, dichloromethyl, etc.
            halogen_parts = []
            halogen_prefix_map = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
            for hal in ['Br', 'Cl', 'F', 'I']:  # alphabetical by prefix
                count = halogen_counts[hal]
                if count > 0:
                    mp = get_multiplier_prefix(count, halogen_prefix_map[hal]) if count > 1 else ''
                    halogen_parts.append(f'{mp}{halogen_prefix_map[hal]}')

            name = '(' + ''.join(halogen_parts) + 'methyl)'
            return {'name': name, 'atoms': sub_atoms, 'is_complex': True}

    return None


def _identify_alkyl_group(mol, start_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify alkyl substituent.

    Uses BFS to find all connected carbons and counts total carbons
    to determine alkyl name.
    """
    # BFS to find all atoms in the substituent
    visited = {start_idx}
    queue = deque([start_idx])
    carbon_count = 0
    all_atoms = []

    while queue:
        current_idx = queue.popleft()
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        if current_atom.GetSymbol() == 'C':
            carbon_count += 1
        else:
            # Non-carbon in the chain - this is not a simple alkyl
            # For a phase, we'll handle more complex substituents
            pass

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_atoms:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    # Check if this is a pure alkyl (only carbons and hydrogens)
    for idx in all_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() not in ('C', 'H'):
            # Contains a heteroatom -> not a simple alkyl. DECLINE so the recursive
            # `name_substituent` fallback (below, ~line 1401) names it from
            # structure. (B2 bug 2: a vinyl-STARTED fragment carrying a
            # downstream heteroatom -- -CH=CH-CH2OH in p-coumaryl alcohol -- was
            # mislabelled 'ethenyl' here with all_atoms, DROPPING the CH2OH tail ->
            # `4-ethenylphenol` (atom drop, -suppressed). A true vinyl
            # -CH=CH2 is all-carbon, never enters this branch, and is named by the
            # multiple-bond decline below + the recursive fallback.)
            return None

    # Wave2 constitution-conservation guard: the carbon-count name below
    # ('octyl' from carbon_count==8) is only honest for an ACYCLIC SATURATED
    # all-C fragment. A styryl arm -CH=CH-C6H5 walked into the far aromatic
    # ring and was mislabelled '(4E)-4-octyl...' (8 C counted through the ring,
    # the phenyl + the C=C both dropped — a different molecule that kept a
    # nonsensical (4E) descriptor). Ring membership or any multiple bond in the
    # fragment -> decline; the shared substituent namer (name_substituent_
    # fragment) or fail-closed handles it. A plain vinyl -CH=CH2 is declined here
    # too (its C=C trips the multiple-bond guard below) and named 'ethenyl' by the
    # recursive `name_substituent` fallback.
    ri = mol.GetRingInfo()
    if any(ri.NumAtomRings(i) > 0 for i in all_atoms):
        return None
    for idx in all_atoms:
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            j = nb.GetIdx()
            if j in all_atoms:
                b = mol.GetBondBetweenAtoms(idx, j)
                if b is not None and b.GetBondTypeAsDouble() != 1.0:
                    return None

    # Check for branched alkyl (isopropyl, tert-butyl, etc.)
    name = _get_alkyl_name(mol, start_idx, carbon_count, ring_atoms, sub_atoms=all_atoms)
    if name:
        return {'name': name, 'atoms': all_atoms}

    return None


def _get_alkyl_name(mol, start_idx: int, carbon_count: int, ring_atoms: Set[int],
                    sub_atoms: list = None) -> Optional[str]:
    """
    Get the name for an alkyl substituent.

    Uses name_substituent_fragment for branched/complex substituents,
    falls back to get_alkyl_name(carbon_count) for linear alkyls.

    Args:
        mol: RDKit Mol object
        start_idx: First carbon of the substituent (bonded to ring)
        carbon_count: Number of carbons in the substituent
        ring_atoms: Set of ring atom indices
        sub_atoms: Optional list of all atom indices in the substituent
    """
    from ..assembly.substituent_naming import name_substituent_fragment

    # If we have atom context, try name_substituent_fragment (handles retained
    # names like sec-butyl/isopropyl and branched substituents)
    if sub_atoms is not None and len(sub_atoms) > 0:
        recursive_name = name_substituent_fragment(
            mol, sub_atoms, start_idx, list(ring_atoms)
        )
        if recursive_name:
            return recursive_name

    # Fallback: simple carbon-count naming
    try:
        return get_alkyl_name(carbon_count)
    except (ValueError, KeyError):
        return None


def _identify_functionalized_chain(mol, start_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify a chain with functional group attached to benzene ring.

    This handles cases like `-CCCC(=O)O` (butanoic acid chain) that the
    simple alkyl detection misses because they contain heteroatoms.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the first atom attached to the ring
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name' (a proper substituent name), 'atoms', 'chain_length',
        'functional_group', or None if not a functionalized chain or cannot be named.
    """
    # BFS to find all atoms in the substituent
    visited = {start_idx}
    queue = deque([start_idx])
    all_atoms = []
    carbon_count = 0
    has_heteroatom = False

    while queue:
        current_idx = queue.popleft()
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        symbol = current_atom.GetSymbol()
        if symbol == 'C':
            carbon_count += 1
        elif symbol != 'H':
            has_heteroatom = True

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_atoms:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    # Only consider if there's a heteroatom (indicating functional group)
    if not has_heteroatom:
        return None

    # Check for common functional groups
    functional_group = _detect_chain_functional_group(mol, all_atoms)
    if not functional_group:
        return None

    # Wave2 constitution-conservation guard safety): the name
    # below is generated from carbon_count ALONE, so it is only honest when
    # the collected branch IS a linear, unbranched, acyclic, saturated
    # all-C chain bearing exactly one terminal FG. Anything else used to be
    # silently collapsed into a fabricated linear chain that dropped or
    # mutated atoms (Ar-CH2-SiH2-CH2-Ar' emitted as '8-carboxyoctyl': the Si
    # dropped, the far ring's carbons re-linearised — a different molecule
    # passing the coverage gate at 0.79). Reject -> the caller falls through
    # to the exact recursive namers or fails closed.
    _FG_O_COUNT = {'carboxylic_acid': 2, 'aldehyde': 1, 'alcohol': 1}
    _o_count = 0
    for a_idx in all_atoms:
        _a = mol.GetAtomWithIdx(a_idx)
        if _a.IsInRing() or _a.GetSymbol() not in ('C', 'O'):
            return None
        if _a.GetSymbol() == 'O':
            _o_count += 1
    if _o_count != _FG_O_COUNT[functional_group]:
        return None
    # Walk the carbon skeleton from the attachment: it must be one unbranched
    # path covering every collected carbon, through single C-C bonds only,
    # with the FG on the terminal carbon.
    _branch_set = set(all_atoms)
    _path = [start_idx]
    _seen_c = {start_idx}
    _cur = start_idx
    while True:
        _nxt = [
            n.GetIdx() for n in mol.GetAtomWithIdx(_cur).GetNeighbors()
            if n.GetIdx() in _branch_set and n.GetSymbol() == 'C'
            and n.GetIdx() not in _seen_c
        ]
        if len(_nxt) > 1:
            return None  # branched carbon skeleton
        if not _nxt:
            break
        _b = mol.GetBondBetweenAtoms(_cur, _nxt[0])
        if _b is None or _b.GetBondTypeAsDouble() != 1.0:
            return None  # unsaturation would be dropped by the alkyl stem
        _cur = _nxt[0]
        _seen_c.add(_cur)
        _path.append(_cur)
    if len(_path) != carbon_count:
        return None  # carbons not on one contiguous path (e.g. ether bridge)
    _t_o = [
        n for n in mol.GetAtomWithIdx(_path[-1]).GetNeighbors()
        if n.GetIdx() in _branch_set and n.GetSymbol() == 'O'
    ]
    if len(_t_o) != _FG_O_COUNT[functional_group]:
        return None  # FG not terminal (mid-chain OH would get a wrong locant)

    # Generate a proper substituent name based on FG type and chain length
    sub_name = _name_functionalized_chain_substituent(carbon_count, functional_group)
    if not sub_name:
        # Cannot produce a valid name; return None so caller can handle gracefully
        return None

    # Wrap MARKLESS compound substituent names in parentheses per IUPAC
    #: "hydroxymethyl" -> "(hydroxymethyl)". w2f p1 (research C.6):
    # a name that ALREADY carries an enclosing mark ('(4-bromophenyl)(chloro)
    # methyl', '(oxan-2-yl)oxy') must NOT take plain parens here — the naive
    # startswith('(')/endswith(')') test misfires on interior/trailing-stem
    # marks and produced the non-PIN parens-in-parens '((...)...)'. Left BARE,
    # it flows to the escalating consumers (format_substituent_prefix / the
    # _omit branch), which apply brackets ('[(4-bromophenyl)(chloro)
    # methyl]').
    from ..assembly.naming_utils import needs_brackets
    is_compound = needs_brackets(sub_name)
    if is_compound and '(' not in sub_name and '[' not in sub_name:
        sub_name = f'({sub_name})'

    return {
        'name': sub_name,
        'atoms': all_atoms,
        'chain_length': carbon_count,
        'functional_group': functional_group,
        'is_complex': is_compound,
    }


# Mapping from chain length (carbons) to substituent prefix stem
_CHAIN_SUB_STEMS = {
    1: "methyl", 2: "ethyl", 3: "propyl", 4: "butyl", 5: "pentyl",
    6: "hexyl", 7: "heptyl", 8: "octyl", 9: "nonyl", 10: "decyl",
}

# Mapping from functional group type to substituent prefix modifier
# These convert a chain with FG into a proper IUPAC prefix substituent name
_FG_SUB_PREFIX = {
    'carboxylic_acid': {
        # -C(=O)OH chain: named as "carboxylalkyl" (e.g., 2-carboxyethyl for -CH2CH2COOH)
        # or simply use the acyl prefix approach
        1: "carboxy",            # just -COOH
        2: "carboxymethyl",      # -CH2COOH
        3: "2-carboxyethyl",     # -CH2CH2COOH
        4: "3-carboxypropyl",    # -(CH2)3COOH
        5: "4-carboxybutyl",     # -(CH2)4COOH
    },
    'aldehyde': {
        # -CHO on the ring carbon (carbon NOT absorbable) -> formyl.
        1: "formyl",             # -CHO
        # Moving-base-atom: the -CHO carbon is absorbed into the chain and expressed
        # as 'oxo' at the terminal carbon (= chain length, opposite the attachment).
        # '1-oxo...yl' (oxo at the acyl carbon) is the disfavoured CAS form note m).
        2: "2-oxoethyl",         # -CH2CHO
        3: "3-oxopropyl",        # -(CH2)2CHO
    },
    'alcohol': {
        # The hydroxy is numbered from the free valence: the attachment is
        # locant 1, so -OH on the far carbon of an N-carbon chain is locant N). The
        # 1-carbon form elides the locant (hydroxymethyl). single-FG-substituent
        # fix: this map mirrors the structure-based namer (_name_polyfunctional_
        # acyclic_substituent, now single-FG) — the Phase-172 NOTE deferred the
        # locant because the sibling parent_to_prefix path dropped it too; that path
        # is now fixed (Tier 1.96 / Step 2c-poly), so both are consistently located.
        1: "hydroxymethyl",      # -CH2OH (1-carbon: locant elided)
        2: "2-hydroxyethyl",     # -CH2CH2OH
        3: "3-hydroxypropyl",    # -(CH2)2CH2OH
    },
}


def _name_functionalized_chain_substituent(carbon_count: int, functional_group: str) -> Optional[str]:
    """
    Generate a proper IUPAC substituent name for a functionalized chain.

    Args:
        carbon_count: Number of carbon atoms in the chain
        functional_group: Type of functional group ('carboxylic_acid', 'aldehyde', 'alcohol')

    Returns:
        Substituent prefix name string, or None if cannot be named
    """
    # Try specific FG + chain length lookup
    fg_map = _FG_SUB_PREFIX.get(functional_group, {})
    if carbon_count in fg_map:
        return fg_map[carbon_count]

    # Fallback: generic naming based on FG type
    if functional_group == 'carboxylic_acid' and carbon_count > 0:
        if carbon_count == 1:
            return "carboxy"
        # The carboxy carbon is NOT a backbone carbon, so the backbone is
        # (N-1) carbons; the carboxy sits on the carbon farthest from the attachment
        # = backbone locant (N-1), numbered from the free valence).
        backbone = carbon_count - 1
        alkyl = _CHAIN_SUB_STEMS.get(backbone)
        if alkyl:
            return f"{backbone}-carboxy{alkyl}" if backbone > 1 else f"carboxy{alkyl}"

    if functional_group == 'alcohol' and carbon_count > 0:
        # Hydroxy on the far carbon, numbered from the free valence: locant
        # = carbon_count; the 1-carbon form elides it (hydroxymethyl). single-FG
        # fix — consistent with the structure-based namer (the Phase-172 deferral is
        # resolved now that parent_to_prefix / Tier 1.96 are also located).
        alkyl = _CHAIN_SUB_STEMS.get(carbon_count)
        if alkyl:
            return f"{carbon_count}-hydroxy{alkyl}" if carbon_count > 1 else f"hydroxy{alkyl}"

    if functional_group == 'aldehyde' and carbon_count > 0:
        if carbon_count == 1:
            return "formyl"  # -CHO on the ring carbon; not absorbable
        alkyl = _CHAIN_SUB_STEMS.get(carbon_count)
        if alkyl:
            # Moving-base-atom: -CHO carbon absorbed -> 'oxo' at the terminal carbon
            # (= carbon_count). '1-oxo...yl' acyl form is non-PIN note m).
            return f"{carbon_count}-oxo{alkyl}"

    # Cannot name this chain
    return None


def _detect_chain_functional_group(mol, chain_atoms: List[int]) -> Optional[str]:
    """
    Detect what functional group is on a chain.

    Checks for carboxylic acid, alcohol, and aldehyde patterns.

    Args:
        mol: RDKit Mol object
        chain_atoms: List of atom indices in the chain

    Returns:
        Functional group name ('carboxylic_acid', 'alcohol', 'aldehyde') or None
    """
    chain_set = set(chain_atoms)

    # Check for carboxylic acid pattern: C(=O)O with O having H
    for idx in chain_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue

        neighbors = list(atom.GetNeighbors())
        o_double = None
        o_single = None

        for nbr in neighbors:
            if nbr.GetSymbol() == 'O':
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    o_double = nbr
                elif bond.GetBondType() == Chem.BondType.SINGLE:
                    if nbr.GetTotalNumHs() >= 1:  # -OH
                        o_single = nbr

        if o_double and o_single:
            return 'carboxylic_acid'

    # Check for aldehyde pattern: C(=O)H (must check before alcohol)
    for idx in chain_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'C' and atom.GetTotalNumHs() >= 1:
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'O':
                    bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        return 'aldehyde'

    # Check for alcohol pattern: C-O-H
    for idx in chain_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() >= 1:
            # Check it's not part of carboxylic acid (already checked above)
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'C':
                    c_atom = nbr
                    has_double_o = False
                    for c_nbr in c_atom.GetNeighbors():
                        if c_nbr.GetSymbol() == 'O' and c_nbr.GetIdx() != atom.GetIdx():
                            bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), c_nbr.GetIdx())
                            if bond.GetBondType() == Chem.BondType.DOUBLE:
                                has_double_o = True
                    if not has_double_o:
                        return 'alcohol'

    return None


def _molecule_principal_group(mol) -> Optional[str]:
    """The molecule-level principal characteristic group name, or None.

    For call sites that have no ``MolecularFeatures`` to read
    ``features.principal_group`` from. Routes through the same
    ``seniority.get_principal_group`` authority the namer uses, so the answer
    cannot diverge from ``features.principal_group``. Fails closed to ``None``
    (= legacy numbering) on any perception failure.
    """
    return molecule_principal_group_and_fgs(mol)[0]


def molecule_principal_group_and_fgs(mol) -> Tuple[Optional[str], Optional[Dict]]:
    """The molecule-level principal group AND the detected FGs, in ONE perception.

    ``principal_group_ring_atoms`` needs both, and detecting the functional groups
    twice for one molecule is the only reason the two were ever separate. Routes
    through the same ``seniority.get_principal_group`` authority the namer uses, so
    the answer cannot diverge from ``features.principal_group``. Fails closed to
    ``(None, None)`` (= legacy numbering) on any perception failure.
    """
    try:
        from ..perception.functional_groups import detect_functional_groups
        from .seniority import get_principal_group
        detected = detect_functional_groups(mol)
        pg_name, _ = get_principal_group(mol, detected)
        return pg_name, detected
    except Exception:
        return None, None


def principal_group_ring_atoms(
    mol,
    ring_atoms: Tuple[int, ...],
    substituents: Dict[int, List[Dict]],
    principal_group: Optional[str] = None,
    detected_fgs: Optional[Dict] = None,
) -> Set[int]:
    """(c): the ring atoms bearing the PRINCIPAL characteristic group.

    This is the input to ``orient_benzene``'s criterion-0 tier. §
    "NUMBERING" (heading at the Blue Book) states at the Blue Book: "When several structural
    features appear in cyclic and acyclic compounds, low locants are assigned to
    them in the following decreasing order of seniority:". That order ranks
    **(c) principal characteristic groups and free valences (suffixes);**
    (the Blue Book) four places ABOVE **(f) detachable alphabetized prefixes, all
    considered together in a series of increasing numerical order;** (the Blue Book) and
    five above **(g) lowest locants for the substituent cited first as a prefix in
    the name;** (the Blue Book). So the suffix locant set is minimised FIRST and the
    prefixes take what is left. Benzene has no fixed numbering (a, the Blue Book) and no
    indicated hydrogen (b, the Blue Book), so (c) decides whenever it applies.

    Every caller used to derive this set from the ``is_suffix`` marker alone,
    which made it EMPTY for phenols: a ring -OH is still spelled as the *prefix*
    ``hydroxy`` when the ring is numbered and is promoted to the ``-ol`` suffix
    only later, inside ``name_substituted_benzene``. Criterion (c) therefore never
    ran and criterion (f) handed locant 1 to the prefix -- emitting
    ``1-chlorobenzene-2,3,4,5,6-pentol`` for the PIN
    ``6-chlorobenzene-1,2,3,4,5-pentol``.

    Whether the group IS promoted is not decided here: it is asked of
    ``benzene_prefix_suffix_promotion``, the single authority
    ``name_substituted_benzene`` also reads.

    ⚠ Phase C Task 9b: this used to ask the seniority tables directly --
    "does ``get_suffix(pg, is_ring=True)`` exist and is ``get_prefix(pg)`` on this
    ring?" -- which is a test for promotion ELIGIBILITY, not promotion. **76** of
    the 136 seniority entries have both a ring suffix and a prefix form and only
    **two** are ever promoted, so for the other 74 criterion (c) was minimising the
    locant of a group the name still spells as a detachable prefix, inverting (f)
    and (g). The decision is still LOCANT-FREE -- it depends only on WHICH groups
    are present, never on where -- which is what lets it be consulted BEFORE the
    ring is numbered.

     is respected: only the SENIOR group is the principal characteristic
    group. ``principal_group`` is the molecule-level answer from
    ``seniority.get_principal_group``, so a ring carrying both -ol and -thiol
    yields the -ol atoms only (matching the emitted ``2-sulfanylphenol``, whose
    numbering hint previously contradicted the name by anchoring the thiol).

    FAILS TOWARD CURRENT BEHAVIOUR: whenever the answer is not available (no
    principal group, no ring suffix for it, nothing promoted,...) this returns the
    legacy ``is_suffix`` union, so the change is a strict improvement rather than a
    coin flip. In particular a ring with NO
    principal characteristic group -- ``Cc1c(C)c(C)c(C)c(C)c1Cl`` -- has
    ``principal_group is None`` and returns the empty set, leaving criterion (c)
    vacuous and (f)+(g) legitimately in charge of
    ``1-chloro-2,3,4,5,6-pentamethylbenzene``.

    Args:
        mol: RDKit Mol object (unused today; kept so callers pass the full context
            and a future rule can consult the graph without a signature change).
        ring_atoms: The benzene ring's atom indices.
        substituents: Dict from ``get_benzene_substituents``.
        principal_group: ``features.principal_group`` -- the molecule-level PCG
            name from ``seniority.get_principal_group``. ``None`` disables the
            promotion tier (legacy behaviour).
        detected_fgs: ``features.functional_groups``, used only for the shared
            functional-class guard.

    Returns:
        Set of ring atom indices bearing the principal characteristic group;
        empty when criterion (c) does not apply.
    """
    from .seniority import get_suffix

    if not isinstance(substituents, dict):
        return set()

    ring_set = set(ring_atoms or ())
    suffix_atoms: Dict[str, Set[int]] = defaultdict(set)
    prefix_atoms: Dict[str, Set[int]] = defaultdict(set)
    amine_candidate_atoms: Set[int] = set()

    for atom_idx, subs in substituents.items():
        if ring_set and atom_idx not in ring_set:
            continue
        if not isinstance(subs, list):
            continue
        for sub in subs:
            if not isinstance(sub, dict):
                continue
            if sub.get('is_suffix'):
                if sub.get('suffix_name'):
                    suffix_atoms[sub['suffix_name']].add(atom_idx)
            elif sub.get('name'):
                prefix_atoms[sub['name']].add(atom_idx)
                if sub.get('amine_candidate') is not None:
                    amine_candidate_atoms.add(atom_idx)

    # The legacy answer, and the fail-closed floor for every early return below.
    legacy: Set[int] = set()
    for _atoms in suffix_atoms.values():
        legacy |= _atoms

    if not principal_group:
        return legacy

    # A group named by functional class has no ring suffix at all, so the tables
    # answer "no PCG suffix" for it directly.
    pg_ring_suffix = get_suffix(principal_group, is_ring=True)
    if not pg_ring_suffix:
        return legacy

    # Already spelled as a suffix at orientation time: take THAT group's atoms
    # only -- a co-occurring junior suffix is not the principal group).
    if pg_ring_suffix in suffix_atoms:
        return set(suffix_atoms[pg_ring_suffix])

    # Still spelled as a prefix. Anchor it ONLY if the shared authority says the
    # NAME will actually promote it to exactly this suffix.
    #
    # ⚠ Phase C Task 9b (C-2): this used to be `get_prefix(principal_group) in
    # prefix_atoms`, i.e. "the principal group HAS a prefix form and it is on this
    # ring". That is true for 76 seniority entries and only 2 are ever promoted, so
    # criterion (c) was minimising the locant of a group the name still spells as a
    # detachable prefix -- which inverts (f) and (g). The anchor and the promotion
    # are now ONE decision instead of two that merely happened to agree for hydroxy.
    promoted_suffix, promoted_prefixes = benzene_prefix_suffix_promotion(
        set(suffix_atoms), set(prefix_atoms), detected_fgs,
        has_amine_candidates=bool(amine_candidate_atoms),
    )
    if promoted_suffix != pg_ring_suffix:
        return legacy

    if promoted_suffix == 'amine':
        # Identified by the promotion marker, not by a prefix name: the prefix may
        # be 'amino', '(N-methylamino)', 'anilino',... and only the marked ones
        # are promoted.
        return set(amine_candidate_atoms)

    anchored: Set[int] = set()
    for _name in promoted_prefixes:
        anchored |= prefix_atoms.get(_name, set())
    return anchored or legacy


def benzene_prefix_citation_locants(
    oriented: List[int],
    substituents: Dict[int, List[Dict]],
    principal_group_positions: Optional[Set[int]] = None,
) -> Tuple[Tuple[int, ...], ...]:
    """(g): the per-prefix locant sets, in alphabetical CITATION order.

    §** "NUMBERING"** (``the Blue Book Blue Book``) criterion
    **(g)** (``:3307``) reads verbatim:

        (g) lowest locants for the substituent cited first as a prefix in the
        name;

    worked at ``:3315`` ``4-methyl-5-nitrooctanedioic acid (PIN)`` and, decisively
    for a ring, at ``:3317`` ``1-methyl-4-nitronaphthalene (PIN) (not
    4-methyl-1-nitronaphthalene)``. The same rule is stated a second time, with a
    *benzene* worked example, as §** "Low locants are assigned to the
    prefix cited first in the name"** (``:26085``): ``1-bromo-2-chloroethane
    (PIN)`` and ``1-azido-4-isocyanatobenzene (PIN)`` (``:26094``).

    "Cited first" is the alphanumerical order, so the sequence is keyed
    on the shared ``alpha_sort_key`` (di/tri ignored, iso/neo/cyclo/sec/tert
    included) rather than a hand-rolled sort. Every candidate orientation of one
    molecule carries the SAME set of prefix names, so only the locants differ and
    the returned tuple can be compared lexicographically.

    A substituent is a prefix here unless it is the suffix:
      * atoms in ``principal_group_positions`` bear the principal characteristic
        group, so the group itself is criterion (c)'s business, not (g)'s -- BUT any
        italic-*N* substituent it carries IS cited as a prefix in the name
        (``N-(4-aminophenyl)-...``) and therefore enters at that locant. Without
        this, two suffix instances distinguished ONLY by their N-substituents give
        an empty (g) key and the tie falls through to the canonical last resort,
        which orders by symmetry class rather than by citation -- measured: it named
        ``Nc1ccc(Nc2ccc(Nc3ccccc3)cc2)cc1`` with ``(4-aminophenyl)`` on the HIGHER
        nitrogen, though ``aminoanilino`` is cited before ``anilino``.
      * an ``is_suffix`` substituent outside that set is a JUNIOR suffix that
        ``name_substituted_benzene`` demotes back to its own prefix form,
        so it IS cited as a prefix and enters under ``_SUFFIX_TO_PREFIX``;
      * when no PCG set is supplied every ``is_suffix`` substituent is the
        suffix, so all of them are excluded.

    For a single-instance suffix this is a no-op: criterion (c) has already fixed
    that locant, so the entry is identical in every surviving candidate.

    Returns:
        Tuple of locant tuples, ordered by the prefix's citation position.
    """
    pcg_set = set(principal_group_positions or ())
    by_name: Dict[str, List[int]] = defaultdict(list)

    def _n_substituent_names(sub):
        """The italic-N prefixes this suffix instance will cite, if any."""
        names = sub.get('n_substituents')
        if not names:
            candidate = sub.get('amine_candidate')
            if isinstance(candidate, dict):
                names = candidate.get('n_substituents')
        return names or ()

    for i, atom_idx in enumerate(oriented):
        for sub in substituents.get(atom_idx) or ():
            if not isinstance(sub, dict):
                continue
            if atom_idx in pcg_set:
                for n_name in _n_substituent_names(sub):
                    if n_name:
                        by_name[n_name].append(i + 1)
                continue
            if sub.get('is_suffix'):
                if not pcg_set:
                    continue
                suffix_name = sub.get('suffix_name')
                name = _SUFFIX_TO_PREFIX.get(suffix_name, suffix_name)
            else:
                name = sub.get('name')
            if name:
                by_name[name].append(i + 1)

    # Ordered by the SHARED total order, not by the raw ``alpha_sort_key``
    # string. 's preamble (:3442) makes this a two-stage comparison --
    # "Nonitalic Roman letters are considered first... When all the Roman
    # letters are identical, the set of locants... are compared" -- and a single
    # key string conflates the stages, because it still carries digits and,
    # before the enclosing-mark strip, brackets. `4-[(1R)-1-chloroethyl]phenoxy`
    # keyed as `1-chloroethylphenoxy`, whose leading '1' (ASCII 49) sorts under
    # every letter, so it was cited ahead of `chloroethyl` and took locant 1 --
    # the wrong numbering for gold row W2F-P8-01. `prefix_citation_sort_key`
    # separates the stages (letters, then locants, then (j) configuration)
    # and is the same authority the rest of the cascade consults.
    from ..assembly.naming_utils import prefix_citation_sort_key
    return tuple(
        tuple(sorted(by_name[name]))
        for name in sorted(by_name, key=prefix_citation_sort_key)
    )


def _canonical_orbit_key(mol, oriented: List[int]) -> Tuple[int, ...]:
    """The LAST-RESORT tie-break: canonical symmetry-class ranks, never input order.

    ``Chem.CanonicalRankAtoms(mol, breakTies=False)`` is a canonical graph
    invariant -- verified invariant under ``RenumberAtoms`` -- so this key depends
    only on the STRUCTURE, which is what a numbering cascade must terminate in. A
    cascade that ends in candidate-enumeration order is not an implementation of
    : it is a coin flip that happens to agree on the molecules tested, and
    that is exactly how the same molecule came to be named
    ``6-chloro-2-methylphenol`` from one SMILES and ``2-chloro-6-methylphenol``
    from another.

    With ``breakTies=False`` symmetry-equivalent atoms share a rank, so two
    orientations tie here only when they are related by an automorphism of the
    ranked graph -- i.e. they carry the same substituent at every locant and
    therefore produce the SAME name. The order is total up to name equality.

    Fails closed to an empty key (no discrimination) if RDKit cannot rank the
    molecule; the earlier tiers have then already reduced the field to
    name-identical candidates.
    """
    try:
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    except Exception:
        return ()
    try:
        return tuple(ranks[a] for a in oriented)
    except (IndexError, TypeError):
        return ()


def orient_benzene(
    mol,
    ring_atoms: Tuple[int, ...],
    substituents: Dict[int, List[Dict]],
    principal_group_positions: Optional[Set[int]] = None,
) -> List[int]:
    """
    Orient benzene ring to give lowest locants to substituents.

    §** "NUMBERING"** (``the Blue Book Blue Book``) is an ORDERED
    cascade. Benzene has no fixed numbering (a, ``:3227``), no indicated hydrogen
    (b, ``:3246``), no added indicated hydrogen (d, ``:3270``), no
    saturation/unsaturation choice (e, ``:3288``) and no skeletal atom in a
    nonstandard valence state (h, ``:3320``), so the applicable criteria are
    exactly (c), (f), (g) -- and the cascade must then TERMINATE IN THE STRUCTURE,
    never in the order RDKit happened to enumerate the ring:

    0. **(c)** ``:3256`` *principal characteristic groups and free valences
       (suffixes)* -- lowest locant SET for the principal characteristic group,
       four places above (f). Applied only when ``principal_group_positions`` is
       supplied; ``None`` disables the tier.
    1. **(f)** ``:3301`` *detachable alphabetized prefixes, all considered
       together in a series of increasing numerical order*, worked at ``:3305``
       (*"the locant set '4,5,8' is lower than '4,7,8'"*) -- first point of
       difference over the substituted positions.
    2. **(g)** ``:3307`` *lowest locants for the substituent cited first as a
       prefix in the name*, worked at ``:3315`` and ``:3317``
       (``1-methyl-4-nitronaphthalene (PIN) (not 4-methyl-1-nitronaphthalene)``);
       restated as § ``:26085`` with the benzene PIN
       ``1-azido-4-isocyanatobenzene`` (``:26094``). See
       ``benzene_prefix_citation_locants``.
    3. **last resort** canonical symmetry-class orbits
       (``_canonical_orbit_key``) -- a structure-derived invariant, so the answer
       cannot depend on how the molecule was spelled.

    ⚠ Phase C Task 9b: tiers 2 and 3 are new. Before them the cascade ended
    at (f) plus a crude "position 1 goes to the alphabetically first substituent
    AT position 1" heuristic, and fell through to candidate-enumeration order --
    which made the emitted name a function of the input SMILES. Tier 2 subsumes
    that heuristic: a (g) tie means every prefix holds the same locants in every
    surviving candidate, so the substituent at position 1 is the same too and the
    heuristic could not have discriminated either.

    Monosubstituted rings short-circuit (the substituent is at locant 1), and an
    unsubstituted ring returns the input order because every orientation is the
    same name.

    Args:
        mol: RDKit Mol object (required for the canonical last-resort tier)
        ring_atoms: Tuple of atom indices in the benzene ring (ordered)
        substituents: Dict from get_benzene_substituents
        principal_group_positions: Optional set of ring atom indices bearing the
            principal characteristic group (c)). Default None = no PCG
            tier.

    Returns:
        List of ring atom indices reordered so position 1 is first
    """
    ring_list = list(ring_atoms)
    n = len(ring_list)  # Should be 6

    # Get substituted positions (indices in ring_list)
    substituted_indices = []
    for i, atom_idx in enumerate(ring_list):
        if atom_idx in substituents:
            substituted_indices.append(i)

    if not substituted_indices:
        # Unsubstituted benzene - any orientation is fine
        return ring_list

    if len(substituted_indices) == 1:
        # Monosubstituted - that position becomes 1
        start_pos = substituted_indices[0]
        return _rotate_list(ring_list, start_pos)

    # Polysubstituted - every starting position in both directions (12 for a ring
    # of 6). The cascade below narrows these; it must never terminate while more
    # than one survives, because their enumeration order is the input atom order.
    candidates = []

    for start_pos in range(n):
        for direction in [1, -1]:  # 1 = clockwise, -1 = counterclockwise
            oriented = _build_oriented_ring(ring_list, start_pos, direction)
            candidates.append((oriented, _calculate_locants(oriented, substituents)))

    # --- Criterion 0 (c)): principal characteristic group lowest locant ---
    # Keep only the orientations giving the PCG its lowest locant set, BEFORE the
    # substituent-set tier below. No-op (byte-identical) when no PCG is supplied.
    if principal_group_positions:
        pcg_set = set(principal_group_positions)

        def _pcg_locants(oriented):
            return sorted(i + 1 for i, a in enumerate(oriented) if a in pcg_set)

        best_pcg = None
        for oriented, _ in candidates:
            pl = _pcg_locants(oriented)
            if pl and (best_pcg is None or _compare_locant_sets(pl, best_pcg) < 0):
                best_pcg = pl
        if best_pcg is not None:
            candidates = [c for c in candidates if _pcg_locants(c[0]) == best_pcg]

    # --- Criterion (f), the Blue Book, worked at:3305): the detachable
    # alphabetized prefixes considered TOGETHER, first point of difference.
    # Criterion (c) above has already fixed the suffix locant set, so minimising
    # the substituted positions as one series is (f) over what remains -- and when
    # no PCG was supplied every substituent IS a detachable prefix, so it is (f)
    # exactly.
    best_locants = None
    for _, locants in candidates:
        if best_locants is None or _compare_locant_sets(locants, best_locants) < 0:
            best_locants = locants

    best_candidates = [
        oriented for oriented, locants in candidates if locants == best_locants
    ]

    if len(best_candidates) == 1:
        return best_candidates[0]

    # --- Criterion (g) (the Blue Book, worked at:3315/:3317; restated
    # the Blue Book with the benzene PIN '1-azido-4-isocyanatobenzene' the Blue Book):
    # lowest locants for the substituent cited first as a prefix in the name.
    # Compared lexicographically over the alphabetical citation sequence, so a tie
    # on the first-cited prefix falls to the second, and so on.
    def _g_key(oriented):
        return benzene_prefix_citation_locants(
            oriented, substituents, principal_group_positions,
        )

    best_g = min(_g_key(o) for o in best_candidates)
    best_candidates = [o for o in best_candidates if _g_key(o) == best_g]

    if len(best_candidates) == 1:
        return best_candidates[0]

    # --- Last resort: canonical symmetry-class orbits, NEVER the order in which
    # the candidates were enumerated (which is the input SMILES' atom order).
    # Every survivor here is name-identical, so this only makes the CHOICE
    # reproducible; see _canonical_orbit_key.
    return min(best_candidates, key=lambda o: _canonical_orbit_key(mol, o))


def _rotate_list(lst: List, start: int) -> List:
    """Rotate list so element at index start becomes first."""
    return lst[start:] + lst[:start]


def _build_oriented_ring(
    ring_list: List[int],
    start_pos: int,
    direction: int
) -> List[int]:
    """
    Build an oriented version of the ring.

    Args:
        ring_list: Original list of ring atom indices
        start_pos: Index to start from
        direction: 1 for clockwise, -1 for counterclockwise

    Returns:
        Reordered list with start_pos first, going in specified direction
    """
    n = len(ring_list)
    oriented = []

    for i in range(n):
        idx = (start_pos + i * direction) % n
        oriented.append(ring_list[idx])

    return oriented


def _calculate_locants(
    oriented_ring: List[int],
    substituents: Dict[int, List[Dict]]
) -> List[int]:
    """
    Calculate locant set for a given orientation.

    Args:
        oriented_ring: Ring atoms in order (position 0 = locant 1)
        substituents: Dict from get_benzene_substituents

    Returns:
        Sorted list of locants (1-indexed positions)
    """
    locants = []
    for i, atom_idx in enumerate(oriented_ring):
        if atom_idx in substituents:
            locants.append(i + 1)  # Locants are 1-indexed

    return sorted(locants)


def name_substituted_benzene(
    mol,
    ring_atoms: Tuple[int, ...],
    oriented_ring: List[int],
    substituents: Dict[int, List[Dict]],
    detected_fgs: Optional[Dict] = None,
) -> str:
    """
    Generate systematic name for substituted benzene.

    IUPAC 2013 PIN Rules:
    - Use numeric locants (not ortho/meta/para) for polysubstituted
    - Monosubstituted benzenes do NOT include locant (it's always 1)
    - Alphabetize substituent prefixes
    - Use multiplicative prefixes (di-, tri-) for repeated substituents
    - Format: locants-substituent-benzene (or just substituent-benzene for mono)
    - Special case: benzonitrile (C6H5CN) uses suffix-style naming per
    - Ring-attached principal groups use suffix form

    Args:
        mol: RDKit Mol object
        ring_atoms: Original ring atom tuple
        oriented_ring: Oriented ring from orient_benzene
        substituents: Dict from get_benzene_substituents

    Returns:
        IUPAC name string (e.g., "chlorobenzene" or "1,4-dimethylbenzene"),
        or None when any substituent is an unnameable sentinel (Wave2 T3a
        conservation guard: emitting without it would drop its atoms).
    """
    # Wave2 T3a: decline (fail closed) when any branch could not be named —
    # see get_benzene_substituents. The caller treats None as handler decline.
    for _subs in substituents.values():
        for _s in _subs:
            if _s.get('unnameable'):
                return None

    # Build locant-to-substituent mapping
    # oriented_ring[0] = position 1, oriented_ring[1] = position 2, etc.
    atom_to_locant = {atom_idx: i + 1 for i, atom_idx in enumerate(oriented_ring)}

    # Collect stereodescriptors for ring atoms (benzene carbons are sp2 so
    # typically no R/S, but substituents attached at ring positions may carry
    # E/Z on bonds to ring atoms). We pass the ring atom_to_locant mapping.
    from ..perception.stereo import assign_stereochemistry
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    assign_stereochemistry(mol)
    stereo_descriptors = collect_stereodescriptors(mol, atom_to_locant)

    # Separate suffix-type FGs from prefix-type substituents
    suffix_groups: Dict[str, List[int]] = defaultdict(list)
    prefix_groups: Dict[str, List[int]] = defaultdict(list)
    # Track N-substituents for amides keyed by suffix_name
    n_substituents_map: Dict[str, List[str]] = {}
    # (plan P1AM Task 6): demoted-prefix stem override by suffix_name
    demoted_prefix_overrides: Dict[str, str] = {}
    # C4: track amine prefixes that CAN be promoted to the aniline suffix.
    # Maps locant -> {'prefix_name': <existing prefix>, 'n_substituents': [...]}.
    amine_candidates: Dict[int, Dict] = {}

    for atom_idx in oriented_ring:
        if atom_idx not in substituents:
            continue

        for sub_info in substituents[atom_idx]:
            locant = atom_to_locant[atom_idx]
            if sub_info.get('is_suffix'):
                suffix_groups[sub_info['suffix_name']].append(locant)
                # Track N-substituents if present
                if 'n_substituents' in sub_info and sub_info['n_substituents']:
                    n_substituents_map[sub_info['suffix_name']] = sub_info['n_substituents']
                # (plan P1AM Task 6): carry a demoted-prefix stem
                # override (hydrazinecarboximidoyl) if the detector set one.
                if sub_info.get('demoted_prefix_override'):
                    demoted_prefix_overrides[sub_info['suffix_name']] = \
                        sub_info['demoted_prefix_override']
            else:
                prefix_groups[sub_info['name']].append(locant)
                # C4: record a promotable amine (payload set by
                # _identify_nitrogen_group). The prefix name is kept in
                # prefix_groups so nothing changes unless it is promoted below.
                cand = sub_info.get('amine_candidate')
                if cand is not None:
                    amine_candidates[locant] = {
                        'prefix_name': sub_info['name'],
                        'n_substituents': cand.get('n_substituents', []),
                    }

    # Sort locants within each group
    for name in suffix_groups:
        suffix_groups[name].sort()
    for name in prefix_groups:
        prefix_groups[name].sort()

    # ---: Reclassify hydroxyl as suffix when it IS the principal group ---
    # _identify_oxygen_group always returns hydroxy as prefix. When hydroxyl is
    # the principal group (no higher-seniority suffix FG detected), move it to
    # suffix_groups so it routes to the phenol/ol naming path.
    # Guard: also check detected_fgs for FGs that use functional class naming
    # (isocyanate and the cyanate family). These FGs are not in _SUFFIX_PRIORITY, so
    # suffix_groups would be empty even though OH may not be the true principal
    # group. Blacklist approach: only block known functional-class-naming FGs.
    # ⚠ Phase C Task 9b: the comment here used to say "isocyanate, azide, etc."
    # -- azide was never in the effective set (dead key, see _FUNCTIONAL_CLASS_FGS),
    # and per (the Blue Book) it must not be. The remaining members are a KNOWN
    # RESIDUAL DEFECT recorded with citations at _FUNCTIONAL_CLASS_FGS.
    # Phase C Task 9b: the promotion decision itself now comes from the SHARED
    # authority ``benzene_prefix_suffix_promotion``, which the (c) numbering
    # anchor also reads. Before this, the anchor computed its own answer from the
    # seniority tables and agreed with this block only for hydroxy -- see C-2 in
    # `internal notes`.
    _promoted_suffix, _promoted_prefixes = benzene_prefix_suffix_promotion(
        set(suffix_groups), set(prefix_groups), detected_fgs,
        has_amine_candidates=bool(amine_candidates),
    )
    # ⚠ The guard is "no suffix SENIOR to -ol", not "no suffix at all".
    # Phase C tranche C REGRESSION, caught by an A/B against HEAD before shipping:
    # adding the `-thiol` suffix form (for `benzenethiol`, the Blue Book) made this
    # `not suffix_groups` test false whenever an SH was present, so the promotion was
    # skipped and the JUNIOR thiol won the suffix -- `Sc1ccccc1O` went from the correct
    # `2-sulfanylphenol` to `2-hydroxybenzenethiol`. Any future junior suffix form would
    # have reintroduced the same inversion, so the guard is now seniority-aware rather
    # than existence-aware.
    # SENIORITY_ORDER ranks: primary/secondary alcohol 88/89, phenol 91, thiol 94,
    # amines 102+. So -ol outranks -thiol and must claim the suffix when both are
    # present, leaving the SH as a `sulfanyl` prefix.
    # _OL_JUNIOR_SUFFIXES is module level (Phase C Task 9); it is read by
    # ``benzene_prefix_suffix_promotion``, which is where the seniority test that
    # used to be inlined here now lives.
    if _promoted_suffix == 'ol':
        # No suffix FG senior to -ol, and no competing functional-class FG -- hydroxyl
        # IS the principal group. Move from prefix to suffix, and demote any junior
        # suffix that was holding the slot back to its own prefix form. The prefix
        # NAME comes from the shared authority rather than a literal here, so the
        # anchor and the promotion cannot disagree about which prefix moves.
        _promoted_locants: List[int] = []
        for _pname in _promoted_prefixes:
            _promoted_locants.extend(prefix_groups.pop(_pname, []))
        suffix_groups['ol'] = sorted(_promoted_locants)
        for _junior in list(suffix_groups):
            if _junior in _OL_JUNIOR_SUFFIXES:
                _jlocs = suffix_groups.pop(_junior)
                _jprefix = _SUFFIX_TO_PREFIX.get(_junior, _junior)
                prefix_groups[_jprefix].extend(_jlocs)
                prefix_groups[_jprefix].sort()
    # When a suffix SENIOR to -ol is present (acid, aldehyde, etc.), hydroxyl stays
    # as prefix "hydroxy" -- correct per IUPAC seniority rules.

    # --- C4 /: reclassify an amine as the '-amine'
    # (aniline) SUFFIX when it IS the molecule-level principal group. ---
    # ``benzene_prefix_suffix_promotion`` returns 'amine' only when NOTHING else
    # holds the suffix -- including the '-ol' it would itself have promoted -- so
    # any senior group (acid/aldehyde/nitrile/ol/etc.) keeps the amine as its
    # 'amino'/(N-alkylamino)/anilino PREFIX -> 4-aminophenol, 4-aminobenzoic acid,
    # 4-(N-methylamino)benzoic acid stay unchanged. The two promotions are mutually
    # exclusive by construction now, not by statement order.
    if _promoted_suffix == 'amine':
        amine_locants = sorted(amine_candidates.keys())
        # Remove each promoted amine from its prefix group (by locant), so the
        # amine no longer appears as a substituent prefix.
        for loc in amine_locants:
            pname = amine_candidates[loc]['prefix_name']
            if pname in prefix_groups and loc in prefix_groups[pname]:
                prefix_groups[pname].remove(loc)
                if not prefix_groups[pname]:
                    del prefix_groups[pname]
        suffix_groups['amine'] = amine_locants
        if len(amine_locants) == 1:
            # Single amine (aniline) path: N-substituents keyed under 'amine'.
            n_subs = amine_candidates[amine_locants[0]]['n_substituents']
            if n_subs:
                n_substituents_map['amine'] = n_subs
        else:
            # C4b: multi-amine (benzene-x,y-diamine) path. Carry the FULL
            # per-locant N-substituent map so the assembler can build the
            # 'N-...benzene-x,y-diamine' PIN instead of dropping the
            # N-substituent (which produced a WRONG structure ->
            # suppressed the whole molecule to 'unknown').
            by_locant = {
                loc: amine_candidates[loc]['n_substituents']
                for loc in amine_locants
                if amine_candidates[loc]['n_substituents']
            }
            if by_locant:
                n_substituents_map['__amine_by_locant__'] = by_locant

    # If suffix groups exist, use suffix naming path
    if suffix_groups:
        name = _assemble_benzene_with_suffix(
            mol, suffix_groups, prefix_groups, n_substituents_map,
            atom_to_locant, oriented_ring, demoted_prefix_overrides,
            # Phase C tranche B: the licence is evaluated PER SCOPE, and a
            # stereodescriptor in this scope is an essential locant that restores every
            # other one, the Blue Book). The suffix assembler could not see them.
            stereo_descriptors=stereo_descriptors,
        )
        if name and stereo_descriptors:
            stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
            name = f"{stereo_prefix}{name}"
        # F-B / gate-independent re-anchor: whenever an N-substituted
        # sulfonamide contributed to the name -- as the suffix (N,4-dimethyl...
        # sulfonamide) or demoted to a '{N-subs}sulfamoyl' prefix -- the italic-N
        # substituent names come from name_substituent_fragment, which can DROP a
        # benzylic stereodescriptor or MIS-NAME a functional-group branch and yield
        # a name for a DIFFERENT molecule that still OPSIN-parses. Require the whole
        # name to round-trip InChIKey-exact; fail closed otherwise.
        if name and n_substituents_map.get('sulfonamide'):
            name = _reanchor_name_to_mol(mol, name)
        return name

    # === PREFIX-ONLY PATH (existing logic) ===

    # Check for nitrile - special handling for benzonitrile naming (fix)
    # IUPAC 2013 PIN: benzonitrile (not cyanobenzene) per
    if 'nitrile' in prefix_groups:
        nitrile_locants = prefix_groups['nitrile']
        if len(nitrile_locants) == 1:
            # Single nitrile: use benzonitrile as parent
            # Remove nitrile from substituent groups since it becomes the parent
            del prefix_groups['nitrile']

            # Re-orient so nitrile is at position 1 for locant calculation
            nitrile_locant = nitrile_locants[0]

            # Other substituents become prefixes relative to benzonitrile
            if not prefix_groups:
                # Pure benzonitrile
                name = "benzonitrile"
                if stereo_descriptors:
                    stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
                    name = f"{stereo_prefix}{name}"
                return name

            # Build prefixes for other substituents
            # Need to recalculate locants relative to nitrile at position 1
            name = _name_substituted_benzonitrile(
                prefix_groups, nitrile_locant, atom_to_locant, oriented_ring
            )
            if stereo_descriptors:
                stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
                name = f"{stereo_prefix}{name}"
            return name

    # Count total number of substituents
    total_substituents = sum(len(locs) for locs in prefix_groups.values())

    # For monosubstituted benzenes, omit the locant (it's always 1)
    is_monosubstituted = total_substituents == 1
    _omit = should_omit_locant_one(
        context="prefix",
        is_ring=True,
        is_heterocyclic=False,  # Benzene is always carbocyclic
        is_monosubstituted=is_monosubstituted,
    )

    # (the Blue Book): a ring whose every substitutable position carries the SAME
    # prefix omits all locants -- 'hexamethylbenzene', 'hexafluorobenzene'. Distinct
    # from the monosubstituted licence above (that one has no locant to cite), so it
    # needs its own branch: the MULTIPLIER must survive, and the `_omit` branch below
    # emits the bare name without one.
    _l5_omit = (not _omit) and _benzene_l5_uniform_licence(
        mol, oriented_ring, {}, prefix_groups, stereo_descriptors
    )

    # Build prefix strings in citation order. Uses the shared TOTAL order
    # rather than the raw `alpha_sort_key` string: (:3442) compares
    # nonitalic Roman letters FIRST and only then locants, and a single key
    # string conflates the two because it still carries digits. That is what
    # cited `4-[(1R)-1-chloroethyl]phenoxy` (key `1-chloroethylphenoxy`, leading
    # '1' = ASCII 49) ahead of `chloroethyl`. The citation order here must agree
    # with `benzene_prefix_citation_locants`, which assigns the locants under
    # (g) -- if the two disagree the name's order and its numbering come
    # from different rules.
    prefixes = []
    for name in sorted(prefix_groups.keys(), key=prefix_citation_sort_key):
        locants = prefix_groups[name]
        count = len(locants)

        if _l5_omit:
            # An empty locant list is format_substituent_prefix's own documented
            # elision path: it drops the locant string and its hyphen while
            # keeping the multiplier, so 'methyl' x6 -> 'hexamethyl'.
            prefix_str = format_substituent_prefix(name, [], count)
        elif _omit:
            # Monosubstituted: no locant. A compound or complex prefix takes
            # enclosing marks, the Blue Book) by the same rule as
            # the polysubstituted branch below (format_substituent_prefix):
            # enclose_if_compound, the union of needs_brackets,
            # is_complex_substituent and an inner enclosing mark, escalated by the
            # marks already inside it,:7444; '(2-methylbut-2-en-1-yl)
            # benzene', '[2-(nitromethyl)propyl]benzene', '[(methoxymethoxy)methyl]
            # benzene'); a fully enclosed name ('(2-methylpropyl)') is unchanged.
            # is_complex_substituent alone cited a chalcogen prefix on a ring
            # substituent bare ('cyclohexylsulfanylbenzene'), against
            # '(cyclopentylselanyl)benzene (PIN)',:27834).
            from ..assembly.naming_utils import enclose_if_compound
            prefix_str = enclose_if_compound(name)
        else:
            # Polysubstituted: include locants
            prefix_str = format_substituent_prefix(name, locants, count)

        prefixes.append(prefix_str)

    # Join prefixes with proper hyphenation
    prefix_part = _join_benzene_prefixes(prefixes)

    # Build final name
    name = f"{prefix_part}benzene"

    # Prepend stereo prefix if descriptors exist
    if stereo_descriptors:
        stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
        name = f"{stereo_prefix}{name}"

    return name


def _assemble_benzene_with_suffix(
    mol,
    suffix_groups: Dict[str, List[int]],
    prefix_groups: Dict[str, List[int]],
    n_substituents_map: Dict[str, List[str]],
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
    demoted_prefix_overrides: Optional[Dict[str, str]] = None,
    stereo_descriptors: Optional[List] = None,
) -> str:
    """
    Assemble benzene name with suffix functional groups.

    Picks the highest-priority suffix FG, builds locants + multiplier + suffix,
    then adds remaining FGs as prefixes.

    Args:
        mol: RDKit Mol object
        suffix_groups: Dict of suffix_name -> list of locants
        prefix_groups: Dict of prefix_name -> list of locants
        n_substituents_map: Dict of suffix_name -> list of N-alkyl names
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring
        stereo_descriptors: The scope's stereodescriptors, needed because a
            stereodescriptor is an essential locant and (the Blue Book) then
            restores every locant in the scope. ``None`` means "not established" and
            fails closed (the licence is refused).

    Returns:
        IUPAC name string with suffix FG
    """
    # Pick highest-priority suffix
    chosen_suffix = None
    for sfx in _SUFFIX_PRIORITY:
        if sfx in suffix_groups:
            chosen_suffix = sfx
            break

    if chosen_suffix is None:
        # Shouldn't happen, but fallback to first suffix
        chosen_suffix = next(iter(suffix_groups))
    # Suite fix j6: a carbothioic S-acid (SENIORITY_ORDER rank 2) outranks an
    # imidic acid (rank 11) but is not in _SUFFIX_PRIORITY; keep it the suffix
    # (the imidic acid demotes to 'C-hydroxycarbonimidoyl'), as before the
    # imidic acid had a suffix form here.
    if (chosen_suffix == 'carboximidic acid'
            and 'carbothioic S-acid' in suffix_groups):
        chosen_suffix = 'carbothioic S-acid'

    chosen_locants = suffix_groups[chosen_suffix]
    chosen_count = len(chosen_locants)

    # Single carbonitrile: delegate to existing benzonitrile path
    # (which uses the retained name "benzonitrile" and proper renumbering)
    if chosen_suffix == 'carbonitrile' and chosen_count == 1:
        # Convert suffix+prefix groups back to prefix-only for benzonitrile path
        nitrile_locant = chosen_locants[0]
        all_prefix = dict(prefix_groups)
        # Add any other suffix FGs as prefixes
        for sfx_name, sfx_locants in suffix_groups.items():
            if sfx_name == 'carbonitrile':
                continue
            prefix_form = _SUFFIX_TO_PREFIX.get(sfx_name, sfx_name)
            if prefix_form:
                all_prefix[prefix_form] = sfx_locants
        if not all_prefix:
            return "benzonitrile"
        return _name_substituted_benzonitrile(
            all_prefix, nitrile_locant, atom_to_locant, oriented_ring
        )

    # Remaining suffix FGs become prefixes
    remaining_prefix_groups = dict(prefix_groups)
    for sfx_name, sfx_locants in suffix_groups.items():
        if sfx_name == chosen_suffix:
            continue
        # Convert to prefix form
        prefix_form = _SUFFIX_TO_PREFIX.get(sfx_name, sfx_name)
        # D1: a demoted N-/N'-substituted amidine must carry its
        # italic-N locants onto the carbamimidoyl prefix, e.g.
        # 4-(N-methylcarbamimidoyl)benzoic acid. n_substituents_map is keyed by
        # the SUFFIX name ('carboximidamide') and holds (nlocant, name) tuples;
        # prepend the N/N' prefix to the base 'carbamimidoyl'. The resulting
        # hyphenated name is 'complex', so format_substituent_prefix wraps it in
        # parentheses. Bare carbamimidoyl (no N-subs) keeps the plain prefix.
        if sfx_name == 'carboximidamide':
            # (BB 34490/34496, plan P1AM Task 6): a hydrazino-
            # bearing ring amidine demotes to 'hydrazinecarboximidoyl', not the
            # N-dropping 'carbamimidoyl'. The override carries no N/N' subs
            # (decorated hydrazino forms are fail-closed at detection).
            _override = (demoted_prefix_overrides or {}).get('carboximidamide')
            if _override:
                prefix_form = _override
            else:
                n_subs = n_substituents_map.get('carboximidamide', [])
                n_prefix = _build_amidine_n_prefix(n_subs)
                if n_prefix:
                    prefix_form = f"{n_prefix}{prefix_form}"
        #: a demoted N-substituted sulfonamide is the
        # '{N-substituents}sulfamoyl' substituent prefix, the N-substituents cited
        # WITHOUT the italic N -- BB '2-(dimethylsulfamoyl)benzene-1-sulfonic acid'
        # (PIN,:32982), 'phenylsulfamoyl' (:32985). The bare 'sulfamoyl' would DROP
        # them (a wrong constitution). Build it; fail closed on a shape the builder
        # does not cover (distinct N-substituents -> the nested 'ethyl(methyl)-'
        # form). The caller re-anchors the whole name gate-independently, so a
        # fragment-namer mis-spelling of an N-substituent fails closed, never ships.
        if sfx_name == 'sulfonamide' and n_substituents_map.get('sulfonamide'):
            prefix_form = _build_n_substituted_sulfamoyl_prefix(
                n_substituents_map['sulfonamide'])
            if not prefix_form:
                return None
        if prefix_form:
            remaining_prefix_groups[prefix_form] = sfx_locants

    #: Handle hydroxyl suffix -- phenol retained name for benzene
    if chosen_suffix == 'ol':
        ol_locants = chosen_locants
        if len(ol_locants) == 1:
            # Single OH on benzene: use "phenol" retained name
            return _name_substituted_phenol(
                remaining_prefix_groups, ol_locants[0],
                atom_to_locant, oriented_ring
            )
        else:
            # Multiple OH on benzene: benzene-1,2-diol, benzene-1,2,3,4,5,6-hexol.
            # Use systematic naming with multiplied -ol suffix; elides
            # the multiplier-final 'a' before '-ol' (hexa+ol → hexol).
            from ..assembly.naming_utils import (
                SIMPLE_MULTIPLIERS,
                _join_multiplied_suffix,
            )
            mult = SIMPLE_MULTIPLIERS.get(len(ol_locants), str(len(ol_locants)))
            loc_str = ','.join(str(l) for l in sorted(ol_locants))
            multiplied = _join_multiplied_suffix(mult, 'ol')
            # (the Blue Book): when every substitutable ring position bears the
            # SAME decoration the locants are omitted -- 'benzenehexol' (PIN) the Blue Book.
            # Deny by default, the Blue Book): any prefix, any stereodescriptor,
            # any partial substitution and the locants all come back.
            if _benzene_l5_uniform_licence(
                mol, oriented_ring, {'ol': ol_locants}, remaining_prefix_groups,
                stereo_descriptors,
            ):
                return f"benzene{multiplied}"
            suffix_part = f"benzene-{loc_str}-{multiplied}"
            if remaining_prefix_groups:
                prefix_part = _build_prefix_string_with_locants(
                    remaining_prefix_groups, mono_needs_locant=True
                )
                return f"{prefix_part}{suffix_part}"
            return suffix_part

    # C4: amine suffix -- 'aniline' retained name for benzene.
    # Mirrors the hydroxy->phenol path above (amine sits last in seniority).
    if chosen_suffix == 'amine':
        amine_locants = chosen_locants
        if len(amine_locants) == 1:
            # Single amine on benzene: 'aniline' retained parent + N-substituents.
            n_subs = n_substituents_map.get('amine', [])
            return _name_substituted_aniline(
                remaining_prefix_groups, amine_locants[0],
                atom_to_locant, oriented_ring, n_subs
            )
        else:
            # Multiple amines -> 'benzene-<locants>-<mult>amine' parent.
            # C4b: when one/more of the amine nitrogens carry N-substituents
            # (e.g. Nc1ccc(NC)cc1 -> N-methylbenzene-1,4-diamine), those must
            # be cited as italic-N locant prefixes on the diamine parent.
            # Previously they were DROPPED, yielding a wrong structure that
            # then suppressed to 'unknown'.
            n_by_locant = n_substituents_map.get('__amine_by_locant__', {})
            return _name_substituted_benzenediamine(
                remaining_prefix_groups, amine_locants, n_by_locant,
            )

    # Check for special "benzoic acid" retained base name:
    # Single carboxylic acid -> "benzoic acid" base
    if chosen_suffix == 'carboxylic acid' and chosen_count == 1:
        return _name_substituted_benzoic_acid(
            remaining_prefix_groups, chosen_locants[0],
            atom_to_locant, oriented_ring
        )

    # W3-P04 /: single sulfonic/sulfinic acid on benzene.
    # Bare -> 'benzenesulfonic acid' (no locant,). With ANY other
    # ring substituent the systematic ring suffix MUST cite its locant per the
    # Blue Book PINs: '4-aminobenzene-1-sulfonic acid' ,
    # '2-(dimethylsulfamoyl)benzene-1-sulfonic acid' ,
    # '4-benzamidobenzene-1-sulfonic acid' . The general path below drops
    # this '-1-' (emitting the valid-but-non-PIN 'sulfanilic'-style form), so a
    # substituted case is intercepted here and re-anchored to the acid (position
    # 1). Sulfinic follows the identical pattern.
    if chosen_suffix in ('sulfonic acid', 'sulfinic acid') and chosen_count == 1:
        if not remaining_prefix_groups:
            return f"benzene{chosen_suffix}"
        renumbered_groups = _renumber_relative_to(
            remaining_prefix_groups, chosen_locants[0]
        )
        prefix_part = _build_prefix_string_with_locants(
            renumbered_groups, mono_needs_locant=True
        )
        return f"{prefix_part}benzene-1-{chosen_suffix}"

    # Check for benzamide-based naming:
    # Single carboxamide -> use "benzamide" as retained base
    if chosen_suffix == 'carboxamide' and chosen_count == 1:
        n_subs = n_substituents_map.get('carboxamide', [])
        return _name_substituted_benzamide(
            remaining_prefix_groups, chosen_locants[0],
            atom_to_locant, oriented_ring, n_subs
        )

    # Check for benzenesulfonamide-based naming:
    # Single sulfonamide -> "benzenesulfonamide". The whole name is re-anchored by
    # the caller (name_substituted_benzene) whenever N-substituents are present.
    if chosen_suffix == 'sulfonamide' and chosen_count == 1:
        return _name_substituted_benzenesulfonamide(
            remaining_prefix_groups, chosen_locants[0],
            atom_to_locant, oriented_ring,
            n_substituents=n_substituents_map.get('sulfonamide', []),
        )

    # F-B 0-wrong: a MULTI-instance sulfonamide (di/poly) carrying N-substituents
    # needs the superscripted N^1/N^3 locants of, which are not
    # built here. The general multi-suffix assembler below cannot cite them and
    # would silently DROP the N-substituents (benzene-1,4-disulfonamide for an
    # N,N'-dimethyl input -- a wrong constitution). Fail closed. A PRIMARY
    # disulfonamide (no N-substituents) is unaffected and still names below.
    if chosen_suffix == 'sulfonamide' and n_substituents_map.get('sulfonamide'):
        return None

    # Task Y): single sulfinamide on benzene ->
    # 'benzenesulfinamide'. Mirrors the sulfonimidamide rule immediately below:
    # only the unsubstituted parent is built, because a ring substituent would
    # need the merged N/numeric prefix list and N-substitution is not built for
    # this class at all. ANY additional ring substituent -> fail closed.
    if chosen_suffix == 'sulfinamide' and chosen_count == 1:
        if remaining_prefix_groups:
            return None
        return "benzenesulfinamide"

    # Wave-2 P1AM, BB 34173): single sulfonimidamide on benzene ->
    # 'benzenesulfonimidamide'. Only the unsubstituted parent is built here;
    # ANY additional ring substituent (or N/N' substitution, which carries
    # N/N' locant semantics not yet built) -> fail closed (return None).
    if chosen_suffix == 'sulfonimidamide' and chosen_count == 1:
        if remaining_prefix_groups:
            return None
        return "benzenesulfonimidamide"

    # Wave2 ring-hydrazide: a single carbohydrazide on benzene
    # keeps the RETAINED acyl stem — 'benzohydrazide' (PIN, substitutable the
    # same way as benzamide) — not the systematic 'benzenecarbohydrazide':
    # 'benzohydrazide', '4-methylbenzohydrazide'. (Multi-instance falls through
    # to the general 'benzene-1,4-dicarbohydrazide' path below,.)
    if chosen_suffix == 'carbohydrazide' and chosen_count == 1:
        if not remaining_prefix_groups:
            return "benzohydrazide"
        renumbered_groups = _renumber_relative_to(
            remaining_prefix_groups, chosen_locants[0]
        )
        prefix_part = _build_prefix_string_with_locants(
            renumbered_groups, mono_needs_locant=True
        )
        return f"{prefix_part}benzohydrazide"

    # C1: single sulfonohydrazide on benzene. Renumber relative to
    # the group (position 1) and cite the locant of any other substituent,
    # mirroring the benzenesulfonamide path: 'benzenesulfonohydrazide',
    # '4-methylbenzenesulfonohydrazide'.
    if chosen_suffix in ('sulfonohydrazide',
                         'sulfinohydrazonohydrazide') and chosen_count == 1:
        if not remaining_prefix_groups:
            return f"benzene{chosen_suffix}"
        renumbered_groups = _renumber_relative_to(
            remaining_prefix_groups, chosen_locants[0]
        )
        prefix_part = _build_prefix_string_with_locants(
            renumbered_groups, mono_needs_locant=True
        )
        return f"{prefix_part}benzene{chosen_suffix}"

    # Suite fix j6 (TRIAGE g8 C23): single imidic acid on benzene ->
    # '-carboximidic acid'. (the Blue Book) 'benzene-
    # carboximidic acid (PIN)' (bare, no locant); with other substituents the
    # PIN cites '-1-' on the group: '2-(propanimidoylselanyl)benzene-1-
    # carboximidic acid (PIN)' (:32013). Several imidic acids fall through to
    # the general assembler ('benzene-1,4-dicarboximidic acid').
    if chosen_suffix == 'carboximidic acid' and chosen_count == 1:
        if not remaining_prefix_groups:
            return "benzenecarboximidic acid"
        renumbered_groups = _renumber_relative_to(
            remaining_prefix_groups, chosen_locants[0]
        )
        prefix_part = _build_prefix_string_with_locants(
            renumbered_groups, mono_needs_locant=True
        )
        return f"{prefix_part}benzene-1-carboximidic acid"

    # C2: single amidine on benzene -> '-carboximidamide' suffix.
    # Bare -> 'benzenecarboximidamide' (no locant). With other substituents the
    # PIN carries an explicit '-1-' on the group (unlike the carbohydrazide path
    # above): '4-methylbenzene-1-carboximidamide' (OPSIN-verified). Multi-instance
    # (-> 'benzene-1,4-dicarboximidamide') falls through to the general assembler.
    if chosen_suffix == 'carboximidamide' and chosen_count == 1:
        n_subs = n_substituents_map.get('carboximidamide', [])
        n_prefix = _build_amidine_n_prefix(n_subs)
        if not remaining_prefix_groups:
            return f"{n_prefix}benzenecarboximidamide"
        renumbered_groups = _renumber_relative_to(
            remaining_prefix_groups, chosen_locants[0]
        )
        prefix_part = _build_prefix_string_with_locants(
            renumbered_groups, mono_needs_locant=True
        )
        return f"{prefix_part}{n_prefix}benzene-1-carboximidamide"

    # Single carbaldehyde: delegate to benzaldehyde retained name path
    # "benzaldehyde" is an IUPAC retained name
    if chosen_suffix == 'carbaldehyde' and chosen_count == 1:
        return _name_substituted_benzaldehyde(
            remaining_prefix_groups, chosen_locants[0],
            atom_to_locant, oriented_ring
        )

    # C- (V-2): single chalcogen aldehyde on benzene. No retained base
    # exists (unlike benzaldehyde), so the systematic 'benzenecarbo*aldehyde'
    # base is used, with the suffix renumbered to locant 1 when other
    # substituents are present (e.g. 4-methylbenzene-1-carboselenaldehyde).
    if chosen_suffix in (
        'carbothialdehyde', 'carboselenaldehyde', 'carbotelluraldehyde'
    ) and chosen_count == 1:
        return _name_substituted_chalcogen_carbaldehyde(
            remaining_prefix_groups, chosen_locants[0], chosen_suffix,
            atom_to_locant, oriented_ring
        )

    # General suffix assembly for multi-suffix or non-retained cases
    # Build suffix part: benzene-{locants}-{multiplier}{suffix}
    multiplier = get_suffix_multiplier_prefix(chosen_count, chosen_suffix) if chosen_count > 1 else ""
    locant_str = ",".join(str(loc) for loc in chosen_locants)

    # Build prefix part from remaining groups.
    # W3-P04 /: the general multi-suffix path must PRESERVE
    # prefix locants — remaining_prefix_groups is already keyed by the final ring
    # locants (same numbering as chosen_locants), so a bare _build_prefix_string
    # would drop e.g. the '4-' on sulfo ('sulfobenzene-1,2-dicarboxylic acid' ->
    # OPSIN reparses to the wrong isomer ->). Use the locant-preserving
    # builder (mono_needs_locant=True) exactly like the retained-base branches.
    prefix_part = _build_prefix_string_with_locants(
        remaining_prefix_groups, mono_needs_locant=True
    )

    # Assemble: {prefix}benzene-{locants}-{multiplier}{suffix}
    if chosen_count > 1:
        return f"{prefix_part}benzene-{locant_str}-{multiplier}{chosen_suffix}"
    elif prefix_part:
        # A single systematic suffix beside a prefix cites its locant:
        # (the Blue Book) omits the '1' only where no locant is needed, and
        # the PINs print it -- 'sodium 4-methylbenzene-1-thiolate (PIN)' (:43599),
        # '3-[(4-sulfanylphenyl)disulfanyl]benzene-1-thiol (PIN)' (:27593), as the
        # sulfonic-acid branch above does ('4-aminobenzene-1-sulfonic acid').
        # '4-methylbenzenethiol' was shipped for the first.
        return f"{prefix_part}benzene-{locant_str}-{chosen_suffix}"
    else:
        # Monosubstituted: benzene{suffix} (no locant, no hyphen)
        return f"{prefix_part}benzene{chosen_suffix}"


def _name_substituted_benzoic_acid(
    prefix_groups: Dict[str, List[int]],
    acid_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
) -> str:
    """
    Name substituted benzoic acid derivatives.

    Uses "benzoic acid" as the retained base name. Position 1 is the
    carboxylic acid position. Other substituents get locants relative to it.

    Args:
        prefix_groups: Dict of prefix name -> locants (non-acid substituents)
        acid_locant: Locant of the carboxylic acid in original numbering
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring

    Returns:
        IUPAC name like "2-hydroxybenzoic acid"
    """
    if not prefix_groups:
        return "benzoic acid"

    # Renumber relative to acid position (acid = position 1)
    renumbered_groups = _renumber_relative_to(prefix_groups, acid_locant)

    # Build prefixes
    prefix_part = _build_prefix_string_with_locants(renumbered_groups, mono_needs_locant=True)

    return f"{prefix_part}benzoic acid"


def _name_substituted_phenol(
    prefix_groups: Dict[str, List[int]],
    ol_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
) -> str:
    """
    Name substituted phenol derivatives.

    Uses "phenol" as the retained base name per IUPAC.
    Position 1 is the hydroxyl position. Other substituents get locants
    relative to it.

    Args:
        prefix_groups: Dict of prefix name -> locants (non-OH substituents)
        ol_locant: Locant of the hydroxyl in original numbering
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring

    Returns:
        IUPAC name like "2-methylphenol" or "4-chlorophenol"
    """
    if not prefix_groups:
        return "phenol"

    # Renumber relative to OH position (OH = position 1)
    renumbered_groups = _renumber_relative_to(prefix_groups, ol_locant)

    # Build prefixes
    prefix_part = _build_prefix_string_with_locants(renumbered_groups, mono_needs_locant=True)

    return f"{prefix_part}phenol"


def _name_substituted_aniline(
    prefix_groups: Dict[str, List[int]],
    amine_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
    n_substituents: List[str],
) -> str:
    """Name substituted aniline derivatives (C4,.

    Uses 'aniline' as the retained base name. Position 1 is the amine-bearing
    carbon. Ring substituents get locants relative to it; N-substituents are
    cited with italic-N locants and are alphabetized TOGETHER with the ring
    substituents detachable-prefix ordering — the N-locant is not a
    numeral, so ordering is by the substituent base name).

    Models _name_substituted_phenol (ring prefixes + renumbering) plus the
    N-substituent prefix builder of _name_substituted_benzamide.

    Args:
        prefix_groups: Non-amine ring substituent groups.
        amine_locant: Locant of the amine in original ring numbering.
        atom_to_locant: Mapping from atom index to locant.
        oriented_ring: The oriented ring.
        n_substituents: List of N-substituent names (e.g. ['methyl'],
            ['phenyl'], ['ethyl', 'ethyl']).

    Returns:
        IUPAC name like 'aniline', 'N-methylaniline', '4-methylaniline',
        'N,N-dimethylaniline', '4-fluoro-N-methylaniline'.
    """
    from ..assembly.naming_utils import _wrap_n_substituent, enclose_if_compound

    # Build the individual N-substituent prefix strings (with N/N,N locants).
    #: a COMPOUND N-substituent (e.g. '3-methylbutyl') takes its own
    # enclosing marks BEFORE the 'N-' locant is prefixed -- enclose_if_compound
    # adds the base '(...)' (and escalates -> when the name already carries a
    # mark); _wrap_n_substituent then deliberately only escalates further. Without
    # the enclose step the prefix read 'N-3-methylbutylaniline' instead of
    # 'N-(3-methylbutyl)aniline' (matches the polyfunctional N-substituent path).
    n_prefix_entries: List[Tuple[str, str]] = []  # (alpha_key, rendered)
    if n_substituents:
        if len(n_substituents) == 2 and n_substituents[0] == n_substituents[1]:
            # The shared primitive joins the multiplier ('N,N-di(dodecyl)',
            # (c), the Blue Book).
            from ..assembly.naming_utils import multiplied_component
            _nm = n_substituents[0]
            rendered = (f"N,N-"
                        f"{multiplied_component(2, _nm, _wrap_n_substituent(enclose_if_compound(_nm)))}")
            n_prefix_entries.append((alpha_sort_key(n_substituents[0]), rendered))
        else:
            for name in n_substituents:
                rendered = f"N-{_wrap_n_substituent(enclose_if_compound(name))}"
                n_prefix_entries.append((alpha_sort_key(name), rendered))

    # Renumber ring substituents relative to the amine position (= position 1).
    renumbered_groups = _renumber_relative_to(prefix_groups, amine_locant)

    # Build the individual ring-substituent prefix strings.
    ring_prefix_entries: List[Tuple[str, str]] = []  # (alpha_key, rendered)
    for name in renumbered_groups:
        locants = sorted(renumbered_groups[name])
        rendered = format_substituent_prefix(name, locants, len(locants))
        ring_prefix_entries.append((alpha_sort_key(name), rendered))

    if not n_prefix_entries and not ring_prefix_entries:
        return "aniline"

    # Alphabetize N- and ring-substituents together.
    all_entries = n_prefix_entries + ring_prefix_entries
    # `e[1]` is an already-RENDERED prefix string, so its leading locants are
    # PARENT locants -> parent_locants=True compares a prefix's OWN
    # locants, and a parent locant is assigned BY this very order). `[1:]` keeps
    # the total-order tier so an exact tie cannot fall through to list order.
    all_entries.sort(key=lambda e: (e[0], prefix_citation_sort_key(
        e[1], parent_locants=True)[1:]))
    prefix_part = "-".join(rendered for _key, rendered in all_entries)

    return f"{prefix_part}aniline"


def _name_substituted_benzenediamine(
    prefix_groups: Dict[str, List[int]],
    amine_locants: List[int],
    n_by_locant: Dict[int, List[str]],
) -> str:
    """Name a benzene poly-amine (>=2 -NH2/-NHR/-NR2) as the
    'benzene-<locants>-<mult>amine' PIN, citing N-substituents on any
    N-substituted nitrogen as italic-N prefixes (C4b, /.

    C4 promoted every amine to the diamine suffix but silently dropped the
    N-substituent, producing e.g. 'benzene-1,4-diamine' for
    N-methylbenzene-1,4-diamine — a WRONG structure that then
    suppressed to 'unknown'. This restores the correct PIN for the class.

    Numbering (all inputs are in the current oriented-ring locant space,
    1..6, cyclic): the ring is renumbered over all 12 symmetry operations
    (6 rotations x 2 directions). Selection criteria, in order:
      1. lowest locant set for the amino (parent-suffix) positions,
      2. lowest locants for the N-substituted nitrogens (so a single
         N-substituted amine lands on position 1 -> cited as plain 'N-'),
      3. lowest locants for the detachable ring-substituent prefixes,
      4. numeric tie-break.

    N-substituent citation: the lowest-locant substituted nitrogen is cited
    with a bare 'N' (or 'N,N-' for a disubstituted single nitrogen); any
    additional substituted nitrogen is cited with its numeric ring locant
    ('N<locant>-', OPSIN-valid,. Ring and N prefixes are
    alphabetized together.

    Args:
        prefix_groups: Non-amine ring substituent groups (current locants).
        amine_locants: Current locants of the amino-bearing ring carbons.
        n_by_locant: current-locant -> list of N-substituent names, ONLY for
            nitrogens that carry substituents (bare -NH2 omitted).

    Returns:
        IUPAC name, e.g. 'N-methylbenzene-1,4-diamine',
        'N,N-dimethylbenzene-1,4-diamine', 'N-phenylbenzene-1,2-diamine'.
    """
    from ..assembly.naming_utils import (
        SIMPLE_MULTIPLIERS,
        _join_multiplied_suffix,
        _wrap_n_substituent,
    )

    amine_set = set(amine_locants)

    def renumber(start_pos: int, direction: int, old: int) -> int:
        # old current locant (1..6) -> new locant under this symmetry op.
        return ((direction * ((old - 1) - start_pos)) % 6) + 1

    best = None  # (amine_key, n_key, ring_key, tiebreak, payload)
    for start_pos in range(6):
        for direction in (1, -1):
            # New amine locants.
            new_amines = sorted(renumber(start_pos, direction, a)
                                for a in amine_locants)
            # New locants of the N-substituted nitrogens.
            new_n_locs = sorted(renumber(start_pos, direction, loc)
                                for loc in n_by_locant)
            # New ring-substituent locants.
            new_ring: Dict[str, List[int]] = defaultdict(list)
            for name, locs in prefix_groups.items():
                for loc in locs:
                    new_ring[name].append(renumber(start_pos, direction, loc))
            ring_key = tuple(sorted(
                loc for locs in new_ring.values() for loc in locs
            ))
            # Map new-N-locant -> substituent names (for citation).
            new_n_map = {
                renumber(start_pos, direction, loc): names
                for loc, names in n_by_locant.items()
            }
            tiebreak = (tuple(new_amines), tuple(new_n_locs), ring_key)
            key = (tuple(new_amines), tuple(new_n_locs), ring_key, tiebreak)
            if best is None or key < best[0]:
                best = (key, new_amines, new_ring, new_n_map)

    _key, new_amines, new_ring, new_n_map = best

    # --- N-substituent prefix entries (alpha_key, rendered) ---
    # (the Blue Book): "Superscript arabic numbers, which are the
    # locants of the parent structure, are used to differentiate the nitrogen
    # atoms of di- and polyamines... except for geminal amines" -- so EVERY
    # substituted N of a benzene poly-amine carries its ring locant, e.g.
    # 'N1-(4-aminophenyl)-N4-phenylbenzene-1,4-diamine (PIN)' (:26404),
    # 'N4-(7-chloroquinolin-4-yl)-N1,N1-diethylpentane-1,4-diamine' (:4675).
    # The producer used to cite the lowest substituted N with a bare 'N'
    # ('N,N-dimethylbenzene-1,4-diamine', 'N-methyl-N4-methyl...'). Identical
    # substituent names are multiplied across the N atoms:
    # 'N1,N4-dimethyl', 'N1,N1-dimethyl'. A compound substituent takes its own
    # enclosing marks first, as in the aniline producer above.
    from ..assembly.naming_utils import enclose_if_compound
    n_locs_by_name: Dict[str, List[int]] = {}
    for n_loc in sorted(new_n_map):
        for nm in new_n_map[n_loc]:
            n_locs_by_name.setdefault(nm, []).append(n_loc)
    n_prefix_entries: List[Tuple[str, str]] = []
    for nm, locs in n_locs_by_name.items():
        locs = sorted(locs)
        tags = ",".join(f"N{loc}" for loc in locs)
        body = _wrap_n_substituent(enclose_if_compound(nm))
        from ..assembly.naming_utils import multiplied_component as _mc
        n_prefix_entries.append((alpha_sort_key(nm), f"{tags}-{_mc(len(locs), nm, body)}"))

    # --- Ring-substituent prefix entries ---
    ring_prefix_entries: List[Tuple[str, str]] = []
    for name in new_ring:
        locants = sorted(new_ring[name])
        rendered = format_substituent_prefix(name, locants, len(locants))
        ring_prefix_entries.append((alpha_sort_key(name), rendered))

    # --- Parent suffix: benzene-<locants>-<mult>amine ---
    mult = SIMPLE_MULTIPLIERS.get(len(new_amines), str(len(new_amines)))
    loc_str = ",".join(str(l) for l in sorted(new_amines))
    suffix_part = f"benzene-{loc_str}-{_join_multiplied_suffix(mult, 'amine')}"

    all_entries = n_prefix_entries + ring_prefix_entries
    if not all_entries:
        return suffix_part
    # `e[1]` is an already-RENDERED prefix string, so its leading locants are
    # PARENT locants -> parent_locants=True compares a prefix's OWN
    # locants, and a parent locant is assigned BY this very order). `[1:]` keeps
    # the total-order tier so an exact tie cannot fall through to list order.
    all_entries.sort(key=lambda e: (e[0], prefix_citation_sort_key(
        e[1], parent_locants=True)[1:]))
    prefix_part = "-".join(rendered for _k, rendered in all_entries)
    return f"{prefix_part}{suffix_part}"


def _detect_amidine_n_substituents(mol, c_idx: int, parent_atoms: Set[int]):
    """D1 /: detect N/N'-substituents on an amidine C.

    Given the amidine carbon ``c_idx`` and the set of ``parent_atoms`` (the ring
    or chain atoms that carry the amidine, excluded so they are never walked into
    a substituent), return a list of ``(nlocant, name)`` tuples where:

      * ``nlocant == 'N'`` -> the amino (SINGLE-bonded, -NH-) nitrogen, and
      * ``nlocant == "N'"`` -> the imino (DOUBLE-bonded, =N-) nitrogen,

    per OPSIN amidine suffixRule labels /N1 (single) and /N2 (double). Each
    substituent is named through the existing ``name_substituent_fragment``
    pipeline (pure alkyl/nameable fragments only).

    Fail-closed: returns ``None`` if any N-substituent is NOT a nameable pure
    fragment (so the caller keeps the bare form and stays honest). An
    unsubstituted amidine returns ```` (empty list).
    """
    from ..assembly.substituent_naming import name_substituent_fragment
    c_atom = mol.GetAtomWithIdx(c_idx)
    entries = []
    for nbr in c_atom.GetNeighbors():
        if nbr.GetSymbol() != 'N':
            continue
        n_idx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(c_idx, n_idx)
        # Determine the italic-N locant strictly from the C-N bond order.
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            nlocant = "N'"          # imino / =N nitrogen
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            nlocant = "N"           # amino / -NH- nitrogen
        else:
            return None             # aromatic/other -> not a plain amidine; fail closed
        # Walk this nitrogen's non-carbon-parent substituents (skip H implicitly).
        for sub in nbr.GetNeighbors():
            sub_idx = sub.GetIdx()
            if sub_idx == c_idx or sub_idx in parent_atoms:
                continue
            if sub.GetSymbol() == 'O':
                # Wave2: amidoxime -C(=N-OH)-NH2 is an N'-hydroxy
                # (or N'-alkyloxy) amidine — the O on the imino N is the N'-hydroxy
                # substituent, NOT a fail-closed case. -OH -> 'hydroxy'; -O-R ->
                # '{alkyl}oxy'. Only the imino (=N, N') nitrogen bears it; a charged
                # / ring / multi-heteroatom O still fails closed.
                if (sub.GetFormalCharge() != 0
                        or mol.GetRingInfo().NumAtomRings(sub_idx) > 0):
                    return None
                o_heavy = [x for x in sub.GetNeighbors() if x.GetIdx() != n_idx]
                if not o_heavy:
                    entries.append((nlocant, 'hydroxy'))
                    continue
                if (len(o_heavy) == 1 and o_heavy[0].GetSymbol() == 'C'
                        and sub.GetTotalNumHs() == 0):
                    from ..assembly.substituent_prefix_forms import get_alkoxy_prefix
                    oxy = get_alkoxy_prefix(
                        mol, (sub_idx, o_heavy[0].GetIdx(), o_heavy[0].GetIdx()),
                        None,
                    )
                    if oxy:
                        entries.append((nlocant, oxy))
                        continue
                return None
            if sub.GetSymbol() != 'C':
                # A non-carbon substituent on the amidine N (e.g. N-N) is not
                # a plain N-alkyl/aryl amidine -> fail closed.
                return None
            #, BB 30468): an N-substituent that is itself an
            # imidoyl carbon R-C(=NH)- is the '{stem}animidoyl' prefix
            # ('ethanimidoyl' for CH3-C(=NH)-, 'methanimidoyl' for HC(=NH)-).
            # Detect: sub C with exactly one DOUBLE bond to a terminal, neutral,
            # unsubstituted =NH (degree 1) whose remaining heavy neighbours form
            # a pure alkyl. A substituted imino N fails closed (return None).
            _imino_ns = [
                x for x in sub.GetNeighbors()
                if x.GetSymbol() == 'N'
                and mol.GetBondBetweenAtoms(
                    sub_idx, x.GetIdx()).GetBondTypeAsDouble() == 2.0
            ]
            if _imino_ns:
                if len(_imino_ns) != 1:
                    return None
                _imN = _imino_ns[0]
                _imN_heavy = [
                    y for y in _imN.GetNeighbors() if y.GetIdx() != sub_idx
                ]
                if (_imN.GetFormalCharge() != 0 or _imN_heavy
                        or mol.GetRingInfo().NumAtomRings(sub_idx) > 0):
                    return None
                _im_alkyl, _im_cc = _collect_pure_alkyl(
                    mol, sub_idx,
                    set(parent_atoms) | {c_idx, n_idx, _imN.GetIdx()},
                )
                if _im_alkyl is None or _im_cc < 1:
                    return None
                from ..data.chain_names import get_chain_prefix
                entries.append(
                    (nlocant, f"{get_chain_prefix(_im_cc)}animidoyl")
                )
                continue
            alkyl_atoms, carbon_count = _collect_pure_alkyl(
                mol, sub_idx, set(parent_atoms) | {c_idx, n_idx}
            )
            if alkyl_atoms is None or carbon_count == 0:
                return None
            rec_name = name_substituent_fragment(
                mol, alkyl_atoms, sub_idx, list(set(parent_atoms) | {c_idx, n_idx})
            )
            if not rec_name:
                return None
            entries.append((nlocant, rec_name))
    return entries


def _build_amidine_n_prefix(n_substituents) -> str:
    """Build the N/N'-locant prefix for an amidine (carboximidamide) suffix.

    C2/D1: amidine N-substituent locants are N (the amino,
    single-bonded nitrogen) and N' (the imino, =N nitrogen). Entries are
    (nlocant, name) tuples where nlocant is 'N' or "N'". Identical substituent
    names are collapsed onto a shared multiplier /:
      [('N','methyl')] -> "N-methyl"
      [("N'",'methyl')] -> "N'-methyl"
      [('N','methyl'),("N'",'methyl')] -> "N,N'-dimethyl"
      [('N','methyl'),('N','methyl')] -> "N,N-dimethyl"
      [('N','ethyl'),("N'",'methyl')] -> "N-ethyl-N'-methyl" (alpha: ethyl<methyl)
      [('N','methyl'),("N'",'ethyl')] -> "N'-ethyl-N-methyl" (alpha: ethyl<methyl)
    Citation order is ALPHANUMERICAL by substituent NAME; the italic-N
    locant is only a tie-break, NEVER the primary key (the Blue Book dicarboximidamide
    PIN 'N''1-ethyl-N1,N1-dimethyl...' cites ethyl before dimethyl across N''1>N1).
    No trailing hyphen, so it can be concatenated directly before the base name.
    Empty list -> "" (plain amidine, both nitrogens unsubstituted).
    """
    from ..assembly.naming_utils import _wrap_n_substituent
    if not n_substituents:
        return ""

    # Group locants by substituent name so identical substituents share a
    # multiplier (N,N'-dimethyl) rather than being cited twice (N-methyl-N'-methyl).
    by_name: Dict[str, List[str]] = defaultdict(list)
    for nloc, name in n_substituents:
        by_name[name].append(nloc)

    def _loc_sort(nloc: str):
        # 'N' (unprimed) before "N'" (primed) at first point of difference.
        return (nloc.count("'"), nloc)

    segments = []  # (alpha_key, rendered)
    for name, nlocs in by_name.items():
        nlocs_sorted = sorted(nlocs, key=_loc_sort)
        loc_str = ",".join(nlocs_sorted)
        count = len(nlocs_sorted)
        if count > 1:
            from ..assembly.naming_utils import multiplied_component as _mc
            rendered = f"{loc_str}-{_mc(count, name, _wrap_n_substituent(name))}"
        else:
            rendered = f"{loc_str}-{_wrap_n_substituent(name)}"
        segments.append((alpha_sort_key(name), _loc_sort(nlocs_sorted[0]), rendered))

    # (the Blue Book): distinct substituents are cited in ALPHANUMERICAL order
    # by NAME; the italic-N locant is only a tie-break (the Blue Book dicarboximidamide
    # PIN cites 'N''1-ethyl' before 'N1,N1-dimethyl', i.e. ethyl<methyl wins over
    # the higher N-locant). Primary key = alpha (s[0]); tie-break = locant (s[1]).
    segments.sort(key=lambda s: (s[0], s[1]))
    return "-".join(rendered for _k, _l, rendered in segments)


def _name_substituted_benzamide(
    prefix_groups: Dict[str, List[int]],
    amide_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
    n_substituents: List[str],
) -> str:
    """
    Name substituted benzamide derivatives.

    Uses "benzamide" as the retained base name. Position 1 is the
    carboxamide position. Other substituents get locants relative to it.

    Args:
        prefix_groups: Non-amide substituent groups
        amide_locant: Locant of the amide in original numbering
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring
        n_substituents: List of N-alkyl names (e.g., ['methyl'] for N-methylbenzamide)

    Returns:
        IUPAC name like "4-methylbenzamide" or "N-methylbenzamide"
    """
    if not prefix_groups and not n_substituents:
        return "benzamide"

    # Renumber relative to amide position (amide = position 1)
    renumbered_groups = _renumber_relative_to(prefix_groups, amide_locant)

    # ------------------------------------------------------------------ #
    # (:7038) clause (b) (:7067) -- ONE multiplying prefix per #
    # substituent NAME, over the WHOLE locant set. #
    # ------------------------------------------------------------------ #
    # The N-substituents used to be spelled into their own string and glued in
    # front of the ring-prefix string. That produced TWO distinct defects, both
    # of which round-tripped cleanly through OPSIN + InChIKey (a round trip
    # proves the STRUCTURE, never the SPELLING):
    #
    # MERGE CNC(=O)c1ccc(C)cc1 -> N-methyl-4-methylbenzamide
    # PIN N,4-dimethylbenzamide (:32879)
    # ORDER CNC(=O)c1cccc(Cl)c1 -> N-methyl-3-chlorobenzamide
    # PIN 3-chloro-N-methylbenzamide (:32881)
    #
    # They are NOT one defect. `chloro` and `methyl` are different names and can
    # never merge -- that one is purely alphanumerical ORDER, and the
    # Blue Book settles the direction with a sulfonamide PIN that cites the
    # parent's `3-chloro` BEFORE the N-prefix (:32881).
    #
    # Both vanish under one structural change: multiplicity is a property of the
    # substituent NAME, not of which atom carries it (b), whose own
    # example list prints `dimethyl`), so an N-methyl and a ring 4-methyl are ONE
    # group of two. Put the N-substituents in the SAME name-keyed bucket carrying
    # the italic locant 'N', and `_build_prefix_string_with_locants` then does the
    # merging (name-keyed dict + count) and the alphanumerical ordering
    # (alpha_sort_key) that it already did for ring substituents alone.
    #
    # "Citation of locants" (:2869) is deny-by-default, so the whole
    # merged set is cited: `N,4-`. The Blue Book's own mixed italic/numeral sets
    # confirm the rendering -- `N,N,N,1-tetramethyl...` (:42213) and
    # `N,1,4-triphenyl...` (:42460) -- and `format_substituent_prefix` already
    # reproduces both byte-exactly, so NO new prefix formatter is added here.
    #
    # This is the same move `rules/heterocycles.py` makes for ring-N substituents
    # (Task P). It differs in one respect worth stating: there the ring N is a
    # NUMBERED skeletal atom and takes an arabic numeral, whereas an amide N is
    # not numbered and keeps its italic 'N' -- which is exactly why the merged
    # set here is mixed, and why `locant_sort_key` is needed to order it.
    merged: Dict[str, List] = {
        name: list(locants) for name, locants in renumbered_groups.items()
    }
    for sub_name in n_substituents:
        merged.setdefault(sub_name, []).append("N")

    return f"{_build_prefix_string_with_locants(merged, mono_needs_locant=True)}benzamide"


def _reanchor_name_to_mol(mol, name: Optional[str]) -> Optional[str]:
    """Gate-INDEPENDENT OPSIN InChIKey re-anchor (the 8afa533c guard-2 precedent).

    Returns ``name`` iff it OPSIN-parses to the SAME molecule as ``mol``
    (InChIKey-exact, so stereo- and atom-count-sensitive), else None.

    F-B needs this because the italic-N substituent names come from
    ``name_substituent_fragment``, a shared helper that can DROP a benzylic
    stereodescriptor (``1-phenylethyl`` for a chiral centre), MIS-NAME a
    functional-group branch (``-C(=O)OH`` -> ``formyl``), or otherwise yield a
    name that denotes a DIFFERENT molecule which still OPSIN-parses cleanly. Such
    a name is invisible to the OPSIN-validity gate and, for stereo, even ships at
    the default gate via the BBR stereo carve-out. The producer must therefore be
    honest ON ITS OWN: re-anchor, and fail closed on mismatch / no-parse / no-jar
    (the cascade then supplies a valid fallback, never a wrong molecule).
    """
    if not name:
        return None
    try:
        ref_key = inchikey_of(mol)
        if not ref_key:
            return None
        from ..namer import _validity_gate_name_to_smiles
        smi = _validity_gate_name_to_smiles(name)
        if not smi:
            return None
        got = Chem.MolFromSmiles(smi)
        if got is None:
            return None
        return name if Chem.MolToInchiKey(got) == ref_key else None
    except Exception:
        return None


def _build_n_substituted_sulfamoyl_prefix(
    n_substituents: List[str],
) -> Optional[str]:
    """: the '{N-substituents}sulfamoyl' substituent prefix.

    The N-substituents are cited WITHOUT the italic N (the Blue Book
    ``dimethylsulfamoyl``,:32985 ``phenylsulfamoyl``). CONSERVATIVE: only the
    single-substituent (``methylsulfamoyl``, ``phenylsulfamoyl``) and
    identical-multiplied (``dimethylsulfamoyl``) cases are built. DISTINCT
    N-substituents need the nested ``ethyl(methyl)sulfamoyl`` form, which is not
    built -> return None (fail closed). The caller re-anchors the whole name, so a
    fragment-namer mis-spelling is caught regardless.
    """
    names = [n for n in (n_substituents or []) if n]
    if not names or len(names) > 2:
        return None
    if len(set(names)) != 1:
        return None  # distinct N-substituents: nested form not built
    base = names[0]
    if len(names) == 2:
        from ..assembly.naming_utils import multiplied_component as _mc
        base = _mc(2, base, base)
    # A substituted substituent -> enclosed per ('(methylsulfamoyl)',
    # '(dimethylsulfamoyl)'). Pre-enclose here; the downstream formatter leaves an
    # already-bracketed name alone, so there is no double-enclosing.
    return f"({base}sulfamoyl)"


def _name_substituted_benzenesulfonamide(
    prefix_groups: Dict[str, List[int]],
    sulfonamide_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
    n_substituents: Optional[List[str]] = None,
) -> str:
    """
    Name substituted benzenesulfonamide derivatives (F-B).

    Uses "benzenesulfonamide" as the base name. Position 1 is the sulfonamide
    position. Two Blue Book rules are applied here:

    * **(c) / ** (``the Blue Book`` / ``:2869``): the
      suffix locant ``1`` is omitted ONLY on a MONOSUBSTITUTED homogeneous
      monocyclic ring. A RING substituent makes the ring DI-substituted, so the
      ``1`` is cited — ``4-methylbenzene-1-sulfonamide`` — exactly as the
      sulfonic-acid sibling path already emits ``4-methylbenzene-1-sulfonic
      acid``. An N-substituent alone does not substitute the RING, so the bare
      ``N-methylbenzenesulfonamide`` keeps no ``1``.
    * ** + ** (``:32879`` ``N,4-dimethyl…benzamide (PIN)``):
      the italic-``N`` substituents and the ring numerals form ONE merged,
      alphanumerically ordered prefix list. This is the identical structural move
      ``_name_substituted_benzamide`` makes — the N-substituents go into the
      SAME name-keyed bucket carrying the italic locant ``N`` and
      ``_build_prefix_string_with_locants`` does the pooling (``N,4-dimethyl``)
      and the ``N``-before-numeral ordering.

    Args:
        prefix_groups: Non-sulfonamide RING substituent groups.
        sulfonamide_locant: Locant of the sulfonamide in original numbering.
        atom_to_locant: Mapping from atom index to locant.
        oriented_ring: The oriented ring.
        n_substituents: Italic-N substituent names (e.g. ['methyl']).

    Returns:
        IUPAC name like "4-methylbenzene-1-sulfonamide",
        "N,4-dimethylbenzene-1-sulfonamide", or "N-methylbenzenesulfonamide".
    """
    n_substituents = n_substituents or []

    # Renumber ring prefixes so the sulfonamide sits at position 1, then merge in
    # the italic-N substituents (locant 'N') as one alphabetised list.
    renumbered_groups = _renumber_relative_to(prefix_groups, sulfonamide_locant)
    merged: Dict[str, List] = {
        name: list(locants) for name, locants in renumbered_groups.items()
    }
    for sub_name in n_substituents:
        merged.setdefault(sub_name, []).append("N")

    if not merged:
        return "benzenesulfonamide"

    prefix_part = _build_prefix_string_with_locants(merged, mono_needs_locant=True)

    # A RING substituent (prefix_groups) — not merely an N-substituent — cites '1'.
    if prefix_groups:
        return f"{prefix_part}benzene-1-sulfonamide"
    return f"{prefix_part}benzenesulfonamide"


def _name_substituted_benzaldehyde(
    prefix_groups: Dict[str, List[int]],
    aldehyde_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
) -> str:
    """
    Name substituted benzaldehyde derivatives.

    Uses 'benzaldehyde' as retained base per IUPAC.
    Position 1 = CHO-bearing carbon.

    Pattern follows _name_substituted_benzoic_acid.

    Args:
        prefix_groups: Non-aldehyde substituent groups
        aldehyde_locant: Locant of the aldehyde in original numbering
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring

    Returns:
        IUPAC name like "4-chlorobenzaldehyde"
    """
    if not prefix_groups:
        return "benzaldehyde"

    # Renumber relative to aldehyde position (aldehyde = position 1)
    renumbered_groups = _renumber_relative_to(prefix_groups, aldehyde_locant)

    # Build prefixes
    prefix_part = _build_prefix_string_with_locants(renumbered_groups, mono_needs_locant=True)

    return f"{prefix_part}benzaldehyde"


def _name_substituted_chalcogen_carbaldehyde(
    prefix_groups: Dict[str, List[int]],
    suffix_locant: int,
    suffix_name: str,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
) -> str:
    """
    Name a benzene bearing a single chalcogen aldehyde (-CH=S/Se/Te).

     C- (V-2). Unlike benzaldehyde, the chalcogen analogues have no
    retained base name, so the systematic 'benzenecarbo*aldehyde'
    base is used:

      - unsubstituted: ``benzenecarboselenaldehyde``
      - substituted: ``4-methylbenzene-1-carboselenaldehyde`` (the principal
        group takes locant 1 and is cited because other prefixes are present,
        

    Args:
        prefix_groups: Non-suffix substituent groups.
        suffix_locant: Locant of the chalcogen-aldehyde carbon (original numbering).
        suffix_name: 'carbothialdehyde' / 'carboselenaldehyde' / 'carbotelluraldehyde'.
        atom_to_locant: Mapping from atom index to locant.
        oriented_ring: The oriented ring.
    """
    if not prefix_groups:
        return f"benzene{suffix_name}"

    # Renumber relative to the chalcogen-aldehyde position (= position 1).
    renumbered_groups = _renumber_relative_to(prefix_groups, suffix_locant)
    prefix_part = _build_prefix_string_with_locants(
        renumbered_groups, mono_needs_locant=True
    )

    return f"{prefix_part}benzene-1-{suffix_name}"


def _renumber_relative_to(
    groups: Dict[str, List[int]],
    reference_locant: int,
) -> Dict[str, List[int]]:
    """
    Renumber substituent locants relative to a reference position.

    The reference position becomes position 1 (criterion (c) has already put the
    principal characteristic group there). The two remaining numberings -- clockwise
    and counterclockwise -- are then separated by the SAME ordered criteria
    ``orient_benzene`` uses, or the two paths would disagree about the identical
    molecule:

    * **(f)** ``the Blue Book Blue Book`` *detachable alphabetized prefixes,
      all considered together in a series of increasing numerical order* (worked at
      ``:3305``);
    * **(g)** ``:3307`` *lowest locants for the substituent cited first as a prefix
      in the name* (worked at ``:3315``, and at ``:3317``
      ``1-methyl-4-nitronaphthalene (PIN) (not 4-methyl-1-nitronaphthalene)``;
      restated as § ``:26085``).

    ⚠ Phase C Task 9b: (g) is new here. Without it a tie on (f) silently kept
    ``direction = 1`` -- the direction ``orient_benzene`` happened to hand over,
    i.e. the input SMILES' atom order. That is what made ONE molecule,
    ``Cc1cccc(Cl)c1O``, come out as ``2-chloro-6-methylphenol`` from one spelling
    and ``6-chloro-2-methylphenol`` from another: both directions give the prefix
    set {2,6}, so only (g) can decide, and ``chloro`` is cited before ``methyl``.

    No further tier is needed: if (f) and (g) both tie then every name holds the
    same locants in both directions, so the two ``converted`` maps are equal and
    the choice is not observable. The order is total.

    Args:
        groups: Dict of name -> list of locants in original numbering
        reference_locant: The locant that should become position 1

    Returns:
        Dict of name -> list of renumbered locants
    """
    best_groups = None
    best_key = None

    for direction in [1, -1]:
        converted: Dict[str, List[int]] = defaultdict(list)

        for name, locants in groups.items():
            for old_loc in locants:
                diff = (old_loc - reference_locant) * direction
                new_loc = (diff % 6) + 1
                if new_loc == 1:
                    # Position 1 is reserved for the principal group
                    new_loc = 6
                converted[name].append(new_loc)

        # Sort locants within each group
        for name in converted:
            converted[name].sort()

        # (f) the whole prefix series, then (g) per prefix in citation order.
        all_locants = sorted([loc for locs in converted.values() for loc in locs])
        citation_locants = tuple(
            tuple(converted[name])
            for name in sorted(converted, key=alpha_sort_key)
        )
        key = (all_locants, citation_locants)

        if best_key is None or key < best_key:
            best_key = key
            best_groups = dict(converted)

    return best_groups if best_groups else {}


_BENZENE_PARENT_HYDRIDE_CACHE = []


def _benzene_parent_hydride():
    """The parent hydride benzene, C6H6, atom ``i`` <-> ring locant ``i + 1``.

     speaks of the substitutable positions of the PARENT, so the licence must
    be evaluated against the undecorated ring: on the input molecule a fully
    substituted ring carbon has zero hydrogens, and the count that decides the licence
    would be lost. Benzene's six carbons each carry exactly one substitutable H, which
    is precisely why ``benzenehexol`` (the Blue Book) omits while
    ``cyclohexane-1,2,3,4,5,6-hexol`` (the Blue Book) retains.
    """
    if not _BENZENE_PARENT_HYDRIDE_CACHE:
        from rdkit import Chem
        _BENZENE_PARENT_HYDRIDE_CACHE.append(Chem.MolFromSmiles('c1ccccc1'))
    return _BENZENE_PARENT_HYDRIDE_CACHE[0]


def _benzene_l5_uniform_licence(
    mol,
    oriented_ring: List[int],
    suffix_locants_by_kind: Optional[Dict[str, List[int]]],
    prefix_groups: Optional[Dict[str, List[int]]],
    stereo_descriptors: Optional[List] = None,
) -> bool:
    """ (the Blue Book): may this benzene ring omit ALL of its locants?

     Phase C tranche B Task 1. **DENY BY DEFAULT** -- "Citation of locants"
    (the Blue Book) says *"if any locants are essential... then all locants must be cited
    for the parent structure or that structural unit"*, so this returns True only when
    the ring is completely substituted **in the same way**, and False for everything it
    cannot positively establish.

    Licensed (all must hold):
      * the parent is a six-membered all-carbon aromatic ring -- the benzene handler's
        own parent hydride, whose every position carries exactly one substitutable H;
      * every one of the six positions is decorated, and every decoration is the same
        kind (suffix and prefix kinds are distinguished, so ``5 x -ol + 1 x chloro``
        is heterogeneous and denied -- the Blue Book *"In case of partial substitution or
        modification, all numerical prefixes must be indicated"*);
      * nothing else in the scope forces locants: no stereodescriptor, no isotopic
        label (the Blue Book).

    ⚠ The decision is delegated to ``assembly.locant_omission`` -- the ONE place the
    Blue Book licences live -- and this function only marshals benzene's scope into it.
    Do not re-derive the rule here.

    ⚠ It counts HYDROGENS, not positions. The whole boundary pair
    (``benzenehexol`` omits, ``cyclohexane-1,2,3,4,5,6-hexol`` retains) turns on the
    ring carbon having one substitutable H rather than two, and a predicate that counts
    positions strips the inositols -- which the Blue Book spells out in one sentence.

    ``stereo_descriptors=None`` means "the caller did not establish them" and fails
    closed, because a stereodescriptor is an essential locant in the same scope.

    ⚠ **Measured with mutation testing, 2026-07-28 (22 mutations, 20 caught) -- do not
    "simplify" the two redundant guards.**

      * disabling the aromatic/all-carbon loop broke NO test, and that was a REAL
        DEFECT rather than redundancy: the licence measures against benzene's parent
        hydride, so without the loop it licensed ``hexamethylcyclohexane``. Two
        witnesses (one per reason the loop can fire) now cover it;
      * the ``1 <= loc <= 6`` range check and the ``len(oriented_ring) != 6`` check
        each survive mutation *individually*, because ``l5_uniform_complete`` validates
        every atom index against the parent hydride's atom count and requires every
        substitutable position to be decorated -- so a locant of 7 becomes atom index 6
        (out of range) and a five-membered ring leaves position 6 undecorated. Removing
        the primitive's index validation IS caught (by
        ``test_out_of_range_index_denies``). They are kept because they state the rule
        at benzene's own scope and defend the licence if the primitive is ever
        loosened -- not because they are independently exercised.
    """
    # (the Blue Book) ambient scope -- see locant_omission.forced_locant_scope.
    # The isotope path names an isotope-STRIPPED molecule, so this function's own
    # `has_isotope` test is structurally unreachable-True there (measured: a trace recorded
    # isotopes_seen_in_mol= for a 13C input). MEASURED consequence:
    # `Cc1c(C)c(C)c(C)c(C)[13c]1C` shipped `hexamethyl(13C1)benzene` against
    # (the Blue Book). Conditional by construction: the decorator enters the scope only after
    # establishing that the label needs a locant, so `(13C1)benzenehexol` -- correct per
    # (the Blue Book), all six positions one orbit -- is unaffected.
    from ..assembly.locant_omission import locants_are_forced
    if locants_are_forced():
        return False

    from ..assembly.locant_omission import l5_uniform_complete, scope_forces_locants

    if mol is None or not oriented_ring or len(oriented_ring) != 6:
        return False

    # Confirm the perceived parent really is a benzene ring before licensing anything
    # against benzene's parent hydride.
    for idx in oriented_ring:
        try:
            atom = mol.GetAtomWithIdx(int(idx))
        except (OverflowError, RuntimeError, ValueError, TypeError):
            return False
        if atom.GetSymbol() != 'C' or not atom.GetIsAromatic():
            return False

    # Marshal the scope: ring locant (1..6) -> decoration kind, keyed so a suffix and a
    # prefix of the same spelling can never be mistaken for "the same way".
    decoration_of: Dict[int, str] = {}
    all_suffix_locants: List = []
    all_prefix_locants: List = []
    for source, groups, bucket in (
        ('suffix', suffix_locants_by_kind or {}, all_suffix_locants),
        ('prefix', prefix_groups or {}, all_prefix_locants),
    ):
        for kind, locants in groups.items():
            for loc in (locants or []):
                bucket.append(loc)
                if isinstance(loc, bool) or not isinstance(loc, int):
                    return False
                if not 1 <= loc <= 6:
                    return False
                if loc - 1 in decoration_of:
                    return False        # two decorations claiming one position
                decoration_of[loc - 1] = f"{source}:{kind}"

    if stereo_descriptors is None:
        stereo_text = None              # not established -> fail closed
    else:
        stereo_text = 'stereo' if stereo_descriptors else ''

    has_isotope = any(a.GetIsotope() for a in atoms_of(mol))

    if scope_forces_locants(
        prefix_locants=all_prefix_locants,
        suffix_locants=all_suffix_locants,
        stereo_text=stereo_text,
        has_indicated_h=False,
        has_isotope=has_isotope,
        # A ring assembly / multiplicative name / skeletal replacement never reaches
        # this handler: it names ONE benzene ring as the whole parent. Measured
        # 2026-07-28 -- a validated call-trace recorded zero hits here for
        # 1,1'-biphenyl and 1,1'-oxydibenzene.
        is_multiplicative=False,
        is_ring_assembly=False,
        has_skeletal_replacement=False,
    ):
        return False

    return l5_uniform_complete(
        _benzene_parent_hydride(), decoration_of=decoration_of)


def _build_prefix_string(prefix_groups: Dict[str, List[int]]) -> str:
    """
    Build prefix part string from prefix groups.

    For groups where locants are all 1 and single, omits locants.
    Handles monosubstituted (no locant needed).

    Args:
        prefix_groups: Dict of prefix name -> list of locants

    Returns:
        Prefix string to prepend to parent name
    """
    if not prefix_groups:
        return ""

    total = sum(len(locs) for locs in prefix_groups.values())
    is_mono = total == 1
    _omit = should_omit_locant_one(
        context="prefix",
        is_ring=True,
        is_heterocyclic=False,  # Benzene is always carbocyclic
        is_monosubstituted=is_mono,
    )

    prefixes = []
    for name in sorted(prefix_groups.keys(), key=alpha_sort_key):
        locants = prefix_groups[name]
        count = len(locants)

        if _omit:
            if name.startswith('(') or name.startswith('['):
                # Already has enclosing marks -- keep as-is
                prefix_str = name
            elif is_complex_substituent(name):
                # Complex substituent needs enclosing marks per IUPAC
                prefix_str = f"({name})"
            else:
                prefix_str = name
        else:
            prefix_str = format_substituent_prefix(name, locants, count)

        prefixes.append(prefix_str)

    return _join_benzene_prefixes(prefixes)


def _build_prefix_string_with_locants(
    prefix_groups: Dict[str, List[int]],
    mono_needs_locant: bool = True,
) -> str:
    """
    Build prefix string where locants are always included (for substituted retained names).

    For "4-methylbenzamide", the locant 4 is needed even for mono-substitution.

    Args:
        prefix_groups: Dict of prefix name -> list of locants
        mono_needs_locant: If True, even single substituents include locant

    Returns:
        Prefix string like "4-methyl" or "2-hydroxy"
    """
    if not prefix_groups:
        return ""

    from ..assembly.naming_utils import locant_sort_key

    prefixes = []
    for name in sorted(prefix_groups.keys(), key=alpha_sort_key):
        # (:2869): the whole locant set of the group is cited, in order.
        # A MERGED group may hold italic ('N') and numeric locants together, and
        # sorted(['N', 4]) raises TypeError -- locant_sort_key puts the italic
        # letters first, matching `N,N,N,1-tetramethyl...` (:42213).
        locants = sorted(prefix_groups[name], key=locant_sort_key)
        count = len(locants)

        prefix_str = format_substituent_prefix(name, locants, count)
        prefixes.append(prefix_str)

    result = _join_benzene_prefixes(prefixes)

    return result


def _name_substituted_benzonitrile(
    substituent_groups: Dict[str, List[int]],
    nitrile_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int]
) -> str:
    """
    Name a substituted benzonitrile.

    IUPAC 2013: substituents are numbered relative to the nitrile position (position 1).
    Example: 4-chlorobenzonitrile, 4-methylbenzonitrile

    The nitrile carbon position becomes position 1 in the benzonitrile numbering.
    For a 6-membered ring with nitrile at old position N:
    - Position N becomes 1
    - Other positions are renumbered going clockwise (or counterclockwise for lowest locants)

    Args:
        substituent_groups: Dict of substituent name -> list of locants (in original numbering)
        nitrile_locant: The locant of the nitrile in the original numbering
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring

    Returns:
        IUPAC name string (e.g., "4-chlorobenzonitrile")
    """
    # Try both directions (clockwise and counterclockwise) and pick lowest locants
    best_groups = None
    best_locant_set = None

    for direction in [1, -1]:
        converted_groups: Dict[str, List[int]] = defaultdict(list)

        for name, locants in substituent_groups.items():
            for old_loc in locants:
                # Calculate new position relative to nitrile at position 1
                # direction = 1: clockwise numbering from nitrile
                # direction = -1: counterclockwise numbering from nitrile
                # Formula: new_pos = ((old_pos - nitrile_pos) * direction % 6) + 1
                # This ensures nitrile_pos -> 1, and other positions follow in order
                diff = (old_loc - nitrile_locant) * direction
                new_loc = (diff % 6) + 1
                if new_loc == 1:
                    # Position 1 is reserved for nitrile; this shouldn't happen
                    # for other substituents, but handle edge case
                    new_loc = 7 - new_loc  # Map to position 6 (opposite direction)
                converted_groups[name].append(new_loc)

        # Sort locants within each group
        for name in converted_groups:
            converted_groups[name].sort()

        # Calculate locant set for comparison
        all_locants = sorted([loc for locs in converted_groups.values() for loc in locs])

        if best_locant_set is None or all_locants < best_locant_set:
            best_locant_set = all_locants
            best_groups = dict(converted_groups)

    # Build prefix strings. Both mono- and poly-substituted use the shared
    # formatter, which cites the locant (the other substituent is never at
    # position 1 -- the nitrile is) AND applies enclosing marks to a
    # complex name. (BLOCKER 2: the old monosubstituted f-string
    # `f"{locants[0]}-{name}"` never bracketed, shipping the malformed
    # `4-(1E)-3-hydroxyprop-1-en-1-ylbenzonitrile`; siblings via
    # format_substituent_prefix bracketed correctly.)
    prefixes = []
    for name in sorted(best_groups.keys(), key=alpha_sort_key):
        locants = best_groups[name]
        count = len(locants)
        prefixes.append(format_substituent_prefix(name, locants, count))

    # Join prefixes
    prefix_part = _join_benzene_prefixes(prefixes)

    return f"{prefix_part}benzonitrile"


def _join_benzene_prefixes(prefixes: List[str]) -> str:
    """
    Join benzene substituent prefixes with proper hyphenation.

    When one prefix ends with a letter and the next starts with a digit,
    a hyphen is needed.

    Args:
        prefixes: List of formatted prefix strings

    Returns:
        Joined prefix string
    """
    if not prefixes:
        return ""

    if len(prefixes) == 1:
        return prefixes[0]

    result = prefixes[0]
    for i in range(1, len(prefixes)):
        current = prefixes[i]

        # Check if we need a hyphen
        if result and current:
            last_char = result[-1]
            first_char = current[0]

            # Hyphen needed between alpha/closing-mark and a following LOCANT
            # e.g. "1-(N,N-dimethylamino)" + "4-amino" needs a hyphen after ")";
            # likewise after a bracket/brace-closed complex prefix
            # ("1-[4-(1-chloroethyl)phenoxy]" + "4-methyl",.
            #
            # The test used to be `first_char.isdigit`, which misses an ITALIC
            # locant because it starts with a LETTER: "3-chloro" + "N-methyl"
            # glued into "3-chloroN-methylbenzamide". OPSIN parsed that to the
            # correct InChIKey, so no structural oracle could catch it.
            from ..assembly.naming_utils import starts_with_locant
            if ((last_char.isalpha() or last_char in ')]}')
                    and (first_char.isdigit() or starts_with_locant(current))):
                result += "-"

        result += current

    return result


def _select_benzene_parent_ring(mol, principal_group_atoms=None,
                                principal_group=None) -> Optional[Tuple[int, ...]]:
    """Deterministically select the benzene ring to use as the parent.

     C- (V-3): ``get_benzene_ring`` returns the FIRST benzene ring in SSSR
    order, which is SMILES-atom-order-dependent. For a molecule with two benzene
    rings connected by a bridge (an aralkyl ether, e.g. benzyl phenyl ether
    ``c1ccccc1COc1ccccc1``) the choice of parent flipped with the input spelling
    -> non-determinism (``(phenoxymethyl)benzene`` vs ``benzoxybenzene``).

    Among the benzene rings, prefer one that is **carbon-linked** (a ring atom
    has a non-ring carbon neighbour) over a purely heteroatom-linked ring, then
    break ties by the lowest canonical-atom-rank tuple. The carbon-linked
    preference keeps the bridging ether oxygen in the substituent prefix
    (-> ``phenoxymethyl``, the form) and avoids the heteroatom-linked
    parent path that mis-names ``-O-CH2-Ar``. Single-benzene molecules (the
    overwhelming majority) are unaffected — the lone ring is returned.
    """
    benzene_rings = [r for r in mol.GetRingInfo().AtomRings()
                     if is_benzene_ring(mol, r)]
    if not benzene_rings:
        return None
    if len(benzene_rings) == 1:
        return benzene_rings[0]

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))

    def _carbon_linked(ring: Tuple[int, ...]) -> bool:
        ring_set = set(ring)
        for a in ring:
            for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
                if nbr.GetIdx() not in ring_set and nbr.GetSymbol() == 'C':
                    return True
        return False

    def _substituent_count(ring: Tuple[int, ...]) -> int:
        ring_set = set(ring)
        return sum(
            1 for a in ring
            for nbr in mol.GetAtomWithIdx(a).GetNeighbors()
            if nbr.GetIdx() not in ring_set and nbr.GetAtomicNum() > 1
        )

    def _pcg_count(ring: Tuple[int, ...]) -> int:
        # (the Blue Book): "The senior parent structure has the
        # maximum number of substituents corresponding to the principal
        # characteristic group (suffix)". It comes BEFORE every ring criterion
        # below: COc1cc(O)cc(C)c1Oc1cc(C)cc(O)c1O is named on the ring with two
        # -OH ('3-(4-hydroxy-2-methoxy-6-methylphenoxy)-5-methylbenzene-1,2-diol'),
        # not '4-(2,3-dihydroxy-5-methylphenoxy)-3-methoxy-5-methylphenol', which
        # the equal substituent counts left to the canonical-rank tie-break
        # (TRIAGE rows 75, 85). Counted per principal-group instance.
        if not principal_group_atoms:
            return 0
        from .parent_selection import is_principal_group_on_ring
        ring_set = set(ring)
        return sum(1 for m in principal_group_atoms
                   if is_principal_group_on_ring(mol, ring_set, [m], principal_group))

    def _key(ring: Tuple[int, ...]):
        # first (above); then carbon-linked (0 < 1); then
        # (Wave-2 C2): the ring with the GREATER number of substituent
        # attachments is the parent (the BB dicyano-phenoxy example — this tier
        # also killed a genuine spelling-dependence: the old rank-only tie-break
        # flipped parent rings with the input SMILES order); then lowest
        # canonical-rank tuple (a structure-derived, spelling-independent total
        # order).
        return (-_pcg_count(ring),
                0 if _carbon_linked(ring) else 1,
                -_substituent_count(ring),
                tuple(sorted(ranks[a] for a in ring)))

    return min(benzene_rings, key=_key)


def _benzene_parent_candidates(mol) -> List[Tuple[int, ...]]:
    """The benzene rings that tie for the parent role after the structure-derived
    pre-filter of ``_select_benzene_parent_ring`` (carbon-linked, then greater
    substituent count).

    A single-benzene molecule returns its lone ring (so naming is byte-identical
    to the old single-parent path). When two or more benzene rings tie — the
     diaryl-linked-by-heteroatom class (e.g. a stereo-differing diaryl
    ether) — ALL tied rings are returned so ``name_benzene_derivative`` can name
    each candidate and select the preferred parent by the alphanumerical order of
    the complete names with the 'R' < 'S' tie-break. The
    canonical-rank total order that formerly broke the tie is retained only as a
    defensive last resort inside the name comparison.
    """
    benzene_rings = [r for r in mol.GetRingInfo().AtomRings()
                     if is_benzene_ring(mol, r)]
    if len(benzene_rings) <= 1:
        return benzene_rings

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))

    def _carbon_linked(ring: Tuple[int, ...]) -> bool:
        ring_set = set(ring)
        for a in ring:
            for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
                if nbr.GetIdx() not in ring_set and nbr.GetSymbol() == 'C':
                    return True
        return False

    def _substituent_count(ring: Tuple[int, ...]) -> int:
        ring_set = set(ring)
        return sum(
            1 for a in ring
            for nbr in mol.GetAtomWithIdx(a).GetNeighbors()
            if nbr.GetIdx() not in ring_set and nbr.GetAtomicNum() > 1
        )

    def _tier(ring: Tuple[int, ...]):
        return (0 if _carbon_linked(ring) else 1, -_substituent_count(ring))

    best_tier = min(_tier(r) for r in benzene_rings)
    tied = [r for r in benzene_rings if _tier(r) == best_tier]
    # deterministic order for the defensive fallback (canonical-rank tuple)
    tied.sort(key=lambda r: tuple(sorted(ranks[a] for a in r)))
    return tied


_STEREO_PAREN_BLOCK_RE = re.compile(r'\((\d*[RSEZ*](?:,\d*[RSEZ*])*)\)')


def _alphanumerical_name_key(name: str):
    """ / preference key for choosing between candidate parent
    names of a diaryl-linked scaffold.

    Returns ``(letters, locants, stereo)`` so ``min`` selects the preferred PIN:
      * ``letters`` — the Roman letters in order of appearance, lowercased, with
        stereodescriptor blocks removed / line 3446: stereodescriptors are
        NOT part of the alphanumerical comparison) — primary;
      * ``locants`` — the numeric locants in order of appearance —
        secondary (compared only when the letters are identical);
      * ``stereo`` — the 'R'/'S' descriptor letters in order of appearance
        ('R' < 'S') — final tie-break, applied only when letters and
        locants are identical (the constitution-symmetric stereo-differing case).
    """
    stereo = tuple(
        ch for block in _STEREO_PAREN_BLOCK_RE.findall(name)
        for ch in block if ch in 'RS'
    )
    core = _STEREO_PAREN_BLOCK_RE.sub('', name)
    letters = ''.join(ch.lower() for ch in core if ch.isalpha())
    locants = tuple(int(m) for m in re.findall(r'\d+', core))
    return (letters, locants, stereo)


def _preferred_benzene_parent_ring(mol):
    """Name each candidate parent benzene ring and return the (ring, name) whose
    complete name is preferred by (alphanumerical order) with the
     'R' < 'S' tie-break. Single-benzene molecules return their lone
    (ring, name). Returns None when no benzene ring can be named.

    This is the single authority for the diaryl-linked-by-heteroatom parent
    choice /: both ``name_benzene_derivative`` and the main
    namer dispatch consult it, so the pipeline names exactly the ring this
    selects. The comparison is over the NAME STRINGS (structure-derived, so
    deterministic — never atom-order-dependent).
    """
    candidates = _benzene_parent_candidates(mol)
    if not candidates:
        return None
    # Computed ONCE for the molecule (it is ring-independent) rather than per
    # candidate ring, so the extra perception cost is paid at most once.
    _pg = _molecule_principal_group(mol)
    scored = []
    for ring_atoms in candidates:
        substituents = get_benzene_substituents(mol, ring_atoms)
        if not substituents:
            nm = "benzene"
        else:
            # W3-P04 (c)): anchor the principal characteristic group to
            # the lowest locant before detachable substituents (mirrors the
            # composer._assemble_benzene_name path and namer.py Branch 3, so the
            # NAME and locant HINT agree).
            # Phase C Task 9: routed through the shared authority. This site
            # has no ``features``, so it derives the molecule-level principal group
            # from the same seniority entry point the rest of the pipeline uses.
            # It is only reached for MULTI-benzene molecules whose principal group
            # is None or a secondary amine (see composer._assemble_benzene_name),
            # and it passes no detected_fgs to name_substituted_benzene either, so
            # the promotion guard stays consistent with the name built here.
            pcg_positions = principal_group_ring_atoms(
                mol, ring_atoms, substituents, principal_group=_pg,
            )
            oriented_ring = orient_benzene(
                mol, ring_atoms, substituents,
                principal_group_positions=pcg_positions or None,
            )
            nm = name_substituted_benzene(mol, ring_atoms, oriented_ring, substituents)
        if nm:
            scored.append((ring_atoms, nm))
    if not scored:
        return None
    return min(scored, key=lambda rn: _alphanumerical_name_key(rn[1]))


def name_benzene_derivative(mol) -> Optional[str]:
    """
    Generate name for a benzene derivative.

    This is the main entry point for benzene naming.

    Args:
        mol: RDKit Mol object

    Returns:
        IUPAC name string, or None if not a benzene derivative
    """
    # Preferred parent benzene ring / for diaryl scaffolds;
    # the lone ring for a single-benzene molecule).
    preferred = _preferred_benzene_parent_ring(mol)
    if preferred is None:
        return None
    return preferred[1]
