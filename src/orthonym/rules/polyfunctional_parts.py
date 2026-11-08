"""The routes and the per-group steps of ``polyfunctional.name_polyfunctional``, moved here verbatim. A route
returns the name (or None, a decline) or ``_GO_ON`` when it does not apply; a group step returns True when it
handled the group (the loop goes on to the next one), a filter step returns that flag and the matches it keeps."""
from collections import defaultdict  # noqa: F401

from .polyfunctional import (  # noqa: F401
    BRANCH_HANDLED_FGS, List, _CHAIN_END_ACID_DERIVATIVE_FGS, _SPLIT_DEFAULT_PGS, _acyl_is_simple_saturated_chain, _chain_end_acid_derivative_prefixes, _fg_located_atoms, _find_fg_center_atom, _name_amidine_chain_side, _name_ring_as_parent_polyfunctional, format_fg_prefix, get_non_principal_fg_locants, logger,
)

#: a route that does not apply
_GO_ON = object()


def _route_without_chain(atom_to_locant, features, mol, principal_chain, principal_group):
    """No principal chain or locant map: the routes that name the molecule without one."""
    if not principal_chain or not atom_to_locant:
        # "When anions are identical functional class multiplicative nomenclature is used"
        # (the Blue Book): a polyester of one alcoholic component whose acid components carry a ring (a ring ether,
        # a saturated ring with a hydroxy group) has no chain, so ``_route_ester_second`` below is never reached for it;
        # the multiplicative name is offered here, and a molecule it does not name goes on as before.
        if principal_group == "ester":
            _ester_matches = features.functional_groups.get("ester", [])
            if len(_ester_matches) >= 2 and not any(
                    mol.GetBondBetweenAtoms(_m[0], _m[2]).IsInRing() for _m in _ester_matches):     # no lactone
                from ..rules.esters import _try_functional_class_diol_diester
                _class_name = _try_functional_class_diol_diester(mol, _ester_matches)
                if _class_name:
                    return _class_name
        # --- a phase-02: Ring-as-parent polyfunctional path ---
        # When principal_chain is None but the molecule is cyclic, attempt
        # ring-as-parent naming. Conservative guards prevent intercepting
        # compounds better handled by specialized fallthrough handlers:
        # - Only monocyclic (1 ring) -- fused/polycyclic have specialized handlers
        # - Only fully saturated rings -- unsaturated rings need -ene/-yne handling
        # - No aromatic rings -- benzene/pyridine/etc. have retained name handlers
        if getattr(features, 'is_cyclic', False) and not getattr(features, 'chain_is_parent', False) and principal_group:
            ring_info = mol.GetRingInfo()
            n_rings = ring_info.NumRings()
            if n_rings == 1:
                ring_atoms = list(ring_info.AtomRings()[0])
                ring_set = set(ring_atoms)
                ring_size = len(ring_atoms)
                total_heavy = mol.GetNumHeavyAtoms()

                # Guard: ring must be a significant portion of the molecule.
                # If the ring is < 40% of heavy atoms, it's likely a
                # substituent on a chain parent, not the parent itself.
                # E.g., sphingolipid with cyclohexane ring but 50+ chain atoms.
                # W2F-P3: EXCEPT when the principal characteristic
                # group is attached to the ring — then the ring necessarily
                # holds the PCG and MUST be the parent (the acyclic branch bears
                # only junior groups), regardless of relative size. Without this
                # exception a large FG-bearing acyclic substituent (item 15) drops
                # the ring below the size heuristic and the whole molecule fails
                # closed even though cyclohexane-1-carboxylic acid is the PIN parent.
                _pcg_on_ring = False
                for _match in (getattr(features, 'principal_group_atoms', None) or []):
                    for _a in _match:
                        if _a in ring_set:
                            _pcg_on_ring = True
                            break
                        for _nb in mol.GetAtomWithIdx(_a).GetNeighbors():
                            if _nb.GetIdx() in ring_set:
                                _pcg_on_ring = True
                                break
                        if _pcg_on_ring:
                            break
                    if _pcg_on_ring:
                        break
                if (total_heavy > 0 and ring_size / total_heavy < 0.35
                        and not _pcg_on_ring):
                    pass  # Fall through to return None
                else:
                    # Check ring is non-aromatic. A NON-aromatic ring double/triple
                    # bond is now allowed through: _name_ring_as_parent_polyfunctional
                    # renders skeletal -ene/-yne for an all-carbon
                    # monocycle and fails closed (returns None) for anything it
                    # cannot render faithfully. Aromatic monocycles still have
                    # dedicated retained-name handlers and are excluded here.
                    is_aromatic = any(
                        mol.GetAtomWithIdx(idx).GetIsAromatic()
                        for idx in ring_atoms
                    )

                    if not is_aromatic:
                        ring_name = _name_ring_as_parent_polyfunctional(
                            features,
                        )
                        if ring_name:
                            return ring_name
        return None
    return _GO_ON


def principal_amides_off_chain(features) -> bool:
    """Several N-substituted amides (secondary or tertiary) are the principal group and the acyl carbon of one of them is
    not on the principal chain: the two amides of ``CNC(=O)CNCC(=O)NCC``, joined by an azanediyl group, with the chain
    on the one side. The chain-suffix machinery expresses the amides of the chain as the suffix and a prefix pair and no
    N-substituent of the amide that is the suffix, so it names another molecule ('2-{[2-(ethylamino)-2-oxoethyl]amino}-1-
    (methylamino)acetamide'), and a chain that holds no acyl carbon names it as 'aminopropanamide'. /:
    the amide is the principal group and its N-substituents are located by N, so the name is built from an acyl chain
    (the amide handler's, for one amide) or from the split of the decomposition engine, which names the amide of an acid
    and an amine."""
    pg = getattr(features, 'principal_group', None)
    if pg not in ('secondary_amide', 'tertiary_amide'):
        return False
    matches = (getattr(features, 'functional_groups', None) or {}).get(pg) or []
    chain = set(getattr(features, 'principal_chain', None) or ())
    return len(matches) > 1 and any(m[0] not in chain for m in matches)


def chain_fallback_declines(features) -> bool:
    """The chain fallback of ``assembly.handlers.general_acyclic`` names no molecule whose principal group is an ester (one
    substitutive name can only write the acid's anion:, the Blue Book) or N-substituted amides of
    which the chain holds not every acyl carbon (:func:`principal_amides_off_chain`: a chain with no acyl carbon is named
    'aminopropanamide'); the decomposition engine names both."""
    return getattr(features, 'principal_group', None) == 'ester' or principal_amides_off_chain(features)


def _route_amide_principal(features, mol, non_principal, principal_chain, principal_group):
    """An N-substituted amide as the principal group."""
    if principal_group in ('secondary_amide', 'tertiary_amide'):
        _amide_matches = features.functional_groups.get(principal_group, [])
        if len(_amide_matches) == 1:
            _match = list(_amide_matches[0])
            _acyl_c = next(
                (i for i in _match
                 if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                 and any(
                     nb.GetSymbol() == 'O'
                     and mol.GetBondBetweenAtoms(i, nb.GetIdx())
                            .GetBondTypeAsDouble() == 2.0
                     for nb in mol.GetAtomWithIdx(i).GetNeighbors())),
                None,
            )
            if _acyl_c is not None:
                # Fail closed on any DEFINED stereocentre/stereobond: the
                # delegated path emits no parent-level descriptors (the T3d
                # oxime-decline precedent).
                #
                # a phase (E3 Task 5 follow-on): `Chem.FindMolChiralCenters`
                # has an undocumented DESTRUCTIVE side effect — its internal
                # legacy re-perception pass clears every bond's `_CIPCode`
                # property (the E/Z label `rdCIPLabeler.AssignCIPLabels` /
                # the `centres` bridge already set during perception),
                # while leaving `Bond.GetStereo` itself untouched. Verified
                # in isolation: a bond reporting `(1,2,HasProp=1,'E')`
                # BEFORE the call reports `(1,2,HasProp=0,None)` straight
                # after, though STEREOE survives. Called on the SHARED
                # `mol` (not a copy), this permanently erases the E/Z
                # descriptors for every LATER consumer in this SAME naming
                # attempt — including this function's own final
                # `collect_stereodescriptors` call far below, which is why a
                # stereo-bearing on-chain amide (e.g. anandamide's
                # 5,8,11-trienoyl homologue) always emitted a name with
                # "0 R/S + N E/Z but name lacks descriptors" once this branch
                # ran, WITH OR WITHOUT the N-substituent fix. Call it on a
                # disposable COPY so the probe cannot clobber the shared
                # mol's already-assigned CIP labels; `_has_stereo` itself is
                # unaffected (chirality centres are copy-invariant).
                from rdkit import Chem as _Chem

                from .amides import (
                    get_n_substituents,
                )
                from .amides import (
                    name_amide as _rules_name_amide,
                )
                _has_stereo = bool(
                    _Chem.FindMolChiralCenters(
                        _Chem.Mol(mol), includeUnassigned=False)
                ) or any(
                    b.GetStereo() != _Chem.BondStereo.STEREONONE
                    for b in mol.GetBonds()
                )
                _n_subs = get_n_substituents(mol, tuple(_match))
                _n_atom = next(
                    (i for i in _match
                     if mol.GetAtomWithIdx(i).GetSymbol() == 'N'), None)
                # Coverage: every heavy N-neighbour besides the acyl C must
                # have produced a named substituent (get_n_substituents
                # silently omits unnameable ones), and every junior FG must
                # live wholly inside those N-substituent fragments.
                _claimed = set()
                for _s in _n_subs:
                    _claimed |= set(_s.get('atoms') or [])
                _expected_n_branches = 0
                if _n_atom is not None:
                    _expected_n_branches = sum(
                        1 for nb in mol.GetAtomWithIdx(_n_atom).GetNeighbors()
                        if nb.GetAtomicNum() > 1 and nb.GetIdx() != _acyl_c
                    )
                _fgs_contained = all(
                    set(_fm).issubset(_claimed)
                    for _fg_matches in (non_principal or {}).values()
                    for _fm in _fg_matches
                )
                _amide_guards_ok = (
                    not _has_stereo and _claimed
                    and len(_n_subs) == _expected_n_branches
                    and _fgs_contained
                )
                if _amide_guards_ok and _acyl_is_simple_saturated_chain(
                        mol, _acyl_c, _n_atom):
                    _nm = _rules_name_amide(mol, tuple(_match))
                    if _nm:
                        return _nm
                # A1 (deeper-amide, /: the count-based
                # rules.amides.name_amide above names ONLY a simple saturated
                # acyl. When the acyl is BRANCHED or UNSATURATED but ON the
                # principal chain (paracetamol's isobutyryl / acryloyl cousins:
                # CC(C)C(=O)Nc1ccc(O)cc1, C=CC(=O)Nc1ccc(O)cc1), route to the
                # acyl-AWARE composer namer _assemble_amide_name instead of
                # falling through. It names the acyl chain via the general chain
                # machinery (branch prefixes + prop-2-en/-yn unsaturation come
                # for free, the unified acid/amide suffix-swap) and the
                # N-substituent via
                # get_n_substituents, whose fragment already contains the junior
                # ring-FG scoped by disjoint atom ownership (no double-count —
                # verified). SAME coverage guards as the simple path (single
                # amide; no defined stereo; every N-branch named; every junior
                # FG contained in the N-substituent fragments). Previously this
                # fell through to the chain machinery, which mis-rooted the
                # N-aryl and abstained. Prefix ORDER follows the existing uniform
                # amide convention; the / N/C merge-and-order fix
                # is a separate conformance lever tracked by
                # test_amide_merged_prefix_p16_3_3::test_chain_amide_merges_too.
                if _amide_guards_ok and _acyl_c in set(principal_chain):
                    from ..assembly.composer import (
                        _assemble_amide_name as _composer_name_amide,
                    )
                    from ..assembly.composer import (
                        _generate_prefixes as _gen_prefixes_am,
                    )
                    # a review BLOCKER guard : _assemble_amide_name appends
                    # _generate_prefixes(features) at PARENT level. That set is
                    # honest for a LOCATED acyl-chain substituent (a '2-methyl'
                    # branch, locant 2) but RE-EXPRESSES a junior FG that lives
                    # inside an N-substituent yet cannot be placed on the acyl
                    # parent — it comes back UNLOCATED ('cyano', 'dihydroxy'),
                    # double-counting it: C=CC(=O)NCC#N ->
                    # 'N-(cyanomethyl)cyanoprop-2-enamide',
                    # C=CC(=O)N(CCO)CCO -> 'N,N-bis(2-hydroxyethyl)dihydroxyprop-2-enamide'
                    # (parseable WRONG molecules gate-OFF). The acyl carbon is C1,
                    # so EVERY legitimate acyl-chain substituent carries a locant;
                    # an unlocated prefix is the double-count signature. Delegate
                    # only when every generated prefix is located, else fall
                    # through (fail closed — restores the pre-change behaviour).
                    _has_unlocated_prefix = any(
                        (not getattr(_p, 'locants', None))
                        and not str(getattr(_p, 'text', ''))[:1].isdigit()
                        for _p in _gen_prefixes_am(features)
                    )
                    if not _has_unlocated_prefix:
                        _nm2 = _composer_name_amide(features, "pin")
                        if _nm2 and _nm2 != "amide":
                            return _nm2
                # (A ring-attached anilide — benzoyl / cyclohexanecarbonyl on an
                # N-aryl bearing a junior FG — never reaches here: its acyl carbon
                # is not on the principal chain and its principal_chain is empty,
                # so it returns None at the top guard and is named by the lever-C
                # intercept in handlers/tier_a_ring.py via _name_ring_attached_anilide.)
                # OFF-chain acyl (N-side is the chain): the chain-suffix
                # machinery below is structurally wrong for this shape (double-
                # expresses the amide) -> fail closed. ON-chain acyl (the
                # ordinary N-aryl / N-alkyl amide, e.g. paracetamol
                # CC(=O)Nc1ccc(O)cc1 whose junior phenol lives on the N-aryl):
                # name_amide is the correct producer and was tried above; if its
                # guards did not hold, fall through to the chain machinery (which
                # correctly handles the cases the delegation does not cover).
                # Previously the acyl-on-chain N-aryl amide fell straight through
                # and the chain path mis-rooted the N-aryl as a C1 substituent
                # ('1-(4-hydroxyanilino)ethanamide', OPSIN-unparseable -> abstain).
                if _acyl_c not in set(principal_chain):
                    return None
        elif principal_amides_off_chain(features):
            # the chain-suffix machinery below cannot name them (see ``principal_amides_off_chain``): fail closed, as
            # for the off-chain acyl above, so that it is not offered to a probe
            return None
    return _GO_ON


def _route_ester_first(features, mol, principal_group):
    """An ester principal group: the first of its two early routes."""
    if principal_group == "ester":
        _ester_matches_all = features.functional_groups.get("ester", [])
        _halide_fgs = ("acid_chloride", "acid_bromide",
                       "acid_fluoride", "acid_iodide")
        if (len(_ester_matches_all) == 1
                and any(features.functional_groups.get(h)
                        for h in _halide_fgs)):
            from ..rules.acid_halides import (
                HALIDE_WORDS,
                name_carbonic_monoester_acyl_halide,
            )
            _em = _ester_matches_all[0]
            _acyl_c = _em[0]
            _c_at = mol.GetAtomWithIdx(_acyl_c)
            _hal = next((nb for nb in _c_at.GetNeighbors()
                         if nb.GetSymbol() in ('Cl', 'Br', 'F', 'I')), None)
            _o_dbl = next((nb.GetIdx() for nb in _c_at.GetNeighbors()
                           if nb.GetSymbol() == 'O'
                           and mol.GetBondBetweenAtoms(
                               _acyl_c, nb.GetIdx()).GetBondTypeAsDouble() == 2.0),
                          None)
            if _hal is not None and _o_dbl is not None:
                _hal_pg = {'Cl': 'acid_chloride', 'Br': 'acid_bromide',
                           'F': 'acid_fluoride', 'I': 'acid_iodide'}[
                    _hal.GetSymbol()]
                _cc = name_carbonic_monoester_acyl_halide(
                    mol, (_acyl_c, _o_dbl, _hal.GetIdx()), HALIDE_WORDS[_hal_pg]
                )
                if _cc is not None:
                    return _cc
    return _GO_ON


def _route_ester_second(features, mol, principal_chain, principal_group):
    """An ester principal group: the second of its two early routes."""
    if principal_group == "ester":
        _ester_matches_all = features.functional_groups.get("ester", [])
        if len(_ester_matches_all) >= 2:
            # "When anions are identical functional class multiplicative nomenclature is used"
            # (the Blue Book): the ester groups of one alcoholic component, named as one multivalent organyl group
            # in front of the multiplied anion ('oxydi(ethane-2,1-diyl) diacetate', '2-oxopropane-1,3-diyl diacetate');
            # the groups beside the esters are prefixes of the organyl group. Declines (None) for any other polyester.
            from ..rules.esters import _try_functional_class_diol_diester
            _polyol_pin = _try_functional_class_diol_diester(mol, _ester_matches_all)
            if _polyol_pin:
                return _polyol_pin
        if len(_ester_matches_all) == 1:
            from ..rules.esters import name_polyfunctional_ester_via_acid
            _ester_pin = name_polyfunctional_ester_via_acid(
                mol, _ester_matches_all[0]
            )
            if _ester_pin:
                return _ester_pin
        elif len(_ester_matches_all) == 2:
            # a phase (diacylglycerol-shape fix, PHASE6-DIESTER-OL-
            # SENIORITY): + -- a partially-esterified
            # polyol carrying TWO different noncyclic esters plus a free
            # hydroxyl. The senior acid (longer principal chain) stays the
            # functional-class parent; the other ester becomes an acyloxy
            # prefix and the free -OH a hydroxy prefix, both on the 'yl'
            # word -- NOT the demotion below, which wrongly promotes
            # the junior hydroxy class to principal (inverting: ester
            # class 9 outranks hydroxy class 17). Fail-closed: declines
            # (None) for anything outside its narrow scope (a tied acid
            # length, a ring acid, an off-chain decoration,...), falling
            # through to unchanged.
            from ..rules.esters import name_polyfunctional_diester_free_hydroxy
            _diester_pin = name_polyfunctional_diester_free_hydroxy(
                mol, _ester_matches_all, principal_chain,
            )
            if _diester_pin:
                return _diester_pin
    return _GO_ON


def _group_on_chain_parent(chain_set, features, fg_name, matches, mol, ring_fg_groups):
    """A chain parent: keep the matches of the group on the chain."""
    if getattr(features, 'chain_is_parent', False):
        ring_atom_set = set()
        for rg in getattr(features, 'ring_substituents_as_groups', []):
            ring_atom_set.update(rg)

        chain_matches = []
        for match in matches:
            # Check if this FG's center atom is on the chain
            center = _find_fg_center_atom(mol, match, fg_name)
            if center is not None and center in chain_set and center not in ring_atom_set:
                chain_matches.append(match)
            elif any(a in chain_set and a not in ring_atom_set for a in _fg_located_atoms(mol, match)):
                # The group's own atoms decide where it sits (``_fg_located_atoms``): a tertiary-alcohol match also holds
                # the carbinol carbon's three neighbours, one of which can be the chain atom a ring branch hangs from,
                # so testing every atom kept the ring's OH and cited it a second time on the chain ('(2E)-3-(1,4-dihydroxy-
                # 6-oxocyclohex-2-en-1-yl)-3-hydroxyprop-2-enoic acid' for the acid that has no 3-hydroxy group).
                chain_matches.append(match)
            else:
                # FG is on a ring - track for ring substituent naming
                ring_fg_groups[fg_name].append(match)
        matches = chain_matches
        if not matches:
            return True, matches
    return False, matches


def _group_in_named_branch(_branch_atoms, features, fg_name, matches, mol, non_principal):
    """A group of a branch that the branch prefix names."""
    if _branch_atoms and fg_name in BRANCH_HANDLED_FGS:
        filtered_matches = []
        for match in matches:
            # The group's own atoms decide where it sits: a secondary-alcohol
            # match also holds the carbinol carbon's two neighbours, one of
            # which is the chain atom the branch hangs from, so testing every
            # atom kept the branch's OH and cited it a second time on the
            # chain ('3-bromo-2-(2-bromo-1-hydroxyethyl)-2,4-dihydroxybutanoic
            # acid' for the PIN '3-bromo-2-(2-bromo-1-hydroxyethyl)-4-hydroxy-
            # butanoic acid (PIN)',, the Blue Book).
            _located = _fg_located_atoms(mol, match)
            if not all(a in _branch_atoms for a in _located):
                filtered_matches.append(match)
                continue
            # Check if this FG is on a small branch with carbons
            # (i.e., a branch where _name_heteroatom_substituent handles the FG)
            on_small_branch = False
            for _pos, sub_list in features.substituents.items():
                for sub_atoms in sub_list:
                    sub_set = set(sub_atoms)
                    if all(a in sub_set for a in _located):
                        # This branch contains the entire FG match
                        c_count = sum(1 for a in sub_atoms
                                      if mol.GetAtomWithIdx(a).GetSymbol() == 'C')
                        if 1 <= c_count <= 3:
                            on_small_branch = True
                            break
                if on_small_branch:
                    break
            if not on_small_branch:
                filtered_matches.append(match)
        original_count = len(non_principal[fg_name])
        matches = filtered_matches
        if not matches:
            if original_count > 0:
                logger.debug(
                    "substituent_all_candidates_filtered polyfunc_bugb: fg_name=%s filtered=%d "
                    "(substituent naming handles these on small branches)",
                    fg_name, original_count,
                )
            return True, matches
    return False, matches


def _group_primary_amine(_N_SUBSTITUTED_AMINE_FGS, features, fg_name, matches, mol):
    """A primary or aromatic amine group."""
    if fg_name in ({'primary_amine', 'aromatic_amine'}
                   | _N_SUBSTITUTED_AMINE_FGS):
        _amide_n_atoms = set()
        for _ak in ('primary_amide', 'secondary_amide', 'tertiary_amide',
                    'thioamide', 'selenoamide', 'telluroamide'):
            for _am in features.functional_groups.get(_ak, ()):
                _amide_n_atoms.update(
                    a for a in _am
                    if mol.GetAtomWithIdx(a).GetSymbol() == 'N')
        if _amide_n_atoms:
            matches = [
                _m for _m in matches
                if not any(a in _amide_n_atoms for a in _m
                           if mol.GetAtomWithIdx(a).GetSymbol() == 'N')]
            if not matches:
                return True, matches
    return False, matches


def _group_substituted_amine(_N_SUBSTITUTED_AMINE_FGS, features, fg_name, matches, mol):
    """A secondary or tertiary amine group beside substituents."""
    if fg_name in _N_SUBSTITUTED_AMINE_FGS and features.substituents:
        _amine_kept = []
        for _match in matches:
            _n_owned_by_branch = False
            for _n in (a for a in _match
                       if mol.GetAtomWithIdx(a).GetSymbol() == 'N'):
                for _sub_list in features.substituents.values():
                    if any(_n in _sub_atoms for _sub_atoms in _sub_list):
                        _n_owned_by_branch = True
                        break
                if _n_owned_by_branch:
                    break
            if not _n_owned_by_branch:
                _amine_kept.append(_match)
        if not _amine_kept:
            logger.debug(
                "suppress_duplicate_bare_amino_prefix polyfunc: skip bare 'amino' FG-prefix for %s — "
                "N-substituted amine named in full by the substituent walk",
                fg_name,
            )
            return True, matches
        matches = _amine_kept
    return False, matches


def _group_acyl_halide(_ACYL_HALIDE_HALO, all_prefixes, atom_to_locant, chain_set, fg_name, matches):
    """An acyl halide on the chain."""
    if fg_name in _ACYL_HALIDE_HALO and chain_set:
        _ah_locs = []
        for _m in matches:
            _c = _m[0]  # acyl-halide carbonyl carbon (SMARTS index 0)
            if _c in chain_set:
                _loc = atom_to_locant.get(_c)
                if _loc is not None:
                    _ah_locs.append(_loc)
        if _ah_locs and len(_ah_locs) == len(matches):
            _ah_locs.sort()
            _n = len(_ah_locs)
            all_prefixes.append(format_fg_prefix('oxo', _ah_locs, _n))
            all_prefixes.append(
                format_fg_prefix(_ACYL_HALIDE_HALO[fg_name], _ah_locs, _n)
            )
            return True
    return False


def _group_primary_amide(_amidine_excluded_n, all_prefixes, atom_to_locant, chain_set, fg_name, matches):
    """A primary amide on the chain."""
    if fg_name == 'primary_amide' and chain_set:
        _pa_locs = []
        _pa_ns = []
        for _m in matches:
            _pac = _m[0]
            if _pac in chain_set:
                _loc = atom_to_locant.get(_pac)
                if _loc is not None:
                    _pa_locs.append(_loc)
                    _pa_ns.append(_m[2])
        if _pa_locs and len(_pa_locs) == len(matches):
            _pa_locs.sort()
            _n = len(_pa_locs)
            all_prefixes.append(format_fg_prefix('oxo', _pa_locs, _n))
            all_prefixes.append(format_fg_prefix('amino', _pa_locs, _n))
            _amidine_excluded_n.update(_pa_ns)
            return True
    return False


def _group_peroxy_acid(_amidine_excluded_n, all_prefixes, atom_to_locant, chain_set, fg_name, matches):
    """A peroxy acid on the chain."""
    if fg_name == 'peroxy_acid' and chain_set:
        _px_locs = []
        _px_os = []
        for _m in matches:
            _pxc = _m[0]
            if _pxc in chain_set:
                _loc = atom_to_locant.get(_pxc)
                if _loc is not None:
                    _px_locs.append(_loc)
                    _px_os.extend([_m[1], _m[2], _m[3]])
        if _px_locs and len(_px_locs) == len(matches):
            _px_locs.sort()
            _n = len(_px_locs)
            all_prefixes.append(format_fg_prefix('oxo', _px_locs, _n))
            all_prefixes.append(
                format_fg_prefix('hydroperoxy', _px_locs, _n))
            _amidine_excluded_n.update(_px_os)
            return True
    return False


def _group_chain_end_acid_derivative(_amidine_excluded_n, all_prefixes, atom_to_locant, chain_set, fg_name, matches, mol, principal_group):
    """A chain-end acid derivative."""
    if fg_name in _CHAIN_END_ACID_DERIVATIVE_FGS and chain_set:
        _ce = _chain_end_acid_derivative_prefixes(
            mol, fg_name, matches, chain_set, atom_to_locant,
            principal_group)
        if _ce is not None:
            _ce_groups, _ce_atoms = _ce
            for _pf, _locs in _ce_groups.items():
                _locs = sorted(_locs)
                all_prefixes.append(
                    format_fg_prefix(_pf, _locs, len(_locs)))
            _amidine_excluded_n.update(_ce_atoms)
            return True
    return False


def _acyl_side_has_carboxy(mol, acyl_c: int, n_idx: int) -> bool:
    """True when the acyl side of -NH-CO-R (from ``acyl_c``, away from ``n_idx``) holds a -COOH: the walk spells such an acyl
    '(3-carboxy-1-oxopropyl)amino', not the '3-carboxypropanamido' of method (1) (the Blue Book), so the
    amide arm keeps the older route that spells it."""
    from ..assembly.decorated_n_branch import _is_carboxy_carbon
    seen, stack = {acyl_c, n_idx}, [nb.GetIdx() for nb in mol.GetAtomWithIdx(acyl_c).GetNeighbors() if nb.GetIdx() != n_idx]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        if _is_carboxy_carbon(mol.GetAtomWithIdx(i)):
            return True
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors())
    return False


def _group_secondary_amide(_amidine_excluded_n, all_prefixes, atom_to_locant, chain_set, fg_name, matches, mol):
    """A secondary amide on the chain."""
    if fg_name == 'secondary_amide' and chain_set:
        from ..assembly.composer import _name_r_group
        _sa_ok = True
        _sa_units = {}  # amide C -> (amino-side prefix, N-side branch atoms)
        for _m in matches:
            _sac, _san = _m[0], _m[2]
            if (_sac not in chain_set and any(nb.GetIdx() in chain_set for nb in mol.GetAtomWithIdx(_san).GetNeighbors())
                    and not _acyl_side_has_carboxy(mol, _sac, _san)):
                continue    # a reversed amide -NH-CO-R on the chain: the substituent walk names it ('2-aminopropanamido')
            if _sac not in chain_set:
                _sa_ok = False
                break
            _r_atoms = [
                nb.GetIdx()
                for nb in mol.GetAtomWithIdx(_san).GetNeighbors()
                if nb.GetIdx() != _sac and nb.GetAtomicNum() > 1]
            if len(_r_atoms) != 1:
                _sa_ok = False
                break
            _r_name = _name_r_group(mol, _r_atoms[0], exclude_atoms={_san})
            if not _r_name:
                _sa_ok = False
                break
            # (the Blue Book): 'anilino' is the retained PREFERRED
            # PREFIX for C6H5-NH- with FULL SUBSTITUTION ALLOWED, so a
            # substituted ring substitutes ONTO 'anilino' (the Blue Book
            # '4-chloroanilino (preferred prefix) | (4-chlorophenyl)amino').
            # The old else-branch put the general-nomenclature form in a PIN
            # ('5-(4-chlorophenylamino)-5-oxopentanoic acid' shipped) — and not
            # even the well-formed general spelling, which the Blue Book writes with
            # the inner enclosure. Bare 'anilino' stays bare (the Blue Book).
            # Non-phenyl rings keep the legacy '({R}amino)'.
            from .ring_substituents import anilino_preferred_prefix
            _amino_prefix = anilino_preferred_prefix(_r_name)
            if _amino_prefix is None:
                # /: an N-substituent whose name carries
                # locants takes its own marks inside the '(...amino)' unit,
                # one level in: '2-[di(butan-2-yl)amino]butan-2-ol (PIN)'
                # (the Blue Book). The bare glue shipped
                # '3-(propan-2-ylamino)-3-oxo...' at pin_verified.
                from ..assembly.naming_utils import (
                    apply_enclosing_marks as _enc_marks,
                    needs_brackets as _needs_br,
                )
                _r_inner = (_enc_marks(_r_name, -1)
                            if _needs_br(_r_name) else _r_name)
                _amino_prefix = _enc_marks(f'{_r_inner}amino', -1)
            # Collect the whole N-side branch (N + R subtree), never crossing
            # the amide C, to exclude it from the alkyl substituent walk.
            _seen = {_sac}
            _stack = [_san]
            _branch = []
            while _stack:
                _a = _stack.pop()
                if _a in _seen:
                    continue
                _seen.add(_a)
                _branch.append(_a)
                for _nb in mol.GetAtomWithIdx(_a).GetNeighbors():
                    if _nb.GetIdx() not in _seen:
                        _stack.append(_nb.GetIdx())
            _sa_units[_sac] = (_amino_prefix, _branch)
        if _sa_ok and _sa_units:
            _sa_prefixes = []
            for _sac, (_amino_prefix, _branch) in _sa_units.items():
                _loc = atom_to_locant.get(_sac)
                if _loc is None:
                    _sa_ok = False
                    break
                _sa_prefixes.append(format_fg_prefix('oxo', [_loc], 1))
                _sa_prefixes.append(
                    format_fg_prefix(_amino_prefix, [_loc], 1))
            if _sa_ok:
                all_prefixes.extend(_sa_prefixes)
                for _sac, (_amino_prefix, _branch) in _sa_units.items():
                    _amidine_excluded_n.update(_branch)
                return True
    return False


def _group_tertiary_amide(_amidine_excluded_n, all_prefixes, atom_to_locant, chain_set, fg_name, matches, mol):
    """A tertiary amide on the chain."""
    if fg_name == 'tertiary_amide' and chain_set:
        # (local aliases only: a bare 'name_substituent' import here would make
        # that name local to the whole function and unbind its other uses)
        from ..assembly.substituent_enumerator import (
            name_substituent as _ta_name_substituent,
        )
        from ..errors import is_refusal_sentinel as _ta_is_refusal
        from ..assembly.naming_utils import (
            apply_enclosing_marks as _ta_enc_marks,
            needs_brackets as _ta_needs_br,
        )
        _ta_prefixes = []
        _ta_branches = []
        for _m in matches:
            _tac, _tan = _m[0], _m[2]
            _loc = atom_to_locant.get(_tac) if _tac in chain_set else None
            if _loc is None or mol.GetAtomWithIdx(_tan).GetTotalNumHs():
                # an N-H amide also matches this SMARTS; it is the
                # secondary block's (or no) business
                _ta_prefixes = []
                break
            _seen = {_tac}
            _stack = [_tan]
            _branch = []
            while _stack:
                _a = _stack.pop()
                if _a in _seen:
                    continue
                _seen.add(_a)
                _branch.append(_a)
                for _nb in mol.GetAtomWithIdx(_a).GetNeighbors():
                    if _nb.GetIdx() not in _seen and _nb.GetAtomicNum() > 1:
                        _stack.append(_nb.GetIdx())
            if set(_branch) & chain_set:
                _ta_prefixes = []  # the N side reaches back into the parent
                break
            _n_name = _ta_name_substituent(mol, sorted(_branch), _tan)
            if (not _n_name or _n_name == 'substituent' or ' ' in _n_name
                    or _ta_is_refusal(_n_name)):
                _ta_prefixes = []
                break
            if _ta_needs_br(_n_name):
                _n_name = _ta_enc_marks(_n_name, -1)
            _ta_prefixes.append(format_fg_prefix('oxo', [_loc], 1))
            _ta_prefixes.append(format_fg_prefix(_n_name, [_loc], 1))
            _ta_branches.append(_branch)
        if _ta_prefixes:
            all_prefixes.extend(_ta_prefixes)
            for _branch in _ta_branches:
                _amidine_excluded_n.update(_branch)
            return True
    return False


def _group_sulfonamide_n(_SULFONAMIDE_N_ATTACH_FGS, chain_set, fg_name, matches, mol):
    """A sulfonamide joined through its nitrogen."""
    if fg_name in _SULFONAMIDE_N_ATTACH_FGS and chain_set:
        _s_on_chain = False
        for _m in matches:
            for _a in _m:
                if mol.GetAtomWithIdx(_a).GetSymbol() != 'S':
                    continue
                if any(nb.GetIdx() in chain_set
                       for nb in mol.GetAtomWithIdx(_a).GetNeighbors()):
                    _s_on_chain = True
                break
            if _s_on_chain:
                break
        if not _s_on_chain:
            return True
    return False


def _group_thioamide(_amidine_excluded_n, all_prefixes, atom_to_locant, chain_set, fg_name, matches, mol):
    """A thioamide on the chain."""
    if fg_name == 'thioamide' and chain_set:
        _ta_ok = True
        _ta_units = {}  # thioamide C -> (S idx, N idx)
        for _m in matches:
            _tac, _tas, _tan = _m[0], _m[1], _m[2]
            if _tac not in chain_set:
                _ta_ok = False
                break
            _ta_units[_tac] = (_tas, _tan)
        if _ta_ok and _ta_units:
            _ta_prefixes = []
            for _tac, (_tas, _tan) in _ta_units.items():
                _loc = atom_to_locant.get(_tac)
                _amino = _name_amidine_chain_side(mol, _tan, _tac, 'amino')
                if _loc is None or _amino is None:
                    _ta_ok = False
                    break
                _ta_prefixes.append(format_fg_prefix(_amino, [_loc], 1))
                _ta_prefixes.append(
                    format_fg_prefix('sulfanylidene', [_loc], 1))
            if _ta_ok:
                all_prefixes.extend(_ta_prefixes)
                for _tac, (_tas, _tan) in _ta_units.items():
                    _amidine_excluded_n.add(_tan)
                return True
    return False


def _group_amidine(_amidine_excluded_n, all_prefixes, atom_to_locant, chain_set, fg_name, matches, mol):
    """An amidine on the chain."""
    if fg_name == 'amidine' and chain_set:
        _seen_amc = {}
        _am_ok = True
        for _m in matches:
            _amc = _dbl = _sgl = None
            for _a in _m:
                _at = mol.GetAtomWithIdx(_a)
                if _at.GetSymbol() != 'C':
                    continue
                _dN = [
                    nb.GetIdx() for nb in _at.GetNeighbors()
                    if nb.GetSymbol() == 'N'
                    and mol.GetBondBetweenAtoms(
                        _a, nb.GetIdx()).GetBondTypeAsDouble() == 2.0
                ]
                _sN = [
                    nb.GetIdx() for nb in _at.GetNeighbors()
                    if nb.GetSymbol() == 'N'
                    and mol.GetBondBetweenAtoms(
                        _a, nb.GetIdx()).GetBondTypeAsDouble() == 1.0
                ]
                if len(_dN) == 1 and len(_sN) == 1:
                    _amc, _dbl, _sgl = _a, _dN[0], _sN[0]
                    break
            if _amc is None or _amc not in chain_set:
                _am_ok = False
                break
            _seen_amc[_amc] = (_dbl, _sgl)
        if _am_ok and _seen_amc:
            _am_prefixes = []
            _am_hydrazinyl_tails = {}
            for _amc, (_dbl, _sgl) in _seen_amc.items():
                _loc = atom_to_locant.get(_amc)
                # (BB 34490, plan P1AM Task 6): when the
                # single-bonded N's sole heavy branch is a terminal NH2,
                # the -NH-NH2 unit is ONE 'hydrazinyl' prefix (PIN
                # '3-hydrazinyl-3-iminopropanoic acid'), never the
                # fragmenting '(aminoamino)'.
                _sgl_branches = [
                    nb.GetIdx()
                    for nb in mol.GetAtomWithIdx(_sgl).GetNeighbors()
                    if nb.GetIdx() != _amc and nb.GetAtomicNum() > 1]
                _hydrazinyl_tail = None
                if (len(_sgl_branches) == 1
                        and mol.GetAtomWithIdx(_sgl_branches[0]).GetSymbol() == 'N'
                        and mol.GetAtomWithIdx(_sgl_branches[0]).GetDegree() == 1):
                    _amino = 'hydrazinyl'
                    _hydrazinyl_tail = _sgl_branches[0]
                else:
                    _amino = _name_amidine_chain_side(mol, _sgl, _amc, 'amino')
                _imino = _name_amidine_chain_side(mol, _dbl, _amc, 'imino')
                if _loc is None or _amino is None or _imino is None:
                    _am_ok = False
                    break
                _am_prefixes.append(format_fg_prefix(_amino, [_loc], 1))
                _am_prefixes.append(format_fg_prefix(_imino, [_loc], 1))
                if _hydrazinyl_tail is not None:
                    _am_hydrazinyl_tails[_amc] = _hydrazinyl_tail
            if _am_ok:
                all_prefixes.extend(_am_prefixes)
                for _amc, (_dbl, _sgl) in _seen_amc.items():
                    _amidine_excluded_n.add(_dbl)
                    _amidine_excluded_n.add(_sgl)
                    if _amc in _am_hydrazinyl_tails:
                        _amidine_excluded_n.add(_am_hydrazinyl_tails[_amc])
                return True
    return False


def _group_hydrazonamide(_amidine_excluded_n, all_prefixes, atom_to_locant, chain_set, fg_name, matches, mol):
    """A hydrazonamide on the chain."""
    if fg_name == 'hydrazonamide' and chain_set:
        _hz_ok = True
        _hz_units = {}
        for _m in matches:
            _hc = _m[0]  # SMARTS [CX3](=[NX2][NX2,NX3])[NX3]: match[0]=C
            if _hc not in chain_set:
                _hz_ok = False
                break
            _at = mol.GetAtomWithIdx(_hc)
            _dbl = next((nb.GetIdx() for nb in _at.GetNeighbors()
                         if nb.GetSymbol() == 'N'
                         and mol.GetBondBetweenAtoms(
                             _hc, nb.GetIdx()).GetBondTypeAsDouble() == 2.0),
                        None)
            _sgl = next((nb.GetIdx() for nb in _at.GetNeighbors()
                         if nb.GetSymbol() == 'N'
                         and mol.GetBondBetweenAtoms(
                             _hc, nb.GetIdx()).GetBondTypeAsDouble() == 1.0),
                        None)
            if _dbl is None or _sgl is None:
                _hz_ok = False
                break
            # =N-NH2 arm must be exactly one terminal NH2 (unsubstituted)
            _dbl_branches = [nb.GetIdx()
                             for nb in mol.GetAtomWithIdx(_dbl).GetNeighbors()
                             if nb.GetIdx() != _hc and nb.GetAtomicNum() > 1]
            if (len(_dbl_branches) != 1
                    or mol.GetAtomWithIdx(_dbl_branches[0]).GetSymbol() != 'N'
                    or mol.GetAtomWithIdx(_dbl_branches[0]).GetDegree() != 1):
                _hz_ok = False
                break
            _hz_units[_hc] = (_dbl, _dbl_branches[0], _sgl)
        if _hz_ok and _hz_units:
            _hz_prefixes = []
            for _hc, (_dbl, _nh2, _sgl) in _hz_units.items():
                _loc = atom_to_locant.get(_hc)
                _amino = _name_amidine_chain_side(mol, _sgl, _hc, 'amino')
                if _loc is None or _amino is None:
                    _hz_ok = False
                    break
                _hz_prefixes.append(format_fg_prefix(_amino, [_loc], 1))
                _hz_prefixes.append(
                    format_fg_prefix('hydrazinylidene', [_loc], 1))
            if _hz_ok:
                all_prefixes.extend(_hz_prefixes)
                for _hc, (_dbl, _nh2, _sgl) in _hz_units.items():
                    _amidine_excluded_n.update({_dbl, _nh2, _sgl})
                return True
    return False


def _group_imidohydrazide(_amidine_excluded_n, all_prefixes, atom_to_locant, chain_set, fg_name, matches, mol):
    """An imidohydrazide on the chain."""
    if fg_name == 'imidohydrazide' and chain_set:
        _ih_ok = True
        _ih_units = {}
        for _m in matches:
            if len(_m) < 4:
                _ih_ok = False
                break
            _ihc, _ih_dbl, _ih_sgl, _ih_nh2 = _m[0], _m[1], _m[2], _m[3]
            if _ihc not in chain_set:
                _ih_ok = False
                break
            # -NH-NH2 arm must terminate in a degree-1 NH2
            if mol.GetAtomWithIdx(_ih_nh2).GetDegree() != 1:
                _ih_ok = False
                break
            _ih_units[_ihc] = (_ih_dbl, _ih_sgl, _ih_nh2)
        if _ih_ok and _ih_units:
            _ih_prefixes = []
            for _ihc, (_ih_dbl, _ih_sgl, _ih_nh2) in _ih_units.items():
                _loc = atom_to_locant.get(_ihc)
                _imino = _name_amidine_chain_side(mol, _ih_dbl, _ihc, 'imino')
                if _loc is None or _imino is None:
                    _ih_ok = False
                    break
                _ih_prefixes.append(format_fg_prefix('hydrazinyl', [_loc], 1))
                _ih_prefixes.append(format_fg_prefix(_imino, [_loc], 1))
            if _ih_ok:
                all_prefixes.extend(_ih_prefixes)
                for _ihc, (_ih_dbl, _ih_sgl, _ih_nh2) in _ih_units.items():
                    _amidine_excluded_n.update({_ih_dbl, _ih_sgl, _ih_nh2})
                return True
    return False


def _group_hydrazonic_acid(_amidine_excluded_n, all_prefixes, atom_to_locant, chain_set, fg_name, matches):
    """A hydrazonic acid on the chain."""
    if fg_name == 'hydrazonic_acid' and chain_set:
        _hzc_ok = True
        _hzc_units = {}
        for _m in matches:
            if len(_m) < 4:
                _hzc_ok = False
                break
            _c, _nim, _nam, _o = _m[0], _m[1], _m[2], _m[3]
            if _c not in chain_set:
                _hzc_ok = False
                break
            _hzc_units[_c] = (_nim, _nam, _o)
        if _hzc_ok and _hzc_units:
            _hzc_prefixes = []
            for _c, (_nim, _nam, _o) in _hzc_units.items():
                _loc = atom_to_locant.get(_c)
                if _loc is None:
                    _hzc_ok = False
                    break
                _hzc_prefixes.append(format_fg_prefix('hydroxy', [_loc], 1))
                _hzc_prefixes.append(
                    format_fg_prefix('hydrazinylidene', [_loc], 1))
            if _hzc_ok:
                all_prefixes.extend(_hzc_prefixes)
                for _c, (_nim, _nam, _o) in _hzc_units.items():
                    _amidine_excluded_n.update({_nim, _nam, _o})
                return True
    return False


def _group_without_prefix_form(_omit_mononuclear_locants, all_prefixes, atom_to_locant, chain_set, features, fg_name, matches, mol, prefix_form, principal_chain, principal_group):
    """A group with no prefix form of its own."""
    if not prefix_form:
        # a phase tier-3 fallback : when a composite loser FG
        # has no clean strict-IUPAC prefix (the substituent_no_prefix_form case), decompose it into
        # its ordered sub-group prefix components instead of dropping it. Gated
        # behind the default-OFF flag (Stage A byte-identical); the split only
        # fires for table-listed composites (ester/thioester/iminoester) and is
        # OPSIN-RT gated (FAIL-CLOSED,). Lazy import (Pattern-S3) avoids a
        # cycle.: iminoester -> imino + alkoxy (2), the chain-
        # end simple-prefix form '4-imino-4-methoxybutanoic acid').
        components = None
        _anchored = []
        _split_default = (
            principal_group in _SPLIT_DEFAULT_PGS
            and fg_name in ("ester", "thioester", "iminoester")
        )
        if getattr(features, "_enable_group_splitting", False) or _split_default:
            from ..assembly.group_splitting import (
                _get_default_oracle,
                split_composite_fg,
            )
            # W2F-P2: split ONLY the match whose carbonyl C
            # is IN the principal chain (in-chain ester C -> oxo + R-oxy /
            # R-sulfanyl); acyloxy-orientation matches (carbonyl off-chain)
            # keep today's continue (composer acyloxy path owns them). The
            # exactly-one guard keeps multi-(thio)ester acids fail-closed
            # (v1) and the locant computation single-valued (fixes the
            # brief item-6 '9,9' defect: locants from anchored ONLY).
            _anchored = [m for m in matches if m and m[0] in chain_set]
            if len(_anchored) == 1:
                _oracle = (
                    getattr(features, "_split_oracle", None)
                    or _get_default_oracle()
                )
                components = split_composite_fg(
                    fg_name, mol, _anchored[0], principal_chain,
                    oracle=_oracle,
                )
        if not components:
            # By-design: FGs using functional class naming (ester→alkoxycarbonyl,
            # secondary_amide→acylamino, thioether, etc.) are handled by
            # specialized naming paths, not as simple prefixes. Unchanged drop
            # path for the OFF / non-splittable / RT-rejected cases.
            logger.debug(
                "substituent_no_prefix_form substituent_skip: reason=no_fg_prefix_form fg_name=%s",
                fg_name,
            )
            return True
        # native: each split component re-enters the existing prefix
        # pipeline (format_fg_prefix + alpha_sort_key) sharing the central-carbon
        # locant (comp.locants is None -> the caller's locants apply to both).
        split_locants = get_non_principal_fg_locants(
            mol, _anchored, principal_chain, atom_to_locant, fg_name
        )
        for comp in components:
            _comp_locants = comp.locants or split_locants
            all_prefixes.append(
                format_fg_prefix(
                    comp.prefix_form,
                    [] if _omit_mononuclear_locants else _comp_locants,
                    comp.count)
            )
        return True
    return False


def _route_amine_on_one_nitrogen(_principal_amine_n_atoms, all_prefixes, mol, principal_chain):
    """The principal amine groups on one nitrogen."""
    if len(_principal_amine_n_atoms) == 1:
        from collections import Counter as _NCounter

        from ..assembly.composer import _name_r_group, _wrap_n_substituent

        _amine_n_atom = next(iter(_principal_amine_n_atoms))
        _chain_set_amine = set(principal_chain)
        _n_sub_names: List[str] = []
        for _pnb in mol.GetAtomWithIdx(_amine_n_atom).GetNeighbors():
            _pni = _pnb.GetIdx()
            # Skip the parent-chain attachment carbon and hydrogens;
            # the remaining heavy neighbours are the N-substituents.
            if _pni in _chain_set_amine or _pnb.GetAtomicNum() <= 1:
                continue
            _rn = _name_r_group(mol, _pni, exclude_atoms={_amine_n_atom})
            if not _rn:
                # Fail closed. This block is the ONLY producer of the principal
                # amine's N-substituent prefixes -- the branch is deliberately
                # excluded from the substituent walk a few lines above -- so
                # skipping an un-nameable one does not degrade the name, it
                # DELETES atoms from it. Measured on
                # 'CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)OCC(COP(=O)(O)OCCNC)'
                # 'OC(=O)CCCCCCCCCCCCCCCCC': the phosphate-bearing arm returned
                # None here and the emitted name was
                # '1-((11Z,14Z)-icosa-11,14-dienoyloxy)-N-methyl-2-'
                # '(stearoyloxy)propanamine' -- a molecule with no phosphate at
                # all. Before the same arm was FABRICATED as
                # 'N-tritetracontyl', a C43 chain; removing the fabrication made
                # the refusal honest and exposed this caller (the contributor guide #9).
                return None
            _n_sub_names.append(_rn)
        if _n_sub_names:
            _n_counts = _NCounter(_n_sub_names)
            for _rn in sorted(_n_counts.keys()):
                _rc = _n_counts[_rn]
                if _rc == 1:
                    all_prefixes.append(f"N-{_wrap_n_substituent(_rn)}")
                else:
                    from ..assembly.naming_utils import multiplied_component as _mc
                    all_prefixes.append(
                        f"N,N-{_mc(_rc, _rn, _wrap_n_substituent(_rn))}"
                    )
    return _GO_ON


def _route_amide_prefixes(_AMIDE_PRINCIPAL_FGS, all_prefixes, features, mol, principal_chain, principal_group):
    """An N-substituted amide principal group with the prefixes assembled."""
    if principal_group in _AMIDE_PRINCIPAL_FGS:
        _amide_matches_ns = features.functional_groups.get(
            principal_group, []) or []
        if len(_amide_matches_ns) == 1:
            _amide_match_ns = _amide_matches_ns[0]
            _amide_n_atom_ns = next(
                (a for a in _amide_match_ns
                 if mol.GetAtomWithIdx(a).GetSymbol() == 'N'), None)
            _amide_acyl_c_ns = next(
                (a for a in _amide_match_ns
                 if mol.GetAtomWithIdx(a).GetSymbol() == 'C'
                 and any(
                     nb.GetSymbol() == 'O'
                     and mol.GetBondBetweenAtoms(a, nb.GetIdx())
                            .GetBondTypeAsDouble() == 2.0
                     for nb in mol.GetAtomWithIdx(a).GetNeighbors())),
                None,
            )
            if (_amide_n_atom_ns is not None
                    and _amide_acyl_c_ns is not None
                    and _amide_acyl_c_ns in set(principal_chain)):
                from collections import Counter as _NCounterAmide

                from ..assembly.composer import _name_r_group, _wrap_n_substituent
                from ..assembly.naming_utils import enclose_if_compound

                _an_sub_names: List[str] = []
                for _anb in mol.GetAtomWithIdx(_amide_n_atom_ns).GetNeighbors():
                    _ani = _anb.GetIdx()
                    # Skip the acyl carbon itself and hydrogens; the remaining
                    # heavy neighbours are the amide's N-substituent(s).
                    if _ani == _amide_acyl_c_ns or _anb.GetAtomicNum() <= 1:
                        continue
                    _arn = _name_r_group(
                        mol, _ani, exclude_atoms={_amide_n_atom_ns})
                    if not _arn:
                        # Fail closed -- see docstring above: this is now the
                        # ONLY producer of this amide's N-substituent prefix
                        # (the branch was excluded from the walk), so skipping
                        # an un-nameable one would DELETE atoms, not degrade.
                        return None
                    #: a COMPOUND R group (e.g. "2-hydroxyethyl")
                    # takes its own enclosing marks before the "N-" locant is
                    # prefixed -- `_name_r_group` returns the BARE compound
                    # token (verified: "2-hydroxyethyl", no parens), and
                    # `_wrap_n_substituent` deliberately only ESCALATES an
                    # enclosure that is already there (its own docstring: "do
                    # not need brackets" for a name with none). Without this,
                    # the emitted prefix read "N-2-hydroxyethyl..." instead of
                    # "N-(2-hydroxyethyl)...".
                    _arn = enclose_if_compound(_arn)
                    _an_sub_names.append(_arn)
                if _an_sub_names:
                    _an_counts = _NCounterAmide(_an_sub_names)
                    for _arn in sorted(_an_counts.keys()):
                        _arc = _an_counts[_arn]
                        if _arc == 1:
                            all_prefixes.append(
                                f"N-{_wrap_n_substituent(_arn)}")
                        else:
                            from ..assembly.naming_utils import multiplied_component as _mc
                            all_prefixes.append(
                                f"N,N-{_mc(_arc, _arn, _wrap_n_substituent(_arn))}"
                            )
    return _GO_ON
