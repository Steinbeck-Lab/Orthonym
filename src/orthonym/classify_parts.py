"""The phases of ``Orthonym._classify`` (``namer.py``), moved here verbatim: the parent ring choice, the
cyclic classification with its ring assembly, PAH, senior ring, heterocyclic and carbocyclic steps, and the chain
parent. Each takes the namer (``self``) and the features it fills in; a phase that ends the classification returns
True. ``_classify`` calls them in its order."""
from .namer import (Chem, _build_ring_info_for_parent_selection, _demote_offring_principal_group_matches,
                    build_atom_to_locant, classify_ring, find_principal_chain, get_ring_info, get_ring_systems,
                    orient_chain, suffix_heteroatoms)


def ring_parent_choice(self, features):
    """A cyclic molecule: the parent choice between its ring systems and a chain."""
    if features.is_cyclic:
        from .rules.n_branch import reanchor_bridging_amine
        from .rules.parent_selection import select_parent

        # Get ring atoms to exclude when finding chain
        all_ring_atoms = set()
        for ring in features.ring_systems:
            all_ring_atoms.update(ring)

        # Find potential principal chain (excluding ring atoms)
        # IMPORTANT: namer.py does the chain finding, then passes result to select_parent
        potential_chain = find_principal_chain(
            features.mol,
            features.functional_groups,
            features.principal_group,
            exclude_atoms=all_ring_atoms
        )
        # 'Maximum number of substituents cited as prefixes' (the Blue Book:
        # 21604): "The preferred IUPAC name is based on the senior parent structure
        # that has the maximum number of substituents cited as prefixes" (the hydro
        # prefixes excepted, as the rule says); example (4),:21624, counts the
        # ring substituent of the parent chain. The call above leaves a ring bonded
        # to the chain out of that count, so two chains of one length bearing one
        # prefix each tie and the input's atom order decided ('methyl 2-benzyl
        # propanoate' for some spellings of methyl 2-methyl-3-phenylpropanoate).
        # The chain is the one makes senior (ring branches counted, located
        # and ordered as prefixes); the call above stands where that comparison is
        # undecided (None) and for a group whose producer is not known to build on
        # the chain it is given (``chains.GIVEN_CHAIN_GROUPS``). A chain of another
        # length is never taken. A chain of one carbon is compared too when the group is an
        # amine whose nitrogen is bonded to several carbons (``chains.p45_pending``).
        #
        # The comparison is made after ``select_parent`` has chosen the chain (below), not
        # before it. It only ever swaps in a chain of the same length that holds the same
        # principal group (an amine's nitrogen holds every chain it bonds to), and the scorer ranks a chain
        # against a ring on the count of principal groups it holds, its atom class and
        # then 'ring senior to chain',,, none of which tells
        # two such chains apart. So the choice between ring and chain does not depend on
        # which of the two is passed, and a ring parent never reads the chain. Counting the
        # ring branches of a large ring system names whole macrocyclic branches (about 2 s
        # for each branch of a 105-atom fragment, six branches a call): run before parent
        # selection it cost +35 s on an allyl oligosaccharide and +20 s on a decacyclic
        # macrocycle, whose parent is the ring either way.
        from .perception.chains import p45_pending, p45_principal_chain
        _p45_pending = p45_pending(features, potential_chain)

        # Only do parent selection if we found a meaningful chain (>= 2
        # carbons) OR (C- / V-4, a 1-carbon chain whose
        # principal-characteristic-group carbon is NOT ring-attached: the
        # ring then cannot express the PCG as a ring suffix, so the 1-carbon
        # parent (formamide / formic acid / methanal) must be parent-eligible
        # (else an N-aryl formamide falls to the ring -> 'carbamoylbenzene').
        # select_parent applies the same ring-neighbour test, so a
        # ring-attached single PCG carbon (benzaldehyde/benzamide) is
        # unaffected.
        _run_parent_selection = bool(potential_chain) and len(potential_chain) >= 2
        if (potential_chain and len(potential_chain) == 1
                and features.principal_group and features.principal_group_atoms):
            _single = potential_chain[0]
            if not any(
                nbr.GetIdx() in all_ring_atoms
                for nbr in features.mol.GetAtomWithIdx(_single).GetNeighbors()
            ):
                _run_parent_selection = True
            else:
                # Wave2 T3a: skeletal suffixes (-ol/-amine/-thiol/-imine,...)
                # have no exocyclic-carbon form, so a ring-attached single
                # PCG carbon still cannot put the suffix on the ring; nor can
                # the sulfonic/sulfinic acid and amide suffixes.
                # Parent selection is eligible when the single carbon holds
                # the suffix and the ring does not (rules.parent_selection).
                from .rules.parent_selection import single_carbon_hosts_suffix
                _run_parent_selection = single_carbon_hosts_suffix(
                    features.mol, features.principal_group,
                    features.principal_group_atoms, _single, all_ring_atoms)
        if _run_parent_selection:
            # a phase: delegate ring-type dispatch to helper.
            # Replaces the prior inline fused-hetero-only block; new
            # helper covers fused-hetero / PAH / benzene / simple-
            # hetero / spiro-stub / VB-stub / else->None per.
            _ring_info = _build_ring_info_for_parent_selection(features)
            # a phase: stash on features so downstream pool.add call sites read it without a
            # signature change (transient, not a MolecularFeatures field,; not frozen).
            features._ring_info = _ring_info

            # Pass pre-computed chain to select_parent
            def _select(chain):
                return select_parent(
                    mol=features.mol,
                    ring_systems=features.ring_systems,
                    principal_chain=chain,
                    principal_group=features.principal_group,
                    principal_group_atoms=features.principal_group_atoms,
                    ring_info=_ring_info,
                    #: 0 for every normal call (byte-identical);
                    # >0 only on a best-effort offer-retry inner instance.
                    _offer_rank=self._forced_parent_rank,
                )

            selection = _select(potential_chain)
            if _p45_pending and selection.parent_type == 'chain':
                # (see above): the chain is the parent, so the chain
                # makes senior is the one to pass; selection is made again only when
                # it differs, with the arguments of the call above.
                _senior = p45_principal_chain(features, potential_chain)
                if _senior != potential_chain:
                    potential_chain = _senior
                    selection = _select(potential_chain)
            features.parent_selection_result = selection  # a phase (V18 Appendix A.5)
            features.principal_group_atoms = reanchor_bridging_amine(
                features.mol, features.principal_group, features.principal_group_atoms, selection.parent_atoms)
            #: publish the top-level pool size so the offer-retry
            # can bound its loop at the real pool length. Guarded to the true
            # top-level molecule (visited set empty) so a substituent's own
            # parent selection never clobbers it.
            try:
                from .assembly.fragment_naming import is_top_level_naming as _sp13_top
                if _sp13_top():
                    self._top_parent_pool_size = getattr(
                        selection, 'parent_pool_size', 1)
            except Exception:
                pass

            if selection.parent_type == 'chain':
                features.chain_is_parent = True
                # is_cyclic stays True -- ring data needed for ring-as-substituent naming (a phase)
                features.principal_chain = selection.parent_atoms
                features.ring_substituents_as_groups = selection.substituent_rings


def cyclic_ring(self, features):
    """The classification of a cyclic parent; True when it ends the classification."""
    if not features.is_cyclic:
        return False
    # Check for ring assemblies FIRST (identical disconnected ring systems)
    # Must come before fused/polycyclic classification because ring assemblies
    # have 2+ separate ring systems that would otherwise be misrouted
    _stop, _principal_ring_system = ring_systems_and_pah(self, features)
    if _stop:
        return True

    ring_info = get_ring_info(features.mol)
    atom_rings = ring_info['atom_rings']

    # Wave2 determinism): among monocyclic candidate rings,
    # atom_rings[0] (first SSSR ring) is SMILES-order-dependent. When the
    # principal characteristic group sits on EXACTLY ONE ring, makes
    # that ring senior (it precedes the seniority tiebreak) — so move
    # it to the front deterministically. Without this, a molecule with two
    # equal rings where only one bears the PCG (e.g. 4-styrylbenzoic acid:
    # the COOH benzene vs the styryl phenyl) flipped parent by spelling,
    # dropping the acid to 'ethenylbenzene' for some orderings. Fused/PAH
    # systems returned earlier; chain-parent cases are unaffected (the PCG
    # then isn't on a ring). Tie (PCG on >1 ring) keeps SSSR order.
    if (atom_rings and len(atom_rings) >= 2
            and features.principal_group_atoms
            and not features.chain_is_parent):
        from .rules.parent_selection import is_principal_group_on_ring
        _pcg_rings = [
            r for r in atom_rings
            if is_principal_group_on_ring(
                features.mol, set(r),
                features.principal_group_atoms,
                features.principal_group)
        ]
        if len(_pcg_rings) == 1 and _pcg_rings[0] is not atom_rings[0]:
            atom_rings = [_pcg_rings[0]] + [
                r for r in atom_rings if r is not _pcg_rings[0]
            ]

    if atom_rings:
        # Select the most senior ring system per IUPAC
        # and store it for downstream use (e.g., ring-vs-chain
        # comparison in the composer). The monocyclic dispatch
        # path below uses atom_rings[0] which preserves SSSR
        # cyclic traversal order needed by orientation functions.
        # Complex multi-ring (fused/bridged) systems are handled
        # by _classify_complex_ring in the composer.
        #: reuse the one authoritative computation (see top of block)
        senior_ring_and_type(self, features, _principal_ring_system, atom_rings)

        # Check if this is a benzene ring
        from .rules.benzene import get_benzene_substituents, is_benzene_ring
        if is_benzene_ring(features.mol, features.principal_ring):
            features.is_benzene = True
            # C- (V-3): with >1 benzene ring and no principal
            # characteristic group (aralkyl/diaryl ethers, e.g. benzyl
            # phenyl ether), `atom_rings[0]` is SMILES-order-dependent, so
            # the parent flipped with the input spelling -> non-determinism
            # ('(phenoxymethyl)benzene' vs 'benzoxybenzene'). Pick the
            # parent ring deterministically (carbon-linked preference +
            # canonical rank). Single-benzene and PCG-bearing rings are
            # left to the existing principal-ring selection.
            from .rules.benzene import (
                _preferred_benzene_parent_ring,
                _select_benzene_parent_ring,
            )
            _benzene_rings = [
                r for r in features.mol.GetRingInfo().AtomRings()
                if is_benzene_ring(features.mol, r)
            ]
            if len(_benzene_rings) > 1 and not features.principal_group:
                # / (diaryl-linked-by-heteroatom):
                # pick the parent whose COMPLETE name is preferred by
                # alphanumerical order (R<S tie-break), naming each tied
                # candidate. Falls back to the structure-only selector
                # when the name-based choice is unavailable.
                _pref = _preferred_benzene_parent_ring(features.mol)
                if _pref is not None:
                    features.principal_ring = _pref[0]
                else:
                    _bz = _select_benzene_parent_ring(features.mol)
                    if _bz is not None:
                        features.principal_ring = _bz
            features.benzene_ring = features.principal_ring
            features.benzene_substituents = get_benzene_substituents(
                features.mol, features.principal_ring, features.principal_group
            )
        elif features.ring_type and features.ring_type.startswith('heterocyclic'):
            # Heterocyclic ring: classify, detect substituents, and orient
            heterocyclic_ring(self, features)
        else:
            # Non-benzene, non-heterocyclic ring: detect substituents and orient
            carbocyclic_ring(self, features)
    return False


def ring_systems_and_pah(self, features):
    """``(stop, the principal ring system)``: ring assemblies and polycyclic aromatic hydrocarbons first."""
    if len(features.ring_systems) >= 2:
        from .rules.branched_assembly import detect_assembly_or_branched
        assembly_info = detect_assembly_or_branched(features.mol, features.ring_systems, features)
        if not assembly_info and not features.chain_is_parent:
            # / (the Blue Book,:24153): two ring systems of
            # one skeleton that differ in hydrogenation are a ring assembly named with
            # 'hydro' prefixes ('2,3-dihydro-1,1'-biphenyl'). The detector offers a name
            # only when it is built and OPSIN reads it back to this structure, so a
            # molecule it declines keeps the ring classification it had. A chain parent never
            # reads the assembly, so no name is built for it.
            from .rules.ring_assemblies import detect_hydro_ring_assembly
            assembly_info = detect_hydro_ring_assembly(
                features.mol, features.ring_systems)
        if assembly_info:
            features.ring_assembly_info = assembly_info
            if not features.chain_is_parent:
                return True, None  # Skip other ring classification for assemblies

    # (a phase) chokepoint consolidation: compute the senior ring
    # system ONCE via select_principal_ring_system) and carry it
    # on the ParentSelectionResult, so the PAH guard, the among-rings
    # selector, and the derived senior_ring_system / principal_ring all
    # read ONE authoritative value instead of recomputing the identical
    # pure call. Behavior-preserving by construction (same args, same
    # pure function); this only de-duplicates two identical calls into one.
    from .rules.ring_selection import select_principal_ring_system_with_groups
    _principal_ring_system = (
        select_principal_ring_system_with_groups(features)
        if features.ring_systems else None
    )
    # Carry it on the one authoritative result. `selection` is the same
    # ParentSelectionResult set by the select_parent call above (None when
    # there was no chain candidate, so select_parent never ran).
    selection = features.parent_selection_result
    if selection is not None:
        selection.principal_ring_system = _principal_ring_system

    # Check for polycyclic aromatics FIRST (naphthalene, anthracene, etc.)
    # These take precedence over single-ring classification
    from .rules.polycyclics import get_polycyclic_substituents, identify_polycyclic
    pah_name = identify_polycyclic(features.mol)
    if pah_name and len(features.ring_systems) >= 2:
        # task 9 (A-i): the PAH early-return must not preempt the
        # parent decision. With >=2 ring systems, take the PAH
        # route ONLY when the PAH system survives:
        # (a) — a principal characteristic group on a
        # DIFFERENT ring system makes that system the parent
        # (2-(naphthalen-2-yl)cyclohexan-1-ol, not naphthalene);
        # (b) — the among-rings winner must BE the PAH
        # system (2-(naphthalen-2-yl)furan: furan is senior).
        from .rules.polycyclics import get_polycyclic_core_atoms
        _core = get_polycyclic_core_atoms(features.mol, pah_name)
        _core_set = set(_core) if _core else set()
        if _core_set:
            # counts the principal groups on each ring system: one with MORE than the PAH is the
            # parent ('4-[2-(1-hydroxynaphthalen-2-yl)ethyl]benzene-1,2-diol'). Only a PAH with strictly more
            # skips the veto below;:19422 "the same number of a characteristic group must be
            # present in all cyclic structures to be compared", so a tie (none on either included) runs it.
            _pah_more = False
            if features.principal_group_atoms:
                from .rules.parent_selection import _count_pg_on_ring as _n_pg
                _pg_args = (features.principal_group_atoms, features.principal_group)
                _n_pah = _n_pg(features.mol, _core_set, *_pg_args)
                _n_other = max((_n_pg(features.mol, set(rs), *_pg_args) for rs in features.ring_systems
                                if not (set(rs) & _core_set)), default=0)
                if _n_other > _n_pah:
                    pah_name = None
                _pah_more = _n_pah > _n_other
            #: reuse the one authoritative computation; on a tie with groups, compares only the
            # ring systems that hold the most (a tertiary amine N on fluorene and on two phenyls: not carbazole)
            _senior = _principal_ring_system
            if pah_name and not _pah_more and features.principal_group_atoms and _n_pah:
                from .rules.ring_selection import select_principal_ring_system as _sel
                _senior = _sel(features.mol, [rs for rs in features.ring_systems if set(rs) & _core_set
                                              or _n_pg(features.mol, set(rs), *_pg_args) == _n_pah])
            if pah_name:
                # / build precedes: when the PAH has more principal groups, a
                # senior heterocyclic SUBSTITUENT ring must not steal the parent (5-(1,3-dioxo-...-isoindol-2-yl)-
                # naphthalene-1-carboxylic acid). Only the among-rings tie-break is skipped.
                if (_senior and not (set(_senior) & _core_set)
                        and not _pah_more):
                    pah_name = None
    if pah_name:
        features.polycyclic_name = pah_name
        features.polycyclic_substituents = get_polycyclic_substituents(
            features.mol, pah_name,
            principal_group=features.principal_group,
        )
        # Set ring type for consistency
        features.ring_type = 'aromatic'
        if not features.chain_is_parent:
            return True, None  # Skip other ring classification for PAHs
    return False, _principal_ring_system


def senior_ring_and_type(self, features, _principal_ring_system, atom_rings):
    """The senior ring system and its ring type."""
    principal = _principal_ring_system
    features.senior_ring_system = principal if principal else atom_rings[0]

    # For multi-ring-system molecules, use a SSSR ring from
    # the senior system as principal_ring when the senior
    # system is strictly LARGER than the default ring's system.
    # This ensures fused/bridged senior systems (imidazopyridine,
    # xanthene) take precedence over small monocyclic rings
    # (benzene) per IUPAC.
    #
    #.1 S2: select_principal_ring_system, with the S1
    # unsaturation tiebreak) is now AUTHORITATIVE among ring
    # systems. The legacy ``size_diff >= 3`` heuristic that DISCARDED
    # the winner for equal/near-equal senior systems is deleted
    # (it produced (furan-2-yl)benzene instead of 2-phenylfuran and
    # pyridinylcyclohexane instead of 4-cyclohexylpyridine). Gated only
    # by the two correctness guards below; verified on the among-rings
    # PIN gold corpus (benchmarks/the gold set/among_rings_gold.json).
    if principal and len(features.ring_systems) >= 2:
        senior_set = set(principal)
        # Find the ring system that contains the default ring
        default_system = set(atom_rings[0])
        for rs in features.ring_systems:
            if set(atom_rings[0]).issubset(rs):
                default_system = rs
                break

        # Guard (a): the principal characteristic group must be
        # IN the parent. If the PG sits on the default ring system,
        # that ring stays parent regardless of ring seniority.
        from .rules.parent_selection import is_principal_group_on_ring
        pg_on_default = False
        pg_on_senior = False
        if features.principal_group_atoms:
            pg_on_default = is_principal_group_on_ring(
                features.mol, default_system,
                features.principal_group_atoms,
                features.principal_group,
            )
            # C4 +: a BRIDGING characteristic group
            # (e.g. a secondary amine N bonded to a carbon of EACH
            # ring — diaryl/aryl-heteroaryl amines) satisfies
            # for BOTH ring systems. When it also sits on the SENIOR
            # ring, no longer disqualifies the senior ring, so
            # the seniority tiebreak decides -> the senior ring
            # is the parent (pyridine over benzene ->
            # N-phenylpyridin-4-amine, not (aminopyridinyl)benzene).
            pg_on_senior = is_principal_group_on_ring(
                features.mol, senior_set,
                features.principal_group_atoms,
                features.principal_group,
            )

        # Guard (b) Phase-171 / CHEBI:59269: when the CHAIN is the
        # parent, both rings are mere substituents; reassigning
        # principal_ring here perturbs stereodescriptor emission on the
        # chain handler (the documented +1 self-test regressor of the
        # 171 trial). Among-ring seniority does not decide a chain
        # parent — keep the default ring. (See test_among_rings_gold.py
        # TestPhase171StereoGuard.)
        if ((not pg_on_default or pg_on_senior)
                and not features.chain_is_parent
                and senior_set != default_system):
            best_ring = atom_rings[0]
            best_overlap = 0
            for ring in atom_rings:
                overlap = len(set(ring) & senior_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_ring = ring
            features.principal_ring = best_ring
        else:
            # Wave-2 C2: when the PG sits on TWO OR
            # MORE benzene rings (the dicyano-diaryl-ether BB
            # example), atom_rings[0] is SMILES-order-dependent
            # — the parent flipped with the spelling (a genuine
            # NEW-NONDET gate catch). Break the tie with the
            # deterministic substituent-count + canonical-rank
            # selector; everything else keeps atom_rings[0].
            _pg_ring_count = 0
            if features.principal_group_atoms:
                for rs in features.ring_systems:
                    if is_principal_group_on_ring(
                            features.mol, set(rs),
                            features.principal_group_atoms,
                            features.principal_group):
                        _pg_ring_count += 1
            _det_ring = None
            if _pg_ring_count >= 2:
                from .rules.benzene import (
                    _select_benzene_parent_ring as _sbpr,
                )
                from .rules.benzene import (
                    is_benzene_ring as _ibr,
                )
                if all(_ibr(features.mol, r) for r in atom_rings):
                    #: the ring with more principal-group
                    # instances wins before any substituent count.
                    _det_ring = _sbpr(features.mol,
                                      features.principal_group_atoms,
                                      features.principal_group)
            features.principal_ring = (
                _det_ring if _det_ring is not None
                else atom_rings[0]
            )
    else:
        features.principal_ring = atom_rings[0]
    features.ring_type = classify_ring(features.mol, features.principal_ring)

    # (BB 25207): the principal ketone-family group sits
    # in BOTH the ring and a pendant chain; the ring won parent
    # selection (more of the group, or ring default on tie), so an
    # off-ring match must NOT become a second ring '-one' suffix —
    # demote it to its chain, which the ring-substituent enumerator
    # then names as an oxo-alkyl prefix (2-oxobutyl). Fail-closed:
    # leaves principal_group_atoms unchanged on any ambiguity.
    # "Ring" here is the parent ring SYSTEM, not the one ring that
    # principal_ring names: rings joined by a spiro atom are one
    # parent hydride, the Blue Book, "the naming of
    # parent hydrides containing free spiro unions"), and both of its
    # ketones are suffixes, 'spiro[4.5]decane-1,7-dione (PIN)'
    #,:28404). Measured against the single ring, the
    # ketone on the second spiro ring was 'off-ring' and the 1:1 tie
    # demoted it to 'oxo' ('1-oxospiro[4.5]decan-7-one').
    if not getattr(features, 'chain_is_parent', False):
        _pr_set = set(features.principal_ring or ())
        _parent_system = next(
            (s for s in get_ring_systems(
                features.mol, include_spiro=True)
             if _pr_set and _pr_set <= s),
            _pr_set)
        _demote_offring_principal_group_matches(
            features, _parent_system
        )

    # (b) /: a mancude monocyclic hydrocarbon
    # (all-carbon, RDKit-aromatic, NOT benzene, n>=7) is named as the
    # cyclo-polyene (cyclodeca-1,3,5,7,9-pentaene), never as an
    # [n]annulene component prefix and never dropped as "unknown".
    # RDKit marks it aromatic (10-pi Huckel), routing it to the
    # aromatic decline; kekulise it so it takes the existing
    # cycloalkene (polyene) path exactly like cyclooctatetraene.
    if features.ring_type == 'aromatic':
        from .rules.cycloalkanes import (
            is_mancude_monocyclic_hydrocarbon as _is_mancude_ring,
        )
        if _is_mancude_ring(features.mol, features.principal_ring):
            _km = Chem.Mol(features.mol)
            try:
                Chem.Kekulize(_km, clearAromaticFlags=True)
                features.mol = _km
                features.ring_type = 'cycloalkene'
            except Exception:
                pass  # fail closed: leave as aromatic (declines)


def heterocyclic_ring(self, features):
    """A heterocyclic principal ring: classify, detect substituents, and orient."""
    from .rules.heterocycles import (
        classify_heterocycle,
        get_heterocycle_substituents,
        orient_heterocycle_with_substituents,
        ring_principal_suffix_atoms,
    )

    features.heterocycle_info = classify_heterocycle(
        features.mol, features.principal_ring
    )

    # Detect substituent positions for proper orientation
    ring_set = set(features.principal_ring)
    sub_positions = set()
    for idx in features.principal_ring:
        atom = features.mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in ring_set:
                sub_positions.add(idx)
                break

    # / ring-suffix fix : identify the ring atoms bearing the
    # principal characteristic group so the suffix gets the lowest
    # locant (c)), after the heteroatom. Same extraction as
    # the cycloalkene branch: a ring C whose FG-match heteroatom is
    # exocyclic (C=O ketone, C-OH alcohol/phenol, C-NH2 amine,...).
    pg_ring_atoms = set()
    if features.principal_group:
        from .rules.seniority import get_prefix as _pg_get_prefix
        _pg_prefix = _pg_get_prefix(features.principal_group)
        from .rules.seniority import SUFFIX_FORMS as _PG_SUFFIX_FORMS
        # the ring suffix of the principal group is an N-substituted
        # amide suffix ('-carboxamide'): the heterocycle assembler
        # cites it. An ester or anhydride class is named by
        # functional class nomenclature (a lactone ring by 'oxo'),
        # so it cites no ring suffix to number first.
        _pg_ring_suffix = (_PG_SUFFIX_FORMS.get(features.principal_group)
                           or ("", ""))[1]
        _pg_has_suffix_form = _pg_ring_suffix.endswith("carboxamide")
        # The principal characteristic group is a CLASS, not a
        # single FG label. A ring bearing an exocyclic primary
        # alcohol (-CH2OH) plus secondary ring alcohols (ring -OH)
        # has principal_group == primary_alcohol, but the RING
        # carbons that carry the -ol SUFFIX are the secondary ones
        # (the assembly already unions them via the shared 'hydroxy'
        # prefix). Collect every same-prefix FG class so orient
        # gives the suffix-bearing ring carbons the lowest locants
        # (c)) DETERMINISTICALLY (alcohol-class union);
        # without this the suffix-locant tie is broken by SMILES
        # atom order -> non-deterministic numbering .
        for _fg, _matches in features.functional_groups.items():
            # A principal group with no prefix form but an amide ring
            # suffix (an N-substituted amide: 'secondary_amide',
            # 'tertiary_amide') still anchors its own matches: its suffix
            # ('-carboxamide') takes the lowest locant before the
            # prefixes, (c) (the Blue Book) --
            # 'N,N,6-trimethylpyridine-3-carboxamide', not 'N,N,2-
            # trimethylpyridine-5-carboxamide' ('2-chloropyridine-3-
            # carboxamide (PIN)',:32895).
            if (_fg != features.principal_group
                    or _pg_prefix is not None
                    or not _pg_has_suffix_form) and (
                    _pg_prefix is None
                    or _pg_get_prefix(_fg) != _pg_prefix):
                continue
            for match in _matches:
                match_set = set(match)
                if not (match_set & ring_set):
                    # CASE A (twin of the cycloalkane branch
                    # below, ~3627): a WHOLLY-EXOCYCLIC appended
                    # suffix (-carboxylic acid, -carbaldehyde,
                    # -carbonitrile, -carboxamide,...). The
                    # entire FG match lies OFF the ring, so the
                    # CASE-B ring-C-bonded-heteroatom test never
                    # fires and pg_ring_atoms stayed empty --
                    # the ring-direction tie for a symmetric-
                    # heteroatom ring (furan/thiophene/pyrrole)
                    # was then broken arbitrarily by canonical
                    # rank, giving the suffix the HIGHER locant
                    # (2-methylfuran-5-carboxylic acid). The
                    # expressed-suffix ANCHOR is the ring atom
                    # bonded to the match carbon; per (c)
                    # it takes the lowest locant BEFORE the
                    # detachable prefix (5-methylfuran-2-
                    # carboxylic acid). Same-prefix
                    # scoping is inherited from the enclosing
                    # loop, so only principal-group-class
                    # suffixes are anchored. (A shared helper
                    # unifying this with the cycloalkane twin +
                    # general_engine + parent-selection is a
                    # tracked follow-up.)
                    # NARROW (protect W6B-LOCK): anchor ONLY a
                    # genuine C-APPENDED suffix -- the ring-bonded
                    # match carbon must carry a characteristic
                    # MULTIPLE bond to a non-carbon match atom
                    # (C=O aldehyde/acid/amide, C#N nitrile). The
                    # same-prefix union also admits an
                    # exocyclic -CH2OH / -CH2NH2 whose ring-bonded
                    # carbon is sp3 single-bonded: that is a
                    # (hydroxymethyl)/(aminomethyl) PREFIX, not a
                    # ring-appended -ol/-amine suffix, so it must
                    # NOT be anchored (else the oxepane pentol
                    # loses the lowest locants to the hydroxymethyl
                    # carbon -- protect W6B-LOCK regression).
                    for atom_idx in match:
                        atom = features.mol.GetAtomWithIdx(atom_idx)
                        if atom.GetSymbol() != 'C':
                            continue
                        _is_suffix_c = any(
                            b.GetBondTypeAsDouble() > 1.0
                            and b.GetOtherAtom(atom).GetIdx() in match_set
                            and b.GetOtherAtom(atom).GetSymbol() != 'C'
                            for b in atom.GetBonds()
                        )
                        if not _is_suffix_c:
                            continue
                        for nbr in atom.GetNeighbors():
                            if nbr.GetIdx() in ring_set:
                                pg_ring_atoms.add(nbr.GetIdx())
                                break
                    continue
                for atom_idx in match:
                    if atom_idx not in ring_set:
                        continue
                    atom = features.mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() != 'C':
                        continue
                    for nbr in atom.GetNeighbors():
                        nbr_idx = nbr.GetIdx()
                        if (nbr_idx not in ring_set
                                and nbr_idx in match_set
                                and nbr.GetSymbol() != 'C'):
                            pg_ring_atoms.add(atom_idx)
                            break

    # (c) NUMBERING (the Blue Book): a principal
    # characteristic group expressed as a SUFFIX takes the lowest
    # locant BEFORE a detachable prefix (f). The same-prefix loop
    # above anchors only a CARBON-centred appended suffix, so it
    # MISSES two ring-suffix families whose suffix decision lives
    # in get_heterocycle_substituents: (A) the pseudoketone
    # -one/-dione/-thione ring carbons (principal group is a
    # lactam-declined amide / cyclic imide / lactone-declined
    # ester / thioamide, whose get_prefix is None) and (B) a ring
    # atom bearing an exocyclic SULFUR-oxoacid suffix (S is not a
    # carbon). Without these the (f) substituent set decided
    # and put the SUBSTITUTED ring atom at locant 1
    # (1-methylimidazolidine-2,5-dione, 1-methylpiperazine-4-
    # sulfonic acid). Feed the suffix atoms exactly as the carboxy
    # path does so (c) governs.
    pg_ring_atoms |= ring_principal_suffix_atoms(
        features.mol, features.principal_ring,
        features.principal_group, features.functional_groups
    )
    _ring_set_pg = set(features.principal_ring)
    _pg_on_ring = any(
        a in _ring_set_pg or any(
            n.GetIdx() in _ring_set_pg
            for n in features.mol.GetAtomWithIdx(a).GetNeighbors())
        for m in (features.principal_group_atoms
                  or features.functional_groups.get(
                      features.principal_group, ()))
        for a in m)
    if not pg_ring_atoms and not _pg_on_ring:
        # the principal group is carried off the ring: the heterocycle
        # assembler still cites a ring suffix of its own (c))
        from .rules.heterocycles import (
            ring_suffix_atoms_the_assembler_cites,
        )
        pg_ring_atoms |= ring_suffix_atoms_the_assembler_cites(
            features.mol, features.principal_ring,
            features.principal_group)

    # Orient considering heteroatoms, the principal group, then
    # other substituents for lowest locants order).
    oriented, atom_to_locant = orient_heterocycle_with_substituents(
        features.mol, features.principal_ring, sub_positions,
        principal_group_atoms=pg_ring_atoms if pg_ring_atoms else None,
        mancude_hydrogen=True,
    )
    features.oriented_heterocycle = oriented
    features.heterocycle_atom_to_locant = atom_to_locant

    # Get substituent details for naming; pass the principal group so
    # a senior FG on the ring is emitted as a SUFFIX, not a prefix.
    features.heterocycle_substituents = get_heterocycle_substituents(
        features.mol,
        features.principal_ring,
        oriented,
        atom_to_locant,
        principal_group=features.principal_group
    )


def carbocyclic_ring(self, features):
    """A non-benzene, non-heterocyclic principal ring: substituents and orientation."""
    from .rules.cycloalkanes import (
        get_ring_double_bonds,
        get_ring_substituents,
        get_ring_triple_bonds,
        orient_cycloalkane,
        orient_cycloalkene,
    )

    # Get ring substituents
    features.ring_substituents = get_ring_substituents(
        features.mol, features.principal_ring
    )

    # Get ring double bonds, and the triple bonds:
    # (the Blue Book) numbers both, as one set first
    # (e)(ii),:3290).
    features.ring_double_bonds = get_ring_double_bonds(
        features.mol, features.principal_ring
    )
    features.ring_triple_bonds = get_ring_triple_bonds(
        features.mol, features.principal_ring
    )

    # Determine principal group atoms on the ring
    # IUPAC /: principal group gets lowest locant
    # We identify ring C atoms that directly bear the principal
    # group's characteristic heteroatom (e.g., C=O for ketone,
    # C-OH for alcohol). The characteristic heteroatom must be
    # bonded DIRECTLY to a ring carbon (not via an exocyclic C).
    pg_ring_atoms = set()
    if features.principal_group and features.principal_group in features.functional_groups:
        from .rules.parent_selection import (
            SKELETAL_SUFFIX_PGS as _SKEL_SFX_RING)
        ring_set = set(features.principal_ring)
        # Breadth job 3 review fixes: the principal characteristic
        # group is the WHOLE equal-seniority class the suffix cites
        # (DD5,, as on the chain path below -- the
        # principal SUBTYPE's matches alone left a secondary or
        # tertiary ring -OH out of the anchor set when a primary
        # -CH2OH decided the subtype. And an -ol/-amine/-thiol/-one
        # decorates a skeletal atom: it has no exocyclic-carbon
        # suffix form (SKELETAL_SUFFIX_PGS), so a wholly exocyclic
        # match (the -CH2OH of 'hydroxymethyl') is a substituent and
        # anchors nothing. (c) (the Blue Book) numbers
        # the suffixes first: '4-(hydroxymethyl)cyclohexan-1-ol',
        # not '1-(hydroxymethyl)cyclohexan-4-ol'.
        _pg_ring_matches = (features.principal_group_atoms
                            or features.functional_groups[features.principal_group])
        for match in _pg_ring_matches:
            match_set = set(match)
            if not (match_set & ring_set):
                if features.principal_group in _SKEL_SFX_RING:
                    continue
                #.1 S4: wholly-exocyclic match = an
                # APPENDED suffix (-carbaldehyde, -carboxylic
                # acid, -carbonitrile). Its expressed-suffix
                # ANCHOR (the ring atom bonded to the match
                # carbon) takes the lowest locant per
                # (c) (2-methylcyclohexane-1-
                # carbaldehyde, never 1-methyl-2-). Matches
                # with no ring contact at all (a CHO at the
                # end of a demoted chain) contribute nothing.
                for atom_idx in match:
                    atom = features.mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() != 'C':
                        continue
                    for nbr in atom.GetNeighbors():
                        if nbr.GetIdx() in ring_set:
                            pg_ring_atoms.add(nbr.GetIdx())
                            break
                continue
            for atom_idx in match:
                if atom_idx in ring_set:
                    atom = features.mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() != 'C':
                        continue
                    # Check if this ring C is bonded to a non-ring
                    # HETEROATOM that is in the FG match
                    for nbr in atom.GetNeighbors():
                        nbr_idx = nbr.GetIdx()
                        if (nbr_idx not in ring_set
                                and nbr_idx in match_set
                                and nbr.GetSymbol() != 'C'):
                            pg_ring_atoms.add(atom_idx)
                            break

    # v52 P1 Task D3 (c), the Blue Book): the loops
    # above only anchor a CARBON-rooted appended suffix
    # (-carboxylic acid, -carbaldehyde,...) -- ``atom.GetSymbol
    # != 'C': continue`` skips every atom of a sulfonic_acid /
    # sulfinic_acid match, whose root is S, not C. So a ring
    # bearing -SO3H / -SO2H never got its suffix atom anchored
    # and (f) (the substituent set) decided numbering
    # instead, giving 'cyclohex-1-ene-4-sulfonic acid' where
    # (c) requires the suffix at locant 1
    # ('cyclohex-3-ene-1-sulfonic acid'). Reuse the SAME
    # ring_principal_suffix_atoms anchor the heterocycle branch
    # above already feeds for this exact family (its case B) --
    # not a new mechanism, the shared one applied to the
    # non-benzene/non-heterocyclic ring path too.
    from .rules.heterocycles import ring_principal_suffix_atoms
    pg_ring_atoms |= ring_principal_suffix_atoms(
        features.mol, features.principal_ring,
        features.principal_group, features.functional_groups
    )

    # Orient the ring based on type
    if features.ring_type == 'cycloalkane':
        features.oriented_ring = orient_cycloalkane(
            features.mol,
            features.principal_ring,
            features.ring_substituents,
            principal_group_atoms=pg_ring_atoms if pg_ring_atoms else None
        )
    elif features.ring_type == 'cycloalkene':
        features.oriented_ring = orient_cycloalkene(
            features.mol,
            features.principal_ring,
            features.ring_double_bonds,
            features.ring_substituents,
            principal_group_atoms=pg_ring_atoms if pg_ring_atoms else None,
            triple_bond_atoms=features.ring_triple_bonds,
        )
        # Calculate ring double and triple bond locants
        # (wrap-aware: the closure bond positions (0, n-1) is
        # locant n)
        if features.oriented_ring:
            from .rules.cycloalkanes import ring_double_bond_locant
            oriented = features.oriented_ring
            n_ring = len(oriented)
            for _bonds, _field in (
                    (features.ring_double_bonds, 'ring_double_bond_locants'),
                    (features.ring_triple_bonds, 'ring_triple_bond_locants')):
                if not _bonds:
                    continue
                locants = []
                for a1, a2 in _bonds:
                    pos1 = oriented.index(a1)
                    pos2 = oriented.index(a2)
                    locants.append(
                        ring_double_bond_locant(pos1, pos2, n_ring)
                    )
                setattr(features, _field, sorted(locants))


def chain_parent(self, features):
    """The principal chain of an acyclic parent, or of a chain parent of a cyclic molecule."""
    if not features.is_cyclic or features.chain_is_parent:
        if not features.chain_is_parent:
            # Normal acyclic molecule - find principal chain
            # Guard: exclude ring atoms even in "acyclic" path (defensive)
            _ring_exclude = set()
            ri = features.mol.GetRingInfo()
            for ring in ri.AtomRings():
                _ring_exclude.update(ring)
            features.principal_chain = find_principal_chain(
                features.mol,
                features.functional_groups,
                features.principal_group,
                exclude_atoms=_ring_exclude if _ring_exclude else None
            )

        if features.principal_chain:
            # Collect principal group atom indices for orientation
            pg_atom_set: set = set()
            if features.principal_group and features.principal_group in features.functional_groups:
                from .rules.parent_selection import (
                    SKELETAL_SUFFIX_PGS,
                    _pg_attachment_atoms,
                )
                # (c) (the Blue Book): the principal characteristic
                # group gets the lowest locants, and that group is the WHOLE
                # equal-seniority class the suffix cites -- DD5
                # already unions primary/secondary/tertiary -ol (and the amine
                # classes) into features.principal_group_atoms, which the suffix
                # locants are counted from. Orienting on the principal SUBTYPE's
                # matches alone left a secondary -OH out of criterion (a), so
                # OCC=CCC(O)CCO tied {1,7}/{1,7} and fell to the ene locant:
                # 'hept-2-ene-1,5,7-triol' (PIN 'hept-5-ene-1,3,7-triol',
                # {1,3,7} < {1,5,7}; TRIAGE row 95 rt-14). Non-union groups have
                # principal_group_atoms == their own matches (unchanged).
                _pg_matches = (features.principal_group_atoms
                               or features.functional_groups[features.principal_group])
                _chain_set = set(features.principal_chain)
                for match in _pg_matches:
                    if features.principal_group in SKELETAL_SUFFIX_PGS:
                        # task 9: the -one family's locant atom is
                        # the carbonyl carbon (PG_ATTACHMENT_INDICES);
                        # feeding the whole match (incl. the FLANKING
                        # carbon) made orientation criterion (a) tie at
                        # {1,2} both ways for 2-carbon ketone chains and
                        # fall through to alphabetics ('...ethan-2-one').
                        _match_atoms = set(_pg_attachment_atoms(
                            features.principal_group, match))
                    else:
                        _match_atoms = set(match)
                    pg_atom_set.update(_match_atoms)
                    # normalizes a match to (heteroatom, bearing C), and
                    # for an N-substituted amine that bearing C can be an
                    # N-alkyl carbon OFF the chain, leaving the instance
                    # invisible to criterion (a): CCN(CC)CCCC(C)N shipped
                    # 'N5,N5-diethylpentane-2,5-diamine' (PIN 'N1,N1-diethyl-
                    # pentane-1,4-diamine', cf. BB:4675). The suffix locant of
                    # such an instance is the chain atom bonded to its
                    # heteroatom, so anchor there. Only the heteroatom's own
                    # chain neighbours count (an -OH on a side-branch carbon
                    # is a substituent, never anchored).
                    if not (_match_atoms | set(match)) & _chain_set:
                        for _a in match:
                            _atom = features.mol.GetAtomWithIdx(_a)
                            if _atom.GetAtomicNum() in (1, 6):
                                continue
                            pg_atom_set.update(
                                nb.GetIdx() for nb in _atom.GetNeighbors()
                                if nb.GetIdx() in _chain_set)

            # Get initial substituents for orientation criterion (d)
            # This is needed BEFORE orientation to apply lowest-locant rule
            initial_subs = self._find_substituents_by_atom(
                features.mol,
                features.principal_chain
            )

            # Orient chain using IUPAC 2013 criteria
            features.principal_chain = orient_chain(
                chain=features.principal_chain,
                mol=features.mol,
                principal_group_atoms=pg_atom_set,
                double_bonds=features.double_bonds,
                triple_bonds=features.triple_bonds,
                substituent_positions=initial_subs, principal_group_heteroatoms=suffix_heteroatoms(features),
            )

            # Build atom-to-locant mapping from oriented chain
            features.atom_to_locant = build_atom_to_locant(features.principal_chain)

            # Get substituents with final locant-based positions
            features.substituents = self._find_substituents(
                features.mol,
                features.principal_chain
            )
