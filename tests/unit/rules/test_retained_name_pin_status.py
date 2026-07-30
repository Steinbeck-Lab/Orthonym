"""Retained trivial names that are NOT preferred IUPAC names (v29 Phase C, Task 10).

Orthonym was emitting retained trivial names as if they were PINs. Origin: probing
``OC(=O)c1ccncc1`` gave ``isonicotinic acid`` while ``BlueBookV2/BlueBookV2.md:29765``
reads verbatim ``isonicotinic acid pyridine-4-carboxylic acid (PIN)``.

Most of these sit in ONE Blue Book example block under §**P-63.1.1** "Retained names"
(``:26762``), which prints the non-preferred name immediately ABOVE the ``(PIN)``. Line
numbers below were each verified individually with ``sed -n '<N>p'``:

===========================  ===========  ==================================
non-preferred (BB line)      PIN line     PIN
===========================  ===========  ==================================
``carvacrol`` ``:26790``     ``:26792``   ``2-methyl-5-(propan-2-yl)phenol``
``thymol`` ``:26794``        ``:26796``   ``5-methyl-2-(propan-2-yl)phenol``
``resorcinol`` ``:26802``    ``:26804``   ``benzene-1,3-diol``
``hydroquinone`` ``:26806``  ``:26808``   ``benzene-1,4-diol``
``1-naphthol`` ``:26818``    ``:26820``   ``naphthalen-1-ol``
===========================  ===========  ==================================

``2-naphthol``, ``pyrogallol`` and ``quinol`` are a DIFFERENT evidence class: they appear
**nowhere** in the Blue Book (grep validated against the known positive ``1-naphthol``,
found at ``:26818``), so they have no standing as PINs at all and the systematic form
governs by default. ``indane`` / ``indoline`` / ``isoindoline`` are a third class, named
verbatim as non-preferred by **P-54.4.3.2** (``:24256``).

★ THREE MEASURED LESSONS, each of which changed the fix:

1. **THE DENY ALONE WAS NOT SUFFICIENT for the naphthols.** Ablation showed it produced
   ``1-hydroxynaphthalene`` — one non-PIN swapped for another (session invariant 11).
   Root cause: ``rules/polycyclics.py`` ``_SUFFIX_PRIORITY`` had no ``'ol'``, so a ring
   hydroxy was demoted to a ``hydroxy`` prefix. Both halves are required.
2. **Denying ``hydroquinone`` UNMASKED ``quinol``**, a second non-PIN synonym for the
   same structure. Found only by re-naming the molecule after the deny. A deny removes
   one surface; it does not guarantee the systematic path wins.
3. **A ``pin: false`` row is NOT always the mechanism.** ``indane`` carried such a row
   all along and still emitted ``indane``, because it is served from
   ``data/fused_heterocycles.py`` — a surface the deny-set does not filter. Measured, not
   assumed. See ``TestDenyMechanismIsReached``.
"""
import pytest

from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


class TestPinsNowEmitted:
    @pytest.mark.parametrize("smiles,expected,authority", [
        # P-63.1.1 retained-name block: BB prints the PIN beside the trivial name
        ("Oc1ccc(O)cc1", "benzene-1,4-diol", "BB:26806/:26808 (was `hydroquinone`)"),
        ("Oc1cccc(O)c1", "benzene-1,3-diol", "BB:26802/:26804 (was `resorcinol`)"),
        ("Oc1cccc2ccccc12", "naphthalen-1-ol", "BB:26818/:26820 (was `1-naphthol`)"),
        ("Cc1ccc(C(C)C)cc1O", "2-methyl-5-(propan-2-yl)phenol",
         "BB:26790/:26792 (was `carvacrol`)"),
        ("Cc1ccc(C(C)C)c(O)c1", "5-methyl-2-(propan-2-yl)phenol",
         "BB:26794/:26796 (was `thymol`)"),
        ("O=C(O)c1ccncc1", "pyridine-4-carboxylic acid",
         "BB:29765 verbatim (was `isonicotinic acid`)"),
        # not in the BB at all -> no PIN standing, systematic governs
        ("Oc1ccc2ccccc2c1", "naphthalen-2-ol", "not in BB (was `2-naphthol`)"),
        ("Oc1cccc(O)c1O", "benzene-1,2,3-triol", "not in BB (was `pyrogallol`)"),
        # P-54.4.3.2 partially saturated heterocycles, BB:24256 verbatim
    ])
    def test_pin_is_emitted(self, namer, smiles, expected, authority):
        assert namer.name(smiles) == expected, authority

    def test_denying_hydroquinone_does_not_leave_quinol(self, namer):
        """★ Invariant 11. Denying `hydroquinone` unmasked `quinol`, another non-PIN
        for the same structure. `quinol` appears nowhere in the Blue Book. This test
        exists because the first deny looked like it worked and did not."""
        got = namer.name("Oc1ccc(O)cc1")
        assert got == "benzene-1,4-diol", got
        assert "quinol" not in got



class TestTheSuffixPromotionHalf:
    """``polycyclics.py`` had no ``'ol'`` in ``_SUFFIX_PRIORITY``, so a ring hydroxy
    could never be a suffix. These reach ``name_substituted_polycyclic`` directly (the
    retained-name table never intercepted them), so they are fixed by the promotion
    alone and prove that half independently of any deny row."""

    def test_anthracen_9_ol(self, namer):
        """Was `9-hydroxyanthracene`. BB:26822/:26824 print `9-anthrol` /
        `anthracen-9-ol (PIN)`. Records ZERO retained-name lookups (measured), so it is
        NOT a table defect and must not be given a deny row."""
        assert namer.name("Oc1c2ccccc2cc2ccccc12") == "anthracen-9-ol"

    def test_naphthalene_diol(self, namer):
        """Was `1,4-dihydroxynaphthalene`. Proves the promotion handles MULTIPLE
        locants, not just a mono case."""
        assert namer.name("Oc1ccc(O)c2ccccc12") == "naphthalene-1,4-diol"

    def test_amino_plus_ol_now_renders(self, namer):
        """Before the promotion this failed closed to a double prefix. `-ol` is senior
        to `-amine` (P-41), which is why the ol promotion is ordered FIRST — the amine
        promotion is guarded by `not suffix_groups`."""
        assert namer.name("Nc1ccc(O)c2ccccc12") == "1-aminonaphthalen-4-ol"

    def test_senior_suffix_keeps_hydroxy_as_a_prefix(self, namer):
        """Deny-by-default: with a senior suffix present the hydroxy must NOT be
        promoted. Correct before this change and must stay correct."""
        assert namer.name("OC(=O)c1ccc(O)c2ccccc12") == \
            "4-hydroxynaphthalene-1-carboxylic acid"


class TestDenyMechanismIsReached:
    """★ the contributor guide invariant 13, applied to the deny table itself.

    A ``pin: false`` row only suppresses a name on the surfaces the deny-set filters.
    ``indane`` proved this the hard way: it had a row AND still emitted ``indane``.
    """

    def test_indane_family_rows_are_marked_as_no_ops(self):
        """These three rows are retained as belt-and-braces but are NOT the mechanism.
        The note must say so, or a future session will trust them."""
        import json
        import pathlib
        d = json.loads(pathlib.Path(
            "src/orthonym/data/iupac_2013_pin_list.json").read_text())
        rows = {r["name"]: r for r in d["entries"]}
        for nm in ("indane", "indoline", "isoindoline"):
            assert nm in rows, nm
            assert rows[nm]["pin"] is False
            assert "NO-OP" in rows[nm]["note"], \
                f"{nm}: the note must record that this row does not fire"
            assert rows[nm]["citation"] == "P-54.4.3.2", \
                f"{nm}: P-54.4.3.2 (BB:24256) is the governing rule"

    def test_fused_heterocycle_table_now_holds_the_PINs(self):
        """v29 Phase C: the indane-family rename has LANDED.

        The first attempt was reverted because the ``name`` field doubles as the
        'spirobi' component (P-24.3.1); ``_name_spirobi_core`` now derives the hydro
        prefixes from the graph and hoists them outside the bracket, so the two
        consumers no longer conflict. See ``test_spirobi_hydro_hoisting.py``.

        Asserted as DATA and in BOTH directions: the retained non-PINs must be GONE,
        not merely joined by the PINs, or a stray duplicate row would keep emitting
        the non-preferred name (session invariant 13 -- presence in a table is not
        evidence the table is reached, and absence must be checked explicitly)."""
        from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
        names = {v.get("name") for v in FUSED_HETEROCYCLE_DATA.values()}
        assert "2,3-dihydro-1H-indene" in names
        assert "2,3-dihydro-1H-indole" in names
        assert "2,3-dihydro-1H-isoindole" in names
        assert "indane" not in names
        assert "indoline" not in names
        assert "isoindoline" not in names

    @pytest.mark.parametrize("name", [
        "1-naphthol", "2-naphthol", "hydroquinone", "quinol", "resorcinol",
        "pyrogallol", "thymol", "carvacrol", "isonicotinic acid",
    ])
    def test_deny_rows_exist_without_hc_override(self, name):
        """Omitting ``hc_override`` is what makes ONE row filter BOTH the hand-curated
        surface (``_PIN_DENY_HC``) and the OPSIN-import surface (``_is_promotable``).
        ``carvacrol``/``thymol`` live only on the import side and ``pyrogallol`` on
        both, so this is load-bearing, not stylistic."""
        import json
        import pathlib
        d = json.loads(pathlib.Path(
            "src/orthonym/data/iupac_2013_pin_list.json").read_text())
        rows = {r["name"]: r for r in d["entries"]}
        assert name in rows, f"{name} has no deny row"
        assert rows[name]["pin"] is False
        assert "hc_override" not in rows[name], \
            f"{name}: hc_override would stop this row filtering the hand-curated surface"


class TestSuffixPriorityOrder:
    """``_SUFFIX_PRIORITY`` ordering, asserted as DATA.

    ★ Why this test exists: mutation testing showed that deleting ``'ol'`` from
    ``polycyclics.py``'s ``_SUFFIX_PRIORITY`` breaks NO end-to-end test. The promotion
    is guarded by ``not suffix_groups``, so a promoted ``ol`` is always the only key and
    the ``next(iter(...))`` fallback selects it either way. The entry is a guard for a
    state that is currently unreachable but one change away — if a hydroxy ever arrives
    already flagged ``is_suffix``, ``suffix_groups`` could hold ``ol`` beside a senior
    suffix and the fallback would be free to pick the junior one.

    An untested unreachable guard is how dead code accumulates and how a wrong comment
    survives (this one had one). Pinning the order as data makes the guard mutable-and-
    caught without pretending an end-to-end witness exists.
    """

    def _priority(self):
        import inspect
        import re

        from orthonym.rules.polycyclics import name_substituted_polycyclic
        src = inspect.getsource(name_substituted_polycyclic)
        i = src.index("_SUFFIX_PRIORITY = [")
        body = src[i + len("_SUFFIX_PRIORITY = ["):src.index("]", i)]
        return re.findall(r"'([^']+)'", body)

    def test_ol_sits_between_carbaldehyde_and_amine(self):
        """P-41 seniority: ... > carbaldehyde > ol > amine."""
        pr = self._priority()
        assert "ol" in pr, f"the -ol suffix guard was removed: {pr}"
        assert "carbaldehyde" in pr and "amine" in pr, pr
        assert pr.index("carbaldehyde") < pr.index("ol") < pr.index("amine"), pr

    def test_acids_outrank_ol(self):
        pr = self._priority()
        for senior in ("carboxylic acid", "sulfonic acid", "carboxamide",
                       "carbonitrile"):
            assert pr.index(senior) < pr.index("ol"), f"{senior} must outrank -ol: {pr}"


class TestUnchangedControls:
    """Names that were ALREADY correct. A deny-set edit is exactly the kind of change
    that moves neighbours, so the whole neighbourhood is pinned."""

    @pytest.mark.parametrize("smiles,expected", [
        ("Oc1ccccc1O", "benzene-1,2-diol"),          # pyrocatechol, already denied
        ("Cc1cc(C)cc(C)c1", "1,3,5-trimethylbenzene"),  # mesitylene
        ("COc1ccccc1", "methoxybenzene"),            # anisole
        ("Oc1ccccc1", "phenol"),                     # a genuine retained PIN
        ("OC(=O)c1cccnc1", "pyridine-3-carboxylic acid"),   # nicotinic, already denied
        ("Nc1ccccc1C(=O)O", "2-aminobenzoic acid"),  # anthranilic, already denied
        ("OCC(O)CO", "propane-1,2,3-triol"),         # glycerol, already denied
        ("Nc1cccc2ccccc12", "naphthalen-1-amine"),   # the amine promotion still works
        ("OC(=O)c1cccc2ccccc12", "naphthalene-1-carboxylic acid"),
        ("Cc1cccc2ccccc12", "1-methylnaphthalene"),
        # P-54.4.3.2's other list-mates and near neighbours: all already correct,
        # which is what sized this cluster at exactly three stale rows
        ("C1CCc2ccccc2O1", "3,4-dihydro-2H-1-benzopyran"),
        ("C1CCc2ccccc2N1", "1,2,3,4-tetrahydroquinoline"),
        ("C1Cc2ccccc2CN1", "1,2,3,4-tetrahydroisoquinoline"),
        ("C1Cc2ccccc2S1", "2,3-dihydro-1-benzothiophene"),
        ("C1CCC2CCCC2C1", "octahydro-1H-indene"),
    ])
    def test_control_unchanged(self, namer, smiles, expected):
        assert namer.name(smiles) == expected


class TestKnownAdjacentDefect:
    @pytest.mark.xfail(strict=True, reason=(
        "The `-ol` promotion shipped here covers the POLYCYCLIC path "
        "(rules/polycyclics.py). The fused-heterocycle path has the same defect and is "
        "not yet fixed: a ring hydroxy stays a prefix. Recorded rather than left "
        "unlogged; needs the same promotion where the fused_heterocycles parent is "
        "assembled."))
    def test_dihydroindenol_should_use_the_ol_suffix(self, namer):
        assert namer.name("OC1Cc2ccccc2C1") == "2,3-dihydro-1H-inden-2-ol"


class TestIndaneFamilyUnblocked:
    """indane / indoline / isoindoline -- derived, attempted, reverted, then SHIPPED.

    **P-54.4.3.2** (``:24256``) names all three verbatim as non-preferred: *"The retained
    names for the partially saturated heterocycles, indane, indoline, isoindoline, and
    chromane, isochromane and their chalcogen analogues are not used as preferred IUPAC
    names…"*, and ``:16988``/``:16992``/``:16999`` print the PINs. The diagnosis was never
    in doubt; the BLOCKER was.

    ★ THE FIRST RENAME WAS REJECTED BY THE GATE -- 1 protect + 3 target regressions.
    Root cause: ``fused_heterocycles``'s ``name`` field feeds TWO consumers. Standalone
    naming wants the saturated PIN, but ``rules/spiro.py:_name_spirobi_core`` embedded it
    as the SPIRO COMPONENT, and **P-24.3.1** (``:10146``) requires the bracket to hold the
    *component ring system*, with hydrogen cited OUTSIDE it. Every worked example is
    mancude -- ``1,1'-spirobi[indene] (PIN)`` (``:10164``),
    ``1H,1'H-2,2'-spirobi[naphthalene] (PIN)`` (``:10158``). The bare rename produced
    ``1,2'-spirobi[2,3-dihydro-1H-indene]``.

    ⚠ AND THE SHORTCUT WOULD HAVE BEEN WORSE. Putting the mancude component in the
    bracket without hoisting the hydro prefixes gives ``1,2'-spirobi[1H-indene]`` -- the
    UNSATURATED molecule. That is a wrong STRUCTURE, not a wrong spelling (session
    invariant 11). It was reverted rather than shipped.

    ✅ UNBLOCKED (v29 Phase C) by deriving the saturation from the GRAPH instead:
    ``_name_spirobi_core`` computes the maximum noncumulative double-bond assignment over
    the ASSEMBLED skeleton (the spiro atom excluded, since its four single ring bonds make
    it sp3 by construction) and hoists the hydro prefixes plus any indicated hydrogen in
    front of the spiro locants, per P-24.3.2 (``:10152``) and the ``:46336`` template. The
    derivation and its five Blue-Book-PIN validations live in
    ``tests/unit/rules/test_spirobi_hydro_hoisting.py``.

    ORDERED FIX: (1) teach ``_name_spirobi_core`` to hoist hydro prefixes onto the
    assembly with primed/unprimed locants; (2) rename the three table rows; (3) correct
    the two gold rows that enforce the non-PIN (``gold_pins`` protect ``C1Cc2ccccc2C1``
    → ``indane``, and the ``5-methylindoline`` target).
    """

    @pytest.mark.parametrize("smiles,pin", [
        ("C1Cc2ccccc2C1", "2,3-dihydro-1H-indene"),
        ("C1Cc2ccccc2N1", "2,3-dihydro-1H-indole"),
        ("C1NCc2ccccc21", "2,3-dihydro-1H-isoindole"),
        ("Cc1ccc2c(c1)CCC2", "5-methyl-2,3-dihydro-1H-indene"),
        ("Cc1ccc2c(c1)CCN2", "5-methyl-2,3-dihydro-1H-indole"),
    ])
    def test_indane_family_pin(self, namer, smiles, pin):
        """v29 Phase C: no longer xfail -- the spiro blocker below is fixed."""
        assert namer.name(smiles) == pin

    def test_the_spiro_form_that_used_to_block_it(self, namer):
        """The row the first rename attempt regressed.

        Kept as the tripwire it always was: hydro must never appear INSIDE the
        spirobi bracket (P-24.3.1 / P-24.3.2). Now also asserts the WHOLE name, so
        a regression to the retained ``spirobi[indane]`` form -- or to the
        unsaturated ``spirobi[1H-indene]`` shortcut -- fails here too.
        """
        got = namer.name("C1Cc2ccccc2C11Cc2ccccc2C1")
        assert "spirobi" in got, got
        assert "2,3-dihydro-1H-indene]" not in got, \
            f"hydro must not be inside the spiro bracket (P-24.3.1): {got!r}"
        assert "spirobi[indane]" not in got, \
            f"'indane' is not a PIN (P-54.4.3.2, BB:24256): {got!r}"
        assert got == "1',2,3,3'-tetrahydro-1,2'-spirobi[indene]", got
