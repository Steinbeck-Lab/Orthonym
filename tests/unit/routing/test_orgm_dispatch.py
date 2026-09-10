"""a phase routing tests for ORGANOMETALLIC StoutClass member.

Mirrors tests/unit/routing/test_dispatch_table.py — the LOCKED analog for
CFR registration + predicate purity + cascade-continuation patterns
(PATTERNS lines 1037-1064).

internal notes + ENFORCEMENT: ORGM CFR entry at priority 50 (intercepts
BEFORE SALT@100); side_effect_inventory=; _is_organometallic is
predicate-pure (no mol/features mutation).

NEVER uses @pytest.mark.xfail (internal notes) — honest-fail-on-data.
"""
import inspect
import pytest
from rdkit import Chem

from orthonym.routing.dispatch_table import (
    DISPATCH_TABLE, StoutClass,
    _is_organometallic, _handle_organometallic,
)


@pytest.mark.unit
class TestOrgmDispatch:
    """a phase routing-layer tests for ORGM."""

    def test_organometallic_member_exists(self):
        """internal notes: StoutClass.ORGANOMETALLIC is uncommented + registered."""
        assert hasattr(StoutClass, 'ORGANOMETALLIC')
        assert StoutClass.ORGANOMETALLIC.value == 'organometallic'

    def test_organometallic_in_dispatch_table(self):
        """ inheritance: every enum member must be in DISPATCH_TABLE."""
        assert StoutClass.ORGANOMETALLIC in DISPATCH_TABLE

    def test_orgm_priority_is_50(self):
        """CONTEXT D-02 + RESEARCH §4.1: priority 50 (LOWER than SALT@100)."""
        assert DISPATCH_TABLE[StoutClass.ORGANOMETALLIC].priority == 50

    def test_orgm_tier_is_1(self):
        """Tier 1 (charged-species + dot-disconnected category per audit § 4.2)."""
        assert DISPATCH_TABLE[StoutClass.ORGANOMETALLIC].tier == 1

    def test_orgm_side_effect_inventory_is_empty(self):
        """internal notes HARD INVARIANT: side_effect_inventory MUST be ."""
        entry = DISPATCH_TABLE[StoutClass.ORGANOMETALLIC]
        assert entry.side_effect_inventory == ()

    def test_orgm_predicate_callable(self):
        entry = DISPATCH_TABLE[StoutClass.ORGANOMETALLIC]
        assert callable(entry.predicate)

    def test_orgm_handler_callable(self):
        entry = DISPATCH_TABLE[StoutClass.ORGANOMETALLIC]
        assert callable(entry.handler)

    def test_orgm_iupac_section_contains_p69(self):
        """iupac_section cites or Salzer 1999."""
        entry = DISPATCH_TABLE[StoutClass.ORGANOMETALLIC]
        assert (
            'P-69' in entry.iupac_section
            or 'Salzer' in entry.iupac_section
        )

    def test_orgm_priority_below_salt(self):
        """internal notes: ORGM@50 fires BEFORE SALT@100."""
        orgm_prio = DISPATCH_TABLE[StoutClass.ORGANOMETALLIC].priority
        salt_prio = DISPATCH_TABLE[StoutClass.SALT].priority
        assert orgm_prio < salt_prio

    def test_orgm_predicate_mol_none_returns_false(self):
        """Defensive guard: _is_organometallic(None,...) returns False."""
        assert _is_organometallic(None, '', '') is False

    def test_orgm_predicate_purity_mol_unchanged(self):
        """internal notes: _is_organometallic mutates nothing."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        atom_count_pre = mol.GetNumAtoms()
        bond_count_pre = mol.GetNumBonds()
        _is_organometallic(mol, '', Chem.MolToSmiles(mol))
        assert mol.GetNumAtoms() == atom_count_pre
        assert mol.GetNumBonds() == bond_count_pre

    def test_orgm_predicate_true_for_ferrocene(self):
        """Ferrocene is an organometallic."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        assert _is_organometallic(mol, '', Chem.MolToSmiles(mol)) is True

    def test_orgm_predicate_true_for_methyllithium(self):
        """MeLi is a σ-bonded organometallic."""
        mol = Chem.MolFromSmiles('[Li][CH3]')
        assert _is_organometallic(mol, '', Chem.MolToSmiles(mol)) is True

    def test_orgm_predicate_false_for_ethanol(self):
        """Ethanol is NOT an organometallic."""
        mol = Chem.MolFromSmiles('CCO')
        assert _is_organometallic(mol, '', Chem.MolToSmiles(mol)) is False

    def test_orgm_predicate_false_for_acetic_acid(self):
        """Risk R-02: C=O group must NOT trigger ORGM predicate."""
        mol = Chem.MolFromSmiles('CC(=O)O')
        assert _is_organometallic(mol, '', Chem.MolToSmiles(mol)) is False

    def test_orgm_handler_returns_name_for_ferrocene(self):
        """Handler returns the byte-identical PIN string for ferrocene."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = _handle_organometallic(
            mol, '', Chem.MolToSmiles(mol), style='pin'
        )
        assert result == 'ferrocene'

    def test_orgm_handler_returns_none_for_ethanol(self):
        """Cascade-continuation: non-ORGM compound returns None."""
        mol = Chem.MolFromSmiles('CCO')
        result = _handle_organometallic(
            mol, '', Chem.MolToSmiles(mol), style='pin'
        )
        assert result is None

    def test_orgm_handler_systematic_style(self):
        """Handler respects style='systematic'."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = _handle_organometallic(
            mol, '', Chem.MolToSmiles(mol), style='systematic'
        )
        assert result == 'bis(η⁵-cyclopentadienyl)iron(II)'

    def test_orgm_below_other_cfr_priorities(self):
        """ORGM@50 is lower (fires earlier) than every other CFR entry.

        Per internal notes: ORGM intercepts BEFORE SALT@100 (lowest pre-Phase-161
        priority), so its priority is below all existing CFR entries.
        """
        orgm_prio = DISPATCH_TABLE[StoutClass.ORGANOMETALLIC].priority
        for cls, entry in DISPATCH_TABLE.items():
            if cls == StoutClass.ORGANOMETALLIC:
                continue
            # + a phase/7/10: a small set of parent-hydride /
            # inorganic recognizers deliberately fire BEFORE ORGM@50 so a free
            # inorganic oxoacid (silicic acid O[Si](O)(O)O), a mononuclear hydride
            # (SF6, trimethylarsane), a chalcogen chain (trisulfane), a polyazane
            # (hydrazine) or a di-nuclear Group-14/15 catenated hydride
            # (germylstibane) is named substitutively, not claimed as an element
            # organometallic. These are the intentional exceptions to "ORGM is the
            # global priority minimum" (INORGANIC_ACID@40, MONONUCLEAR_HYDRIDE@45,
            # CHALCOGEN_CHAIN@46, POLYAZANE@47, CATENATED_HYDRIDE@47.5,
            # DINUCLEAR_HYDRIDE@48, KETENE@49 (Wave-2 completion: disiloxane,
            # C=C=O -> ethenone), RING_CHALCOGEN_OXIDE@49.5 (Wave-2 completion
            # B4: dibenzothiophene 5-oxide), HETEROIMINE@48.7 (W2E-P1FG:
            # CP=N -> 1-methylphosphanimine), HYDRO_FUSED_PEROXOL@49.6 (W2E-P1FG),
            # THIOIMIDE@49.7 (W2E-D3: CC(=S)NC(C)=S ->
            # N-(ethanethioyl)ethanethioamide) — all < 50, all mutually decline).
            # (Wave-3 additions to the sub-50 interceptor set: FREE_HOMONUCLEAR_
            # G14_HYDRIDE@47.7 (disilane/digermene), CYCLIC_POLYESTER@49.8
            # (glycolide -> 1,4-dioxane-2,5-dione), and W3-P13 POLYCHALCOGEN_OXIDE
            # @46.5 (...disulfane-1,2-dione) + LAMBDA_SULFANE_IMINE_OXIDE@48.75
            # (S,S-diethyl-N-phenyl-lambda4-sulfanimine) — all < 50, all mutually
            # decline. Further wave-3 sub-50 interceptors:
            # HETEROCHALCOGEN_ABA@47.55 (pure-chalcogen a[ba]n parent hydride,
            # dithioxane), HOMONUCLEAR_PNICTOGEN_CHAIN@47.6,
            # PNICTOGEN_CARBOXYLIC_ACID@47.65, ACYL_CHALCOGENCHAIN_PSEUDOKETONE
            # @48.65 — all element-hydride-family recognizers with predicates
            # disjoint from ORGM (they mutually decline); all < 50.)
            if cls in (StoutClass.INORGANIC_ACID, StoutClass.MONONUCLEAR_HYDRIDE,
                       StoutClass.CHALCOGEN_CHAIN, StoutClass.POLYAZANE,
                       StoutClass.CATENATED_HYDRIDE, StoutClass.DINUCLEAR_HYDRIDE,
                       StoutClass.KETENE, StoutClass.RING_CHALCOGEN_OXIDE,
                       StoutClass.AZINIC_DERIVATIVE, StoutClass.HETERONE,
                       StoutClass.SULFINE, StoutClass.PSEUDOKETONE_HETERO,
                       StoutClass.HETEROIMINE, StoutClass.HYDRO_FUSED_PEROXOL,
                       StoutClass.THIOIMIDE,
                       StoutClass.FREE_HOMONUCLEAR_G14_HYDRIDE,
                       StoutClass.CYCLIC_POLYESTER,
                       StoutClass.POLYCHALCOGEN_OXIDE,
                       StoutClass.LAMBDA_SULFANE_IMINE_OXIDE,
                       StoutClass.HETEROCHALCOGEN_ABA,
                       StoutClass.HOMONUCLEAR_PNICTOGEN_CHAIN,
                       StoutClass.PNICTOGEN_CARBOXYLIC_ACID,
                       StoutClass.ACYL_CHALCOGENCHAIN_PSEUDOKETONE):
                continue
            assert orgm_prio < entry.priority, (
                f"ORGM@{orgm_prio} not lower than {cls.name}@{entry.priority}; "
                f"CONTEXT D-02 violation."
            )

    def test_orgm_predicate_inspect_no_seniority_import(self):
        """internal notes: _is_organometallic does NOT trigger rules.seniority import."""
        src = inspect.getsource(_is_organometallic)
        assert 'seniority' not in src

    def test_orgm_handler_inspect_signature(self):
        """Handler accepts (mol, smiles, canonical_smiles, features=None, style='pin')."""
        sig = inspect.signature(_handle_organometallic)
        params = sig.parameters
        assert 'mol' in params
        assert 'smiles' in params
        assert 'canonical_smiles' in params

    def test_orgm_predicate_pure_under_repeat_invocation(self):
        """Repeated calls to predicate produce identical results (no hidden state)."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        smi = Chem.MolToSmiles(mol)
        r1 = _is_organometallic(mol, '', smi)
        r2 = _is_organometallic(mol, '', smi)
        r3 = _is_organometallic(mol, '', smi)
        assert r1 == r2 == r3 == True  # noqa: E712 — explicit equality

    def test_orgm_handler_pure_under_repeat_invocation(self):
        """Repeated handler calls produce identical results."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        smi = Chem.MolToSmiles(mol)
        r1 = _handle_organometallic(mol, '', smi, style='pin')
        r2 = _handle_organometallic(mol, '', smi, style='pin')
        assert r1 == r2 == 'ferrocene'
