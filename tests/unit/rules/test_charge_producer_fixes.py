""" a phase -- three narrow 0-wrong charge-producer fixes.

a trace: internal notes (the classifier-veto
premise was REFUTED; these are the 3 measured WRONG-guess producers instead,
each independently RT-validated). Plan: docs/superpowers/plans/
2026-08-12-phase3-correctness-tail-hardening.md Sec 3A.

3A-a: rules/salts.py -- is_salt/name_salt must require the disconnected
      fragment set to be net-charge-balanced (a neutral salt, by definition,
      always is). A charge-imbalanced assembly (e.g. a bare metal cation +
      one anion, net nonzero) is an out-of-scope ionic/coordination complex,
      not a salt -- decline rather than guess balanced stoichiometry.
3A-b: assembly/substituent_enumerator.py + assembly/substituent_naming.py --
      a compound-substituent fragment carrying a REAL (non-internal) formal
      charge must route through the existing charge-aware cation_to_prefix
      primitive when the cation is the fragment's own attach atom,
      and fail closed (never guess) otherwise.
3A-c: rules/ions.py::name_anion -- the single-anion fall-through must never
      accept _try_neutralize_and_name's bare NEUTRAL-skeleton name as the
      anion's final name (that always drops the charge and denotes a
      different molecule); decline instead.
"""
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import name_compound
from orthonym.validation.opsin_roundtrip import opsin_parse


def _full_rt(smiles: str, name: str) -> bool:
    """OPSIN parse `name` -> canonical InChIKey; compare to `smiles`'s."""
    if not name:
        return False
    o = opsin_parse(name)
    if not o:
        return False
    m1 = Chem.MolFromSmiles(smiles)
    m2 = Chem.MolFromSmiles(o)
    if m1 is None or m2 is None:
        return False
    return inchi.MolToInchiKey(m1) == inchi.MolToInchiKey(m2)


# ============================================================================
# 3A-a: charge-balance guard in rules/salts.py
# ============================================================================


class TestSaltChargeBalance:
    def test_charge_imbalanced_pd_acetate_never_ships_wrong_name(self):
        """CC(=O)[O-].[Pd+2] is net +1 (one acetate anion, one Pd2+ cation) --
        NOT a neutral salt. name_salt must never ship the balanced-implying
        'palladium(II) acetate' (which OPSIN round-trips to the DIACETATE
        Pd(OAc)2, a different molecule than the 1:1 input)."""
        smi = "CC(=O)[O-].[Pd+2]"
        name = name_compound(smi)
        assert name != "palladium(II) acetate", name
        # Whatever IS emitted (an honest abstain / descriptive fallback) must
        # never claim to be this different, balanced molecule.
        if name and "acetate" in name:
            assert _full_rt(smi, name), name

    def test_is_salt_declines_charge_imbalanced_fragment_set(self):
        from orthonym.rules.salts import is_salt
        mol = Chem.MolFromSmiles("CC(=O)[O-].[Pd+2]")
        assert is_salt(mol) is False

    def test_name_salt_declines_charge_imbalanced_fragment_set(self):
        from orthonym.rules.salts import name_salt
        mol = Chem.MolFromSmiles("CC(=O)[O-].[Pd+2]")
        assert name_salt(mol) == ""

    def test_regression_quaternary_ammonium_iodide_unchanged(self):
        smi = "C[N+](C)(C)C.[I-]"
        name = name_compound(smi)
        assert name == "N,N,N-trimethylmethanaminium iodide", name
        assert _full_rt(smi, name), name

    def test_regression_sodium_acetate_unchanged(self):
        assert name_compound("CC(=O)[O-].[Na+]") == "sodium acetate"

    def test_regression_potassium_chloride_unchanged(self):
        assert name_compound("[K+].[Cl-]") == "potassium chloride"

    def test_regression_calcium_diacetate_unchanged(self):
        # a genuine charge-BALANCED divalent salt (Ca2+ + 2x acetate-, net 0)
        # must still name -- the guard must not reject legitimate stoichiometry.
        name = name_compound("[Ca+2].[O-]C(C)=O.[O-]C(C)=O")
        assert "calcium" in name and "acetate" in name, name

    def test_regression_is_salt_positive_still_true(self):
        from orthonym.rules.salts import is_salt
        assert is_salt(Chem.MolFromSmiles("[Na+].[Cl-]")) is True

    def test_regression_is_salt_negative_cases_unchanged(self):
        from orthonym.rules.salts import is_salt
        assert is_salt(Chem.MolFromSmiles("CCO")) is False
        assert is_salt(Chem.MolFromSmiles("[Na+]")) is False
        assert is_salt(Chem.MolFromSmiles("[NH3+]CC([O-])=O")) is False


# ============================================================================
# 3A-b: wire cation_to_prefix into the compound-substituent recursion
# ============================================================================


class TestCationSubstituentPrefix:
    def test_choline_phosphate_lipid_never_ships_wrong_substituent_guess(self):
        """A choline-phosphate-ester compound substituent (-O-P(=O)(OH)-O-CH2-
        CH2-N+(CH3)3) used to leak through the charge-blind recursion (a trace-
        traced call site: substituent_naming.py::name_substituent_fragment
        Step 5, reached via perception/chains.py's substituent-cost
        evaluation) as '(2-phosphonooxy-N,N,N-trimethylethan-1-aminium)yl' --
        an OPSIN-unparseable guess. That specific leak must be gone (RT-
        confirmed via a direct trace on parent_to_prefix's call args before/
        after this fix -- internal notes).

        NOTE: this molecule also has a SEPARATE, unrelated wrong-candidate
        source (a different decomposition names the choline head as if it
        were the WHOLE molecule, 'hydroxy-N,N,N-trimethylethanaminium' --
        an atom-drop bug, not a charge-blind-prefix bug) that catches
        via the OPSIN backstop in most environments but not universally; it
        is OUT OF SCOPE for this narrowly-scoped 3A-b fix (a different
        subsystem produces it) and is reported separately, not asserted on
        here.
        """
        smi = "CCCCCCCCCCCCCCCCC(COP(=O)(O)OCC[N+](C)(C)C)O"
        name = name_compound(smi)
        assert name is not None
        assert "aminiumyl" not in name, name
        assert "phosphonooxy-N,N,N-trimethylethan" not in name, name

    def test_direct_attach_cation_prefix_routes_through_cation_to_prefix(self):
        """A fragment whose OWN attach atom IS the cationic centre (the
        direct-attachment shape cation_to_prefix is built for) must produce
        an RT-correct, OPSIN-valid prefix, never a guess. Built via the
        primitive directly (same call the fixed call sites now make)."""
        from orthonym.assembly.substituent_naming import cation_to_prefix
        # trimethylammonio-methyl fragment: C[N+](C)(C)C attached via the N.
        mol = Chem.MolFromSmiles("C[N+](C)(C)CC")
        n_idx = [a.GetIdx() for a in mol.GetAtoms()
                 if a.GetSymbol() == "N" and a.GetFormalCharge() > 0][0]
        n_atom = mol.GetAtomWithIdx(n_idx)
        # the ethyl carbon is the "parent" side; everything else on N is a
        # substituent of the onium prefix.
        ethyl_c = [nb.GetIdx() for nb in n_atom.GetNeighbors()
                   if mol.GetAtomWithIdx(nb.GetIdx()).GetSymbol() == "C"
                   and mol.GetAtomWithIdx(nb.GetIdx()).GetDegree() == 2][0]
        prefix = cation_to_prefix(mol, n_idx, ethyl_c)
        assert prefix == "trimethylazaniumyl", prefix

    def test_name_substituent_fragment_names_pendant_onium_branch(self):
        """ a phase enabler: the "no nested-substituent
        composer exists for that shape yet" premise this test used to assert
        is no longer true -- `_name_polyfunctional_acyclic_substituent`
        (assembly/substituent_naming.py, Pass 1d) now composes a pendant
        onium branch as a locanted detachable prefix, reusing the existing
        structured `cation_to_prefix` primitive rather than falling through
        to the charge-blind `parent_to_prefix` guess this class of test was
        written to forbid. The guess this test guarded against denoted a
        DIFFERENT molecule (an OPSIN-unparseable '...aminiumyl' suffix
        surgery, or a silently-mis-anchored attachment); the value asserted
        below is RT-verified as the SAME molecule instead:

            OPSIN.parse("2-(trimethylazaniumyl)ethoxybenzene")
                -> C[N+](CCOC1=CC=CC=C1)(C)C
                -> canonical C[N+](C)(C)CCOc1ccccc1

        i.e. benzene + this exact -O-CH2-CH2-N+(CH3)3 fragment, confirmed via
        OPSIN (an independent implementation, not the code under test).
        `test_out_of_scope_remote_charge_still_fails_closed` below keeps this
        test's original guard alive for a charge shape the new pass does NOT
        cover (a carbocation branch, not an onium element)."""
        from orthonym.assembly.substituent_naming import name_substituent_fragment
        # -O-CH2-CH2-N+(CH3)3 fragment attached to a parent via the ether O;
        # the charge is 2 atoms away from attach_idx (the O).
        mol = Chem.MolFromSmiles("CCCC(OCC[N+](C)(C)C)CC")
        o_idx = [a.GetIdx() for a in mol.GetAtoms()
                 if a.GetSymbol() == "O"][0]
        parent_c = [nb.GetIdx() for nb in mol.GetAtomWithIdx(o_idx).GetNeighbors()
                    if not (nb.GetSymbol() == "C" and nb.GetDegree() == 2
                            and any(nn.GetSymbol() == "C" for nn in nb.GetNeighbors()
                                    if nn.GetIdx() != o_idx))]
        # sub_atoms = every fragment atom reachable from O without crossing
        # back into the parent chain.
        visited = {parent_c[0]} if parent_c else set()
        stack = [o_idx]
        frag = []
        while stack:
            a = stack.pop()
            if a in visited:
                continue
            visited.add(a)
            frag.append(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                if nb.GetIdx() not in visited:
                    stack.append(nb.GetIdx())
        result = name_substituent_fragment(mol, frag, o_idx, parent_c)
        assert result == "2-(trimethylazaniumyl)ethoxy", result

    def test_out_of_scope_remote_charge_still_fails_closed(self):
        """A charge shape the new Pass 1d does NOT cover -- a carbocation
        branch (carbon is not in `_ONIUM_PREFIX_STEM`, so `cation_to_prefix`
        declines it) -- must still fail closed (None), preserving this test
        class's original guard: never fall through to the charge-blind
        `parent_to_prefix` guess."""
        from orthonym.assembly.substituent_naming import (
            _name_polyfunctional_acyclic_substituent,
        )
        # -CH2-CH(OH)-CH2-[CH2+]: the hydroxy forces entry into the
        # polyfunctional path; the terminal carbocation is out of scope.
        mol = Chem.MolFromSmiles("C(O)CC[CH2+]")
        attach = 0
        frag = list(a.GetIdx() for a in mol.GetAtoms())
        result = _name_polyfunctional_acyclic_substituent(mol, frag, attach, set())
        assert result is None, result

    def test_regression_neutral_compound_substituent_unchanged(self):
        """A neutral (uncharged) compound substituent must be completely
        unaffected -- the new charge guard is a no-op for it."""
        # 2-hydroxyethyl-bearing molecule: neutral compound substituent path.
        smi = "OCCC1=CC=CC=C1O"
        name = name_compound(smi)
        assert name is not None
        assert "unknown" not in name


# ============================================================================
# 3A-c: charge-aware fall-through in rules/ions.py::name_anion
# ============================================================================


class TestAnionChargeAwareFallthrough:
    def test_dihydrogenborate_anion_never_ships_boric_acid(self):
        """[O-]B(O)O (the dihydrogenborate anion) must never ship 'boric
        acid' -- that name denotes the NEUTRAL B(OH)3, a different molecule
        (OPSIN round-trips 'boric acid' to OB(O)O, not the charged input)."""
        smi = "[O-]B(O)O"
        name = name_compound(smi)
        assert name != "boric acid", name
        if name and "unknown" not in name and "not supported" not in name:
            assert _full_rt(smi, name), name

    def test_name_anion_declines_rather_than_drop_charge(self):
        from orthonym.rules.ions import name_anion
        mol = Chem.MolFromSmiles("[O-]B(O)O")
        assert name_anion(mol) == ""

    def test_regression_neutral_boric_acid_unchanged(self):
        assert name_compound("OB(O)O") == "boric acid"

    def test_regression_common_anions_unchanged(self):
        assert name_compound("CC(=O)[O-]") == "acetate"
        assert name_compound("[Cl-]") == "chloride"
        assert name_compound("CO[O-]") is not None

    def test_regression_oxoacid_anion_path_unchanged(self):
        # sulfonate/sulfinate/phosphonate branch is untouched by 3A-c.
        name = name_compound("CS(=O)(=O)[O-]")
        assert name is not None and "unknown" not in name
