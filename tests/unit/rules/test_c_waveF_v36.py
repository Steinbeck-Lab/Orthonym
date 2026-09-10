""" Milestone-C Wave F -- spelling quality + skeletal-namer correctness.

Task 1 (spelling quality, core-namer item 3): the best-effort recursive
substituent namer must PREFER the substitutive/retained spelling
(``methoxy`` / ``pyridin-2-yl`` / ``sulfinyl``) over the skeletal-replacement
('a') form (``1-oxaethyl`` / ``1-azacyclohexa-1,3,5-trien-2-yl`` /
``1-oxo-1-thiaethyl``) whenever the substitutive form can be built and round-
trips. Both spellings round-trip, so this is QUALITY-only (not 0-wrong).

Task 2 (correctness): the 2,2,2-trifluoroethoxy fragment ``-OCC(F)(F)F`` must
name as ``2,2,2-trifluoroethoxy`` (or abstain), never the WRONG-constitution
``3,3-difluoro-1-oxabutan-1-yl`` the skeletal namer used to emit.

Naming is deterministic and JVM-free; the round-trip assertions call OPSIN via
``opsin_parse`` (a JVM). Targeted-file run only (whole-suite OPSIN deadlock).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.metrics.provenance import best_effort_ctx


PANTOPRAZOLE_DIF = "COc1ccnc(CS(=O)c2[nH]c3ccc(OC(F)F)cc3n2)c1OC"
PANTOPRAZOLE_TRIF = "COc1ccnc(CS(=O)c2nc3ccc(OC(F)(F)F)cc3[nH]2)c1OC"


def _best_effort_namer():
    from orthonym.namer import Orthonym
    return Orthonym(style="pin", general_fallback=True,
                     general_fallback_unverified=True,
                     allow_aromatic_general=True)


def _inchikey(smiles):
    m = Chem.MolFromSmiles(smiles)
    return inchi.MolToInchiKey(m) if m else None


def _rt_ok(name, smiles):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    got = opsin_parse(name)
    if not got:
        return False
    m = Chem.MolFromSmiles(got)
    return bool(m) and inchi.MolToInchiKey(m) == _inchikey(smiles)


# --------------------------------------------------------------------------
# Task 1 -- substitutive/retained over 'a'-replacement
# --------------------------------------------------------------------------

# The C2 substituent of pantoprazole(diF): -S(=O)-CH2-[3,4-dimethoxypyridine].
# Fragment-side attach atom is the sulfinyl S (idx 8).
_PANTO_SUB_ATOMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 23, 24, 25]
_PANTO_SUB_ATTACH = 8


class TestSubstitutiveOverAReplacement:
    def test_sulfinyl_rooted_compound_substituent_is_substitutive(self):
        """The -S(=O)-CH2-Ar substituent names substitutively, not by
        skeletal replacement."""
        from orthonym.assembly.substituent_enumerator import name_substituent
        mol = Chem.MolFromSmiles(PANTOPRAZOLE_DIF)
        tok = best_effort_ctx.set(True)
        try:
            name = name_substituent(mol, _PANTO_SUB_ATOMS, _PANTO_SUB_ATTACH,
                                    allow_mancude=True)
        finally:
            best_effort_ctx.reset(tok)
        assert name, "substituent namer abstained"
        # substitutive/retained spellings present
        assert "sulfinyl" in name
        assert "methoxy" in name
        assert "pyridin" in name
        # 'a'-replacement spellings gone
        assert "oxaethyl" not in name
        assert "thiaethyl" not in name
        assert "azacyclohexa" not in name

    @pytest.mark.parametrize("smiles", [PANTOPRAZOLE_DIF, PANTOPRAZOLE_TRIF])
    def test_pantoprazole_best_effort_substitutive_and_roundtrips(self, smiles):
        namer = _best_effort_namer()
        name = namer.name(smiles)
        assert name and "unknown" not in name.lower(), \
            f"abstained/sentinel: {name!r}"
        assert "pyridin" in name and "methoxy" in name and "sulfinyl" in name, \
            f"not substitutive: {name!r}"
        assert "azacyclohexa" not in name and "oxaethyl" not in name \
            and "thiaethyl" not in name, f"still 'a'-replacement: {name!r}"
        assert _rt_ok(name, smiles), f"round-trip failed: {name!r}"


# --------------------------------------------------------------------------
# Task 2 -- 2,2,2-trifluoroethoxy miscount (WRONG constitution)
# --------------------------------------------------------------------------

# (2,2,2-trifluoroethoxy)benzene: F-C(F)(F)-CH2-O-benzene.
# The -O-CH2-CF3 substituent fragment is atoms {0,1,2,3,4,5}, attach O = 5.
_TFEO = "FC(F)(F)COc1ccccc1"
_TFEO_FRAG = [5, 4, 1, 0, 2, 3]
_TFEO_ATTACH = 5
LANSOPRAZOLE = "Cc1ccnc(CS(=O)c2[nH]c3ccccc3n2)c1OCC(F)(F)F"


class TestTrifluoroethoxyMiscount:
    def test_backbone_excludes_halogen(self):
        """The replacement-chain backbone must not step onto a monovalent
        halogen -- the miscount root cause (O-C-C-F -> '1-oxabutane')."""
        from orthonym.rules.terminal_fragment import (
            _backbone_from, _canonical_ranks)
        mol = Chem.MolFromSmiles(_TFEO)
        ranks = _canonical_ranks(mol)
        bb = _backbone_from(mol, set(_TFEO_FRAG), _TFEO_ATTACH, ranks,
                            stop_at_ring=False)
        syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in bb]
        assert syms == ["O", "C", "C"], f"backbone walked into a halogen: {syms}"

    def test_terminal_fragment_correct_constitution(self):
        """terminal_fragment now names -O-CH2-CF3 with the RIGHT constitution
        (3 F over a 3-atom O-C-C chain), not the old 2-F/4-atom miscount."""
        from orthonym.rules.terminal_fragment import terminal_fragment_name
        mol = Chem.MolFromSmiles(_TFEO)
        r = terminal_fragment_name(mol, set(_TFEO_FRAG), _TFEO_ATTACH)
        assert r is not None and r.name
        assert "trifluoro" in r.name
        assert "difluoro" not in r.name and "oxabutan" not in r.name, \
            f"still miscounting: {r.name!r}"
        assert _rt_ok(f"({r.name})benzene", _TFEO)

    def test_fragment_prefers_trifluoroethoxy(self):
        from orthonym.assembly.substituent_enumerator import name_substituent
        mol = Chem.MolFromSmiles(_TFEO)
        tok = best_effort_ctx.set(True)
        try:
            name = name_substituent(mol, _TFEO_FRAG, _TFEO_ATTACH,
                                    allow_mancude=True)
        finally:
            best_effort_ctx.reset(tok)
        assert name == "2,2,2-trifluoroethoxy"
        assert _rt_ok(f"({name})benzene", _TFEO)

    def test_lansoprazole_roundtrips_with_trifluoroethoxy(self):
        namer = _best_effort_namer()
        name = namer.name(LANSOPRAZOLE)
        assert name and "unknown" not in name.lower(), \
            f"abstained/sentinel: {name!r}"
        assert "2,2,2-trifluoroethoxy" in name, f"missing correct chain: {name!r}"
        assert "oxabutan" not in name and "difluoro" not in name, \
            f"miscount leaked: {name!r}"
        assert _rt_ok(name, LANSOPRAZOLE), f"round-trip failed: {name!r}"


class TestCanariesUnchanged:
    """Shared substituent/ring machinery -- these PINs must stay byte-identical."""

    @pytest.mark.parametrize("smiles,expected", [
        ("c1ccccc1", "benzene"),
        ("Cc1ccccc1", "toluene"),
        ("COc1ccccc1", "anisole"),  # a review RISK 7: bare anisole IS the PIN (the Blue Book / the Blue Book)
        ("c1ccncc1", "pyridine"),
        ("CS(=O)c1ccccc1", "(methanesulfinyl)benzene"),
        ("c1ccc2ccccc2c1", "naphthalene"),
    ])
    def test_pin_byte_identical(self, smiles, expected):
        from orthonym import name_compound
        assert name_compound(smiles) == expected
