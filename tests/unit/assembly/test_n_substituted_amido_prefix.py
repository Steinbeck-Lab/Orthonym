"""v30 F-amido — an N-substituted acylamino branch `-N(R')-C(=O)-R` is the
P-66.1.1.4.3 method (1) `{N-R'}{acyl}amido` PREFIX (N-methylacetamido /
N-methylformamido / N-ethylpropanamido), NOT the ugly general-engine replacement
name (`1,2-dimethyl-3-oxa-1-azaprop-2-en-1-yl`) nor a fail-closed abstention.

PIN authority P-66.1.1.4.3 method (1), BlueBookV2.md:32995; verbatim
`2-(N-methylpropanamido)benzene-1-sulfonic acid (PIN)` :33040. The amido-prefix
form is PIN ONLY as a PREFIX (a senior characteristic group present); when the
amide is the PRINCIPAL group it is the SUFFIX (`N-methyl-N-(quinolin-4-yl)acetamide`,
:33048, `4-(N-methylacetamido)quinoline` explicitly NOT PIN). `_name_amino_branch`
is a substituent namer, only reached when parent selection already made the amide a
prefix, so seniority is handled upstream (verified: OC(=O)CCN(C)C(C)=O routes here;
CCN(C)C(C)=O names `N-ethyl-N-methylacetamide` at the suffix path, untouched).

Approach:
rebuild a mono-acid from the acyl subgraph, name it, `oic acid->amido` /
`carboxylic acid->carboxamido` (+ retained acetamido/formamido), render the
N-substituent as an `N-` prefix. We reuse the strict `linear_acyl_amido_prefix`
for the acyl core (which already proves atom coverage) + the recursive substituent
namer for R'. Fail closed (None) on a branched/unsaturated/ring/hetero acyl or an
unnameable R' -> those keep abstaining, never a wrong or atom-dropped name.
"""
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent, _name_amino_branch
from orthonym.assembly.substituent_naming import n_substituted_acyl_amido_prefix


# ---- helper-level positives -------------------------------------------------

def _frag(smiles):
    m = Chem.MolFromSmiles(smiles)
    return m


def test_helper_n_methylacetamido():
    # OC(=O)CCN(C)C(C)=O : N=5, methyl=6, acyl C=7, acyl-methyl=8, acyl O=9
    m = _frag("OC(=O)CCN(C)C(C)=O")
    assert n_substituted_acyl_amido_prefix(m, 5, {5, 6, 7, 8, 9}, {0, 1, 2, 3, 4}) \
        == "N-methylacetamido"


def test_helper_n_methylformamido():
    # OC(=O)CCN(C)C=O : N=5, methyl=6, formyl C=7, formyl O=8
    m = _frag("OC(=O)CCN(C)C=O")
    assert n_substituted_acyl_amido_prefix(m, 5, {5, 6, 7, 8}, {0, 1, 2, 3, 4}) \
        == "N-methylformamido"


def test_helper_n_ethylacetamido():
    # OC(=O)CCN(CC)C(C)=O : N=5, ethyl=6,7 ; acyl C=8, acyl-methyl=9, O=10
    m = _frag("OC(=O)CCN(CC)C(C)=O")
    assert n_substituted_acyl_amido_prefix(m, 5, {5, 6, 7, 8, 9, 10}, {0, 1, 2, 3, 4}) \
        == "N-ethylacetamido"


# ---- helper-level fail-closed (no wrong / atom-dropped emission) ------------

def test_helper_branched_acyl_fails_closed():
    # isobutyryl acyl (branched) -> linear_acyl_amido_prefix refuses -> None
    m = _frag("OC(=O)CCN(C)C(=O)C(C)C")
    n = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "N")
    frag = set(range(m.GetNumAtoms())) - {0, 1, 2, 3, 4}
    assert n_substituted_acyl_amido_prefix(m, n, frag, {0, 1, 2, 3, 4}) is None


def test_helper_unsaturated_acyl_fails_closed():
    # acryloyl acyl (unsaturated) -> refused
    m = _frag("OC(=O)CCN(C)C(=O)C=C")
    n = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "N")
    frag = set(range(m.GetNumAtoms())) - {0, 1, 2, 3, 4}
    assert n_substituted_acyl_amido_prefix(m, n, frag, {0, 1, 2, 3, 4}) is None


def test_helper_ring_n_fails_closed():
    # amide N inside a ring is not a bare tertiary amide-N prefix
    m = _frag("O=C1CCCN1C")
    n = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "N")
    frag = set(range(m.GetNumAtoms()))
    assert n_substituted_acyl_amido_prefix(m, n, frag, set()) is None


# ---- fable review of the first F-amido cut: R' must be ENCLOSED, not raw -----
# The raw `f"N-{r_name}..."` left R' unbracketed -> (a) regressed a valid RT-exact
# emission to an abstention (unparseable `N-2,2-dimethylpropyl...`) and (b) shipped
# an OPSIN-unparseable stereo name. Fix: reuse the amide N-substituent machinery
# (_name_n_substituent + format_n_substitution) -> correct P-16.5.1.1 marks + stereo.

def test_helper_neopentyl_R_bracketed():
    m = _frag("OC(=O)CCN(CC(C)(C)C)C(C)=O")
    assert n_substituted_acyl_amido_prefix(m, 5, set(range(m.GetNumAtoms())) - {0, 1, 2, 3, 4},
                                           {0, 1, 2, 3, 4}) == "N-(2,2-dimethylpropyl)acetamido"


def test_helper_isobutyl_R_bracketed():
    m = _frag("OC(=O)CCN(CC(C)C)C(C)=O")
    assert n_substituted_acyl_amido_prefix(m, 5, set(range(m.GetNumAtoms())) - {0, 1, 2, 3, 4},
                                           {0, 1, 2, 3, 4}) == "N-(2-methylpropyl)acetamido"


def test_helper_2chlorophenyl_R_bracketed():
    m = _frag("OC(=O)CCN(c1ccccc1Cl)C(C)=O")
    assert n_substituted_acyl_amido_prefix(m, 5, set(range(m.GetNumAtoms())) - {0, 1, 2, 3, 4},
                                           {0, 1, 2, 3, 4}) == "N-(2-chlorophenyl)acetamido"


def test_helper_stereo_R_bracket_escalated():
    # BLOCKER (b): a stereocentre R' -> the descriptor must be embedded and the mark
    # escalated to [] (P-16.5.4), never spliced raw after `N-`.
    # v33 Engine 4 change-asserted-value: `(S)-` -> `(1S)-`. The carrier chain
    # numbering `1-phenylethyl` cites was not reaching the substituent stereo
    # emitter, so the descriptor shipped unlocanted. VERIFIED rule,
    # `BlueBookV2.md:44643`, heading `## **P-91.3** NAMING OF STEREOISOMERS`:
    # a substituent-group stereodescriptor is "preceded by a numerical or letter
    # locant to describe the position of the stereogenic unit *when such locants
    # are present*" -- worked `(PIN)` example `[(1R)-1-chloropropyl]benzene`.
    # The property under test (embedded descriptor + `[]` escalation) is
    # unchanged; only the descriptor gained its locant. Both spellings round-trip
    # through OPSIN 2.9.0 to the input's full InChIKey.
    m = _frag("OC(=O)CCN([C@@H](C)c1ccccc1)C(C)=O")
    assert n_substituted_acyl_amido_prefix(m, 5, set(range(m.GetNumAtoms())) - {0, 1, 2, 3, 4},
                                           {0, 1, 2, 3, 4}) == "N-[(1S)-1-phenylethyl]acetamido"


def test_helper_thioacyl_R_fails_closed():
    # finding 2: a mixed imide -N(C=O)(C=S) is not P-66.1.1.4.3 method (1) -> fail closed
    # (the general fallback then names it via a valid replacement name, never junk here).
    m = _frag("OC(=O)CCN(C(C)=S)C(C)=O")
    assert n_substituted_acyl_amido_prefix(m, 5, set(range(m.GetNumAtoms())) - {0, 1, 2, 3, 4},
                                           {0, 1, 2, 3, 4}) is None


def test_helper_misnamed_R_reanchor_fails_closed():
    # fable RE-review BLOCKER: name_substituent mis-names -CH2-S-CH3 as `methylsulfanyl`
    # (drops the CH2 -- pre-existing fragment-namer defect). The gate-INDEPENDENT OPSIN
    # re-anchor (build the amide, require same InChIKey as the capped fragment) must fail
    # closed so the general fallback names it instead of shipping a wrong molecule.
    m = _frag("OC(=O)CCN(CSC)C(C)=O")  # R' = -CH2-S-CH3
    assert n_substituted_acyl_amido_prefix(m, 5, set(range(m.GetNumAtoms())) - {0, 1, 2, 3, 4},
                                           {0, 1, 2, 3, 4}) is None


def test_helper_misnamed_R_selena_reanchor_fails_closed():
    m = _frag("OC(=O)CCN(C[Se]C)C(C)=O")  # R' = -CH2-Se-CH3, same defect shape
    assert n_substituted_acyl_amido_prefix(m, 5, set(range(m.GetNumAtoms())) - {0, 1, 2, 3, 4},
                                           {0, 1, 2, 3, 4}) is None


def test_helper_sentinel_R_fails_closed():
    # fable RISK 2: name_substituent may return the `substituent` placeholder ->
    # never splice `N-substituentacetamido` (invariant 16). is_refusal_sentinel + the
    # re-anchor both catch it.
    m = _frag("OC(=O)CCN(N=O)C(C)=O")  # R' = -N=O (recursion/depth fallback)
    assert n_substituted_acyl_amido_prefix(m, 5, set(range(m.GetNumAtoms())) - {0, 1, 2, 3, 4},
                                           {0, 1, 2, 3, 4}) is None


def test_helper_isotope_fails_closed():
    # finding 3: an isotopic label would be dropped by the amido stem -> fail closed
    # (honest without the gate; verified under ORTHONYM_SELF_CONSISTENCY_GATE=off).
    m = _frag("OC(=O)CCN(C)C(=O)[13CH3]")
    assert n_substituted_acyl_amido_prefix(m, 5, set(range(m.GetNumAtoms())) - {0, 1, 2, 3, 4},
                                           {0, 1, 2, 3, 4}) is None


# ---- end-to-end through name_substituent (proves the intercept path) --------

def test_end_to_end_name_substituent():
    m = _frag("OC(=O)CCN(C)C(C)=O")
    assert name_substituent(m, {5, 6, 7, 8, 9}, 5, allow_mancude=True) == "N-methylacetamido"
    assert _name_amino_branch(m, {5, 6, 7, 8, 9}, 5, {0, 1, 2, 3, 4}) == "N-methylacetamido"
