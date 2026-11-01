"""The zwitterion of a fully ionised carboxylic acid with one primary ammonium group and other nitrogen atoms.

 "Anionic and cationic centers on different parent structures" (the Blue Book): "Zwitterionic
compounds with anionic and cationic centers on different parent structures may be named by prefixing the name of
the cationic center... to the name of the anionic parent structure"; INTRODUCTION (:42411): "anionic
centers are preferred for lower locants and become the parent structure, into which the cationic part is
substituted". The anion is the parent, the group -NH3(+) is the prefix 'azaniumyl' and every carboxylate that is
not the suffix of the parent is the prefix 'carboxylato' "Prefixes for anionic centers derived from acid
characteristic groups",:41199: '-CO-O(-) carboxylato (preferred prefix)',:41209), as in 'pentanedioate' with
'4-azaniumyl-4-carboxylatobutanamido'.

When every carboxy group of the molecule is ionised, the parent of the anion is the parent of the neutral acid:
 "Choice of an anionic parent structure" (:41261) takes first "the parent with the maximum number of anionic
centers" (:41265), the neutral acid takes the one with the maximum number of carboxy groups as suffixes, and the two
counts are one count. The parent is therefore the neutral name's, the acid ending becomes the anion's, and each 'carboxy'
and the 'amino' of the ammonium group are re-expressed as 'carboxylato' and 'azaniumyl'.

``charged_router._name_primary_amine_azaniumyl_zwitterion`` builds this name when every nitrogen atom of the molecule
is an ammonium group, where the one 'amino' of the neutral name is the one to change. A neutral name with other 'amino'
groups ('{[(1-carboxyethyl)amino]...}', 'N-substituted amides') has several, and the text does not say which stands for
the ammonium nitrogen: each is offered, and the name that ships is the one that is cited in the order of
(``alphanumerical_order_check``) and that OPSIN reads back with the hydron on the atom the input has it
(``charged_router._full_inchikey_rt_ok``, which checks the protonation site). The neutral name is built inside
``naming_utils.amino_cited_as_azaniumyl``, which orders and numbers its prefixes as the ion's, so 'anilino' (which stands
between 'amino' and 'azaniumyl') is cited before 'azaniumyl'. A decline returns ''.

A phosphate monoester dianion in the molecule ('(2S)-2-azaniumyl-3-(phosphonatooxy)propanoate', phosphoserine) is cited as
the prefix 'phosphonatooxy' of the carboxylate parent, and the name is recorded as no PIN: (:41261) takes "the parent
with the maximum number of anionic centers, including anionic suffixes" (:41265), which would be the phosphate ester (two
centres) if an ester were an anionic parent, and the book is silent on that (the maintainers' ruling of 2026-11-01).
"""
import re

from rdkit import Chem

#: a free carboxylic acid group: the molecule is fully ionised when it has none
_FREE_CARBOXY = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')

#: the dianion of a phosphoric monoester, ``R-O-P(=O)(O-)O-``: its prefix is 'phosphonatooxy',:41195)
_PHOSPHATE_MONOESTER_DIANION = Chem.MolFromSmarts('[O-]P(=O)([O-])O[#6]')

#: a hydroxy group on phosphorus: the phosphate is not fully ionised
_PHOSPHORUS_HYDROXY = Chem.MolFromSmarts('P[OX2H1]')

#: 'carboxy' as a prefix ('carboxylato' and 'carboxylic acid' are other words)
_CARBOXY_PREFIX = re.compile(r'carboxy(?!l)')

#: 'amino' as a prefix on a name that has a letter before it only in a compound ('...]amino', '2-amino', 'aminomethyl')
_AMINO_PREFIX = re.compile(r'(?<![A-Za-z])amino')


def fully_ionised_azaniumyl_zwitterion(mol, cations, anions, style: str) -> str:
    """The name of a zwitterion of one primary -NH3(+) and carboxylates only, no carboxy group left un-ionised, with the
    anion as the parent, or ''.

    The neutral molecule is named by the same route as the builder of the all-ammonium case; the acid suffix of that
    name becomes the anion's, every 'carboxy' prefix 'carboxylato', and one 'amino' prefix 'azaniumyl'. Each 'amino'
    is tried in order of position; the first name that reads back to the full InChIKey and to the protonation site of
    the input ships."""
    from .charged_router import (_full_inchikey_rt_ok, _is_malformed_parent, _is_primary_protonated_amine,
                                 _reenter, classify_anion)
    if len(cations) != 1 or not anions:
        return ''
    if not _is_primary_protonated_amine(mol.GetAtomWithIdx(cations[0]['atom_idx'])):
        return ''
    phosphates = mol.GetSubstructMatches(_PHOSPHATE_MONOESTER_DIANION)
    phosphate_oxygens = {i for m in phosphates for i in (m[0], m[3])}
    if (any(classify_anion(mol, a) != 'carboxylate' for a in anions if a['atom_idx'] not in phosphate_oxygens)
            or mol.HasSubstructMatch(_FREE_CARBOXY) or mol.HasSubstructMatch(_PHOSPHORUS_HYDROXY)):
        return ''
    from .ions import name_carboxylate_anion
    from .salts import _neutralize_zwitterion
    neutral = _neutralize_zwitterion(mol)
    if neutral is None:
        return ''
    from ..assembly.naming_utils import amino_cited_as_azaniumyl
    from .amino_acids import retained_amino_acid_names_left_out
    try:
        # the neutral name is ordered and numbered for the ion's prefix, and is the systematic one ('aminoacetic acid')
        with amino_cited_as_azaniumyl(), retained_amino_acid_names_left_out():
            neutral_name = _reenter(Chem.MolToSmiles(neutral, canonical=True), style)
    except (RecursionError, ValueError, RuntimeError):
        return ''
    if not neutral_name or _is_malformed_parent(neutral_name):
        return ''
    anion_name = name_carboxylate_anion(neutral_name)
    if not anion_name or anion_name == neutral_name:
        return ''
    anion_name = _CARBOXY_PREFIX.sub('carboxylato', anion_name)
    if phosphates:
        # the phosphate monoester of the neutral name is '(phosphonooxy)'; its anion is '(phosphonatooxy)'
        if anion_name.count('phosphonooxy') != len(phosphates):
            return ''
        anion_name = anion_name.replace('phosphonooxy', 'phosphonatooxy')
    from ..validation.spelling.checks_marks import alphanumerical_order_check
    for hit in _AMINO_PREFIX.finditer(anion_name):
        name = anion_name[:hit.start()] + 'azaniumyl' + anion_name[hit.end():]
        if alphanumerical_order_check(mol, name) is None and _full_inchikey_rt_ok(mol, name):
            if phosphates:
                from ..metrics.provenance import record_non_pin_label
                record_non_pin_label(name, rule="P-72.7",
                                     detail="the book does not say whether the carboxylate or the phosphate ester "
                                            "dianion is the anionic parent (:41265, :41289); the carboxylate parent is "
                                            "shipped below the PIN")
            return name   # cited in the order of and read back to the full InChIKey and the protonation site
    return ''
