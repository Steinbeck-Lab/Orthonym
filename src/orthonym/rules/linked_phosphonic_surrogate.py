"""The sulfonic acid surrogate of a linked phosphonic acid (``multiplicative._linked_phosphonic_acids_name``)."""
from typing import Optional

from rdkit import Chem


def _principal_group_is_sulfonic(surrogate) -> bool:
    """ "SENIORITY ORDER FOR CLASSES", Table 4.1 (the Blue Book): "7a 'Suffix' acids in the order
    carboxylic (not including carbonic or polycarbonic acids...), sulfonic, sulfinic,...". The surrogate is named
    '...sulfonic acid' only when its principal group is the sulfonic acid; with a carboxylic acid (or any senior
    group) elsewhere its name ends otherwise. A surrogate whose principal group cannot be read is not decided
    (True)."""
    try:
        from ..perception.functional_groups import detect_functional_groups
        from .seniority import get_principal_group
        group, _ = get_principal_group(surrogate, detect_functional_groups(surrogate))
    except Exception:
        return True
    return group is None or 'sulfon' in group


def name_sulfonic_surrogate(surrogate) -> Optional[str]:
    """The name of ``surrogate``, the linked molecule with every phosphonic acid group drawn as a sulfonic acid
    group; None, without naming it, when a sulfonic acid cannot be its principal group (the caller reads only a name
    that ends '(di|tri|tetra)sulfonic acid')."""
    if not _principal_group_is_sulfonic(surrogate):
        return None
    from ..namer import Orthonym
    return Orthonym(style='pin', _disable_opsin_validity_gate=True).name(Chem.MolToSmiles(surrogate))
