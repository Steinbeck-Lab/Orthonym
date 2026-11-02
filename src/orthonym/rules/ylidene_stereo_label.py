"""The E/Z of a ring '-ylidene' double bond cited inside the prefix is below the PIN.

 (1) (the Blue Book): the PIN cites that descriptor at the front of the name with the locant of
the parent's attachment atom (method (a)); cited at the front of the substituent group (method (b)) the name is valid
in general nomenclature only.
"""


def label_ylidene_ez_in_prefix(mol, ring_list, stereo_prefix: str, attachment_atom=None) -> None:
    """Record ``stereo_prefix`` as below the PIN when a ring atom of ``ring_list`` has a configured double bond to an
    atom outside the ring (the '-ylidene' bond), whose descriptor the prefix then cites. The bond from the attachment
    atom to a chain atom of the parent is cited by the host instead (method (a),
    ``stereochemistry.ylidene_bond_on_chain``), so it is no such bond."""
    from .stereochemistry import ylidene_bond_on_chain
    ring = set(ring_list)
    if stereo_prefix and any(b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtomIdx(i) not in ring
                             and b.HasProp('_CIPCode')
                             and not (i == attachment_atom and ylidene_bond_on_chain(mol, b))
                             for i in ring for b in mol.GetAtomWithIdx(i).GetBonds()):
        from ..metrics.provenance import record_non_pin_label
        record_non_pin_label(stereo_prefix, rule="P-93.5.1.4.2.1", detail="ylidene E/Z cited in the prefix")


def ylidene_bonds_to_chain(mol, ring_list, attachment_atom):
    """The stereo double bond from the attachment atom to an acyclic parent atom: a ring '-ylidene' group on a chain.

     (1) (the Blue Book): method (a), the PIN method (:48279), cites the E/Z of that bond at the
    front of the name with the locant of the parent's attachment atom, and the chain host does
    (``stereochemistry._is_true_exocyclic``), so the prefix leaves it out. A bond to a ring atom of the host keeps
    its prefix descriptor, method (b), below the PIN (``ylidene_stereo_label``). At the attachment atom of a ring
    the one exocyclic double bond is the bond to the parent."""
    from .stereochemistry import ylidene_bond_on_chain
    ring = set(ring_list)
    return tuple(b.GetIdx() for b in mol.GetAtomWithIdx(attachment_atom).GetBonds()
                 if b.GetOtherAtomIdx(attachment_atom) not in ring and ylidene_bond_on_chain(mol, b))
