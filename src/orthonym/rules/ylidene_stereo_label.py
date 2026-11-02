"""The E/Z of a ring '-ylidene' double bond cited inside the prefix is below the PIN.

 (1) (the Blue Book): the PIN cites that descriptor at the front of the name with the locant of
the parent's attachment atom (method (a)); cited at the front of the substituent group (method (b)) the name is valid
in general nomenclature only.
"""


def label_ylidene_ez_in_prefix(mol, ring_list, stereo_prefix: str) -> None:
    """Record ``stereo_prefix`` as below the PIN when a ring atom of ``ring_list`` has a configured double bond to an
    atom outside the ring (the '-ylidene' bond), whose descriptor the prefix then cites."""
    ring = set(ring_list)
    if stereo_prefix and any(b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtomIdx(i) not in ring
                             and b.HasProp('_CIPCode') for i in ring for b in mol.GetAtomWithIdx(i).GetBonds()):
        from ..metrics.provenance import record_non_pin_label
        record_non_pin_label(stereo_prefix, rule="P-93.5.1.4.2.1", detail="ylidene E/Z cited in the prefix")
