"""Roadmap N5a: the label guard reads the end atoms of an acyclic 'a' chain.

 (the Blue Book, heading 'Skeletal replacement ('a') nomenclature for
acyclic parent hydrides'): "The chain must be terminated by a C atom or one of the
following heteroatoms: P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, or Tl." (repeated
at,:23430). (:23348): an 'a' name is a PIN only for "four or more
heterounits... in a unbranched chain containing at least one carbon atom".

``non_pin_vocabulary`` counted the locants only, so an 'a' chain with four or more
heteroatoms that ends on N, O, S, Se or Te passed the guard: the engine labelled
'N-(11-amino-3,7-disulfanylidene-2,8,12-trioxa-4,5,6-trithia-10-azadodec-11-en-1-yl)urea'
pin_unverified (milestone1500 and druglike2000, best-effort tier).

The label decision moved to the writers that keep an 'a' chain
(``book_prefixes.a_chain_licensed``); the tests are in test_nq_forms_label_records.py.
"""

from orthonym.rules.pin_vocabulary import non_pin_vocabulary


def test_no_pattern_re_reads_the_a_chain_ends():
    # (the Blue Book) is decided from the atoms by the chain writers
    assert non_pin_vocabulary("1,4,7,10-tetraoxaundecyl") is None
    assert non_pin_vocabulary("2,5,8,11-tetraoxadodecane") is None
