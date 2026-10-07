"""Blue Book rows that a spelling check flags, each with the reason the check and the book
disagree; shared by the two book-corpus tests (``test_pin_spelling_bluebook.py``, the PIN rows of
the conformance report; ``test_pin_spelling_book_names.py``, every name the book prints as a PIN or
a preferred prefix). Each test adds the rows only its corpus holds.

Keys are (book line, name as printed, rule)."""

_LISTING = ("a listing row prints the ring component without its indicated hydrogen; P-14.7.1 "
            "(:3721) and P-25.7.1.3.1 (:14605, :14607) require it in a PIN")

#: listing rows of (Table 2.9, heading:11700, rows:11708-:11714) and
#: (:11801-:11811), and the amplificant table (:14896); the Table 2.9 listing also
#: prints 'indole (PIN)'
LISTING_ROWS = {
    (11708, "arsindole", "P-14.7.1"): "P-25.2.1 listing row (:11708); " + _LISTING,
    (11708, "phosphindole", "P-14.7.1"): "P-25.2.1 listing row (:11708); " + _LISTING,
    (11710, "isoarsindole", "P-14.7.1"): "P-25.2.1 listing row (:11710); " + _LISTING,
    (11710, "isophosphindole", "P-14.7.1"): "P-25.2.1 listing row (:11710); " + _LISTING,
    (11801, "phenoxaphosphinine", "P-14.7.1"): "P-25.2.2.3 listing row (:11801); " + _LISTING,
    (11805, "phenoxarsinine", "P-14.7.1"): "P-25.2.2.3 listing row (:11805); " + _LISTING,
    (11807, "phenoxastibinine", "P-14.7.1"): "P-25.2.2.3 listing row (:11807); " + _LISTING,
    (11811, "phenothiarsinine", "P-14.7.1"): "P-25.2.2.3 listing row (:11811); " + _LISTING,
    (14898, "pyran", "P-14.7.1"): "P-26.2.2.1 amplificant table (:14896); " + _LISTING,
    (14898, "pyrrole", "P-14.7.1"): "P-26.2.2.1 amplificant table (:14896); the PIN is 1H-pyrrole (:3721)",
}

#: rows against the book's own rule: (:3448) alphabetizes 'chlorophenyl' before
#: 'hydroxy' (this stereo example cites N-hydroxy first); (:2869) 'all locants must
#: be cited' (the book writes the suffix locant beside other locants 8 times for
#: cyclohexane-1-carboxylic acid and 12 times for ethan-1-ol, and omits it once each, these rows)
OWN_RULE_ROWS = {
    (47666, "(Z)-N-hydroxy(4-chlorophenyl)(phenyl)methanimine", "P-14.5"): "book row against :3448",
    (30363, "4-[(hydroxysulfanyl)carbonyl]cyclohexanecarboxylic acid", "P-14.3.3"): "book row against :2869",
    (40135, "2-(osmocen-1-yl)ethanol", "P-14.3.3"): "book row against :2869",
}
