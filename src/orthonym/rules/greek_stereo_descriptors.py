"""Shared Greek stereodescriptor letters for natural-product nomenclature.

IUPAC 2013 (the Blue Book) writes the steroid ring-face and the
carbohydrate anomeric configuration with the GREEK letters ``α``, ``β``, ``ξ`` —
"The stereodescriptors 'α', 'β', and 'ξ' … are cited before the name of the
fundamental parent structure", "The symbols 'α', 'β', or 'ξ' … are placed
immediately at the beginning of the name". The prose uses the English words
'alpha'/'beta' only to *name* the letters; every emitted NAME uses the symbol
(the Blue Book contains ``5α`` 44×, ``α-D-`` 41×, ``β-D-`` 58×, and the spelled-out
form inside a name 0×). OPSIN 2.9.0 parses the Greek symbol and the ASCII word to
the identical structure, so routing every stereodescriptor emitter through these
constants is a pure spelling-conformance change with no round-trip effect
(mirrors the ``rules/lambda_convention.LAMBDA`` fix for ``λ``).

This is the single source of truth: every steroid ring-face and sugar anomeric
emitter imports ALPHA/BETA/XI from here, never a bare ``'alpha'``/``'beta'``.
"""

ALPHA = "α"
BETA = "β"
XI = "ξ"
