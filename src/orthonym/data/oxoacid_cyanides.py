"""The acid-derived functional-class groups of the sulfonic, sulfinic, selenonic, seleninic, telluronic
and tellurinic acids: their halides (sulfonyl, sulfinyl) and their cyanides.

 "Cyanides corresponding to the sulfur, selenium, and tellurium acids"
(the Blue Book): "Cyanides formally formed by replacing the -OH group of sulfonic acids,
sulfinic acids, and similar Se and Te acids are named by functional class nomenclature." Its
examples are ``methanesulfonyl cyanide (PIN)`` (:34811) and ``benzeneseleninyl cyanide (PIN)``
(:34813). "Acyl pseudohalides from suffix acids" (:31460) gives the grammar: the name
of the acyl group, then ``cyanide`` as a separate word, "preceded by a multiplicative prefix, as
needed". (:31260) gives the acyl group: "by changing the 'ic acid' ending of the
suffix to 'yl'" (``benzeneseleninic acid`` -> ``benzeneseleninyl``).

The one table of the ending of an acid name and the ending of its acyl name is
``ACYL_ENDING_OF_ACID_ENDING``; every other table here is read off it. A group of
``perception/functional_groups.py`` is looked up in ``ACID_ACYL_ENDINGS``: the ending of the name of
its parent acid and the ending of the name of its acyl group.
"""

# the ending of the name of an acid R-EOx-OH -> the ending of the name of its acyl group
ACYL_ENDING_OF_ACID_ENDING = {
    "sulfonic acid": "sulfonyl",
    "sulfinic acid": "sulfinyl",
    "selenonic acid": "selenonyl",
    "seleninic acid": "seleninyl",
    "telluronic acid": "telluronyl",
    "tellurinic acid": "tellurinyl",
}

# perception group -> the ending of the name of its parent acid
_ACID_ENDING_OF_GROUP = {
    "sulfonyl_halide": "sulfonic acid",
    "sulfinyl_halide": "sulfinic acid",
    "sulfonyl_cyanide": "sulfonic acid",
    "sulfinyl_cyanide": "sulfinic acid",
    "selenonyl_cyanide": "selenonic acid",
    "seleninyl_cyanide": "seleninic acid",
    "telluronyl_cyanide": "telluronic acid",
    "tellurinyl_cyanide": "tellurinic acid",
}

# perception group -> (ending of the parent acid's name, ending of the acyl group's name)
ACID_ACYL_ENDINGS = {
    group: (acid, ACYL_ENDING_OF_ACID_ENDING[acid])
    for group, acid in _ACID_ENDING_OF_GROUP.items()
}

# the cyanide groups of it
OXOACID_CYANIDE_ENDINGS = {
    group: endings for group, endings in ACID_ACYL_ENDINGS.items() if group.endswith("_cyanide")
}

OXOACID_CYANIDE_GROUPS = tuple(OXOACID_CYANIDE_ENDINGS)
