"""Fused parents that OPSIN 2.9.0 and OPSIN's main branch number differently (measured 2026-11-18).

The engine numbers a fused parent that no table holds by OPSIN's reading of its fusion name
(``rules.bridged_fused_pin.parents``, sources ``*_name+opsin``). OPSIN's main branch changed how it lays
out fused ring systems (commits 093a195c and 75ff41fe, "could consider rings in a horizontal line that
weren't fused", "can help with 7 membered rings"), so for the parents below the two versions put the
same locant on different atoms. A name numbered by OPSIN 2.9.0 on such a parent is read by OPSIN main as
another structure, or not at all. While the OPSIN in use is not a main-branch build, these parents are not
used (``rules.bridged_fused_pin.parents.fused_parent``), and the molecule takes its von Baeyer name as it
does under OPSIN main.

OPSIN main builds from PR #316 ("Accept locant 8 for the pyrrolizine ring-fusion carbon", cb88edec) up to the fix
of dan2097/opsin#359 (a2f1a995, 2026-11-19) read locant 8 of pyrrolizine as its fusion carbon 7a. A bridged
pyrrolizine numbers its first bridge atom 8, the Blue Book: "Bridge atoms are numbered continuing
from the highest locant of the fused ring system"), so such a build reads the right name as another structure. A
bridged pyrrolizine is therefore not named while the OPSIN in use fails the probe
(``opsin_capabilities.bridged_pyrrolizine_reads_right``); OPSIN 2.9.0 and the fixed main read the names as written.

How the table was measured: every fused or bridged ring token in the names of the PubChem run and the six
paper sets (7,680,652 tokens, 31,550 fused parents, 13,197 bridged parents) and the engine's own parent
tables (323 names) and retained ring parents (63) were read with OPSIN 2.9.0 and OPSIN main 760d57c0f132
as 'perhydro' + parent (else the bare parent, else a name that carries it). A parent is listed when some
locant that 2.9.0 puts on an atom is on another atom, or on no atom, in main's reading. Builds of OPSIN main
from 9f30df5cc22c (2026-10-23) on read the 48 affected paper-set names as 760d57c0f132 does; an older main build
(ORTHONYM_OPSIN=main@<older commit>) is not measured and counts as main. Tools and the
per-parent verdicts: `internal notes`.
"""

#: Fused parent names (as ``fused_parent`` spells them, indicated hydrogen dropped) that OPSIN 2.9.0 and
#: OPSIN main number differently.
LAYOUT_MISNUMBERED_PARENTS = frozenset({
    "[1,4]benzodioxonino[5,6-d]pyrimidine",
    "[1,4]benzodioxonino[7,6-c]pyridazine",
    "[1,4]benzothiazino[4,3-a][1]benzazacycloundecine",
    "[1]benzoxacycloundecino[8,9-c]pyrrole",
    "[1]benzoxonino[5,6-c]pyrrole",
    "benzo[3,4]cyclohepta[1,2-d]isoindole",
    "benzo[4,5]cyclodeca[1,2-b]oxirene",
    "benzo[7,8]azuleno[5,6-b]oxirene",
    "benzo[8,9]cyclonona[1,2-c]pyridazine",
    "benzo[f]cyclopropa[d]quinoline",
    "chryseno[4a,4b-b]oxirene",
    "cyclopenta[4,5]cyclodeca[1,2-b]oxirene",
    "cyclopenta[5,6]cyclododeca[1,2-b]oxirene",
    "cyclopenta[e]cyclopropa[c]indole",
    "cyclopropa[c]pyrrolo[1,2-a]azocine",
    "cyclopropa[c]thieno[2,3-e]indole",
    "cyclopropa[d][3]benzazacycloheptadecine",
    "cyclopropa[k][4,23,2,7]benzodioxadiazacyclohentriacontine",
    "dibenzo[b,i][1]oxacycloundecine",
    "dicyclopenta[d,h]isoindole",
    "dipyrido[1,2-a:3',4'-h]azonine",
    "dipyrido[1,2-b:2',3'-k][1,2,5]triazacyclotridecine",
    "dipyrido[1,2-b:4',3'-k][10,1,2,5]oxatriazacyclotridecine",
    "indolo[1',2':4,5][1,4,7,10,13]pentaazacyclopentadecino[1,2-b]isoquinoline",
    "naphtho[1,2-b][4,6,5]benzodioxaphosphacycloundecine",
    "naphtho[1,2-b][4]benzazonine",
    "phenanthro[4a,4-d][1,3]oxazole",
    "piceno[4,4a-b]azirine",
    "piceno[4a,4-b]azete",
    "pyrido[1,2-b][2,7,11,14]benzotetraazacyclononadecine",
    "pyrido[1,2-c][12,2,3,6]benzoxatriazacyclopentadecine",
    "pyrido[1,2-c][2,3,6,10]benzotetraazacyclotridecine",
    "pyrido[1,2-c][2,3]benzodiazacyclotridecine",
    "pyrido[1,2-d][4]benzazacyclotridecine",
    "pyrrolo[1,2-a][1,6]benzodiazonine",
})

#: Fused parents to which OPSIN main gives a locant 2.9.0 does not (pyrrolizine: 8 on C7a), a locant that the
#: first bridge atom of a bridged derivative takes; such a parent is not the parent of a bridged fused name
#: (``rules.bridged_fused_pin.selection.best_splits``). As a fusion component ('thieno[2,3-b]pyrrolizine') it is
#: numbered by the fused system, so it stays in use there.
BRIDGE_LOCANT_ALIASED_PARENTS = frozenset({"pyrrolizine"})
