# Phane names

A phane name replaces each ring or ring system of a structure with one atom, names the chain or ring that is left, and then says which rings stand for which atoms. `1,4(1,4)-dibenzenacyclohexaphane` is [2.2]paracyclophane: six nodes in a ring, and nodes 1 and 4 are benzene rings, each joined at its positions 1 and 4. The recommendations (section P-52.2.5) make the phane name the preferred name for two kinds of structure:

- **Cyclophanes.** A ring made of rings or ring systems and of atoms or chains, at least one of the rings or ring systems being mancude (benzene, naphthalene, pyridine) and joined to the rest at nonadjacent positions. When those conditions hold, the phane name is preferred to a fusion, bridged fused, von Baeyer or spiro name of the same structure.
- **Linear phanes.** Four or more rings or ring systems, two of them terminal, joined by atoms or chains (or directly to each other, as in a ring assembly), with at least seven nodes in all. The phane name is preferred to the substitutive and the multiplicative name: `2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphane` is the preferred name of the structure that is also named 1-phenoxy-4-(4-phenoxyphenoxy)benzene.

## How a phane name is checked

OPSIN 2.9.0, the default, has no grammar for phane names. OPSIN's main branch reads them, so a phane name is checked in full only with `--opsin main` ([Install](../start/install.md#choosing-the-opsin-version)). Then it is read back like every other name: OPSIN reads the name, and the name ships only when the structure it gives has your full InChIKey.

```console
$ orthonym --opsin main "O(c1ccccc1)c1ccc(Oc2ccc(Oc3ccccc3)cc2)cc1"
2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphane
```

With the default OPSIN 2.9.0, Orthonym does not build phane names, with one exception. The exception is the monocyclic cyclophane of benzene rings (`1,4(1,4)-dibenzenacyclohexaphane`), which it still ships, checked by building the structure back from the name itself; the row says `carveout:phane` and the tier is {tier}`pin_unverified`. A linear phane, a cyclophane with any other skeleton and a cyclophane with substituents are not named as phanes; at the default tier they are declined (for the large rings described below).

## A non-phane name of a cyclophane is not a preferred name

For a cyclophane the preferred name is the phane name, so Orthonym never gives another name for it as a preferred name. This holds for the large rings the rules describe (ten atoms or more) with either OPSIN. Where it cannot build the phane name, the default tier declines the structure, and the wider tiers still return the fusion, bridged fused, von Baeyer or spiro name, labelled {tier}`systematic_verified`:

```console
$ orthonym "C1CCC2=CC(=CC=C2)CCCC12CCCC1=CC(=CC=C1)CCC2"

C1CCC2=CC(=CC=C2)CCCC12CCCC1=CC(=CC=C1)CCC2	NO_VERIFIED_PIN: unknown organic compound
$ orthonym --emit-tier best-effort "C1CCC2=CC(=CC=C2)CCCC12CCCC1=CC(=CC=C1)CCC2"
5,5'-spirobi[bicyclo[7.3.1]trideca-1(12),9(13),10-triene]
$ orthonym --opsin main "C1CCC2=CC(=CC=C2)CCCC12CCCC1=CC(=CC=C1)CCC2"
4,12(1,3)-dibenzenaspiro[7.7]pentadecaphane
```

The first command prints an empty name and the reason on the error stream ([Declines](../declines.md)). Earlier versions returned the spiro name as a verified preferred name; it is not one.

## Stereodescriptors

The CIP descriptors of the atoms of a phane skeleton are cited at the front of the name, with the composite locant of an atom of a ring that a superatom stands for: `(2E,5^1R,5^2S,5^5R,8Z,11E,15^1R,15^2S,15^4S,17E)-4,6,10,14,16-pentaoxa-1,19(1),7,13(1,4)-tetrabenzena-5,15(1,2)-dicyclohexananonadecaphane-2,8,11,17-tetraene-5^5,15^4-dicarboxylic acid`. The descriptors of a double bond of a cyclic phane are cited from eight nodes on, as the recommendations require; a cyclic phane of seven nodes or fewer with a specified double bond is declined, because OPSIN does not read an omitted descriptor as its configuration. When two numberings are equally low, the one that gives R or Z the lower locant is used. The same holds for the descriptors inside the prefixes: `2-[(1Z)-prop-1-en-1-yl]-5-[(1E)-prop-1-en-1-yl]-1,4(1,4)-dibenzenacyclohexaphane`, not the E prefix first. The read-back compares the isomeric SMILES as well as the InChIKey.

## Hydro prefixes and indicated hydrogen

A ring that is a hydro derivative of a mancude ring system, or that needs indicated hydrogen, is cited by its mancude parent, and the hydrogen atoms are written before the parent hydride name with the composite locants of the ring atoms: `1^4H-1(3,5)-pyrana-4(1,3)-benzenacyclohexaphane`, `1^1,1^2-dihydro-1(2)-quinolina-3,5(1,4),7(1)-tribenzenaheptaphane`, `1^1H,2^1H,3^1H,4^1H,5^1H,6^1H,7^1H-1,7(2),2,3,4,5,6(2,5)-heptapyrrolaheptaphane`. The indicated hydrogen of a ring is numbered before its attachment locants, as the name of its parent hydride is: `1^2H-1(6)-pyrana-...`, not `1^6H-1(2)-pyrana-...`. A cyclophane of more than seven nodes whose hydro ring keeps a double bond has no certified name: the recommendations cite a descriptor for that bond (`1^2Z`, in `(1^2Z,2E)-1^1,1^4,1^5,1^6-tetrahydro-1(2,6)-pyridina-7(1,3)-benzenacyclododecaphan-2-ene`), and OPSIN does not read it. The default tier declines such a structure; the wider tiers return the name without the descriptor, labelled below the preferred name.

## A phane as a substituent

When another parent is senior to the phane (it has a senior atom such as silicon, or it cites more of the principal characteristic group), the preferred name keeps that parent and cites the linear phane as a substituent prefix, with the suffix `yl` at the atom that bears the free valence:

```console
$ orthonym --opsin main "C[Si](C)(C)c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1"
trimethyl[2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphan-1^4-yl]silane
```

With the default OPSIN 2.9.0 such a structure is declined at the default tier, as before. When two identical parents hold every principal characteristic group, the recommendations prefer the multiplicative name that cites the phane as its central group (`3,3'-[1,7(1),3,5(1,4)-tetrabenzenaheptaphane-1^4,7^4-diyl]dipropanoic acid`); that name is not built, and the name that cites the phane in the prefix of one parent is not given as a preferred name.

## Examples

Five structures whose preferred names the recommendations print, and what each OPSIN gives:

| Preferred name | Structure | OPSIN 2.9.0 (default) | `--opsin main` |
|:--|:--|:--|:--|
| `1,4(1,4)-dibenzenacyclohexaphane` | `C1CC2=CC=C(C=C2)CCC2=CC=C1C=C2` | the name, {tier}`pin_unverified` | the name, {tier}`pin_verified` |
| `1(1,3)-benzenacycloheptadecaphane` | `C1CCCCCCCCCCCCCCCC=2C=C1C=CC2` | declined | the name |
| `2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphane` | `O(C1=CC=CC=C1)C1=CC=C(C=C1)OC1=CC=C(C=C1)OC1=CC=CC=C1` | declined | the name |
| `2,4,6-trithia-1,7(1),3,5(1,4)-tetrabenzenaheptaphane` | `S(C1=CC=CC=C1)C1=CC=C(C=C1)SC1=CC=C(C=C1)SC1=CC=CC=C1` | declined | the name |
| `4(3,5)-pyridina-1,7(1),2,3,5,6(1,3)-hexabenzenaheptaphane` | `N1=CC(=CC(=C1)C1=CC(=CC=C1)C1=CC(=CC=C1)C1=CC=CC=C1)C1=CC(=CC=C1)C1=CC(=CC=C1)C1=CC=CC=C1` | declined | the name |

The last example is a ring assembly of seven rings. The recommendations name ring assemblies of seven nodes or more, with at least four rings, as phanes.

## What is not built yet

Orthonym does not build a phane name for a structure whose rings carry a double bond to an outside atom (the oxo group of a quinolone), for a ring that is a cycloalkene, or for a peri-fused ring system. Such a structure is declined at the default tier, even with `--opsin main`; the wider tiers return a name that is not a phane name, labelled below the preferred name. A cyclophane whose large ring has nine atoms or fewer is outside what the recommendations decide here, and it keeps the name and the label the engine gives it.
