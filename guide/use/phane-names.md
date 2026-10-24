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

Orthonym does not yet build a phane name for a structure with stereodescriptors, for one whose rings carry hydro prefixes or indicated hydrogen, or for a phane that is a substituent of another parent. Such a structure is declined at the default tier, even with `--opsin main`; the wider tiers return a name that is not a phane name, labelled below the preferred name. A cyclophane whose large ring has nine atoms or fewer is outside what the recommendations decide here, and it keeps the name and the label the engine gives it.
