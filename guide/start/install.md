# Install

Orthonym is a Python package. It needs two things on your machine and downloads two more.

| You need | Why |
|:--|:--|
| Python 3.10 or newer | the engine is written in Python |
| A Java runtime, version 11 or newer, on your `PATH` | OPSIN, which reads every name back, and centres, which assigns the CIP stereodescriptors, are Java programs |

Check the Java runtime with `java -version`. Any Java 11+ runtime works (for example OpenJDK or Temurin).

## Install the package

```console
$ pip install "git+https://github.com/Steinbeck-Lab/Orthonym.git"
$ orthonym --fetch-jars
```

The first line installs Orthonym and the packages it needs, among them RDKit, which reads the structures. The second line downloads the two Java programs Orthonym uses, checks each against the SHA-256 checksum recorded in Orthonym, and prints where they are:

```console
$ orthonym --fetch-jars
[orthonym] opsin 2.9.0: /home/you/.cache/orthonym/jars/opsin-cli-2.9.0-jar-with-dependencies.jar
[orthonym] centres 1.2.1: /home/you/.cache/orthonym/jars/centres-cli-1.2.1.jar
```

`pip install` already tries this download for you. Run `orthonym --fetch-jars` once anyway: it also re-checks the files, and it tells you at once if something is missing.

## The two jars

Orthonym does not ship any Java program. It uses two, each pinned to one version, and downloads them from their official releases:

| Jar | Version | Used for | Licence |
|:--|:--|:--|:--|
| OPSIN, `opsin-cli-2.9.0-jar-with-dependencies.jar` | 2.9.0 | reading every name back into a structure | MIT (the jar bundles jna-inchi, LGPL-2.1, and others) |
| centres, `centres-cli-1.2.1.jar` | 1.2.1 | CIP stereodescriptors (*R*/*S*, *E*/*Z*) | BSD-2-Clause (the jar bundles CDK, LGPL-2.1+) |

If a jar is missing, Orthonym stops with an error that says so, rather than naming with fewer checks.

## Three settings for the jars

| Setting | Effect |
|:--|:--|
| `ORTHONYM_OPSIN_JAR=/path/to/opsin-cli-2.9.0-jar-with-dependencies.jar` | use this OPSIN jar |
| `ORTHONYM_CENTRES_JAR=/path/to/centres-cli-1.2.1.jar` | use this centres jar |
| `ORTHONYM_JAR_DIR=/path/to/dir` | keep the downloaded jars here (the default is your user cache, `~/.cache/orthonym/jars`) |

## On a machine with no internet

1. On a machine with internet, run `orthonym --fetch-jars` and copy the two files it prints.
2. On the offline machine, point Orthonym at the copies:

   ```console
   $ export ORTHONYM_OPSIN_JAR=/opt/jars/opsin-cli-2.9.0-jar-with-dependencies.jar
   $ export ORTHONYM_CENTRES_JAR=/opt/jars/centres-cli-1.2.1.jar
   $ export ORTHONYM_NO_DOWNLOAD=1
   ```

   `ORTHONYM_NO_DOWNLOAD=1` stops Orthonym from trying to download anything.

## Naming without the jars

`ORTHONYM_ALLOW_REDUCED=1` lets Orthonym name without the jars. Then no name is read back by OPSIN and the stereodescriptors come from RDKit instead of centres. Use it only when you know you want that.

Next: [your first name](first-name.md).
