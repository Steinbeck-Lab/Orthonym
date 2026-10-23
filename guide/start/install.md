# Install

Orthonym is a Python package. It needs two things on your machine and downloads two more.

| You need | Why |
|:--|:--|
| Python 3.10 or newer | the engine is written in Python |
| A Java runtime, version 11 or newer, on your `PATH` | OPSIN, which reads names back for the round-trip check, and centres, which assigns the CIP stereodescriptors, are Java programs |

Check the Java runtime with `java -version`. Any Java 11+ runtime works (for example OpenJDK or Temurin).

## Install the package

```console
$ pip install orthonym
$ orthonym --fetch-jars
```

With conda, the first line is `conda install -c conda-forge orthonym`; the conda package also installs a Java runtime. To install the newest code from GitHub instead, use `pip install "git+https://github.com/Steinbeck-Lab/Orthonym.git"`.

The first line installs Orthonym and the packages it needs, among them RDKit, which reads the structures. The second line downloads the two Java programs Orthonym uses, checks each against the SHA-256 checksum recorded in Orthonym, and prints where they are:

```console
$ orthonym --fetch-jars
[orthonym] opsin 2.9.0: /home/you/.cache/orthonym/jars/opsin-cli-2.9.0-jar-with-dependencies.jar
[orthonym] centres 1.2.1: /home/you/.cache/orthonym/jars/centres-cli-1.2.1.jar
```

If you skip `orthonym --fetch-jars`, the first call that names a molecule downloads the two jars (about 17 MB) and says so; later calls are fast. An install from GitHub or from a clone already tries this download while pip builds the package. Run `orthonym --fetch-jars` once anyway: it also re-checks the files, and it tells you at once if something is missing. If your shell does not find `orthonym` right after the install, open a new terminal (or run `hash -r`); `python -m orthonym` works in any case.

## The two jars

Orthonym does not ship any Java program. It uses two, each pinned to one version, and downloads them from their official releases:

| Jar | Version | Used for | Licence |
|:--|:--|:--|:--|
| OPSIN, `opsin-cli-2.9.0-jar-with-dependencies.jar` | 2.9.0 | round-trip validation of names | MIT (the jar bundles jna-inchi, LGPL-2.1, and others) |
| centres, `centres-cli-1.2.1.jar` | 1.2.1 | CIP stereodescriptors (*R*/*S*, *E*/*Z*) | BSD-2-Clause (the jar bundles CDK, LGPL-2.1+) |

A missing jar is downloaded: when `pip` builds the package, on first use (unless `ORTHONYM_NO_DOWNLOAD=1` is set), or by `orthonym --fetch-jars`. Downloaded jars and jars in the jar directory are checked against the pinned SHA-256. A jar runs with your rights and the JVM opens it by path, so Orthonym also loads a jar from the jar directory only when no other user can write that directory, or rename it or any directory above it (they could swap the jar after the check). Symlinks are resolved first, so a symlinked `~/.cache` is judged where it leads, and the jar is used by that resolved path. A group-writable directory passes only when the group is your own private group (your primary group, named like you, with no other member and, on Linux, no ACL on the directory (ACLs are not looked at on macOS, so an entry added there with `chmod +a`, which the mode bits do not show, is not seen), as on Debian and Ubuntu; a user with no entry in the password database, as in `docker run -u 1234`, has none), or when the directory lies inside one that only you can enter (a `0700` `~/.cache`). A directory that fails the check stops Orthonym with an error that names every directory at fault and the command that repairs them, `chmod go-w <directories>`; you can also point `ORTHONYM_JAR_DIR` at a directory of your own. On a shared or site installation, where the directories belong to another user or to root and you cannot change them, use the jars your site provides: set `ORTHONYM_OPSIN_JAR` and `ORTHONYM_CENTRES_JAR`. `ORTHONYM_ALLOW_REDUCED=1`, which lets a missing jar pass by naming without it, does not apply to this error: it is a setup to fix, not a missing jar. A jar you point to with `ORTHONYM_OPSIN_JAR` or `ORTHONYM_CENTRES_JAR` is used as given. Orthonym stops with an error only when a jar can be neither found nor downloaded. Without a Java runtime it declines every molecule.

## Three settings for the jars

| Setting | Effect |
|:--|:--|
| `ORTHONYM_OPSIN_JAR=/path/to/opsin-cli-2.9.0-jar-with-dependencies.jar` | use this OPSIN jar |
| `ORTHONYM_CENTRES_JAR=/path/to/centres-cli-1.2.1.jar` | use this centres jar |
| `ORTHONYM_JAR_DIR=/path/to/dir` | keep the downloaded jars here (the default is your user cache, `~/.cache/orthonym/jars`) |

## Choosing the OPSIN version

Orthonym reads every name back with OPSIN 2.9.0, the pinned release jar. You can use OPSIN's main branch instead, which reads some name forms that 2.9.0 does not (phane names, for example):

```console
$ orthonym --opsin main "CCO"
ethanol
$ ORTHONYM_OPSIN=main orthonym "CCO"       # the same, set in the environment
$ orthonym --opsin main --fetch-jars       # build it before the first use
```

| Value | OPSIN |
|:--|:--|
| `2.9.0` (also: nothing set) | the release jar. This is the default. |
| `main` | the newest commit of OPSIN's main branch |
| `main@<commit>` | that one commit of OPSIN's main branch (a full or an abbreviated hash), for a run you want to repeat |

The first use of a commit builds a jar on your machine: Orthonym fetches the source of that commit from GitHub, compiles it with `javac` against the libraries of the 2.9.0 jar, checks the result by reading three names, and keeps it in the subdirectory `opsin-main` of the jar directory as `opsin-main-<first 12 characters of the commit>.jar`, next to a `.json` file that says how it was built. The build usually takes under a minute. Later runs use that jar and need neither GitHub nor a JDK. `main` asks GitHub for the newest commit at most once a day; when GitHub cannot be reached, it uses the last commit it knows. The same commit, the same JDK and the same 2.9.0 jar always give the same jar.

What to know before you choose `main`:

- It needs a JDK (`javac`, version 11 or newer), not only a Java runtime, for the build.
- The jar directory must pass the check above (no other user can write it or rename it or a directory above it), and the cache inside it is private to your user. A jar from it runs with your rights, so Orthonym creates `opsin-main` with mode 700 and uses a built jar only when `opsin-main`, the jar and its `.json` file are real files of yours that group and others cannot write, and the SHA-256 in the `.json` file is the jar's. Anything else stops Orthonym with an error that names the path, or counts as not built.
- A commit named with `main@<commit>` must be on OPSIN's main branch: before it is built, Orthonym asks GitHub whether the commit is in the history of that branch (the source archive service also serves commits that exist only in forks), and it refuses when GitHub cannot say. A commit that is already built is used without asking.
- One OPSIN serves the whole run, so choose it before the first name. A jar you give with `ORTHONYM_OPSIN_JAR` wins over this setting.
- A build that fails (no JDK, no GitHub, a commit that does not compile, a jar that fails its check) stops Orthonym with an error that names the cause. Orthonym does not fall back to 2.9.0 in that case; unset `ORTHONYM_OPSIN` or use `--opsin 2.9.0` to get the release jar. With `ORTHONYM_ALLOW_REDUCED=1` a jar that cannot be built is handled like a missing jar: Orthonym names without OPSIN. A value that is none of `2.9.0`, `main` and `main@<commit>` is always an error, also then.
- With `ORTHONYM_NO_DOWNLOAD=1`, a commit that is not built yet is not built when Orthonym names a structure; `orthonym --opsin main --fetch-jars` is the explicit request and still builds.
- `orthonym --opsin main --version` adds a second line with the build in use.

## On a machine with no internet

1. On a machine with internet, run `orthonym --fetch-jars` and copy the two files it prints.
2. On the offline machine, point Orthonym at the copies:

   ```console
   $ export ORTHONYM_OPSIN_JAR=/opt/jars/opsin-cli-2.9.0-jar-with-dependencies.jar
   $ export ORTHONYM_CENTRES_JAR=/opt/jars/centres-cli-1.2.1.jar
   $ export ORTHONYM_NO_DOWNLOAD=1
   ```

   `ORTHONYM_NO_DOWNLOAD=1` stops Orthonym from downloading a missing jar when it names a structure. `orthonym --fetch-jars` is an explicit request and still downloads.

## Naming without the jars

`ORTHONYM_ALLOW_REDUCED=1` lets Orthonym name without the jars. Then no name is read back by OPSIN and the stereodescriptors come from RDKit instead of centres. Use it only when you know you want that.

Next: [your first name](first-name.md).
